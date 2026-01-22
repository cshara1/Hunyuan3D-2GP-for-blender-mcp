# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A unified model worker and server that exposes both:
1. REST API for Blender Addon (with Auth)
2. Gradio UI for Web Interaction (with Auth and Share)
"""
# --- Monkeypatch for hy3dgen/mmgp compatibility (MUST be before ANY imports) ---
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    # cached_download was removed in 0.26.0. 
    # We map it to hf_hub_download which is the modern equivalent for HF files.
    print("Applying monkeypatch for huggingface_hub.cached_download...")
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# --- Monkeypatch for optimum-quanto/diffusers compatibility removed (Manual offloading does not use mmgp/optimum-quanto) ---




import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys

import traceback
import uuid
import random
import shutil
import time
import secrets
from io import BytesIO
from pathlib import Path
from glob import glob

import torch
import trimesh
import uvicorn
from PIL import Image
import gradio as gr
from fastapi import FastAPI, Request, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# --- Monkeypatch moved to top of file ---

# Manual offloading implementation
print(">>> Manual offloading enabled (MMGP removed)")

# from hymotion.utils.t2m_runtime import T2MRuntime # Moved to __init__ to allow env var setup
import torch

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.shapegen.utils import logger as shapegen_logger

LOGDIR = '.'
MAX_SEED = 1e7

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."




class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


handler = None

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class PipelineOffloader:
    """Context manager for manual offloading of pipelines."""
    def __init__(self, pipeline: object, device: str = "cuda", offload_to: str = "cpu"):
        self.pipeline = pipeline
        self.device = device
        self.offload_to = offload_to

    def __enter__(self):
        self._move_pipeline(self.device)
        return self.pipeline

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._move_pipeline(self.offload_to)
        if self.offload_to == "cpu":
            torch.cuda.empty_cache()

    def _move_pipeline(self, target_device):
        msg = f"Moving pipeline {type(self.pipeline).__name__} to {target_device}..."
        logger.info(msg)
        # Force print to original stdout/stderr to ensure visibility
        try:
            sys.__stdout__.write(f"DEBUG: {msg}\n")
            sys.__stdout__.flush()
        except:
            pass
        
        # Standard Diffusers / PyTorch Modules with .to()
        if hasattr(self.pipeline, "to"):
            self.pipeline.to(target_device)
        
        # Wrapped pipelines (e.g. HunyuanDiTPipeline which holds .pipe)
        elif hasattr(self.pipeline, "pipe") and hasattr(self.pipeline.pipe, "to"):
            self.pipeline.pipe.to(target_device)
            # Also handle text encoders explicitly if needed
            if hasattr(self.pipeline.pipe, "text_encoder") and self.pipeline.pipe.text_encoder:
                 self.pipeline.pipe.text_encoder.to(target_device)
            if hasattr(self.pipeline.pipe, "text_encoder_2") and self.pipeline.pipe.text_encoder_2:
                 self.pipeline.pipe.text_encoder_2.to(target_device)
        
        # Texture Generation Pipeline (holds .models dict)
        elif hasattr(self.pipeline, "models") and isinstance(self.pipeline.models, dict):
             for name, model in self.pipeline.models.items():
                 if hasattr(model, "to"):
                     model.to(target_device)
                 elif hasattr(model, "model") and hasattr(model.model, "to"):
                     model.model.to(target_device)
        
        # Fallback using .cuda() / .cpu() if available
        elif target_device == "cuda" and hasattr(self.pipeline, "cuda"):
             self.pipeline.cuda()
        elif target_device == "cpu" and hasattr(self.pipeline, "cpu"):
             self.pipeline.cpu()
        else:
             logger.warning(f"Pipeline {type(self.pipeline)} does not support manual offloading (no .to() method found).")
        torch.cuda.empty_cache()


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

# --- Helper Functions from gradio_app.py ---

def get_example_img_list():
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))

def get_example_txt_list():
    txt_list = list()
    if os.path.exists('./assets/example_prompts.txt'):
        for line in open('./assets/example_prompts.txt', encoding='utf-8'):
            txt_list.append(line.strip())
    return txt_list

def get_example_mv_list():
    mv_list = list()
    root = './assets/example_mv_images'
    if os.path.exists(root):
        for mv_dir in os.listdir(root):
            view_list = []
            for view in ['front', 'back', 'left', 'right']:
                path = os.path.join(root, mv_dir, f'{view}.png')
                if os.path.exists(path):
                    view_list.append(path)
                else:
                    view_list.append(None)
            mv_list.append(view_list)
    return mv_list

def gen_save_folder(max_size=200):
    os.makedirs(SAVE_DIR, exist_ok=True)
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    return new_folder

def export_mesh(mesh, save_folder, textured=False, type='glb'):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    import html
    # Determine paths
    if textured:
        file_name = f"textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
    else:
        file_name = f"white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
    
    offset = 50 if textured else 10
    
    # Check if template exists, fallback if not
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, template_name)
    if not os.path.exists(template_path):
        template_html = """
        <html><body><model-viewer src="#src#" auto-rotate camera-controls style="width: #width#px; height: #height#px;"></model-viewer></body></html>
        """
    else:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_html = f.read()

    # Construct absolute path for the GLB file and create a /file= URL
    glb_path = os.path.join(save_folder, file_name)
    glb_url = f"/file={os.path.abspath(glb_path)}"

    # Replace placeholders in template
    template_html = template_html.replace('#height#', f'{height - offset}')
    template_html = template_html.replace('#width#', f'{width}')
    template_html = template_html.replace('#src#', glb_url)

    # Use srcdoc to embed HTML directly, avoiding file download issues
    escaped_html = html.escape(template_html)
    iframe_tag = f'<iframe srcdoc="{escaped_html}" height="{height}" width="100%" frameborder="0"></iframe>'

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

def replace_property_getter(instance, property_name, new_getter):
    original_class = type(instance)
    original_property = getattr(original_class, property_name)
    custom_class = type(f'Custom{original_class.__name__}', (original_class,), {})
    new_property = property(new_getter, original_property.fset)
    setattr(custom_class, property_name, new_property)
    instance.__class__ = custom_class
    return instance

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


# --- Model Worker ---

class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2mini',
                 tex_model_path='tencent/Hunyuan3D-2',
                 subfolder='hunyuan3d-dit-v2-mini-turbo',
                 device='cuda',
                 enable_tex=False,
                 enable_t23d=False,
                 enable_motion=True):
        self.model_path = model_path
        self.device = device
        self.mv_mode = '2mv' in model_path or 'mv' in model_path.lower()
        if self.mv_mode:
            logger.info("Multi-view mode detected for Shape Generation.")
        self.has_texturegen = enable_tex
        self.has_t2i = enable_t23d
        
        logger.info(f"Loading the model {model_path} on device {device} ...")

        self.rembg = BackgroundRemover()
        
        # Initialize Shape Generation Pipeline
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device="cpu", # Load to cpu initially for manual offloading
        )
        self.pipeline.enable_flashvdm()
        self.pipeline.enable_flashvdm()

        # Initialize Text-to-Image Pipeline
        if enable_t23d:
            try:
                self.pipeline_t2i = HunyuanDiTPipeline(
                    'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                    device="cpu" # Load to cpu initially
                )
            except Exception as e:
                logger.error(f"Failed to load Text-to-Image model: {e}")
                self.has_t2i = False

        # Initialize Texture Generation Pipeline
        if enable_tex:
            try:
                # Pre-download models to debug potentially masked errors in hy3dgen
                import huggingface_hub
                logger.info(f"Pre-downloading texture models from {tex_model_path}...")
                
                # Retrieve snapshot path explicitly
                snapshot_path = huggingface_hub.snapshot_download(
                    repo_id=tex_model_path, 
                    allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0/*"]
                )
                logger.info(f"Texture models pre-downloaded to: {snapshot_path}")
                
                # Pass the absolute path to bypass internal weak resolution logic
                self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(snapshot_path)
                
                # Manual CUDA Migration for Texture Pipeline
                # Manual CUDA Migration for Texture Pipeline logic removed in favor of mmgp.offload
                # if self.device == "cuda" and torch.cuda.is_available(): ...
            except Exception as e:
                logger.error(f"Failed to load Texture Generation model: {e}")
                import traceback
                traceback.print_exc()
                self.has_texturegen = False

        # Helper Workers
        self.floater_remove_worker = FloaterRemover()
        self.degenerate_face_remove_worker = DegenerateFaceRemover()
        self.face_reduce_worker = FaceReducer()

        # Initialize Motion Generation Runtime
        self.enable_motion = enable_motion
        self.motion_runtime = None
        self.motion_init_error = None
        
        if self.enable_motion:
            try:
                
                # Define potential checkpoint paths (prioritized)
                possible_paths = [
                    # 1. Local path (relative to api_server.py) - Worked in user logs previously
                    "ckpts/tencent/HY-Motion-1.0-Lite",
                    # 2. Sibling directory path
                    "../HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite",
                    # 3. Standard absolute path in container if applicable
                    "/root/Hunyuan3D-2GP/ckpts/tencent/HY-Motion-1.0-Lite" 
                ]
                
                motion_config = None
                motion_ckpt = None
                checked_paths = []

                for base_path in possible_paths:
                    cfg = os.path.join(base_path, "config.yml")
                    ckpt = os.path.join(base_path, "latest.ckpt")
                    checked_paths.append(cfg)
                    
                    if os.path.exists(cfg):
                        motion_config = cfg
                        motion_ckpt = ckpt
                        break
                
                if not motion_config:
                     msg = f"Motion checkpoints not found.\nChecked paths:\n" + "\n".join(checked_paths) + "\nPlease ensure 'config.yml' exists in one of these locations."
                     logger.warning(msg)
                     self.motion_init_error = msg
                     self.enable_motion = False

                if self.enable_motion and motion_config:
                    logger.info(f"Loading Motion Runtime from {motion_config}...")
                    # Ensure quantization is set default to int4 for memory efficiency
                    if "QWEN_QUANTIZATION" not in os.environ:
                        os.environ["QWEN_QUANTIZATION"] = "int4"
                        
                    
                    disable_pe_env = os.environ.get("DISABLE_PROMPT_ENGINEERING", "False").lower() == "true"
                    prompt_model_path_env = os.environ.get("PROMPT_MODEL_PATH", None)

                    from hymotion.utils.t2m_runtime import T2MRuntime
                    self.motion_runtime = T2MRuntime(
                        config_path=motion_config,
                        ckpt_name=motion_ckpt,
                        device_ids=None, # Load to CPU initially by passing None or handling inside T2MRuntime if possible? 
                        # Actually T2MRuntime might conform to device_ids immediately. 
                        # Let's assume we handle it via .to() if it supports it, or recreate.
                        # Looking at T2MRuntime, it loads to device. Let's load to CPU if possible or clear cache.
                        # For now, let's keep it on CPU by not passing CUDA device IDs if that works, or move it after.
                        disable_prompt_engineering=disable_pe_env,
                        prompt_engineering_model_path=prompt_model_path_env
                    )
                    # Manually ensure modules are on cpu
                    if hasattr(self.motion_runtime, "to"):
                         self.motion_runtime.to("cpu")
                    elif hasattr(self.motion_runtime, "model"):
                         self.motion_runtime.model.to("cpu") # Example fallback
                    logger.info("Motion Runtime initialized.")
            except Exception as e:
                logger.error(f"Failed to load Motion Runtime: {e}")
                import traceback
                traceback.print_exc()
                self.motion_init_error = f"Runtime init failed: {str(e)}"
                self.enable_motion = False

    def get_queue_length(self):
        # Placeholder for semaphore usage if needed
        return 0

    def generate(self, uid, params):
        logger.info(f"Worker.generate: Job {uid}, MV Mode: {self.mv_mode}")
        # --- Preprocessing (Image / Text) ---
        if self.mv_mode:
            # Multi-view expects a dictionary
            image_dict = {}
            if 'image' in params:
                img = params["image"]
                image_dict['front'] = load_image_from_base64(img) if isinstance(img, str) else img
            
            # Support other views if provided in params
            for view in ['back', 'left', 'right']:
                key = f'image_{view}'
                if key in params:
                    img = params[key]
                    image_dict[view] = load_image_from_base64(img) if isinstance(img, str) else img
            
            if not image_dict:
                 # Fallback to T2I if no image provided and T2I is enabled
                 if 'text' in params and self.has_t2i:
                    text = params["text"]
                    image_dict['front'] = self.pipeline_t2i(text)
                 else:
                    raise ValueError("No input image or text provided for MV mode")
            
            # Apply background removal to all views
            for k, v in image_dict.items():
                image_dict[k] = self.rembg(v)
            image = image_dict
        else:
            if 'image' in params:
                image = params["image"]
                if isinstance(image, str): # Base64 string
                    image = load_image_from_base64(image)
            else:
                if 'text' in params and self.has_t2i:
                    text = params["text"]
                    with PipelineOffloader(self.pipeline_t2i, self.device):
                        image = self.pipeline_t2i(text)
                else:
                    raise ValueError("No input image or text provided")

            image = self.rembg(image)
        # Note: params are modified in-place to include the PIL image for texture generation later
        
        # --- Shape Generation ---
        if 'mesh' in params:
             # Load existing mesh if provided (e.g. for texturing only)
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            seed = params.get("seed", 1234)
            generator = torch.Generator(self.device).manual_seed(seed)
            
            # Extract generation parameters
            steps = params.get("num_inference_steps", 5) # Default to 5 for turbo
            guidance_scale = params.get('guidance_scale', 5.0)
            octree_resolution = params.get("octree_resolution", 256)
            num_chunks = params.get("num_chunks", 8000)
            
            start_time = time.time()
            with PipelineOffloader(self.pipeline, self.device):
                outputs = self.pipeline(
                    image=image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    output_type='mesh'
                )
            mesh = export_to_trimesh(outputs)[0]
            logger.info("--- Shape Gen: %s seconds ---" % (time.time() - start_time))

        # --- Post-Processing & Texturing ---
        if params.get('texture', False) and self.has_texturegen:
            start_time = time.time()
            mesh = self.floater_remove_worker(mesh)
            mesh = self.degenerate_face_remove_worker(mesh)
            mesh = self.face_reduce_worker(mesh, max_facenum=params.get('face_count', 20000)) # Default face count
            # For texture generation, we use the 'front' view as reference
            tex_image = image['front'] if self.mv_mode else image
            with PipelineOffloader(self.pipeline_tex, self.device):
                mesh = self.pipeline_tex(mesh, tex_image)
            logger.info("--- Texture Gen: %s seconds ---" % (time.time() - start_time))

        # --- Save ---
        file_type = params.get('type', 'glb')
        save_path = os.path.join(SAVE_DIR, f'{str(uid)}.{file_type}')
        
        if file_type == 'glb' or file_type == 'obj':
            # Include normals if textured
            include_normals = params.get('texture', False)
            if file_type == 'glb':
                mesh.export(save_path, include_normals=include_normals)
            else:
                mesh.export(save_path, include_normals=include_normals)
        else:
            mesh.export(save_path)

        torch.cuda.empty_cache()
        # For result, return the main image used
        main_image = image['front'] if self.mv_mode else image
        return save_path, uid, mesh, main_image

    def generate_motion(self, uid, params):
        if not self.enable_motion or not self.motion_runtime:
             reason = getattr(self, "motion_init_error", "Unknown reason (disabled by flag?)")
             raise RuntimeError(f"Motion generation disabled. Reason: {reason}")
        
        logger.info(f"Worker.generate_motion: Job {uid}")
        text = params.get("text", "")
        duration = float(params.get("duration", 3.0)) # Default 3s
        seed = str(params.get("seed", random.randint(0, 1000000)))
        
        # Use existing save dir relative to CWD if easier, or formatted absolute path
        save_dir = os.path.abspath(SAVE_DIR)
        
        try:
            # seeds_csv expects string like "123, 456"
            # Manually move motion runtime to device
            # Note: T2MRuntime might not have a simple .to() method covering all submodules (text encoder, vae, denoiser)
            # We assume it exposes a way or we access internal modules.
            # Checking T2MRuntime implementation in hymotion/utils/t2m_runtime.py would be ideal.
            # Assuming it has a .to() or .cuda() method.
            if hasattr(self.motion_runtime, "to"):
                 self.motion_runtime.to(self.device)
            elif hasattr(self.motion_runtime, "cuda"):
                 self.motion_runtime.cuda()
            
            try:
                html, fbx_files, raw_output = self.motion_runtime.generate_motion(
                    text=text,
                    seeds_csv=str(seed),
                    duration=duration,
                    cfg_scale=params.get("cfg_scale", 7.5),
                    output_format="fbx",
                    output_dir=save_dir,
                    output_filename=f"{uid}_motion",
                    use_special_game_feat=False
                )
            finally:
                 # Move back to CPU
                 if hasattr(self.motion_runtime, "to"):
                      self.motion_runtime.to("cpu")
                 elif hasattr(self.motion_runtime, "cpu"):
                      self.motion_runtime.cpu()
                 torch.cuda.empty_cache()
            
            fbx_path = fbx_files[0] if fbx_files else None
            return html, fbx_path
        except Exception as e:
            logger.error(f"Motion Generation Failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def rewrite_text(self, text, enable_rewrite=True, enable_duration=True):
        if not self.enable_motion or not self.motion_runtime:
             reason = getattr(self, "motion_init_error", "Unknown reason (disabled by flag?)")
             # Just return original text with warning in logs, or raise error? UI handles return tuple
             logger.warning(f"Rewrite disabled. Reason: {reason}")
             return text, 5.0 # Default fallback
        
        try:
             # T2MRuntime.rewrite_text_and_infer_time might not be available if PromptRewriter is disabled
             # But the method itself usually handles logic check or we check here
             if not self.motion_runtime.prompt_rewriter:
                  return text, 5.0
             
             pred_duration, rewritten = self.motion_runtime.rewrite_text_and_infer_time(text)
             return rewritten, pred_duration
        except Exception as e:
             logger.error(f"Rewrite failed: {e}")
             return text, 5.0

# --- FastAPI App & Auth ---

app = FastAPI()
security = HTTPBasic()

# Global variables for Auth
AUTH_USER = None
AUTH_PASS = None

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not AUTH_USER or not AUTH_PASS:
        return True # Auth disabled
    
    current_username_bytes = credentials.username.encode("utf8")
    current_password_bytes = credentials.password.encode("utf8")
    correct_username_bytes = AUTH_USER.encode("utf8")
    correct_password_bytes = AUTH_PASS.encode("utf8")
    
    is_correct_username = secrets.compare_digest(current_username_bytes, correct_username_bytes)
    is_correct_password = secrets.compare_digest(current_password_bytes, correct_password_bytes)
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routes ---

@app.post("/generate", dependencies=[Depends(check_auth)])
async def generate_api(request: Request, background_tasks: BackgroundTasks):
    logger.info("API: Generate Request Received")
    params = await request.json()
    uid = uuid.uuid4()
    
    try:
        # Offload blocking generation to background task
        # We wrap worker.generate to catch exceptions if needed, but worker catches internal errors usually.
        # But worker.generate returns values which we discard here (results are saved to disk).
        background_tasks.add_task(worker.generate, uid, params)
        
        # Return Job ID immediately to prevent timeout
        logger.info(f"API: Job {uid} started in background")
        return JSONResponse({"job_id": str(uid), "status": "RUN"}, status_code=200)
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": "Internal Server Error"}, status_code=500)

@app.get("/status/{uid}", dependencies=[Depends(check_auth)])
async def status_api(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    if not os.path.exists(save_file_path):
        return JSONResponse({'status': 'processing'}, status_code=200)
    else:
        # Ideally we should stream file, but legacy API returns base64
        with open(save_file_path, 'rb') as f:
            base64_str = base64.b64encode(f.read()).decode()
        return JSONResponse({'status': 'completed', 'model_base64': base64_str}, status_code=200)

@app.post("/generate_motion", dependencies=[Depends(check_auth)])
async def generate_motion_endpoint(request: Request):
    logger.info("API: Generate Motion Request Received")
    params = await request.json()
    uid = params.get("uid", str(uuid.uuid4()))
    
    if not worker.enable_motion:
        raise HTTPException(status_code=500, detail="Motion generation is disabled on this server.")

    try:
        loop = asyncio.get_event_loop()
        # Run in executor to avoid blocking main thread
        result = await loop.run_in_executor(None, worker.generate_motion, uid, params)
        html_content, result_path = result
        
        if result_path and os.path.exists(result_path):
             with open(result_path, 'rb') as f:
                 data = base64.b64encode(f.read()).decode()
             
             return JSONResponse({"status": "completed", "result_path": result_path, "model_base64": data, "html_viz": html_content})
        else:
             return JSONResponse({"status": "failed", "error": "No output generated"}, status_code=500)

    except Exception as e:
        logger.error(f"Error during motion generation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Gradio UI ---

def build_gradio_app(worker, args):
    # UI Configuration
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder
    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    
    HAS_TEXTUREGEN = worker.has_texturegen
    HAS_T2I = worker.has_t2i

    # Example Data
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    example_mvs = get_example_mv_list()
    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']
    HTML_OUTPUT_PLACEHOLDER = f'''
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    '''

    # --- UI Logic Functions (Adapters to Worker) ---
    
    def ui_shape_generation(caption, image, mv_front, mv_back, mv_left, mv_right, 
                            steps, guidance, seed, octree, rembg, chunks, rand_seed):
        
        seed = int(randomize_seed_fn(seed, rand_seed))
        save_folder = gen_save_folder()
        
        # Prepare params
        params = {
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "octree_resolution": octree,
            "num_chunks": chunks,
            "texture": False,
        }
        
        # Handle Inputs
        # Handle Inputs
        # Handle Inputs (Common for both MV and Standard modes)
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGBA")
            params["image"] = image
        elif caption and caption.strip():
            if not worker.has_t2i:
                    raise gr.Error("Text-to-3D is disabled. Please provide an image or restart with --enable_t23d.")
            params["text"] = caption
        
        # MV Specific Extras
        if MV_MODE:
            # Map extra views if present (if exposed in UI)
            if mv_front is not None: params["image_front"] = mv_front
            if mv_back is not None: params["image_back"] = mv_back
            if mv_left is not None: params["image_left"] = mv_left
            if mv_right is not None: params["image_right"] = mv_right

        # Validation
        if 'image' not in params and 'text' not in params and not any(k.startswith('image_') for k in params):
                raise gr.Error("Please provide an Input Image or Text Prompt.")
        
        # Call Worker
        try:
            uid = uuid.uuid4()
            file_path, uid, mesh, processed_image = worker.generate(uid, params)
            
            # Post-generation UI Logic (Viewer HTML, Stats)
            stats = {
                'vertices': mesh.vertices.shape[0],
                'faces': mesh.faces.shape[0]
            }
            
            # Use native Gradio Model3D viewer instead of custom HTML
            # path_white = export_mesh(mesh, save_folder, textured=False)
            # html_white = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH, textured=False)
            path_white = export_mesh(mesh, save_folder, textured=False)
            
            return path_white, path_white, stats, seed, save_folder, processed_image, mesh
            
        except Exception as e:
            raise gr.Error(str(e))

    def ui_generation_all(caption, image, mv_front, mv_back, mv_left, mv_right, 
                          steps, guidance, seed, octree, rembg, chunks, rand_seed):
        seed = int(randomize_seed_fn(seed, rand_seed))
        save_folder = gen_save_folder()
        
        params = {
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "octree_resolution": octree,
            "num_chunks": chunks,
            "texture": True,
        }
        
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGBA")
            params["image"] = image
        elif caption is not None:
             params["text"] = caption

        try:
            uid = uuid.uuid4()
            file_path, uid, mesh, processed_image = worker.generate(uid, params)
            
            stats = {'vertices': mesh.vertices.shape[0], 'faces': mesh.faces.shape[0]}
            
            path_white = export_mesh(mesh, save_folder, textured=False) # Save white version too
            path_tex = file_path # This is the textured one returned by worker
            
            # html_tex = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
            
            return path_white, path_tex, path_tex, stats, seed
            
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(str(e))

        except Exception as e:
            traceback.print_exc()
            raise gr.Error(str(e))

    def ui_rewrite(text):
        if not text: return "", 5.0
        rewritten, duration = worker.rewrite_text(text)
        return rewritten, duration

    def ui_generate_motion(text, rewritten_text, duration, cfg, seed, randomize_seed):
        seed = int(randomize_seed_fn(seed, randomize_seed))
        prompt = rewritten_text if rewritten_text and rewritten_text.strip() else text
        
        params = {
            "text": prompt,
            "duration": duration,
            "cfg_scale": cfg,
            "seed": seed
        }
        
        try:
            uid = uuid.uuid4()
            html_content, fbx_path = worker.generate_motion(uid, params)
            
            # iframe wrapping
            if html_content:
                escaped_html = html_content.replace('"', "&quot;")
                iframe_html = f"""
                    <iframe srcdoc="{escaped_html}" width="100%" height="750px" style="border: none; border-radius: 12px;"></iframe>
                """
            else:
                iframe_html = "<p>No Visualization Available</p>"
                
            return iframe_html, fbx_path, seed
            
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(str(e))

    # --- UI Layout ---
    
    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0 Unified Server') as demo:
        gr.Markdown("# Hunyuan3D-2 Unified Server")
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab('Image Prompt'):
                        image = gr.Image(label='Image', type='filepath', image_mode='RGBA', height=290)
                    with gr.Tab('Text Prompt'):
                        caption = gr.Textbox(label='Text Prompt')
                
                with gr.Row():
                    btn = gr.Button(value='Gen Shape', variant='primary')
                    btn_all = gr.Button(value='Gen Textured Shape', variant='primary', visible=HAS_TEXTUREGEN)
                
                # Hidden state containers
                save_folder_state = gr.State()
                mesh_state = gr.State() # Store mesh object if needed for export
                processed_image_state = gr.State()

                with gr.Accordion("Advanced Options", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=1234)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=5 if TURBO_MODE else 30)
                    octree = gr.Slider(label="Octree Resolution", minimum=16, maximum=512, step=16, value=256)
                    guidance = gr.Number(label="Guidance", value=5.0)
                    chunks = gr.Slider(label="Chunks", minimum=1000, maximum=5000000, value=8000)
                    rembg = gr.Checkbox(label="Remove Background", value=True)
                    # MV inputs placeholder
                    mv_f = gr.State(None)
                    mv_b = gr.State(None)
                    mv_l = gr.State(None)
                    mv_r = gr.State(None)

            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab('Generated Mesh'):
                        # html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER)
                        html_gen_mesh = gr.Model3D(label="3D Preview", clear_color=[0.0, 0.0, 0.0, 0.0])
                    with gr.Tab('Mesh Info'):
                        stats = gr.Json({})
                
                # Hidden outputs
                file_out = gr.File(label="White Mesh", visible=False)
                file_out2 = gr.File(label="Textured Mesh", visible=False)

        # Event Wirings
        btn.click(
            ui_shape_generation,
            inputs=[caption, image, mv_f, mv_b, mv_l, mv_r, steps, guidance, seed, octree, rembg, chunks, randomize_seed],
            outputs=[file_out, html_gen_mesh, stats, seed, save_folder_state, processed_image_state, mesh_state]
        )
        
        btn_all.click(
            ui_generation_all,
            inputs=[caption, image, mv_f, mv_b, mv_l, mv_r, steps, guidance, seed, octree, rembg, chunks, randomize_seed],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        )

        # --- HY-Motion UI Section ---
        gr.Markdown("---")
        gr.Markdown("# HY-Motion Generation")
        
        with gr.Row():
            with gr.Column(scale=2):
                motion_input = gr.Textbox(
                    label="📝 Motion Input Text", 
                    placeholder="Enter text to generate motion (e.g. 'A person walking forward')"
                )
                
                with gr.Row():
                     motion_rewrite_btn = gr.Button("🔄 Rewrite Text", variant="secondary")
                
                motion_rewritten = gr.Textbox(
                    label="✏️ Rewritten Text", 
                    interactive=True,
                    placeholder="Rewritten text will appear here. You can edit it manually."
                )
                
                motion_gen_btn = gr.Button("🚀 Generate Motion", variant="primary")
                
                with gr.Accordion("Motion Advanced Options", open=False):
                    motion_duration = gr.Slider(0.5, 12.0, value=5.0, step=0.1, label="Duration (s)")
                    motion_cfg = gr.Number(value=7.5, label="Guidance Scale")
                    motion_seed = gr.Number(value=1234, label="Seed")
                    motion_rand_seed = gr.Checkbox(value=True, label="Randomize Seed")
                    
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Motion Visualization"):
                        motion_html = gr.HTML(label="Visualization")
                    with gr.Tab("Files"):
                        motion_file = gr.File(label="Download FBX")
                
        # Motion Event Wiring
        motion_rewrite_btn.click(
            ui_rewrite,
            inputs=[motion_input],
            outputs=[motion_rewritten, motion_duration]
        )
        
        motion_gen_btn.click(
            ui_generate_motion,
            inputs=[motion_input, motion_rewritten, motion_duration, motion_cfg, motion_seed, motion_rand_seed],
            outputs=[motion_html, motion_file, motion_seed]
        )

    return demo


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--enable_tex', action='store_true')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--turbo', action='store_true')
    parser.add_argument('--share', action='store_true', help="Enable Gradio Share Link")
    parser.add_argument('--auth-user', type=str, help="Username for authentication")
    parser.add_argument('--auth-pass', type=str, help="Password for authentication")
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--profile', type=str, default="3", help="Offload profile (1-4). Default 3 (Balanced).")
    parser.add_argument('--verbose', type=str, default="1", help="Verbose level for offloading.")
    parser.add_argument('--disable-motion', action='store_true', help="Disable motion generation features.")
    parser.add_argument("--quantize-text-encoder", type=str, default="int4", choices=["int4", "int8", "none"], help="Quantization level for text encoder (int4, int8, none)")
    parser.add_argument("--disable-prompt-engineering", action="store_true", help="Disable prompt rewriting (saves VRAM)")
    parser.add_argument("--prompt-cpu-mode", action="store_true", help="Run prompt rewriter on CPU")
    parser.add_argument("--prompt-model-path", type=str, default="Qwen/Qwen3-8B", help="Path to prompt rewriter model")
    args = parser.parse_args()

    # Apply Turbo defaults if needed
    if args.turbo:
        if "turbo" not in args.subfolder:
            args.subfolder += "-turbo"

    # Set Environment Variables for HY-Motion
    os.environ["QWEN_QUANTIZATION"] = args.quantize_text_encoder
    if args.disable_prompt_engineering:
        os.environ["DISABLE_PROMPT_ENGINEERING"] = "True"
    if args.prompt_cpu_mode:
        os.environ["PROMPT_CPU_MODE"] = "true"
    if args.prompt_model_path:
        os.environ["PROMPT_MODEL_PATH"] = args.prompt_model_path
        
    # Default to using HF models for HY-Motion if not specified (since local ckpts are often empty)
    if "USE_HF_MODELS" not in os.environ:
        os.environ["USE_HF_MODELS"] = "1"

    logger.info(f"Starting server with args: {args}")

    # Set Auth Globals
    if args.auth_user and args.auth_pass:
        AUTH_USER = args.auth_user
        AUTH_PASS = args.auth_pass
        logger.info("Authentication enabled.")
        gradio_auth = (args.auth_user, args.auth_pass)
    else:
        gradio_auth = None

    # Initialize Worker
    worker = ModelWorker(
        model_path=args.model_path,
        tex_model_path=args.texgen_model_path,
        subfolder=args.subfolder,
        device=args.device,
        enable_tex=True if not args.turbo else args.enable_tex, # Default enable tex for non-turbo? Configurable.
        enable_t23d=args.enable_t23d,
        enable_motion=not args.disable_motion
    )

    # --- Manual Offloading Setup ---
    # We don't need complex setup here as we handle it per-request in generate()
    logger.info("Server ready with manual offloading strategy.")

    # Build Gradio UI
    demo = build_gradio_app(worker, args)

    # Static Files for Gradio HTML viewer
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

    # Launch Strategy
    if args.share:
        logger.info("Launching in SHARE mode. API endpoints will only be available on the local forwarded port if supported by Gradio tunnel.")
        print("!"*80)
        print("WARNING: --share mode is active. The Gradio UI will be publicly accessible.")
        print("The custom API endpoints (/generate) are enabled locally.")
        print("!"*80)

        # Launch Gradio with prevent_thread_lock=True to allow us to attach routes
        _, _, shared_url = demo.launch(
            share=True, 
            auth=gradio_auth, 
            server_name=args.host, 
            server_port=args.port, 
            allowed_paths=[os.path.abspath(SAVE_DIR)],
            prevent_thread_lock=True
        )
        
        # Attach our custom API endpoint to Gradio's internal FastAPI app
        # This makes /generate available on the port Gradio is running on (e.g. 8081)
        demo.app.include_router(app.router)
        logger.info("Custom API endpoints attached to Gradio server.")
        logger.info(f"Public Share URL: {shared_url}")
        
        # Keep the main thread alive
        demo.block_thread()
    else:
        # Default behavior: Unified Server
        app = gr.mount_gradio_app(app, demo, path="/", auth=gradio_auth, allowed_paths=[os.path.abspath(SAVE_DIR)])
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
