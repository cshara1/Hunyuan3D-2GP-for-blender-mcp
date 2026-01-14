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
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
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
from mmgp import offload

# --- Monkeypatch for hy3dgen compatibility ---
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    # cached_download was removed in 0.26.0. 
    # We map it to hf_hub_download which is the modern equivalent for HF files,
    # or just warn if it fails.
    print("Applying monkeypatch for huggingface_hub.cached_download...")
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

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
                 enable_t23d=False):
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
            device=device,
        )
        self.pipeline.enable_flashvdm()

        # Initialize Text-to-Image Pipeline
        if enable_t23d:
            try:
                self.pipeline_t2i = HunyuanDiTPipeline(
                    'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                    device=device
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
                huggingface_hub.snapshot_download(repo_id=tex_model_path, allow_patterns=["hunyuan3d-delight-v2-0/*"])
                huggingface_hub.snapshot_download(repo_id=tex_model_path, allow_patterns=["hunyuan3d-paint-v2-0/*"])
                logger.info("Texture models pre-downloaded successfully.")
                
                self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)
                
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
        if MV_MODE:
            # Not fully implemented in this unified script version yet for MV inputs mapping
            # But let's support single image/text flow primarily for now as base
             pass
        else:
            if image is not None:
                params["image"] = image
            elif caption and caption.strip():
                if not worker.has_t2i:
                     raise gr.Error("Text-to-3D is disabled. Please provide an image or restart with --enable_t23d.")
                params["text"] = caption
            else:
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

    # --- UI Layout ---
    
    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0 Unified Server') as demo:
        gr.Markdown("# Hunyuan3D-2 Unified Server")
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab('Image Prompt'):
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
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
    args = parser.parse_args()

    # Apply Turbo defaults if needed
    if args.turbo:
        if "turbo" not in args.subfolder:
            args.subfolder += "-turbo"

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
        enable_t23d=args.enable_t23d
    )

    # --- MMGP Profile Offloading ---
    try:
        profile = int(args.profile)
        kwargs = {}
        
        # Monkeypatch _execution_device for mmgp compatibility if needed
        replace_property_getter(worker.pipeline, "_execution_device", lambda self: "cuda")
        
        pipe = offload.extract_models("i23d_worker", worker.pipeline)
        
        if worker.has_texturegen:
            pipe.update(offload.extract_models("texgen_worker", worker.pipeline_tex))
            # Enable slicing for VAE in texture gen (copied from gradio_app.py)
            worker.pipeline_tex.models["multiview_model"].pipeline.vae.use_slicing = True
            
        if worker.has_t2i:
            pipe.update(offload.extract_models("t2i_worker", worker.pipeline_t2i))
            
        if profile < 5:
            kwargs["pinnedMemory"] = "i23d_worker/model"
        if profile != 1 and profile != 3:
            kwargs["budgets"] = { "*": 2200 }
            
        offload.default_verboseLevel = verboseLevel = int(args.verbose)
        logger.info(f"Applying mmgp offload profile: {profile}")
        offload.profile(pipe, profile_no=profile, verboseLevel=int(args.verbose), **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to apply mmgp offloading: {e}")
        traceback.print_exc()

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
