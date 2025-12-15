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

import os
import random
import shutil
import time
from glob import glob
from pathlib import Path

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uuid
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import HEDdetector # Although imported, HEDdetector isn't used in the sketch logic currently. Canny is used.

from hy3dgen.shapegen.utils import logger

MAX_SEED = int(1e7)


def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def get_example_mv_list():
    print('Loading example mv list ...')
    mv_list = list()
    root = './assets/example_mv_images'
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
        print(f"Removed the oldest folder: {oldest_dir}")
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")
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
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', f'{related_path}/')
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(
        f'Find html file {output_html_path}, {os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """


def _gen_shape(
    caption=None,
    image=None,
    sketch=None,
    sketch_prompt=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    device_choice="GPU" # <--- ADDED DEVICE CHOICE ARG
):
    # Determine target device based on user choice and availability
    target_device = "cuda" if device_choice == "GPU" and torch.cuda.is_available() else "cpu"
    # Store original devices safely, checking if workers exist
    original_i23d_device = str(i23d_worker.device) if 'i23d_worker' in globals() and i23d_worker else 'cpu'
    original_s2i_device = str(s2i_worker.device) if 's2i_worker' in globals() and s2i_worker and HAS_S2I else None
    original_t2i_device = str(t2i_worker.device) if 't2i_worker' in globals() and t2i_worker and HAS_T2I and args.enable_t23d else None

    # --- WRAP in try...finally for model movement ---
    try:
        # Move models to the target device for inference
        if 'i23d_worker' in globals() and i23d_worker: i23d_worker.to(target_device)
        if 's2i_worker' in globals() and s2i_worker and HAS_S2I: s2i_worker.to(target_device)
        if 't2i_worker' in globals() and t2i_worker and HAS_T2I and args.enable_t23d: t2i_worker.to(target_device)
        print(f"--- Running generation on {target_device.upper()} ---")

        # --- EXISTING LOGIC STARTS HERE ---
        if not MV_MODE and image is None and caption is None and sketch is None:
            raise gr.Error("Please provide either a caption, an image, or a sketch.") # Corrected error message
        if MV_MODE:
             if mv_image_front is None and mv_image_back is None and mv_image_left is None and mv_image_right is None:
                 raise gr.Error("Please provide at least one view image for MultiView mode.") # Corrected error message for MV
             image = {}
             if mv_image_front: image['front'] = mv_image_front
             if mv_image_back: image['back'] = mv_image_back
             if mv_image_left: image['left'] = mv_image_left
             if mv_image_right: image['right'] = mv_image_right


        seed = int(randomize_seed_fn(seed, randomize_seed))
        octree_resolution = int(octree_resolution)
        if caption: print('prompt is', caption)
        save_folder = gen_save_folder()
        stats = {
            'model': {
                'shapegen': f'{args.model_path}/{args.subfolder}',
                'texgen': f'{args.texgen_model_path}',
            },
            'params': {
                'caption': caption,
                'sketch_prompt': sketch_prompt, # Added sketch prompt to stats
                'steps': steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'octree_resolution': octree_resolution,
                'check_box_rembg': check_box_rembg,
                'num_chunks': num_chunks,
                'device_choice': device_choice # Added device choice to stats
            }
        }
        time_meta = {}

        if image is None: # Only generate if no image is provided
            if caption and HAS_T2I and args.enable_t23d:
                start_time = time.time()
                try:
                    # Pass the correct device generator
                    t2i_generator = torch.Generator(device=target_device).manual_seed(int(seed))
                    image = t2i_worker(caption, generator=t2i_generator) # Removed redundant call, pass generator
                except Exception as e:
                    logger.error(f"Text-to-Image generation failed: {e}")
                    raise gr.Error(f"Text to 3D failed during image generation.")
                time_meta['text2image'] = time.time() - start_time
            elif sketch and HAS_S2I:
                start_time = time.time()
                if not sketch_prompt:
                    raise gr.Error("Please provide a text prompt to describe your sketch.")
                # Extract the composite image from the Sketchpad dictionary
                sketch_pil = sketch["composite"]
                # Convert sketch (RGBA) to RGB and then to Canny edge map
                sketch_rgb = sketch_pil.convert("RGB")
                sketch_np = np.array(sketch_rgb)
                low_threshold = 100
                high_threshold = 200
                canny_image_np = cv2.Canny(sketch_np, low_threshold, high_threshold)
                canny_image_np = canny_image_np[:, :, None]
                canny_image_np = np.concatenate([canny_image_np, canny_image_np, canny_image_np], axis=2)
                canny_image_pil = Image.fromarray(canny_image_np)
                try:
                     # Pass the correct device generator
                    s2i_generator = torch.Generator(device=target_device).manual_seed(int(seed))
                    s2i_output = s2i_worker(
                        sketch_prompt,
                        image=canny_image_pil,
                        num_inference_steps=20, # Keep relatively low for speed
                        generator=s2i_generator # Pass generator here
                    )
                    image = s2i_output.images[0]
                except Exception as e:
                    logger.error(f"Sketch-to-Image generation failed: {e}")
                    raise gr.Error(f"Sketch to 3D failed during image generation.")
                time_meta['sketch2image'] = time.time() - start_time
            else:
                 # Handle cases where T2I/S2I might be disabled or input missing
                 raise gr.Error("Required model (T2I or S2I) not available or input missing.")

        # Background removal (rmbg_worker likely runs on CPU, keep as is)
        if MV_MODE:
            start_time = time.time()
            for k, v in image.items():
                if v and (check_box_rembg or v.mode == "RGB"): # Check if view exists
                    img = rmbg_worker(v.convert('RGB'))
                    image[k] = img
            time_meta['remove background'] = time.time() - start_time
        else:
             if image and (check_box_rembg or image.mode == "RGB"): # Check if image exists
                start_time = time.time()
                image = rmbg_worker(image.convert('RGB'))
                time_meta['remove background'] = time.time() - start_time


        # image to white model
        start_time = time.time()
        generator = torch.Generator(device=target_device) # Use target_device
        generator = generator.manual_seed(int(seed))

        # Check if i23d_worker exists before calling
        if 'i23d_worker' not in globals() or not i23d_worker:
             raise gr.Error("Image-to-3D model failed to load.")

        outputs = i23d_worker( # This is now on the target_device
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type='mesh'
        )
        time_meta['shape generation'] = time.time() - start_time
        logger.info(f"---Shape generation on {target_device.upper()} takes {time.time() - start_time:.2f} seconds ---")

        tmp_start = time.time()
        mesh = export_to_trimesh(outputs)[0]
        time_meta['export to trimesh'] = time.time() - tmp_start

        stats['number_of_faces'] = mesh.faces.shape[0]
        stats['number_of_vertices'] = mesh.vertices.shape[0]

        stats['time'] = time_meta
        # Handle case where image might be None if generation failed earlier
        main_image = None
        if isinstance(image, dict): # MV_MODE
             main_image = image.get('front') # Use .get for safety
        else: # Single image mode
             main_image = image

        return mesh, main_image, save_folder, stats, seed

    # --- ENSURE MODELS ARE MOVED BACK ---
    finally:
        print(f"--- Moving models back to original devices if needed ---")
        # Move back to original devices (often CPU if offloaded)
        if 'i23d_worker' in globals() and i23d_worker and str(i23d_worker.device) != original_i23d_device:
             print(f"Moving i23d_worker back to {original_i23d_device}")
             i23d_worker.to(original_i23d_device)
        if 's2i_worker' in globals() and s2i_worker and HAS_S2I and original_s2i_device and str(s2i_worker.device) != original_s2i_device:
            print(f"Moving s2i_worker back to {original_s2i_device}")
            s2i_worker.to(original_s2i_device)
        if 't2i_worker' in globals() and t2i_worker and HAS_T2I and args.enable_t23d and original_t2i_device and str(t2i_worker.device) != original_t2i_device:
             print(f"Moving t2i_worker back to {original_t2i_device}")
             t2i_worker.to(original_t2i_device)
        # Clean cache if running on GPU to save VRAM
        if target_device == "cuda":
            print("Clearing CUDA cache.")
            torch.cuda.empty_cache()
    # --- END try...finally block ---


def generation_all(
    caption=None,
    image=None,
    sketch=None,
    sketch_prompt=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    device_choice="GPU" # <--- ADDED DEVICE CHOICE
):
    target_device = "cuda" if device_choice == "GPU" and torch.cuda.is_available() else "cpu"
    original_texgen_device = str(texgen_worker.device) if 'texgen_worker' in globals() and texgen_worker and HAS_TEXTUREGEN else None

    try:
        start_time_0 = time.time()
        # Pass device_choice down
        mesh, image, save_folder, stats, seed = _gen_shape(
            caption,
            image,
            sketch,
            sketch_prompt,
            mv_image_front=mv_image_front,
            mv_image_back=mv_image_back,
            mv_image_left=mv_image_left,
            mv_image_right=mv_image_right,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            octree_resolution=octree_resolution,
            check_box_rembg=check_box_rembg,
            num_chunks=num_chunks,
            randomize_seed=randomize_seed,
            device_choice=device_choice # <--- PASS IT HERE
        )
        path = export_mesh(mesh, save_folder, textured=False)

        # Texture Generation - Move texgen_worker if needed
        if HAS_TEXTUREGEN and 'texgen_worker' in globals() and texgen_worker:
            texgen_worker.to(target_device)
            print(f"--- Running texture generation on {target_device.upper()} ---")
            tmp_time = time.time()
            textured_mesh = texgen_worker(mesh, image) # Runs on target_device
            logger.info(f"---Texture Generation on {target_device.upper()} takes {time.time() - tmp_time:.2f} seconds ---")
            stats['time']['texture generation'] = time.time() - tmp_time
            path_textured = export_mesh(textured_mesh, save_folder, textured=True)
            model_viewer_html_textured = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)

        else: # Handle case where texture gen is disabled or failed to load
             path_textured = path # Use untextured path
             model_viewer_html_textured = "<p style='color:red;'>Texture Generation Disabled or Failed to Load.</p>"
             print("--- Texture generation skipped (disabled or unavailable) ---")
             if 'time' not in stats: stats['time'] = {} # Ensure time dict exists
             stats['time']['texture generation'] = 0 # Indicate no time spent


        # Postprocessing (assuming these run on CPU is fine)
        tmp_time = time.time()
        mesh_reduced = face_reduce_worker(mesh) # Apply reduction to the base mesh if needed elsewhere
        logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
        stats['time']['face reduction'] = time.time() - tmp_time

        # Update total time and metadata
        stats['time']['total'] = time.time() - start_time_0
        if HAS_TEXTUREGEN and 'texgen_worker' in globals() and texgen_worker:
            textured_mesh.metadata['extras'] = stats
        else:
            mesh.metadata['extras'] = stats # Add stats to the base mesh if no texture


        # Note: low_vram_mode check is handled within _gen_shape and the finally block here
        return (
            gr.update(value=path),
            gr.update(value=path_textured),
            model_viewer_html_textured,
            stats,
            seed,
        )

    finally:
        # Move texgen_worker back
        if 'texgen_worker' in globals() and texgen_worker and HAS_TEXTUREGEN and original_texgen_device and str(texgen_worker.device) != original_texgen_device:
            print(f"Moving texgen_worker back to {original_texgen_device}")
            texgen_worker.to(original_texgen_device)
        # Clean cache if running on GPU
        if target_device == "cuda":
            print("Clearing CUDA cache.")
            torch.cuda.empty_cache()


def shape_generation(
    caption=None,
    image=None,
    sketch=None,
    sketch_prompt=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
    device_choice="GPU" # <--- ADDED DEVICE CHOICE
):
    # No try...finally needed here as _gen_shape handles its models
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption,
        image,
        sketch,
        sketch_prompt,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
        device_choice=device_choice # <--- PASS IT HERE
    )
    # Ensure stats['time'] exists before adding total
    if 'time' not in stats: stats['time'] = {}
    stats['time']['total'] = time.time() - start_time_0
    mesh.metadata['extras'] = stats

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)

    # _gen_shape already called empty_cache if on GPU in its finally block
    return (
        gr.update(value=path),
        model_viewer_html,
        stats,
        seed,
    )


def build_app():
    # --- Use global default_device defined in __main__ ---
    global default_device
    # --- END CHANGE ---

    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if MV_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'
    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
    {title}
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    <div align="center">
      <a href="https://github.com/tencent/Hunyuan3D-2">Github</a> &ensp;
      <a href="http://3d-models.hunyuan.tencent.com">Homepage</a> &ensp;
      <a href="https://3d.hunyuan.tencent.com">Hunyuan3D Studio</a> &ensp;
      <a href="#">Technical Report</a> &ensp;
      <a href="https://huggingface.co/Tencent/Hunyuan3D-2"> Pretrained Models</a> &ensp;
    </div>
    """
    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1480px;
    }
    .mv-image button .wrap {
        font-size: 10px;
    }
    .mv-image .icon-wrap {
        width: 20px;
    }
    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0', analytics_enabled=False, css=custom_css) as demo:
        gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt', visible=not MV_MODE) as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)

                    with gr.Tab('Sketch Prompt', id='tab_sketch_prompt', visible=HAS_S2I and not MV_MODE) as tab_sp:
                        sketch_prompt = gr.Textbox(label='Sketch Prompt',
                                                 placeholder='Describe your sketch (e.g., "a red armchair")',
                                                 info='Describe what the sketch is, to guide the image generation.')
                        sketch = gr.Sketchpad(label="Draw your Sketch", type="pil", image_mode="RGBA", height=290)

                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='HunyuanDiT will be used to generate image.',
                                             info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('MultiView Prompt', visible=MV_MODE) as tab_mv:
                        with gr.Row():
                            mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')

                with gr.Row():
                    btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                    btn_all = gr.Button(value='Gen Textured Shape',
                                        variant='primary',
                                        visible=HAS_TEXTUREGEN,
                                        min_width=100)

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                    with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(label='Generation Mode',
                                            info='Recommendation: Turbo for most cases, Fast for very complex cases, Standard seldom use.',
                                            choices=['Turbo', 'Fast', 'Standard'], value='Turbo')
                        decode_mode = gr.Radio(label='Decoding Mode',
                                               info='The resolution for exporting mesh from generated vectset',
                                               choices=['Low', 'Standard', 'High'],
                                               value='Standard')
                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                        # --- ADDED DEVICE CHOICE UI ---
                        device_choice = gr.Radio(
                            label="Processing Device",
                            choices=["CPU", "GPU"] if has_gpu else ["CPU"], # Only show GPU if available
                            value=default_device, # Use the checked default from __main__
                            info="Select CPU or GPU. GPU is faster."
                        )
                        # --- END ADDITION ---
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background', min_width=100)
                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True, min_width=100)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                        )
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps')
                            octree_resolution = gr.Slider(maximum=512, minimum=16, value=256, label='Octree Resolution')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100)
                    with gr.Tab("Export", id='tab_export'):
                        with gr.Row():
                            file_type = gr.Dropdown(label='File Type', choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100)
                            reduce_face = gr.Checkbox(label='Simplify Mesh', value=False, min_width=100)
                            export_texture = gr.Checkbox(label='Include Texture', value=False,
                                                         visible=False, min_width=100)
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000,
                                                    label='Target Face Number')
                        with gr.Row():
                            confirm_export = gr.Button(value="Transform", min_width=100)
                            file_export = gr.DownloadButton(label="Download", variant='primary',
                                                            interactive=False, min_width=100)

            with gr.Column(scale=6):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')

            with gr.Column(scale=3 if MV_MODE else 2):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery', visible=not MV_MODE) as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=18)
                    # Sketch Gallery could be added here if needed, linking to sketch input
                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery', visible=HAS_T2I and not MV_MODE) as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label=None, examples_per_page=18)
                    with gr.Tab('MultiView to 3D Gallery', id='tab_mv_gallery', visible=MV_MODE) as tab_mv:
                        with gr.Row():
                            gr.Examples(examples=example_mvs,
                                        inputs=[mv_image_front, mv_image_back, mv_image_left, mv_image_right],
                                        label=None, examples_per_page=6)

        gr.HTML(f"""
        <div align="center">
        Activated Model - Shape Generation ({args.model_path}/{args.subfolder}) ; Texture Generation ({'Hunyuan3D-2' if HAS_TEXTUREGEN else 'Unavailable'})
        </div>
        """)
        if not HAS_TEXTUREGEN:
            gr.HTML("""
            <div style="margin-top: 5px;"  align="center">
                <b>Warning: </b>
                Texture synthesis is disable due to missing requirements,
                 please install requirements following <a href="https://github.com/Tencent/Hunyuan3D-2?tab=readme-ov-file#install-requirements">README.md</a>to activate it.
            </div>
            """)
        if not args.enable_t23d:
             gr.HTML("""
             <div style="margin-top: 5px;"  align="center">
                 <b>Warning: </b>
                 Text to 3D is disable. To activate it, please run `python gradio_app.py --enable_t23d`.
             </div>
             """)
        # --- ADD WARNING FOR SKETCH-TO-IMAGE FAILURE ---
        if not HAS_S2I:
             gr.HTML("""
             <div style="margin-top: 5px;"  align="center">
                 <b style='color:red;'>Warning: </b>
                 Sketch to 3D failed to load (ControlNet model). Check logs and internet connection.
             </div>
             """)
        # --- END ADDITION ---


        tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
        tab_sp.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery) # Link sketch tab to image gallery
        if HAS_T2I:
            tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)

        # --- UPDATED BUTTON CLICK INPUTS ---
        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                sketch,
                sketch_prompt,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                device_choice # Pass device choice
            ],
            outputs=[file_out, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                sketch,
                sketch_prompt,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                device_choice # Pass device choice
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )
        # --- END UPDATED BUTTON CLICK INPUTS ---


        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value == 'Fast':
                return gr.update(value=10)
            else:
                return gr.update(value=30)
        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value == 'Low':
                return gr.update(value=196)
            elif value == 'Standard':
                return gr.update(value=256)
            else:
                return gr.update(value=384)
        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], outputs=[octree_resolution])

        def on_export_click(file_out, file_out2, file_type, reduce_face, export_texture, target_face_num):
            if file_out is None:
                raise gr.Error('Please generate a mesh first.')

            print(f'exporting {file_out}')
            print(f'reduce face to {target_face_num}')
            if export_texture:
                # Check if texture generation actually succeeded
                if not HAS_TEXTUREGEN or file_out2 is None:
                     raise gr.Error('Textured mesh not available. Generate with "Gen Textured Shape" or fix texture model loading.')
                mesh = trimesh.load(file_out2)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=True, type=file_type)
                # for preview
                save_folder_preview = gen_save_folder() # Use different folder for preview to avoid name clashes
                _ = export_mesh(mesh, save_folder_preview, textured=True)
                model_viewer_html = build_model_viewer_html(save_folder_preview, height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
            else:
                mesh = trimesh.load(file_out)
                # Postprocessing (assuming these run on CPU)
                mesh = floater_remove_worker(mesh)
                mesh = degenerate_face_remove_worker(mesh)
                if reduce_face:
                    mesh = face_reduce_worker(mesh, target_face_num)

                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=False, type=file_type)
                # for preview
                save_folder_preview = gen_save_folder()
                _ = export_mesh(mesh, save_folder_preview, textured=False)
                model_viewer_html = build_model_viewer_html(save_folder_preview, height=HTML_HEIGHT, width=HTML_WIDTH, textured=False)

            print(f'export to {path}')
            return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected='export_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )

    return demo


    # --- Global variables for workers (initialized in __main__) ---
    # Define them here so functions above can see them
    texgen_worker = None
    t2i_worker = None
    s2i_worker = None
    i23d_worker = None
    rmbg_worker = None
    floater_remove_worker = None
    degenerate_face_remove_worker = None
    face_reduce_worker = None
    # --- END GLOBALS ---

if __name__ == '__main__':
    import argparse

    # --- PASTE THE GLOBAL DECLARATIONS HERE ---
    global texgen_worker, t2i_worker, s2i_worker, i23d_worker
    global rmbg_worker, floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker
    # --- END PASTE ---

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda', help="Initial device to load models onto (cuda or cpu)")
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    args = parser.parse_args()


    # --- ADDED GPU CHECK and default device determination ---
    has_gpu = torch.cuda.is_available()
    # Respect user's command line choice if possible, otherwise default based on availability
    initial_device_arg = args.device if (args.device == 'cuda' and has_gpu) or args.device == 'cpu' else ('cuda' if has_gpu else 'cpu')
    default_device = "GPU" if initial_device_arg == 'cuda' else "CPU" # Default for the UI Radio button
    print(f"CUDA Available: {has_gpu}. Initial model device: {initial_device_arg}. UI Default: {default_device}")
    # --- END CHANGE ---


    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px;
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    example_mvs = get_example_mv_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']


    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            # Apply offloading based on initial device choice and low_vram mode
            if args.low_vram_mode or initial_device_arg == 'cpu':
                 print("Enabling CPU offload for TexGen.")
                 texgen_worker.enable_model_cpu_offload()
            else:
                 texgen_worker.to(initial_device_arg) # Move to initial device if not offloading
            HAS_TEXTUREGEN = True
        except Exception as e:
            print(e)
            print("Failed to load texture generator.")
            print('Please try to install requirements by following README.md')
            HAS_TEXTUREGEN = False


    HAS_T2I = False # Default to False
    if args.enable_t23d:
        try:
            from hy3dgen.text2image import HunyuanDiTPipeline
            # Load T2I worker to the initial device determined earlier
            t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=initial_device_arg)
            HAS_T2I = True
            logger.info(f"Loaded Text-to-Image pipeline on {initial_device_arg}.")
        except Exception as e:
            print(e)
            print("Failed to load Text-to-Image generator even though --enable_t23d was set.")
            HAS_T2I = False


    HAS_S2I = False # Default to False
    try:
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        s2i_worker = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )
        s2i_worker.scheduler = UniPCMultistepScheduler.from_config(s2i_worker.scheduler.config)
        # Apply offloading based on initial device choice and low_vram mode
        if args.low_vram_mode or initial_device_arg == 'cpu':
             print("Enabling CPU offload for Sketch2Image.")
             s2i_worker.enable_model_cpu_offload()
        else:
             s2i_worker.to(initial_device_arg) # Move to initial device if not offloading

        HAS_S2I = True
        logger.info(f"Loaded Sketch-to-Image ControlNet pipeline on {initial_device_arg} (or offloaded).")
    except Exception as e:
        print(e)
        print("Failed to load Sketch-to-Image generator.")
        HAS_S2I = False


    # Load the main i23d_worker (shape generator) - Use initial_device_arg
    try:
        from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier, \
            Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.shapegen.pipelines import export_to_trimesh
        from hy3dgen.rembg import BackgroundRemover

        i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model_path,
            subfolder=args.subfolder,
            use_safetensors=True,
            device=initial_device_arg, # Use determined initial device
        )
        logger.info(f"Loaded Image-to-3D pipeline on {initial_device_arg}.")

        if args.enable_flashvdm:
            mc_algo = 'mc' if initial_device_arg in ['cpu', 'mps'] else args.mc_algo # Check initial_device_arg
            i23d_worker.enable_flashvdm(mc_algo=mc_algo)
        if args.compile:
            i23d_worker.compile()

        # Load other workers (these usually run on CPU)
        rmbg_worker = BackgroundRemover()
        floater_remove_worker = FloaterRemover()
        degenerate_face_remove_worker = DegenerateFaceRemover()
        face_reduce_worker = FaceReducer()

    except Exception as e:
         print(f"FATAL: Failed to load core Image-to-3D model or its dependencies: {e}")
         # Consider exiting if the core model fails
         # exit()
         i23d_worker = None # Ensure it's None if loading failed

    # FastAPI setup
    app = FastAPI()
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    # Ensure env_maps source exists before copying
    if os.path.exists('./assets/env_maps'):
         shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)
    else:
         print("Warning: './assets/env_maps' not found, skipping copy.")


    # Initial cache clear if starting on GPU
    if initial_device_arg == 'cuda':
        torch.cuda.empty_cache()

    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)