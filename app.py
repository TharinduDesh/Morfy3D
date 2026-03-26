import gradio as gr
import trimesh
import os
import uuid
from PIL import Image
import torch
import numpy as np

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

# XAI: Occlusion Sensitivity
from xai_occlusion import occlusion_sensitivity_heatmap

# Stage-02: Refinement + Dimension Scaling
from stage2_refine import refine_mesh, scale_mesh_to_dimensions


# =========================
# Global Variables
# =========================
SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)
SUPPORTED_FORMATS = ["glb", "obj", "ply", "stl"]

generated_mesh_cache = {"mesh": None}


# =========================
# 1) Load Models
# =========================
print("Loading the 3D generation model...")
try:
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
    )
    print("Main 3D Model loaded successfully.")
except Exception as e:
    print(f"Error loading 3D model: {e}")
    pipeline = None

print("Loading Background Remover...")
try:
    rmbg_worker = BackgroundRemover()
    print("Background Remover loaded successfully.")
except Exception as e:
    print(f"Error loading BackgroundRemover: {e}")
    rmbg_worker = None


# --- Sketch-to-Image (Optional / Safe) ---
sketch_pipe = None
print("Loading Sketch-to-Image (SDXL + T2I-Adapter, local files)...")

try:
    # Guarded import: app still launches even if diffusers isn't installed
    from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler

    # Local adapter folder path
    adapter_path = "models/t2i-adapter-sketch-sdxl"
    adapter = T2IAdapter.from_pretrained(
        adapter_path,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    # Local SDXL single file
    base_model_path = "models/sd_xl_base_1.0_0.9vae.safetensors"
    sketch_pipe = StableDiffusionXLAdapterPipeline.from_single_file(
        base_model_path,
        adapter=adapter,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    sketch_pipe.enable_attention_slicing()
    sketch_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sketch_pipe.scheduler.config)

    # Essential for 8GB VRAM GPUs
    sketch_pipe.enable_model_cpu_offload()

    print("Sketch-to-Image models loaded successfully.")
except Exception as e:
    print(f"Sketch-to-Image not available: {e}")
    sketch_pipe = None


# =========================
# Helper Functions
# =========================
def load_css(file_path="style.css"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load CSS file: {e}")
        return ""


def gen_save_folder(base_dir=SAVE_DIR):
    new_folder_name = str(uuid.uuid4())
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"Created new save folder: {new_folder_path}")
    return new_folder_path


def export_mesh_file(mesh, save_folder, file_type="glb", base_name="generated_model"):
    if file_type not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported export format: {file_type}")

    path = os.path.join(save_folder, f"{base_name}.{file_type}")
    print(f"Attempting to export mesh to: {path}")
    try:
        mesh.export(path)
        print(f"Successfully exported mesh to {path}")
        return path
    except Exception as e:
        print(f"Error during mesh export to {file_type}: {e}")
        raise


def preprocess_image(input_image: Image.Image) -> Image.Image:
    """For 3D model input images (upload or sketch-refined)."""
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    print("Processing image...")
    try:
        if rmbg_worker is not None:
            print("Removing background...")
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")
            processed_image = rmbg_worker(input_image)
        else:
            print("Background remover missing, using raw image.")
            processed_image = input_image.convert("RGB") if input_image.mode != "RGB" else input_image
    except Exception as e:
        print(f"Error in background removal: {e}")
        processed_image = input_image.convert("RGB") if input_image.mode != "RGB" else input_image

    return processed_image


def _unwrap_mesh(output_list):
    out0 = output_list[0]
    return out0[0] if isinstance(out0, list) else out0


def generate_mesh_from_pil(pil_img: Image.Image, fast: bool = False, seed: int = 0) -> trimesh.Trimesh:
    if pipeline is None:
        raise gr.Error("Model could not be loaded. Cannot generate.")
    if pil_img is None:
        raise gr.Error("No image provided.")

    temp_input_path = "temp_input.png"
    pil_img.save(temp_input_path)

    device = pipeline.device if hasattr(pipeline, "device") else "cuda"
    g = torch.Generator(device=device).manual_seed(int(seed))

    if fast:
        output_list = pipeline(
            image=temp_input_path,
            num_inference_steps=14,
            guidance_scale=4.0,
            octree_resolution=256,
            enable_pbar=False,
            generator=g,
            output_type="trimesh",
        )
    else:
        output_list = pipeline(
            image=temp_input_path,
            num_inference_steps=50,
            guidance_scale=5.0,
            octree_resolution=384,
            enable_pbar=True,
            generator=g,
            output_type="trimesh",
        )

    return _unwrap_mesh(output_list)


# =========================
# Sketch preprocessing
# =========================
def normalize_sketch_for_adapter_robust(img: Image.Image) -> Image.Image:
    """
    Robust sketch normalization to what the adapter expects:
      - Black background
      - White strokes/lines
    Handles: black-on-white paper, pencil/gray lines, inverted (white-on-black), noisy photos.
    Output: RGB.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img).astype(np.uint8)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)

    # Contrast stretch (helps faint pencil strokes)
    gmin, gmax = int(gray.min()), int(gray.max())
    if gmax > gmin:
        gray = ((gray - gmin) * (255.0 / (gmax - gmin))).clip(0, 255).astype(np.uint8)

    # Percentile-based thresholds
    p10 = int(np.percentile(gray, 10))
    p90 = int(np.percentile(gray, 90))

    # Candidate masks: dark strokes vs light strokes
    dark_strokes = gray < max(35, p10 + 18)
    light_strokes = gray > min(220, p90 - 18)

    dark_ratio = float(dark_strokes.mean())
    light_ratio = float(light_strokes.mean())

    def score(r: float) -> float:
        # Prefer sparse-ish lines, reject empty or filled
        if r < 0.002:
            return 0.0
        if r > 0.55:
            return 0.0
        # peak around ~8% coverage
        return float(np.exp(-((r - 0.08) ** 2) / (2 * (0.06 ** 2))))

    use_dark = score(dark_ratio) >= score(light_ratio)
    line_mask = dark_strokes if use_dark else light_strokes

    out = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    out[line_mask] = 255
    return Image.fromarray(out, mode="RGB")


# =========================
# Sketch-to-Image (Sketchpad)
# =========================
def process_sketch_to_preview(sketch_data, prompt):
    """Convert a sketchpad drawing to a refined image using SDXL + T2I-Adapter."""
    if sketch_pipe is None:
        raise gr.Error("Sketch-to-Image model not loaded. Check diffusers install + model paths in /models.")
    if not prompt:
        raise gr.Error("Please provide a prompt to guide the sketch conversion.")
    if sketch_data is None:
        raise gr.Error("Please draw something first.")

    try:
        # Sketchpad may return PIL directly OR dict with "composite"
        if isinstance(sketch_data, dict) and "composite" in sketch_data:
            sketch_img = sketch_data["composite"]
        elif isinstance(sketch_data, Image.Image):
            sketch_img = sketch_data
        else:
            raise gr.Error("Sketch data format not supported. Try drawing again.")

        sketch_img = sketch_img.convert("RGB")
        sketch_for_adapter = normalize_sketch_for_adapter_robust(sketch_img)

        print(f"Refining sketchpad sketch with prompt: '{prompt}'")
        refined_image = sketch_pipe(
            prompt=prompt + ", 3d model style, high quality",
            image=sketch_for_adapter,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]

        print("Sketchpad refinement complete.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return refined_image

    except Exception as e:
        print(f"Error in sketchpad processing: {e}")
        raise gr.Error(f"Sketch processing failed: {e}")


# =========================
# Sketch-to-Image (Uploaded sketch image)
# =========================
def process_uploaded_sketch_to_preview(sketch_image, prompt):
    """Convert an uploaded sketch image to a refined image using SDXL + T2I-Adapter."""
    if sketch_pipe is None:
        raise gr.Error("Sketch-to-Image model not loaded. Check diffusers install + model paths in /models.")
    if sketch_image is None:
        raise gr.Error("Please upload a sketch image first.")
    if not prompt:
        raise gr.Error("Please provide a prompt to guide the sketch conversion.")

    try:
        sketch_img = sketch_image.convert("RGB")
        sketch_for_adapter = normalize_sketch_for_adapter_robust(sketch_img)

        print(f"Refining uploaded sketch with prompt: '{prompt}'")
        refined_image = sketch_pipe(
            prompt=prompt + ", 3d model style, high quality",
            image=sketch_for_adapter,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]

        print("Uploaded sketch refinement complete.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return refined_image

    except Exception as e:
        print(f"Error in uploaded sketch processing: {e}")
        raise gr.Error(f"Uploaded sketch processing failed: {e}")


# =========================
# 2) Generation Function
# =========================
def generate_and_cache_model(
    input_image,
    sketchpad_preview_image,        # refined image from sketchpad
    uploaded_sketch_preview_image,  # refined image from uploaded sketch
    enable_xai,
    xai_method,
    xai_grid,
    xai_segments,
    xai_compactness,
    xai_mode,
    xai_max_cells,
    xai_patch_scale,
    enable_stage2,
    stage2_strength,
    enable_dim,
    dim_units,
    dim_keep_aspect,
    dim_align_bed,
    dim_w,
    dim_l,
    dim_h,
):
    global generated_mesh_cache

    if pipeline is None:
        raise gr.Error("Model could not be loaded. Cannot generate.")

    # Choose image source priority:
    # 1) uploaded sketch refined preview
    # 2) sketchpad refined preview
    # 3) normal uploaded image
    final_input = (
        uploaded_sketch_preview_image
        if uploaded_sketch_preview_image is not None
        else sketchpad_preview_image
        if sketchpad_preview_image is not None
        else input_image
    )

    if final_input is None:
        raise gr.Error("Please upload an image OR generate a sketch preview first.")

    processed_image = preprocess_image(final_input)

    print(
        f"Starting 3D shape generation... "
        f"(XAI: {enable_xai}, Stage-02: {enable_stage2}, Dimensions: {enable_dim})"
    )

    # Stage-01
    mesh = generate_mesh_from_pil(processed_image, fast=False, seed=0)
    print("Stage-01 generation complete.")

    # Stage-02 refinement
    if enable_stage2:
        try:
            lvl = int(stage2_strength)
            print(f"Stage-02 refining... strength={lvl}")
            mesh = refine_mesh(mesh, strength=lvl, enable_smoothing=True)
            print("Stage-02 refinement complete.")
        except Exception as e:
            print(f"Stage-02 refinement failed: {e}")

    # Exact dimension scaling
    if enable_dim:
        try:
            w = float(dim_w) if dim_w not in (None, "", 0) else None
            l = float(dim_l) if dim_l not in (None, "", 0) else None
            h = float(dim_h) if dim_h not in (None, "", 0) else None

            print(
                f"Scaling to dimensions (units={dim_units}, keep_aspect={dim_keep_aspect}, align_bed={dim_align_bed}) "
                f"W={w}, L={l}, H={h}"
            )
            mesh = scale_mesh_to_dimensions(
                mesh,
                target_w=w,
                target_l=l,
                target_h=h,
                units=str(dim_units),
                keep_aspect=bool(dim_keep_aspect),
                align_to_bed=bool(dim_align_bed),
            )
            print("Dimension scaling complete.")
        except Exception as e:
            print(f"Dimension scaling failed: {e}")

    # Save viewer model
    viewer_save_folder = gen_save_folder()
    viewer_model_path = export_mesh_file(mesh, viewer_save_folder, file_type="glb", base_name="viewer_model")

    # Cache mesh
    generated_mesh_cache["mesh"] = mesh
    print("Mesh cached in memory for export.")

    # XAI heatmap
    heatmap_image = None
    xai_summary = ""
    xai_table = []

    if enable_xai:
        try:
            method = str(xai_method)
            grid = int(xai_grid)
            n_segments = int(xai_segments)
            compactness = float(xai_compactness)
            max_cells = int(xai_max_cells)
            mode = str(xai_mode)
            patch_scale = float(xai_patch_scale)  # kept for UI compatibility

            print(
                f"Running XAI occlusion... "
                f"method={method}, grid={grid}, n_segments={n_segments}, "
                f"compactness={compactness}, max_cells={max_cells}, mode={mode}, patch_scale={patch_scale}"
            )

            def _xai_generate_mesh(img_pil: Image.Image) -> trimesh.Trimesh:
                return generate_mesh_from_pil(img_pil, fast=True, seed=0)

            xai_result = occlusion_sensitivity_heatmap(
                input_img=processed_image,
                generate_mesh_from_pil=_xai_generate_mesh,
                method=method,
                grid=grid,
                n_segments=n_segments,
                compactness=compactness,
                occlusion_mode=mode,
                max_regions=max_cells,
                n_samples=1024,
                seed=0,
            )

            heatmap_image = xai_result["overlay"]

            faith = xai_result.get("faithfulness", {})
            xai_summary = (
                f"Top-k mesh change: {faith.get('topk_mesh_change', 0.0):.4f}\n"
                f"Bottom-k mesh change: {faith.get('bottomk_mesh_change', 0.0):.4f}\n"
                f"Faithfulness gap: {faith.get('faithfulness_gap', 0.0):.4f}"
            )

            top_regions = xai_result.get("ranked_regions", [])[:5]
            xai_table = [[int(r.region_id), round(float(r.score), 4), int(r.area)] for r in top_regions]

            print("XAI heatmap generation complete.")
        except Exception as e:
            print(f"XAI heatmap failed: {e}")
            heatmap_image = None
            xai_summary = f"XAI failed: {e}"
            xai_table = []

    return (
        viewer_model_path,
        gr.update(interactive=True),
        gr.update(value=None, interactive=False),
        heatmap_image,
        xai_summary,
        xai_table,
    )


# =========================
# 3) Export Function
# =========================
def export_cached_model(export_format):
    global generated_mesh_cache
    mesh = generated_mesh_cache.get("mesh")

    if mesh is None:
        return gr.update(value=None, interactive=False)
    if export_format not in SUPPORTED_FORMATS:
        return gr.update(value=None, interactive=False)

    export_save_folder = gen_save_folder()
    try:
        exported_path = export_mesh_file(mesh, export_save_folder, file_type=export_format)
        print(f"Providing path for download: {exported_path}")
        return gr.update(value=exported_path, interactive=True)
    except Exception as e:
        print(f"Export failed: {e}")
        return gr.update(value=None, interactive=False)


# =========================
# 4) Gradio UI
# =========================
custom_css = load_css()

print("Launching Enhanced Morfy interface...")

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css=custom_css,
    title="✨ Morfy - AI 3D Generator",
) as morfy_app:

    with gr.Row(elem_classes="main-header"):
        gr.Markdown(
            """
            # <span style="display: block; text-align: center; font-size: 4rem; font-weight: 900; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">✨ Morfy ✨</span>
            <h3 style="text-align: center;">Transform 2D Images into Stunning 3D Models</h3>
            <div style="display: flex; align-items: center; justify-content: center; margin: 2rem 0;">
                <span class="status-indicator"></span>
                <strong style="color: rgba(255, 255, 255, 0.9); font-size: 1.05rem;">AI-Powered • Real-time Processing • Explainable AI • Sketch-to-3D</strong>
            </div>
            """
        )

    with gr.Row(equal_height=True):
        # Input Section
        with gr.Column(scale=1, elem_classes="input-card"):

            with gr.Tabs() as input_tabs:
                with gr.Tab("📸 Image Upload", id="tab_upload"):
                    input_image = gr.Image(type="pil", label="Upload Image", elem_classes="image-container", height=300)

                with gr.Tab("🎨 Sketch to Image", id="tab_sketch"):
                    sketch_prompt = gr.Textbox(label="Describe Sketch", placeholder="e.g. A wooden chair, a sports car")

                    with gr.Row(equal_height=True):
                        sketch_pad = gr.Sketchpad(type="pil", label="Draw Here", elem_classes="sketch-canvas", scale=1)
                        sketch_preview_out = gr.Image(type="pil", label="Refined Image Preview", interactive=False, scale=1)

                    sketch_preview_btn = gr.Button("🔍 1. Refine Sketch to Image", variant="secondary")

                    if sketch_pipe is None:
                        gr.Markdown(
                            "⚠️ **Sketch-to-Image not loaded.**\n\n"
                            "Check that `diffusers` is installed and your local model files exist in the `models/` folder."
                        )

                with gr.Tab("🖼️ Upload Sketch Image", id="tab_sketch_upload"):
                    sketch_upload = gr.Image(type="pil", label="Upload Sketch Image", elem_classes="image-container", height=300)
                    sketch_upload_prompt = gr.Textbox(label="Describe Sketch", placeholder="e.g. A wooden chair, a sports car")
                    sketch_upload_preview_out = gr.Image(type="pil", label="Refined Image Preview", interactive=False, height=300)
                    sketch_upload_preview_btn = gr.Button("🔍 1. Refine Uploaded Sketch", variant="secondary")

                    if sketch_pipe is None:
                        gr.Markdown(
                            "⚠️ **Sketch-to-Image not loaded.**\n\n"
                            "Check that `diffusers` is installed and your local model files exist in the `models/` folder."
                        )

            # Stage-02 Refinement
            stage2_checkbox = gr.Checkbox(
                label="Enable Stage-02 Refinement",
                value=True,
                info="Cleans mesh + fixes normals + applies smoothing to improve surface quality.",
            )
            stage2_strength = gr.Slider(0, 3, value=2, step=1, label="Refinement strength (0–3)")

            # 3D Printing Dimensions
            with gr.Accordion("📏 3D Printing Dimensions (Exact Size)", open=False):
                dim_enable = gr.Checkbox(label="Scale model to exact dimensions", value=False)
                dim_units = gr.Dropdown(choices=["mm", "cm", "in"], value="mm", label="Units")
                dim_keep_aspect = gr.Checkbox(label="Keep proportions (uniform scale)", value=True)
                dim_align_bed = gr.Checkbox(label="Align model to print bed (Z=0)", value=True)

                with gr.Row():
                    dim_w = gr.Number(value=None, label="Width (X)")
                    dim_l = gr.Number(value=None, label="Length/Depth (Y)")
                    dim_h = gr.Number(value=None, label="Height (Z)")

                gr.Markdown(
                    "Tip: Set **Height (Z)** only (e.g., 120mm) to scale uniformly. "
                    "If you turn off **Keep proportions**, X/Y/Z will scale independently (may distort)."
                )

            # XAI
            xai_checkbox = gr.Checkbox(
                label="Enable Explainable AI (Occlusion Heatmap)",
                value=False,
                info="Runs multiple fast generations to estimate which image regions influence the 3D output.",
            )
            with gr.Accordion("🧠 XAI Settings (Occlusion)", open=False):
                xai_method = gr.Dropdown(
                    choices=["superpixel", "grid"],
                    value="superpixel",
                    label="Explanation region type",
                )
                xai_grid = gr.Slider(4, 12, value=8, step=1, label="Grid size (NxN)")
                xai_segments = gr.Slider(12, 80, value=36, step=1, label="Number of superpixels")
                xai_compactness = gr.Slider(1, 30, value=10, step=1, label="Superpixel compactness")
                xai_mode = gr.Dropdown(choices=["blur", "gray", "mean"], value="blur", label="Occlusion mode")
                xai_max_cells = gr.Slider(4, 64, value=16, step=1, label="Max occlusion cells (speed control)")
                xai_patch_scale = gr.Slider(1.0, 2.0, value=1.2, step=0.1, label="Patch scale")

            generate_button = gr.Button("🚀 2. Generate 3D Model", variant="primary", size="lg")

            with gr.Group(elem_classes="examples-section"):
                gr.Markdown("### 💡 **Examples**")
                gr.Examples(examples=[["input/demo.png"], ["input/demo2.png"]], inputs=input_image, label="")

        # Output Section
        with gr.Column(scale=1, elem_classes="output-card"):
            with gr.Tabs():
                with gr.Tab("Generated 3D Model"):
                    output_model_viewer = gr.Model3D(
                        label="3D Viewer",
                        elem_classes="model3d-container",
                        height=400,
                        show_label=False,
                    )
                with gr.Tab("XAI Heatmap"):
                    xai_heatmap_output = gr.Image(label="Occlusion Sensitivity", show_label=False, height=400)
                    xai_summary_output = gr.Textbox(label="Faithfulness Summary", interactive=False)
                    xai_table_output = gr.Dataframe(
                        headers=["Region ID", "Score", "Area"],
                        datatype=["number", "number", "number"],
                        label="Top Important Regions",
                        interactive=False,
                    )

            gr.Markdown("### 💾 **Export Model**")
            with gr.Row():
                export_format_dropdown = gr.Dropdown(choices=SUPPORTED_FORMATS, value="glb", label="Format", scale=2)
                export_button = gr.Button("📤 Export", scale=1, interactive=False, variant="secondary")

            download_button = gr.DownloadButton(label="⬇️ Download", variant="secondary", interactive=False, size="lg")

    with gr.Accordion("⚙️ Advanced Settings & Information", open=False, elem_classes="accordion"):
        gr.Markdown(
            "### 🔧 Technical Specifications\n"
            "- **Model:** Hunyuan3D-2mini\n"
            "- **Sketch-to-Image:** SDXL + T2I-Adapter (local files, sketch normalized to white-on-black)\n"
            "- **Stage-02:** Mesh refinement + optional exact-dimension scaling for 3D printing\n"
            "- **XAI:** Occlusion Sensitivity Heatmap (model-agnostic)\n"
            "- **XAI speed-up:** Uses fewer diffusion steps and lower octree resolution during occlusion runs"
        )

    with gr.Row(elem_classes="footer-section"):
        gr.Markdown("### 🏗️ **Morfy** - Next-Generation AI 3D Generation Platform")

    # --- EVENTS ---
    sketch_preview_btn.click(
        fn=process_sketch_to_preview,
        inputs=[sketch_pad, sketch_prompt],
        outputs=[sketch_preview_out],
    )

    sketch_upload_preview_btn.click(
        fn=process_uploaded_sketch_to_preview,
        inputs=[sketch_upload, sketch_upload_prompt],
        outputs=[sketch_upload_preview_out],
    )

    generate_button.click(
        fn=generate_and_cache_model,
        inputs=[
            input_image,
            sketch_preview_out,
            sketch_upload_preview_out,
            xai_checkbox,
            xai_method,
            xai_grid,
            xai_segments,
            xai_compactness,
            xai_mode,
            xai_max_cells,
            xai_patch_scale,
            stage2_checkbox,
            stage2_strength,
            dim_enable,
            dim_units,
            dim_keep_aspect,
            dim_align_bed,
            dim_w,
            dim_l,
            dim_h,
        ],
        outputs=[
            output_model_viewer,
            export_button,
            download_button,
            xai_heatmap_output,
            xai_summary_output,
            xai_table_output,
        ],
    )

    export_button.click(
        fn=export_cached_model,
        inputs=[export_format_dropdown],
        outputs=[download_button],
    )

    export_format_dropdown.change(
        fn=export_cached_model,
        inputs=[export_format_dropdown],
        outputs=[download_button],
    )


if __name__ == "__main__":
    morfy_app.launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=7860)