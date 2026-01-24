import gradio as gr
import trimesh
import os
import uuid
from PIL import Image
import torch

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

# NEW: Occlusion Sensitivity XAI
from xai_occlusion import occlusion_sensitivity_heatmap

# --- Global Variables ---
SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)

SUPPORTED_FORMATS = ["glb", "obj", "ply", "stl"]

# --- 1. Load the Models ---
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


# --- Helper Functions ---
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


# --- Cache for Generated Mesh ---
generated_mesh_cache = {"mesh": None}


def preprocess_image(input_image: Image.Image) -> Image.Image:
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    # --- Pre-processing: Background Removal ---
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
    """
    Your pipeline annotation says List[List[trimesh.Trimesh]], but some versions return [mesh].
    This makes it robust.
    """
    out0 = output_list[0]
    return out0[0] if isinstance(out0, list) else out0


def generate_mesh_from_pil(pil_img: Image.Image, fast: bool = False, seed: int = 0) -> trimesh.Trimesh:
    """
    PIL.Image -> trimesh.Trimesh
    fast=True uses cheaper settings for XAI occlusion runs (many repeated calls).
    """
    if pipeline is None:
        raise gr.Error("Model could not be loaded. Cannot generate.")
    if pil_img is None:
        raise gr.Error("No image provided.")

    temp_input_path = "temp_input.png"
    pil_img.save(temp_input_path)

    # Deterministic generator for comparable occlusion runs
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

    mesh = _unwrap_mesh(output_list)
    return mesh


# --- 2. Generation Function ---
def generate_and_cache_model(
    input_image,
    enable_xai,
    xai_grid,
    xai_mode,
    xai_max_cells,
    xai_patch_scale,
):
    global generated_mesh_cache

    if pipeline is None:
        raise gr.Error("Model could not be loaded. Cannot generate.")
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    processed_image = preprocess_image(input_image)

    print(f"Starting 3D shape generation... (XAI Enabled: {enable_xai})")

    # 1) Always generate once for the main output (QUALITY)
    mesh = generate_mesh_from_pil(processed_image, fast=False, seed=0)
    print("Generation complete.")

    viewer_save_folder = gen_save_folder()
    viewer_model_path = export_mesh_file(mesh, viewer_save_folder, file_type="glb", base_name="viewer_model")

    generated_mesh_cache["mesh"] = mesh
    print("Mesh cached in memory for export.")

    # 2) XAI (Occlusion Sensitivity Heatmap)
    heatmap_image = None
    if enable_xai:
        try:
            grid = int(xai_grid)
            max_cells = int(xai_max_cells)
            patch_scale = float(xai_patch_scale)
            mode = str(xai_mode)

            print(
                f"Running XAI occlusion... grid={grid}, max_cells={max_cells}, mode={mode}, patch_scale={patch_scale}"
            )

            # FAST generator for occlusion runs
            def _xai_generate_mesh(img_pil: Image.Image) -> trimesh.Trimesh:
                return generate_mesh_from_pil(img_pil, fast=True, seed=0)

            heatmap_image, _scores = occlusion_sensitivity_heatmap(
                input_img=processed_image,
                generate_mesh_from_pil=_xai_generate_mesh,
                grid=grid,
                occlusion_mode=mode,
                patch_scale=patch_scale,
                max_cells=max_cells,
                n_samples=1024,
                seed=0,
            )
            print("XAI heatmap generation complete.")
        except Exception as e:
            print(f"XAI heatmap failed: {e}")
            heatmap_image = None

    # Return: Model Path, Enable Export Btn, Disable Download Btn, Heatmap Image
    return viewer_model_path, gr.update(interactive=True), gr.update(value=None, interactive=False), heatmap_image


# --- 3. Export Function ---
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


# --- 4. Load External CSS ---
def load_css(file_path="style.css"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load CSS file: {e}")
        return ""


custom_css = load_css()

# --- 5. Create and Launch Interface ---
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
            # <span style="font-size: 4rem; font-weight: 900; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">✨ Morfy</span>
            <h3 style="text-align: center;">Transform 2D Images into Stunning 3D Models</h3>
            <div style="display: flex; align-items: center; justify-content: center; margin: 2rem 0;">
                <span class="status-indicator"></span>
                <strong style="color: rgba(255, 255, 255, 0.9); font-size: 1.05rem;">AI-Powered • Real-time Processing • Explainable AI</strong>
            </div>
            """
        )

    with gr.Row(equal_height=True):
        # Input Section
        with gr.Column(scale=1, elem_classes="input-card"):
            gr.Markdown("### 📸 **Input Image**")
            input_image = gr.Image(
                type="pil",
                label="Upload",
                elem_classes="image-container",
                height=300,
                show_label=False,
            )

            xai_checkbox = gr.Checkbox(
                label="Enable Explainable AI (Occlusion Heatmap)",
                value=False,
                info="Runs multiple fast generations to estimate which image regions influence the 3D output.",
            )

            with gr.Accordion("🧠 XAI Settings (Occlusion)", open=False):
                xai_grid = gr.Slider(4, 12, value=8, step=1, label="Grid size (NxN)")
                xai_mode = gr.Dropdown(choices=["blur", "gray"], value="blur", label="Occlusion mode")
                xai_max_cells = gr.Slider(4, 64, value=16, step=1, label="Max occlusion cells (speed control)")
                xai_patch_scale = gr.Slider(1.0, 2.0, value=1.2, step=0.1, label="Patch scale")

            generate_button = gr.Button("✨ Generate 3D Model", variant="primary", size="lg")

            with gr.Group(elem_classes="examples-section"):
                gr.Markdown("### 💡 **Examples**")
                gr.Examples(
                    examples=[["input/demo.png"], ["input/demo2.png"]],
                    inputs=input_image,
                    label="",
                )

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
                    xai_heatmap_output = gr.Image(
                        label="Occlusion Sensitivity",
                        show_label=False,
                        height=400,
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
            "- **XAI:** Occlusion Sensitivity Heatmap (model-agnostic)\n"
            "- **Metric:** Mesh change via Chamfer distance on sampled surface points\n"
            "- **XAI speed-up:** Uses fewer diffusion steps and lower octree resolution during occlusion runs"
        )

    with gr.Row(elem_classes="footer-section"):
        gr.Markdown("### 🏗️ **Morfy** - Next-Generation AI 3D Generation Platform")

    # Event Handlers
    generate_button.click(
        fn=generate_and_cache_model,
        inputs=[input_image, xai_checkbox, xai_grid, xai_mode, xai_max_cells, xai_patch_scale],
        outputs=[output_model_viewer, export_button, download_button, xai_heatmap_output],
    )

    export_button.click(
        fn=export_cached_model,
        inputs=[export_format_dropdown],
        outputs=[download_button],
    )

if __name__ == "__main__":
    morfy_app.launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=7860)
