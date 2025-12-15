import gradio as gr
import trimesh
import os
import uuid
from pathlib import Path
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# --- Global Variables ---
SAVE_DIR = 'output'
os.makedirs(SAVE_DIR, exist_ok=True)

SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

# --- 1. Load the Model ---
print("Loading the 3D generation model...")
try:
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini-turbo'
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

# --- Helper Functions ---
def gen_save_folder(base_dir=SAVE_DIR):
    new_folder_name = str(uuid.uuid4())
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"Created new save folder: {new_folder_path}")
    return new_folder_path

def export_mesh_file(mesh, save_folder, file_type='glb', base_name='generated_model'):
    if file_type not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported export format: {file_type}")
    
    path = os.path.join(save_folder, f'{base_name}.{file_type}')
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

# --- 2. Generation Function ---
def generate_and_cache_model(input_image):
    global generated_mesh_cache
    if pipeline is None:
         raise gr.Error("Model could not be loaded. Cannot generate.")
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    temp_input_path = "temp_input.png"
    input_image.save(temp_input_path)

    print("Starting 3D shape generation...")
    output_list = pipeline(image=temp_input_path)
    mesh = output_list[0]
    print("Generation complete.")

    viewer_save_folder = gen_save_folder()
    viewer_model_path = export_mesh_file(mesh, viewer_save_folder, file_type='glb', base_name='viewer_model')

    generated_mesh_cache["mesh"] = mesh
    print("Mesh cached in memory for export.")

    return viewer_model_path, gr.update(interactive=True), gr.update(value=None, interactive=False)

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
        mesh_to_export = mesh
        exported_path = export_mesh_file(mesh_to_export, export_save_folder, file_type=export_format)
        print(f"Providing path for download: {exported_path}")
        return gr.update(value=exported_path, interactive=True)
    except Exception as e:
        print(f"Export failed: {e}")
        return gr.update(value=None, interactive=False)

# --- 4. Premium Modern CSS ---
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --dark-bg: #0a0a1f;
    --darker-bg: #050510;
    --card-bg: rgba(255, 255, 255, 0.03);
    --card-hover-bg: rgba(255, 255, 255, 0.06);
    --card-border: rgba(255, 255, 255, 0.08);
    --card-border-hover: rgba(102, 126, 234, 0.4);
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.7);
    --text-tertiary: rgba(255, 255, 255, 0.5);
    --accent-primary: #667eea;
    --accent-secondary: #764ba2;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.15);
    --shadow-md: 0 4px 24px rgba(0, 0, 0, 0.25);
    --shadow-lg: 0 10px 50px rgba(0, 0, 0, 0.35);
    --shadow-glow: 0 0 30px rgba(102, 126, 234, 0.3);
    --blur-effect: blur(20px);
    --border-radius-sm: 12px;
    --border-radius-md: 16px;
    --border-radius-lg: 24px;
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.gradio-container {
    background: var(--darker-bg) !important;
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(102, 126, 234, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(118, 75, 162, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(240, 147, 251, 0.05) 0%, transparent 50%);
    min-height: 100vh;
    padding: 2.5rem !important;
}

/* Animated Background Elements */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        linear-gradient(45deg, transparent 30%, rgba(102, 126, 234, 0.03) 30%, rgba(102, 126, 234, 0.03) 70%, transparent 70%),
        linear-gradient(-45deg, transparent 30%, rgba(118, 75, 162, 0.03) 30%, rgba(118, 75, 162, 0.03) 70%, transparent 70%);
    background-size: 100px 100px;
    pointer-events: none;
    z-index: 0;
}

/* Header Section */
.main-header {
    background: var(--card-bg);
    backdrop-filter: var(--blur-effect);
    -webkit-backdrop-filter: var(--blur-effect);
    border: 1px solid var(--card-border);
    border-radius: var(--border-radius-lg);
    padding: 3.5rem 2.5rem;
    margin-bottom: 2.5rem;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
    z-index: 1;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    background: var(--primary-gradient);
}

.main-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

.main-header h1 {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3.5rem !important;
    font-weight: 900 !important;
    margin-bottom: 0.75rem !important;
    letter-spacing: -0.03em;
    line-height: 1.1 !important;
    text-align: center;
}

.main-header h1 span {
    display: inline-block;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.main-header h3 {
    color: var(--text-secondary) !important;
    font-size: 1.4rem !important;
    font-weight: 400 !important;
    margin-bottom: 2rem !important;
    letter-spacing: -0.01em;
}

.main-header p {
    color: var(--text-secondary) !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    max-width: 900px;
    margin: 0 auto;
}

/* Status Indicator */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    background: var(--success-color);
    border-radius: 50%;
    margin-right: 0.75rem;
    animation: pulse-glow 2s infinite;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
}

@keyframes pulse-glow {
    0% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    50% {
        box-shadow: 0 0 0 12px rgba(16, 185, 129, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

/* Card Styles */
.input-card, .output-card {
    background: var(--card-bg) !important;
    backdrop-filter: var(--blur-effect);
    -webkit-backdrop-filter: var(--blur-effect);
    border: 1px solid var(--card-border) !important;
    border-radius: var(--border-radius-lg) !important;
    padding: 2.5rem !important;
    box-shadow: var(--shadow-md);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.input-card::before, .output-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: var(--primary-gradient);
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
    z-index: 0;
}

.input-card:hover, .output-card:hover {
    border-color: var(--card-border-hover) !important;
    box-shadow: var(--shadow-lg), var(--shadow-glow);
    transform: translateY(-4px);
}

.input-card:hover::before, .output-card:hover::before {
    opacity: 0.03;
}

/* Section Headers */
.input-card h3, .output-card h3 {
    color: var(--text-primary) !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    margin-bottom: 1.5rem !important;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid rgba(102, 126, 234, 0.2);
}

.input-card h3::before {
    content: '';
    width: 4px;
    height: 24px;
    background: var(--primary-gradient);
    border-radius: 2px;
}

/* Image Container */
.image-container {
    border-radius: var(--border-radius-md) !important;
    overflow: hidden;
    border: 2px solid var(--card-border) !important;
    transition: all 0.3s ease;
    background: rgba(0, 0, 0, 0.3) !important;
}

.image-container:hover {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 25px rgba(102, 126, 234, 0.4);
    transform: scale(1.01);
}

/* Fix for image upload display */
.image-container img {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}

/* Hide the upload placeholder when image is present */
.image-container [data-testid="image"] {
    min-height: 300px !important;
}

.image-container .upload-container {
    min-height: 300px !important;
}

/* Ensure single image display */
.image-container > div {
    display: block !important;
}

.image-container [data-testid="image"] > div {
    width: 100% !important;
}

/* Better image upload area */
.image-container .image-container, 
.image-container [data-testid="image"],
.image-container .wrap {
    background: transparent !important;
}

/* Hide upload icon when image is loaded */
.image-container:has(img) .upload-text {
    display: none !important;
}

/* Center the upload area content */
.image-container .upload-text {
    color: var(--text-secondary) !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}

/* Upload area styling */
.image-container [data-testid="image"] {
    background: rgba(102, 126, 234, 0.05) !important;
    border: 2px dashed rgba(102, 126, 234, 0.3) !important;
    border-radius: var(--border-radius-md) !important;
    transition: all 0.3s ease !important;
}

.image-container [data-testid="image"]:hover {
    background: rgba(102, 126, 234, 0.08) !important;
    border-color: rgba(102, 126, 234, 0.5) !important;
}

/* Upload button styling */
.upload-button {
    background: var(--primary-gradient) !important;
    border: none !important;
    color: white !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: var(--border-radius-sm) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.upload-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
}

/* Image preview improvements */
.image-container img {
    border-radius: var(--border-radius-sm) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

/* Clear/Remove button */
.image-container button[aria-label="Clear"],
.image-container button[aria-label="Remove Image"] {
    background: rgba(239, 68, 68, 0.9) !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    transition: all 0.3s ease !important;
}

.image-container button[aria-label="Clear"]:hover,
.image-container button[aria-label="Remove Image"]:hover {
    background: rgba(239, 68, 68, 1) !important;
    transform: scale(1.1);
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.5) !important;
}

/* Button Styles */
button {
    border-radius: var(--border-radius-sm) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: none !important;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
}

button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

button:hover::before {
    width: 300px;
    height: 300px;
}

button[variant="primary"] {
    background: var(--primary-gradient) !important;
    color: white !important;
    padding: 1.1rem 2.5rem !important;
    font-size: 1.05rem !important;
}

button[variant="primary"]:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md), 0 0 35px rgba(102, 126, 234, 0.5);
}

button[variant="primary"]:active {
    transform: translateY(-1px);
}

button[variant="secondary"] {
    background: rgba(255, 255, 255, 0.08) !important;
    color: var(--text-primary) !important;
    backdrop-filter: var(--blur-effect);
    border: 1px solid var(--card-border) !important;
}

button[variant="secondary"]:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: var(--accent-primary) !important;
    box-shadow: var(--shadow-sm), 0 0 20px rgba(102, 126, 234, 0.3);
}

button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    transform: none !important;
}

/* Export Section Styling */
.export-section {
    background: rgba(102, 126, 234, 0.05);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: var(--border-radius-md);
    padding: 2rem;
    margin-top: 2rem;
}

.export-section h3 {
    color: var(--text-primary) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin-bottom: 1.5rem !important;
}

/* Dropdown Styling */
select, .dropdown {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--border-radius-sm) !important;
    color: var(--text-primary) !important;
    padding: 0.85rem 1rem !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease;
    cursor: pointer;
}

select:hover, .dropdown:hover {
    border-color: var(--accent-primary) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 0 15px rgba(102, 126, 234, 0.2);
}

select:focus, .dropdown:focus {
    outline: none;
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

/* Model3D Container */
.model3d-container {
    border-radius: var(--border-radius-md) !important;
    overflow: hidden;
    border: 2px solid var(--card-border) !important;
    background: rgba(0, 0, 0, 0.4) !important;
    box-shadow: inset 0 2px 15px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}

.model3d-container:hover {
    border-color: rgba(102, 126, 234, 0.3) !important;
    box-shadow: inset 0 2px 15px rgba(0, 0, 0, 0.4), 0 0 25px rgba(102, 126, 234, 0.2);
}

/* Examples Section */
.examples-section {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--border-radius-md) !important;
    padding: 1.75rem !important;
    margin-top: 2rem;
    transition: all 0.3s ease;
}

.examples-section:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    border-color: rgba(102, 126, 234, 0.2) !important;
}

.examples-section h3 {
    color: var(--text-primary) !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin-bottom: 1.25rem !important;
}

/* Info Box Styles */
.info-box {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
    border-left: 4px solid var(--accent-primary);
    border-radius: var(--border-radius-sm);
    padding: 1.5rem;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-sm);
}

.info-box strong {
    color: var(--text-primary);
    display: block;
    margin-bottom: 0.75rem;
    font-size: 1.1rem;
}

.info-box p {
    color: var(--text-secondary) !important;
    line-height: 1.6;
    margin: 0.5rem 0;
}

/* Accordion */
.accordion {
    background: var(--card-bg) !important;
    backdrop-filter: var(--blur-effect);
    border: 1px solid var(--card-border) !important;
    border-radius: var(--border-radius-md) !important;
    margin-top: 2.5rem;
    overflow: hidden;
    transition: all 0.3s ease;
}

.accordion:hover {
    border-color: rgba(102, 126, 234, 0.3) !important;
}

.accordion summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    padding: 1.75rem !important;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 1.05rem;
}

.accordion summary:hover {
    background: rgba(255, 255, 255, 0.05);
}

.accordion[open] {
    box-shadow: var(--shadow-md);
}

.accordion[open] summary {
    border-bottom: 1px solid var(--card-border);
    background: rgba(102, 126, 234, 0.05);
}

/* Footer */
.footer-section {
    background: var(--card-bg);
    backdrop-filter: var(--blur-effect);
    border: 1px solid var(--card-border);
    border-radius: var(--border-radius-md);
    padding: 2.5rem;
    margin-top: 3rem;
    text-align: center;
}

.footer-section h3 {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    line-height: 1.8;
}

/* Labels */
label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.75rem !important;
    display: block;
}

/* Markdown Content */
.gr-form p, .gr-box p {
    color: var(--text-secondary) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

/* Separator */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--card-border), transparent);
    margin: 2rem 0;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--accent-primary);
    border-radius: 10px;
    border: 2px solid var(--dark-bg);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-secondary);
}

/* Loading State */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.loading {
    animation: shimmer 2s infinite;
    background: linear-gradient(
        to right,
        rgba(255, 255, 255, 0.05) 0%,
        rgba(255, 255, 255, 0.15) 50%,
        rgba(255, 255, 255, 0.05) 100%
    );
    background-size: 1000px 100%;
}

/* Responsive Design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1.25rem !important;
    }
    
    .main-header h1 {
        font-size: 2.25rem !important;
    }
    
    .main-header {
        padding: 2.5rem 1.75rem !important;
    }
    
    .input-card, .output-card {
        padding: 1.75rem !important;
    }
    
    button[variant="primary"] {
        padding: 0.9rem 2rem !important;
        font-size: 0.95rem !important;
    }
}

/* Utility Classes */
.text-gradient {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.glow-on-hover:hover {
    box-shadow: var(--shadow-glow);
}
"""

# --- 5. Create and Launch Interface ---
print("Launching Enhanced Morfy interface...")

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
    ),
    css=custom_css,
    title="✨ Morfy - AI 3D Generator"
) as morfy_app:

    # Header
    with gr.Row(elem_classes="main-header"):
        gr.Markdown(
            """
            # <span style="font-size: 4rem; font-weight: 900; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">✨ Morfy</span>
            
            <h3 style="text-align: center;">Transform 2D Images into Stunning 3D Models</h3>
            
            <div style="display: flex; align-items: center; justify-content: center; margin: 2rem 0;">
                <span class="status-indicator"></span>
                <strong style="color: rgba(255, 255, 255, 0.9); font-size: 1.05rem;">AI-Powered • Real-time Processing • Production Ready</strong>
            </div>
            
            <p style="text-align: center; max-width: 900px; margin: 0 auto; color: rgba(255, 255, 255, 0.7); font-size: 1.05rem; line-height: 1.7;">
            Upload any image and watch AI create a detailed 3D model in minutes. Perfect for game development,
            3D printing, AR/VR applications, and creative projects. Powered by Hunyuan3D-2mini Turbo technology.
            </p>
            """,
            elem_classes="header-content"
        )

    # Main Content
    with gr.Row(equal_height=True):
        # Input Section
        with gr.Column(scale=1, elem_classes="input-card"):
            gr.Markdown("### 📸 **Input Image**")
            gr.Markdown(
                """
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                     border-left: 3px solid #667eea; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <strong style="color: #667eea;">💡 Tip:</strong> 
                    <span style="color: rgba(255, 255, 255, 0.8);">Upload a clear, well-lit image for best results</span>
                </div>
                """
            )
            input_image = gr.Image(
                type="pil",
                label="Upload Your Image",
                elem_classes="image-container",
                height=400,
                show_label=False,
                sources=["upload", "clipboard"],
                show_download_button=False
            )

            generate_button = gr.Button(
                "✨ Generate 3D Model",
                variant="primary",
                size="lg",
                scale=1
            )

            # Examples
            with gr.Group(elem_classes="examples-section"):
                gr.Markdown("### 💡 **Try These Examples**")
                gr.Examples(
                    examples=[
                        ["input/demo.png"], ["input/demo2.png"], ["input/demo3.png"],
                        ["input/demo5.png"], ["input/demo6.png"], ["input/demo7.png"]
                    ],
                    inputs=input_image,
                    label="",
                    examples_per_page=3
                )

        # Output Section
        with gr.Column(scale=1, elem_classes="output-card"):
            gr.Markdown("### 🎮 **Generated 3D Model**")
            output_model_viewer = gr.Model3D(
                label="Interactive 3D Viewer",
                elem_classes="model3d-container",
                height=400,
                show_label=False
            )

            gr.Markdown(
                """
                <div class="info-box">
                <strong>🎉 Model Generated Successfully!</strong>
                <p><strong>Interactive Controls:</strong></p>
                <p>• 🔄 <strong>Rotate:</strong> Click and drag to rotate the model</p>
                <p>• 🔍 <strong>Zoom:</strong> Use mouse wheel or pinch gesture</p>
                <p>• 🖐️ <strong>Pan:</strong> Right-click and drag to move</p>
                </div>
                """
            )

            # Export Controls
            gr.Markdown("---")
            gr.Markdown("### 💾 **Export Model**")
            
            with gr.Row():
                export_format_dropdown = gr.Dropdown(
                    label="🎯 Select Export Format",
                    choices=SUPPORTED_FORMATS,
                    value='glb',
                    scale=2,
                    info="Choose your preferred 3D file format"
                )
                export_button = gr.Button(
                    "📤 Export",
                    scale=1,
                    interactive=False,
                    variant="secondary"
                )

            download_button = gr.DownloadButton(
                label="⬇️ Download Exported File",
                variant="secondary",
                interactive=False,
                size="lg"
            )

    # Advanced Settings
    with gr.Accordion("⚙️ Advanced Settings & Information", open=False, elem_classes="accordion"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    ### 🔧 **Technical Specifications**
                    
                    **Model Information:**
                    - **Architecture:** Hunyuan3D-2mini (Turbo Edition)
                    - **Backend:** CPU Optimized / GPU Accelerated (Auto-detect)
                    - **Processing Mode:** Flow Matching Pipeline
                    
                    **Performance Metrics:**
                    - **GPU Processing:** 10-30 seconds per model
                    - **CPU Processing:** 2-5 minutes per model
                    - **Model Quality:** High-fidelity 3D reconstruction
                    
                    **Supported Formats:**
                    - **GLB:** Recommended for web/AR (optimized)
                    - **OBJ:** Universal format with texture support
                    - **PLY:** Point cloud and mesh data
                    - **STL:** Perfect for 3D printing
                    
                    ### 📊 **Best Practices for Optimal Results**
                    
                    ✅ **Image Requirements:**
                    - High-resolution images (min 512x512px)
                    - Well-lit, clear subject matter
                    - Clean background separation
                    - Front-facing or 3/4 view angles
                    
                    ✅ **What Works Best:**
                    - Product photography
                    - Character portraits
                    - Architectural elements
                    - Simple geometric objects
                    
                    ❌ **Avoid:**
                    - Extreme angles or occlusions
                    - Heavy motion blur
                    - Complex transparent materials
                    - Very dark or overexposed images
                    
                    ### 📁 **File Management**
                    
                    - Maximum upload size: 10MB
                    - Supported formats: PNG, JPG, JPEG
                    - Models are temporarily cached for export
                    - Each export creates a unique file
                    """
                )

    # Footer
    with gr.Row(elem_classes="footer-section"):
        gr.Markdown(
            """
            ### 🏗️ **Morfy** - Next-Generation AI 3D Generation Platform
            
            **Transform Any 2D Image into High-Quality 3D Models** | Powered by Hunyuan3D Technology
            
            ---
            
            *Processing times may vary based on image complexity and hardware specifications*
            
            *For optimal performance, we recommend using GPU-accelerated systems*
            
            Built with ❤️ using Gradio & Tencent Hunyuan3D | © 2024 Morfy AI
            """
        )

    # Event Handlers
    generate_button.click(
        fn=generate_and_cache_model,
        inputs=[input_image],
        outputs=[output_model_viewer, export_button, download_button]
    )

    export_button.click(
        fn=export_cached_model,
        inputs=[export_format_dropdown],
        outputs=[download_button]
    )

# Launch Configuration
if __name__ == "__main__":
    morfy_app.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )