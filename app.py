import gradio as gr
import trimesh
import os
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# --- 1. Load the Model ---
print("Loading the 3D generation model...")
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-dit-v2-mini-turbo'
)
print("Model loaded successfully.")

# --- 2. Define the Final Generation Function ---
def generate_3d_model(input_image):
    if input_image is None:
        raise gr.Error("Please upload an image first.")
    
    temp_input_path = "temp_input.png"
    input_image.save(temp_input_path)

    print("Starting 3D shape generation on CPU...")
    output_list = pipeline(image=temp_input_path)
    mesh = output_list[0]
    print("Generation complete.")

    output_model_path = "output/generated_model.glb"
    os.makedirs("output", exist_ok=True)
    mesh.export(output_model_path)
    print(f"Model saved to {output_model_path}")

    return output_model_path

# --- 3. Custom CSS for Modern UI ---
custom_css = """
/* Global Styles */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    min-height: 100vh;
}

/* Header styling */
.main-header {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    text-align: center;
    color: #2d3748 !important;
}

.main-header h1 {
    color: #2d3748 !important;
    margin-bottom: 1rem;
}

.main-header h3 {
    color: #4a5568 !important;
    margin-bottom: 1.5rem;
}

.main-header p {
    color: #718096 !important;
}

.main-header strong {
    color: #667eea !important;
}

/* Card-like containers */
.input-card, .output-card {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin: 0.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    color: #2d3748 !important;
}

.input-card:hover, .output-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.45);
}

.input-card h3, .output-card h3 {
    color: #2d3748 !important;
    margin-bottom: 1rem;
}

.input-card p, .output-card p {
    color: #4a5568 !important;
}

.input-card strong, .output-card strong {
    color: #667eea !important;
}

.input-card ul, .output-card ul {
    color: #4a5568 !important;
}

.input-card li, .output-card li {
    color: #4a5568 !important;
}

/* Modern button styling */
button[variant="primary"] {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    margin: 1rem 0 !important;
}

button[variant="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Image upload area styling */
.image-container {
    border: 2px dashed #667eea !important;
    border-radius: 15px !important;
    background: rgba(102, 126, 234, 0.05) !important;
    transition: all 0.3s ease !important;
    min-height: 200px !important;
}

.image-container:hover {
    border-color: #764ba2 !important;
    background: rgba(118, 75, 162, 0.08) !important;
}

/* Examples section */
.examples-section {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 15px;
    padding: 1rem;
    margin-top: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #2d3748 !important;
}

.examples-section h3 {
    color: #2d3748 !important;
}

.examples-section p {
    color: #4a5568 !important;
}

/* Model viewer styling */
.model3d-container {
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    min-height: 400px !important;
}

/* Footer styling */
.footer-section {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1.5rem;
    margin-top: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.18);
    text-align: center;
    color: white;
}

/* Status indicators */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #4CAF50;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Accordion styling */
.accordion {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin: 1rem 0;
}

/* Responsive design */
@media (max-width: 768px) {
    .input-card, .output-card {
        margin: 0.5rem 0;
        padding: 1rem;
    }
    
    .main-header {
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
}

/* Loading animation */
.loading-spinner {
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-top: 3px solid #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""

# --- 4. Create and Launch the Modern Gradio Interface ---
print("Launching Modern Morfy interface...")

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="Morfy - AI 3D Generator"
) as morfy_app:
    
    # Header Section with modern styling
    with gr.Row(elem_classes="main-header"):
        gr.Markdown(
            """
            # 🚀 Morfy AI - Neural 3D Generator
            
            ### Transform 2D Images into Stunning 3D Models
            
            <div style="display: flex; align-items: center; justify-content: center; margin: 1rem 0;">
                <span class="status-indicator"></span>
                <strong>Powered by Hunyuan3D • CPU Optimized • Real-time Processing</strong>
            </div>
            
            Upload any image and watch AI create a detailed 3D model in minutes. Perfect for game development, 
            3D printing, AR/VR applications, and creative projects.
            """,
            elem_classes="header-content"
        )
    
    # Main content area with modern cards
    with gr.Row(equal_height=True):
        # Input Section
        with gr.Column(scale=1, elem_classes="input-card"):
            gr.Markdown("### 📸 **Input Image**")
            input_image = gr.Image(
                type="pil", 
                label="Upload Your Image",
                elem_classes="image-container",
                height=250
            )
            
            submit_button = gr.Button(
                "🎯 Generate 3D Model", 
                variant="primary",
                size="lg"
            )
            
            # Examples section with modern styling
            with gr.Group(elem_classes="examples-section"):
                gr.Markdown("### 💡 **Try These Examples**")
                gr.Examples(
                    examples=[
                        ["input/demo.png"],
                        ["input/demo2.png"],
                        ["input/demo3.png"],
                        ["input/demo5.png"],
                        ["input/demo6.png"],
                        ["input/demo7.png"]
                    ],
                    inputs=input_image,
                    label="",
                    examples_per_page=3
                )

        # Output Section
        with gr.Column(scale=1, elem_classes="output-card"):
            gr.Markdown("### 🎮 **Generated 3D Model**")
            output_model = gr.Model3D(
                label="Your 3D Creation",
                elem_classes="model3d-container",
                height=400
            )
            
            # Download section
            gr.Markdown(
                """
                **🎉 Model Ready!** 
                
                Interact with your 3D model above:
                - **Rotate**: Click and drag
                - **Zoom**: Mouse wheel or pinch
                - **Pan**: Right-click and drag
                
                The model is automatically saved as `generated_model.glb`
                """
            )

    # Advanced options (collapsible)
    with gr.Accordion("⚙️ Advanced Settings", open=False, elem_classes="accordion"):
        with gr.Row():
            gr.Markdown(
                """
                ### Processing Information
                - **Model**: Hunyuan3D-2mini (Turbo)
                - **Backend**: CPU Optimized
                - **Average Time**: 2-3 minutes
                - **Output Format**: GLB (compatible with Blender, Unity, etc.)
                - **Supported Inputs**: PNG, JPG, JPEG (max 10MB)
                """
            )

    # Modern footer
    with gr.Row(elem_classes="footer-section"):
        gr.Markdown(
            """
            ### 🏗️ **Project NeuralForge** - Feasibility Study
            
           
            
            ---
            
            *Processing times may vary based on image complexity and hardware specifications*
            """
        )
    
    # Enhanced function with progress tracking
    def generate_3d_with_progress(input_image):
        if input_image is None:
            raise gr.Error("⚠️ Please upload an image first.")
        
        temp_input_path = "temp_input.png"
        input_image.save(temp_input_path)

        print("Starting 3D shape generation on CPU...")
        output_list = pipeline(image=temp_input_path)
        mesh = output_list[0]
        
        output_model_path = "output/generated_model.glb"
        os.makedirs("output", exist_ok=True)
        mesh.export(output_model_path)
        
        print(f"Model saved to {output_model_path}")
        
        return output_model_path

    # Connect the button to the enhanced function
    submit_button.click(
        fn=generate_3d_with_progress, 
        inputs=input_image, 
        outputs=output_model
    )

# Launch with modern configuration
if __name__ == "__main__":
    morfy_app.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )