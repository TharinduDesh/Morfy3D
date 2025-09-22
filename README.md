🤖 Morfy: AI 3D Model Generator
Morfy is a user-friendly application that transforms 2D images into 3D models using artificial intelligence. This project is the practical implementation of a feasibility study for "Project NeuralForge," leveraging the powerful open-source Hunyuan3D model. It's designed to run on a local CPU, making 3D content creation more accessible.

(Recommendation: Run your app, take a nice screenshot, upload it to a site like Imgur, and add the link here to showcase your UI.)

✨ Features
Image-to-3D Conversion: Upload a single 2D image of an object and generate a 3D mesh.

CPU Optimized: The application is configured to run on a standard computer's CPU, removing the need for an expensive dedicated GPU.

Modern UI: Built with Gradio, the interface is clean, modern, and easy to use, featuring clickable examples.

Powered by Hunyuan3D: Based on a state-of-the-art 3D generation model for high-quality results.

🛠️ Setup and Installation
Follow these steps to set up and run the project on your local machine.

Prerequisites
Git: To clone the repository.

Python: Version 3.10 or newer.

Microsoft C++ Build Tools: (Windows only) Required for compiling custom components. Download here. During installation, select the "Desktop development with C++" workload.

Step 1: Clone the Repository
Open your terminal and clone this repository.

git clone [https://github.com/your-username/Morfy-AI-Generator.git](https://github.com/your-username/Morfy-AI-Generator.git)
cd Morfy-AI-Generator

Step 2: Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

Step 3: Install Dependencies
Install PyTorch: This project requires PyTorch. Install the appropriate version for your system (CPU or GPU) from the official PyTorch website.

Install Python Packages: Install the remaining packages from the requirements.txt file.

pip install -r requirements.txt
pip install -e .

Install Custom Components: These require the C++ compiler you installed earlier.

# Install custom_rasterizer
cd hy3dgen/texgen/custom_rasterizer
python setup.py install
cd ../../..

# Install differentiable_renderer
cd hy3dgen/texgen/differentiable_renderer
python setup.py install
cd ../../..

Step 4: Download Model Weights
The first time you run the application, it will automatically download the required model weights from Hugging Face (approximately 3.8 GB). This is a one-time process.

🚀 How to Run
With all the dependencies installed and your virtual environment active, you can start the application with a single command:

python app.py

After a few moments, you'll see a message in your terminal:
Running on local URL:  http://127.0.0.1:7860

Open this URL in your web browser to start using Morfy!

🗺️ Future Roadmap
This project is a fully functional prototype. Future development can focus on:

Cloud GPU Deployment: Moving the application to a service like Google Colab to enable high-speed GPU-based generation.

Texture Generation: Implementing the second stage of the Hunyuan3D model to generate high-quality, colored textures for the models.

Advanced XAI Features: Researching the model's source code to implement more advanced Explainable AI features, such as visualizing the denoising process.

🙏 Acknowledgements
This project is built upon the incredible open-source work of the Tencent Hunyuan team. Their Hunyuan3D-2 model is the core technology that powers Morfy.