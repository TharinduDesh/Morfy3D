<p align="center">
  <img src="https://i.imgur.com/U539doJ.png" width="200">
</p>

<h1 align="center">🤖 Morfy: AI 3D Model Generator</h1>

<p align="center">
“Transforming 2D imagination into 3D reality.”
</p>

<p align="center">
  <img src="https://i.imgur.com/45uFcZj.png" width="800">
</p>

---

## 🔥 Abstract
Morfy is a user-friendly application that transforms 2D images into 3D models using artificial intelligence. This project is the practical implementation of a feasibility study for "Project NeuralForge," leveraging the powerful open-source Hunyuan3D model. It's designed to run on a standard computer's CPU, making 3D content creation more accessible for developers, artists, and hobbyists. The core of Morfy is a modern Gradio interface that provides a seamless user experience for generating 3D assets directly from image uploads.

---

## ✨ Core Features
- **Image-to-3D Conversion:** Upload a single 2D image of an object and generate a 3D mesh.  
- **CPU Optimized:** Runs on a standard computer's CPU, removing the need for a dedicated GPU.  
- **Modern UI:** Clean, modern, and easy to use, built with Gradio.  
- **Powered by Hunyuan3D:** Based on a state-of-the-art 3D generation model for high-quality results.  

---

## 🛠️ Getting Started

### Prerequisites
- **Git:** To clone the repository.  
- **Python:** Version 3.10 or newer.  
- **Microsoft C++ Build Tools (Windows only):** Required for compiling custom components.  

### Step 1: Clone the Repository
```bash
git clone https://github.com/TharinduDesh/Morfy3D.git
cd Morfy3D
```

### Step 2: Create Virtual Environment
```bash
python -m venv morfy_env
source morfy_env/bin/activate   # On Linux/Mac
morfy_env\Scripts\activate    # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python main_script.py
```

### Step 5: Access Morfy
After running the script, Morfy will launch a **local Gradio web interface**.  
Open the displayed link in your browser to start generating 3D models from 2D images.

---

## 📂 Project Structure
```
Morfy3D/
├── main_script.py      # Core application script
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└── assets/             # Logos, icons, and other visuals
```

---

## 🙌 Acknowledgements
- **Hunyuan3D Model** – For powering the 3D generation engine.  
- **Gradio** – For providing the seamless user interface.  
- Open-source contributors who made this project possible.
