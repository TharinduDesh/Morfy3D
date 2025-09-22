# Import the necessary pipeline from the Hunyuan3D code
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh # A library for handling 3D meshes
import os

print("Step 1: Downloading the model (this might take a while)...")

# This line downloads and loads the 'mini-turbo' model we chose.
# It will automatically cache it so it doesn't re-download every time.
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini', 
    subfolder='hunyuan3d-dit-v2-mini-turbo'
)

# Move the model to your GPU for processing
pipeline = pipeline.to('cuda')

print("Step 2: Model loaded. Starting 3D shape generation...")

# This is the main function call. It takes the input image and generates the mesh.
# 'assets/demo.png' is an example image included with the project.
mesh = pipeline(image='assets/demo.png')[0]

print("Step 3: Shape generation complete! Saving the output file...")

# Create an 'output' folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Save the generated mesh to a file named 'output.glb' in the 'output' folder.
# .glb is a standard 3D file format.
mesh.export('output/output.glb')

print("All done! Check for 'output.glb' in the 'output' folder.")