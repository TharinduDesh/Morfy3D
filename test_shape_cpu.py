from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh
import os

print("--- RUNNING CPU-ONLY DIAGNOSTIC TEST ---")
print("Step 1: Loading the model (will use local files)...")

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini', 
    subfolder='hunyuan3d-dit-v2-mini-turbo'
)

if pipeline is None:
    print("\n--- ERROR ---")
    print("Failed to load the model pipeline.")
    exit()

# The line pipeline.to('cuda') has been REMOVED to force CPU usage.

print("Step 2: Model loaded. Starting 3D shape generation on CPU.")
print("!!! THIS WILL BE VERY SLOW. PLEASE BE PATIENT. !!!")

mesh = pipeline(image='assets/demo7.png')[0]

print("Step 3: Shape generation complete! Saving the output file...")

os.makedirs("output", exist_ok=True)
mesh.export('output/output_cpu6.glb')

print("\nAll done! CPU test successful. Check for 'output_cpu.glb'.")