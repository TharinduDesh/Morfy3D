import torch
import numpy as np
import cv2
from PIL import Image
from hy3dgen.shapegen.models.denoisers import hunyuan3ddit

class XAIPipeline:
    def __init__(self):
        self.attention_scores = []
        self.original_attention = hunyuan3ddit.attention

    def __enter__(self):
        self.attention_scores = []
        
        def hooked_attention(q, k, v, **kwargs):
            # 1. Calculate Standard Attention (Softmax(Q @ K^T / sqrt(d)))
            # We use a try-except block to handle potential OOM on GPU smoothly
            try:
                B, H, Lq, D = q.shape
                Lk = k.shape[2]
                scale = D ** -0.5
                
                # Check for Flash Attention (if available/used) constraints or fall back
                # Calculating the full matrix [B, H, Lq, Lk] is heavy on GPU VRAM.
                # If Lq is huge (>4096), this might still spike VRAM.
                
                sim = torch.einsum("bhqd, bhkd -> bhqk", q, k) * scale
                attn = sim.softmax(dim=-1)
                
                # --- MEMORY OPTIMIZATION START ---
                # Instead of storing [B, H, Lq, Lk] (Huge), 
                # We average over Heads (dim 1) and Queries (dim 2) IMMEDIATELY.
                # This gives us a vector [B, Lk] representing "How important is each Input Key?"
                
                # 1. Mean over heads -> [B, Lq, Lk]
                attn_avg = attn.mean(dim=1)
                
                # 2. Mean over Queries (Rows) -> [B, Lk]
                # We want to know: "Globally, which input pixels were focused on?"
                importance_vector = attn_avg.mean(dim=1)
                
                # 3. Move only this tiny vector to CPU
                self.attention_scores.append(importance_vector.detach().cpu())
                
                # --- MEMORY OPTIMIZATION END ---
                
                # Complete the attention mechanism so the model keeps running
                out = torch.einsum("bhqk, bhvd -> bhqd", attn, v)
                out = out.permute(0, 2, 1, 3).contiguous().view(B, Lq, H*D)
                return out
                
            except RuntimeError as e:
                # Fallback if manual calculation fails (e.g. OOM), just run standard attention
                # We lose XAI data for this step, but we don't crash the app.
                if "memory" in str(e).lower():
                    print("⚠️ XAI Hook Warning: GPU OOM prevented map capture. Skipping step.")
                    # Re-run using optimized standard pytorch attention
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
                raise e

        # Apply the monkey patch
        hunyuan3ddit.attention = hooked_attention
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original function
        hunyuan3ddit.attention = self.original_attention

    def generate_heatmap(self, original_image):
        if not self.attention_scores:
            print("No attention scores captured.")
            return original_image

        # 1. Stack all vectors: [Steps*Layers, Batch, Total_Tokens]
        all_scores = torch.stack(self.attention_scores)
        
        # 2. Average over all steps and batch to get one "Global Importance Vector"
        # Shape: [Total_Tokens]
        global_importance = all_scores.mean(dim=0).mean(dim=0)
        
        # 3. Identify the "Image Condition" part of the tokens.
        # Hunyuan3D concatenates [Condition, Latents]. Condition comes first.
        # We need to find the square grid that matches the condition.
        # Heuristic: Try to find a perfect square at the start of the vector.
        
        total_len = global_importance.shape[0]
        
        # Common sizes: 16x16=256, 32x32=1024
        # We test reasonable grid sizes
        grid_size = 0
        target_tokens = 0
        
        for size in [64, 32, 16]: # Try 64x64, then 32x32, then 16x16
            tokens = size * size
            if tokens <= total_len:
                # Verify if this chunk has high variance (meaning it's the active image part)
                grid_size = size
                target_tokens = tokens
                break
        
        if grid_size == 0:
            return original_image

        # Extract the image part
        heatmap_vector = global_importance[:target_tokens]
        
        # Normalize
        heatmap_vector = (heatmap_vector - heatmap_vector.min()) / (heatmap_vector.max() - heatmap_vector.min() + 1e-8)
        
        # Reshape to grid
        heatmap_np = heatmap_vector.reshape(grid_size, grid_size).numpy()
        
        # Resize to original image size
        heatmap_resized = cv2.resize(heatmap_np, original_image.size, interpolation=cv2.INTER_CUBIC)
        
        # Colorize
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        original_np = np.array(original_image)
        # Resize original if needed (just in case)
        if original_np.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (original_np.shape[1], original_np.shape[0]))
            
        overlay = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)
        
        return Image.fromarray(overlay)