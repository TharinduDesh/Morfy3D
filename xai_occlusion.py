# xai_occlusion.py
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
import trimesh
from scipy.spatial import cKDTree


def occlude_patch(img: Image.Image, x0: int, y0: int, x1: int, y1: int, mode: str = "blur") -> Image.Image:
    """mode: 'blur' or 'gray'"""
    img = img.convert("RGB")
    out = img.copy()
    patch = img.crop((x0, y0, x1, y1))

    if mode == "blur":
        patch = patch.filter(ImageFilter.GaussianBlur(radius=12))
    elif mode == "gray":
        patch = Image.new("RGB", patch.size, (128, 128, 128))
    else:
        raise ValueError("mode must be 'blur' or 'gray'")

    out.paste(patch, (x0, y0))
    return out


def chamfer_distance_mesh(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_samples: int = 1024) -> float:
    """
    Fast-ish symmetric chamfer on sampled surface points.
    Returns a single float score: higher means 'more changed'.
    """
    pa, _ = trimesh.sample.sample_surface(mesh_a, n_samples)
    pb, _ = trimesh.sample.sample_surface(mesh_b, n_samples)

    ta = cKDTree(pa)
    tb = cKDTree(pb)

    d_ab, _ = tb.query(pa, k=1)
    d_ba, _ = ta.query(pb, k=1)

    return float(np.mean(d_ab) + np.mean(d_ba))


def _normalize(scores: np.ndarray) -> np.ndarray:
    smin, smax = float(scores.min()), float(scores.max())
    if abs(smax - smin) < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - smin) / (smax - smin)).astype(np.float32)


def heatmap_overlay(img: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    heat: HxW float in [0,1]. Produces a simple red/yellow overlay (no matplotlib/opencv).
    """
    img = img.convert("RGB")
    base = np.array(img).astype(np.float32) / 255.0

    cmap = np.zeros_like(base)
    cmap[..., 0] = heat                 # red
    cmap[..., 1] = heat * 0.6           # green
    cmap[..., 2] = 0.0                  # blue

    out = (1 - alpha) * base + alpha * cmap
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def occlusion_sensitivity_heatmap(
    input_img: Image.Image,
    generate_mesh_from_pil,
    grid: int = 8,
    occlusion_mode: str = "blur",
    patch_scale: float = 1.0,
    max_cells: int = 16,
    n_samples: int = 1024,
    seed: int = 0,
):
    """
    generate_mesh_from_pil: callable(PIL.Image) -> trimesh.Trimesh
    Returns:
      overlay_img (PIL.Image), scores_grid (grid x grid float32)
    """
    rng = np.random.default_rng(seed)
    img = input_img.convert("RGB")
    W, H = img.size

    cell_w = W / grid
    cell_h = H / grid

    # Baseline
    baseline_mesh = generate_mesh_from_pil(img)

    # Choose subset of cells for speed
    cells = [(i, j) for j in range(grid) for i in range(grid)]
    if max_cells is not None and max_cells < len(cells):
        pick = rng.choice(len(cells), size=max_cells, replace=False)
        cells = [cells[k] for k in pick]

    scores = np.zeros((grid, grid), dtype=np.float32)

    for (i, j) in cells:
        x0 = int(i * cell_w)
        y0 = int(j * cell_h)
        x1 = int((i + 1) * cell_w)
        y1 = int((j + 1) * cell_h)

        if patch_scale != 1.0:
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            pw = int((x1 - x0) * patch_scale)
            ph = int((y1 - y0) * patch_scale)
            x0 = max(0, cx - pw // 2)
            x1 = min(W, cx + pw // 2)
            y0 = max(0, cy - ph // 2)
            y1 = min(H, cy + ph // 2)

        occ = occlude_patch(img, x0, y0, x1, y1, mode=occlusion_mode)
        occ_mesh = generate_mesh_from_pil(occ)

        score = chamfer_distance_mesh(baseline_mesh, occ_mesh, n_samples=n_samples)
        scores[j, i] = score

    norm = _normalize(scores)
    heat_img = Image.fromarray((norm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    heat = np.array(heat_img).astype(np.float32) / 255.0

    overlay = heatmap_overlay(img, heat, alpha=0.45)
    return overlay, scores
