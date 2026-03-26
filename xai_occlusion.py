# xai_occlusion.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from PIL import Image, ImageFilter
import trimesh
from scipy.spatial import cKDTree
from skimage.segmentation import slic, mark_boundaries
from skimage.transform import resize


@dataclass
class RegionScore:
    region_id: int
    score: float
    area: int


def chamfer_distance_mesh(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_samples: int = 1024) -> float:
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
    img = img.convert("RGB")
    base = np.array(img).astype(np.float32) / 255.0

    cmap = np.zeros_like(base)
    cmap[..., 0] = heat
    cmap[..., 1] = heat * 0.6
    cmap[..., 2] = 0.0

    out = (1 - alpha) * base + alpha * cmap
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def build_grid_segments(width: int, height: int, grid: int) -> np.ndarray:
    seg = np.zeros((height, width), dtype=np.int32)
    cell_w = width / grid
    cell_h = height / grid

    rid = 0
    for j in range(grid):
        for i in range(grid):
            x0 = int(i * cell_w)
            y0 = int(j * cell_h)
            x1 = int((i + 1) * cell_w)
            y1 = int((j + 1) * cell_h)
            seg[y0:y1, x0:x1] = rid
            rid += 1
    return seg


def build_superpixel_segments(img: Image.Image, n_segments: int = 36, compactness: float = 10.0) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    seg = slic(
        arr,
        n_segments=int(n_segments),
        compactness=float(compactness),
        start_label=0,
        channel_axis=-1,
    )
    return seg.astype(np.int32)


def mask_region(
    img: Image.Image,
    segments: np.ndarray,
    region_id: int,
    mode: str = "blur",
    blur_radius: int = 12
) -> Image.Image:
    arr = np.array(img.convert("RGB")).copy()
    mask = segments == region_id

    if not np.any(mask):
        return Image.fromarray(arr)

    if mode == "gray":
        arr[mask] = np.array([128, 128, 128], dtype=np.uint8)

    elif mode == "mean":
        mean_rgb = arr[~mask].mean(axis=0) if np.any(~mask) else np.array([128, 128, 128])
        arr[mask] = mean_rgb.astype(np.uint8)

    elif mode == "blur":
        blurred = np.array(img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        arr[mask] = blurred[mask]

    else:
        raise ValueError("mode must be one of: blur, gray, mean")

    return Image.fromarray(arr)


def segments_to_heatmap(segments: np.ndarray, region_scores: Dict[int, float]) -> np.ndarray:
    heat = np.zeros_like(segments, dtype=np.float32)
    for rid, score in region_scores.items():
        heat[segments == rid] = score
    return _normalize(heat)


def rank_regions(segments: np.ndarray, region_scores: Dict[int, float]) -> List[RegionScore]:
    out: List[RegionScore] = []
    for rid, score in region_scores.items():
        area = int(np.sum(segments == rid))
        out.append(RegionScore(region_id=int(rid), score=float(score), area=area))
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def evaluate_faithfulness(
    img: Image.Image,
    baseline_mesh: trimesh.Trimesh,
    generate_mesh_from_pil: Callable[[Image.Image], trimesh.Trimesh],
    segments: np.ndarray,
    ranked_regions: List[RegionScore],
    n_remove: int = 3,
    mask_mode: str = "blur",
    n_samples: int = 1024,
) -> Dict[str, float]:
    if len(ranked_regions) == 0:
        return {
            "topk_mesh_change": 0.0,
            "bottomk_mesh_change": 0.0,
            "faithfulness_gap": 0.0,
        }

    top_ids = [r.region_id for r in ranked_regions[:n_remove]]
    bottom_ids = [r.region_id for r in ranked_regions[-n_remove:]]

    def apply_many(region_ids: List[int]) -> Image.Image:
        out = img.copy()
        for rid in region_ids:
            out = mask_region(out, segments, rid, mode=mask_mode)
        return out

    top_img = apply_many(top_ids)
    bottom_img = apply_many(bottom_ids)

    top_mesh = generate_mesh_from_pil(top_img)
    bottom_mesh = generate_mesh_from_pil(bottom_img)

    top_score = chamfer_distance_mesh(baseline_mesh, top_mesh, n_samples=n_samples)
    bottom_score = chamfer_distance_mesh(baseline_mesh, bottom_mesh, n_samples=n_samples)

    return {
        "topk_mesh_change": float(top_score),
        "bottomk_mesh_change": float(bottom_score),
        "faithfulness_gap": float(top_score - bottom_score),
    }


def occlusion_sensitivity_heatmap(
    input_img: Image.Image,
    generate_mesh_from_pil: Callable[[Image.Image], trimesh.Trimesh],
    method: str = "superpixel",          # "superpixel" or "grid"
    grid: int = 8,
    n_segments: int = 36,
    compactness: float = 10.0,
    occlusion_mode: str = "blur",
    max_regions: int | None = 16,
    n_samples: int = 1024,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    img = input_img.convert("RGB")
    w, h = img.size

    baseline_mesh = generate_mesh_from_pil(img)

    if method == "grid":
        segments = build_grid_segments(w, h, grid)
    elif method == "superpixel":
        segments = build_superpixel_segments(img, n_segments=n_segments, compactness=compactness)
    else:
        raise ValueError("method must be 'grid' or 'superpixel'")

    region_ids = np.unique(segments).tolist()

    if max_regions is not None and max_regions < len(region_ids):
        chosen = rng.choice(region_ids, size=max_regions, replace=False)
        chosen = sorted(int(x) for x in chosen)
    else:
        chosen = region_ids

    region_scores: Dict[int, float] = {}

    for rid in chosen:
        occ = mask_region(img, segments, rid, mode=occlusion_mode)
        occ_mesh = generate_mesh_from_pil(occ)
        score = chamfer_distance_mesh(baseline_mesh, occ_mesh, n_samples=n_samples)
        region_scores[int(rid)] = float(score)

    ranked = rank_regions(segments, region_scores)
    heat = segments_to_heatmap(segments, region_scores)
    overlay = heatmap_overlay(img, heat, alpha=0.45)

    faithfulness = evaluate_faithfulness(
        img=img,
        baseline_mesh=baseline_mesh,
        generate_mesh_from_pil=generate_mesh_from_pil,
        segments=segments,
        ranked_regions=ranked,
        n_remove=min(3, max(1, len(ranked) // 5)),
        mask_mode=occlusion_mode,
        n_samples=n_samples,
    )

    boundary_vis = mark_boundaries(np.array(img).astype(np.float32) / 255.0, segments, color=(1, 1, 0))
    boundary_vis = (boundary_vis * 255).astype(np.uint8)
    boundary_img = Image.fromarray(boundary_vis)

    return {
        "overlay": overlay,
        "boundary": boundary_img,
        "segments": segments,
        "region_scores": region_scores,
        "ranked_regions": ranked,
        "faithfulness": faithfulness,
    }