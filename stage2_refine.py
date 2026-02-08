from __future__ import annotations

import numpy as np
import trimesh


def _as_trimesh(mesh) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values()]
        mesh = trimesh.util.concatenate(geoms) if geoms else None

    if isinstance(mesh, list):
        mesh = mesh[0] if mesh else None

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got: {type(mesh)}")

    return mesh


def _cleanup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()

    v = mesh.vertices
    if not np.isfinite(v).all():
        mask = np.isfinite(v).all(axis=1)
        mesh.update_vertices(mask)

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()

    try:
        mesh.fix_normals()
    except Exception:
        _ = mesh.vertex_normals

    return mesh


def _taubin_smooth(mesh: trimesh.Trimesh, iterations: int = 10, lam: float = 0.5, mu: float = -0.53) -> trimesh.Trimesh:
    mesh = mesh.copy()
    try:
        trimesh.smoothing.filter_taubin(mesh, lamb=lam, nu=mu, iterations=iterations)
        return mesh
    except Exception:
        try:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)
        except Exception:
            pass
        return mesh


def refine_mesh(mesh, strength: int = 2, enable_smoothing: bool = True) -> trimesh.Trimesh:
    mesh = _as_trimesh(mesh)
    mesh = _cleanup(mesh)

    if not enable_smoothing or int(strength) <= 0:
        return mesh

    iters = {1: 6, 2: 12, 3: 20}.get(int(strength), 12)
    mesh = _taubin_smooth(mesh, iterations=iters, lam=0.5, mu=-0.53)

    try:
        mesh.fix_normals()
    except Exception:
        pass

    return mesh


def _unit_to_mm(units: str) -> float:
    u = (units or "mm").lower().strip()
    if u == "mm":
        return 1.0
    if u == "cm":
        return 10.0
    if u in ("in", "inch", "inches"):
        return 25.4
    raise ValueError("units must be one of: mm, cm, in")


def scale_mesh_to_dimensions(
    mesh: trimesh.Trimesh,
    target_w: float | None = None,   # X
    target_l: float | None = None,   # Y
    target_h: float | None = None,   # Z
    units: str = "mm",
    keep_aspect: bool = True,
    align_to_bed: bool = True,
) -> trimesh.Trimesh:
    """
    Scales mesh so its bounding box matches requested dimensions.
    Convention: X=width, Y=length/depth, Z=height.

    keep_aspect=True  -> uniform scaling (preserves proportions)
    keep_aspect=False -> non-uniform scaling (hits exact XYZ but can distort)
    align_to_bed=True -> moves model so its lowest Z sits at Z=0 (print bed)
    """
    mesh = _as_trimesh(mesh).copy()

    mm = _unit_to_mm(units)

    # Convert targets to mm (internal)
    tx = float(target_w) * mm if target_w and float(target_w) > 0 else None
    ty = float(target_l) * mm if target_l and float(target_l) > 0 else None
    tz = float(target_h) * mm if target_h and float(target_h) > 0 else None

    if tx is None and ty is None and tz is None:
        return mesh

    bounds = mesh.bounds
    current = bounds[1] - bounds[0]  # [sx, sy, sz]
    cx, cy, cz = float(current[0]), float(current[1]), float(current[2])

    eps = 1e-9
    if cx < eps or cy < eps or cz < eps:
        raise ValueError("Mesh has near-zero size on an axis; cannot scale reliably.")

    if keep_aspect:
        s_candidates = []
        if tx is not None:
            s_candidates.append(tx / cx)
        if ty is not None:
            s_candidates.append(ty / cy)
        if tz is not None:
            s_candidates.append(tz / cz)

        s = float(np.mean(s_candidates))
        sx = sy = sz = s
    else:
        sx = (tx / cx) if tx is not None else 1.0
        sy = (ty / cy) if ty is not None else 1.0
        sz = (tz / cz) if tz is not None else 1.0

    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = sx, sy, sz
    mesh.apply_transform(S)

    if align_to_bed:
        zmin = float(mesh.bounds[0][2])
        T = np.eye(4)
        T[2, 3] = -zmin
        mesh.apply_transform(T)

    return mesh
