import warnings
from typing import Tuple, cast

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

with warnings.catch_warnings():
    # Ignore all warnings
    warnings.simplefilter("ignore")
    CUDA_CTX = dr.RasterizeCudaContext()
    GL_CTX = dr.RasterizeGLContext()


def get_proj_mat(
    K: torch.Tensor,
    img_wh: Tuple[int, int],
    znear: float = 0.001,
    zfar: float = 1000.0,
) -> torch.Tensor:
    """
    Args:
        K: (3, 3).
        img_wh: (2,).

    Returns:
        proj_mat: (4, 4).
    """
    W, H = img_wh
    # Assume a camera model without distortion.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fovx = 2.0 * torch.arctan(W / (2.0 * fx)).item()
    fovy = 2.0 * torch.arctan(H / (2.0 * fy)).item()
    t = znear * np.tan(0.5 * fovy).item()
    b = -t
    r = znear * np.tan(0.5 * fovx).item()
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (cx - W / 2) / W * 2, 0.0],
            [0.0, 2 * n / (t - b), (cy - H / 2) / H * 2, 0.0],
            [0.0, 0.0, (f + n) / (f - n), -f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=K.device,
    )


@torch.inference_mode()
def render_mesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    w2c: torch.Tensor,
    proj_mat: torch.Tensor,
    img_wh: Tuple[int, int],
) -> dict:
    """
    Args:
        verts: (V, 3).
        faces: (F, 3).
        w2c: (4, 4).
        proj_mat: (4, 4).
        img_wh: (2,).
    """
    W, H = img_wh
    ctx = CUDA_CTX if max(W, H) <= 2048 else GL_CTX

    # Maintain two sets of typed faces for different ops.
    faces_int32 = faces.to(torch.int32)
    faces_int64 = faces.to(torch.int64)

    mvp = proj_mat @ w2c
    verts_clip = torch.einsum(
        "ij,nj->ni", mvp, F.pad(verts, pad=(0, 1), value=1.0)
    ).contiguous()
    rast, _ = cast(tuple, dr.rasterize(ctx, verts_clip[None], faces_int32, (H, W)))

    # Render mask.
    mask = (rast[..., -1:] > 0).to(torch.float32)
    mask = cast(torch.Tensor, dr.antialias(mask, rast, verts_clip, faces_int32))[
        0
    ].clamp(0, 1)

    # Render normal in camera space.
    i0, i1, i2 = faces_int64[:, 0], faces_int64[:, 1], faces_int64[:, 2]
    v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    face_normals = torch.einsum("ij,nj->ni", w2c[:3, :3], face_normals)
    vert_normals = torch.zeros_like(verts)
    vert_normals.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vert_normals.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vert_normals.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
    vert_normals = torch.where(
        torch.sum(vert_normals * vert_normals, -1, keepdim=True) > 1e-20,
        vert_normals,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vert_normals.device),
    )
    normal = F.normalize(
        cast(
            torch.Tensor,
            dr.interpolate(vert_normals[None].contiguous(), rast, faces_int32),
        )[0],
        dim=-1,
    )
    normal = cast(torch.Tensor, dr.antialias(normal, rast, verts_clip, faces_int32))[
        0
    ].clamp(-1, 1)
    # Align normal coordinate to get to the blue-pinkish side of normals.
    normal[..., [1, 2]] *= -1.0
    normal = mask * (normal + 1.0) / 2.0

    return {
        # (H, W, 3).
        "normal": normal,
    }
