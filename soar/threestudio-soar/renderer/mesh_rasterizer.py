import math
from dataclasses import dataclass

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.typing import *

# ----------------------------------------------------------------------------
# Helpers.


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def mesh_render(
    glctx,
    mtx,
    pos,
    pos_idx,
    uv,
    uv_idx,
    tex,
    resolution,
    enable_mip=False,
    max_mip_level=0,
):
    pos_clip = transform_pos(mtx, pos)
    # pos_clip = torch.einsum(
    #     "ij,nj->ni", mtx, F.pad(pos, pad=(0, 1), value=1.0)
    # ).contiguous()[None]
    # rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    rast_out, rast_out_db = dr.rasterize(
        glctx, pos_clip, pos_idx, resolution=[resolution[0], resolution[1]]
    )

    if enable_mip:
        texc, texd = dr.interpolate(
            uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs="all"
        )
        color = dr.texture(
            tex[None, ...],
            texc,
            texd,
            filter_mode="linear-mipmap-linear",
            max_mip_level=max_mip_level,
        )
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode="linear")

    alpha = torch.clamp(rast_out[..., -1:], 0, 1)
    color = color * alpha  # Mask out background.
    return color, alpha


@threestudio.register("mesh-rasterizer")
class DiffMesh(Rasterizer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 1.0
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        threestudio.info(
            "[Note] Gaussian Splatting doesn't support material and background now."
        )
        super().configure(geometry, material, background)
        self.background_tensor = torch.tensor(
            self.cfg.back_ground_color, dtype=torch.float32, device="cuda"
        )
        self.glctx = dr.RasterizeGLContext()

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        gt=False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        return
