import math
from dataclasses import dataclass

import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer

from threestudio.utils.typing import *

from .gaussian_batch_renderer import GaussianBatchRenderer


@threestudio.register("gaussiansurfel-rasterizer")
class DiffGaussian(Rasterizer, GaussianBatchRenderer):
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: float = 1.0
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        offset: bool = False
        use_explicit: bool = False

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

    def forward(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        patch_size: list = [float("inf"), float("inf")],
        scaling_modifier=1.0,
        override_color=None,
        gt=False,
        render_front=True,
        #  normal_crop=False,
        stage=0,
        **kwargs,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        # bg_color *= 0
        pc = self.geometry

        points = pc.get_xyz
        rot = pc.get_rotation

        if not gt:
            idx = kwargs.get("gt_index")
            if "gt_a_smpl" in kwargs:
                with torch.no_grad():
                    root, mat, scale = self.geometry.smpl_guidance(
                        points, smpl_parms=kwargs["gt_a_smpl"], zero_out=True
                    )
            else:
                root, mat, scale = self.geometry.smpl_guidance(
                    points, idx=idx, zero_out=True
                )
            if idx is None:
                points_attributes = pc.attribute_field(points.detach())
            else:
                idx = idx % len(pc.smpl_guidance.smpl_parms["body_pose"])
                points_attributes = pc.attribute_field(
                    points.detach(), #z=pc.latent_pose[idx]
                )
            (
                attribute_color,
                attribute_scale,
                attribute_offsets,
            ) = (
                points_attributes["shs"],
                points_attributes["scales"],
                points_attributes["offsets"],
            )
            points = (
                torch.einsum("bnxy,bny->bnx", mat[..., :3, :3], points[None])
                + mat[..., :3, 3]
            )[0]
            if self.cfg.offset:
                points = points + attribute_offsets
            points, T = transform_point_cloud(points, "+z,+x,+y")
            rot_mat = quaternion_to_matrix(rot)
            rot_mat = torch.matmul(mat[..., :3, :3], rot_mat)
            rot_mat = torch.matmul(T.T, rot_mat)
            rot = matrix_to_quaternion(rot_mat)
            rot = torch.nn.functional.normalize(rot, p=2, dim=-1)[0]
        else:
            if "gt_a_smpl" in kwargs:
                with torch.no_grad():
                    root, mat, scale = self.geometry.smpl_guidance(
                        points, smpl_parms=kwargs["gt_a_smpl"])
            else:
                root, mat, scale = self.geometry.smpl_guidance(
                    points,
                    idx=kwargs["gt_index"],
                )
            points_attributes = pc.attribute_field(
                points.detach(), z=None
            )
            (
                attribute_color,
                attribute_scale,
                attribute_offsets,
            ) = (
                points_attributes["shs"],
                points_attributes["scales"],
                points_attributes["offsets"],
            )

            points = (
                torch.einsum("bnxy,bny->bnx", mat[..., :3, :3], points[None])
                + mat[..., :3, 3]
            )[0]

            if self.cfg.offset:
                points = points + attribute_offsets

            rot_mat = quaternion_to_matrix(rot)
            rot_mat = torch.matmul(mat[..., :3, :3], rot_mat)
            rot = matrix_to_quaternion(rot_mat)
            rot = torch.nn.functional.normalize(rot, p=2, dim=-1)[0]
            
            if not self.training:
                bg_color = torch.ones_like(bg_color)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # viewpoint_camera.to_device()
        # viewpoint_camera.update()

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            patch_bbox=viewpoint_camera.random_patch(patch_size[0], patch_size[1]),
            prcppoint=viewpoint_camera.prcppoint,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_front=False,  # render_front,
            sort_descending=not render_front,
            debug=False,
            config=pc.config,
        )

        raster_settings_occ = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            patch_bbox=viewpoint_camera.random_patch(patch_size[0], patch_size[1]),
            prcppoint=viewpoint_camera.prcppoint,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_front=True,  # render_front,
            sort_descending=False,
            debug=False,
            config=pc.config,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rasterizer_occ = GaussianRasterizer(raster_settings=raster_settings_occ)

        means3D = points
        means2D = screenspace_points
        opacity = pc.get_opacity
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None

        if self.cfg.use_explicit:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = attribute_scale.repeat(1, 3)  #
        
        # For FS-Human
        # if gt: 
        #     scales *= scale
        
        scales[..., -1] = -1e10
        rotations = rot
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.cfg.use_explicit:
            colors_precomp = pc.get_colors
        else:
            colors_precomp = attribute_color
        # colors_precomp = pc.get_colors #attribute_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if stage == 0:
            (
                rendered_image,
                rendered_normal,
                rendered_depth,
                rendered_opac,
                radii,
            ) = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=torch.ones_like(opacity),
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        elif stage == 1:
            (
                rendered_image,
                rendered_normal,
                rendered_depth,
                rendered_opac,
                radii,
            ) = rasterizer(
                means3D=means3D,  # .detach(),
                means2D=means2D,  # .detach(),
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=torch.ones_like(opacity),
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        occ = pc.get_occ.repeat(1, 3)
        (rendered_occ, _, _, _, _) = rasterizer_occ(
            means3D=means3D.detach(),
            means2D=means2D.detach(),
            shs=shs,
            colors_precomp=occ,
            opacities=torch.ones_like(opacity),
            scales=scales.detach(),
            rotations=rotations.detach(),
            cov3D_precomp=cov3D_precomp,
        )
        mask = rendered_opac > 1e-5
        normal_mask = mask.repeat(3, 1, 1)
        rendered_normal[~normal_mask] = rendered_normal[~normal_mask].detach()
        rendered_normal[1] *= -1
        rendered_normal[2] *= -1
        curv = normal2curv(rendered_normal, rendered_opac.detach() > 1e-5)
        rendered_normal = (rendered_normal + 1) / 2
        depth_normal = depth2normal(
            rendered_depth, rendered_opac.detach() > 1e-5, viewpoint_camera
        )
        depth_normal[1:] *= -1
        depth_normal = (depth_normal + 1) / 2

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "normal": rendered_normal,
            "depth": rendered_depth,
            "pred_normal": depth_normal,
            "mask": rendered_opac,
            "occ": rendered_occ,
            "curv": curv,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


def transform_point_cloud(xyz, dirs):
    """
    Creates a transformation matrix based on directions and applies it to the point cloud.
    xyz: NumPy array of shape (N, 3)
    dirs: String, directions for transformation (e.g., '+y,+x,+z')
    """
    valid_dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
    dir2vec = {
        "+x": np.array([1, 0, 0]),
        "+y": np.array([0, 1, 0]),
        "+z": np.array([0, 0, 1]),
        "-x": np.array([-1, 0, 0]),
        "-y": np.array([0, -1, 0]),
        "-z": np.array([0, 0, -1]),
    }
    # Initialize transformation matrix

    T = np.zeros((3, 3))

    # Map directions to transformation matrix
    for i, dir in enumerate(dirs.split(",")):
        if dir in valid_dirs:
            T[:, i] = dir2vec[dir]
        else:
            raise ValueError(f"Invalid direction: {dir}")

    T = torch.from_numpy(T).float().to(xyz.device)

    # Apply transformation
    transformed_xyz = torch.matmul(xyz, T)
    # transformed_xyz = np.dot(xyz, T)
    return transformed_xyz, T


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(
        torch.arange(0, shape[0]),
        torch.arange(0, shape[1]),
        torch.arange(0, shape[2]),
        indexing="ij",
    )
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= camera.prcppoint[0] * camera.image_width
    p[..., 1:2] -= camera.prcppoint[1] * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2, 2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode="replicate")
    mask = torch.nn.functional.pad(
        mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode="replicate"
    ).to(torch.bool)

    p_c = (p[:, 1:-1, 1:-1, :]) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:, :-2, 1:-1, :] - p_c) * mask[:, :-2, 1:-1, :]
    p_l = (p[:, 1:-1, :-2, :] - p_c) * mask[:, 1:-1, :-2, :]
    p_b = (p[:, 2:, 1:-1, :] - p_c) * mask[:, 2:, 1:-1, :]
    p_r = (p[:, 1:-1, 2:, :] - p_c) * mask[:, 1:-1, 2:, :]

    n_ul = torch.cross(p_u, p_l)
    n_ur = torch.cross(p_r, p_u)
    n_br = torch.cross(p_b, p_r)
    n_bl = torch.cross(p_l, p_b)

    # n_ul = torch.nn.functional.normalize(torch.cross(p_u, p_l), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.cross(p_r, p_u), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.cross(p_b, p_r), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.cross(p_l, p_b), dim=-1)

    # n_ul = torch.nn.functional.normalize(torch.cross(p_l, p_u), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.cross(p_u, p_r), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.cross(p_r, p_b), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.cross(p_b, p_l), dim=-1)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    # n *= -torch.sum(camVDir * camN, -1, True).sign() # no cull back

    mask = mask[0, 1:-1, 1:-1, :]

    # n = gaussian_blur(n, filter_size, 1) * mask

    n = torch.nn.functional.normalize(n, dim=-1)
    # n[..., 1] *= -1
    # n *= -1

    n = (n * mask).permute([2, 0, 1])
    return n


def normal2curv(normal, mask):
    # normal = normal.detach()
    n = normal.permute([1, 2, 0])
    m = mask.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode="replicate")
    m = torch.nn.functional.pad(
        m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode="replicate"
    ).to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:, :-2, 1:-1, :] - n_c) * m[:, :-2, 1:-1, :]
    n_l = (n[:, 1:-1, :-2, :] - n_c) * m[:, 1:-1, :-2, :]
    n_b = (n[:, 2:, 1:-1, :] - n_c) * m[:, 2:, 1:-1, :]
    n_r = (n[:, 1:-1, 2:, :] - n_c) * m[:, 1:-1, 2:, :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(1, 0, True)
    return curv