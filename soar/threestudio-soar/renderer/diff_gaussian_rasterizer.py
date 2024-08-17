import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from torchvision.utils import save_image

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer
from threestudio.utils.ops import (
    get_cam_info_gaussian,
    get_mvp_matrix,
    get_projection_matrix,
)
from threestudio.utils.typing import *

from ..geometry.gaussian_base import BasicPointCloud, Camera
from .gaussian_batch_renderer import GaussianBatchRenderer
from .mesh_rasterizer import mesh_render, transform_pos


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


def uv_to_grid(uv_idx_map, resolution):
    """
    uv_idx_map: shape=[batch, N_uvcoords, 2], ranging between 0-1
    this function basically reshapes the uv_idx_map and shift its value range to (-1, 1) (required by F.gridsample)
    the sqaure of resolution = N_uvcoords
    """
    bs = uv_idx_map.shape[0]
    grid = uv_idx_map.reshape(bs, resolution, resolution, 2) * 2 - 1.0
    grid = grid.transpose(1, 2)
    return grid


def camera_to_mvp(K, c2w, img_wh):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W, H = img_wh
    fovx = 2.0 * torch.arctan(W / (2.0 * fx)).item()
    fovy = 2.0 * torch.arctan(H / (2.0 * fy)).item()

    cx = (cx - W / 2) / W * 2
    cy = (cy - H / 2) / H * 2

    def _get_proj_mat(
        fovx: float,
        fovy: float,
        cx: float,
        cy: float,
        znear: float = 0.001,
        zfar: float = 1000.0,
        device: str = "cpu",
    ):
        # Projection matrix takes camera points to NDC space (c2n).
        t = znear * np.tan(0.5 * fovy).item()
        b = -t
        r = znear * np.tan(0.5 * fovx).item()
        l = -r
        n = znear
        f = zfar
        return torch.tensor(
            [
                [2 * n / (r - l), 0.0, cx, 0.0],
                [0.0, 2 * n / (t - b), cy, 0.0],
                [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
                [0.0, 0.0, 1.0, 0.0],
            ],
            device=device,
        )

    proj_mat = _get_proj_mat(fovx, fovy, cx, cy, device=K.device)
    mvp = proj_mat @ torch.linalg.inv(c2w)

    return mvp


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


class Depth2Normal(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delzdelxkernel = torch.tensor(
            [
                [0.00000, 0.00000, 0.00000],
                [-1.00000, 0.00000, 1.00000],
                [0.00000, 0.00000, 0.00000],
            ]
        )
        self.delzdelykernel = torch.tensor(
            [
                [0.00000, -1.00000, 0.00000],
                [0.00000, 0.00000, 0.00000],
                [0.0000, 1.00000, 0.00000],
            ]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        delzdelxkernel = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
        delzdelx = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelxkernel, padding=1
        ).reshape(B, C, H, W)
        delzdelykernel = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
        delzdely = F.conv2d(
            x.reshape(B * C, 1, H, W), delzdelykernel, padding=1
        ).reshape(B, C, H, W)
        normal = -torch.cross(delzdelx, delzdely, dim=1)
        return normal


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
        self.normal_module = Depth2Normal()
        self.glctx = dr.RasterizeGLContext()

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
                print("[Warning] No gt_a_smpl with idx", idx)
                root, mat, scale = self.geometry.smpl_guidance(
                    points, idx=idx, zero_out=True
                )
            if idx is None:
                points_attributes = pc.attribute_field(points.detach())
            else:
                idx = idx % len(pc.smpl_guidance.smpl_parms["body_pose"])
                points_attributes = pc.attribute_field(
                    points.detach(), z=pc.latent_pose[idx]
                )
            (
                attribute_color,
                attribute_scale,
                attribute_opacity,
                attribute_quat,
                attribute_offsets,
            ) = (
                points_attributes["shs"],
                points_attributes["scales"],
                points_attributes["opacities"],
                points_attributes["quats"],
                points_attributes["offsets"],
            )
            # if stage == 1:
            #     rot = attribute_quat.clone()
            points = (
                torch.einsum("bnxy,bny->bnx", mat[..., :3, :3], points[None])
                + mat[..., :3, 3]
            )[0]
            if self.cfg.offset:
                points = points + attribute_offsets
                print("offset", attribute_offsets)
            points, T = transform_point_cloud(points, "+z,+x,+y")
            # points, T = transform_point_cloud(points, "-z,+x,-y")
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
                    #  normal_crop=normal_crop,
                )
            points_attributes = pc.attribute_field(
                points.detach(), z=None #pc.latent_pose[kwargs["gt_index"]]
            )
            (
                attribute_color,
                attribute_scale,
                attribute_opacity,
                attribute_quat,
                attribute_offsets,
            ) = (
                points_attributes["shs"],
                points_attributes["scales"],
                points_attributes["opacities"],
                points_attributes["quats"],
                points_attributes["offsets"],
            )

            points = (
                torch.einsum("bnxy,bny->bnx", mat[..., :3, :3], points[None])
                + mat[..., :3, 3]
            )[0]

            if self.cfg.offset:
                points = points + attribute_offsets
                print("offset", attribute_offsets)
            rot_mat = quaternion_to_matrix(rot)
            rot_mat = torch.matmul(mat[..., :3, :3], rot_mat)
            # _, T = transform_point_cloud(points, "-x,+y,+z")
            # rot_mat = torch.matmul(T.T, rot_mat)
            rot = matrix_to_quaternion(rot_mat)
            rot = torch.nn.functional.normalize(rot, p=2, dim=-1)[0]
            
            # if True: #not self.training:
            #     # for FS-Human
            #     points = (points - root) * scale
            #     points[..., 0] = -points[..., 0]
                
            #     normal = quaternion2rotmat(rot)[..., 2]
            #     normal[..., 0] *= -1
            #     rot = normal2rotation(normal, normalize=True)
                
            if not self.training:
                bg_color = torch.ones_like(bg_color)
            #  if not render_front:
            #      ref_z = self.geometry.smpl_guidance.normal_perfect_parms["transl"][
            #          kwargs["gt_index"], -1
            #      ]
            #      #  points_ = points.clone()
            #      points[:, 2] = 2 * ref_z - points[:, 2]
            #      #  import viser

            #      #  server = viser.ViserServer(port=30037)
            #      #  server.add_point_cloud(
            #      #      "/p1", points_.detach().cpu().numpy()[::10], (255, 0, 0)
            #      #  )
            #      #  server.add_point_cloud(
            #      #      "/p2", points.detach().cpu().numpy()[::10], (0, 255, 0)
            #      #  )
            #      #  __import__("ipdb").set_trace()
        
        

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
        # if pipe.compute_cov3D_python:
        #     cov3D_precomp = pc.get_covariance(scaling_modifier)
        # else:
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
        # breakpoint()
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

        # comp_rgb_bg = None
        # if not gt:
        #     batch_idx = kwargs["batch_idx"]
        #     rays_d = kwargs["rays_d"][batch_idx]
        #     rays_o = kwargs["rays_o"][batch_idx]
        #     comp_rgb_bg = self.background(dirs=rays_d.unsqueeze(0))
        #     _, H, W = rendered_image.shape

        #     rendered_image = rendered_image + (1 - rendered_opac) * comp_rgb_bg.reshape(
        #         H, W, 3
        #     ).permute(2, 0, 1)

        # else:
        #     batch_idx = kwargs["batch_idx"]
        #     rays_d = kwargs["gt_rays_d"][0]
        #     rays_o = kwargs["gt_rays_o"][0]
        #     comp_rgb_bg = self.background(dirs=rays_d.unsqueeze(0))
        # comp_rgb_bg = comp_rgb_bg[0]

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
            # "comp_bg": comp_rgb_bg,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (
        W - 1
    )
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (
        H - 1
    )
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(
        B, N, C, H, W, 3
    )  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)  # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz


def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(
        depth_image[None, None, None, ...], intrinsic_matrix[None, ...]
    )
    xyz_cam = xyz_cam.reshape(-1, 3)
    xyz_world = torch.cat(
        [xyz_cam, torch.ones_like(xyz_cam[..., 0:1])], axis=-1
    ) @ torch.inverse(extrinsic_matrix).transpose(0, 1)
    xyz_world = xyz_world[..., :3]

    return xyz_world


def depth_pcd2normal(xyz):
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
    top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = xyz[..., 1 : hd - 1, 2:wd, :]
    left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)
    return xyz_normal


def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix)  # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world)

    return xyz_normal

def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    # R = torch.eye(4).expand([len(q), 4, 4]).to(q.device)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
    ], -1).reshape([len(q), 3, 3]);
    return R

def rotmat2quaternion(R, normalize=False):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] + 1e-6
    r = torch.sqrt(1 + tr) / 2
    # print(torch.sum(torch.isnan(r)))
    q = torch.stack([
        r,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * r),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * r),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * r)
    ], -1)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1)
    return q

def normal2rotation(n, normalize=False):
    # construct a random rotation matrix from normal
    # it would better be positive definite and orthogonal
    n = torch.nn.functional.normalize(n)
    # w0 = torch.rand_like(n)
    w0 = torch.tensor([[1, 0, 0]]).expand(n.shape).to(n.device)
    # epsilon = 1e-6
    # dot_product = torch.abs(torch.sum(n * w0, dim=-1, keepdim=True))

    # # Check if the vectors are nearly parallel
    # if torch.any(dot_product > 1.0 - epsilon):
    #     w0 = torch.tensor([[0, 1, 0]]).expand(n.shape).to(n.device)
    R0 = w0 - torch.sum(w0 * n, -1, True) * n
    R0 *= torch.sign(R0[:, :1])
    R0 = torch.nn.functional.normalize(R0)
    R1 = torch.cross(n, R0)
    
    R1 *= torch.sign(R1[:, 1:2]) * torch.sign(n[:, 2:])
    R = torch.stack([R0, R1, n], -1)

    # q = rotmat2quaternion(R, normalize)
    q = matrix_to_quaternion(R)

    return q

