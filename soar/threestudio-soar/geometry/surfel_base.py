#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import quaternion_to_matrix
from simple_knn._C import distCUDA2
from torch import nn
from torch.utils.cpp_extension import load

import threestudio
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.typing import *

# from scene.colmap_loader import qvec2rotmat
from ..utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    normal2rotation,
    quaternion2rotmat,
    strip_symmetric,
)
from ..utils.graphics_utils import BasicPointCloud
from ..utils.image_utils import world2scrn
from ..utils.sh_utils import RGB2SH, SH2RGB
from ..utils.system_utils import mkdir_p
from .gaussian_io import GaussianIO
from .sdf_fields import HashMLPSDFField


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

    # Apply transformation
    transformed_xyz = np.dot(xyz, T)
    return transformed_xyz, T


@threestudio.register("gaussiansurfel-base")
class GaussianSurfelModel(BaseGeometry, GaussianIO):
    @dataclass
    class Config(BaseGeometry.Config):
        max_num: int = 500000
        sh_degree: int = 0
        position_lr_init: Any = 0.001
        position_lr_final: Any = 0.00001
        position_lr_delay_mult: Any = 0.01
        position_lr_max_steps: Any = 2000
        camera_lr: Any = 0.0
        scale_lr: Any = 0.003
        feature_lr: Any = 0.01
        opacity_lr: Any = 0.05
        scaling_lr: Any = 0.005
        rotation_lr: Any = 0.005
        pred_normal: bool = False
        normal_lr: Any = 0.001
        field_lr: Any = 0.01
        occ_lr: Any = 0.01
        background_lr: Any = 0.01
        latent_pose_lr: Any = 0.01

        densification_interval: int = 50
        prune_interval: int = 50
        opacity_reset_interval: int = 100000
        densify_from_iter: int = 100
        prune_from_iter: int = 100
        densify_until_iter: int = 2000
        prune_until_iter: int = 2000
        densify_grad_threshold: Any = 0.01
        min_opac_prune: Any = 0.005
        split_thresh: Any = 0.02
        radii2d_thresh: Any = 1000

        sphere: bool = True  # False
        prune_big_points: bool = False
        color_clip: Any = 2.0

        geometry_convert_from: str = ""
        load_ply_only_vertex: bool = False
        init_num_pts: int = 100
        pc_init_radius: float = 0.8
        opacity_init: float = 0.1

        shap_e_guidance_config: dict = field(default_factory=dict)

        smpl_guidance_config: dict = field(default_factory=dict)

    cfg: Config

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def configure(self) -> None:
        super().configure()
        self.active_sh_degree = 0
        self.max_sh_degree = self.cfg.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.scale_gradient_accum = torch.empty(0)
        self.rot_gradient_accum = torch.empty(0)
        self.opac_gradient_accum = torch.empty(0)

        self.percent_dense = 0.01
        self.spatial_lr_scale = 0
        # try:
        #     self.config = [args.surface, args.normalize_depth, args.perpix_depth]
        # except AttributeError:
        self.config = [True, True, True]
        self.setup_functions()
        self.utils_mod = load(
            name="cuda_utils",
            sources=[
                "custom/threestudio-soar/utils/ext.cpp",
                "custom/threestudio-soar/utils/cuda_utils.cu",
            ],
            extra_cuda_cflags=['-allow-unsupported-compiler', '--expt-relaxed-constexpr']
        )
        self.opac_reset_record = [0, 0]

        self.denom = torch.empty(0)
        # if self.cfg.pred_normal:
        #     self._normal = torch.empty(0)
        self.optimizer = None
        self.setup_functions()

        if self.cfg.geometry_convert_from.startswith("shap-e:"):
            shap_e_guidance = threestudio.find("shap-e-guidance")(
                self.cfg.shap_e_guidance_config
            )
            # print('cfg:', self.cfg.shap_e_guidance_config, self.cfg); exit()
            prompt = self.cfg.geometry_convert_from[len("shap-e:") :]
            xyz, color = shap_e_guidance(prompt)

            dirs = "-y,+x,+z"
            threestudio.info("Transforming point cloud with directions: %s" % dirs)
            xyz, _ = transform_point_cloud(xyz, dirs)
            xyz, color = self.add_points(xyz, color)
            pcd = BasicPointCloud(
                points=xyz * self.cfg.pc_init_radius,
                colors=color,
                normals=np.zeros((xyz.shape[0], 3)),
            )
            self.create_from_pcd(pcd, 10)
            self.training_setup()

        # Support Initialization from OpenLRM, Please see https://github.com/Adamdad/threestudio-lrm
        elif self.cfg.geometry_convert_from.startswith("lrm:"):
            lrm_guidance = threestudio.find("lrm-mvdream-guidance")(
                self.cfg.shap_e_guidance_config
            )
            prompt = self.cfg.geometry_convert_from[len("lrm:") :]
            xyz, color, trans = lrm_guidance(prompt)

            if trans:
                dirs = "-y,+x,+z"
                threestudio.info("Transforming point cloud with directions: %s" % dirs)
                xyz, _ = transform_point_cloud(xyz, dirs)
            xyz, color = self.add_points(xyz, color)
            pcd = BasicPointCloud(
                points=xyz * self.cfg.pc_init_radius,
                colors=color,
                normals=np.zeros((xyz.shape[0], 3)),
            )
            self.create_from_pcd(pcd, 10)
            self.training_setup()

        elif self.cfg.geometry_convert_from.startswith("smpl:"):
            self.cfg.smpl_guidance_config.update(
                {"seq": self.cfg.geometry_convert_from[len("smpl:") :]}
            )
            smpl_guidance = threestudio.find("smpl-guidance")(
                self.cfg.smpl_guidance_config
            )
            self.smpl_guidance = smpl_guidance
            xyz = self.smpl_guidance.query_points[0].detach().cpu().numpy()
            # xyz = np.zeros_like(self.smpl_guidance.vertices.detach().cpu().numpy())
            dirs = "+x,+y,+z"
            # dirs = "+z,+x,+y"

            threestudio.info("Transforming point cloud with directions: %s" % dirs)
            positions, _ = transform_point_cloud(xyz, dirs)

            shs = np.random.random((positions.shape[0], 3)) / 255.0
            C0 = 0.28209479177387814
            colors = shs * C0 + 0.5
            colors = np.ones_like(colors) * 0.5
            normals = np.zeros_like(positions)
            pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
            self.cano_texture = torch.ones((256, 256, 1), device="cuda") * 1e-2
            self.cano_texture = torch.logit(self.cano_texture)
            self.cano_texture = torch.nn.Parameter(
                self.cano_texture.requires_grad_(True)
            )

            self.geom_featmap = (
                torch.ones(1, 32, 64, 64).normal_(mean=0.0, std=0.01).float().cuda()
            )
            self.geom_featmap = torch.nn.Parameter(
                self.geom_featmap.requires_grad_(True)
            )

            self.create_from_pcd(pcd, 10)
            self.training_setup()

            xyzs = self.get_xyz
            rots = self.get_rotation
            rot_mat = quaternion_to_matrix(rots)
            rot_normal = rot_mat[:, :, 2]
            points_perturbed_up = xyzs + 0.001 * rot_normal
            # points_perturbed_down = xyzs - 0.001 * rot_normal
            reset_points = torch.cat([xyzs, points_perturbed_up], 0)
            reset_colors = torch.ones_like(torch.cat([self.get_colors] * 2, 0)) * 0.5
            reset_scales = torch.cat([self.get_scaling] * 2, 0)
            reset_rots = torch.cat([self.get_rotation] * 2, 0)

            self.attribute_field.reset_field(
                reset_points, reset_colors, reset_scales, reset_rots
            )

        elif self.cfg.geometry_convert_from.startswith(
            "resume:"
        ):  # os.path.exists(self.cfg.geometry_convert_from):
            threestudio.info(
                "Loading point cloud from %s" % self.cfg.geometry_convert_from
            )
            if self.cfg.geometry_convert_from.endswith(".ckpt"):
                seq_name = self.cfg.geometry_convert_from.split(":")[1]
                ckpt_path = self.cfg.geometry_convert_from[
                    len("resume:" + seq_name + ":") :
                ]
                self.cfg.smpl_guidance_config.update({"seq": seq_name})
                smpl_guidance = threestudio.find("smpl-guidance")(
                    self.cfg.smpl_guidance_config
                )
                self.smpl_guidance = smpl_guidance
                ckpt_dict = torch.load(ckpt_path)
                num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
                pcd = BasicPointCloud(
                    points=np.zeros((num_pts, 3)),
                    colors=np.zeros((num_pts, 3)),
                    normals=np.zeros((num_pts, 3)),
                )
                self.create_from_pcd(pcd, 10)
                self.training_setup()
                new_ckpt_dict = {}
                for key in self.state_dict():
                    if ckpt_dict["state_dict"].__contains__("geometry." + key):
                        new_ckpt_dict[key] = ckpt_dict["state_dict"]["geometry." + key]
                    else:
                        new_ckpt_dict[key] = self.state_dict()[key]
                # new_ckpt_dict["_colors"] = (
                #     torch.ones_like(new_ckpt_dict["_colors"]) * 0.5
                # )
                #  new_ckpt_dict["_occ"] = torch.logit(
                #      torch.ones_like(new_ckpt_dict["_occ"]) * 1e-2
                #  )
                # breakpoint()
                # del new_ckpt_dict[
                #     "attribute_field.mlp_base_offsets.tcnn_encoding.params"
                # ]
                self.load_state_dict(new_ckpt_dict, strict=False)

                xyzs = self.get_xyz
                rots = self.get_rotation
                rot_mat = quaternion_to_matrix(rots)
                rot_normal = rot_mat[:, :, 2]
                points_perturbed_up = xyzs + 0.001 * rot_normal
                points_perturbed_down = xyzs - 0.001 * rot_normal

                # import viser
                # server = viser.ViserServer()
                # server.add_point_cloud("points", fused_point_cloud.detach().cpu().numpy(), colors=np.array([0, 255, 0]), point_size=0.001)
                # server.add_point_cloud("perturbed", points_perturbed.detach().cpu().numpy(), colors=np.array([255, 0, 0]), point_size=0.001)
                # breakpoint()
                reset_points = torch.cat(
                    [xyzs, points_perturbed_up, points_perturbed_down], 0
                )
                reset_colors = (
                    torch.ones_like(torch.cat([self.get_colors] * 3, 0)) * 0.5
                )
                reset_scales = torch.cat([self.get_scaling] * 3, 0)
                reset_rots = torch.cat([self.get_rotation] * 3, 0)

                #  self.attribute_field.reset_field(
                #      reset_points, reset_colors, reset_scales, reset_rots
                #  )
            elif self.cfg.geometry_convert_from.endswith(".ply"):
                if self.cfg.load_ply_only_vertex:
                    plydata = PlyData.read(self.cfg.geometry_convert_from)
                    vertices = plydata["vertex"]
                    positions = np.vstack(
                        [vertices["x"], vertices["y"], vertices["z"]]
                    ).T
                    if vertices.__contains__("red"):
                        colors = (
                            np.vstack(
                                [vertices["red"], vertices["green"], vertices["blue"]]
                            ).T
                            / 255.0
                        )
                    else:
                        shs = np.random.random((positions.shape[0], 3)) / 255.0
                        C0 = 0.28209479177387814
                        colors = shs * C0 + 0.5
                    normals = np.zeros_like(positions)
                    pcd = BasicPointCloud(
                        points=positions, colors=colors, normals=normals
                    )
                    self.create_from_pcd(pcd, 10)
                else:
                    self.load_ply(self.cfg.geometry_convert_from)
                self.training_setup()
        else:
            threestudio.info("Geometry not found, initilization with random points")
            num_pts = self.cfg.init_num_pts
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = self.cfg.pc_init_radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)

            shs = np.random.random((num_pts, 3)) / 255.0
            C0 = 0.28209479177387814
            color = shs * C0 + 0.5
            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((num_pts, 3))
            )

            self.create_from_pcd(pcd, 10)
            self.training_setup()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.scale_gradient_accum,
            self.rot_gradient_accum,
            self.opac_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.config,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            scale_gradient_accum,
            rot_gradient_accum,
            opac_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.config,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.scale_gradient_accum = scale_gradient_accum
        self.rot_gradient_accum = rot_gradient_accum
        self.opac_gradient_accum = opac_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # print(self._scaling)
        return self.scaling_activation(self._scaling)
        # scaling_2d = torch.cat([self._scaling[..., :2], torch.full_like(self._scaling[..., 2:], -1e10)], -1)
        # return self.scaling_activation(scaling_2d)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_delta_xyz(self):
        return self._xyz - self.original_pos

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_colors(self):
        return torch.sigmoid(self._colors)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_occ(self):
        return torch.sigmoid(self._occ)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    @property
    def get_normal(self):
        return quaternion2rotmat(self.get_rotation)[..., 2]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]  # .repeat(1, 3)

        # scales = torch.log(torch.ones((len(fused_point_cloud), 3)).cuda() * 0.02)

        if self.config[0] > 0:
            # if np.abs(np.sum(pcd.normals)) < 1:
            #     dup = 4
            #     fused_point_cloud = torch.cat([fused_point_cloud for _ in range(dup)], 0)
            #     fused_color = torch.cat([fused_color for _ in range(dup)], 0)
            #     scales = torch.cat([scales for _ in range(dup)], 0)
            #     normals = np.random.rand(len(fused_point_cloud), 3) - 0.5
            #     normals /= np.linalg.norm(normals, 2, 1, True)
            # else:
            #     normals = pcd.normals
            # breakpoint()
            # rots = normal2rotation(torch.from_numpy(normals).to(torch.float32)).to("cuda")
            rots = self.smpl_guidance.init_q
            # scales[..., -1] -= 1e10 # squeeze z scaling
            # scales[..., -1] = 0
            # print(pcd.normals)
            # exit()
            # rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
            # rots = self.rotation_activation(rots)
        else:
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

        colors = torch.logit(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self.original_pos = fused_point_cloud.clone().detach()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._colors = nn.Parameter(colors.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        occ = torch.ones((fused_point_cloud.shape[0], 1), device="cuda") * 1e-2
        occ = torch.logit(occ)
        self._occ = nn.Parameter(occ.requires_grad_(True))

        if not hasattr(self, "latent_pose"):
            self.latent_pose = torch.zeros(
                (len(self.smpl_guidance.smpl_parms["body_pose"]), 2), device="cuda"
            )
            self.latent_pose = torch.nn.Parameter(self.latent_pose.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        cano_points = self.smpl_guidance.query_points
        self.aabb = torch.stack(
            [cano_points.min(dim=1)[0], cano_points.max(dim=1)[0]]
        ).to("cuda")
        center = self.aabb.mean(dim=0)
        self.aabb = (self.aabb - center) * 1.5 + center
        diagonal = torch.norm(self.aabb[1] - self.aabb[0], 2)
        self.radius = 1e-1
        self.attribute_field = HashMLPSDFField(self.aabb).to("cuda")

        # exit()

    def training_setup(self, training_args=None):
        if training_args is None:
            training_args = self.cfg
        else:
            self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.rot_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._colors], "lr": training_args.feature_lr, "name": "color"},
            {
                "params": self.attribute_field.encoding.parameters(),
                "lr": training_args.field_lr,
                "name": "attribute_field_encoding",
            },
            {
                "params": self.attribute_field.quat_encoding.parameters(),
                "lr": training_args.field_lr,
                "name": "attribute_field_quat_encoding",
            },
            {
                "params": self.attribute_field.mlp_base_shs.parameters(),
                "lr": training_args.field_lr,
                "name": "attribute_field_shs",
            },
            {
                "params": self.attribute_field.mlp_base_quats.parameters(),
                "lr": training_args.field_lr,
                "name": "attribute_field_quats",
            },
            {
                "params": self.attribute_field.mlp_base_scales.parameters(),
                "lr": training_args.field_lr * 10,
                "name": "attribute_field_scales",
            },
            {
                "params": self.attribute_field.mlp_base_offsets.parameters(),
                "lr": training_args.field_lr * 0.01,
                "name": "attribute_field_offests",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._occ],
                "lr": training_args.occ_lr,
                "name": "occ",
            },
            # {
            #     "params": self.background.parameters(),
            #     "lr": training_args.background_lr,
            #     "name": "background",
            # },
            {
                "params": [self.latent_pose],
                "lr": training_args.latent_pose_lr,
                "name": "latent_pose",
            },
        ]

        if len(self.config) <= 3:
            self.config.append(training_args.camera_lr > 0)
        else:
            self.config[3] = training_args.camera_lr > 0
        self.config = torch.tensor(self.config, device="cuda").float()
        # self.optimizer = torch.optim.SGD(l)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    # def save_pcl(self, path):
    #     v = self.get_xyz
    #     n = self.get_normal
    #     c = SH2RGB(self._features_dc)[:, 0]
    #     save_pcl('test/pcl.ply', v, n, c)

    def reset_opacity(self, ratio, iteration):
        # if len(self._xyz) < self.opac_reset_record[0] * 1.05 and iteration < self.opac_reset_record[1] + 3000:
        #     print(len(self._xyz), self.opac_reset_record, 'notreset')
        #     return
        # print(len(self._xyz), self.opac_reset_record, 'reset')
        # self.opac_reset_record = [len(self._xyz), iteration]

        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * ratio))
        opacities_new = inverse_sigmoid(self.get_opacity * ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )

        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._colors = optimizable_tensors["color"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]
        self.rot_gradient_accum = self.rot_gradient_accum[valid_points_mask]
        self.opac_gradient_accum = self.opac_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if "attribute" in group["name"]:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_colors,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "color": new_colors,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._colors = optimizable_tensors["color"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.scale_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.rot_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opac_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(
        self, grads, grad_threshold, scene_extent, N=2, pre_mask=True
    ):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        # print(selected_pts_mask.dtype, pre_mask.dtype)
        # selected_pts_mask *= pre_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        if self.config[0] > 0:
            new_scaling[:, -1] = -1e10
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_colors = self._colors[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_colors,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, pre_mask=True):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        # selected_pts_mask += (grad_rot > grad_rot_thrsh).squeeze()
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        selected_pts_mask *= pre_mask

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_colors = self._colors[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_colors,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def adaptive_prune(self, min_opacity, extent):

        # print(sum(grad_rot > 1.2) / len(grad_rot))
        # print(sum(grad_pos > max_grad) / len(grad_pos), max_grad)

        n_ori = len(self._xyz)

        # prune
        # prune_mask =
        # opac_thrsh = torch.tensor([min_opacity, 1])
        opac_temp = self.get_opacity
        prune_opac = (opac_temp < min_opacity).squeeze()
        # prune_opac += (opac_temp > opac_thrsh[1]).squeeze()

        # scale_thrsh = torch.tensor([2e-4, 0.1]) * extent
        scale_min = self.get_scaling[:, :2].min(1).values
        scale_max = self.get_scaling[:, :2].max(1).values
        prune_scale = scale_max > 0.5 * extent
        prune_scale += (scale_min * scale_max) < (1e-8 * extent**2)
        # print(prune_scale.sum())

        prune_vis = (self.denom == 0).squeeze()
        prune = prune_opac + prune_vis + prune_scale
        self.prune_points(prune)
        # print(f'opac:{prune_opac.sum()}, scale:{prune_scale.sum()}, vis:{prune_vis.sum()} extend:{extent}')
        # print(f'prune: {n_ori}-->{len(self._xyz)}')

    def adaptive_densify(self, max_grad, extent):
        grad_pos = self.xyz_gradient_accum / self.denom
        grad_scale = self.scale_gradient_accum / self.denom
        grad_rot = self.rot_gradient_accum / self.denom
        grad_opac = self.opac_gradient_accum / self.denom
        grad_pos[grad_pos.isnan()] = 0.0
        grad_scale[grad_scale.isnan()] = 0.0
        grad_rot[grad_rot.isnan()] = 0.0
        grad_opac[grad_opac.isnan()] = 0.0

        # densify
        # opac_lr = [i['lr'] for i in self.optimizer.param_groups if i['name'] == 'opacity'][0]
        larger = torch.le(grad_scale, 1e-7)[:, 0]  # if opac_lr == 0 else True
        # print(grad_opac.min(), grad_opac.max(), grad_opac.mean())
        denser = torch.le(grad_opac, 2)[:, 0]
        pre_mask = denser * larger

        self.densify_and_clone(grad_pos, max_grad, extent, pre_mask=pre_mask)
        self.densify_and_split(grad_pos, max_grad, extent)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # print(self.xyz_gradient_accum.shape)
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        # print(self.xyz_gradient_accum.shape)
        # print(self._scaling.grad.shape)
        # exit()
        # points = self.get_xyz
        # points_attributes = self.attribute_field(points[None].detach())
        # attribute_scale = points_attributes["scales"]
        # self.scale_gradient_accum[update_filter] += attribute_scale.grad[
        #     update_filter, :2
        # ].sum(1, True)

        self.scale_gradient_accum[update_filter] += self._scaling.grad[
            update_filter, :2
        ].sum(1, True)
        # print(self._scaling.grad)
        self.rot_gradient_accum[update_filter] += torch.norm(
            self._rotation[update_filter], dim=-1, keepdim=True
        )
        self.opac_gradient_accum[update_filter] += self._opacity[update_filter]
        self.denom[update_filter] += 1

    def mask_prune(self, cams, pad=4):
        batch_size = 32
        batch_num = len(cams) // batch_size + int(len(cams) % batch_size != 0)
        cams_batch = [
            cams[i * batch_size : min(len(cams), (i + 1) * batch_size)]
            for i in range(batch_num)
        ]
        for c in cams_batch:
            _, _, inMask, outView = world2scrn(self._xyz.detach(), c, pad)
            visible = inMask.all(0) * ~(outView.all(0))
            self.prune_points(~visible)

    def to_occ_grid(self, cutoff, grid_dim_max=512, bound_overwrite=None):
        if bound_overwrite is None:
            xyz_min = self._xyz.min(0)[0]
            xyz_max = self._xyz.max(0)[0]
            xyz_len = xyz_max - xyz_min
            xyz_min -= xyz_len * 0.1
            xyz_max += xyz_len * 0.1
        else:
            xyz_min, xyz_max = bound_overwrite
        xyz_len = xyz_max - xyz_min

        # print(xyz_min, xyz_max, xyz_len)

        # grid_dim_max = 1024
        grid_len = xyz_len.max() / grid_dim_max
        grid_dim = (xyz_len / grid_len + 0.5).to(torch.int32)

        # breakpoint()
        points_attributes = self.attribute_field(self.get_xyz.detach())
        attribute_color, attribute_scale, attribute_opacity, attribute_quat = (
            points_attributes["shs"],
            points_attributes["scales"],
            points_attributes["opacities"],
            points_attributes["quats"],
        )
        grid = self.utils_mod.gaussian2occgrid(
            xyz_min,
            xyz_max,
            grid_len,
            grid_dim,
            self.get_xyz,
            self.get_rotation,
            attribute_scale.repeat(1, 3),
            torch.ones_like(self.get_opacity),
            torch.tensor([cutoff]).to(torch.float32).cuda(),
        )

        # print('here')
        # x, y, z = torch.meshgrid(torch.arange(0, grid_dim[0]), torch.arange(0, grid_dim[1]), torch.arange(0, grid_dim[2]), indexing='ij')

        # print('here')
        # exit()

        # grid_cord = torch.stack([x, y, z], -1).cuda()

        return grid, -xyz_min, 1 / grid_len, grid_dim

    @torch.no_grad()
    def update_states(
        self, iteration, visibility_filter, radii, viewspace_point_tensor
    ):
        # Densification
        # breakpoint()
        cameras_extent = self.radius
        if iteration > self.cfg.densify_from_iter:
            # Keep track of max radii in image-space for pruning
            print("update_states iteration", iteration)

            for i in range(len(visibility_filter)):
                radii_i = radii[i]
                visibility_filter_i = visibility_filter[i]
                viewspace_point_tensor_i = viewspace_point_tensor[i]

                self.max_radii2D = torch.max(self.max_radii2D, radii_i.float())
                self.add_densification_stats(
                    viewspace_point_tensor_i, visibility_filter_i
                )
                # self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
                # self.add_densification_stats(viewspace_point_tensor, visibility_filter)
            min_opac = 0.1  # if iteration <= opt.densify_from_iter else 0.1
            # min_opac = 0.05 if iteration <= opt.densify_from_iter else 0.005
            # if iteration % opt.pruning_interval == 0:
            if iteration % self.cfg.densification_interval == 0:
                if iteration > self.cfg.prune_from_iter:
                    self.adaptive_prune(min_opac, cameras_extent)
                self.adaptive_densify(self.cfg.densify_grad_threshold, cameras_extent)

            if (
                iteration - 1
            ) % self.cfg.opacity_reset_interval == 0 and self.cfg.opacity_lr > 0:
                self.reset_opacity(0.12, iteration)
