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
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

import threestudio
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.misc import C
from threestudio.utils.typing import *

from ..model.network import POP_no_unet
from .gaussian_io import GaussianIO
from .sdf_fields import HashMLPSDFField

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = (
        covs[:, 0],
        covs[:, 1],
        covs[:, 2],
        covs[:, 3],
        covs[:, 4],
        covs[:, 5],
    )

    # eps must be small enough !!!
    inv_det = 1 / (
        a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24
    )
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = (
        -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f)
        - x * y * inv_b
        - x * z * inv_c
        - y * z * inv_e
    )

    power[power > 0] = -1e10  # abnormal values... make weights 0

    return torch.exp(power)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


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


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    prcppoint: torch.Tensor

    def random_patch(self, h_size=float("inf"), w_size=float("inf")):
        h = self.image_height
        w = self.image_width
        h_size = min(h_size, h)
        w_size = min(w_size, w)
        h0 = random.randint(0, h - h_size)
        w0 = random.randint(0, w - w_size)
        h1 = h0 + h_size
        w1 = w0 + w_size
        return torch.tensor([h0, w0, h1, w1]).to(torch.float32).to("cuda")
        # return torch.tensor([0, 0, h, w]).to(torch.float32).to("cuda")


@threestudio.register("gaussiandreamer-base")
class GaussianBaseModel(BaseGeometry, GaussianIO):
    @dataclass
    class Config(BaseGeometry.Config):
        max_num: int = 500000
        sh_degree: int = 0
        position_lr: Any = 0.001
        scale_lr: Any = 0.003
        feature_lr: Any = 0.01
        opacity_lr: Any = 0.05
        scaling_lr: Any = 0.005
        rotation_lr: Any = 0.005
        pred_normal: bool = False
        normal_lr: Any = 0.001
        field_lr: Any = 0.01
        occ_lr: Any = 0.01

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
        self.denom = torch.empty(0)
        if self.cfg.pred_normal:
            self._normal = torch.empty(0)
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
            # breakpoint()
            self.smpl_guidance = smpl_guidance
            xyz = self.smpl_guidance.vertices.detach().cpu().numpy()
            # xyz = np.zeros_like(self.smpl_guidance.vertices.detach().cpu().numpy())
            dirs = "+x,+y,+z"
            # dirs = "-y,+x,+z"
            # dirs = "+x,+z,+y"
            # dirs = "+z,+x,+y"

            # dirs = "+z,+x,-y"

            # dirs = "+z,+x,+y" Zero Global Orient
            threestudio.info("Transforming point cloud with directions: %s" % dirs)
            positions, _ = transform_point_cloud(xyz, dirs)
            # positions = xyz

            # if trans:
            #     dirs = "-y,+x,+z"
            #     threestudio.info("Transforming point cloud with directions: %s" % dirs)
            #     xyz, _ = transform_point_cloud(xyz, dirs)
            # xyz, color = self.add_points(xyz, color)
            # pcd = BasicPointCloud(
            #     points=xyz * self.cfg.pc_init_radius, colors=color, normals=np.zeros((xyz.shape[0], 3))
            # )
            # self.create_from_pcd(pcd, 10)
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
            # self.net = POP_no_unet(
            #     c_geom=32,
            #     # c_geom=self.net_parms.c_geom, # channels of the geometric features
            #     # geom_layer_type=self.net_parms.geom_layer_type, # the type of architecture used for smoothing the geometric feature tensor
            #     nf=32, # num filters for the unet
            #     hsize=64, # hidden layer size of the ShapeDecoder MLP
            #     # up_mode=self.net_parms.up_mode,# upconv or upsample for the upsampling layers in the pose feature UNet
            #     # use_dropout=bool(self.net_parms.use_dropout), # whether use dropout in the pose feature UNet
            #     # uv_feat_dim=2, # input dimension of the uv coordinates
            # ).cuda()
            self.create_from_pcd(pcd, 10)
            self.training_setup()

        elif os.path.exists(self.cfg.geometry_convert_from):
            threestudio.info(
                "Loading point cloud from %s" % self.cfg.geometry_convert_from
            )
            if self.cfg.geometry_convert_from.endswith(".ckpt"):
                ckpt_dict = torch.load(self.cfg.geometry_convert_from)
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
                self.load_state_dict(new_ckpt_dict)
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

    @property
    def get_scaling(self):
        if self.cfg.sphere:
            return self.scaling_activation(
                torch.mean(self._scaling, dim=-1).unsqueeze(-1).repeat(1, 3)
            )
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        # return self.rotation_activation(self._rotation)
        return self.static_rots

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

    @property
    def get_normal(self):
        if self.cfg.pred_normal:
            return self._normal
        else:
            raise ValueError("Normal is not predicted")

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def add_points(self, coords, rgb, num_points=100000):
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))

        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        points = np.random.uniform(
            low=np.asarray(bbox.min_bound),
            high=np.asarray(bbox.max_bound),
            size=(num_points, 3),
        )

        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

        points_inside = []
        color_inside = []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]] + 0.2 * np.random.random(3))

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords, coords], axis=0)
        all_rgb = np.concatenate([all_rgb, rgb], axis=0)
        return all_coords, all_rgb

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # breakpoint()
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        threestudio.info(
            f"Number of points at initialisation:{fused_point_cloud.shape[0]}"
        )

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        # scales = torch.log(torch.tensor(0.005))[..., None].repeat(1, 3)
        scales = torch.log(torch.sqrt(dist2) * 2)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self.static_rots = rots.clone().detach()
        opacities = inverse_sigmoid(
            self.cfg.opacity_init
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        self.original_pos = fused_point_cloud.clone().detach()
        self._xyz = nn.Parameter(
            (torch.zeros_like(fused_point_cloud)).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        logit_colors = torch.logit(torch.from_numpy(pcd.colors).to("cuda").float())
        self._colors = nn.Parameter(logit_colors.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        occ = torch.ones((fused_point_cloud.shape[0], 1), device="cuda") * 1e-2
        occ = torch.logit(occ)
        self._occ = nn.Parameter(occ.requires_grad_(True))
        if self.cfg.pred_normal:
            normals = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")
            self._normal = nn.Parameter(normals.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.fused_point_cloud = fused_point_cloud.cpu().clone().detach()
        self.features = features.cpu().clone().detach()
        self.scales = scales.cpu().clone().detach()
        self.rots = rots.cpu().clone().detach()
        self.opacities = opacities.cpu().clone().detach()

        cano_points = self.smpl_guidance.query_points
        self.aabb = torch.stack(
            [cano_points.min(dim=1)[0], cano_points.max(dim=1)[0]]
        ).to("cuda")
        center = self.aabb.mean(dim=0)
        self.aabb = (self.aabb - center) * 1.5 + center
        # breakpoint()
        self.attribute_field = HashMLPSDFField(self.aabb).to("cuda")
        self.gen_field = HashMLPSDFField(self.aabb).to("cuda")

    def training_setup(self):
        training_args = self.cfg
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": C(training_args.position_lr, 0, 0),
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": C(training_args.feature_lr, 0, 0),
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": C(training_args.feature_lr, 0, 0) / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._colors],
                "lr": C(training_args.feature_lr, 0, 0),
                "name": "colors",
            },
            {
                "params": [self._opacity],
                "lr": C(training_args.opacity_lr, 0, 0),
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": C(training_args.scaling_lr, 0, 0),
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": C(training_args.rotation_lr, 0, 0),
                "name": "rotation",
            },
            {
                "params": [self._occ],
                "lr": C(training_args.occ_lr, 0, 0),
                "name": "occ",
            },
            {
                "params": self.attribute_field.encoding.parameters(),
                "lr": C(training_args.field_lr, 0, 0),
                "name": "attribute_field_encoding",
            },
            {
                "params": self.attribute_field.mlp_base_shs.parameters(),
                "lr": C(training_args.field_lr, 0, 0),
                "name": "attribute_field_shs",
            },
            {
                "params": self.attribute_field.mlp_base_scales.parameters(),
                "lr": C(training_args.field_lr * 10, 0, 0),
                "name": "attribute_field_scales",
            },
            {
                "params": self.attribute_field.mlp_base_opacities.parameters(),
                "lr": C(training_args.field_lr, 0, 0),
                "name": "attribute_field_opacities",
            },
            {
                "params": self.gen_field.encoding.parameters(),
                "lr": C(training_args.field_lr, 0, 0),
                "name": "gen_field_encoding",
            },
            {
                "params": self.gen_field.mlp_base_shs.parameters(),
                "lr": C(training_args.field_lr, 0, 0),
                "name": "gen_field_shs",
            },
            {
                "params": self.gen_field.mlp_base_scales.parameters(),
                "lr": C(training_args.field_lr, 0, 0),
                "name": "gen_field_scales",
            },
            {
                "params": [self.cano_texture],
                "lr": C(training_args.occ_lr, 0, 0),
                "name": "cano_texture",
            },
            # {
            #     "params": self.smpl_guidance.param,
            #     "lr": C(5e-3, 0, 0),
            #     "name": "smpl_params",
            # }
        ]
        if self.cfg.pred_normal:
            l.append(
                {
                    "params": [self._normal],
                    "lr": C(training_args.normal_lr, 0, 0),
                    "name": "normal",
                },
            )

        self.optimize_params = [
            "xyz",
            "f_dc",
            "f_rest",
            "opacity",
            "scaling",
            "rotation",
        ]
        self.optimize_list = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def merge_optimizer(self, net_optimizer):
        l = self.optimize_list
        for param in net_optimizer.param_groups:
            l.append(
                {
                    "params": param["params"],
                    "lr": param["lr"],
                }
            )
        self.optimizer = torch.optim.Adam(l, lr=0.0)
        return self.optimizer

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue
            if param_group["name"] == "xyz":
                param_group["lr"] = C(
                    self.cfg.position_lr, 0, iteration, interpolation="linear"
                )
            # if param_group["name"] == "scaling":
            #     param_group["lr"] = C(
            #         self.cfg.scaling_lr, 0, iteration, interpolation="linear"
            #     )
            # if param_group["name"] == "f_dc":
            #     param_group["lr"] = C(
            #         self.cfg.feature_lr, 0, iteration, interpolation="linear"
            #     )
            # if param_group["name"] == "f_rest":
            #     param_group["lr"] = (
            #         C(self.cfg.feature_lr, 0, iteration, interpolation="linear") / 20.0
            #     )
            # if param_group["name"] == "opacity":
            #     param_group["lr"] = C(
            #         self.cfg.opacity_lr, 0, iteration, interpolation="linear"
            #     )
            # if param_group["name"] == "rotation":
            #     param_group["lr"] = C(
            #         self.cfg.rotation_lr, 0, iteration, interpolation="linear"
            #     )
            # if param_group["name"] == "normal":
            #     param_group["lr"] = C(
            #         self.cfg.normal_lr, 0, iteration, interpolation="linear"
            #     )
        self.color_clip = C(self.cfg.color_clip, 0, iteration)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(self.get_opacity * 0.9)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def to(self, device="cpu"):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._opacity = self._opacity.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._normal = self._normal.to(device)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if ("name" in group) and group["name"] == name:
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
            if ("name" in group) and (group["name"] in self.optimize_params):
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
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if ("name" in group) and (group["name"] in self.optimize_params):
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
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
        new_opacities,
        new_scaling,
        new_rotation,
        new_normal=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if self.cfg.pred_normal:
            d.update({"normal": new_normal})

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.cfg.pred_normal:
            self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.cfg.split_thresh,
        )

        # divide N to enhance robustness
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1) / N
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        else:
            new_normal = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_normal,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.cfg.split_thresh,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.cfg.pred_normal:
            new_normal = self._normal[selected_pts_mask]
        else:
            new_normal = None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_normal,
        )

    def densify(self, max_grad):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad)
        self.densify_and_split(grads, max_grad)

    def prune(self, min_opacity, max_screen_size, iteration):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    @torch.no_grad()
    def update_states(
        self,
        iteration,
        visibility_filter,
        radii,
        viewspace_point_tensor,
    ):
        if self._xyz.shape[0] >= self.cfg.max_num + 100:
            prune_mask = torch.randperm(self._xyz.shape[0]).to(self._xyz.device)
            prune_mask = prune_mask > self.cfg.max_num
            self.prune_points(prune_mask)
            return
        # Keep track of max radii in image-space for pruning
        # loop over batch
        bs = len(viewspace_point_tensor)
        for i in range(bs):
            radii_i = radii[i]
            visibility_filter_i = visibility_filter[i]
            viewspace_point_tensor_i = viewspace_point_tensor[i]

            self.max_radii2D = torch.max(self.max_radii2D, radii_i.float())
            # breakpoint()
            self.add_densification_stats(viewspace_point_tensor_i, visibility_filter_i)

        if (
            iteration > self.cfg.prune_from_iter
            and iteration < self.cfg.prune_until_iter
            and iteration % self.cfg.prune_interval == 0
        ):
            self.prune(self.cfg.min_opac_prune, self.cfg.radii2d_thresh, iteration)
            if iteration % self.cfg.opacity_reset_interval == 0:
                self.reset_opacity()

        if (
            iteration > self.cfg.densify_from_iter
            and iteration < self.cfg.densify_until_iter
            and iteration % self.cfg.densification_interval == 0
        ):
            self.densify(self.cfg.densify_grad_threshold)
