from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from torch import Tensor, nn


class LearnedVariance(nn.Module):
    """Variance network in NeuS

    Args:
        init_val: initial value in NeuS variance network
    """

    variance: Tensor

    def __init__(self, init_val):
        super().__init__()
        self.register_parameter(
            "variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, x: Float[Tensor, "1"]) -> Float[Tensor, "1"]:
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(
            self.variance * 10.0
        )

    def get_variance(self) -> Float[Tensor, "1"]:
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


class HashMLPSDFField(Field):
    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        color_dim: int = 3,
        use_linear: bool = False,
        num_levels: int = 16,
        max_res: int = 2048,  # 1024
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        device: Literal["cpu", "cuda"] = "cuda",
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.use_linear = use_linear

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        print("implementation", implementation)
        self.encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.quat_encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.color_dim = color_dim

        if not self.use_linear:
            # network = MLP(
            #     in_dim=self.encoding.get_out_dim(),
            #     num_layers=num_layers,
            #     layer_width=hidden_dim,
            #     out_dim=1,
            #     activation=nn.ReLU(),
            #     out_activation=None,
            #     implementation=implementation,
            # )
            self.mlp_base_shs = MLP(
                in_dim=self.encoding.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )
            self.mlp_base_scales = MLP(
                in_dim=self.encoding.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.mlp_base_quats = MLP(
                in_dim=self.encoding.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=4,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.mlp_base_offsets = MLP(
                in_dim=self.encoding.get_out_dim() + 2,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=None,
                # implementation=implementation,
                implementation="torch",
            )
            self.mlp_base_offsets.layers[-1].weight.data.zero_()
            self.mlp_base_offsets.layers[-1].bias.data.zero_()
            self.mlp_base_opacities = MLP(
                in_dim=self.encoding.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )
            # self.mlp_base_shs = torch.nn.Sequential(
            #     nn.Linear(self.encoding.get_out_dim(), 64),
            #     nn.ReLU(),
            #     nn.Linear(64, 64),
            #     nn.ReLU(),
            #     nn.Linear(64, 3),
            # )
            # self.mlp_base_scales = torch.nn.Sequential(
            #     nn.Linear(self.encoding.get_out_dim(), 64),
            #     nn.ReLU(),
            #     nn.Linear(64, 64),
            #     nn.ReLU(),
            #     nn.Linear(64, 1),
            # )
        else:
            self.linear = torch.nn.Linear(self.encoding.get_out_dim(), 1)

        self.device = device

    def get_attributes(
        self, xyzs, z=None, pose=None, is_normalized=False
    ) -> Tuple[Tensor, None]:

        if not is_normalized:
            positions = SceneBox.get_normalized_positions(xyzs, self.aabb)
            # Make sure the tcnn gets inputs between 0 and 1.
            selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
            positions = positions * selector[..., None]
            positions_flat = positions.view(-1, 3)
        else:
            positions = xyzs
            positions_flat = xyzs.view(-1, 3)

        if not self.use_linear:
            x = self.encoding(positions_flat).to(positions)
            out_shs = self.mlp_base_shs(x).to(positions)
            out_scales = self.mlp_base_scales(x).to(positions)
            # out_shs = torch.sigmoid(out_shs)
            out_scales = torch.sigmoid(out_scales) * 2e-2
            x_quats = self.quat_encoding(positions_flat).to(positions)
            out_quats = self.mlp_base_quats(x_quats).to(positions)
            # out_quats = torch.tanh(out_quats)
            out_quats = F.normalize(out_quats, p=2, dim=-1)
            if z is None:
                z = torch.zeros_like(xyzs[..., :2])
            else:
                z = z[None].expand(len(x), -1)
            x_offsets = torch.cat([x, z], dim=-1)
            out_offsets = self.mlp_base_offsets(x_offsets).to(positions)
            out_opacities = self.mlp_base_opacities(x).to(positions)
            # if pose is not None:
            #     pose = pose.reshape(-1, 23 * 3)
            #     pose = pose.repeat(len(x), 1)
            #     pose_feat = self.pose2feat(pose)
            #     geo_pose = torch.cat([out_geo, pose_feat], dim=-1)
            #     out_deform = self.geo2deform(geo_pose)
            #     out_quat = self.geo2quat(geo_pose)
            # else:
            #     out_deform = torch.tensor(0.0).to(self.device)
            #     # out_quat = torch.tensor(0.0).to(self.device)
        else:
            raise NotImplementedError

        return out_shs, out_scales, out_quats, out_offsets, out_opacities

    def forward(
        self,
        xyzs: Tensor,
        pose: Optional[Tensor] = None,
        z=None,
        is_normalized: bool = False,
    ) -> Dict[str, Tensor]:
        (
            out_shs,
            out_scales,
            out_quats,
            out_offsets,
            out_opacities,
        ) = self.get_attributes(xyzs, z=z, pose=pose, is_normalized=is_normalized)
        return {
            "shs": out_shs,
            "scales": out_scales,
            "quats": out_quats,
            "offsets": out_offsets,
            "opacities": out_opacities,
        }

    def reset_field(
        self, xyzs: Tensor, gt_shs: Tensor, gt_scales: Tensor, gt_quats: Tensor
    ):

        xyzs = xyzs.detach()
        gt_shs_ = gt_shs.detach()
        gt_scales_ = gt_scales.detach()
        gt_quats_ = gt_quats.detach()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        for i in range(1000):
            out_attributes = self(xyzs)

            out_shs, out_scales, out_quats, out_offsets = (
                out_attributes["shs"],
                out_attributes["scales"],
                out_attributes["quats"],
                out_attributes["offsets"],
            )
            loss = (
                F.mse_loss(out_shs, gt_shs_)
                + 1000 * F.mse_loss(out_scales, gt_scales_)
                + F.mse_loss(out_quats, gt_quats_)
                # + 10 * F.mse_loss(out_offsets, torch.zeros_like(out_offsets))
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f"resetting field - step: {i}, loss: {loss.item()}")
