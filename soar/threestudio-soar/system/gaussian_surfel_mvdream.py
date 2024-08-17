import math
import os
import random
from dataclasses import dataclass, field

import lpips
import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from torchvision.utils import save_image

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *

from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..utils.loss_utils import l1_loss_w, ssim

import skimage
from skimage.metrics import structural_similarity as ski_ssim
import lpips

loss_fn_alex = lpips.LPIPS(net='alex').cuda()
loss_fn_test_vgg = lpips.LPIPS(net='vgg', version='0.1').cuda()

def scale_gradients_hook(grad, mask=None):
    grad_copy = grad.clone()  # Make a copy to avoid in-place modifications
    # Assume we want to scale the gradients of the first 5 rows by half
    if mask is not None:
        grad_copy *= mask
    return grad_copy


@threestudio.register("gaussiansurfel-mvdream-system")
class SurfelMVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        training_stage: int = 0

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        #  self.loss_fn_vgg = lpips.LPIPS(net="alex").cuda()
        self.loss_fn_vgg = lpips.LPIPS(net="vgg").cuda()
        for param in self.loss_fn_vgg.parameters():
            param.requires_grad = False

        self.sds_start = 0 if self.cfg.training_stage == 1 else 500
        self.resampled = []
        self.psnrs = []
        self.lpips = []
        self.ssims = []

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        # if hasattr(self, "merged_optimizer"):
        #     return [optim]
        # if hasattr(self.cfg.optimizer, "name"):
        #     net_optim = parse_optimizer(self.cfg.optimizer, self)
        #     # optim = self.geometry.merge_optimizer(net_optim)
        #     breakpoint()
        #     self.merged_optimizer = True
        # else:
        #     self.merged_optimizer = False
        bg_optim = torch.optim.Adam(
            self.background.parameters(),
            lr=self.cfg.optimizer.params.background.lr,
        )

        return [optim]  # , bg_optim]

    def on_load_checkpoint(self, checkpoint):
        num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        return

    def forward(self, batch: Dict[str, Any], head_flag=False) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        # if gt:
        #     outputs = self.renderer.gt_forward(batch)
        # else:
        mode = "full"
        if self.global_step >= 200 and self.global_step < 500:
            mode = "occ"
        elif self.global_step >= 500:
            mode = "gen"
        outputs_all = self.renderer.batch_forward(
            batch, mode="gen", head_flag=head_flag, stage=self.cfg.training_stage
        )
        return outputs_all

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        iteration = self.global_step
        # gt_out = self(batch, gt=True)
        # pose_optimizer = torch.optim.SparseAdam(
        #     self.geometry.smpl_guidance.param, 5.0e-3
        # )
        head_flag = random.random() < 0.4
        out, gt_out = self(batch, head_flag=head_flag)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        guidance_normal_inp = out["comp_normal"].clone()
        guidance_occ_mask = out["comp_occ"]
        comp_bg = gt_out["comp_bg"]
        viewspace_point_tensor = out["viewspace_points"]

        gt_visibility_filter = gt_out["visibility_filter"]
        gt_radii = gt_out["radii"]
        gt_viewspace_point_tensor = gt_out["viewspace_points"]

        save_dir = self.get_save_dir()
        if iteration % 50 == 0:
            test_dir = os.path.join(save_dir, f"test_{iteration}")
            os.makedirs(test_dir, exist_ok=True)
            rendered_out = torch.cat([gt_out["comp_rgb"], batch["gt_rgb"]], dim=0)
            rendered_out = rendered_out.permute(0, 3, 1, 2)

            rendered_out_mask = torch.cat(
                [gt_out["comp_mask"], batch["gt_mask"][..., None]], dim=0
            )
            rendered_out_mask = rendered_out_mask.permute(0, 3, 1, 2)

            if "gt_normal_F" in batch:
                rendered_out_normal = torch.cat(
                    [
                        normal2rgb(gt_out["comp_normal"]),
                        batch["gt_normal_F"],
                        # batch["gt_normal_B"],
                    ],
                    dim=0,
                )
            else:
                rendered_out_normal = normal2rgb(gt_out["comp_normal"])
            rendered_out_normal = rendered_out_normal.permute(0, 3, 1, 2)
            pred_normal = out["comp_pred_normal"]
            pred_normal = normal2rgb(pred_normal).permute(0, 3, 1, 2)

            if "comp_occ" in out:
                rendered_out_occ = out["comp_occ"]
                rendered_out_occ = rendered_out_occ.permute(0, 3, 1, 2)
                save_image(
                    rendered_out_occ,
                    os.path.join(test_dir, f"test_{iteration}_occ.png"),
                )

            rendered_depth = gt_out["comp_depth"]
            rendered_depth = depth2rgb(
                rendered_depth[0].permute(2, 0, 1), batch["gt_mask"]
            )

            rendered_curv = out["comp_curv"]
            rendered_curv = rendered_curv.permute(0, 3, 1, 2)
            save_image(
                rendered_out, os.path.join(test_dir, f"test_{iteration}_test.png")
            )
            save_image(
                rendered_out_mask, os.path.join(test_dir, f"test_{iteration}_mask.png")
            )
            save_image(
                rendered_out_normal,
                os.path.join(test_dir, f"test_{iteration}_normal.png"),
            )
            save_image(
                pred_normal, os.path.join(test_dir, f"test_{iteration}_pred_normal.png")
            )
            save_image(
                rendered_depth, os.path.join(test_dir, f"test_{iteration}_depth.png")
            )
            save_image(
                rendered_curv, os.path.join(test_dir, f"test_{iteration}_curv.png")
            )
            save_image(
                normal2rgb(guidance_normal_inp).permute(0, 3, 1, 2),
                os.path.join(test_dir, f"test_{iteration}_guidance_normal.png"),
            )
            save_image(
                guidance_inp.permute(0, 3, 1, 2),
                os.path.join(test_dir, f"test_{iteration}_guidance.png"),
            )
            # breakpoint()

        # timestep = 0.98 + (self.true_global_step / 1200) * (0.02 - 0.98)
        def get_sd_step_ratio(step, start, end):

            len = end - start
            if (step + 1) <= start:
                return 1.0 / len
            if (step + 1) >= end:
                return 1.0
            ratio = min(1, (step - start + 1) / len)
            ratio = max(1.0 / len, ratio)
            return ratio

        step_ratio = get_sd_step_ratio(self.true_global_step, 0, 1200)
        # timestep = (())
        # guidance_mask[guidance_mask < 0.5] = 0.0
        # guidance_mask[guidance_mask >= 0.5] = 1.0
        # if head_flag:
        #     guidance_inp.register_hook(
        #         lambda grad: scale_gradients_hook(
        #             grad, mask=(torch.exp(-3 * guidance_mask.detach()))
        #         )
        #     )
        # guidance_normal_inp.register_hook(
        #         lambda grad: scale_gradients_hook(
        #             grad, mask=(torch.exp(-3 * guidance_mask.detach()))
        #         )
        #     )
        # else:

        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        imagedream_kwargs = {}
        is_imagedream = (
            self.cfg.guidance_type == "imagedream-multiview-diffusion-guidance"
        )
        if is_imagedream:
            if self.cfg.training_stage == 1:
                imagedream_kwargs["ref_rgb"] = batch["gt_rgb_crop"][0].permute(2, 0, 1)
                imagedream_kwargs["ref_mask"] = batch["gt_mask_crop"]
            elif self.cfg.training_stage == 0:
                imagedream_kwargs["ref_rgb"] = batch["gt_normal_F"][0].permute(2, 0, 1)
                imagedream_kwargs["ref_mask"] = batch["gt_normal_mask"]
                if imagedream_kwargs["ref_rgb"].shape[1] != 512:
                    imagedream_kwargs["ref_rgb"] = torch.nn.functional.interpolate(
                        imagedream_kwargs["ref_rgb"][None],
                        (512, 512),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    imagedream_kwargs["ref_mask"] = torch.nn.functional.interpolate(
                        imagedream_kwargs["ref_mask"][None].float(),
                        (512, 512),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
            #seems like the ref_w2c is not used in the imagedream (not important)
            # imagedream_kwargs["ref_w2c"] = self.geometry.smpl_guidance.effective_w2cs[
            #     batch["gt_index"]
            # ]
            imagedream_kwargs["comp_bg"] = comp_bg[0].permute(2, 0, 1)

        if self.cfg.training_stage == 1:
            if self.cfg.loss["lambda_occ"] > 0.0:
                guidance_inp.register_hook(
                    lambda grad: scale_gradients_hook(
                        grad, mask=(torch.exp(-3 * guidance_occ_mask.detach()))
                    )
                )
            guidance_out = self.guidance(
                guidance_inp,
                self.prompt_utils,
                **batch,
                rgb_as_latents=False,
                **imagedream_kwargs,
            )

            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss_sds += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )
        elif self.cfg.training_stage == 0:
            if self.cfg.loss["lambda_occ"] > 0.0:
                guidance_normal_inp.register_hook(
                    lambda grad: scale_gradients_hook(
                        grad, mask=(torch.exp(-3 * guidance_occ_mask.detach()))
                    )
                )
            guidance_normal_out = self.guidance(
                guidance_normal_inp,
                self.prompt_utils,
                **batch,
                rgb_as_latents=False,
                normal_flag=True,
                **imagedream_kwargs,
            )

            for name, value in guidance_normal_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss_sds += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # check_points = self.geometry.smpl_guidance.query_points + self.geometry.get_xyz
        # check_points_att = self.geometry.attribute_field(check_points)
        # check_color, check_scale = check_points_att["shs"], check_points_att["scales"]
        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            loss_position = xyz_mean.mean()
            self.log(f"train/loss_position", loss_position)
            loss += self.C(self.cfg.loss["lambda_position"]) * loss_position

        delta_mean = None
        if self.cfg.loss["lambda_delta"] > 0.0:
            delta_mean = self.geometry.get_delta_xyz.norm(dim=-1)
            loss_delta = delta_mean.mean()
            self.log(f"train/loss_delta", loss_delta)
            loss += self.C(self.cfg.loss["lambda_delta"]) * loss_delta
            #  print("Delta mean: ", delta_mean.mean())

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_sparsity"] > 0.0:
            # loss_sparsity = (out["comp_mask"] ** 2 + 0.01).sqrt().mean()
            # self.log("train/loss_sparsity", loss_sparsity)
            # loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            loss_sparsity = -(self.geometry.get_opacity - 0.5).pow(2).mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.cfg.loss["lambda_scales"] > 0.0:
            if not self.cfg.renderer["use_explicit"]:
                points = self.geometry.get_xyz
                out_attributes = self.geometry.attribute_field(points)
                scales, offsets = out_attributes["scales"], out_attributes["offsets"]
            else:
                scales = self.geometry.get_scaling
            scale_sum = torch.mean(scales)  # self.geometry.get_scaling)
            # scale_sum = torch.mean(check_scale)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        # if (
        #     self.cfg.loss["lambda_offsets"] > 0.0
        #     and not self.cfg.renderer["use_explicit"]
        # ):
        #     # points = self.geometry.get_xyz
        #     # out_attributes = self.geometry.attribute_field(points)
        #     # scales, offsets = out_attributes["scales"], out_attributes["offsets"]
        #     # if iteration % 50 == 0:
        #     #     breakpoint()
        #     offset_sum = torch.mean(torch.abs(offsets))
        #     latent_mean = torch.mean(torch.abs(self.geometry.latent_pose))
        #     self.log(f"train/offsets", offset_sum)
        #     loss += self.C(self.cfg.loss["lambda_offsets"]) * (offset_sum + latent_mean)

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv

        mask = batch["gt_mask"] > 1e-5
        normal_mask = batch["gt_normal_mask"] > 1e-5
        gt_rgb_blended = batch["gt_rgb"] * batch["gt_mask"][..., None] + gt_out["rand_bg"] * (
            1 - batch["gt_mask"][..., None]
        )
        if True:
            if self.cfg.loss["lambda_recon"] > 0.0:
                loss_recon = 0.8 * l1_loss_w(
                    gt_out["comp_rgb"][mask], batch["gt_rgb"][mask]
                ) + 0.2 * (
                    1
                    - ssim(
                        gt_out["comp_rgb"].permute(0, 3, 1, 2),
                        gt_rgb_blended.permute(0, 3, 1, 2),
                    )
                )
                loss_recon = loss_recon * self.C(self.cfg.loss["lambda_recon"])
                self.log(f"train/loss_recon", loss_recon)
                loss += loss_recon

            if self.cfg.loss["lambda_mask"] > 0.0:
                loss_recon = torch.abs(
                    gt_out["comp_mask"] - batch["gt_mask"][..., None]
                ).mean() * self.C(self.cfg.loss["lambda_mask"])
                self.log(f"train/loss_mask", loss_recon)
                loss += loss_recon

            if self.cfg.loss["lambda_normal_F"] > 0.0 and "gt_normal_F" in batch:
                # normal_mask_float = normal_mask.float()
                loss_normal = (
                    0.2
                    * cos_loss(
                        gt_out["comp_normal"][[0]],
                        batch["gt_normal_F"],
                        normal_mask,
                        thrsh=0,
                        weight=1,
                    )
                    + 1 * self.loss_fn_vgg(
                        (
                            (
                                gt_out["comp_normal"][[0]]
                                 * batch["gt_normal_mask"][..., None]
                            ).permute(0, 3, 1, 2)
                            - 0.5
                        )
                        * 2,
                        (
                            (
                                batch["gt_normal_F"] * batch["gt_normal_mask"][..., None]#* normal_mask_float[..., None]
                            ).permute(0, 3, 1, 2)
                            - 0.5
                        )
                        * 2,
                    ).mean()
                ) * self.C(self.cfg.loss["lambda_normal_F"])
                self.log(f"train/loss_normal_F", loss_normal)
                loss += loss_normal

                # if self.cfg.loss["lambda_normal_mask"] > 0.0:
                #     loss_normal_mask = torch.abs(
                #         gt_out["comp_normal_mask"][[0]] - batch["gt_normal_mask"]
                #     ).mean() * self.C(self.cfg.loss["lambda_normal_mask"])
                #     loss += loss_normal_mask

            if self.cfg.loss["lambda_normal_B"] > 0.0 and "gt_normal_B" in batch:
                #  loss_normal = cos_loss(
                #      gt_out["comp_normal"][[1]],
                #      batch["gt_normal_B"],
                #      normal_mask,
                #      thrsh=0,
                #      weight=1,
                #  ) * self.C(self.cfg.loss["lambda_normal_B"])
                normal_mask_float = normal_mask.float()
                loss_normal = (
                    0.2
                    * cos_loss(
                        gt_out["comp_normal"][[1]],
                        batch["gt_normal_B"],
                        normal_mask,
                        thrsh=0,
                        weight=1,
                    )
                    + self.loss_fn_vgg(
                        (
                            (
                                gt_out["comp_normal"][[1]]
                                * normal_mask_float[..., None]
                            ).permute(0, 3, 1, 2)
                            - 0.5
                        )
                        * 2,
                        (
                            (
                                batch["gt_normal_B"] * normal_mask_float[..., None]
                            ).permute(0, 3, 1, 2)
                            - 0.5
                        )
                        * 2,
                    ).mean()
                ) * self.C(self.cfg.loss["lambda_normal_B"])
                self.log(f"train/loss_normal_B", loss_normal)
                loss += loss_normal

            if self.cfg.loss["lambda_normal_mask"] > 0.0:
                loss_normal_mask = torch.abs(
                    gt_out["comp_normal_mask"][0,...,0] - batch["gt_normal_mask"][0]
                ).mean() * self.C(self.cfg.loss["lambda_normal_mask"])
                loss += loss_normal_mask

            if self.cfg.loss["lambda_vgg"] > 0.0:
                vgg_loss = (
                    self.C(self.cfg.loss["lambda_vgg"])
                    * self.loss_fn_vgg(
                        (gt_out["comp_rgb"].permute(0, 3, 1, 2) - 0.5) * 2,
                        (gt_rgb_blended.permute(0, 3, 1, 2) - 0.5) * 2,
                    ).mean()
                )
                self.log(f"train/vgg_loss", vgg_loss)
                loss += vgg_loss

        if self.cfg.loss["lambda_occ"] > 0.0:
            mask = batch["gt_mask"] > 0.0
            loss_occ = (1 - gt_out["comp_occ"][mask]).mean() * self.C(
                self.cfg.loss["lambda_occ"]
            )
            loss += loss_occ

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
        ):
            loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            )
            self.log(f"train/loss_depth_tv", loss_depth_tv)
            loss += loss_depth_tv

        if self.cfg.loss["lambda_normal_consistency"] > 0.0 and out.__contains__(
            "comp_pred_normal"
        ):
            loss_pred_normal = torch.nn.functional.mse_loss(
                out["comp_pred_normal"], out["comp_normal"]  # .detach()
            )
            loss_pred_normal = cos_loss(
                gt_out["comp_pred_normal"],
                gt_out["comp_normal"],
                thrsh=np.pi * 1 / 10000,
                weight=1,
            )
            if iteration > self.sds_start:
                loss_pred_normal += cos_loss(
                    out["comp_pred_normal"],
                    out["comp_normal"],
                    thrsh=np.pi * 1 / 10000,
                    weight=1,
                )
                loss_pred_normal = loss_pred_normal * 0.5
            self.log(f"train/loss_pred_normal_consistency", loss_pred_normal)
            loss += (
                self.C(self.cfg.loss["lambda_normal_consistency"])
                + 0.1 * min(2 * iteration / 2000, 1)
            ) * loss_pred_normal

        if self.cfg.loss["lambda_curv"] > 0.0 and out.__contains__("comp_curv"):
            loss_curv = torch.abs(out["comp_curv"]).mean() * self.C(
                self.cfg.loss["lambda_curv"]
            )
            self.log(f"train/loss_curv", loss_curv)
            loss += loss_curv

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        if loss_sds > 0 and iteration > self.sds_start:
            loss_sds.backward(retain_graph=True)

        # if gt_viewspace_point_tensor[0].grad is not None:

        if loss > 0:
            # print("Optimizing loss with loss recon: ", loss_recon)
            # if iteration < 200:
            #     loss = loss_occ
            loss.backward()  # retain_graph=True)
            # print('iter: ', iteration, 'gt_view_space_points: ', gt_viewspace_point_tensor[0].grad)
            # self.geometry.update_states(
            #     iteration,
            #     gt_visibility_filter,
            #     gt_radii,
            #     gt_viewspace_point_tensor,
            # )
            # if viewspace_point_tensor[0].grad is not None:
            #     self.geometry.update_states(
            #         iteration,
            #         visibility_filter,
            #         radii,
            #         viewspace_point_tensor,
            #     )
        # if iteration % 50 == 0:
        #     breakpoint()
        # for opti in opt:
        opt.step()
        opt.zero_grad(set_to_none=True)
        # pose_optimizer.zero_grad(set_to_none=True)

        return {"loss": loss_sds + loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_pred_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_pred_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_occ"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "comp_occ" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )
        # self.save_image_grid(
        #     f"it{self.global_step}-{batch['index'][0]}_gt_occ.png",
        #     [
        #         {
        #             "type": "rgb",
        #             "img": batch["comp_occ"][0],
        #             "kwargs": {"data_format": "HWC"},
        #         },
        #     ]
        #     name="validation_step",
        #     step=self.global_step,
        # )
        # save_image(out["comp_occ"].permute(0,3,1,2), f"it{self.global_step}-{batch['index'][0]}_occ.png")

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):

        # breakpoint()
        # if batch_idx % 10 == 0:
        #     depth = out["comp_depth"][0].permute(2, 0, 1)
        #     normal = out["comp_normal"][0].permute(2, 0, 1)
        #     image = out["comp_rgb"][0].permute(2, 0, 1)
        #     mask = out["comp_mask"][0].permute(2, 0, 1)
        #     bound = None
        #     occ_grid, grid_shift, grid_scale, grid_dim = self.geometry.to_occ_grid(
        #         0.0, 512, bound
        #     )

        #     fovy = batch["fovy"][0]
        #     w2c, proj, cam_p = get_cam_info_gaussian(
        #         c2w=batch["c2w"][0], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
        #     )

        #     # import pdb; pdb.set_trace()
        #     viewpoint_cam = Camera(
        #         FoVx=fovy,
        #         FoVy=fovy,
        #         image_width=batch["width"],
        #         image_height=batch["height"],
        #         world_view_transform=w2c,
        #         full_proj_transform=proj,
        #         camera_center=cam_p,
        #         prcppoint=torch.tensor([0.5, 0.5], device=w2c.device),
        #     )
        #     pts = resample_points(viewpoint_cam, depth, normal, image, mask)
        #     grid_mask = grid_prune(
        #         occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=1
        #     )
        #     clean_mask = grid_mask  # * mask_mask
        #     pts = pts[clean_mask]
        #     self.resampled.append(pts)
        out, gt_out = self(batch)
        # rgb_dir = os.path.join(
        #     self.get_save_dir(), "test", f"cam_{str(batch_idx).zfill(2)}", "rgb"
        # )
        # os.makedirs(rgb_dir, exist_ok=True)
        # normal_dir = os.path.join(
        #     self.get_save_dir(), "test", f"cam_{str(batch_idx).zfill(2)}", "normal"
        # )
        # os.makedirs(normal_dir, exist_ok=True)
        # occ_dir = os.path.join(
        #     self.get_save_dir(), "test", f"cam_{str(batch_idx).zfill(2)}", "occ"
        # )
        # os.makedirs(occ_dir, exist_ok=True)
        # for i in range(10):
        #     batch["gt_index"] = (
        #         i * len(self.geometry.smpl_guidance.smpl_parms["body_pose"]) // 10
        #     )
        #     out = self(batch)
        #     save_image(
        #         torch.cat([out["comp_rgb"], out["comp_mask"]], dim=-1).permute(
        #             0, 3, 1, 2
        #         ),
        #         os.path.join(rgb_dir, f"{str(i).zfill(5)}.png"),
        #     )
        #     save_image(
        #         torch.cat([out["comp_normal"], out["comp_mask"]], dim=-1).permute(
        #             0, 3, 1, 2
        #         ),
        #         os.path.join(normal_dir, f"{str(i).zfill(5)}.png"),
        #     )
        #     save_image(
        #         torch.cat([out["comp_occ"], out["comp_mask"]], dim=-1).permute(
        #             0, 3, 1, 2
        #         ),
        #         os.path.join(occ_dir, f"{str(i).zfill(5)}.png"),
        #     )

        
        pred = gt_out["comp_rgb"][0].detach().cpu().numpy()
        gt = batch["gt_rgb"][0].detach().cpu().numpy()
        gt_mask = batch["gt_mask"][0].detach().cpu().numpy()
        choice = gt_mask > 0.5
        gt[~choice] = 1.0
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['gt_index']}.png",
            [
                {
                    "type": "rgb",
                    "img": gt_out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ] + (
                [
                    {
                        "type": "rgb",
                        "img": torch.from_numpy(gt),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "comp_occ" in out
                else []
            ),
            name="test_step",
            step=self.global_step,
        )
        psnr_ = skimage.metrics.peak_signal_noise_ratio(gt, pred)
        self.psnrs.append(psnr_)
        ssim_ = ski_ssim(pred,gt, multichannel=True, channel_axis=-1, data_range=1)
        self.ssims.append(ssim_)
        lpips_ = loss_fn_test_vgg(
            torch.from_numpy(pred)[None].cuda().permute(0, 3, 1, 2) * 2 - 1, 
            torch.from_numpy(gt)[None].cuda().permute(0, 3, 1, 2) * 2 - 1).mean()
        self.lpips.append(lpips_)
        print("PSNR: ", psnr_, "SSIM: ", ssim_, "LPIPS: ", lpips_)
        # if batch["index"][0] == 0:
        #     save_path = self.get_save_path("point_cloud.ply")
        #     self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        threestudio.info("Saving test sequence")
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )
        self.psnrs = np.array(self.psnrs)
        self.ssims = np.array(self.ssims)
        self.lpips = torch.tensor(self.lpips)
        self.lpips = self.lpips.detach().cpu().numpy()
        threestudio.info(f"Average PSNR: {self.psnrs.mean()}; SSIM: {self.ssims.mean()}; LPIPS: {self.lpips.mean()}")
        np.savetxt(os.path.join(self.get_save_dir(), "psnrs.txt"), self.psnrs)
        np.savetxt(os.path.join(self.get_save_dir(), "ssims.txt"), self.ssims)
        np.savetxt(os.path.join(self.get_save_dir(), "lpips.txt"), self.lpips)
        with open(os.path.join(self.get_save_dir(), "average.txt"), "w") as f:
            f.write(f"{self.psnrs.mean()} {self.ssims.mean()} {self.lpips.mean()}")
        # self.save_mesh(self.get_save_dir())

    def save_mesh(self, save_dir, poisson_depth=10):
        resampled = torch.cat(self.resampled, 0)
        mesh_path = f"{save_dir}/poisson_mesh_{poisson_depth}"

        # breakpoint()
        points = self.geometry.get_xyz
        rot = self.geometry.get_rotation
        tmp_joints, mat, ori_mat = self.geometry.smpl_guidance(
            points, idx=0, zero_out=True
        )
        points = (
            torch.einsum("bnxy,bny->bnx", mat[..., :3, :3], points[None])
            + mat[..., :3, 3]
        )[0]
        out_attributes = self.geometry.attribute_field(points)
        colors = out_attributes["shs"]
        rot_mat = quaternion_to_matrix(rot)
        rot_mat = torch.matmul(mat[..., :3, :3], rot_mat)[0]
        # rot = matrix_to_quaternion(rot_mat)
        # rot = torch.nn.functional.normalize(rot, p=2, dim=-1)[0]
        normal = rot_mat[..., 2]
        poisson_mesh(mesh_path, points, normal, colors, poisson_depth, 1 * 1e-4)
        # poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, 1 * 1e-4)
        return


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def depth2rgb(depth, mask):
    #  print(depth.shape, mask.shape)
    sort_d = torch.sort(depth[mask.to(torch.bool)])[0]
    min_d = sort_d[len(sort_d) // 100 * 5]
    max_d = sort_d[len(sort_d) // 100 * 95]
    # min_d = 2.8
    # max_d = 4.6
    # print(min_d, max_d)
    depth = (depth - min_d) / (max_d - min_d) * 0.9 + 0.1

    viridis = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256)))
    depth_draw = viridis(depth.detach().cpu().numpy()[0])[..., :3]
    #  print(
    #      viridis(depth.detach().cpu().numpy()).shape,
    #      depth_draw.shape,
    #      depth.shape,
    #      mask.shape,
    #  )
    depth_draw = torch.from_numpy(depth_draw).to(depth.device).permute([2, 0, 1]) * mask

    return depth_draw


def normal2rgb(normal):
    # print('normals', normal.shape)
    # normal[..., 1:] *= -1
    # normal = (normal + 1) / 2
    return normal


def cos_loss(output, gt, mask=None, thrsh=0, weight=1):
    # breakpoint()
    output_n = output * 2 - 1
    gt_n = gt * 2 - 1
    if mask is not None:
        mask_n = mask.detach()
        output_n = output_n[mask_n]
        gt_n = gt_n[mask_n]
    cos = torch.sum(output_n * gt_n * weight, -1)
    return (1 - cos[cos < np.cos(thrsh)]).mean()


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def depth2wpos(depth, mask, camera):
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
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    pose = camera.world_view_transform.to(device)
    Rinv = pose[:3, :3]
    t = pose[3:, :3]
    camWPos = (camPos - t) @ Rinv.t()

    camWPos = (camWPos[..., :3] * mask).permute([2, 0, 1])

    return camWPos


def resample_points(camera, depth, normal, color, mask):
    camWPos = depth2wpos(depth, mask, camera).permute([1, 2, 0])
    camN = normal.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0]).to(torch.bool)
    mask = mask.detach()[..., 0]
    camN = camN.detach()[mask]
    camWPos = camWPos.detach()[mask]
    camRGB = color.permute([1, 2, 0])[mask]

    Rinv = camera.world_view_transform[:3, :3]

    points = torch.cat([camWPos, camN @ Rinv.t(), camRGB], -1)
    return points


def grid_prune(grid, shift, scale, dim, pts, thrsh=1):
    # print(dim)
    grid_cord = ((pts + shift) * scale).to(torch.long)
    # print(grid_cord.min(), grid_cord.max())
    out = (torch.le(grid_cord, 0) + torch.gt(grid_cord, dim - 1)).any(1)
    # print(grid_cord.min(), grid_cord.max())
    grid_cord = grid_cord.clamp(torch.zeros_like(dim), dim - 1)
    mask = grid[grid_cord[:, 0], grid_cord[:, 1], grid_cord[:, 2]] > thrsh
    mask *= ~out
    # print(grid_cord.shape, mask.shape, mask.sum())
    return mask.to(torch.bool)


import pymeshlab
from pytorch3d.ops import knn_points
from tqdm import tqdm


def poisson_mesh(path, vtx, normal, color, depth, thrsh):

    pbar = tqdm(total=4)
    pbar.update(1)
    pbar.set_description("Poisson meshing")

    # create pcl with normal from sampled points
    ms = pymeshlab.MeshSet()
    pts = pymeshlab.Mesh(vtx.cpu().numpy(), [], normal.cpu().numpy())
    ms.add_mesh(pts)

    # poisson reconstruction
    ms.generate_surface_reconstruction_screened_poisson(
        depth=depth, preclean=True, samplespernode=1.5
    )
    vert = ms.current_mesh().vertex_matrix()
    face = ms.current_mesh().face_matrix()
    ms.save_current_mesh(path + "_plain.ply")

    pbar.update(1)
    pbar.set_description("Mesh refining")
    # knn to compute distance and color of poisson-meshed points to sampled points
    nn_dist, nn_idx, _ = knn_points(
        torch.from_numpy(vert).to(torch.float32).cuda()[None], vtx.cuda()[None], K=4
    )
    nn_dist = nn_dist[0]
    nn_idx = nn_idx[0]
    nn_color = torch.mean(color[nn_idx], axis=1)

    # create mesh with color and quality (distance to the closest sampled points)
    vert_color = nn_color.clip(0, 1).cpu().numpy()
    vert_color = np.concatenate([vert_color, np.ones_like(vert_color[:, :1])], 1)
    ms.add_mesh(
        pymeshlab.Mesh(
            vert,
            face,
            v_color_matrix=vert_color,
            v_scalar_array=nn_dist[:, 0].cpu().numpy(),
        )
    )

    pbar.update(1)
    pbar.set_description("Mesh cleaning")
    # prune outlying vertices and faces in poisson mesh
    ms.compute_selection_by_condition_per_vertex(condselect=f"q>{thrsh}")
    ms.meshing_remove_selected_vertices()

    # fill holes
    ms.meshing_close_holes(maxholesize=300)
    ms.save_current_mesh(path + "_pruned.ply")

    # smoothing, correct boundary aliasing due to pruning
    ms.load_new_mesh(path + "_pruned.ply")
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)
    ms.save_current_mesh(path + "_pruned.ply")

    pbar.update(1)
    pbar.close()
