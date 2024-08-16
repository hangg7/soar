import os
import random
from dataclasses import dataclass, field

import lpips
import numpy as np
import torch
from torchvision.utils import save_image

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *

from ..geometry.gaussian_base import BasicPointCloud
from ..utils.loss_utils import l1_loss_w, ssim


def scale_gradients_hook(grad, mask=None):
    grad_copy = grad.clone()  # Make a copy to avoid in-place modifications
    # Assume we want to scale the gradients of the first 5 rows by half
    if mask is not None:
        grad_copy *= mask
    return grad_copy


@threestudio.register("gaussiandreamer-mvdream-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

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
        self.loss_fn_vgg = lpips.LPIPS(net="alex").cuda()

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

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
            batch, mode="gen", head_flag=head_flag
        )
        return outputs_all

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        #  pose_optimizer = torch.optim.SparseAdam(self.geometry.smpl_guidance.param, 5.0e-3)
        head_flag = random.random() < 0.4
        out, gt_out = self(batch, head_flag=head_flag)

        rendered_out = torch.cat([gt_out["comp_rgb"], batch["gt_rgb"]], dim=0)
        rendered_out = rendered_out.permute(0, 3, 1, 2)
        # print('mask shapes', gt_out["comp_mask"].shape, batch['gt_mask'].shape)
        rendered_out_mask = torch.cat(
            [gt_out["comp_mask"], batch["gt_mask"][..., None]], dim=0
        )
        rendered_out_mask = rendered_out_mask.permute(0, 3, 1, 2)

        rendered_out_normal = torch.cat(
            [gt_out["comp_normal"], batch["gt_normal_F"]], dim=0
        )
        rendered_out_normal = rendered_out_normal.permute(0, 3, 1, 2)
        # .permute(0,3,1,2)
        # rendered_out = torch.flip(rendered_out, [1])
        # print(rendered_out.shape)
        save_image(rendered_out, "test.png")
        save_image(rendered_out_mask, "mask.png")
        save_image(rendered_out_normal, "normal.png")
        # exit()
        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["comp_rgb"]
        guidance_normal_inp = out["comp_pred_normal"]
        guidance_mask = out["comp_mask"]
        viewspace_point_tensor = out["viewspace_points"]

        gt_visibility_filter = gt_out["visibility_filter"]
        gt_radii = gt_out["radii"]
        gt_viewspace_point_tensor = gt_out["viewspace_points"]

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
        if head_flag:
            guidance_inp.register_hook(
                lambda grad: scale_gradients_hook(
                    grad, mask=(torch.exp(-3 * guidance_mask.detach()))
                )
            )
            guidance_normal_inp.register_hook(
                lambda grad: scale_gradients_hook(
                    grad, mask=(torch.exp(-3 * guidance_mask.detach()))
                )
            )
        else:
            guidance_inp.register_hook(
                lambda grad: scale_gradients_hook(
                    grad, mask=(torch.exp(-2 * guidance_mask.detach()))
                )
            )
            guidance_normal_inp.register_hook(
                lambda grad: scale_gradients_hook(
                    grad, mask=(torch.exp(-2 * guidance_mask.detach()))
                )
            )
        save_image(guidance_inp.permute(0, 3, 1, 2), "guidance.png")

        guidance_out = self.guidance(
            guidance_inp,
            self.prompt_utils,
            **batch,
            rgb_as_latents=False,
            aux_text_embeddings=head_flag,
            latent_mask=torch.exp(
                -2 * guidance_mask.detach()
            ),  # , step_ratio=step_ratio#, timestep=timestep
        )
        guidance_normal_out = self.guidance(
            guidance_normal_inp,
            self.prompt_utils,
            **batch,
            rgb_as_latents=False,
            aux_text_embeddings=head_flag,
            normal_flag=True,
        )

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

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        for name, value in guidance_normal_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        check_points = self.geometry.smpl_guidance.query_points + self.geometry.get_xyz
        check_points_att = self.geometry.attribute_field(check_points)
        check_color, check_scale = check_points_att["shs"], check_points_att["scales"]
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
            print("Delta mean: ", delta_mean.mean())

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
            # scale_sum = torch.sum(self.geometry.get_scaling)
            scale_sum = torch.mean(check_scale)
            self.log(f"train/scales", scale_sum)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

        if self.cfg.loss["lambda_tv_loss"] > 0.0:
            loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                out["comp_rgb"].permute(0, 3, 1, 2)
            )
            self.log(f"train/loss_tv", loss_tv)
            loss += loss_tv

        if self.cfg.loss["lambda_recon"] > 0.0:
            # loss_recon =torch.abs(gt_out["comp_rgb"] - batch["gt_rgb"]).mean() * self.C(self.cfg.loss["lambda_recon"])
            loss_recon = 0.8 * l1_loss_w(gt_out["comp_rgb"], batch["gt_rgb"]) + 0.2 * (
                1
                - ssim(
                    gt_out["comp_rgb"].permute(0, 3, 1, 2),
                    batch["gt_rgb"].permute(0, 3, 1, 2),
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

        if self.cfg.loss["lambda_normal_F"] > 0.0:
            loss_normal = torch.abs(
                gt_out["comp_pred_normal"] - batch["gt_normal_F"]
            ).mean() * self.C(self.cfg.loss["lambda_normal_F"])
            self.log(f"train/loss_normal_F", loss_normal)
            loss += loss_normal

        if self.cfg.loss["lambda_vgg"] > 0.0:
            vgg_loss = (
                self.C(self.cfg.loss["lambda_vgg"])
                * self.loss_fn_vgg(
                    (gt_out["comp_rgb"].permute(0, 3, 1, 2) - 0.5) * 2,
                    (batch["gt_rgb"].permute(0, 3, 1, 2) - 0.5) * 2,
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

        if out.__contains__("comp_pred_normal"):
            loss_pred_normal = torch.nn.functional.mse_loss(
                out["comp_pred_normal"], out["comp_normal"]  # .detach()
            )
            loss += loss_pred_normal

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        # loss_sds = 0.0
        iteration = self.global_step
        if loss_sds > 0 and iteration > 0:
            loss_sds.backward(retain_graph=True)
        # if viewspace_point_tensor[0].grad is not None:
        #     self.geometry.update_states(
        #         iteration,
        #         visibility_filter,
        #         radii,
        #         viewspace_point_tensor,
        #     )
        # if gt_viewspace_point_tensor[0].grad is not None:
        #     self.geometry.update_states(
        #         iteration,
        #         gt_visibility_filter,
        #         gt_radii,
        #         gt_viewspace_point_tensor,
        #     )

        if loss > 0:
            # print("Optimizing loss with loss recon: ", loss_recon)
            # if iteration < 200:
            #     loss = loss_occ
            loss.backward()
        opt.step()
        #  pose_optimizer.step()
        opt.zero_grad(set_to_none=True)
        #  pose_optimizer.zero_grad(set_to_none=True)

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
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0]}.png",
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
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_occ" in out
                else []
            ),
            name="test_step",
            step=self.global_step,
        )
        # save_image(out["comp_occ"].permute(0,3,1,2), f"it{self.global_step}-test/{batch['index'][0]}_occ.png")
        if batch["index"][0] == 0:
            save_path = self.get_save_path("point_cloud.ply")
            self.geometry.save_ply(save_path)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
