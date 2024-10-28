import math
import os
import random
from dataclasses import dataclass

import lpips
import numpy as np
import torch
from torchvision.utils import save_image

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *

from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..utils.loss_utils import l1_loss_w, ssim

import skimage
from skimage.metrics import structural_similarity as ski_ssim
import lpips

loss_fn_lpips = lpips.LPIPS(net='vgg', version='0.1').cuda()
loss_fn_lpips = loss_fn_lpips.eval()

def scale_gradients_hook(grad, mask=None):
    grad_copy = grad.clone()  # Make a copy to avoid in-place modifications
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

        self.sds_start = 0 if self.cfg.training_stage == 1 else 500
        self.resampled = []
        self.psnrs = []
        self.lpips = []
        self.ssims = []

    def configure_optimizers(self):
        optim = self.geometry.optimizer
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

        outputs_all = self.renderer.batch_forward(
            batch, mode="gen", head_flag=head_flag, stage=self.cfg.training_stage
        )
        return outputs_all

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        iteration = self.global_step

        head_flag = random.random() < 0.4
        out, gt_out = self(batch, head_flag=head_flag)

        guidance_inp = out["comp_rgb"]
        guidance_normal_inp = out["comp_normal"].clone()
        guidance_occ_mask = out["comp_occ"]
        comp_bg = gt_out["comp_bg"]

        save_dir = self.get_save_dir()
        if iteration % 250 == 0:
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

        if self.cfg.loss["lambda_opacity"] > 0.0:
            scaling = self.geometry.get_scaling.norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
            ).sum()
            self.log(f"train/loss_opacity", loss_opacity)
            loss += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

        if self.cfg.loss["lambda_sparsity"] > 0.0:
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
            loss_normal = (
                0.2
                * cos_loss(
                    gt_out["comp_normal"][[0]],
                    batch["gt_normal_F"],
                    normal_mask,
                    thrsh=0,
                    weight=1,
                )
                + 1 * loss_fn_lpips(
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

        if self.cfg.loss["lambda_normal_B"] > 0.0 and "gt_normal_B" in batch:
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
                + loss_fn_lpips(
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
                    * loss_fn_lpips(
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

        if loss > 0:
            loss.backward()

        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss_sds + loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
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

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):

        out, gt_out = self(batch)
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
        loss_fn_lpips_test = loss_fn_lpips.cpu()
        with torch.no_grad():
            lpips_ = loss_fn_lpips_test(
                torch.from_numpy(pred)[None].permute(0, 3, 1, 2) * 2 - 1, 
                torch.from_numpy(gt)[None].permute(0, 3, 1, 2) * 2 - 1).mean()
            self.lpips.append(lpips_)
        print("PSNR: ", psnr_, "SSIM: ", ssim_, "LPIPS: ", lpips_)

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


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def depth2rgb(depth, mask):
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
    return normal


def cos_loss(output, gt, mask=None, thrsh=0, weight=1):
    output_n = output * 2 - 1
    gt_n = gt * 2 - 1
    if mask is not None:
        mask_n = mask.detach()
        output_n = output_n[mask_n]
        gt_n = gt_n[mask_n]
    cos = torch.sum(output_n * gt_n * weight, -1)
    return (1 - cos[cos < np.cos(thrsh)]).mean()
