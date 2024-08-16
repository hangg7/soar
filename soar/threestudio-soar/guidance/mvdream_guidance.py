import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("mvdream-multiview-diffusion-guidance")
class MultiviewDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_name: str = (
            "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        )
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path).to(
            self.device
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="scheduler",
        )

        threestudio.info(f"Loaded Multiview Diffusion!")

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32] Latent space image

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        step_ratio=None,
        text_embeddings=None,
        aux_text_embeddings=False,
        input_is_latent=False,
        normal_flag=False,
        latent_mask=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        camera = c2w if not aux_text_embeddings else kwargs["head_c2ws"]
        #  print(f"camera: {camera.shape}")
        # breakpoint()

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation,
                azimuth,
                camera_distances,
                self.cfg.view_dependent_prompting,
                aux_text_embeddings=aux_text_embeddings,
                normal_flag=normal_flag,
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (
                    F.interpolate(
                        rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                    )
                    * 2
                    - 1
                )
            else:
                # interp to 256x256 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                if latent_mask is not None:
                    latent_mask = F.interpolate(
                        latent_mask.permute(0, 3, 1, 2),
                        (32, 32),
                        mode="bilinear",
                        align_corners=False,
                    )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if step_ratio is not None:
            timestep = np.round((1 - step_ratio) * self.num_train_timesteps).clip(
                self.min_step, self.max_step
            )
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)

        # if latent_mask is not None:
        #     latents_noisy = latents * (1 - latent_mask) + latents_noisy * latent_mask
        #     latent_model_input = torch.cat([latents_noisy] * 2)
        # else:

        with torch.no_grad():
            latent_model_input = torch.cat([latents_noisy] * 2)
            # pred noise
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(
            2
        )  # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(
                latents_noisy, t, noise_pred
            )

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(
                    latents_noisy, t, noise_pred_text
                )
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                    -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                )
                latents_recon_reshape = latents_recon.view(
                    -1, self.cfg.n_view, *latents_recon.shape[1:]
                )
                factor = (
                    latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8
                ) / (latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

                latents_recon_adjust = latents_recon.clone() * factor.squeeze(
                    1
                ).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = (
                    self.cfg.recon_std_rescale * latents_recon_adjust
                    + (1 - self.cfg.recon_std_rescale) * latents_recon
                )

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = (
                0.5
                * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
                / latents.shape[0]
            )
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # Original SDS
            # w(t), sigma_t^2
            w = 1 - self.alphas_cumprod[t]
            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    #  @torch.inference_mode()
    #  def refine(
    #      self,
    #      rgb: Float[Tensor, "B H W C"],
    #      prompt_utils: PromptProcessorOutput,
    #      elevation: Float[Tensor, "B"],
    #      azimuth: Float[Tensor, "B"],
    #      camera_distances: Float[Tensor, "B"],
    #      c2w: Float[Tensor, "B 4 4"],
    #      guidance_scale=100,
    #      steps=50,
    #      strength=0.8,
    #      **_,
    #  ):
    #      batch_size = rgb.shape[0]
    #      rgb_256 = F.interpolate(
    #          rgb.permute(0, 3, 1, 2),
    #          (256, 256),
    #          mode="bilinear",
    #          align_corners=False,
    #      )
    #      latents = self.encode_images(rgb_256)

    #      self.scheduler.set_timesteps(steps)
    #      init_step = int(steps * strength)
    #      latents = self.scheduler.add_noise(
    #          latents, torch.randn_like(latents), self.scheduler.timesteps[init_step]
    #      )

    #      camera = self.get_camera_cond(c2w, None)
    #      text_embeddings = prompt_utils.get_text_embeddings(
    #          elevation,
    #          azimuth,
    #          camera_distances,
    #          self.cfg.view_dependent_prompting,
    #      )
    #      camera = camera.repeat(2, 1).to(text_embeddings)

    #      context = {
    #          "context": text_embeddings,
    #          "camera": camera,
    #          "num_frames": self.cfg.n_view,
    #      }

    #      for _, t in enumerate(self.scheduler.timesteps[init_step:]):
    #          latent_model_input = torch.cat([latents] * 2)
    #          tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)
    #          noise_pred = self.model.apply_model(latent_model_input, tt, context)
    #          noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
    #          noise_pred = noise_pred_uncond + guidance_scale * (
    #              noise_pred_text - noise_pred_uncond
    #          )
    #          latents = self.scheduler.step(noise_pred, t, latents).prev_sample

    #      imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
    #      return imgs

    #  def decode_latents(self, latents):
    #      imgs = self.model.decode_first_stage(latents)
    #      imgs = ((imgs + 1) / 2).clamp(0, 1)
    #      return imgs
