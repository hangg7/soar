#!/usr/bin/env python3
#
# File   : compute_normal.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 08/03/2024
#
# Distributed under terms of the MIT license.

import os
import os.path as osp
import sys
from glob import glob

import cv2
import imageio.v3 as iio
import numpy as np
import smplx
import torch
import torch.nn.functional as F
import tyro
from tqdm import tqdm

from soar.rendering import get_proj_mat, render_mesh

sys.path.insert(0, osp.join(osp.dirname(__file__), "../submodules/econ/"))
from lib.config import cfg as normal_cfg
from lib.net import NormalNet

sys.path.insert(0, osp.dirname(__file__))
from utils import transform_K_by_bbox


def main(
    data_dir: str,
    normal_checkpoint_path: str = osp.join(
        osp.dirname(__file__), "../datasets/checkpoints/normal.ckpt"
    ),
    model_path: str = osp.join(osp.dirname(__file__), "../datasets/models/"),
):
    img_dir = osp.join(data_dir, "images")
    mask_dir = osp.join(data_dir, "masks")
    normal_F_dir = osp.join(data_dir, "normal_F")
    normal_B_dir = osp.join(data_dir, "normal_B")
    smplx_param_path = osp.join(data_dir, "smplx/params.pth")

    if (
        osp.exists(normal_F_dir)
        and len(os.listdir(img_dir)) == len(os.listdir(normal_F_dir))
        and osp.exists(normal_B_dir)
        and len(os.listdir(img_dir)) == len(os.listdir(normal_B_dir))
    ):
        print("Normals already computed.")
    else:
        device = "cuda"

        os.makedirs(normal_F_dir, exist_ok=True)
        os.makedirs(normal_B_dir, exist_ok=True)

        params = torch.load(smplx_param_path, map_location=device, weights_only=True)
        body_model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            use_face_contour=True,
        ).to(device)

        with torch.inference_mode():
            body_output = body_model(
                **{k: v for k, v in params.items() if k not in ["Ks", "w2c", "img_wh"]}
            )

        normal_cfg.net.in_nml = (("image", 3), ("T_normal_F", 3), ("T_normal_B", 3))
        normal_cfg.net.in_geo = (("normal_F", 3), ("normal_B", 3))
        normal_model = NormalNet(normal_cfg)
        lightning_state_dict = torch.load(
            normal_checkpoint_path, map_location="cpu", weights_only=True
        )["state_dict"]
        normal_state_dict = {}
        for k, v in lightning_state_dict.items():
            if k.startswith("netG."):
                normal_state_dict[k.replace("netG.", "")] = v
        normal_model.load_state_dict(normal_state_dict, strict=False)
        normal_model.to(device)
        normal_model.eval()

        normal_Ks = []
        for frame_idx, (img_path, mask_path) in enumerate(
            zip(
                tqdm(sorted(glob(osp.join(img_dir, "*.png"))), "Computing normals"),
                sorted(glob(osp.join(mask_dir, "*.png"))),
            )
        ):
            input_img = (
                cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            )
            input_img = torch.from_numpy(input_img).to(device)
            if input_img.shape[-1] == 4:
                img_mask = input_img[..., 3:4]
            else:
                img_mask = (
                    cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    / 255.0
                )
                img_mask = torch.from_numpy(img_mask).to(device)  # [..., None]
            input_rgb = input_img[..., :3]
            input_rgb = torch.flip(input_rgb, [-1])
            input_img = (input_rgb * 2 - 1) * img_mask[..., None]

            mask_indices = torch.nonzero(img_mask)
            bbox = torch.cat(
                [mask_indices.min(0)[0].flip(0), mask_indices.max(0)[0].flip(0)]
            )
            bbox_c = bbox[:2] + (bbox[2:] - bbox[:2]) / 2.0
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox_s = max(bbox_w.item(), bbox_h.item()) * 1.1
            bbox = torch.cat([bbox_c - bbox_s / 2.0, bbox_c + bbox_s / 2.0])
            crop_wh = (512, 512)
            grid = (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(
                            bbox[0].item(), bbox[2].item(), crop_wh[0], device=device
                        )
                        / params["img_wh"][0],
                        torch.linspace(
                            bbox[1].item(), bbox[3].item(), crop_wh[1], device=device
                        )
                        / params["img_wh"][1],
                        indexing="xy",
                    ),
                    dim=-1,
                )[None]
                * 2.0
                - 1.0
            )
            cropped_img = F.grid_sample(
                input_img[None].permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                align_corners=False,
            )
            cropped_mask = F.grid_sample(
                img_mask[None, ..., None].permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                align_corners=False,
            )

            # Render normal in the crop space.
            K_c = transform_K_by_bbox(params["Ks"][frame_idx], bbox, crop_wh)
            normal_Ks.append(K_c)

            proj_front = get_proj_mat(
                K_c,
                crop_wh,
                znear=body_output.vertices[frame_idx][:, -1].mean() - 3.0,
            )
            with torch.inference_mode():
                rendered_F_pkg = render_mesh(
                    verts=body_output.vertices[frame_idx],
                    faces=torch.as_tensor(
                        body_model.faces.astype(np.int64), device=device
                    ),
                    w2c=params["w2c"],
                    proj_mat=proj_front,
                    img_wh=crop_wh,
                )
                rendered_F, rendered_F_mask = (
                    rendered_F_pkg["normal"],
                    rendered_F_pkg["mask"],
                )
            proj_back = proj_front.clone()
            proj_back[2] *= -1
            with torch.inference_mode():
                rendered_B_pkg = render_mesh(
                    verts=body_output.vertices[frame_idx],
                    faces=torch.as_tensor(
                        body_model.faces.astype(np.int64), device=device
                    ),
                    w2c=params["w2c"],
                    proj_mat=proj_back,
                    img_wh=crop_wh,
                )
                rendered_B, rendered_B_mask = (
                    rendered_B_pkg["normal"],
                    rendered_B_pkg["mask"],
                )
            rendered_F = (rendered_F * 2.0 - 1.0) * rendered_F_mask
            rendered_B = (rendered_B * 2.0 - 1.0) * rendered_B_mask
            normal_input = {
                "image": cropped_img.to(device),
                "T_normal_F": (rendered_F)[None].permute(0, 3, 1, 2),
                "T_normal_B": (rendered_B)[None].permute(0, 3, 1, 2),
            }
            with torch.no_grad():
                pred_normal_F, pred_normal_B = normal_model(normal_input)
            pred_normal_F = (pred_normal_F + 1.0) / 2.0 * cropped_mask
            pred_normal_F = torch.cat([pred_normal_F, cropped_mask], dim=1)
            pred_normal_B = (pred_normal_B + 1.0) / 2.0 * cropped_mask
            pred_normal_B = torch.cat([pred_normal_B, cropped_mask], dim=1)
            iio.imwrite(
                osp.join(normal_F_dir, f"{frame_idx:05d}.png"),
                (pred_normal_F[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                ),
            )
            iio.imwrite(
                osp.join(normal_B_dir, f"{frame_idx:05d}.png"),
                (pred_normal_B[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                ),
            )

        normal_Ks = torch.stack(normal_Ks)
        params["normal_Ks"] = normal_Ks.cpu()
        torch.save(params, smplx_param_path)


if __name__ == "__main__":
    tyro.cli(main)
