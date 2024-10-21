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
from transforms3d.axangles import mat2axangle, axangle2mat
#from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
#from smplx.smplx import SMPLLayer
from torchvision.utils import save_image
import tyro
from tqdm import tqdm

from soar.rendering import get_proj_mat, render_mesh

sys.path.insert(0, osp.join(osp.dirname(__file__), "../submodules/econ/"))
from lib.config import cfg as normal_cfg
from lib.net import NormalNet

sys.path.insert(0, osp.dirname(__file__))
from utils import transform_K_by_bbox

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def trans_smpl_rt(smpl_data, smpl_layer, T_cw):
    smpl_theta = smpl_data["poses"].reshape((24, 3))
    smpl_rot, smpl_trans = smpl_data["Rh"][0], smpl_data["Th"]
    smpl_R = axangle2mat(
        smpl_rot / (np.linalg.norm(smpl_rot) + 1e-6), np.linalg.norm(smpl_rot)
    )

    T_wh = np.eye(4)
    T_wh[:3, :3], T_wh[:3, 3] = smpl_R.copy(), smpl_trans.squeeze(0).copy()

    T_ch = T_cw.astype(np.float64) @ T_wh.astype(np.float64)

    smpl_global_rot_d, smpl_global_rot_a = mat2axangle(T_ch[:3, :3])
    smpl_global_rot = smpl_global_rot_d * smpl_global_rot_a
    smpl_trans = T_ch[:3, 3]  # 3
    smpl_theta[0] = smpl_global_rot
    beta = smpl_data["shapes"][0][:10]

    # ! Because SMPL global rot is rot around joint-0, have to correct this in the global translation!!
    _pose = axis_angle_to_matrix(torch.from_numpy(smpl_theta)[None])
    so = smpl_layer(
        torch.from_numpy(beta)[None].float(),
        body_pose=_pose[:, 1:].float(),
    )
    j0 = (so.joints[0, 0]).numpy()
    t_correction = (_pose[0, 0].numpy() - np.eye(3)) @ j0
    smpl_trans = smpl_trans + t_correction

    smpl_arranged = {
        "betas": beta[None],
        "body_pose": smpl_theta.flatten()[3:][None],
        "global_orient": smpl_global_rot[None], 
        "transl": smpl_trans[None] 
    }

    return smpl_arranged

def main(
    data_dir: str,
    normal_checkpoint_path: str = osp.join(
        osp.dirname(__file__), "../data/ckpt/normal.ckpt"
    ),
    model_path: str = osp.join(osp.dirname(__file__), "../data/smpl_related/models/"),
):
    img_dir = osp.join(data_dir, "Camera_B1")
    mask_dir = osp.join(data_dir, "mask", "Camera_B1")
    mask_cihp_dir = osp.join(data_dir, "mask_cihp", "Camera_B1")
    normal_F_dir = osp.join(data_dir, "normal_F")
    normal_B_dir = osp.join(data_dir, "normal_B")
    smplx_param_paths = sorted(glob(osp.join(data_dir, "new_params", "*.npy")), key=lambda x: int(x.split('/')[-1].split('.')[0]))

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

        #params = torch.load(smplx_param_path, map_location=device, weights_only=True)
        body_model = smplx.create(
            model_path=model_path,
            model_type="smpl",
            gender="neutral",
            use_pca=False,
            use_face_contour=True,
        ).to(device)

        smpl_layer = smplx.SMPLLayer(
            osp.join(model_path, "smpl"), kid_template_path="data/smpl_related/models/smpl/SMPL_NEUTRAL.pkl")

        
        annots = np.load(osp.join(data_dir, "annots.npy"), allow_pickle=True).item()
        select_view = 0
        cams = annots['cams']
        cam_Ks = np.array(cams['K'])[select_view].astype('float32')
        cam_Rs = np.array(cams['R'])[select_view].astype('float32')
        cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.
        cam_Ds = np.array(cams['D'])[select_view].astype('float32')
        cam_T = cam_Ts[:3, 0]
        E = np.eye(4)
        E[:3, :3] = cam_Rs
        E[:3, 3]= cam_T
        params = {
            "betas": [], "body_pose": [], "global_orient": [], "transl": []
        }
        for smplx_param_path in smplx_param_paths:
            smplx_param = np.load(smplx_param_path, allow_pickle=True).item()
            smpl_arranged = trans_smpl_rt(smplx_param, smpl_layer, E)
           
            for k, v in smpl_arranged.items():
                params[k].append(v)
        for k, v in params.items():
            params[k] = torch.from_numpy(np.concatenate(v)).float().to(device)

        params["w2c"] = torch.inverse(torch.from_numpy(E).float().to(device))
        bg = np.array([0, 0, 0], dtype=np.float32)
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
                tqdm(sorted(glob(osp.join(img_dir, "*.jpg"))), "Computing normals"),
                sorted(glob(osp.join(mask_dir, "*.png"))),
            )
        ):
            input_img = (
                cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            )
            K, D = cam_Ks, cam_Ds
            input_img = cv2.undistort(input_img, K, D)
            input_img = input_img.astype(np.float32) / 255.0
            
            params['img_wh'] = input_img.shape[:2]
            if input_img.shape[-1] == 4:
                img_mask = input_img[..., 3:4]
            else:
                img_mask = (
                    cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                )
                img_mask = (img_mask != 0).astype(np.uint8)
                mask_cihp = cv2.imread(mask_cihp_dir + '/' + mask_path.split('/')[-1], cv2.IMREAD_UNCHANGED)
                mask_cihp = (mask_cihp != 0).astype(np.uint8)
                img_mask = (img_mask | mask_cihp).astype(np.uint8)
                img_mask[img_mask == 1] = 255
            img_mask = cv2.undistort(img_mask, K, D)
            img_mask = img_mask.astype(np.float32) / 255.0
            input_img = input_img * img_mask[..., None] + bg * (1.0 - img_mask[..., None])

            input_img = torch.from_numpy(input_img).to(device)
            img_mask = torch.from_numpy(img_mask).to(device)  # [..., None]

            #save_image(input_img.permute(2,0,1), 'input_img.jpg')
            #save_image(img_mask[None], 'img_mask_.png')
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
            K_c = transform_K_by_bbox(torch.from_numpy(cam_Ks).to(device).float(), bbox, crop_wh)
            normal_Ks.append(K_c)

            proj_front = get_proj_mat(
                K_c,
                crop_wh,
                znear=0.001, #body_output.vertices[frame_idx][:, -1].mean() - 3.0,
            )
            with torch.inference_mode():
                rendered_F_pkg = render_mesh(
                    verts=body_output.vertices[frame_idx],
                    faces=torch.as_tensor(
                        body_model.faces.astype(np.int64), device=device
                    ),
                    w2c=torch.eye(4).to(params["w2c"]),
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
                    w2c=torch.eye(4).to(params["w2c"]),
                    proj_mat=proj_back,
                    img_wh=crop_wh,
                )
                rendered_B, rendered_B_mask = (
                    rendered_B_pkg["normal"],
                    rendered_B_pkg["mask"],
                )
            #save_image(rendered_F.permute(2,0,1), "rendered_F.jpg")
            #save_image(rendered_B.permute(2,0,1), "rendered_B.jpg")
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
        torch.save(params, osp.join(data_dir, "params.pth")) 


if __name__ == "__main__":
    tyro.cli(main)
