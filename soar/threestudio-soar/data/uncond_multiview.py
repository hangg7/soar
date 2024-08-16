import math
import os
import random
from dataclasses import dataclass, field
from glob import glob

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import pickle
import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from transforms3d.axangles import mat2axangle, axangle2mat


def fov2K(fov=90, H=256, W=256):
    if isinstance(fov, torch.Tensor):
        f = H / (2 * torch.tan(fov / 2 * np.pi / 180))
        K = torch.eye(3).repeat(fov.shape[0], 1, 1).to(fov)
        K[:, 0, 0], K[:, 0, 2] = f, W / 2.0
        K[:, 1, 1], K[:, 1, 2] = f, H / 2.0
        return K.clone()
    else:
        f = H / (2 * np.tan(fov / 2 * np.pi / 180))
        K = np.eye(3)
        K[0, 0], K[0, 2] = f, W / 2.0
        K[1, 1], K[1, 2] = f, H / 2.0
        return K.copy()


def transform_K_by_bbox(K, bbox, crop_wh):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x1, y1, x2, y2 = bbox
    W_c, H_c = crop_wh
    fx_c = W_c / (x2 - x1) * fx
    fy_c = H_c / (y2 - y1) * fy
    cx_c = W_c / (x2 - x1) * (cx - x1)
    cy_c = H_c / (y2 - y1) * (cy - y1)
    return torch.tensor(
        [
            [fx_c, 0.0, cx_c],
            [0.0, fy_c, cy_c],
            [0.0, 0.0, 1.0],
        ],
        device=K.device,
    )

def get_projection_matrix_cxcy(
    fovy: Float[Tensor, "B"],
    aspect_wh: float,
    near: float,
    far: float,
    cxcy: Optional[tuple[float, float]] = None,
    img_wh: Optional[tuple[float, float]] = None,
) -> Float[Tensor, "B 4 4"]:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    if cxcy is not None and img_wh is not None:
        cx, cy = cxcy
        W, H = img_wh
        proj_mtx[:, 0, 2] = -(2.0 * cx - W) / W
        proj_mtx[:, 1, 2] = -(2.0 * cy - H) / H
    return proj_mtx


@dataclass
class RandomMultiviewCameraDataModuleConfig(RandomCameraDataModuleConfig):
    dataroot: str = ""
    relative_radius: bool = True
    n_view: int = 1
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    smpl_type: str = "smpl"
    index_range: Tuple[int, int] = (0, 1)

    occ_range: int = 405
    occ_mid: int = 451
    occ_width: int = 86
    rays_d_normalize: bool = True


class RandomMultiviewCameraIterableDataset(RandomCameraIterableDataset):
    def __init__(self, cfg: Any, split: str = "train", *args, **kwargs):
        
        super().__init__(cfg)
        self.zoom_range = self.cfg.zoom_range

        if self.cfg.smpl_type == "smpl":
            img_list = sorted(
                glob(os.path.join(self.cfg.dataroot, "train", "images", "*.png"))
            )
            mask_list = sorted(
                glob(os.path.join(self.cfg.dataroot, "train", "masks", "*.png"))
            )
        else:
            img_list = sorted(glob(os.path.join(self.cfg.dataroot, "images", "*.png")))
            mask_list = sorted(glob(os.path.join(self.cfg.dataroot, "masks", "*.png")))
        normal_F_list = sorted(
            glob(os.path.join(self.cfg.dataroot, "normal_F", "*.png"))
        )
        normal_B_list = sorted(
            glob(os.path.join(self.cfg.dataroot, "normal_B", "*.png"))
        )
        threestudio.info(f"Found {len(img_list)} images in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(mask_list)} masks in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(normal_F_list)} normal_F in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(normal_B_list)} normal_B in {self.cfg.dataroot}")
        assert len(img_list) == len(mask_list) == len(normal_F_list) == len(
            normal_B_list
        ), "Number of images and masks should be the same"
        scene_length = len(img_list)
        num_val = scene_length // 5
        length = int(1 / (num_val) * scene_length)
        offset = length // 2
        val_list = list(range(scene_length))[offset::length]
        train_list = list(set(range(scene_length)) - set(val_list))
        test_list = val_list[:len(val_list) // 2]
        val_list = val_list[len(val_list) // 2:]
        split_type = split
        threestudio.info(f"Using {split_type} split")
        train_list = [0, 4] 
        if split_type == "train":
            self.index_list = train_list
        elif split_type == "val":
            self.index_list = val_list
        elif split_type == "test":
            self.index_list = test_list
        frames_img = []
        frames_mask = []
        frames_normal_F = []
        frames_normal_B = []
        frames_normal_mask = []
        for i, img_path in tqdm(enumerate(img_list)):

            img = cv2.imread(
                img_path, cv2.IMREAD_UNCHANGED
            )  # .astype(np.float32) / 255.0
            if img.shape[-1] == 4:
                mask = img[..., 3]
                img = img[..., :3]
            else:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_UNCHANGED)
            mask[mask > 0] = 1.0
            #  img = img * mask[..., None]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_img.append(img)
            frames_mask.append(mask)

            if len(normal_F_list) > 0:
                normal_F = cv2.imread(normal_F_list[i], cv2.IMREAD_UNCHANGED)
                normal_mask = normal_F[..., 3]
                normal_F = normal_F[..., :3]
                normal_F = cv2.cvtColor(normal_F, cv2.COLOR_BGR2RGB)
                normal_B = cv2.imread(normal_B_list[i], cv2.IMREAD_UNCHANGED)
                normal_B = normal_B[..., :3]
                normal_B = cv2.cvtColor(normal_B, cv2.COLOR_BGR2RGB)
                
                frames_normal_F.append(normal_F)
                frames_normal_B.append(normal_B)
                frames_normal_mask.append(normal_mask)
        frames_img = np.stack(frames_img, axis=0)
        frames_mask = np.stack(frames_mask, axis=0)
        self.frames_img = torch.from_numpy(frames_img).float() / 255.0
        self.frames_mask = torch.from_numpy(frames_mask).float()

        self.frames_img = self.frames_img * self.frames_mask[..., None]

        if len(normal_F_list) > 0:
            frames_normal_F = np.stack(frames_normal_F, axis=0)
            self.frames_normal_F = torch.from_numpy(frames_normal_F).float() / 255.0
            frames_normal_B = np.stack(frames_normal_B, axis=0)
            self.frames_normal_B = torch.from_numpy(frames_normal_B).float() / 255.0
            frames_normal_mask = np.stack(frames_normal_mask, axis=0)
            self.frames_normal_mask = (
                torch.from_numpy(frames_normal_mask).float() / 255.0
            )
        else:
            self.frames_normal_F = []
            self.frames_normal_B = []
            self.frames_normal_mask = []
        gt_width, gt_height = frames_img.shape[2], frames_img.shape[1]
        self.gt_width, self.gt_height = gt_width, gt_height
        self.n_frames = len(self.frames_img)

        if self.cfg.index_range[1] == -1:
            self.cfg.index_range = (0, self.n_frames)
        self.cfg.index_range = (
            max(0, self.cfg.index_range[0]),
            min(self.n_frames, self.cfg.index_range[1]),
        )
        threestudio.info(f"Using index range {self.cfg.index_range}")

        if self.cfg.smpl_type == "smpl":
            camera = np.load(os.path.join(self.cfg.dataroot, "cameras.npz"))
            intrinsic = np.array(camera["intrinsic"])
            extrinsic = np.array(camera["extrinsic"])
            self.intrinsic = torch.from_numpy(intrinsic).float()
            self.extrinsic = torch.from_numpy(extrinsic).float()
        else:
            body_data = torch.load(
                os.path.join(self.cfg.dataroot, "smplx/params.pth"), map_location="cpu"
            )
            self.extrinsic = body_data["w2c"]
            self.intrinsics = body_data["Ks"]
            self.normal_intrinsics = body_data["normal_Ks"]
        print("using smpl_type", self.cfg.smpl_type)

        self.extrinsic[1:3] *= -1

        if self.cfg.smpl_type == "smpl":
            self.smpl_parms = self.load_smpl_param(
                os.path.join(self.cfg.dataroot, "poses_optimized.npz")
            )
        else:
            self.smpl_parms = self.load_smpl_param(
                os.path.join(self.cfg.dataroot, "smplx", "params.pth")
            )
 
        frames_img_crop = []
        frames_mask_crop = []
        frames_rays_d = []
        for i, (img, mask) in enumerate(zip(self.frames_img, self.frames_mask)):
            mask_indices = torch.nonzero(mask)
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
                        torch.linspace(bbox[0].item(), bbox[2].item(), crop_wh[0])
                        / self.frames_img.shape[2],
                        torch.linspace(bbox[1].item(), bbox[3].item(), crop_wh[1])
                        / self.frames_img.shape[1],
                        indexing="xy",
                    ),
                    dim=-1,
                )[None]
                * 2.0
                - 1.0
            )
            cropped_img = F.grid_sample(
                img[None].permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                align_corners=False,
            )
            cropped_mask = F.grid_sample(
                mask[None, ..., None].permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                align_corners=False,
            )
            # breakpoint()
            cropped_directions = get_ray_directions(
                H=crop_wh[1],
                W=crop_wh[0],
                focal=(
                    self.normal_intrinsics[i, 0, 0],
                    self.normal_intrinsics[i, 1, 1],
                ),
                principal=(
                    self.normal_intrinsics[i, 0, 2],
                    self.normal_intrinsics[i, 1, 2],
                ),
            )[None, ...]

            c2w = torch.inverse(self.extrinsic).unsqueeze(0).to(cropped_img.device)
            cropped_rays_o, cropped_rays_d = get_rays(
                cropped_directions, c2w, keepdim=True, normalize=True
            )

            cropped_img = cropped_img[0].permute(1, 2, 0)
            cropped_mask = cropped_mask[0, 0]
            frames_img_crop.append(cropped_img)
            frames_mask_crop.append(cropped_mask)
            frames_rays_d.append(cropped_rays_d)
        frames_img_crop = torch.stack(frames_img_crop, dim=0)
        frames_mask_crop = torch.stack(frames_mask_crop, dim=0)
        self.frames_img_crop = frames_img_crop
        self.frames_mask_crop = frames_mask_crop
        self.frames_rays_d = torch.cat(frames_rays_d, dim=0)

    def load_smpl_param(self, path):
        if self.cfg.smpl_type == "smpl":
            smpl_params = dict(np.load(str(path)))
        else:
            smpl_params = torch.load(str(path))
        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        if self.cfg.smpl_type == "smpl":
            return {
                "betas": smpl_params["betas"].astype(np.float32),
                "body_pose": smpl_params["body_pose"].astype(np.float32),
                "global_orient": smpl_params["global_orient"].astype(np.float32),
                "transl": smpl_params["transl"].astype(np.float32),
            }
        else:
            return {
                "betas": smpl_params["betas"],
                "body_pose": smpl_params["body_pose"],
                "global_orient": smpl_params["global_orient"],
                "transl": smpl_params["transl"],
            }

    def collate(self, batch) -> Dict[str, Any]:
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]

        c2w = torch.eye(4, device="cpu").unsqueeze(0)
        # gt_index = torch.randint(0, self.n_frames, (1,)).item()
        gt_index = torch.randint(0, len(self.index_list), (1,)).item()
        gt_index = self.index_list[gt_index]
        # print("c2w", c2w.shape, c2w);
        # print("proj_mtx", proj_mtx.shape, proj_mtx);
        # gt_c2w = torch.eye(4, device=c2w.device).unsqueeze(0)
        # gt_c2w[:, :3, 3] += torch.tensor([0.0, 0.0, 0.8], device=c2w.device)
        gt_c2w = torch.inverse(self.extrinsic).unsqueeze(0).to(c2w.device)
        # gt_c2w[:3, 1:3] *= -1
        if self.cfg.smpl_type == "smpl":
            gt_fx = self.intrinsic[0, 0]
            gt_fy = self.intrinsic[1, 1]
            gt_cx = self.intrinsic[0, 2]
            gt_cy = self.intrinsic[1, 2]
        else:
            gt_fx = self.intrinsics[gt_index, 0, 0]
            gt_fy = self.intrinsics[gt_index, 1, 1]
            gt_cx = self.intrinsics[gt_index, 0, 2]
            gt_cy = self.intrinsics[gt_index, 1, 2]
        gt_fovy = 2 * torch.atan(self.gt_height / (2 * gt_fy))
        gt_fovx = 2 * torch.atan(self.gt_width / (2 * gt_fx))
        gt_fovy = torch.tensor(gt_fovy, device=c2w.device).unsqueeze(0)
        gt_fovx = torch.tensor(gt_fovx, device=c2w.device).unsqueeze(0)
        gt_width = self.gt_width
        gt_height = self.gt_height
        gt_cx = torch.tensor(gt_cx, device=c2w.device).unsqueeze(0)
        gt_cy = torch.tensor(gt_cy, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.tensor(90.0, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.deg2rad(gt_fovy)
        # gt_height = 512
        # gt_width = 512
        # gt_fovx = 2 * torch.atan(torch.tan(gt_fovy / 2) * self.width / self.height)
        gt_focal_length = 0.5 * gt_height / torch.tan(0.5 * gt_fovy)
        gt_near = 0.1
        if self.cfg.smpl_type == "smplx":
            gt_near = self.smpl_parms["transl"][gt_index][-1].item() - 5.0
        gt_proj_mtx = get_projection_matrix_cxcy(
            gt_fovy,
            gt_width / gt_height,
            gt_near,
            1000.0,
            **(
                {}
                if self.cfg.smpl_type == "smpl"
                else {
                    "cxcy": (gt_cx.item(), gt_cy.item()),
                    "img_wh": (gt_width, gt_height),
                }
            ),
        )

        gt_normal_res = 512
        gt_normal_fx = self.normal_intrinsics[gt_index, 0, 0]
        gt_normal_fy = self.normal_intrinsics[gt_index, 1, 1]
        gt_normal_cx = self.normal_intrinsics[gt_index, 0, 2]
        gt_normal_cy = self.normal_intrinsics[gt_index, 1, 2]
        gt_normal_fovy = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fy))
        gt_normal_fovx = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fx))
        gt_normal_fovy = torch.tensor(gt_normal_fovy, device=c2w.device).unsqueeze(0)
        gt_normal_fovx = torch.tensor(gt_normal_fovx, device=c2w.device).unsqueeze(0)
        gt_normal_cx = torch.tensor(gt_normal_cx, device=c2w.device).unsqueeze(0)
        gt_normal_cy = torch.tensor(gt_normal_cy, device=c2w.device).unsqueeze(0)

        gt_mvp_mtx = get_mvp_matrix(gt_c2w, gt_proj_mtx)
        gt_near = torch.tensor(gt_near, device=c2w.device).unsqueeze(0)
        # breakpoint()
        gt_directions = get_ray_directions(
            H=gt_normal_res,
            W=gt_normal_res,
            focal=(gt_normal_fx, gt_normal_fy),
            principal=(gt_normal_cx, gt_normal_cy),
        )[None, ...]
        # gt_directions = get_ray_directions(
        #     H=gt_height,
        #     W=gt_width,
        #     focal=(gt_fx, gt_fy),
        #     principal=(gt_cx, gt_cy),
        # )[None, ...]
        # print('gt_directions', gt_directions.shape, gt_directions, gt_focal_length.shape, gt_focal_length)
        # gt_directions[:, :, :, :2] = (
        #     gt_directions[:, :, :, :2] / gt_focal_length[:, None, None, None]
        # )

        gt_rays_o, gt_rays_d = get_rays(
            gt_directions, gt_c2w, keepdim=True, normalize=True
        )
        # gt_index = 0
        # print('check fovy', gt_fovy, fovy)
        # print('check directions', gt_directions.shape, directions.shape)
        # print('check rays_o', gt_rays_o.shape, rays_o.shape)
        # print('check rays_d', gt_rays_d.shape, rays_d.shape)
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################
        # fovy = gt_normal_fovy.repeat_interleave(self.cfg.n_view, dim=0)
        # camera_distances = torch.tensor([self.smpl_parms['transl'][gt_index][..., -1]], device=c2w.device).repeat_interleave(
        #     self.cfg.n_view, dim=0)

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(
                    self.cfg.n_view, dim=0
                )
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )
        # breakpoint()
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        # TODO(Hang Gao @ 05/03): This information is incomplete for SMPLX
        # models, but since we are not actually using it so it's okay for now.
        smpl_collate = {
            "betas": self.smpl_parms["betas"][None],
            "body_pose": self.smpl_parms["body_pose"][gt_index][None],
            "global_orient": self.smpl_parms["global_orient"][gt_index][None],
            "transl": self.smpl_parms["transl"][gt_index][None],
        }
        # breakpoint()
        out_dict = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "frames_rays_d": self.frames_rays_d,
            "cam_d": directions,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
            # gt data
            "gt_index": gt_index,
            "gt_rays_o": gt_rays_o,
            "gt_rays_d": gt_rays_d,
            "gt_cam_d": gt_directions,
            "gt_mvp_mtx": gt_mvp_mtx,
            "gt_c2w": gt_c2w,
            "gt_fovx": gt_fovx,
            "gt_fovy": gt_fovy,
            "gt_cx": gt_cx,
            "gt_cy": gt_cy,
            "gt_normal_fovx": gt_normal_fovx,
            "gt_normal_fovy": gt_normal_fovy,
            "gt_normal_cx": gt_normal_cx,
            "gt_normal_cy": gt_normal_cy,
            "gt_normal_res": gt_normal_res,
            "gt_near": gt_near,
            "gt_height": gt_height,
            "gt_width": gt_width,
            "gt_smpl": smpl_collate,
            "gt_rgb": self.frames_img[gt_index : gt_index + 1],
            "gt_mask": self.frames_mask[gt_index : gt_index + 1],
            "gt_rgb_crop": self.frames_img_crop[gt_index : gt_index + 1],
            "gt_mask_crop": self.frames_mask_crop[gt_index : gt_index + 1],
        }
        if len(self.frames_normal_F) > 0:
            out_dict["gt_normal_F"] = self.frames_normal_F[gt_index : gt_index + 1]
            out_dict["gt_normal_B"] = self.frames_normal_B[gt_index : gt_index + 1]
            out_dict["gt_normal_mask"] = self.frames_normal_mask[
                gt_index : gt_index + 1
            ]
        return out_dict

class ValDataset(Dataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        self.zoom_range = self.cfg.zoom_range
        
        if self.cfg.smpl_type == "smpl":
            img_list = sorted(
                glob(os.path.join(self.cfg.dataroot, "train", "images", "*.png"))
            )
            mask_list = sorted(
                glob(os.path.join(self.cfg.dataroot, "train", "masks", "*.png"))
            )
        else:
            img_list = sorted(glob(os.path.join(self.cfg.dataroot, "images", "*.png")))
            mask_list = sorted(glob(os.path.join(self.cfg.dataroot, "masks", "*.png")))
        normal_F_list = sorted(
            glob(os.path.join(self.cfg.dataroot, "normal_F", "*.png"))
        )
        normal_B_list = sorted(
            glob(os.path.join(self.cfg.dataroot, "normal_B", "*.png"))
        )
        threestudio.info(f"Found {len(img_list)} images in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(mask_list)} masks in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(normal_F_list)} normal_F in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(normal_B_list)} normal_B in {self.cfg.dataroot}")
        assert len(img_list) == len(mask_list) == len(normal_F_list) == len(
            normal_B_list
        ), "Number of images and masks should be the same"
        scene_length = len(img_list)
        num_val = scene_length // 5
        length = int(1 / (num_val) * scene_length)
        offset = length // 2
        val_list = list(range(scene_length))[offset::length]
        train_list = list(set(range(scene_length)) - set(val_list))
        test_list = val_list[:len(val_list) // 2]
        val_list = val_list[len(val_list) // 2:]
        split_type = "test"
        threestudio.info(f"Using {split_type} split") 
        if split_type == "train":
            self.index_list = train_list
        elif split_type == "val":
            self.index_list = val_list
        elif split_type == "test":
            self.index_list = test_list
        self.index_list = [0, 1, 2, 3, 4, 5, 6, 7]
        frames_img = []
        frames_mask = []
        frames_normal_F = []
        frames_normal_B = []
        frames_normal_mask = []
        for i, img_path in tqdm(enumerate(img_list)):

            img = cv2.imread(
                img_path, cv2.IMREAD_UNCHANGED
            )  # .astype(np.float32) / 255.0
            if img.shape[-1] == 4:
                mask = img[..., 3]
                img = img[..., :3]
            else:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_UNCHANGED)
            mask[mask > 0] = 1.0
            #  img = img * mask[..., None]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_img.append(img)
            frames_mask.append(mask)

            if len(normal_F_list) > 0:
                normal_F = cv2.imread(normal_F_list[i], cv2.IMREAD_UNCHANGED)
                normal_mask = normal_F[..., 3]
                normal_F = normal_F[..., :3]
                normal_F = cv2.cvtColor(normal_F, cv2.COLOR_BGR2RGB)
                normal_B = cv2.imread(normal_B_list[i], cv2.IMREAD_UNCHANGED)
                normal_B = normal_B[..., :3]
                normal_B = cv2.cvtColor(normal_B, cv2.COLOR_BGR2RGB)
                
                frames_normal_F.append(normal_F)
                frames_normal_B.append(normal_B)
                frames_normal_mask.append(normal_mask)
        frames_img = np.stack(frames_img, axis=0)
        frames_mask = np.stack(frames_mask, axis=0)
        self.frames_img = torch.from_numpy(frames_img).float() / 255.0
        self.frames_mask = torch.from_numpy(frames_mask).float()

        self.frames_img = self.frames_img * self.frames_mask[..., None]

        if len(normal_F_list) > 0:
            frames_normal_F = np.stack(frames_normal_F, axis=0)
            self.frames_normal_F = torch.from_numpy(frames_normal_F).float() / 255.0
            frames_normal_B = np.stack(frames_normal_B, axis=0)
            self.frames_normal_B = torch.from_numpy(frames_normal_B).float() / 255.0
            frames_normal_mask = np.stack(frames_normal_mask, axis=0)
            self.frames_normal_mask = (
                torch.from_numpy(frames_normal_mask).float() / 255.0
            )
        else:
            self.frames_normal_F = []
            self.frames_normal_B = []
            self.frames_normal_mask = []
        gt_width, gt_height = frames_img.shape[2], frames_img.shape[1]
        self.gt_width, self.gt_height = gt_width, gt_height
        self.n_frames = len(self.frames_img)

        if self.cfg.index_range[1] == -1:
            self.cfg.index_range = (0, self.n_frames)
        self.cfg.index_range = (
            max(0, self.cfg.index_range[0]),
            min(self.n_frames, self.cfg.index_range[1]),
        )
        threestudio.info(f"Using index range {self.cfg.index_range}")

        if self.cfg.smpl_type == "smpl":
            camera = np.load(os.path.join(self.cfg.dataroot, "cameras.npz"))
            intrinsic = np.array(camera["intrinsic"])
            extrinsic = np.array(camera["extrinsic"])
            self.intrinsic = torch.from_numpy(intrinsic).float()
            self.extrinsic = torch.from_numpy(extrinsic).float()
        else:
            body_data = torch.load(
                os.path.join(self.cfg.dataroot, "smplx/params.pth"), map_location="cpu"
            )
            self.extrinsic = body_data["w2c"]
            self.intrinsics = body_data["Ks"]
            self.normal_intrinsics = body_data["normal_Ks"]
        print("using smpl_type", self.cfg.smpl_type)

        self.extrinsic[1:3] *= -1

        if self.cfg.smpl_type == "smpl":
            self.smpl_parms = self.load_smpl_param(
                os.path.join(self.cfg.dataroot, "poses_optimized.npz")
            )
        else:
            self.smpl_parms = self.load_smpl_param(
                os.path.join(self.cfg.dataroot, "smplx", "params.pth")
            )
        
    def load_smpl_param(self, path):
        if self.cfg.smpl_type == "smpl":
            smpl_params = dict(np.load(str(path)))
        else:
            smpl_params = torch.load(str(path))
        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        if self.cfg.smpl_type == "smpl":
            return {
                "betas": smpl_params["betas"].astype(np.float32),
                "body_pose": smpl_params["body_pose"].astype(np.float32),
                "global_orient": smpl_params["global_orient"].astype(np.float32),
                "transl": smpl_params["transl"].astype(np.float32),
            }
        else:
            return {
                "betas": smpl_params["betas"],
                "body_pose": smpl_params["body_pose"],
                "global_orient": smpl_params["global_orient"],
                "transl": smpl_params["transl"],
            }
    
    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]

        c2w = torch.eye(4, device="cpu").unsqueeze(0)
        gt_index = self.index_list[index]

        gt_c2w = torch.inverse(self.extrinsic).unsqueeze(0).to(c2w.device)
        # gt_c2w[:3, 1:3] *= -1
        if self.cfg.smpl_type == "smpl":
            gt_fx = self.intrinsic[0, 0]
            gt_fy = self.intrinsic[1, 1]
            gt_cx = self.intrinsic[0, 2]
            gt_cy = self.intrinsic[1, 2]
        else:
            gt_fx = self.intrinsics[gt_index, 0, 0]
            gt_fy = self.intrinsics[gt_index, 1, 1]
            gt_cx = self.intrinsics[gt_index, 0, 2]
            gt_cy = self.intrinsics[gt_index, 1, 2]
        gt_fovy = 2 * torch.atan(self.gt_height / (2 * gt_fy))
        gt_fovx = 2 * torch.atan(self.gt_width / (2 * gt_fx))
        gt_fovy = torch.tensor(gt_fovy, device=c2w.device).unsqueeze(0)
        gt_fovx = torch.tensor(gt_fovx, device=c2w.device).unsqueeze(0)
        gt_width = self.gt_width
        gt_height = self.gt_height
        gt_cx = torch.tensor(gt_cx, device=c2w.device).unsqueeze(0)
        gt_cy = torch.tensor(gt_cy, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.tensor(90.0, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.deg2rad(gt_fovy)
        # gt_height = 512
        # gt_width = 512
        # gt_fovx = 2 * torch.atan(torch.tan(gt_fovy / 2) * self.width / self.height)
        gt_focal_length = 0.5 * gt_height / torch.tan(0.5 * gt_fovy)
        gt_near = 0.1
        if self.cfg.smpl_type == "smplx":
            gt_near = self.smpl_parms["transl"][gt_index][-1].item() - 5.0
        gt_proj_mtx = get_projection_matrix_cxcy(
            gt_fovy,
            gt_width / gt_height,
            gt_near,
            1000.0,
            **(
                {}
                if self.cfg.smpl_type == "smpl"
                else {
                    "cxcy": (gt_cx.item(), gt_cy.item()),
                    "img_wh": (gt_width, gt_height),
                }
            ),
        )

        gt_normal_res = 512
        gt_normal_fx = self.normal_intrinsics[gt_index, 0, 0]
        gt_normal_fy = self.normal_intrinsics[gt_index, 1, 1]
        gt_normal_cx = self.normal_intrinsics[gt_index, 0, 2]
        gt_normal_cy = self.normal_intrinsics[gt_index, 1, 2]
        gt_normal_fovy = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fy))
        gt_normal_fovx = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fx))
        gt_normal_fovy = torch.tensor(gt_normal_fovy, device=c2w.device).unsqueeze(0)
        gt_normal_fovx = torch.tensor(gt_normal_fovx, device=c2w.device).unsqueeze(0)
        gt_normal_cx = torch.tensor(gt_normal_cx, device=c2w.device).unsqueeze(0)
        gt_normal_cy = torch.tensor(gt_normal_cy, device=c2w.device).unsqueeze(0)

        gt_mvp_mtx = get_mvp_matrix(gt_c2w, gt_proj_mtx)
        gt_near = torch.tensor(gt_near, device=c2w.device).unsqueeze(0)
        # breakpoint()
        gt_directions = get_ray_directions(
            H=gt_normal_res,
            W=gt_normal_res,
            focal=(gt_normal_fx, gt_normal_fy),
            principal=(gt_normal_cx, gt_normal_cy),
        )[None, ...]


        gt_rays_o, gt_rays_d = get_rays(
            gt_directions, gt_c2w, keepdim=True, normalize=True
        )

        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################
        # fovy = gt_normal_fovy.repeat_interleave(self.cfg.n_view, dim=0)
        # camera_distances = torch.tensor([self.smpl_parms['transl'][gt_index][..., -1]], device=c2w.device).repeat_interleave(
        #     self.cfg.n_view, dim=0)

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(
                    self.cfg.n_view, dim=0
                )
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )
        # breakpoint()
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        # TODO(Hang Gao @ 05/03): This information is incomplete for SMPLX
        # models, but since we are not actually using it so it's okay for now.
        smpl_collate = {
            "betas": self.smpl_parms["betas"][None],
            "body_pose": self.smpl_parms["body_pose"][gt_index][None],
            "global_orient": self.smpl_parms["global_orient"][gt_index][None],
            "transl": self.smpl_parms["transl"][gt_index][None],
        }
        out_dict = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            # "frames_rays_d": self.frames_rays_d,
            "cam_d": directions,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
            # gt data
            "gt_index": gt_index,
            "gt_rays_o": gt_rays_o,
            "gt_rays_d": gt_rays_d,
            "gt_cam_d": gt_directions,
            "gt_mvp_mtx": gt_mvp_mtx,
            "gt_c2w": gt_c2w,
            "gt_fovx": gt_fovx,
            "gt_fovy": gt_fovy,
            "gt_cx": gt_cx,
            "gt_cy": gt_cy,
            "gt_normal_fovx": gt_normal_fovx,
            "gt_normal_fovy": gt_normal_fovy,
            "gt_normal_cx": gt_normal_cx,
            "gt_normal_cy": gt_normal_cy,
            "gt_normal_res": gt_normal_res,
            "gt_near": gt_near,
            "gt_height": gt_height,
            "gt_width": gt_width,
            "gt_smpl": smpl_collate,
            "gt_rgb": self.frames_img[gt_index : gt_index + 1],
            "gt_mask": self.frames_mask[gt_index : gt_index + 1],
            # "gt_rgb_crop": self.frames_img_crop[gt_index : gt_index + 1],
            # "gt_mask_crop": self.frames_mask_crop[gt_index : gt_index + 1],
        }
        if len(self.frames_normal_F) > 0:
            out_dict["gt_normal_F"] = self.frames_normal_F[gt_index : gt_index + 1]
            out_dict["gt_normal_B"] = self.frames_normal_B[gt_index : gt_index + 1]
            out_dict["gt_normal_mask"] = self.frames_normal_mask[
                gt_index : gt_index + 1
            ]
        return out_dict
    
    def collate(self, batch):
        return batch[0]
    
class FSRandomMultiviewCameraIterableDataset(RandomCameraIterableDataset):
    def __init__(self, cfg: Any, split: str = "train", *args, **kwargs):
        
        super().__init__(cfg)
        self.zoom_range = self.cfg.zoom_range


        img_list = sorted(glob(os.path.join(self.cfg.dataroot, "basecolor", "*.png")))
        mask_list = sorted(glob(os.path.join(self.cfg.dataroot, "mask", "*.png")))
        normal_F_list = sorted(
            glob(os.path.join(self.cfg.dataroot, "normal", "*.png"))
        )
        # normal_B_list = sorted(
        #     glob(os.path.join(self.cfg.dataroot, "normal_B", "*.png"))
        # )
        threestudio.info(f"Found {len(img_list)} images in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(mask_list)} masks in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(normal_F_list)} normal_F in {self.cfg.dataroot}")
        # threestudio.info(f"Found {len(normal_B_list)} normal_B in {self.cfg.dataroot}")
        assert len(img_list) == len(mask_list) == len(normal_F_list), "Number of images and masks should be the same"
        scene_length = len(img_list)
        num_val = scene_length // 5
        length = int(1 / (num_val) * scene_length)
        offset = length // 2
        val_list = list(range(scene_length))[offset::length]
        train_list = list(set(range(scene_length)) - set(val_list))
        test_list = val_list[:len(val_list) // 2]
        val_list = val_list[len(val_list) // 2:]
        split_type = split
        threestudio.info(f"Using {split_type} split")
        train_list = [0, 4] 
        if split_type == "train":
            self.index_list = train_list
        elif split_type == "val":
            self.index_list = val_list
        elif split_type == "test":
            self.index_list = test_list
        frames_img = []
        frames_mask = []
        frames_normal_F = []
        frames_normal_B = []
        frames_normal_mask = []
        for i, img_path in tqdm(enumerate(img_list)):

            img = cv2.imread(
                img_path, cv2.IMREAD_UNCHANGED
            )  # .astype(np.float32) / 255.0
            if img.shape[-1] == 4:
                mask = img[..., 3]
                img = img[..., :3]
            else:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_UNCHANGED)
            mask[mask > 0] = 1.0
            #  img = img * mask[..., None]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_img.append(img)
            frames_mask.append(mask)

            if len(normal_F_list) > 0:
                normal_F = cv2.imread(normal_F_list[i], cv2.IMREAD_UNCHANGED)
                normal_mask = mask * 255
                normal_F = normal_F[..., :3]
                normal_F = cv2.cvtColor(normal_F, cv2.COLOR_BGR2RGB)
                
                frames_normal_F.append(normal_F)
                frames_normal_mask.append(normal_mask)
        frames_img = np.stack(frames_img, axis=0)
        frames_mask = np.stack(frames_mask, axis=0)
        self.frames_img = torch.from_numpy(frames_img).float() / 255.0
        self.frames_mask = torch.from_numpy(frames_mask).float()

        self.frames_img = self.frames_img * self.frames_mask[..., None]

        if len(normal_F_list) > 0:
            frames_normal_F = np.stack(frames_normal_F, axis=0)
            self.frames_normal_F = torch.from_numpy(frames_normal_F).float() / 255.0
            frames_normal_mask = np.stack(frames_normal_mask, axis=0)
            self.frames_normal_mask = (
                torch.from_numpy(frames_normal_mask).float() / 255.0
            )
        else:
            self.frames_normal_F = []
            self.frames_normal_mask = []
        gt_width, gt_height = frames_img.shape[2], frames_img.shape[1]
        self.gt_width, self.gt_height = gt_width, gt_height
        self.n_frames = len(self.frames_img)

        if self.cfg.index_range[1] == -1:
            self.cfg.index_range = (0, self.n_frames)
        self.cfg.index_range = (
            max(0, self.cfg.index_range[0]),
            min(self.n_frames, self.cfg.index_range[1]),
        )
        threestudio.info(f"Using index range {self.cfg.index_range}")


        # body_data = torch.load(
        #     os.path.join(self.cfg.dataroot, "smplx/params.pth"), map_location="cpu"
        # )

        smplx_paths = sorted(glob(os.path.join(self.cfg.dataroot, "..", "..", "training", os.path.basename(self.cfg.dataroot), "smplx", "*.pkl")))
        if 'smplx.pkl' in smplx_paths[-2]:
            meshpose_path = smplx_paths[-2]
        else:
            meshpose_path = smplx_paths[-1]
        #os.path.join(self.cfg.dataroot, "..", "..", "training", os.path.basename(self.cfg.dataroot), "smplx", "mesh_smplx.pkl") 
        with open(meshpose_path, "rb") as f:
            meshpose = pickle.load(f)
        device = torch.device("cpu")
        # self.smpl_parms = {
        #     "betas" : body_data["betas"], 
        #     "global_orient":apose[:, 0:3].to(device),
        #     "body_pose":apose[:, 3:66].to(device),
        #     "jaw_pose":apose[:, 66:69].to(device),
        #     "leye_pose":apose[:, 69:72].to(device),
        #     "reye_pose":apose[:, 72:75].to(device),
        #     "left_hand_pose":apose[:, 75:120].to(device),
        #     "right_hand_pose":apose[:, 120:165].to(device),
        #     "expression":apose[:, 165:175].to(device),
        #     "transl": torch.zeros_like(body_data["transl"][0:1]).to(device),
        # }

        betas = torch.tensor(meshpose['betas'].reshape(-1,10))
        transl = torch.zeros((1,3))
        global_orient = torch.zeros((1,3))
        self.smplx_parms = []
        for smplx_path in smplx_paths[:self.n_frames]:
            with open(smplx_path, "rb") as f:
                smplx_pose = pickle.load(f)
            if smplx_pose.shape[1] == 175:
                expression = smplx_pose[:, 165:175]
            else:
                expression = torch.zeros((1,10))
            smplx_parm = {
                "betas" : betas.to(device),
                "global_orient": smplx_pose[:, :3].to(device),
                "body_pose":smplx_pose[:, 3:66].to(device),
                "jaw_pose":smplx_pose[:, 66:69].to(device),
                "leye_pose":smplx_pose[:, 69:72].to(device),
                "reye_pose":smplx_pose[:, 72:75].to(device),
                "left_hand_pose":smplx_pose[:, 75:120].to(device),
                "right_hand_pose":smplx_pose[:, 120:165].to(device),
                "expression":expression.to(device),
                "transl": transl.to(device),
            }
            self.smplx_parms.append(smplx_parm)
                
       
        angle_90 = torch.full((8,), 90.0)
        thetas = angle_90.to(device) 

        ph = [x for x in range(0, -360, -45)]
        phis = torch.FloatTensor(ph).reshape(-1).to(device)
        # phis = torch.full((8,), 0.0).to(device)
        radii = torch.full((8,), 3.2).to(device)
        poses, dirs = circle_poses(device, radius=radii, theta=thetas, phi=phis)
        
        
        self.extrinsics= torch.inverse(poses) #body_data["w2c"]
        W, H = 1024, 1024
        fov = 20
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        self.intrinsic = torch.tensor(
            [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], device=device
        ).float()
        self.normal_intrinsics = self.intrinsic.cpu() / 2
        
        for i, (ext, normal_F) in enumerate(zip(self.extrinsics, self.frames_normal_F)):
            normal_F_ = normal_F * 2 - 1
            normal_F_local = ext[:3, :3] @ normal_F_.reshape(-1, 3).T
            normal_F_local = normal_F_local.T.reshape(normal_F.shape)
            normal_F_local = normal_F_local / torch.norm(normal_F_local, dim=-1, keepdim=True)
            normal_F_local = (normal_F_local + 1) / 2
            normal_F_local = normal_F_local * self.frames_normal_mask[i][..., None]
            normal_F = normal_F_local
            self.frames_normal_F[i] = normal_F
            
        
        frames_img_crop = []
        frames_mask_crop = []
        frames_rays_d = []
        for i, (img, mask) in enumerate(zip(self.frames_img, self.frames_mask)):
            mask_indices = torch.nonzero(mask)
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
                        torch.linspace(bbox[0].item(), bbox[2].item(), crop_wh[0])
                        / self.frames_img.shape[2],
                        torch.linspace(bbox[1].item(), bbox[3].item(), crop_wh[1])
                        / self.frames_img.shape[1],
                        indexing="xy",
                    ),
                    dim=-1,
                )[None]
                * 2.0
                - 1.0
            )
            cropped_img = F.grid_sample(
                img[None].permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                align_corners=False,
            )
            cropped_mask = F.grid_sample(
                mask[None, ..., None].permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                align_corners=False,
            )
            # breakpoint()
            cropped_directions = get_ray_directions(
                H=crop_wh[1],
                W=crop_wh[0],
                focal=(
                    self.normal_intrinsics[0, 0],
                    self.normal_intrinsics[1, 1],
                ),
                principal=(
                    self.normal_intrinsics[0, 2],
                    self.normal_intrinsics[1, 2],
                ),
            )[None, ...]

            c2w = torch.inverse(self.extrinsics[i]).unsqueeze(0).to(cropped_img.device)
            cropped_rays_o, cropped_rays_d = get_rays(
                cropped_directions, c2w, keepdim=True, normalize=True
            )

            cropped_img = cropped_img[0].permute(1, 2, 0)
            cropped_mask = cropped_mask[0, 0]
            frames_img_crop.append(cropped_img)
            frames_mask_crop.append(cropped_mask)
            frames_rays_d.append(cropped_rays_d)
        frames_img_crop = torch.stack(frames_img_crop, dim=0)
        frames_mask_crop = torch.stack(frames_mask_crop, dim=0)
        self.frames_img_crop = frames_img_crop
        self.frames_mask_crop = frames_mask_crop
        self.frames_rays_d = torch.cat(frames_rays_d, dim=0)

    def load_smpl_param(self, path):
        if self.cfg.smpl_type == "smpl":
            smpl_params = dict(np.load(str(path)))
        else:
            smpl_params = torch.load(str(path))
        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        if self.cfg.smpl_type == "smpl":
            return {
                "betas": smpl_params["betas"].astype(np.float32),
                "body_pose": smpl_params["body_pose"].astype(np.float32),
                "global_orient": smpl_params["global_orient"].astype(np.float32),
                "transl": smpl_params["transl"].astype(np.float32),
            }
        else:
            return {
                "betas": smpl_params["betas"],
                "body_pose": smpl_params["body_pose"],
                "global_orient": smpl_params["global_orient"],
                "transl": smpl_params["transl"],
            }

    def collate(self, batch) -> Dict[str, Any]:
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]

        c2w = torch.eye(4, device="cpu").unsqueeze(0)
        # gt_index = 0
        # gt_index = torch.randint(0, self.n_frames, (1,)).item()
        gt_index = torch.randint(0, len(self.index_list), (1,)).item()
        gt_index = self.index_list[gt_index]
        gt_c2w = torch.inverse(self.extrinsics[gt_index]).unsqueeze(0).to(c2w.device)

        gt_fx = self.intrinsic[0, 0]
        gt_fy = self.intrinsic[1, 1]
        gt_cx = self.intrinsic[0, 2]
        gt_cy = self.intrinsic[1, 2]

        gt_fovy = 2 * torch.atan(self.gt_height / (2 * gt_fy))
        gt_fovx = 2 * torch.atan(self.gt_width / (2 * gt_fx))
        gt_fovy = torch.tensor(gt_fovy, device=c2w.device).unsqueeze(0)
        gt_fovx = torch.tensor(gt_fovx, device=c2w.device).unsqueeze(0)
        gt_width = self.gt_width
        gt_height = self.gt_height
        gt_cx = torch.tensor(gt_cx, device=c2w.device).unsqueeze(0)
        gt_cy = torch.tensor(gt_cy, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.tensor(90.0, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.deg2rad(gt_fovy)
        # gt_height = 512
        # gt_width = 512
        # gt_fovx = 2 * torch.atan(torch.tan(gt_fovy / 2) * self.width / self.height)
        gt_focal_length = 0.5 * gt_height / torch.tan(0.5 * gt_fovy)
        gt_near = 0.01

        gt_proj_mtx = get_projection_matrix_cxcy(
            gt_fovy,
            gt_width / gt_height,
            gt_near,
            1000.0,
            **(
                {}
                if self.cfg.smpl_type == "smpl"
                else {
                    "cxcy": (gt_cx.item(), gt_cy.item()),
                    "img_wh": (gt_width, gt_height),
                }
            ),
        )

        gt_normal_res = 512
        gt_normal_fx = self.normal_intrinsics[0, 0]
        gt_normal_fy = self.normal_intrinsics[1, 1]
        gt_normal_cx = self.normal_intrinsics[0, 2]
        gt_normal_cy = self.normal_intrinsics[1, 2]
        gt_normal_fovy = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fy))
        gt_normal_fovx = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fx))
        gt_normal_fovy = torch.tensor(gt_normal_fovy, device=c2w.device).unsqueeze(0)
        gt_normal_fovx = torch.tensor(gt_normal_fovx, device=c2w.device).unsqueeze(0)
        gt_normal_cx = torch.tensor(gt_normal_cx, device=c2w.device).unsqueeze(0)
        gt_normal_cy = torch.tensor(gt_normal_cy, device=c2w.device).unsqueeze(0)

        gt_mvp_mtx = get_mvp_matrix(gt_c2w, gt_proj_mtx)
        gt_near = torch.tensor(gt_near, device=c2w.device).unsqueeze(0)
        # breakpoint()
        gt_directions = get_ray_directions(
            H=gt_normal_res,
            W=gt_normal_res,
            focal=(gt_normal_fx, gt_normal_fy),
            principal=(gt_normal_cx, gt_normal_cy),
        )[None, ...]

        gt_rays_o, gt_rays_d = get_rays(
            gt_directions, gt_c2w, keepdim=True, normalize=True
        )

        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################
        # fovy = gt_normal_fovy.repeat_interleave(self.cfg.n_view, dim=0)
        # camera_distances = torch.tensor([self.smpl_parms['transl'][gt_index][..., -1]], device=c2w.device).repeat_interleave(
        #     self.cfg.n_view, dim=0)

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(
                    self.cfg.n_view, dim=0
                )
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )
        # breakpoint()
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        # TODO(Hang Gao @ 05/03): This information is incomplete for SMPLX
        # models, but since we are not actually using it so it's okay for now.

        # breakpoint()
        out_dict = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "frames_rays_d": self.frames_rays_d,
            "cam_d": directions,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
            # gt data
            "gt_index": gt_index,
            "gt_rays_o": gt_rays_o,
            "gt_rays_d": gt_rays_d,
            "gt_cam_d": gt_directions,
            "gt_mvp_mtx": gt_mvp_mtx,
            "gt_c2w": gt_c2w,
            "gt_fovx": gt_fovx,
            "gt_fovy": gt_fovy,
            "gt_cx": gt_cx,
            "gt_cy": gt_cy,
            "gt_normal_fovx": gt_normal_fovx,
            "gt_normal_fovy": gt_normal_fovy,
            "gt_normal_cx": gt_normal_cx,
            "gt_normal_cy": gt_normal_cy,
            "gt_normal_res": gt_normal_res,
            "gt_near": gt_near,
            "gt_height": gt_height,
            "gt_width": gt_width,
            "gt_a_smpl": self.smplx_parms[gt_index],
            "gt_rgb": self.frames_img[gt_index : gt_index + 1],
            "gt_mask": self.frames_mask[gt_index : gt_index + 1],
            "gt_rgb_crop": self.frames_img_crop[gt_index : gt_index + 1],
            "gt_mask_crop": self.frames_mask_crop[gt_index : gt_index + 1],
        }
        if len(self.frames_normal_F) > 0:
            out_dict["gt_normal_F"] = self.frames_normal_F[gt_index : gt_index + 1]
            out_dict["gt_normal_B"] = None
            out_dict["gt_normal_mask"] = self.frames_normal_mask[
                gt_index : gt_index + 1
            ]
        return out_dict
    
class FSValDataset(Dataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        self.zoom_range = self.cfg.zoom_range
       
        datapath = os.path.join(self.cfg.dataroot, "..", "..", "testing", os.path.basename(self.cfg.dataroot)) 
        img_list = sorted(glob(os.path.join(datapath, "basecolor", "*.png")), key=lambda x: int(x.split("/")[-1].split(".")[0]))
        mask_list = sorted(glob(os.path.join(datapath, "mask", "*.png")), key=lambda x: int(x.split("/")[-1].split(".")[0]))

        threestudio.info(f"Found {len(img_list)} images in {datapath}")
        threestudio.info(f"Found {len(mask_list)} masks in {datapath}")

        frames_img = []
        frames_mask = []

        for i, img_path in tqdm(enumerate(img_list)):

            img = cv2.imread(
                img_path, cv2.IMREAD_UNCHANGED
            )
            if img.shape[-1] == 4:
                mask = img[..., 3]
                img = img[..., :3]
            else:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_UNCHANGED)
            mask[mask > 0] = 1.0
            #  img = img * mask[..., None]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_img.append(img)
            frames_mask.append(mask)

        frames_img = np.stack(frames_img, axis=0)
        frames_mask = np.stack(frames_mask, axis=0)
        self.frames_img = torch.from_numpy(frames_img).float() / 255.0
        self.frames_mask = torch.from_numpy(frames_mask).float()

        self.frames_img = self.frames_img * self.frames_mask[..., None]

        gt_width, gt_height = frames_img.shape[2], frames_img.shape[1]
        self.gt_width, self.gt_height = gt_width, gt_height
        self.n_frames = len(self.frames_img)

        if self.cfg.index_range[1] == -1:
            self.cfg.index_range = (0, self.n_frames)
        self.cfg.index_range = (
            max(0, self.cfg.index_range[0]),
            min(self.n_frames, self.cfg.index_range[1]),
        )
        threestudio.info(f"Using index range {self.cfg.index_range}")

        # raise NotImplementedError("Not implemented yet")
        # TODO: figure out correct smplx
        body_data = torch.load(
            os.path.join(self.cfg.dataroot, "..", "..", "soar", os.path.basename(self.cfg.dataroot), "smplx/params.pth"), map_location="cpu"
        )
        apose_path = os.path.join(self.cfg.dataroot, "..", "..", "training", "Apose.pkl")
        meshpose_path = os.path.join(self.cfg.dataroot, "..", "..", "training", os.path.basename(self.cfg.dataroot), "smplx", "mesh_smplx.pkl") 
        with open(apose_path, "rb") as f:
            apose = pickle.load(f)
        with open(meshpose_path, "rb") as f:
            tpose = pickle.load(f)
        # breakpoint()
        device = torch.device("cuda")
        # self.smpl_parms = {
        #     "betas" : body_data["betas"], 
        #     "global_orient":apose[:, 0:3].to(device),
        #     "body_pose":apose[:, 3:66].to(device),
        #     "jaw_pose":apose[:, 66:69].to(device),
        #     "leye_pose":apose[:, 69:72].to(device),
        #     "reye_pose":apose[:, 72:75].to(device),
        #     "left_hand_pose":apose[:, 75:120].to(device),
        #     "right_hand_pose":apose[:, 120:165].to(device),
        #     "expression":apose[:, 165:175].to(device),
        #     "transl": torch.zeros_like(body_data["transl"][0:1]).to(device),
        # }
        betas = torch.tensor(tpose['betas'].reshape(-1,10))
        jaw_pose = torch.tensor(tpose['jaw_pose'].reshape(-1,3))
        leye_pose = torch.tensor(tpose['leye_pose'].reshape(-1,3))
        reye_pose = torch.tensor(tpose['reye_pose'].reshape(-1,3))
        right_hand_pose = torch.tensor(tpose['right_hand_pose'].reshape(-1,45))
        left_hand_pose = torch.tensor(tpose['left_hand_pose'].reshape(-1,45))
        transl = torch.tensor(tpose['transl'].reshape(-1,3))
        body_pose = torch.tensor(tpose['body_pose'].reshape(-1,63))
        global_orient = torch.tensor(tpose['global_orient'].reshape(-1,3))
        global_orient = torch.zeros((1,3))
        self.smpl_parms = {
            "betas" : betas.to(device),
            "global_orient":global_orient.to(device),
            "body_pose":body_pose.to(device),
            "jaw_pose":jaw_pose.to(device),
            "leye_pose":leye_pose.to(device),
            "reye_pose":reye_pose.to(device),
            "left_hand_pose":left_hand_pose.to(device),
            "right_hand_pose":right_hand_pose.to(device),
            "expression":torch.zeros((1,10)).to(device),
            "transl":torch.zeros_like(transl).to(device),
        }
        
       
        # breakpoint()
        angle_45 = torch.full((8,), 45.0)
        angle_90 = torch.full((8,), 90.0)
        angle_135 = torch.full((8,), 135.0)
        thetas = torch.cat((angle_45, angle_90, angle_135)).reshape(-1).to(device)

        ph = [x for x in range(-90, 270, 45)]
        phis = torch.FloatTensor(ph).repeat(3).reshape(-1).to(device)
        radii = torch.full((24,), 3.2).to(device)
        poses, dirs = circle_poses(device, radius=radii, theta=thetas, phi=phis)
        
        
        self.extrinsics= torch.inverse(poses) #body_data["w2c"]
        W, H = 1024, 1024
        fov = 20
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        self.intrinsic = torch.tensor(
            [[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], device=device
        ).float()
        self.normal_intrinsics = body_data["normal_Ks"]
        print("using smpl_type", self.cfg.smpl_type)

        # self.extrinsics[:, 1:3] *= -1
        
    
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, index):
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]

        c2w = torch.eye(4, device="cpu").unsqueeze(0)
        gt_index = index 

        gt_c2w = torch.inverse(self.extrinsics[index]).unsqueeze(0).to(c2w.device)
        # gt_c2w[:3, 1:3] *= -1
        gt_fx = self.intrinsic[0, 0]
        gt_fy = self.intrinsic[1, 1]
        gt_cx = self.intrinsic[0, 2]
        gt_cy = self.intrinsic[1, 2]

        gt_fovy = 2 * torch.atan(self.gt_height / (2 * gt_fy))
        gt_fovx = 2 * torch.atan(self.gt_width / (2 * gt_fx))
        gt_fovy = torch.tensor(gt_fovy, device=c2w.device).unsqueeze(0)
        gt_fovx = torch.tensor(gt_fovx, device=c2w.device).unsqueeze(0)
        gt_width = self.gt_width
        gt_height = self.gt_height
        gt_cx = torch.tensor(gt_cx, device=c2w.device).unsqueeze(0)
        gt_cy = torch.tensor(gt_cy, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.tensor(90.0, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.deg2rad(gt_fovy)
        # gt_height = 512
        # gt_width = 512
        # gt_fovx = 2 * torch.atan(torch.tan(gt_fovy / 2) * self.width / self.height)
        gt_focal_length = 0.5 * gt_height / torch.tan(0.5 * gt_fovy)
        gt_near = 0.01
        # if self.cfg.smpl_type == "smplx":
        #     gt_near = self.smpl_parms["transl"][gt_index][-1].item() - 5.0
        gt_proj_mtx = get_projection_matrix_cxcy(
            gt_fovy,
            gt_width / gt_height,
            gt_near,
            1000.0,
            **(
                {}
                if self.cfg.smpl_type == "smpl"
                else {
                    "cxcy": (gt_cx.item(), gt_cy.item()),
                    "img_wh": (gt_width, gt_height),
                }
            ),
        )

        gt_normal_res = 512
        gt_normal_fx = self.normal_intrinsics[0, 0, 0]
        gt_normal_fy = self.normal_intrinsics[0, 1, 1]
        gt_normal_cx = self.normal_intrinsics[0, 0, 2]
        gt_normal_cy = self.normal_intrinsics[0, 1, 2]
        gt_normal_fovy = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fy))
        gt_normal_fovx = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fx))
        gt_normal_fovy = torch.tensor(gt_normal_fovy, device=c2w.device).unsqueeze(0)
        gt_normal_fovx = torch.tensor(gt_normal_fovx, device=c2w.device).unsqueeze(0)
        gt_normal_cx = torch.tensor(gt_normal_cx, device=c2w.device).unsqueeze(0)
        gt_normal_cy = torch.tensor(gt_normal_cy, device=c2w.device).unsqueeze(0)

        gt_mvp_mtx = get_mvp_matrix(gt_c2w, gt_proj_mtx)
        gt_near = torch.tensor(gt_near, device=c2w.device).unsqueeze(0)
        # breakpoint()
        gt_directions = get_ray_directions(
            H=gt_normal_res,
            W=gt_normal_res,
            focal=(gt_normal_fx, gt_normal_fy),
            principal=(gt_normal_cx, gt_normal_cy),
        )[None, ...]


        gt_rays_o, gt_rays_d = get_rays(
            gt_directions, gt_c2w, keepdim=True, normalize=True
        )

        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################
        # fovy = gt_normal_fovy.repeat_interleave(self.cfg.n_view, dim=0)
        # camera_distances = torch.tensor([self.smpl_parms['transl'][gt_index][..., -1]], device=c2w.device).repeat_interleave(
        #     self.cfg.n_view, dim=0)

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(
                    self.cfg.n_view, dim=0
                )
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        # TODO(Hang Gao @ 05/03): This information is incomplete for SMPLX
        # models, but since we are not actually using it so it's okay for now.

        out_dict = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            # "frames_rays_d": self.frames_rays_d,
            "cam_d": directions,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
            # gt data
            "gt_index": gt_index,
            "gt_rays_o": gt_rays_o,
            "gt_rays_d": gt_rays_d,
            "gt_cam_d": gt_directions,
            "gt_mvp_mtx": gt_mvp_mtx,
            "gt_c2w": gt_c2w,
            "gt_fovx": gt_fovx,
            "gt_fovy": gt_fovy,
            "gt_cx": gt_cx,
            "gt_cy": gt_cy,
            "gt_normal_fovx": gt_normal_fovx,
            "gt_normal_fovy": gt_normal_fovy,
            "gt_normal_cx": gt_normal_cx,
            "gt_normal_cy": gt_normal_cy,
            "gt_normal_res": gt_normal_res,
            "gt_near": gt_near,
            "gt_height": gt_height,
            "gt_width": gt_width,
            "gt_a_smpl": self.smpl_parms,
            "gt_rgb": self.frames_img[gt_index : gt_index + 1],
            "gt_mask": self.frames_mask[gt_index : gt_index + 1],
            # "gt_rgb_crop": self.frames_img_crop[gt_index : gt_index + 1],
            # "gt_mask_crop": self.frames_mask_crop[gt_index : gt_index + 1],
        }
        return out_dict
    
    def collate(self, batch):
        return batch[0]
    
class ZJURandomMultiviewCameraIterableDataset(RandomCameraIterableDataset):
    def __init__(self, cfg: Any, split: str = "train", *args, **kwargs):
        
        super().__init__(cfg)
        self.zoom_range = self.cfg.zoom_range


        img_list = sorted(glob(os.path.join(self.cfg.dataroot, "Camera_B1", "*.jpg")))
        mask_list = sorted(glob(os.path.join(self.cfg.dataroot, "mask", "Camera_B1", "*.png")))
        mask_cihp_list = sorted(glob(os.path.join(self.cfg.dataroot, "mask_cihp", "Camera_B1", "*.png")))
        smplx_param_paths = sorted(glob(os.path.join(self.cfg.dataroot, "params", "*.npy")), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        normal_F_list = sorted(
            glob(os.path.join(self.cfg.dataroot, "normal_F", "*.png"))
        )
        normal_B_list = sorted(
            glob(os.path.join(self.cfg.dataroot, "normal_B", "*.png"))
        )
        threestudio.info(f"Found {len(img_list)} images in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(mask_list)} masks in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(normal_F_list)} normal_F in {self.cfg.dataroot}")
        threestudio.info(f"Found {len(normal_B_list)} normal_B in {self.cfg.dataroot}")
        assert len(img_list) == len(mask_list) == len(normal_F_list), "Number of images and masks should be the same"
        
        frames_img = []
        frames_mask = []
        frames_normal_F = []
        frames_normal_B = []
        frames_normal_mask = []
        
        annots = np.load(os.path.join(self.cfg.dataroot, "annots.npy"), allow_pickle=True).item()
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
        bg = np.array([0, 0, 0], dtype=np.float32)
        
        self.n_frames = len(img_list)
        if self.cfg.index_range[1] == -1:
            self.cfg.index_range = (0, self.n_frames)
        self.cfg.index_range = (
            max(0, self.cfg.index_range[0]),
            min(self.n_frames, self.cfg.index_range[1]),
        )
        
        threestudio.info(f"Using index range {self.cfg.index_range}")
        for i, img_path in tqdm(enumerate(img_list[:self.cfg.index_range[1]])):

            img = cv2.imread(
                img_path, cv2.IMREAD_UNCHANGED
            )  # .astype(np.float32) / 255.0
            if img.shape[-1] == 4:
                mask = img[..., 3]
                img = img[..., :3]
            else:
                img_mask = cv2.imread(mask_list[i], cv2.IMREAD_UNCHANGED)
                img_mask = (img_mask != 0).astype(np.uint8)
                mask_cihp = cv2.imread(mask_cihp_list[i], cv2.IMREAD_UNCHANGED)
                mask_cihp = (mask_cihp != 0).astype(np.uint8)
                mask = (img_mask | mask_cihp).astype(np.uint8)
                mask[mask == 1] = 255
            mask[:, self.cfg.occ_mid-self.cfg.occ_width//2:self.cfg.occ_mid+self.cfg.occ_width//2] *= 0
            K, D = cam_Ks, cam_Ds
            img = cv2.undistort(img, K, D)
            img = img.astype(np.float32) / 255.0
            mask = cv2.undistort(mask, K, D)
            mask = mask.astype(np.float32) / 255.0
            img = img * mask[..., None] + bg * (1.0 - mask[..., None])
            #  img = img * mask[..., None]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_img.append(img)
            frames_mask.append(mask)

            if len(normal_F_list) > 0:
                normal_F = cv2.imread(normal_F_list[i], cv2.IMREAD_UNCHANGED)
                normal_mask = normal_F[..., 3]
                normal_F = normal_F[..., :3]
                normal_F = cv2.cvtColor(normal_F, cv2.COLOR_BGR2RGB)
                normal_B = cv2.imread(normal_B_list[i], cv2.IMREAD_UNCHANGED)
                normal_B = normal_B[..., :3]
                normal_B = cv2.cvtColor(normal_B, cv2.COLOR_BGR2RGB)
                
                frames_normal_F.append(normal_F)
                frames_normal_B.append(normal_B)
                frames_normal_mask.append(normal_mask)
        frames_img = np.stack(frames_img, axis=0)
        frames_mask = np.stack(frames_mask, axis=0)
        self.frames_img = torch.from_numpy(frames_img).float()
        self.frames_mask = torch.from_numpy(frames_mask).float()

        self.frames_img = self.frames_img * self.frames_mask[..., None]

        if len(normal_F_list) > 0:
            frames_normal_F = np.stack(frames_normal_F, axis=0)
            self.frames_normal_F = torch.from_numpy(frames_normal_F).float() / 255.0
            frames_normal_B = np.stack(frames_normal_B, axis=0)
            self.frames_normal_B = torch.from_numpy(frames_normal_B).float() / 255.0
            frames_normal_mask = np.stack(frames_normal_mask, axis=0)
            self.frames_normal_mask = (
                torch.from_numpy(frames_normal_mask).float() / 255.0
            )
        else:
            self.frames_normal_F = []
            self.frames_normal_B = []
            self.frames_normal_mask = []
        gt_width, gt_height = frames_img.shape[2], frames_img.shape[1]
        self.gt_width, self.gt_height = gt_width, gt_height

        

        body_data = torch.load(
            os.path.join(self.cfg.dataroot, "params.pth"), map_location="cpu"
        )
        self.extrinsic = torch.eye(4).to(body_data["w2c"])
        self.intrinsic = torch.from_numpy(cam_Ks).float().to(body_data["w2c"])
        self.normal_intrinsics = body_data["normal_Ks"]

        self.extrinsic[1:3] *= -1
        self.smplx_parms = body_data
        
        frames_img_crop = []
        frames_mask_crop = []
        frames_rays_d = []
        for i, (img, mask) in enumerate(zip(self.frames_img, self.frames_mask)):
            mask_indices = torch.nonzero(mask)
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
                        torch.linspace(bbox[0].item(), bbox[2].item(), crop_wh[0])
                        / self.frames_img.shape[2],
                        torch.linspace(bbox[1].item(), bbox[3].item(), crop_wh[1])
                        / self.frames_img.shape[1],
                        indexing="xy",
                    ),
                    dim=-1,
                )[None]
                * 2.0
                - 1.0
            )
            cropped_img = F.grid_sample(
                img[None].permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                align_corners=False,
            )
            cropped_mask = F.grid_sample(
                mask[None, ..., None].permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                align_corners=False,
            )
            # breakpoint()
            cropped_directions = get_ray_directions(
                H=crop_wh[1],
                W=crop_wh[0],
                focal=(
                    self.normal_intrinsics[i, 0, 0],
                    self.normal_intrinsics[i, 1, 1],
                ),
                principal=(
                    self.normal_intrinsics[i, 0, 2],
                    self.normal_intrinsics[i, 1, 2],
                ),
            )[None, ...]

            c2w = torch.inverse(self.extrinsic).unsqueeze(0).to(cropped_img.device)
            cropped_rays_o, cropped_rays_d = get_rays(
                cropped_directions, c2w, keepdim=True, normalize=True
            )

            cropped_img = cropped_img[0].permute(1, 2, 0)
            cropped_mask = cropped_mask[0, 0]
            frames_img_crop.append(cropped_img)
            frames_mask_crop.append(cropped_mask)
            frames_rays_d.append(cropped_rays_d)
        frames_img_crop = torch.stack(frames_img_crop, dim=0)
        frames_mask_crop = torch.stack(frames_mask_crop, dim=0)
        self.frames_img_crop = frames_img_crop
        self.frames_mask_crop = frames_mask_crop
        self.frames_rays_d = torch.cat(frames_rays_d, dim=0) 

    def load_smpl_param(self, path):
        if self.cfg.smpl_type == "smpl":
            smpl_params = dict(np.load(str(path)))
        else:
            smpl_params = torch.load(str(path))
        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        if self.cfg.smpl_type == "smpl":
            return {
                "betas": smpl_params["betas"].astype(np.float32),
                "body_pose": smpl_params["body_pose"].astype(np.float32),
                "global_orient": smpl_params["global_orient"].astype(np.float32),
                "transl": smpl_params["transl"].astype(np.float32),
            }
        else:
            return {
                "betas": smpl_params["betas"],
                "body_pose": smpl_params["body_pose"],
                "global_orient": smpl_params["global_orient"],
                "transl": smpl_params["transl"],
            }

    def collate(self, batch) -> Dict[str, Any]:
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]

        c2w = torch.eye(4, device="cpu").unsqueeze(0)
        # gt_index = 0
        gt_index = torch.randint(0, self.cfg.index_range[1], (1,)).item()
        gt_c2w = torch.inverse(self.extrinsic).unsqueeze(0).to(c2w.device) 

        gt_fx = self.intrinsic[0, 0]
        gt_fy = self.intrinsic[1, 1]
        gt_cx = self.intrinsic[0, 2]
        gt_cy = self.intrinsic[1, 2]

        gt_fovy = 2 * torch.atan(self.gt_height / (2 * gt_fy))
        gt_fovx = 2 * torch.atan(self.gt_width / (2 * gt_fx))
        gt_fovy = torch.tensor(gt_fovy, device=c2w.device).unsqueeze(0)
        gt_fovx = torch.tensor(gt_fovx, device=c2w.device).unsqueeze(0)
        gt_width = self.gt_width
        gt_height = self.gt_height
        gt_cx = torch.tensor(gt_cx, device=c2w.device).unsqueeze(0)
        gt_cy = torch.tensor(gt_cy, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.tensor(90.0, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.deg2rad(gt_fovy)
        # gt_height = 512
        # gt_width = 512
        # gt_fovx = 2 * torch.atan(torch.tan(gt_fovy / 2) * self.width / self.height)
        gt_focal_length = 0.5 * gt_height / torch.tan(0.5 * gt_fovy)
        gt_near = 0.01

        gt_proj_mtx = get_projection_matrix_cxcy(
            gt_fovy,
            gt_width / gt_height,
            gt_near,
            1000.0,
            **(
                {}
                if self.cfg.smpl_type == "smpl"
                else {
                    "cxcy": (gt_cx.item(), gt_cy.item()),
                    "img_wh": (gt_width, gt_height),
                }
            ),
        )

        gt_normal_res = 512
        gt_normal_fx = self.normal_intrinsics[gt_index, 0, 0]
        gt_normal_fy = self.normal_intrinsics[gt_index, 1, 1]
        gt_normal_cx = self.normal_intrinsics[gt_index, 0, 2]
        gt_normal_cy = self.normal_intrinsics[gt_index, 1, 2]
        gt_normal_fovy = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fy))
        gt_normal_fovx = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fx))
        gt_normal_fovy = torch.tensor(gt_normal_fovy, device=c2w.device).unsqueeze(0)
        gt_normal_fovx = torch.tensor(gt_normal_fovx, device=c2w.device).unsqueeze(0)
        gt_normal_cx = torch.tensor(gt_normal_cx, device=c2w.device).unsqueeze(0)
        gt_normal_cy = torch.tensor(gt_normal_cy, device=c2w.device).unsqueeze(0)

        gt_mvp_mtx = get_mvp_matrix(gt_c2w, gt_proj_mtx)
        gt_near = torch.tensor(gt_near, device=c2w.device).unsqueeze(0)
        # breakpoint()
        gt_directions = get_ray_directions(
            H=gt_normal_res,
            W=gt_normal_res,
            focal=(gt_normal_fx, gt_normal_fy),
            principal=(gt_normal_cx, gt_normal_cy),
        )[None, ...]

        gt_rays_o, gt_rays_d = get_rays(
            gt_directions, gt_c2w, keepdim=True, normalize=True
        )

        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################
        # fovy = gt_normal_fovy.repeat_interleave(self.cfg.n_view, dim=0)
        # camera_distances = torch.tensor([self.smpl_parms['transl'][gt_index][..., -1]], device=c2w.device).repeat_interleave(
        #     self.cfg.n_view, dim=0)

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(
                    self.cfg.n_view, dim=0
                )
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )
        # breakpoint()
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        # TODO(Hang Gao @ 05/03): This information is incomplete for SMPLX
        # models, but since we are not actually using it so it's okay for now.
        smpl_collate = {
            "betas": self.smplx_parms["betas"][0][None],
            "body_pose": self.smplx_parms["body_pose"][gt_index][None],
            "global_orient": self.smplx_parms["global_orient"][gt_index][None],
            "transl": self.smplx_parms["transl"][gt_index][None],
        }
        # breakpoint()
        out_dict = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "frames_rays_d": self.frames_rays_d,
            "cam_d": directions,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
            # gt data
            "gt_index": gt_index,
            "gt_rays_o": gt_rays_o,
            "gt_rays_d": gt_rays_d,
            "gt_cam_d": gt_directions,
            "gt_mvp_mtx": gt_mvp_mtx,
            "gt_c2w": gt_c2w,
            "gt_fovx": gt_fovx,
            "gt_fovy": gt_fovy,
            "gt_cx": gt_cx,
            "gt_cy": gt_cy,
            "gt_normal_fovx": gt_normal_fovx,
            "gt_normal_fovy": gt_normal_fovy,
            "gt_normal_cx": gt_normal_cx,
            "gt_normal_cy": gt_normal_cy,
            "gt_normal_res": gt_normal_res,
            "gt_near": gt_near,
            "gt_height": gt_height,
            "gt_width": gt_width,
            "gt_a_smpl": smpl_collate,
            "gt_rgb": self.frames_img[gt_index : gt_index + 1],
            "gt_mask": self.frames_mask[gt_index : gt_index + 1],
            "gt_rgb_crop": self.frames_img_crop[gt_index : gt_index + 1],
            "gt_mask_crop": self.frames_mask_crop[gt_index : gt_index + 1],
        }
        if len(self.frames_normal_F) > 0:
            out_dict["gt_normal_F"] = self.frames_normal_F[gt_index : gt_index + 1]
            out_dict["gt_normal_B"] = self.frames_normal_B[gt_index : gt_index + 1]
            out_dict["gt_normal_mask"] = self.frames_normal_mask[
                gt_index : gt_index + 1
            ]
        return out_dict

from copy import deepcopy
class ZJUValDataset(Dataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range
        self.zoom_range = self.cfg.zoom_range
        
        import smplx
        smpl_layer = smplx.SMPLLayer(
            os.path.join("custom/threestudio-soar/utils/smpl_files", "smpl"), kid_template_path="custom/threestudio-soar/utils/smpl_files/smpl/SMPL_NEUTRAL.pkl")
        
        smplx_param_paths = sorted(glob(os.path.join(self.cfg.dataroot, "new_params", "*.npy")), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        annots = np.load(os.path.join(self.cfg.dataroot, "annots.npy"), allow_pickle=True).item()
        bg = np.array([0, 0, 0], dtype=np.float32)
        frames_img = []
        frames_mask = []
        smpl_index = []
        smplx_params = []
        smpl_arranged_params = []
        intrinsics = []
        for smplx_param_path in smplx_param_paths:
            smplx_param = np.load(smplx_param_path, allow_pickle=True).item()
            smplx_params.append(smplx_param)
        for select_view in range(2, 24):
            img_list = sorted(glob(os.path.join(self.cfg.dataroot, f"Camera_B{select_view}", "*.jpg")))
            mask_list = sorted(glob(os.path.join(self.cfg.dataroot, "mask", f"Camera_B{select_view}", "*.png")))
            mask_cihp_list = sorted(glob(os.path.join(self.cfg.dataroot, "mask_cihp", f"Camera_B{select_view}", "*.png")))
            normal_F_list = sorted(
                glob(os.path.join(self.cfg.dataroot, "normal_F", "*.png"))
            )
            normal_B_list = sorted(
                glob(os.path.join(self.cfg.dataroot, "normal_B", "*.png"))
            )
            threestudio.info(f"Found {len(img_list)} images in {self.cfg.dataroot}")
            threestudio.info(f"Found {len(mask_list)} masks in {self.cfg.dataroot}")
            threestudio.info(f"Found {len(normal_F_list)} normal_F in {self.cfg.dataroot}")
            threestudio.info(f"Found {len(normal_B_list)} normal_B in {self.cfg.dataroot}")
            assert len(img_list) == len(mask_list) == len(normal_F_list), "Number of images and masks should be the same"
            
            cams = annots['cams']
            cam_Ks = np.array(cams['K'])[select_view - 1].astype('float32')
            cam_Rs = np.array(cams['R'])[select_view - 1].astype('float32')
            cam_Ts = np.array(cams['T'])[select_view - 1].astype('float32') / 1000.
            cam_Ds = np.array(cams['D'])[select_view - 1].astype('float32')
            cam_T = cam_Ts[:3, 0]
            E = np.eye(4)
            E[:3, :3] = cam_Rs
            E[:3, 3]= cam_T
            
            self.n_frames = len(img_list)
            if self.cfg.index_range[1] == -1:
                self.cfg.index_range = (0, self.n_frames)
            self.cfg.index_range = (
                max(0, self.cfg.index_range[0]),
                min(self.n_frames, self.cfg.index_range[1]),
            )
            
            threestudio.info(f"Using index range {self.cfg.index_range}")
            for i in tqdm(range(0, self.cfg.index_range[1], 30)):
                img_path = img_list[i]
                img = cv2.imread(
                    img_path, cv2.IMREAD_UNCHANGED
                )  # .astype(np.float32) / 255.0
                if img.shape[-1] == 4:
                    mask = img[..., 3]
                    img = img[..., :3]
                else:
                    img_mask = cv2.imread(mask_list[i], cv2.IMREAD_UNCHANGED)
                    img_mask = (img_mask != 0).astype(np.uint8)
                    mask_cihp = cv2.imread(mask_cihp_list[i], cv2.IMREAD_UNCHANGED)
                    mask_cihp = (mask_cihp != 0).astype(np.uint8)
                    mask = (img_mask | mask_cihp).astype(np.uint8)
                    mask[mask == 1] = 255
                # mask[:, self.cfg.occ_mid-self.cfg.occ_width//2:self.cfg.occ_mid+self.cfg.occ_width//2] *= 0
                K, D = cam_Ks, cam_Ds
                img = cv2.undistort(img, K, D)
                img = img.astype(np.float32) / 255.0
                mask = cv2.undistort(mask, K, D)
                mask = mask.astype(np.float32) / 255.0
                img = img * mask[..., None] + bg * (1.0 - mask[..., None])
                #  img = img * mask[..., None]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames_img.append(img)
                frames_mask.append(mask)
                
                params = {
                    "betas": [], "body_pose": [], "global_orient": [], "transl": []
                }
                smpl_arranged = trans_smpl_rt(deepcopy(smplx_params[i]), smpl_layer, E)
                
                for k, v in smpl_arranged.items():
                    params[k].append(v)
                for k, v in params.items():
                    params[k] = torch.from_numpy(np.concatenate(v)).float().to('cpu')
                smpl_arranged_params.append(params)
                smpl_index.append(i)
                intrinsics.append(cam_Ks)

        frames_img = np.stack(frames_img, axis=0)
        frames_mask = np.stack(frames_mask, axis=0)
        self.frames_img = torch.from_numpy(frames_img).float()
        self.frames_mask = torch.from_numpy(frames_mask).float()

        self.frames_img = self.frames_img * self.frames_mask[..., None]

        gt_width, gt_height = frames_img.shape[2], frames_img.shape[1]
        self.gt_width, self.gt_height = gt_width, gt_height
        self.n_frames = len(self.frames_img)

        body_data = torch.load(
            os.path.join(self.cfg.dataroot, "params.pth"), map_location="cpu"
        )
        self.smplx_parms = body_data
        self.extrinsic = torch.eye(4).to(body_data["w2c"])
        intrinsics = np.stack(intrinsics, axis=0)
        self.intrinsics = torch.from_numpy(intrinsics).float().to(body_data["w2c"])
        self.normal_intrinsics = body_data["normal_Ks"]
        self.smpl_index = smpl_index
        self.smpl_arranged_params = smpl_arranged_params

        self.extrinsic[1:3] *= -1 
        
    
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, index):
        assert (
            self.batch_size % self.cfg.n_view == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]

        c2w = torch.eye(4, device="cpu").unsqueeze(0)
        gt_index = index
        smpl_index = self.smpl_index[index]

        gt_c2w = torch.inverse(self.extrinsic).unsqueeze(0).to(c2w.device)
        # gt_c2w[:3, 1:3] *= -1
        gt_fx = self.intrinsics[gt_index, 0, 0]
        gt_fy = self.intrinsics[gt_index, 1, 1]
        gt_cx = self.intrinsics[gt_index, 0, 2]
        gt_cy = self.intrinsics[gt_index, 1, 2]

        gt_fovy = 2 * torch.atan(self.gt_height / (2 * gt_fy))
        gt_fovx = 2 * torch.atan(self.gt_width / (2 * gt_fx))
        gt_fovy = torch.tensor(gt_fovy, device=c2w.device).unsqueeze(0)
        gt_fovx = torch.tensor(gt_fovx, device=c2w.device).unsqueeze(0)
        gt_width = self.gt_width
        gt_height = self.gt_height
        gt_cx = torch.tensor(gt_cx, device=c2w.device).unsqueeze(0)
        gt_cy = torch.tensor(gt_cy, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.tensor(90.0, device=c2w.device).unsqueeze(0)
        # gt_fovy = torch.deg2rad(gt_fovy)
        # gt_height = 512
        # gt_width = 512
        # gt_fovx = 2 * torch.atan(torch.tan(gt_fovy / 2) * self.width / self.height)
        gt_focal_length = 0.5 * gt_height / torch.tan(0.5 * gt_fovy)
        gt_near = 0.01
        # if self.cfg.smpl_type == "smplx":
        #     gt_near = self.smpl_parms["transl"][gt_index][-1].item() - 5.0
        gt_proj_mtx = get_projection_matrix_cxcy(
            gt_fovy,
            gt_width / gt_height,
            gt_near,
            1000.0,
            **(
                {}
                if self.cfg.smpl_type == "smpl"
                else {
                    "cxcy": (gt_cx.item(), gt_cy.item()),
                    "img_wh": (gt_width, gt_height),
                }
            ),
        )

        gt_normal_res = 512
        gt_normal_fx = self.normal_intrinsics[0, 0, 0]
        gt_normal_fy = self.normal_intrinsics[0, 1, 1]
        gt_normal_cx = self.normal_intrinsics[0, 0, 2]
        gt_normal_cy = self.normal_intrinsics[0, 1, 2]
        gt_normal_fovy = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fy))
        gt_normal_fovx = 2 * torch.atan(gt_normal_res / (2 * gt_normal_fx))
        gt_normal_fovy = torch.tensor(gt_normal_fovy, device=c2w.device).unsqueeze(0)
        gt_normal_fovx = torch.tensor(gt_normal_fovx, device=c2w.device).unsqueeze(0)
        gt_normal_cx = torch.tensor(gt_normal_cx, device=c2w.device).unsqueeze(0)
        gt_normal_cy = torch.tensor(gt_normal_cy, device=c2w.device).unsqueeze(0)

        gt_mvp_mtx = get_mvp_matrix(gt_c2w, gt_proj_mtx)
        gt_near = torch.tensor(gt_near, device=c2w.device).unsqueeze(0)
        # breakpoint()
        gt_directions = get_ray_directions(
            H=gt_normal_res,
            W=gt_normal_res,
            focal=(gt_normal_fx, gt_normal_fy),
            principal=(gt_normal_cx, gt_normal_cy),
        )[None, ...]


        gt_rays_o, gt_rays_d = get_rays(
            gt_directions, gt_c2w, keepdim=True, normalize=True
        )

        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.cfg.n_view).reshape(1, -1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################
        # fovy = gt_normal_fovy.repeat_interleave(self.cfg.n_view, dim=0)
        # camera_distances = torch.tensor([self.smpl_parms['transl'][gt_index][..., -1]], device=c2w.device).repeat_interleave(
        #     self.cfg.n_view, dim=0)

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(
                    self.cfg.n_view, dim=0
                )
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.cfg.n_view, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize
        )
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        # TODO(Hang Gao @ 05/03): This information is incomplete for SMPLX
        # models, but since we are not actually using it so it's okay for now.
        smpl_collate = {
            "betas": self.smplx_parms["betas"][0][None],
            "body_pose": self.smplx_parms["body_pose"][smpl_index][None],
            "global_orient": self.smplx_parms["global_orient"][smpl_index][None],
            "transl": self.smplx_parms["transl"][smpl_index][None],
        }
        out_dict = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            # "frames_rays_d": self.frames_rays_d,
            "cam_d": directions,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy,
            # gt data
            "gt_index": gt_index,
            "gt_rays_o": gt_rays_o,
            "gt_rays_d": gt_rays_d,
            "gt_cam_d": gt_directions,
            "gt_mvp_mtx": gt_mvp_mtx,
            "gt_c2w": gt_c2w,
            "gt_fovx": gt_fovx,
            "gt_fovy": gt_fovy,
            "gt_cx": gt_cx,
            "gt_cy": gt_cy,
            "gt_normal_fovx": gt_normal_fovx,
            "gt_normal_fovy": gt_normal_fovy,
            "gt_normal_cx": gt_normal_cx,
            "gt_normal_cy": gt_normal_cy,
            "gt_normal_res": gt_normal_res,
            "gt_near": gt_near,
            "gt_height": gt_height,
            "gt_width": gt_width,
            "gt_a_smpl": self.smpl_arranged_params[gt_index],
            "gt_rgb": self.frames_img[gt_index : gt_index + 1],
            "gt_mask": self.frames_mask[gt_index : gt_index + 1],
            # "gt_rgb_crop": self.frames_img_crop[gt_index : gt_index + 1],
            # "gt_mask_crop": self.frames_mask_crop[gt_index : gt_index + 1],
        }
        return out_dict
    
    def collate(self, batch):
        return batch[0]
        


@register("mvdream-random-multiview-camera-datamodule")
class RandomMultiviewCameraDataModule(pl.LightningDataModule):
    cfg: RandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = FSRandomMultiviewCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = FSValDataset(self.cfg) #RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def get_view_direction(thetas, phis, overhead, front):
    #                   phis: [B,];          thetas: [B,]
    # front = 0             [-front/2, front/2)
    # side (cam left) = 1   [front/2, 180-front/2)
    # back = 2              [180-front/2, 180+front/2)
    # side (cam right) = 3  [180+front/2, 360-front/2)
    # top = 4               [0, overhead]
    # bottom = 5            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def circle_poses(device, radius=torch.tensor([3.2]), theta=torch.tensor([90]), phi=torch.tensor([0]), return_dirs=False, angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    # centers = torch.stack([
    #     radius * torch.cos(theta) * torch.cos(phi),
    #     radius * torch.cos(theta) * torch.sin(phi),
    #     radius * torch.sin(theta),
    # ], dim=-1) # [B, 3]
    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(theta, phi, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs

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
