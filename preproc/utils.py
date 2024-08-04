import json
import math
import os.path as osp
from glob import glob

import cv2
import imageio.v3 as iio
import matplotlib.colors
import numpy as np
import torch
import torch.nn.functional as F
from roma import rotmat_to_rotvec, rotvec_to_rotmat
from torch import nn
from tqdm import tqdm

from soar.rendering import get_proj_mat, render_mesh


def load_keypoints(kp_dir: str):
    kp_paths = sorted(glob(osp.join(kp_dir, "*.json")))
    keypoints = []
    for path in kp_paths:
        with open(path) as f:
            keypoint_data = json.load(f)
        keypoints.append(
            np.array(
                keypoint_data["people"][0]["pose_keypoints_2d"]
                + keypoint_data["people"][0]["hand_left_keypoints_2d"]
                + keypoint_data["people"][0]["hand_right_keypoints_2d"]
                + keypoint_data["people"][0]["face_keypoints_2d"],
                dtype=np.float32,
            ).reshape(-1, 3)
        )
    keypoints = np.stack(keypoints, axis=0)
    return keypoints


def load_smplerx(smplerx_result_dir: str, device: str):
    N = len(sorted(glob(osp.join(smplerx_result_dir, "*_0.npz"))))
    pose_data = [
        np.load(osp.join(smplerx_result_dir, f"{i:05d}_0.npz")) for i in range(N)
    ]
    # (N, 10).
    betas = torch.cat(
        [torch.from_numpy(pose_data[i]["betas"].astype(np.float32)) for i in range(N)],
        dim=0,
    ).to(device)
    # (N, 3).
    global_orient = torch.cat(
        [
            torch.from_numpy(pose_data[i]["global_orient"].astype(np.float32))
            for i in range(N)
        ],
        dim=0,
    ).to(device)
    # (N, 63).
    body_pose = torch.stack(
        [
            torch.from_numpy(pose_data[i]["body_pose"].reshape(-1).astype(np.float32))
            for i in range(N)
        ],
        dim=0,
    ).to(device)
    # (N, 3).
    transl = torch.cat(
        [torch.from_numpy(pose_data[i]["transl"].astype(np.float32)) for i in range(N)],
        dim=0,
    ).to(device)
    # (N, 45).
    left_hand_pose = torch.stack(
        [
            torch.from_numpy(
                pose_data[i]["left_hand_pose"].reshape(-1).astype(np.float32)
            )
            for i in range(N)
        ],
        dim=0,
    ).to(device)
    # (N, 45).
    right_hand_pose = torch.stack(
        [
            torch.from_numpy(
                pose_data[i]["right_hand_pose"].reshape(-1).astype(np.float32)
            )
            for i in range(N)
        ],
        dim=0,
    ).to(device)
    # (N, 3).
    jaw_pose = torch.cat(
        [
            torch.from_numpy(pose_data[i]["jaw_pose"].astype(np.float32))
            for i in range(N)
        ],
        dim=0,
    ).to(device)
    # (N, 3).
    leye_pose = torch.cat(
        [
            torch.from_numpy(pose_data[i]["leye_pose"].astype(np.float32))
            for i in range(N)
        ],
        dim=0,
    ).to(device)
    # (N, 3).
    reye_pose = torch.cat(
        [
            torch.from_numpy(pose_data[i]["reye_pose"].astype(np.float32))
            for i in range(N)
        ],
        dim=0,
    ).to(device)
    # (N, 10).
    expression = torch.cat(
        [
            torch.from_numpy(pose_data[i]["expression"].astype(np.float32))
            for i in range(N)
        ],
        dim=0,
    ).to(device)
    return (
        betas,
        body_pose,
        global_orient,
        left_hand_pose,
        right_hand_pose,
        jaw_pose,
        leye_pose,
        reye_pose,
        expression,
        transl,
    )


def gmof(x, sigma):
    """Geman-McClure error function."""
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def compute_jitter(x):
    """Compute jitter for the input tensor"""
    if x.shape[0] < 3:
        return torch.zeros(1, device=x.device)
    return torch.linalg.norm(x[2:] + x[:-2] - 2 * x[1:-1], dim=-1)


def compute_smooth_loss(x):
    R = rotation_6d_to_matrix(x)
    R12 = R[1:] @ R[:-1].transpose(-2, -1)
    return (rotmat_to_rotvec(R12) ** 2).sum(-1).mean()


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def get_target_scales(target_kps):
    target_scales = []
    for frame_kps in target_kps:
        valid_kps = frame_kps[frame_kps[..., -1] > 0.3]
        min_x, min_y = valid_kps[:, :2].min(dim=0).values
        max_x, max_y = valid_kps[:, :2].max(dim=0).values
        target_scales.append(max(max_x - min_x, max_y - min_y))
    target_scales = torch.stack(target_scales, dim=0)
    return target_scales


def prepare_smplx_to_openpose137():
    kp_mask = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
        ],
        dtype=torch.float32,
    )
    src_inds = [
        55,
        12,
        17,
        19,
        21,
        16,
        18,
        20,
        0,
        2,
        5,
        8,
        1,
        4,
        7,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        37,
        38,
        39,
        66,
        25,
        26,
        27,
        67,
        28,
        29,
        30,
        68,
        34,
        35,
        36,
        69,
        31,
        32,
        33,
        70,
        52,
        53,
        54,
        71,
        40,
        41,
        42,
        72,
        43,
        44,
        45,
        73,
        49,
        50,
        51,
        74,
        46,
        47,
        48,
        75,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
    ]
    dst_inds = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
    ]

    def convert_kps(kps):
        assert kps.shape[1] == 144
        new_kps = kps.new_zeros((kps.shape[0], 137, 3))
        new_kps[:, dst_inds] = kps[:, src_inds]
        new_kps[:, 8] = 0.5 * (new_kps[:, 9] + new_kps[:, 12])
        new_kps[:, [9, 12], :2] = (
            new_kps[:, [9, 12], :2]
            + 0.25 * (new_kps[:, [9, 12], :2] - new_kps[:, [12, 9], :2])
            + 0.5
            * (
                new_kps[:, [8], :2]
                - 0.5 * (new_kps[:, [9, 12], :2] + new_kps[:, [12, 9], :2])
            )
        )
        return new_kps

    return convert_kps, kp_mask


class SMPLify(nn.Module):
    def __init__(
        self,
        body_model,
        lr=1.0,
        max_iters=20,
        body_steps=20,
        hand_steps=40,
        kp_weight=100.0,
        preserve_weight=60.0,
        smooth_weight=10000.0,
        sigma=100,
        visual_dir=None,
        debug=False,
    ):
        super().__init__()
        self.body_model = body_model
        self.lr = lr
        self.max_iters = max_iters
        self.body_steps = body_steps
        self.hand_steps = hand_steps
        self.kp_weight = kp_weight
        self.preserve_weight = preserve_weight
        self.smooth_weight = smooth_weight
        self.sigma = sigma
        self.visual_dir = visual_dir
        self.debug = debug

        self.convert_kps, kp_mask = prepare_smplx_to_openpose137()
        self.register_buffer("kp_mask", kp_mask)

        self.loss_dict = {}

    def forward(
        self,
        body_output,
        body_params,
        Ks,
        w2c,
        img_wh,
        target_kps,
        target_scales,
        init_params,
        ignore_hands=False,
    ):
        # Keypoint loss.
        pred_joints = self.convert_kps(body_output.joints)
        pred_joints_c = torch.einsum(
            "ij,nkj->nki", w2c[:3], F.pad(pred_joints, (0, 1), value=1.0)  # type: ignore
        )
        pred_kps = torch.einsum("nij,nkj->nki", Ks, pred_joints_c)
        pred_kps = pred_kps[..., :2] / pred_kps[..., 2:].clamp(min=1e-5)
        target_kps = torch.cat(
            [
                target_kps[..., :-1] * target_kps.new_tensor(img_wh),
                target_kps[..., -1:],
            ],
            dim=-1,
        )
        kp_confs = target_kps[..., -1:] * self.kp_mask[:, None]
        if ignore_hands:
            kp_confs[:, 25:-70] = 0.0
        kp_loss = gmof(
            (pred_kps - target_kps[..., :-1]) / target_scales[:, None, None] * 200.0,
            self.sigma,
        )
        kp_loss = (kp_loss * kp_confs).mean()

        # Preverse loss.
        preserve_loss = 0.0
        for k in body_params.keys():
            preserve_loss += torch.linalg.norm(
                body_params[k] - init_params[k], dim=-1
            ).mean()

        # Smooth loss.
        smooth_loss = 0.0
        for k in [
            "body_pose",
            "global_orient",
            "left_hand_pose",
            "right_hand_pose",
        ]:
            smooth_loss += compute_smooth_loss(body_params[k])

        loss = {
            "kp": self.kp_weight * kp_loss,
            "preserve": self.preserve_weight * preserve_loss,
            "smooth": self.smooth_weight * smooth_loss,
        }

        return loss

    @torch.inference_mode()
    def visualize_params(self, params, Ks, w2c, img_wh, target_kps, imgs=None):
        body_output = self.body_model(
            **{
                k: (
                    rotmat_to_rotvec(rotation_6d_to_matrix(v)).reshape(
                        *v.shape[:-1], -1
                    )
                    if k
                    in [
                        "body_pose",
                        "global_orient",
                        "left_hand_pose",
                        "right_hand_pose",
                    ]
                    else v
                )
                for k, v in params.items()
            },
        )
        joints = body_output.joints
        dst_joints = self.convert_kps(joints)
        kp_mask = self.kp_mask.to(torch.bool)  # type: ignore
        dst_joints_c = torch.einsum("ij,nkj->nki", w2c[:3], F.pad(dst_joints, (0, 1), value=1.0))  # type: ignore
        dst_kps = torch.einsum("nij,nkj->nki", Ks, dst_joints_c)
        dst_kps = dst_kps[..., :2] / dst_kps[..., 2:]
        dst_kps = dst_kps / dst_kps.new_tensor(img_wh)
        dst_kps[:, ~kp_mask] = -1.0
        dst_kps = torch.cat(
            [
                dst_kps,
                kp_mask[None, :, None].repeat_interleave(joints.shape[0], dim=0),
            ],
            dim=-1,
        )
        dst_kps = dst_kps.cpu().numpy()
        target_kps = target_kps.cpu().numpy()
        faces = torch.from_numpy(self.body_model.faces.astype(np.int64)).to(Ks.device)
        video = []
        Rs = rotvec_to_rotmat(
            -w2c[:3, 1]
            * torch.linspace(0.0, 2.0 * np.pi, 4, device=Ks.device)[:3, None]
        )
        for i, frame_kps in enumerate(tqdm(dst_kps, leave=False)):
            kp_visual = draw_pose(
                target_kps[i : i + 1], img_wh[1], img_wh[0], (0, 255, 0)
            )
            kp_visual = draw_pose(
                frame_kps[None], img_wh[1], img_wh[0], (255, 0, 0), kp_visual
            )
            if imgs is not None:
                kp_visual = cv2.addWeighted(kp_visual, 0.6, imgs[i], 0.4, 0)
            normal_visuals = []
            current_K = Ks[i].clone()
            current_img_wh = img_wh
            current_scale = 1.0
            if max(current_img_wh) > 2048:
                current_scale = 2048.0 / max(current_img_wh)
                current_K[:2] = current_K[:2] * current_scale
                current_img_wh = tuple(round(x * current_scale) for x in current_img_wh)
            for R in Rs:
                verts = (
                    torch.einsum(
                        "ij,nj->ni",
                        R,
                        body_output.vertices[i] - body_output.joints[i, :1],
                    )
                    + body_output.joints[i, :1]
                )
                normal_visual = (
                    render_mesh(
                        verts,
                        faces,
                        w2c,
                        get_proj_mat(
                            current_K,
                            current_img_wh,
                            znear=params["transl"][i, 2].item() - 0.1,
                        ),
                        current_img_wh,
                    )["normal"]
                    .cpu()
                    .numpy()
                    * 255.0
                ).astype(np.uint8)
                if current_scale != 1.0:
                    normal_visual = cv2.resize(
                        normal_visual, img_wh, interpolation=cv2.INTER_LINEAR
                    )
                normal_visuals.append(normal_visual)
            if imgs is not None:
                normal_overlaid_visual = cv2.addWeighted(
                    normal_visuals[0], 0.6, imgs[i], 0.4, 0
                )
            else:
                normal_overlaid_visual = normal_visuals[0]
            visual = np.concatenate(
                [kp_visual, normal_overlaid_visual, *normal_visuals], axis=1
            )
            if max(img_wh) > 2048:
                visual = cv2.resize(
                    visual, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
                )
            video.append(visual)
        return video

    def create_closure(
        self,
        optimizer,
        body_params,
        Ks,
        w2c,
        img_wh,
        target_kps,
        target_scales,
        init_params,
        ignore_hands=False,
    ):
        def closure():
            optimizer.zero_grad()
            body_output = self.body_model(
                **{
                    k: (
                        rotmat_to_rotvec(rotation_6d_to_matrix(v)).reshape(
                            *v.shape[:-1], -1
                        )
                        if k
                        in [
                            "body_pose",
                            "global_orient",
                            "left_hand_pose",
                            "right_hand_pose",
                        ]
                        else (
                            v
                            if k != "betas"
                            else v.mean(0, keepdim=True).repeat_interleave(
                                Ks.shape[0], dim=0
                            )
                        )
                    )
                    for k, v in body_params.items()
                },
            )
            loss_dict = self(
                body_output,
                body_params,
                Ks,
                w2c,
                img_wh,
                target_kps,
                target_scales,
                init_params,
                ignore_hands=ignore_hands,
            )
            self.loss_dict = loss_dict
            loss = sum(loss_dict.values())
            loss.backward()  # type: ignore
            return loss

        return closure

    def fit(self, init_params, Ks, w2c, img_wh, target_kps, imgs=None):
        init_params = {
            k: (
                matrix_to_rotation_6d(rotvec_to_rotmat(v.reshape(*v.shape[:-1], -1, 3)))
                if k
                in [
                    "body_pose",
                    "global_orient",
                    "left_hand_pose",
                    "right_hand_pose",
                ]
                else (v if k != "betas" else v.mean(0, keepdim=True))
            )
            for k, v in init_params.items()
        }

        params = {
            k: v.clone()
            .detach()
            .requires_grad_(
                k
                in [
                    "betas",
                    "body_pose",
                    "global_orient",
                    "left_hand_pose",
                    "right_hand_pose",
                    "transl",
                ]
            )
            for k, v in init_params.items()
        }

        target_scales = get_target_scales(
            torch.cat(
                [
                    target_kps[..., :-1] * target_kps.new_tensor(img_wh),
                    target_kps[..., -1:],
                ],
                dim=-1,
            )
        )

        # Stage 1.
        optimizer = torch.optim.LBFGS(  # type: ignore
            [
                params["betas"],
                params["body_pose"],
                params["global_orient"],
                params["transl"],
            ],
            lr=self.lr,
            max_iter=self.max_iters,
            line_search_fn="strong_wolfe",
        )
        closure = self.create_closure(
            optimizer,
            params,
            Ks,
            w2c,
            img_wh,
            target_kps,
            target_scales,
            init_params,
            ignore_hands=True,
        )
        if self.visual_dir is not None:
            video = self.visualize_params(params, Ks, w2c, img_wh, target_kps, imgs)
            iio.imwrite(f"{self.visual_dir}/1_000.mp4", video, fps=10)
        for i in (pbar := tqdm(range(self.body_steps))):
            optimizer.zero_grad()
            _ = optimizer.step(closure)
            msg = " ".join([f"{k}: {v.item():.1f}" for k, v in self.loss_dict.items()])
            if self.visual_dir is not None and (
                (self.debug and (i + 1) % 5 == 0)
                or (not self.debug and i == self.body_steps - 1)
            ):
                video = self.visualize_params(params, Ks, w2c, img_wh, target_kps, imgs)
                iio.imwrite(f"{self.visual_dir}/1_{i:03d}.mp4", video, fps=10)
            pbar.set_postfix_str(msg)

        # Stage 2.
        optimizer = torch.optim.LBFGS(  # type: ignore
            [
                params["betas"],
                params["body_pose"],
                params["global_orient"],
                params["left_hand_pose"],
                params["right_hand_pose"],
                params["transl"],
            ],
            lr=self.lr,
            max_iter=self.max_iters,
            line_search_fn="strong_wolfe",
        )
        closure = self.create_closure(
            optimizer,
            params,
            Ks,
            w2c,
            img_wh,
            target_kps,
            target_scales,
            init_params,
            ignore_hands=False,
        )
        for i in (pbar := tqdm(range(self.hand_steps))):
            optimizer.zero_grad()
            _ = optimizer.step(closure)
            msg = " ".join([f"{k}: {v.item():.1f}" for k, v in self.loss_dict.items()])
            if self.visual_dir is not None and (
                (self.debug and (i + 1) % 5 == 0)
                or (not self.debug and i == self.hand_steps - 1)
            ):
                video = self.visualize_params(params, Ks, w2c, img_wh, target_kps, imgs)
                iio.imwrite(f"{self.visual_dir}/2_{i:03d}.mp4", video, fps=10)
            pbar.set_postfix_str(msg)

        params = {
            k: (
                rotmat_to_rotvec(rotation_6d_to_matrix(v)).reshape(*v.shape[:-1], -1)
                if k
                in [
                    "body_pose",
                    "global_orient",
                    "left_hand_pose",
                    "right_hand_pose",
                ]
                else (v if k != "betas" else v.mean(0, keepdim=True))
            )
            for k, v in params.items()
        }
        params["global_orient"] = params["global_orient"][:, 0]
        return params


EPS = 0.01


def draw_pose(pose, H, W, color=None, canvas=None):
    if canvas is None:
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    else:
        canvas = canvas.copy()

    if pose.shape[1] == 135:
        body = pose[:, :18, :2].copy()
        foot = pose[:, 18:24, :2].copy()
        faces = pose[:, 24:92, :2].copy()
        hands = pose[:, 92:, :2].copy()
        body_score = pose[:, :18, -1].copy()
        for i in range(len(body_score)):
            for j in range(len(body_score[i])):
                if body_score[i][j] > 0.3:
                    body_score[i][j] = int(18 * i + j)
                else:
                    body_score[i][j] = -1

        canvas = draw_bodypose(canvas, body[0], body_score, color)
        canvas = draw_foot(canvas, foot, color)
        canvas = draw_handpose(canvas, hands, color)
        canvas = draw_facepose(canvas, faces, color)
    else:
        canvas = draw_facepose(canvas, pose[..., :2].copy(), color)

    return canvas


def draw_bodypose(canvas, candidate, subset, color=None):
    H, W, _ = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i] if color is None else color)  # type: ignore

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(
                canvas,
                (int(x), int(y)),
                4,
                colors[i] if color is None else color,
                thickness=-1,
            )

    return canvas


def draw_handpose(canvas, all_hand_peaks, color=None):
    H, W, _ = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > EPS and y1 > EPS and x2 > EPS and y2 > EPS:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    (
                        matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])  # type: ignore
                        * 255
                        if color is None
                        else color
                    ),
                    thickness=2,
                )

        for keyponit in peaks:
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > EPS and y > EPS:
                cv2.circle(
                    canvas,
                    (x, y),
                    4,
                    (0, 0, 255) if color is None else color,
                    thickness=-1,
                )
    return canvas


def draw_foot(canvas, all_lmks, color=None):
    H, W, _ = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > EPS and y > EPS:
                cv2.circle(
                    canvas,
                    (x, y),
                    3,
                    (255, 255, 255) if color is None else color,
                    thickness=-1,
                )
    return canvas


def draw_facepose(canvas, all_lmks, color=None):
    H, W, _ = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > EPS and y > EPS:
                cv2.circle(
                    canvas,
                    (x, y),
                    3,
                    (255, 255, 255) if color is None else color,
                    thickness=-1,
                )
    return canvas
