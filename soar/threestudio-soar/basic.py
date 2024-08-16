#!/usr/bin/env python3
#
# File   : basic.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 12/26/2023
#
# Distributed under terms of the MIT license.

import time

import cv2
import numpy as np
import torch
import tyro
from ghuman import GaussianTemplateModel
from nerfview import CameraState, LightState, ViewerServer, ViewerStats, with_view_lock
from renderer import render_cam_pcl

render_mode_map = {}
gstemplate = None


def render_fn(
    camera_state: CameraState, img_wh: tuple[int, int], light_state: LightState
):
    fov = camera_state.fov
    c2w = camera_state.c2w
    mode = camera_state.mode
    extras = camera_state.extras
    W, H = img_wh

    focal_length = H / 2.0 / np.tan(fov / 2.0)
    K = np.array(
        [
            [focal_length, 0.0, W / 2.0],
            [0.0, focal_length, H / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # camera_dirs = np.einsum(
    #     "ij,hwj->hwi",
    #     np.linalg.inv(K),
    #     np.pad(
    #         np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), -1) + 0.5,
    #         ((0, 0), (0, 0), (0, 1)),
    #         constant_values=1.0,
    #     ),
    # )
    # dirs = np.einsum("ij,hwj->hwi", c2w[:3, :3], camera_dirs)
    # dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # img = ((dirs + 1.0) / 2.0 * 255.0).astype(np.uint8)
    # print(dirs.shape)
    # xyz = torch.rand((100, 3)).float().cuda()
    # frame = torch.eye(3).unsqueeze(0).repeat(100, 1, 1).float().cuda()
    # scale = torch.rand((100, 3)).float().cuda() * 1e-2
    # opacity = torch.ones((100, 1)).float().cuda()
    # color_map = torch.ones((100, 3)).float().cuda()
    print("camera state:", c2w, "extras:", extras)
    if extras is not None:
        pose_frame = extras["pose_frame"]
        pose = gstemplate.pose_list[[pose_frame]]
        trans = gstemplate.global_trans_list[[pose_frame]]
        trans = torch.tensor([[0.0, 0.0, 2.0]]).float().cuda()
        print("pose:", pose, "trans:", trans)
        xyz, frame, scale, opacity, color_map, _ = gstemplate(
            pose,
            trans,
            additional_dict={"t": pose_frame},  # use time_index from training set
        )
    if light_state is not None:
        direction = torch.from_numpy(light_state.direction).float().cuda()
        color = torch.from_numpy(light_state.color).float().cuda()
        print("light state:", direction, color)
        img = (
            render_cam_pcl(
                xyz[0],
                frame[0],
                scale[0],
                opacity[0],
                color_map[0],
                c2w,
                H,
                W,
                K,
                bg_color=[0.0, 0.0, 0.0],
                lighting=True,
                light_dir=direction,
                diffuse_light_color=color,
                verbose=True,
            )[render_mode_map[mode]]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
    else:
        print("no light state")
        img = (
            render_cam_pcl(
                xyz[0],
                frame[0],
                scale[0],
                opacity[0],
                color_map[0],
                c2w,
                H,
                W,
                K,
                bg_color=[0.0, 0.0, 0.0],
                lighting=False,
            )[render_mode_map[mode]]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
    # img = np.flip(img, 0)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    print("rendered image shape:", img.shape, frame.shape)
    cv2.imwrite("test.png", img * 255)
    return img


@with_view_lock
def train_step(step: int, budget: float = 0.1):
    print(f"Training step {step}...")
    time.sleep(budget)
    print("Done.")


def main(path: str):
    # path = './data/dance/model.pth'
    global gstemplate
    global render_mode_map
    gstemplate = GaussianTemplateModel(path)
    # xyz, frame, scale, opacity, color_map = gstemplate.xyz, gstemplate.frame, gstemplate.scale, gstemplate.opacity, gstemplate.color_map

    render_mode_map["RGB"] = "rgb"
    render_mode_map["Depth"] = "dep"
    render_mode_map["Normals"] = "normals"
    render_mode_map["Depth Normals"] = "dep_normal"

    viewer_stats = ViewerStats()
    server = ViewerServer(render_fn=render_fn)
    host = server.get_host()
    port = server.get_port()
    print("host:", host, "port:", port)
    url = server.request_share_url()
    print("Server started at", url)
    while True:
        if server.player_state == "paused":
            # print("Paused.", url)
            time.sleep(0.01)
        else:
            used_time = server.update_player()
            # print("Updating...", time.time())
            print("Updating...", used_time)
            time.sleep(max(1.0 / server.player_fps - used_time, 0.01))


if __name__ == "__main__":
    # NOTE(Hang Gao @ 01/26): Debug why this not working.
    #  # Use case 1: Just serving the images -- useful when inspecting a
    #  # pretrained checkpoint.
    #  server = ViewerServer(port=30108, render_fn=render_fn)
    #  while True:
    #      with server.lock:
    #          time.sleep(1.0)
    #      time.sleep(0.5)
    tyro.cli(main)
    #  # Use case 1: Just serving the images -- useful when inspecting a
    #  # pretrained checkpoint.
    #  server = ViewerServer(port=30108, render_fn=render_fn)
    #  while True:
    #      time.sleep(1.0)

    # Use case 2: Periodically update the renderer -- useful when training.

    # stats = server.stats
    # max_steps = 10000
    # num_train_rays_per_step = 512 * 512
    # for step in range(max_steps):
    #     while server.training_state == "paused":
    #         time.sleep(0.01)

    #     train_step(step)
    #     num_train_steps_per_sec = 10.0
    #     num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
    #     tic = time.time()

    #     # Update viewer stats.
    #     stats.num_train_rays_per_sec = num_train_rays_per_sec
    #     # Update viewer.
    #     server.update(step, num_train_rays_per_step)
    # server.complete()
