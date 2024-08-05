import json
import os
import os.path as osp
import subprocess
import sys
from glob import glob

import imageio.v3 as iio
import smplx
import torch
import tyro

sys.path.insert(0, osp.dirname(__file__))
from utils import SMPLify, load_keypoints, load_smplerx


def main(
    data_dir: str,
    smplerx_dir: str,
    # smplify-x options.
    model_path: str = osp.join(osp.dirname(__file__), "../datasets/models/"),
    debug: bool = False,
    preserve_weight: float = 60.0,
):
    img_dir = osp.join(data_dir, "images")
    kp_dir = osp.join(data_dir, "keypoints")
    smplx_dir = osp.join(data_dir, "smplx")
    smplerx_result_dir = osp.join(smplx_dir, "smplx")

    if osp.exists(smplerx_result_dir) and len(os.listdir(img_dir)) == len(
        os.listdir(smplerx_result_dir)
    ):
        print("SMPLER-X already computed.")
    else:
        # Run smplerx.
        cmd = f"""
source /home/hangg/.anaconda3/bin/activate smplerx; python inference.py \\
    --num_gpus 1 \\
    --pretrained_model smpler_x_h32 \\
    --agora_benchmark agora_model \\
    --img_path {img_dir} \\
    --output_folder {smplx_dir} \\
    --show_verts \\
    --show_bbox
        """
        print("Running command:", cmd)
        subprocess.call(
            cmd,
            cwd=osp.join(smplerx_dir, "main"),
            shell=True,
            executable="/bin/bash",
        )
    if osp.exists(osp.join(smplx_dir, "params.pth")):
        print("SMPL-X already optimized.")
    else:
        device = "cuda"

        img_paths = sorted(glob(osp.join(img_dir, "*.png")))
        N = len(img_paths)
        imgs = [iio.imread(img_path) for img_path in img_paths]

        body_model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            use_face_contour=True,
        ).to(device)

        (
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
        ) = load_smplerx(smplerx_result_dir, device)

        # (N, 3, 3).
        def load_K(i):
            with open(osp.join(smplx_dir, f"meta/{i:05d}_0.json")) as f:
                data = json.load(f)
            K = torch.eye(3, device=device)
            K[0, 0] = data["focal"][0]
            K[1, 1] = data["focal"][1]
            K[0, 2] = data["princpt"][0]
            K[1, 2] = data["princpt"][1]
            return K

        Ks = torch.stack([load_K(i) for i in range(N)], dim=0)
        # (4, 4).
        w2c = torch.eye(4, device=device)
        img_wh = tuple(imgs[0].shape[:2][::-1])

        keypoints = load_keypoints(kp_dir)
        keypoints[..., :-1] /= img_wh

        smplify = SMPLify(
            body_model,
            visual_dir=smplx_dir,
            preserve_weight=preserve_weight,
            debug=debug,
        ).to(device)
        target_kps = torch.as_tensor(keypoints, device=device)
        optimized_params = smplify.fit(
            dict(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                expression=expression,
                transl=transl,
            ),
            Ks,
            w2c,
            img_wh,
            target_kps,
            imgs,
        )
        optimized_params.update({"Ks": Ks, "w2c": w2c, "img_wh": img_wh})
        for k in optimized_params:
            if isinstance(optimized_params[k], torch.Tensor):
                optimized_params[k] = (
                    optimized_params[k].detach().requires_grad_(False).cpu()
                )
        torch.save(optimized_params, osp.join(smplx_dir, "params.pth"))


if __name__ == "__main__":
    tyro.cli(main)
