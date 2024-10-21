import os
import os.path as osp
import subprocess
import sys
from glob import glob

import cv2
import imageio.v3 as iio
import numpy as np
import tyro
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

sys.path.insert(0, osp.dirname(__file__))
from utils import load_keypoints


def main(
    data_dir: str,
    openpose_dir: str,
    sam_checkpoint_path: str = osp.join(
        osp.dirname(__file__), "../data/ckpt/sam_vit_h_4b8939.pth"
    ),
):
    img_dir = osp.join(data_dir, "images")
    kp_dir = osp.join(data_dir, "keypoints")
    mask_dir = osp.join(data_dir, "masks")

    if osp.exists(kp_dir) and len(os.listdir(img_dir)) == len(os.listdir(kp_dir)) // 2:
        print("Keypoints already computed.")
    else:
        # Run openpose.
        cmd = f"""
./build/examples/openpose/openpose.bin \\
    --image_dir {img_dir} \\
    --write_json {kp_dir} \\
    --write_images {kp_dir} \\
    --display 0 \\
    --hand \\
    --face
        """
        print("Running command:", cmd)
        subprocess.call(cmd, cwd=openpose_dir, shell=True)
    if osp.exists(mask_dir) and len(os.listdir(img_dir)) == len(os.listdir(mask_dir)):
        print("Masks already computed.")
        return
    else:
        os.makedirs(mask_dir, exist_ok=True)
        # Load keypoints.
        keypoints = load_keypoints(kp_dir)
        # Run sam with keypoints as prompt for masks.
        predictor = SamPredictor(
            sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to("cuda")
        )
        for frame_idx, (img_path, kps) in enumerate(
            zip(
                sorted(glob(osp.join(img_dir, "*"))),
                tqdm(keypoints, desc="Prompting SAM"),
            )
        ):
            img = iio.imread(img_path)[..., ::-1]
            predictor.set_image(img)
            kps = kps[:25]
            confident_kps = kps[kps[:, 2] > 0.5, :2]
            masks, _, _ = predictor.predict(
                confident_kps[:, :2], np.ones_like(confident_kps[:, 0])
            )
            mask = masks.sum(axis=0) > 0
            mask = mask.astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask = labels == largest_component_index
            iio.imwrite(
                osp.join(mask_dir, f"{frame_idx:05d}.png"),
                mask,
            )


if __name__ == "__main__":
    tyro.cli(main)
