import os
import os.path as osp
import subprocess
import sys
import tempfile

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), "../submodules/segment-anything-2/"))
from sam2.build_sam import build_sam2_video_predictor

sys.path.insert(0, osp.dirname(__file__))
from utils import load_keypoints


def main(
    data_dir: str,
    openpose_dir: str,
    sam2_checkpoint_path: str = osp.join(
        osp.dirname(__file__),
        "../submodules/segment-anything-2/checkpoints/sam2_hiera_large.pt",
    ),
    sam2_cfg_path: str = "sam2_hiera_l.yaml",
):
    img_dir = osp.join(data_dir, "images")
    kp_dir = osp.join(data_dir, "keypoints")
    mask_dir = osp.join(data_dir, "masks")

    if osp.exists(kp_dir) and len(os.listdir(img_dir)) == len(os.listdir(kp_dir)) // 2:
        print("Keypoints already computed.")
    else:
        # Run openpose.
        subprocess.call(
            f"""
./build/examples/openpose/openpose.bin \
    --image_dir {img_dir} \
    --write_json {kp_dir} \
    --write_images {kp_dir} \
    --display 0 \
    --hand \
    --face
                """,
            cwd=openpose_dir,
            shell=True,
        )
    if osp.exists(mask_dir) and len(os.listdir(img_dir)) == len(os.listdir(mask_dir)):
        print("Masks already computed.")
        return
    else:
        os.makedirs(mask_dir, exist_ok=True)
        # Load keypoints.
        keypoints = load_keypoints(kp_dir)
        # Run sam2 with keypoints as prompt for masks.
        predictor = build_sam2_video_predictor(sam2_cfg_path, sam2_checkpoint_path)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for img_name in os.listdir(img_dir):
                    iio.imwrite(
                        osp.join(tmp_dir, img_name.replace(".png", ".jpg")),
                        iio.imread(osp.join(img_dir, img_name)),
                    )
                inference_state = predictor.init_state(video_path=tmp_dir)
            for frame_idx, kps in enumerate(tqdm(keypoints, desc="Prompting SAM2")):
                kps = kps[:25]
                confident_kps = kps[kps[:, 2] > 0.5, :2]
                _, _, mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=0,
                    points=confident_kps,
                    labels=np.ones_like(confident_kps[:, 0], dtype=np.int32),
                )
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                assert len(obj_ids) == 1, "Expected only one human."
                mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                _, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
                largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask = labels == largest_component_index
                iio.imwrite(
                    osp.join(mask_dir, f"{frame_idx:05d}.png"),
                    mask,
                )


if __name__ == "__main__":
    tyro.cli(main)
