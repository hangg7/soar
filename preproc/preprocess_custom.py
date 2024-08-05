import os.path as osp
import subprocess

import tyro


def main(
    video_path: str,
    data_root: str,
    openpose_dir: str,
    smplerx_dir: str,
    height: int = -1,
    skip_time: int = 1,
    start_time: str = "00:00:00",
    end_time: str | None = None,
):
    preproc_dir = osp.dirname(__file__)

    # Extract frames.
    cmd = f"""
python {preproc_dir}/extract_frames.py \\
    --video-path {video_path} \\
    --data-root {data_root} \\
    --height {height} \\
    --skip-time {skip_time} \\
    --start-time {start_time} \\
    --end-time {end_time}
    """
    print("Running command:", cmd)
    subprocess.call(cmd, shell=True)

    seq_name = osp.splitext(osp.basename(video_path))[0]
    data_dir = osp.join(data_root, seq_name)

    # Compute keypoints and masks.
    cmd = f"""
python {preproc_dir}/compute_kp_and_mask.py \\
    --data-dir {data_dir} \\
    --openpose-dir {openpose_dir}
    """
    print("Running command:", cmd)
    subprocess.call(cmd, shell=True)

    # Compute SMPLER-X.
    cmd = f"""
python {preproc_dir}/compute_smplx.py \\
    --data-dir {data_dir} \\
    --smplerx-dir {smplerx_dir}
    """
    print("Running command:", cmd)
    subprocess.call(cmd, shell=True)

    # Compute normals.
    cmd = f"""
python {preproc_dir}/compute_normal.py \\
    --data-dir {data_dir}
    """
    print("Running command:", cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    tyro.cli(main)
