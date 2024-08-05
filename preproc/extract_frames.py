import os
import os.path as osp
import subprocess

import tyro


def main(
    video_path: str,
    data_root: str,
    height: int = -1,
    skip_time: int = 1,
    start_time: str = "00:00:00",
    end_time: str | None = None,
):
    seq_name = osp.splitext(osp.basename(video_path))[0]
    img_dir = osp.join(data_root, seq_name, "images")

    if osp.exists(img_dir) and len(os.listdir(img_dir)) > 0:
        print("Frames already extracted.")
    else:
        os.makedirs(img_dir, exist_ok=True)
        to_str = f"-to {end_time}" if end_time else ""
        cmd = f"""
ffmpeg -i {video_path} \\
    -vf \"select='not(mod(n,{skip_time}))',scale=-1:{height}\" \\
    -fps_mode vfr \\
    -start_number 0 \\
    -ss {start_time} {to_str} \\
    {img_dir}/%05d.png
        """
        print("Running command:", cmd)
        subprocess.call(cmd, shell=True)
        os.symlink(video_path, osp.join(data_root, seq_name, "video.mp4"))


if __name__ == "__main__":
    tyro.cli(main)
