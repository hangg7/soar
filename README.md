## Setup

```bash
# You can just use my conda environment.
# conda activate soar

# Get models: /home/hangg/projects/soar/datasets/models/ and /home/hangg/projects/soar/datasets/checkpoints/.
# ln -sf /home/hangg/projects/soar/datasets/models/ /home/hangg/zhy/soar/datasets/
# ln -sf /home/hangg/projects/soar/datasets/checkpoints/ /home/hangg/zhy/soar/datasets/

# For proper setup.
pip install -e ".[all]"
pip install -r submodules/econ/requirements.txt
```

## Preprocess custom videos

I have prepared 11 videos in `/home/hangg/projects/soar/datasets/custom/*.mp4`

```bash
CUDA_VISIBLE_DEVICES=8 python preproc/preprocess_custom.py \
    --video-path /home/hangg/projects/soar/datasets/custom/<VIDEO.MP4> \
    --data-root /home/hangg/projects/soar/datasets/custom/ \
    --openpose-dir /home/hangg/zhy/openpose/ \
    --smplerx-dir /home/hangg/projects/smplerx/
```

Proprocessed data look like this:

```
- VIDEO
    - images/*.png
    - keypoints/*.json
    - masks/*.png
    - normal_B/*.png
    - normal_F/*.png
    - smplx/params.pth
        - some SMPLify-X debugging video here as well.
    - video.mp4
```

## Logs

- 08/04
  - @HG: I have processed these videos in `/home/hangg/zhy/soar/datasets/` so you don't need to do it. @ZY You should just try using them for our code. You should also try other sequences.
    - dance_0
    - dance_1
    - dance_2
    - nadia
    - truman
