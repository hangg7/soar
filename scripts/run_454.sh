seq="4540332"
prompt="A woman wearing a black sweater, gray cropped pants, black sneakers"
echo -e "Running Stage 0"

cd ./submodules/threestudio
# python launch.py \
#     --config custom/threestudio-soar/configs/gaussiansurfel_imagedream_s0.yaml \
#     --train \
#     --gpu 0 \
#     data.dataroot="../../data/custom/$seq" \
#     system.prompt_processor.prompt="$prompt" \
#     system.geometry.geometry_convert_from="smpl:$seq" \
#     data.smpl_type='smplx'\
#     system.geometry.smpl_guidance_config.gender="neutral" 
#     # subtag='stage0_vid_rerun'

# echo -e "Running Stage 1"

# python launch.py \
#     --config custom/threestudio-soar/configs/gaussiansurfel_imagedream_s1.yaml \
#     --train \
#     --gpu 0 \
#     data.dataroot="../../data/custom/$seq" \
#     system.prompt_processor.prompt="$prompt" \
#     system.geometry.geometry_convert_from="resume:$seq:$(pwd)/outputs/exp-id-s0-org/$seq/ckpts/last.ckpt" \
#     data.smpl_type='smplx'\
#     system.geometry.smpl_guidance_config.gender="neutral"

python launch.py \
    --config custom/threestudio-soar/configs/gaussiansurfel_imagedream_s1.yaml \
    --test \
    --gpu 0 \
    data.dataroot="../../data/custom/$seq" \
    system.prompt_processor.prompt="$prompt" \
    system.geometry.geometry_convert_from="resume:$seq:$(pwd)/outputs/exp-id-s0-org/$seq/ckpts/last.ckpt" \
    data.smpl_type='smplx'\
    system.geometry.smpl_guidance_config.gender="neutral" \
    resume="/scr/panzhy/soar/outputs/exp-id-s1-org/4540332_A_woman_wearing_a_black_sweater,_gray_cropped_pants,_black_sneakers@20241025-224447/ckpts/epoch=0-step=1000.ckpt"
    # resume="/scr/panzhy/soar/outputs/exp-id-s1-org/4540332_A_woman_wearing_a_black_sweater,_gray_cropped_pants,_black_sneakers@20241023-080205/ckpts/epoch=0-step=1000.ckpt"
