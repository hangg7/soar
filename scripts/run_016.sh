seq="016"
prompt="A man with a light gray T-shirt, gray pants, and dark shoes. "
echo -e "Running Stage 0"

cd ./submodules/threestudio
python launch.py \
    --config custom/threestudio-soar/configs/gaussiansurfel_imagedream_s0_fs.yaml \
    --train \
    --gpu 0 \
    data.dataroot="/home/hangg/zhy/cleanup/threestudio/data/fs-xhumans/training/$seq" \
    system.prompt_processor.prompt="$prompt" \
    system.geometry.geometry_convert_from="smpl:$seq" \
    data.smpl_type='smplx'\
    system.geometry.smpl_guidance_config.gender="male" #\

# echo -e "Running Stage 1"

# python launch.py \
#     --config custom/threestudio-soar/configs/gaussiansurfel_imagedream_s1_fs.yaml \
#     --train \
#     --gpu 0 \
#     data.dataroot="/home/hangg/zhy/cleanup/threestudio/data/fs-xhumans/training/$seq" \
#     system.prompt_processor.prompt="$prompt" \
#     system.geometry.geometry_convert_from="resume:$seq:/home/hangg/zhy/cleanup/threestudio/outputs/exp-id-s0-exp-fs/${seq}_test_fs/ckpts/last.ckpt" \
#     data.smpl_type='smplx'\
#     system.geometry.smpl_guidance_config.gender="male" 
