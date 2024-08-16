seq="bike"
prompt="A man with a red sweater, beige pants, and black shoes. "
echo -e "Running Stage 0"

cd ./submodules/threestudio
echo $PYTHONPATH
python launch.py \
    --config ./custom/threestudio-soar/configs/gaussiansurfel_imagedream_recon_normal.yaml \
    --train \
    --gpu 0 \
    data.dataroot="../../datasets/neuman/$seq" \
    system.prompt_processor.prompt="$prompt" \
    system.geometry.geometry_convert_from="smpl:$seq" \
    data.smpl_type='smplx'\
    system.geometry.smpl_guidance_config.gender="neutral" #\
    #subtag='stage0_vid_nobg_ck_neutral_flag'

# echo -e "Running Stage 1"

# python launch.py \
#     --config custom/threestudio-gaussiandreamer/configs/gaussiansurfel_imagedream_s1.yaml \
#     --train \
#     --gpu 0 \
#     data.dataroot="/home/hangg/zhy/threestudio/data/custom/$seq" \
#     system.prompt_processor.prompt="$prompt" \
#     system.geometry.geometry_convert_from="resume:$seq:/home/hangg/zhy/threestudio/outputs/exp-id-s0-org/$seq@stage0_vid_nobg_ck_neutral_flag/ckpts/last.ckpt" \
#     data.smpl_type='smplx'\
#     system.geometry.smpl_guidance_config.gender="neutral" \
#     subtag='stage1_vid_nobg_ck_neutral'
