seq="CUSTOM" # CUSTOM is a placeholder for the sequence name
prompt="A woman wearing a xxx sweater, xxx pants, xxx sneakers" # This is a placeholder for the prompt, you should replace it with your own prompt
echo -e "Running Stage 0"

cd ./submodules/threestudio
python launch.py \
    --config custom/threestudio-soar/configs/gaussiansurfel_imagedream_s0.yaml \
    --train \
    --gpu 0 \
    data.dataroot="../../data/custom/$seq" \
    system.prompt_processor.prompt="$prompt" \
    system.geometry.geometry_convert_from="smpl:$seq" \
    data.smpl_type='smplx'\
    system.geometry.smpl_guidance_config.gender="neutral" 

echo -e "Running Stage 1"

python launch.py \
    --config custom/threestudio-soar/configs/gaussiansurfel_imagedream_s1.yaml \
    --train \
    --gpu 0 \
    data.dataroot="../../data/custom/$seq" \
    system.prompt_processor.prompt="$prompt" \
    system.geometry.geometry_convert_from="resume:$seq:$(pwd)/outputs/exp-id-s0-org/$seq/ckpts/last.ckpt" \
    data.smpl_type='smplx'\
    system.geometry.smpl_guidance_config.gender="neutral"

