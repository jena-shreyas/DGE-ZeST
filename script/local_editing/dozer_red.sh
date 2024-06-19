python launch.py --config configs/dge_zest.yaml --train --gpu 0 \
    trainer.max_steps=1000 system.prompt_processor.prompt="turn the dozer into red" \
    data.source="/data2/manan/gaussian-splatting/data/mipnerf360/kitchen/" \
    data.max_view_num=20 \
    system.seg_prompt="dozer" \
    system.mask_thres=0.6 \
    system.guidance.camera_batch_size=5 \
    system.guidance.guidance_scale=7.5 \
    system.gs_source="/data2/manan/gaussian-splatting/output/mipnerf360/kitchen/point_cloud/iteration_30000/point_cloud.ply" \
    system.gs_lr_scaler=0.0001 system.gs_final_lr_scaler=0.0001 system.color_lr_scaler=1 \
    system.opacity_lr_scaler=0.0001 system.scaling_lr_scaler=0.0001 system.rotation_lr_scaler=0.0001
