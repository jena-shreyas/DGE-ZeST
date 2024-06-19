python launch.py --config configs/dge_zest.yaml --train --gpu 0 \
    trainer.max_steps=1000 \
    data.source="/data2/manan/2d-gaussian-splatting/data/DTU/scan65/" \
    data.depth_path="/data2/manan/zest_code/demo_assets/depths/DTU/scan65/" \
    data.max_view_num=20 \
    system.mask_thres=0.6 \
    system.guidance.camera_batch_size=5 \
    system.guidance.guidance_scale=7.5 \
    system.gs_source="/data2/manan/gaussian-splatting/output/DTU/scan65/point_cloud/iteration_30000/point_cloud.ply" \
    system.gs_lr_scaler=0.0001 system.gs_final_lr_scaler=0.0001 system.color_lr_scaler=1 \
    system.opacity_lr_scaler=0.0001 system.scaling_lr_scaler=0.0001 system.rotation_lr_scaler=0.0001

    # system.prompt_processor.prompt="turn the dozer into red" \
    # system.seg_prompt="dozer" \

    # --config configs/dge_zest.yaml --train --gpu 0 
    # trainer.max_steps=1000 
    # data.source="/data2/manan/2d-gaussian-splatting/data/DTU/scan65/" 
    # data.depth_path="/data2/manan/zest_code/demo_assets/depths/DTU/scan65/" 
    # data.max_view_num=20 
    # system.mask_thres=0.6 
    # system.guidance.camera_batch_size=5 
    # system.guidance.guidance_scale=7.5 
    # system.gs_source="/data2/manan/gaussian-splatting/output/DTU/scan65/point_cloud/iteration_30000/point_cloud.ply" 
    # system.gs_lr_scaler=0.0001 system.gs_final_lr_scaler=0.0001 system.color_lr_scaler=1 
    # system.opacity_lr_scaler=0.0001 system.scaling_lr_scaler=0.0001 system.rotation_lr_scaler=0.0001