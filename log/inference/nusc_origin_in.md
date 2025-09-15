(base) root@autodl-container-45c740b3b5-e77c9f66:~# cd autodl-tmp/code/sparseocc
(base) root@autodl-container-45c740b3b5-e77c9f66:~/autodl-tmp/code/sparseocc# conda activate sparseocc
(sparseocc) root@autodl-container-45c740b3b5-e77c9f66:~/autodl-tmp/code/sparseocc# bash tools/dist_test.sh ./projects/configs/sparseocc/sparseocc_nusc_spot.py ./work_dirs/sparseocc_nusc_256/best_nuScenes_SSC_mIoU_epoch_24_sparseocc.pth 1
/root/miniconda3/envs/sparseocc/lib/python3.8/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
projects.mmdet3d_plugin
/root/miniconda3/envs/sparseocc/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py:92: UserWarning: The arguments `dropout` in MultiheadAttention has been deprecated, now you can separately set `attn_drop`(float), proj_drop(float), and `dropout_layer`(dict) 
  warnings.warn('The arguments `dropout` in MultiheadAttention '
/root/miniconda3/envs/sparseocc/lib/python3.8/site-packages/mmdet/models/backbones/resnet.py:400: UserWarning: DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead
  warnings.warn('DeprecationWarning: pretrained is deprecated, '

load checkpoint from local path: ./work_dirs/sparseocc_nusc_256/best_nuScenes_SSC_mIoU_epoch_24.pth
2025-09-02 10:04:55,203 - root - INFO - DeformConv2dPack img_view_transformer.depth_net.depth_conv.4 is upgraded to version 2.
[                                                  ] 0/6019, elapsed: 0s, ETA:2025-09-02 10:05:01,895 - mmdet - INFO - | name                                           | #elements or shape   |
|:-----------------------------------------------|:---------------------|
| model                                          | 0.1G                 |
|  module                                        |  0.1G                |
|   module.pts_bbox_head                         |   8.2M               |
|    module.pts_bbox_head.transformer_decoder    |    8.0M              |
|    module.pts_bbox_head.query_embed            |    19.2K             |
|    module.pts_bbox_head.query_feat             |    19.2K             |
|    module.pts_bbox_head.level_embed            |    0.6K              |
|    module.pts_bbox_head.cls_embed              |    3.5K              |
|    module.pts_bbox_head.mask_embed             |    0.1M              |
|    module.pts_bbox_head.label_enc              |    3.3K              |
|   module.img_backbone                          |   23.5M              |
|    module.img_backbone.conv1                   |    9.4K              |
|    module.img_backbone.bn1                     |    0.1K              |
|    module.img_backbone.layer1                  |    0.2M              |
|    module.img_backbone.layer2                  |    1.2M              |
|    module.img_backbone.layer3                  |    7.1M              |
|    module.img_backbone.layer4                  |    15.0M             |
|   module.img_neck                              |   2.0M               |
|    module.img_neck.deblocks                    |    2.0M              |
|   module.img_view_transformer                  |   28.1M              |
|    module.img_view_transformer.dx              |    (3,)              |
|    module.img_view_transformer.bx              |    (3,)              |
|    module.img_view_transformer.nx              |    (3,)              |
|    module.img_view_transformer.frustum         |    (112, 16, 44, 3)  |
|    module.img_view_transformer.depth_net       |    27.8M             |
|   module.img_bev_encoder_backbone              |   42.2M              |
|    module.img_bev_encoder_backbone.resBlock0   |    0.9M              |
|    module.img_bev_encoder_backbone.complBlock1 |    0.7M              |
|    module.img_bev_encoder_backbone.aggreBlock1 |    1.0M              |
|    module.img_bev_encoder_backbone.complBlock2 |    0.6M              |
|    module.img_bev_encoder_backbone.aggreBlock2 |    2.7M              |
|    module.img_bev_encoder_backbone.complBlock3 |    1.2M              |
|    module.img_bev_encoder_backbone.aggreBlock3 |    12.0M             |
|    module.img_bev_encoder_backbone.complBlock4 |    2.4M              |
|    module.img_bev_encoder_backbone.aggreBlock4 |    20.7M             |
|   module.img_bev_encoder_neck                  |   1.3M               |
|    module.img_bev_encoder_neck.input_convs     |    0.3M              |
|    module.img_bev_encoder_neck.lateral_convs   |    25.2K             |
|    module.img_bev_encoder_neck.output_convs    |    37.2K             |
|    module.img_bev_encoder_neck.mask_feature    |    37.1K             |
|    module.img_bev_encoder_neck.up_sample       |    0.9M              |
/root/autodl-tmp/code/sparseocc/projects/mmdet3d_plugin/sparseocc/necks/sparse_feature_pyramid.py:190: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  down_coord = last_layer_bcoord[:, 1:] // 2
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6020/6019, 3.9 task/s, elapsed: 1547s, ETA:     0s{'nuScenes_SC_Precision': 42.94, 'nuScenes_SC_Recall': 27.85, 'nuScenes_SC_IoU': 20.33, 'nuScenes_SSC_mIoU': 13.19, 'nuScenes_SSC_empty_IoU': 97.24, 'nuScenes_SSC_barrier_IoU': 16.05, 'nuScenes_SSC_bicycle_IoU': 7.31, 'nuScenes_SSC_bus_IoU': 15.44, 'nuScenes_SSC_car_IoU': 17.61, 'nuScenes_SSC_construction_vehicle_IoU': 5.89, 'nuScenes_SSC_motorcycle_IoU': 9.16, 'nuScenes_SSC_pedestrian_IoU': 10.45, 'nuScenes_SSC_traffic_cone_IoU': 9.53, 'nuScenes_SSC_trailer_IoU': 5.83, 'nuScenes_SSC_truck_IoU': 12.8, 'nuScenes_SSC_driveable_surface_IoU': 30.74, 'nuScenes_SSC_other_flat_IoU': 21.11, 'nuScenes_SSC_sidewalk_IoU': 19.03, 'nuScenes_SSC_terrain_IoU': 16.53, 'nuScenes_SSC_manmade_IoU': 5.02, 'nuScenes_SSC_vegetation_IoU': 8.74, 'nuScenes_combined_IoU': 33.52}
(sparseocc) root@autodl-container-45c740b3b5-e77c9f66:~/autodl-tmp/code/sparseocc#