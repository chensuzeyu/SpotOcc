(base) root@autodl-container-6764489861-0b1a1fd9:~# cd autodl-tmp/code/sparseocc
(base) root@autodl-container-6764489861-0b1a1fd9:~/autodl-tmp/code/sparseocc# conda activate sparseocc
(sparseocc) root@autodl-container-6764489861-0b1a1fd9:~/autodl-tmp/code/sparseocc# bash tools/dist_test.sh ./projects/configs/sparseocc/sparseocc_kitti.py ./work_dirs/kitti-origin.pth 1
/root/miniconda3/envs/sparseocc/lib/python3.8/site-packages/torch/distributed/launch.py:178: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  warnings.warn(
projects.mmdet3d_plugin
/root/miniconda3/envs/sparseocc/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py:341: UserWarning: The arguments `feedforward_channels` in BaseTransformerLayer has been deprecated, now you should set `feedforward_channels` and other FFN related arguments to a dict named `ffn_cfgs`. 
  warnings.warn(
/root/miniconda3/envs/sparseocc/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py:341: UserWarning: The arguments `ffn_dropout` in BaseTransformerLayer has been deprecated, now you should set `ffn_drop` and other FFN related arguments to a dict named `ffn_cfgs`. 
  warnings.warn(
/root/miniconda3/envs/sparseocc/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py:341: UserWarning: The arguments `ffn_num_fcs` in BaseTransformerLayer has been deprecated, now you should set `num_fcs` and other FFN related arguments to a dict named `ffn_cfgs`. 
  warnings.warn(
load checkpoint from local path: ./work_dirs/kitti-origin.pth
2025-09-02 11:57:13,921 - root - INFO - DeformConv2dPack img_view_transformer.depth_net.depth_conv.4 is upgraded to version 2.
[                                                  ] 0/4071, elapsed: 0s, ETA:2025-09-02 11:57:20,125 - mmdet - INFO - | name                                           | #elements or shape   |
|:-----------------------------------------------|:---------------------|
| model                                          | 0.2G                 |
|  module                                        |  0.2G                |
|   module.pts_bbox_head                         |   8.2M               |
|    module.pts_bbox_head.transformer_decoder    |    8.0M              |
|    module.pts_bbox_head.query_embed            |    19.2K             |
|    module.pts_bbox_head.query_feat             |    19.2K             |
|    module.pts_bbox_head.level_embed            |    0.6K              |
|    module.pts_bbox_head.cls_embed              |    4.1K              |
|    module.pts_bbox_head.mask_embed             |    0.1M              |
|    module.pts_bbox_head.empty_token            |    0.2K              |
|    module.pts_bbox_head.occ_pred_conv          |    20.7K             |
|   module.img_backbone                          |   63.8M              |
|    module.img_backbone.layers                  |    63.8M             |
|   module.img_neck                              |   1.8M               |
|    module.img_neck.deblocks                    |    1.8M              |
|   module.img_view_transformer                  |   44.1M              |
|    module.img_view_transformer.dx              |    (3,)              |
|    module.img_view_transformer.bx              |    (3,)              |
|    module.img_view_transformer.nx              |    (3,)              |
|    module.img_view_transformer.frustum         |    (112, 24, 80, 3)  |
|    module.img_view_transformer.depth_net       |    43.4M             |
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
|   module.img_bev_encoder_neck                  |   1.0M               |
|    module.img_bev_encoder_neck.input_convs     |    0.3M              |
|    module.img_bev_encoder_neck.lateral_convs   |    25.2K             |
|    module.img_bev_encoder_neck.output_convs    |    37.2K             |
|    module.img_bev_encoder_neck.mask_feature    |    37.1K             |
|    module.img_bev_encoder_neck.up_sample       |    0.6M              |
/root/autodl-tmp/code/sparseocc/projects/mmdet3d_plugin/sparseocc/necks/sparse_feature_pyramid.py:190: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').
  down_coord = last_layer_bcoord[:, 1:] // 2
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 4072/4071, 4.7 task/s, elapsed: 858s, ETA:     0s{'semkitti_SC_Precision': 47.93, 'semkitti_SC_Recall': 58.92, 'semkitti_SC_IoU': 35.92, 'semkitti_SSC_mIoU': 12.21, 'semkitti_SSC_unlabeled_IoU': 94.0, 'semkitti_SSC_car_IoU': 25.12, 'semkitti_SSC_bicycle_IoU': 0.68, 'semkitti_SSC_motorcycle_IoU': 1.49, 'semkitti_SSC_truck_IoU': 9.18, 'semkitti_SSC_other-vehicle_IoU': 8.64, 'semkitti_SSC_person_IoU': 2.29, 'semkitti_SSC_bicyclist_IoU': 1.2, 'semkitti_SSC_motorcyclist_IoU': 0.0, 'semkitti_SSC_road_IoU': 59.67, 'semkitti_SSC_parking_IoU': 13.23, 'semkitti_SSC_sidewalk_IoU': 28.06, 'semkitti_SSC_other-ground_IoU': 0.14, 'semkitti_SSC_building_IoU': 15.33, 'semkitti_SSC_fence_IoU': 6.59, 'semkitti_SSC_vegetation_IoU': 19.01, 'semkitti_SSC_trunk_IoU': 3.51, 'semkitti_SSC_terrain_IoU': 30.93, 'semkitti_SSC_pole_IoU': 4.24, 'semkitti_SSC_traffic-sign_IoU': 2.67, 'semkitti_combined_IoU': 48.13}
(sparseocc) root@autodl-container-6764489861-0b1a1fd9:~/autodl-tmp/code/sparseocc#