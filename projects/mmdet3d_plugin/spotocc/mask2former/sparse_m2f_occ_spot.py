# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv3d, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence,
                                         MultiheadAttention,
                                         FFN)
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK, TRANSFORMER_LAYER
from mmcv.runner import ModuleList, force_fp32
from torch.nn.modules.normalization import LayerNorm
import math
from typing import Optional
from torch import Tensor

from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.builder import HEADS, build_loss

from .base.mmdet_utils import (sample_valid_coords_with_frequencies,
                          get_uncertain_point_coords_3d_with_frequency,
                          preprocess_occupancy_gt, point_sample_3d)

from .base.anchor_free_head import AnchorFreeHead
from .base.maskformer_head import MaskFormerHead
from projects.mmdet3d_plugin.utils.semkitti import semantic_kitti_class_frequencies

from einops import rearrange


@ATTENTION.register_module(name='SPOT_CA_0')
class SPOT_CA_0(nn.Module):
    """
    Prototype-based Masked Cross-Attention adapted for MMCV/MMDet.
    Uses Softmax weighted average instead of argmax for differentiable key selection.
    Assumes key/value/query inputs are (SeqLen, Batch, Dim) - batch_first=False.
    Supports multi-head with per-head Top-K selection.
    """
    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 1,
                 pruning_ratio: float = 0.1,
                 min_k: int = 32,
                 dropout: float = 0.0,
                 batch_first: bool = False,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.pruning_ratio = pruning_ratio
        self.min_k = min_k

        assert self.embed_dims % self.num_heads == 0, \
            f"embed_dims ({self.embed_dims}) must be divisible by num_heads ({self.num_heads})."
        self.head_dim = self.embed_dims // self.num_heads

        self.key_proj = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims)
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims)
        )

        self.proj = nn.Linear(embed_dims, embed_dims)
        self.final = nn.Linear(embed_dims, embed_dims)
        self.alpha = nn.Parameter(torch.ones(1, 1, embed_dims))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.alpha, 0.1)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                cross_attn_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:

        if self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            if query_pos is not None:
                query_pos = query_pos.permute(1, 0, 2)
            if key_pos is not None:
                key_pos = key_pos.permute(1, 0, 2)

        residual = query

        query = self.with_pos_embed(query, query_pos)
        key = self.with_pos_embed(key, key_pos)
        value_with_pos = key

        q_proj = self.query_proj(query)
        k_proj = self.key_proj(key)
        # [B, Q, C] / [B, K, C]
        q_bqc = q_proj.permute(1, 0, 2)
        k_bkc = k_proj.permute(1, 0, 2)
        v_bkc = value_with_pos.permute(1, 0, 2)

        # reshape to heads: [B, Q, H, D], [B, K, H, D]
        B, Q, _ = q_bqc.shape
        K = k_bkc.shape[1]
        q_bqhd = q_bqc.view(B, Q, self.num_heads, self.head_dim)
        k_bkhd = k_bkc.view(B, K, self.num_heads, self.head_dim)
        v_bkhd = v_bkc.view(B, K, self.num_heads, self.head_dim)

        # L2 normalize on head dim
        q_bqhd = F.normalize(q_bqhd, dim=-1)
        k_bkhd = F.normalize(k_bkhd, dim=-1)

        # similarity per head: [B, H, Q, K]
        sim = torch.einsum('bqhd, bkhd -> bhqk', q_bqhd, k_bkhd)

        # Apply cross-attention mask
        if cross_attn_mask is not None:
            # cross_attn_mask.shape: [B, Q, K]
            # Expand mask to match sim's shape [B, H, Q, K]
            expanded_mask = cross_attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            # Fill positions where the mask is True with a very small value,
            # so the weight approaches 0 after softmax
            sim.masked_fill_(expanded_mask, float('-inf'))

        key_len = key.shape[0]
        dynamic_k = int(math.ceil(self.pruning_ratio * key_len))
        k_to_use = max(self.min_k, min(dynamic_k, key_len))

        # top-k per head along keys dim
        top_k_sim, top_k_indices = torch.topk(sim, k=k_to_use, dim=3)

        scale_factor = self.head_dim ** -0.5
        attn_weights = F.softmax(top_k_sim * scale_factor, dim=3)

        # values to [B, H, Q, k, D]
        v_bhkd = v_bkhd.permute(0, 2, 1, 3)  # [B, H, K, D]
        v_bhqkd = v_bhkd.unsqueeze(2).expand(-1, -1, Q, -1, -1)
        top_k_idx_exp = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        gathered_values = torch.gather(v_bhqkd, 3, top_k_idx_exp)

        # weighted sum over k: [B, H, Q, D]
        selected_value = (attn_weights.unsqueeze(-1) * gathered_values).sum(dim=3)

        # merge heads back to [B, Q, C]
        selected_value_merge = selected_value.permute(0, 2, 1, 3).reshape(B, Q, self.embed_dims)

        q_proj_permuted = q_proj.permute(1, 0, 2)
        interaction = self.proj(selected_value_merge * q_proj_permuted)
        out = F.normalize(interaction, dim=1) * self.alpha + selected_value_merge
        out = self.final(out)

        out_permuted = out.permute(1, 0, 2)
        query = residual + self.dropout(out_permuted)

        if self.batch_first:
            query = query.permute(1, 0, 2)

        return query


@TRANSFORMER_LAYER.register_module(name='SparseProtoTransformerDecoderLayer_0')
class SparseProtoTransformerDecoderLayer_0(nn.Module):
    """
    Transformer Decoder Layer following PEM structure:
    CrossAttn -> Norm -> SelfAttn -> Norm -> FFN -> Norm
    """
    def __init__(self,
                 pem_ca_cfg=dict(
                     type='SPOT_CA_0',
                     embed_dims=192,
                     num_heads=6,
                     dropout=0.0,
                     batch_first=False),
                 self_attn_cfg=dict(
                     embed_dims=192,
                     num_heads=6,
                     dropout=0.0,
                     batch_first=False),
                 ffn_cfg=dict(
                     embed_dims=192,
                     feedforward_channels=192 * 8,
                     num_fcs=2,
                     ffn_drop=0.0,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg=dict(type='LN'),
                 pre_norm=False,
                 **kwargs):
        super().__init__()

        self.pre_norm = pre_norm

        # Directly instantiate custom PEM Cross Attention
        self.pem_cross_attn = SPOT_CA_0(**pem_ca_cfg)

        # Use standard MultiheadAttention as self-attn
        self.self_attn = MultiheadAttention(**self_attn_cfg)

        # Use standard FFN
        self.ffn = FFN(**ffn_cfg)

        # Three LayerNorms: after cross_attn, after self_attn, after ffn
        self.norms = ModuleList()
        for _ in range(3):
            self.norms.append(LayerNorm(pem_ca_cfg['embed_dims']))

        self.embed_dims = pem_ca_cfg['embed_dims']

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                self_attn_mask: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                cross_attn_mask: Optional[Tensor] = None,
                **kwargs):

        # 1. PEM Cross Attention
        query = self.pem_cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            cross_attn_mask=cross_attn_mask,
        )
        # 2. Norm after Cross Attention
        query = self.norms[0](query)

        # 3. Self Attention (+ residual)
        residual_sa = query
        query_sa = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            key_padding_mask=query_key_padding_mask
        )
        query = residual_sa + query_sa

        # 4. Norm after Self Attention
        query = self.norms[1](query)

        # 5. FFN
        query = self.ffn(query)

        # 6. Norm after FFN
        query = self.norms[2](query)

        return query


# Sparse Mask2former Head for 3D Occupancy Segmentation
@HEADS.register_module()
class SparseProtoMask2FormerOccHead(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 feat_channels,
                 out_channels,
                 num_occupancy_classes=20,
                 final_occ_size=[512, 512, 40],
                 num_queries=100,
                 num_transformer_feat_level=3,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 pooling_attn_mask=True,
                 sample_weight_gamma=0.25,
                 empty_idx=0,
                 with_cp=True,
                 align_corners=True,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 dn_cfg=None,  # DN configuration parameters
                 dn_loss_layers=1, # Number of layers to calculate DN loss, default is 1 (out of 9 layers)
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        
        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = self.num_occupancy_classes
        self.num_queries = num_queries
        self.with_cp = with_cp
        
        ''' Transformer Decoder Related '''
        # number of multi-scale features for masked attention
        self.num_transformer_feat_level = num_transformer_feat_level
        # Compatible with the configuration structure of PEM decoder layers, automatically parses num_heads from pem_ca_cfg/self_attn_cfg
        first_layer_cfg = transformer_decoder['transformerlayers']
        if 'pem_ca_cfg' in first_layer_cfg and 'num_heads' in first_layer_cfg['pem_ca_cfg']:
            self.num_heads = first_layer_cfg['pem_ca_cfg']['num_heads']
        elif 'self_attn_cfg' in first_layer_cfg and 'num_heads' in first_layer_cfg['self_attn_cfg']:
            self.num_heads = first_layer_cfg['self_attn_cfg']['num_heads']
        else:
            print('Warning: Cannot determine num_heads automatically, using default 6.')
            self.num_heads = 6
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution, align the channel of input features
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv3d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
                
        self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        ''' Pixel Decoder Related, skipped '''
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        # create class_weights for semantic_kitti
        self.class_weight = loss_cls.class_weight
        kitti_class_weights = 1 / np.log(semantic_kitti_class_frequencies)
        norm_kitti_class_weights = kitti_class_weights / kitti_class_weights[0]
        norm_kitti_class_weights = norm_kitti_class_weights.tolist()
        # append the class_weight for background
        norm_kitti_class_weights.append(self.class_weight[-1])
        self.class_weight = norm_kitti_class_weights
        
        loss_cls.class_weight = self.class_weight
        
         # DN (Denoising) configuration and components
        if dn_cfg is None:
            dn_cfg = {
                'scalar': 1,  # Number of repetitions for each GT mask
                'noise_scale': 0.01,  # Intensity of feature noise
                'dn_label_noise_ratio': 0.01  # Ratio of adding noise to GT labels
            }
        self.dn_cfg = dn_cfg
        self.dn_loss_layers = dn_loss_layers
        
        # a class embedding layer to convert GT class labels into feature vectors
        self.label_enc = nn.Embedding(self.num_occupancy_classes, feat_channels)
        
        # computing sampling weight        
        sample_weights = 1 / semantic_kitti_class_frequencies
        sample_weights = sample_weights / sample_weights.min()
        self.baseline_sample_weights = sample_weights
        self.sample_weight_gamma = sample_weight_gamma
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.pooling_attn_mask = pooling_attn_mask
        
        # align_corners
        self.align_corners = align_corners

        # for sparse segmentation (interface reserved for configuration compatibility)
        self.empty_idx = empty_idx
        self.final_occ_size = final_occ_size

    def get_sampling_weights(self):
        if type(self.sample_weight_gamma) is list:
            # dynamic sampling weights
            min_gamma, max_gamma = self.sample_weight_gamma
            sample_weight_gamma = np.random.uniform(low=min_gamma, high=max_gamma)
        else:
            sample_weight_gamma = self.sample_weight_gamma
        
        self.sample_weights = self.baseline_sample_weights ** sample_weight_gamma
        
    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv3d):
                caffe2_xavier_init(m, bias=0)
        
        if hasattr(self, "pixel_decoder"):
            self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
    def _prepare_for_dn(self, gt_labels_list, gt_masks_list):
        """
        Prepare noisy queries for Denoising training.
        Reference MultiScaleMaskedTransformerDecoderMaskDN.prepare_for_dn_v5 method.
        
        Args:
            gt_labels_list (list[Tensor]): List of GT labels for each batch
            gt_masks_list (list[Tensor]): List of GT masks for each batch
            
        Returns:
            tuple: (noised_dn_queries, tgt_mask, dn_meta, dn_gt_masks)
                - noised_dn_queries (Tensor): Noisy DN queries, shape (num_dn_queries, batch_size, feat_channels)
                - tgt_mask (Tensor): self-attention mask, shape (total_queries, total_queries) 
                - dn_meta (dict): DN metadata dictionary
                - dn_gt_masks (Tensor): GT masks corresponding to DN queries, shape (batch_size, pad_size, w, h, D)
        """
        scalar = self.dn_cfg['scalar']
        noise_scale = self.dn_cfg['noise_scale'] 
        dn_label_noise_ratio = self.dn_cfg['dn_label_noise_ratio']
        
        # Count the number of GTs in each batch
        num_gts_list = [len(gt_labels) for gt_labels in gt_labels_list]
        max_num_gts = max(num_gts_list) if num_gts_list else 0
        
        # If there are no GTs, return None to indicate that DN will not be performed
        if max_num_gts == 0 or scalar == 0:
            return None
            
        batch_size = len(gt_labels_list)
        pad_size = scalar * max_num_gts  # Total number of DN queries
        
        # Initialize container for DN queries
        dn_queries = torch.zeros([batch_size, pad_size, self.decoder_embed_dims]).cuda()
        
        # Initialize container for DN GT masks
        dn_gt_masks_list = []
        
        # Generate DN queries for each batch
        for batch_idx, (gt_labels, gt_masks) in enumerate(zip(gt_labels_list, gt_masks_list)):
            num_gts = len(gt_labels)
            if num_gts == 0:
                # If there is no GT, create a zero mask for padding. The spatial dimensions
                # need to be obtained from other batches or a fixed size.
                # Default dimensions are used here and will be adjusted in subsequent processing.
                w, h, d = 64, 64, 8  # Default spatial dimensions, will be adjusted by interpolation according to the actual feature map size later.
                final_masks = torch.zeros(pad_size, w, h, d, device=torch.device('cuda'))
                dn_gt_masks_list.append(final_masks)
                continue
                
            # Repeat GT labels scalar times
            repeated_labels = gt_labels.repeat(scalar)
            
            # Repeat GT masks scalar times
            repeated_masks = gt_masks.repeat(scalar, 1, 1, 1)  # repeat scalar times

            # Ensure the mask is of boolean type
            masks_bool = repeated_masks.bool()
            # Get the spatial dimensions of the mask W, H, D
            _, W, H, D = gt_masks.shape
            
            # Calculate the volume of each mask (i.e., the number of True voxels)
            areas = masks_bool.sum(dim=(-1, -2, -3))
            
            # Calculate the noise ratio based on the volume and noise scaling factor
            # noise_scale comes from self.dn_cfg, e.g., 0.4
            noise_ratio = areas * noise_scale / (W * H * D)
            
            # Generate a random noise mask where the probability of being True is determined by noise_ratio
            # noise_ratio[:, None, None, None] is used to broadcast the noise ratio of each mask to its spatial dimensions
            delta_mask = torch.rand_like(masks_bool, dtype=torch.float) < noise_ratio[:, None, None, None]
            
            # Use logical XOR to flip pixels in the mask to apply noise
            # GT=1, delta=1 -> 0 (create a hole in the object)
            # GT=0, delta=1 -> 1 (create noise in the background)
            noised_masks = torch.logical_xor(masks_bool, delta_mask)
            
            # Create a padding tensor to ensure the number of masks is consistent for each batch
            padding_needed = pad_size - repeated_masks.shape[0]
            if padding_needed > 0:
                padding_mask = torch.zeros(padding_needed, *gt_masks.shape[1:], device=gt_masks.device)
                final_masks = torch.cat([repeated_masks, padding_mask], dim=0)
            else:
                final_masks = repeated_masks
            
            dn_gt_masks_list.append(final_masks)
            
            # label noise: randomly replace a portion of GT labels with other classes
            if dn_label_noise_ratio > 0:
                prob = torch.rand_like(repeated_labels.float())
                noise_mask = prob < dn_label_noise_ratio
                # Randomly generate new labels (from 0 to num_occupancy_classes-1)
                random_labels = torch.randint_like(repeated_labels[noise_mask], 0, self.num_occupancy_classes)
                repeated_labels[noise_mask] = random_labels
                
            # Use label_enc to convert labels into feature vectors
            label_features = self.label_enc(repeated_labels)
            
            # noise to the feature vectors
            if noise_scale > 0:
                feature_noise = (torch.rand_like(label_features) * 2 - 1.0) * noise_scale
                noised_features = label_features + feature_noise
            else:
                noised_features = label_features
                
            # Fill the DN features into the corresponding positions
            for i in range(scalar):
                start_idx = i * max_num_gts
                end_idx = start_idx + num_gts
                dn_queries[batch_idx, start_idx:end_idx] = noised_features[i * num_gts:(i + 1) * num_gts]
        
        # Convert to (num_dn_queries, batch_size, feat_channels) format to match Transformer input
        dn_queries = dn_queries.permute(1, 0, 2)
        
        # Construct self-attention mask
        # Total number of queries = DN queries + original learnable queries
        total_queries = pad_size + self.num_queries
        tgt_mask = torch.ones(total_queries, total_queries).cuda() < 0  # Initialize to False (visible)
        
        # Masking between DN queries: DN queries from different groups cannot see each other
        for i in range(scalar):
            group_start = i * max_num_gts
            group_end = (i + 1) * max_num_gts
            for j in range(scalar):
                if i != j:  # Different groups cannot see each other
                    other_start = j * max_num_gts
                    other_end = (j + 1) * max_num_gts
                    tgt_mask[group_start:group_end, other_start:other_end] = True
        
        # Original queries cannot see DN queries
        tgt_mask[pad_size:, :pad_size] = True
        
        # Stack GT masks from all batches
        dn_gt_masks = torch.stack(dn_gt_masks_list, dim=0)  # (batch_size, pad_size, w, h, D)
        
        # Prepare DN metadata
        dn_meta = {
            'pad_size': pad_size,
            'scalar': scalar,
            'max_num_gts': max_num_gts,
            'num_gts_list': num_gts_list,
        }
        
        return dn_queries, tgt_mask, dn_meta, dn_gt_masks
    
    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, w, h).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, w, h).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, w, h).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, x, y, z).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, x, y, z).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, w, h).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        gt_labels = gt_labels.long()
        
        # create sampling weights
        point_indices, point_coords = sample_valid_coords_with_frequencies(self.num_points, 
                gt_labels=gt_labels, gt_masks=gt_masks, sample_weights=self.sample_weights)
        
        point_coords = point_coords[..., [2, 1, 0]]
        mask_points_pred = point_sample_3d(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1), align_corners=self.align_corners).squeeze(1)
        
        # shape (num_gts, num_points)
        gt_points_masks = gt_masks.view(num_gts, -1)[:, point_indices]
        
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label target
        labels = gt_labels.new_full((self.num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = labels.new_ones(self.num_queries).type_as(cls_score)
        class_weights_tensor = torch.tensor(self.class_weight).type_as(cls_score)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = class_weights_tensor[labels[pos_inds]]
        
        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    
    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
               gt_masks_list, img_metas, dn_meta=None):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder layers
                with shape (num_decoder_layers, batch_size, total_queries,
                num_classes+1). Note num_classes+1=`cls_out_channels` 
                includes background. total_queries = dn_queries + num_queries
            all_mask_preds (Tensor): Mask predictions for all decoder layers
                shape (num_decoder_layers, batch_size, total_queries, w, h, d).
            gt_labels_list (list[Tensor]): List of ground truth class labels for each image
                Ground truth class indices for each image with shape (n_gts, ). 
                n_gts is the sum of number of stuff type and number of instance 
                in a image.
            gt_masks_list (list[Tensor]): List of ground truth masks for each image
                Ground truth mask for each image with shape (n_gts, w, h, d).
            img_metas (list[dict]): List of metadata for each image
                List of image meta information.
            dn_meta (dict): DN metadata dictionary, including pad_size, etc.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        
        # Get DN information
        if dn_meta is None:
            dn_meta = {}
        pad_size = dn_meta.get('pad_size', 0)
        
        # Separate predictions for DN and matching
        if pad_size > 0:
            # Case with DN queries: separate DN and matching parts
            dn_cls_scores = [score[:, :pad_size, :] for score in all_cls_scores]
            matching_cls_scores = [score[:, pad_size:, :] for score in all_cls_scores]
            
            dn_mask_preds = [mask[:, :pad_size, :, :, :] for mask in all_mask_preds]
            matching_mask_preds = [mask[:, pad_size:, :, :, :] for mask in all_mask_preds]
        else:
            # Case without DN queries: all predictions are for matching
            dn_cls_scores = []
            dn_mask_preds = []
            matching_cls_scores = all_cls_scores
            matching_mask_preds = all_mask_preds

        # Calculate loss independently for each decoder layer
        num_dec_layers = len(all_cls_scores)
        # Create the same gt_labels_list and gt_masks_list for each decoder layer
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
        # Calculate loss for the matching part (using existing Hungarian matching logic)
        matching_losses_cls, matching_losses_mask, matching_losses_dice = multi_apply(
            self.loss_single, matching_cls_scores, matching_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)
        
        # Calculate loss for the DN part (if DN queries exist)
        dn_losses_cls, dn_losses_mask, dn_losses_dice = [], [], []
        if pad_size > 0:
            # Calculate DN loss only for the first n layers
            num_dn_layers_to_supervise = min(self.dn_loss_layers, num_dec_layers)

            dn_losses_cls, dn_losses_mask, dn_losses_dice = multi_apply(
                self.loss_dn_single, 
                dn_cls_scores[:num_dn_layers_to_supervise], 
                dn_mask_preds[:num_dn_layers_to_supervise],
                all_gt_labels_list[:num_dn_layers_to_supervise], 
                all_gt_masks_list[:num_dn_layers_to_supervise], 
                img_metas_list[:num_dn_layers_to_supervise], 
                [dn_meta for _ in range(num_dn_layers_to_supervise)])
        
        # Build loss dictionary
        loss_dict = dict()
        
        # Matching loss for the last decoder layer
        loss_dict['loss_cls'] = matching_losses_cls[-1]
        loss_dict['loss_mask'] = matching_losses_mask[-1]
        loss_dict['loss_dice'] = matching_losses_dice[-1]
        
        # DN loss for the last decoder layer (if it exists and is supervised)
        # condition: num_dec_layers <= self.dn_loss_layers
        if pad_size > 0 and num_dec_layers <= self.dn_loss_layers:
            loss_dict['loss_cls_dn'] = dn_losses_cls[-1]
            loss_dict['loss_mask_dn'] = dn_losses_mask[-1]
            loss_dict['loss_dice_dn'] = dn_losses_dice[-1]
        
        # Losses for other decoder layers
        num_dec_layer = 0
        for i in range(num_dec_layers - 1):  # All layers except the last one
            # Matching loss
            loss_dict[f'd{num_dec_layer}.loss_cls'] = matching_losses_cls[i]
            loss_dict[f'd{num_dec_layer}.loss_mask'] = matching_losses_mask[i]
            loss_dict[f'd{num_dec_layer}.loss_dice'] = matching_losses_dice[i]
            
            # DN loss (if it exists)
            if pad_size > 0 and i < self.dn_loss_layers:  # dn_loss_layers
                loss_dict[f'd{num_dec_layer}.loss_cls_dn'] = dn_losses_cls[i]
                loss_dict[f'd{num_dec_layer}.loss_mask_dn'] = dn_losses_mask[i]
                loss_dict[f'd{num_dec_layer}.loss_dice_dn'] = dn_losses_dice[i]
            
            num_dec_layer += 1
        # Example of the returned dictionary:
        # {
        # 'loss_cls': 0.1, 
        # 'loss_mask': 0.2, 
        # 'loss_dice': 0.3, 
        # 'd0.loss_cls': 0.05, 
        # 'd0.loss_mask': 0.1, 
        # 'd0.loss_dice': 0.15, 
        # 'd1.loss_cls': 0.03, 
        # 'd1.loss_mask': 0.08, 
        # 'd1.loss_dice': 0.12
        # }

        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, x, y, z).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, x, y, z).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)  # Split along the batch size dimension
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        # Assign ground truth labels and mask targets for each query
        # Return values:
        # labels_list: List of ground truth labels for each query; shape (num_queries, )
        # label_weights_list: List of label weights for each query; shape (num_queries, )
        # mask_targets_list: List of ground truth masks for each query; shape (num_gts, w, h, d)
        # mask_weights_list: List of mask weights for each query; shape (num_queries, )
        # num_total_pos: Number of positive samples; int
        # num_total_neg: Number of negative samples; int
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                gt_labels_list, gt_masks_list, img_metas)
        
        # Stack the ground truth labels and mask weights for each query
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # Special handling for mask_targets: concatenate positive sample masks from all images
        # shape (num_total_gts, w, h, d)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classification loss
        # Use cross-entropy loss with class weights
        cls_scores = cls_scores.flatten(0, 1)  # (batch_size*num_queries, num_classes+1)
        labels = labels.flatten(0, 1)  # (batch_size*num_queries, )
        label_weights = label_weights.flatten(0, 1)  # (batch_size*num_queries, )
        class_weight = cls_scores.new_tensor(self.class_weight)  # (num_classes+1, )
        
        # Use class_weight and labels to calculate the weight for each sample,
        # then sum the weights of all samples to get avg_factor
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum(),
        )

        # extract positive ones
        # shape (batch_size, num_queries, w, h, d) -> (num_total_gts, w, h, d)
        mask_preds = mask_preds[mask_weights > 0]  # use >0 as index
        mask_weights = mask_weights[mask_weights > 0]
        # mask_preds shape becomes (num_positive_queries, w, h, D)
        # mask_weights shape becomes (num_positive_queries,)

        # Handle the case of no positive samples
        if mask_targets.shape[0] == 0:
            # zero match: loss is 0 when there are no positive samples
            # Set dummy values for Dice loss and Mask loss (simply use the sum of predictions)
            # Return the loss values early to avoid subsequent calculation errors
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        ''' 
        randomly sample K points for supervision, which can largely improve the 
        efficiency and preserve the performance. oversample_ratio = 3.0, importance_sample_ratio = 0.75
        '''
        # Randomly sample K points for supervision, which can significantly improve efficiency and maintain performance
        # oversample_ratio = 3.0, importance_sample_ratio = 0.75
        
        with torch.no_grad():
            # Sample points based on prediction uncertainty and class frequencies
            # Input parameter description:
            # mask_preds.unsqueeze(1): channel dimension -> (N,1,H,W,D)
            # self.num_points: number of sampling points (e.g., 12544)
            # self.oversample_ratio=3.0: oversampling ratio
            # self.importance_sample_ratio=0.75: important sample ratio
            point_indices, point_coords = get_uncertain_point_coords_3d_with_frequency(
                mask_preds.unsqueeze(1), None, gt_labels_list, gt_masks_list, 
                self.sample_weights, self.num_points, self.oversample_ratio, 
                self.importance_sample_ratio)
            # Output parameter description:
            # point_indices: sampled point indices; shape (num_total_gts, num_points)
            # point_coords: sampled point coordinates; shape (num_total_gts, num_points, 3)
            
            # Get the ground truth labels at the sampled points
            # shape (num_total_gts, w, h, d) -> (num_total_gts, num_points)
            # Use gather operation to extract labels corresponding to sampled points from mask_targets
            # shape (num_total_gts, w, h) -> (num_total_gts, num_points)
            mask_point_targets = torch.gather(mask_targets.view(mask_targets.shape[0], -1), 
                                        dim=1, index=point_indices)
        
        # Get the predicted values at the sampled points
        # shape (num_queries, w, h) -> (num_queries, num_points)
        mask_point_preds = point_sample_3d(
            mask_preds.unsqueeze(1), point_coords[..., [2, 1, 0]], align_corners=self.align_corners).squeeze(1)
        
        # dice loss
        num_total_mask_weights = reduce_mean(mask_weights.sum())
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, 
                        weight=mask_weights, avg_factor=num_total_mask_weights)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        mask_point_weights = mask_weights.view(-1, 1).repeat(1, self.num_points)
        # Flatten predictions and labels: reshape(-1) -> (num_queries*num_points,)
        mask_point_weights = mask_point_weights.reshape(-1)

        # Calculate average weight
        num_total_mask_point_weights = reduce_mean(mask_point_weights.sum())
        # Calculate Mask loss, usually BCE (Binary Cross-Entropy) loss
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            weight=mask_point_weights,
            avg_factor=num_total_mask_point_weights)

        return loss_cls, loss_mask, loss_dice
    
    def loss_dn_single(self, cls_scores, mask_preds, gt_labels_list,
                       gt_masks_list, img_metas, dn_meta):
        """
        Calculate the loss for Denoising queries.
        DN queries do not require Hungarian matching; the loss is calculated directly based on the GT.
        
        Args:
            cls_scores (Tensor): Classification scores for the DN part, shape (batch_size, pad_size, num_classes+1)
            mask_preds (Tensor): Mask predictions for the DN part, shape (batch_size, pad_size, w, h, d)
            gt_labels_list (list[Tensor]): List of GT labels
            gt_masks_list (list[Tensor]): List of GT masks
            img_metas (list[dict]): Image metadata
            dn_meta (dict): DN metadata
            
        Returns:
            tuple[Tensor]: DN losses (loss_cls_dn, loss_mask_dn, loss_dice_dn)
        """
        scalar = dn_meta['scalar']
        max_num_gts = dn_meta['max_num_gts']
        num_gts_list = dn_meta['num_gts_list']
        pad_size = dn_meta['pad_size']
        
        batch_size = cls_scores.size(0)
        device = cls_scores.device
        
        # Prepare DN labels and mask targets
        dn_labels = []
        dn_masks = []
        dn_label_weights = []
        dn_mask_weights = []
        
        for batch_idx, (gt_labels, gt_masks) in enumerate(zip(gt_labels_list, gt_masks_list)):
            num_gts = len(gt_labels)
            
            # Construct DN labels for the current batch (repeated scalar times)
            batch_dn_labels = torch.full((pad_size,), self.num_classes, dtype=torch.long, device=device)  # Initialize with background class
            batch_dn_label_weights = torch.zeros((pad_size,), dtype=torch.float, device=device)
            
            # Construct DN mask targets and weights for the current batch
            batch_dn_masks = []
            batch_dn_mask_weights = torch.zeros((pad_size,), dtype=torch.float, device=device)
            
            if num_gts > 0:
                # Fill in the real DN labels and masks
                for i in range(scalar):
                    start_idx = i * max_num_gts
                    end_idx = start_idx + num_gts
                    
                    # Set labels and weights
                    batch_dn_labels[start_idx:end_idx] = gt_labels
                    batch_dn_label_weights[start_idx:end_idx] = 1.0
                    
                    # Set mask weights (using class weights)
                    class_weights_tensor = torch.tensor(self.class_weight, device=device)
                    batch_dn_mask_weights[start_idx:end_idx] = class_weights_tensor[gt_labels]
                    
                    # GT masks to the batch mask list
                    for gt_mask in gt_masks:
                        batch_dn_masks.append(gt_mask)
                        
                # Pad the remaining positions with zero masks
                remaining_positions = pad_size - len(batch_dn_masks)
                zero_mask = torch.zeros_like(gt_masks[0]) if len(gt_masks) > 0 else torch.zeros((1, 1, 1), device=device)
                for _ in range(remaining_positions):
                    batch_dn_masks.append(zero_mask)
            else:
                # In case of no GT, pad with zero masks
                zero_mask = torch.zeros((1, 1, 1), device=device)
                for _ in range(pad_size):
                    batch_dn_masks.append(zero_mask)
            
            # Stack masks
            batch_dn_masks = torch.stack(batch_dn_masks, dim=0)
                
            dn_labels.append(batch_dn_labels)
            dn_masks.append(batch_dn_masks)
            dn_label_weights.append(batch_dn_label_weights)
            dn_mask_weights.append(batch_dn_mask_weights)
        
        # Stack all batches
        dn_labels = torch.stack(dn_labels, dim=0)  # (batch_size, pad_size)
        dn_masks = torch.stack(dn_masks, dim=0)    # (batch_size, pad_size, w, h, d) 
        dn_label_weights = torch.stack(dn_label_weights, dim=0)  # (batch_size, pad_size)
        dn_mask_weights = torch.stack(dn_mask_weights, dim=0)    # (batch_size, pad_size)
        
        # Calculate classification loss
        cls_scores_flat = cls_scores.flatten(0, 1)  # (batch_size*pad_size, num_classes+1)
        dn_labels_flat = dn_labels.flatten(0, 1)    # (batch_size*pad_size,)
        dn_label_weights_flat = dn_label_weights.flatten(0, 1)  # (batch_size*pad_size,)
        
        class_weight = cls_scores_flat.new_tensor(self.class_weight)
        loss_cls_dn = self.loss_cls(
            cls_scores_flat,
            dn_labels_flat, 
            dn_label_weights_flat,
            avg_factor=class_weight[dn_labels_flat].sum(),
        )
        
        # Extract valid mask predictions (mask_weights > 0)
        mask_preds_valid = mask_preds[dn_mask_weights > 0]  # (num_valid, w, h, d)
        dn_masks_valid = dn_masks[dn_mask_weights > 0]      # (num_valid, w, h, d) 
        dn_mask_weights_valid = dn_mask_weights[dn_mask_weights > 0]  # (num_valid,)
        
        if mask_preds_valid.shape[0] == 0:
            # When there are no valid masks, return zero loss
            loss_dice_dn = mask_preds.sum() * 0
            loss_mask_dn = mask_preds.sum() * 0
        else:
            # Randomly sample points for supervision (reuse existing sampling logic)
            with torch.no_grad():
                # Prepare aligned GT for DN sampling
                # The shape of dn_masks_valid is (num_valid, w, h, d)
                # The sampling function requires a list of tensors, where each tensor represents the GT masks for one sample
                # Therefore, we split dn_masks_valid into a list, with each element having a shape of (1, w, h, d)
                dn_masks_list_for_sampling = [m.unsqueeze(0) for m in dn_masks_valid]

                # Similarly, create a corresponding list for labels
                dn_labels_valid = dn_labels[dn_mask_weights > 0]
                dn_labels_list_for_sampling = [l.unsqueeze(0) for l in dn_labels_valid]
                
                point_indices, point_coords = get_uncertain_point_coords_3d_with_frequency(
                    mask_preds_valid.unsqueeze(1), 
                    None, 
                    dn_labels_list_for_sampling,  # use the GT labels list
                    dn_masks_list_for_sampling,   # use the GT masks list
                    self.sample_weights, 
                    self.num_points, 
                    self.oversample_ratio, 
                    self.importance_sample_ratio
                )
                
                # Get GT labels for the sampled points
                mask_point_targets = torch.gather(dn_masks_valid.view(dn_masks_valid.shape[0], -1), 
                                                dim=1, index=point_indices)
            
            # Get predicted values at the sampled points
            mask_point_preds = point_sample_3d(
                mask_preds_valid.unsqueeze(1), point_coords[..., [2, 1, 0]], align_corners=self.align_corners).squeeze(1)
            
            # Calculate Dice loss
            num_total_mask_weights = reduce_mean(dn_mask_weights_valid.sum())
            loss_dice_dn = self.loss_dice(mask_point_preds, mask_point_targets,
                                        weight=dn_mask_weights_valid, avg_factor=num_total_mask_weights)
            
            # Calculate Mask loss
            mask_point_preds_flat = mask_point_preds.reshape(-1)
            mask_point_targets_flat = mask_point_targets.reshape(-1) 
            mask_point_weights = dn_mask_weights_valid.view(-1, 1).repeat(1, self.num_points).reshape(-1)
            
            num_total_mask_point_weights = reduce_mean(mask_point_weights.sum())
            loss_mask_dn = self.loss_mask(
                mask_point_preds_flat,
                mask_point_targets_flat,
                weight=mask_point_weights,
                avg_factor=num_total_mask_point_weights)
        
        return loss_cls_dn, loss_mask_dn, loss_dice_dn

    def forward_head(self, 
            decoder_out,
            voxel_features_flat,
            ori_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, w, h).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: (cls_pred, mask_pred)
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(decoder_out)
        # Use all voxel features for prediction, voxel_features_flat shape is [B, N, C]
        nonempty_pred = torch.einsum('bqc, bnc -> bqn', mask_embed, voxel_features_flat)
        mask_3d = rearrange(nonempty_pred, 'b q (h w z) -> b q h w z', h=ori_size[0], w=ori_size[1], z=ori_size[2])
        
        # The PEM version no longer constructs/uses the cross-attention attn_mask
        return cls_pred, mask_3d

    def preprocess_gt(self, gt_occ, img_metas):
        
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, w, h).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, w, h).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, w, h).
        """
        
        num_class_list = [self.num_occupancy_classes] * len(img_metas)
        targets = multi_apply(preprocess_occupancy_gt, gt_occ, num_class_list, img_metas)
        
        labels, masks = targets
        return labels, masks
    
    def forward_train(self,
            voxel_feats,
            img_metas,
            gt_occ,
            **kwargs,
        ):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, w, h).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, w, h).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        # reset the sampling weights
        self.get_sampling_weights()
        
        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_occ, img_metas)
        
        # forward - pass gt_labels and gt_masks to the forward function to support DN
        all_cls_scores, all_mask_preds, dn_meta = self.forward(
            voxel_feats, img_metas, gt_labels_list=gt_labels, gt_masks_list=gt_masks
        )

        # loss (including DN loss and matching loss)
        loss_dict = {}
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas, dn_meta)
        loss_dict.update(losses)
        return loss_dict

    def forward(self, 
            voxel_feats,
            img_metas,
            gt_labels_list=None,
            gt_masks_list=None,
            **kwargs,
        ):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 5D-tensor (B, C, X, Y, Z).
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        """
        
        batch_size = len(img_metas)
        mask_features = voxel_feats[0]  # B, C, W, H, D
        B, _, W, H, D = mask_features.shape
        multi_scale_memorys = voxel_feats[:0:-1]

        # Remove the coarse branch and use the highest resolution features directly

        # Directly use all voxels of the highest resolution features, no sampling needed
        voxel_features_flat = rearrange(mask_features, 'b c h w z -> b (h w z) c')


        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            ''' with flatten features '''
            # projection for input features
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            ''' with level embeddings '''
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            ''' with positional encodings '''
            # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
            mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-3:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        
        tgt_mask = None  # Default self-attention mask is None
        dn_meta = {}     # DN metadata, empty dict during inference
        
        # If in training mode and GT is provided, execute DN logic
        dn_gt_masks = None  # Initialize DN GT masks
        if self.training and gt_labels_list is not None and gt_masks_list is not None:
            dn_result = self._prepare_for_dn(gt_labels_list, gt_masks_list)
            if dn_result is not None:
                dn_queries, tgt_mask, dn_meta, dn_gt_masks = dn_result
                
                # Concatenate DN queries and original queries: DN queries first, then original queries
                query_feat = torch.cat([dn_queries, query_feat], dim=0)
                
                # query_embed also needs to be concatenated accordingly, using DN queries as positional embeddings for the DN part
                dn_embed = dn_queries.clone().detach()
                query_embed = torch.cat([dn_embed, query_embed], dim=0)
        
        ''' directly decode the learnable queries, as simple proposals '''
        cls_pred_list = []
        mask_pred_list = []
        # In the training phase, supervision is required for all decoder layers (including the initial one);
        # in the inference phase, only the final layer is kept to avoid redundant large-volume mask prediction calculations
        if self.training:
            cls_pred, mask_pred = self.forward_head(
                query_feat,
                voxel_features_flat,
                voxel_feats[0].shape[-3:]
            )
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            
            # Generate cross-attention mask for the current decoder layer
            cross_attn_mask = None
            if self.training and dn_result is not None and dn_gt_masks is not None and i < self.dn_loss_layers:
                # Get the spatial dimensions of the current feature map
                # target_size = decoder_inputs[level_idx].shape[0]  # (X*Y*Z, batch_size, C) -> X*Y*Z
                current_feat_shape = multi_scale_memorys[level_idx].shape[-3:]  # (X, Y, Z)
                feat_w, feat_h, feat_d = current_feat_shape

                # Get the dimensions of the original GT masks
                # dn_gt_masks: (batch_size, pad_size, w, h, D)
                gt_w, gt_h, gt_d = dn_gt_masks.shape[-3:]
                
                # Calculate kernel size and stride for pooling
                # The dimension order for PyTorch's Conv3D/MaxPool3D is (d, h, w)
                # but (w, h, d) also works, as long as the order of kernel_size and stride corresponds
                stride_w = gt_w // feat_w
                stride_h = gt_h // feat_h
                stride_d = gt_d // feat_d

                # Use 3D max pooling for downsampling
                downsampled_dn_masks = F.max_pool3d(
                    dn_gt_masks.float(),
                    kernel_size=(stride_w, stride_h, stride_d),
                    stride=(stride_w, stride_h, stride_d)
                )
                # Since the max_pool result is 0 or 1, using < 0.5 for boolean conversion is robust
                
                # Flatten and invert the logic: True means masked (not attended), False means can be attended
                dn_mask_flat = (downsampled_dn_masks < 0.5).flatten(2)  # (batch_size, pad_size, X*Y*Z)
                
                # Create no mask for matching queries (all False, meaning all positions can be attended)
                num_keys = dn_mask_flat.shape[-1]
                matching_mask_flat = torch.zeros(
                    (batch_size, self.num_queries, num_keys), 
                    dtype=torch.bool, 
                    device=query_feat.device
                )
                
                # Concatenate to get the complete cross-attention mask
                cross_attn_mask = torch.cat([dn_mask_flat, matching_mask_flat], dim=1)  # (batch_size, total_queries, X*Y*Z)
            
            # PEM decoder layer: pass self-attention mask (tgt_mask) and cross-attention mask
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                self_attn_mask=tgt_mask,
                query_key_padding_mask=None,
                cross_attn_mask=cross_attn_mask,
            )
            # Training: output layer by layer; Inference: output only once at the last layer
            if self.training or i == self.num_transformer_decoder_layers - 1:
                cls_pred, mask_pred = self.forward_head(
                    query_feat,
                    voxel_features_flat,
                    voxel_feats[0].shape[-3:]
                )
                cls_pred_list.append(cls_pred)
                mask_pred_list.append(mask_pred)
        
        '''
        Returns:
            tuple: A tuple contains three elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
            - dn_meta (dict): DN metadata dictionary, including pad_size, etc. Empty during inference.
        '''
        
        return cls_pred_list, mask_pred_list, dn_meta

    def format_results(self, mask_cls_results, mask_pred_results):
        mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        output_voxels = torch.einsum("bqc, bqxyz->bcxyz", mask_cls, mask_pred)
        
        return output_voxels

    def simple_test(self, 
            voxel_feats,
            img_metas,
            **kwargs,
        ):
        all_cls_scores, all_mask_preds, _ = self.forward(voxel_feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=self.final_occ_size,
            mode='trilinear',
            align_corners=self.align_corners,
        )

        output_voxels = self.format_results(mask_cls_results, mask_pred_results)
        res = {
            'output_voxels': [output_voxels],
            'output_voxel_refine': None,
            'output_points': None,
        }

        return res
    
    # The coarse voxel loss branch has been removed