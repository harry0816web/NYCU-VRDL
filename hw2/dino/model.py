"""
DINO model: backbone, position encoding, deformable transformer,
denoising components, matcher, criterion, postprocess.
All model components in one file (no segmentation).
Reference: Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes" (ICLR 2023).
"""
import copy
import math
import random
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.boxes import nms
from scipy.optimize import linear_sum_assignment

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from ops.modules import MSDeformAttn


# ===========================================================================
# Utilities
# ===========================================================================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor, num_pos_feats=128):
    scale = 2 * math.pi
    dim_t = torch.arange(
        num_pos_feats,
        dtype=torch.float32,
        device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(
            "Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_encoder_output_proposals(
        memory,
        memory_padding_mask,
        spatial_shapes,
        learnedwh=None):
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(
            _cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            indexing='ij')
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat(
            [valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (
        output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(
        ~output_proposals_valid, float('inf'))
    output_memory = memory
    output_memory = output_memory.masked_fill(
        memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(
        ~output_proposals_valid, float(0))
    return output_memory, output_proposals


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for _ in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ===========================================================================
# Positional Encoding (Sine with separate H/W temperatures)
# ===========================================================================

class PositionEmbeddingSineHW(nn.Module):
    def __init__(
            self,
            num_pos_feats=64,
            temperatureH=10000,
            temperatureW=10000,
            normalize=False,
            scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_tx = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx
        dim_ty = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ===========================================================================
# Backbone (multi-scale ResNet with FrozenBN)
# ===========================================================================

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + 1e-5).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 num_channels: list, return_interm_indices: list):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or (
                'layer2' not in name
                and 'layer3' not in name
                and 'layer4' not in name
            ):
                parameter.requires_grad_(False)
        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers["layer{}".format(5 -
                                           len(return_interm_indices) +
                                           idx)] = "{}".format(layer_index)
        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(),
                                 size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, dilation: bool,
                 return_interm_indices: list, batch_norm=FrozenBatchNorm2d):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights="IMAGENET1K_V1", norm_layer=batch_norm)
        assert name in ('resnet50', 'resnet101')
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices):]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


# ===========================================================================
# Deformable Transformer
# ===========================================================================

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes,
                level_start_index, key_padding_mask=None):
        src2 = self.self_attn(
            self.with_pos_embed(
                src,
                pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 decoder_sa_type='sa', module_seq=None):
        super().__init__()
        if module_seq is None:
            module_seq = ['sa', 'ca', 'ffn']
        self.module_seq = module_seq

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(
            activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.decoder_sa_type = decoder_sa_type

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self, tgt, tgt_query_pos=None, tgt_reference_points=None,
                   memory=None, memory_key_padding_mask=None,
                   memory_level_start_index=None, memory_spatial_shapes=None,
                   memory_pos=None, self_attn_mask=None, **kwargs):
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        return tgt

    def forward_ca(self, tgt, tgt_query_pos=None, tgt_reference_points=None,
                   memory=None, memory_key_padding_mask=None,
                   memory_level_start_index=None, memory_spatial_shapes=None,
                   memory_pos=None, **kwargs):
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1), memory_spatial_shapes,
            memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        return tgt

    def forward(self, tgt, tgt_query_pos=None, tgt_query_sine_embed=None,
                tgt_key_padding_mask=None, tgt_reference_points=None,
                memory=None, memory_key_padding_mask=None,
                memory_level_start_index=None, memory_spatial_shapes=None,
                memory_pos=None, self_attn_mask=None, cross_attn_mask=None):
        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(
                    tgt,
                    tgt_query_pos,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_pos)
            elif funcname == 'sa':
                tgt = self.forward_sa(
                    tgt,
                    tgt_query_pos,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_pos,
                    self_attn_mask)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256,
                 num_queries=300, deformable_encoder=False,
                 enc_layer_share=False, two_stage_type='no'):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(
                encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.two_stage_type = two_stage_type

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, pos, spatial_shapes, level_start_index,
                valid_ratios, key_padding_mask,
                ref_token_index=None, ref_token_coord=None):
        output = src
        if self.num_layers > 0 and self.deformable_encoder:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device)
        intermediate_output = []
        intermediate_ref = []
        for layer_id, layer in enumerate(self.layers):
            if self.deformable_encoder:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                output = layer(
                    src=output.transpose(
                        0, 1), pos=pos.transpose(
                        0, 1), key_padding_mask=key_padding_mask).transpose(
                    0, 1)
        if self.norm is not None:
            output = self.norm(output)
        return output, None, None


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False, d_model=256, query_dim=4,
                 modulate_hw_attn=False, num_feature_levels=1,
                 deformable_decoder=False, decoder_query_perturber=None,
                 dec_layer_number=None, rm_dec_query_scale=False,
                 dec_layer_share=False, use_detached_boxes_dec_out=False):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(
                decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.d_model = d_model
        self.deformable_decoder = deformable_decoder

        self.ref_point_head = MLP(
            query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None
        self.modulate_hw_attn = modulate_hw_attn

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None
        self.dec_layer_number = dec_layer_number
        self.rm_detach = None

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            pos=None,
            refpoints_unsigmoid=None,
            level_start_index=None,
            spatial_shapes=None,
            valid_ratios=None):
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = (
                        reference_points[:, :, None]
                        * torch.cat([valid_ratios, valid_ratios], -1)[None, :])
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = (
                        reference_points[:, :, None] * valid_ratios[None, :]
                    )
                query_sine_embed = gen_sineembed_for_position(
                    reference_points_input[:, :, 0, :], self.d_model // 2)
            else:
                query_sine_embed = gen_sineembed_for_position(
                    reference_points, self.d_model // 2)
                reference_points_input = None

            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(
                output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = (
                    query_sine_embed[..., :self.d_model]
                    * self.query_pos_sine_scale(output)
                )

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask)

            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                if self.rm_detach and 'dec' in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_queries=300,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0, activation="relu",
                 normalize_before=False, return_intermediate_dec=True,
                 query_dim=4, num_feature_levels=1,
                 enc_n_points=4, dec_n_points=4,
                 two_stage_type='no', two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 two_stage_pat_embed=0, two_stage_add_query_num=0,
                 random_refpoints_xy=False, dec_layer_number=None,
                 decoder_sa_type='sa', module_seq=None,
                 embed_init_tgt=False, use_detached_boxes_dec_out=False):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation,
            num_feature_levels, nhead, enc_n_points)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm,
            d_model=d_model, num_queries=num_queries,
            deformable_encoder=True, two_stage_type=two_stage_type)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation,
            num_feature_levels, nhead, dec_n_points,
            decoder_sa_type=decoder_sa_type,
            module_seq=module_seq)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model, query_dim=query_dim,
            modulate_hw_attn=True, num_feature_levels=num_feature_levels,
            deformable_decoder=True,
            dec_layer_number=dec_layer_number,
            rm_dec_query_scale=True,
            use_detached_boxes_dec_out=use_detached_boxes_dec_out)

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(
                    torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = True
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (
                two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        if two_stage_type == 'standard':
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None
            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(
                    self.two_stage_add_query_num, d_model)

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        self.dec_layer_number = dec_layer_number

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)
        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight,
                              math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(
            self,
            srcs,
            masks,
            refpoint_embed,
            pos_embeds,
            tgt,
            attn_mask=None):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(
                zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + \
                    self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
        )

        if self.two_stage_type == 'standard':
            input_hw = (
                self.two_stage_wh_embedding.weight[0]
                if self.two_stage_learn_wh
                else None
            )
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, input_hw)
            output_memory = self.enc_output_norm(
                self.enc_output(output_memory))

            enc_outputs_class_unselected = self.enc_out_class_embed(
                output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(
                output_memory) + output_proposals
            topk = self.num_queries
            topk_proposals = torch.topk(
                enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()

            tgt_undetach = torch.gather(
                output_memory, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(
                    1, bs, 1).transpose(0, 1)
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat(
                    [refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(
                1, bs, 1).transpose(0, 1)
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(
                1, bs, 1).transpose(0, 1)
            if refpoint_embed is not None:
                refpoint_embed = torch.cat(
                    [refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_
            init_box_proposal = refpoint_embed_.sigmoid()
        else:
            raise NotImplementedError(
                "unknown two_stage_type {}".format(
                    self.two_stage_type))

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1), memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index, spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios, tgt_mask=attn_mask)

        if self.two_stage_type == 'standard':
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None

        return hs, references, hs_enc, ref_enc, init_box_proposal


# ===========================================================================
# DN (Denoising) Components
# ===========================================================================

def prepare_for_cdn(
        dn_args,
        training,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc):
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i)
                               for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(
                p < (label_noise_ratio * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)

        positive_idx = torch.tensor(
            range(
                len(boxes))).long().cuda().unsqueeze(0).repeat(
            dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) *
                         len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2,
                                           dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + \
                torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (
                known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat(
                [map_known_indice + single_pad * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(),
                               map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(),
                              map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          :single_pad * 2 * i] = True

        dn_meta = {'pad_size': pad_size, 'num_dn_group': dn_number}
    else:
        input_query_label = input_query_bbox = attn_mask = dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(
        outputs_class,
        outputs_coord,
        dn_meta,
        aux_loss,
        _set_aux_loss):
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1],
               'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(
                output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


# ===========================================================================
# DINO Model
# ===========================================================================

class DINO(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, num_feature_levels=1, nheads=8,
                 two_stage_type='no',
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 dn_number=100, dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5, dn_labelbook_size=100):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim)))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            hidden_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1),
                        nn.GroupNorm(
                            32,
                            hidden_dim)))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(
                nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim))])

        self.backbone = backbone
        self.aux_loss = aux_loss

        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for _ in range(
                    transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for _ in range(
                    transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for _ in range(
                    transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed) for _ in range(
                    transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        self.two_stage_type = two_stage_type
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(
                    _bbox_embed)
            if two_stage_class_embed_share:
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(
                    _class_embed)

        self.label_embedding = None
        for layer in self.transformer.decoder.layers:
            layer.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor, targets: List = None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for level_idx, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[level_idx](src))
            masks.append(mask)
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for level_idx in range(_len_srcs, self.num_feature_levels):
                if level_idx == _len_srcs:
                    src = self.input_proj[level_idx](features[-1].tensors)
                else:
                    src = self.input_proj[level_idx](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(),
                                     size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = \
                prepare_for_cdn(
                    dn_args=(targets, self.dn_number, self.dn_label_noise_ratio,
                             self.dn_box_noise_scale),
                    training=self.training, num_queries=self.num_queries,
                    num_classes=self.num_classes, hidden_dim=self.hidden_dim,
                    label_enc=self.label_enc)
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = \
            self.transformer(srcs, masks, input_query_bbox, poss,
                             input_query_label, attn_mask)
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + \
                inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack([layer_cls_embed(layer_hs)
                                     for layer_cls_embed, layer_hs
                                     in zip(self.class_embed, hs)])

        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta, self.aux_loss, self._set_aux_loss)

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord_list)

        if hs_enc is not None:
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class,
                                     'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {
                'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        out['dn_meta'] = dn_meta
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


# ===========================================================================
# Loss: Focal Loss, Matcher, SetCriterion
# ===========================================================================

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


class HungarianMatcher(nn.Module):
    def __init__(
            self,
            cost_class=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            focal_alpha=0.25):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * \
            (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * \
            ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(out_bbox),
            box_ops.box_cxcywh_to_xyxy(tgt_bbox))

        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class
             + self.cost_giou * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - \
                accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {'cardinality_error': card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i]
                                  for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}

        dn_meta = outputs['dn_meta']
        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(
                dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0,
                                     len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        (torch.tensor(
                            range(scalar)) *
                            single_pad).long().cuda() .unsqueeze(1) +
                        t)
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(
                    loss, output_known_lbs_bboxes, targets,
                    dn_pos_idx, num_boxes * scalar, **kwargs))
            l_dict = {k + '_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = {
                'loss_bbox_dn': torch.as_tensor(0.).to(device),
                'loss_giou_dn': torch.as_tensor(0.).to(device),
                'loss_ce_dn': torch.as_tensor(0.).to(device),
                'cardinality_error_dn': torch.as_tensor(0.).to(device),
            }
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    indices,
                    num_boxes))

        if 'aux_outputs' in outputs:
            for idx_aux, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets,
                                           indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx_aux}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx_aux]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}
                        l_dict.update(self.get_loss(
                            loss, aux_outputs_known, targets,
                            dn_pos_idx, num_boxes * scalar, **kwargs))
                    l_dict = {
                        k + f'_dn_{idx_aux}': v for k,
                        v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = {
                        'loss_bbox_dn': torch.as_tensor(0.).to(device),
                        'loss_giou_dn': torch.as_tensor(0.).to(device),
                        'loss_ce_dn': torch.as_tensor(0.).to(device),
                        'cardinality_error_dn': torch.as_tensor(0.).to(device),
                    }
                    l_dict = {k + f'_{idx_aux}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets,
                                       indices, num_boxes, **kwargs)
                l_dict = {k + '_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups
        return output_known_lbs_bboxes, single_pad, num_dn_groups


# ===========================================================================
# PostProcess (sigmoid-based top-k)
# ===========================================================================

class PostProcess(nn.Module):
    def __init__(self, num_select=300, nms_iou_threshold=-1):
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold)
                            for b, s in zip(boxes, scores)]
            results = [
                {'scores': score[idx], 'labels': label[idx], 'boxes': box[idx]}
                for score, label, box, idx in zip(scores, labels, boxes, item_indices)
            ]
        else:
            results = [{'scores': score, 'labels': label, 'boxes': box}
                       for score, label, box in zip(scores, labels, boxes)]
        return results


# ===========================================================================
# Build
# ===========================================================================

def build_backbone_and_pos(config):
    N_steps = config['hidden_dim'] // 2
    position_embedding = PositionEmbeddingSineHW(
        N_steps,
        temperatureH=config.get('pe_temperatureH', 20),
        temperatureW=config.get('pe_temperatureW', 20),
        normalize=True)

    train_backbone = config['lr_backbone'] > 0
    return_interm_indices = config.get('return_interm_indices', [1, 2, 3])
    backbone_net = Backbone(config['backbone'], train_backbone,
                            config.get('dilation', False),
                            return_interm_indices,
                            batch_norm=FrozenBatchNorm2d)
    model = Joiner(backbone_net, position_embedding)
    model.num_channels = backbone_net.num_channels
    return model


def build_model(config):
    num_classes = config['num_classes']
    device = torch.device(config.get('device', 'cuda'))

    backbone = build_backbone_and_pos(config)

    transformer = DeformableTransformer(
        d_model=config['hidden_dim'],
        nhead=config['nheads'],
        num_queries=config['num_queries'],
        num_encoder_layers=config['enc_layers'],
        num_decoder_layers=config['dec_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config.get('dropout', 0.0),
        activation=config.get('transformer_activation', 'relu'),
        normalize_before=config.get('pre_norm', False),
        return_intermediate_dec=True,
        query_dim=4,
        num_feature_levels=config.get('num_feature_levels', 4),
        enc_n_points=config.get('enc_n_points', 4),
        dec_n_points=config.get('dec_n_points', 4),
        two_stage_type=config.get('two_stage_type', 'standard'),
        two_stage_learn_wh=config.get('two_stage_learn_wh', False),
        two_stage_keep_all_tokens=config.get('two_stage_keep_all_tokens', False),
        two_stage_pat_embed=config.get('two_stage_pat_embed', 0),
        two_stage_add_query_num=config.get('two_stage_add_query_num', 0),
        random_refpoints_xy=config.get('random_refpoints_xy', False),
        dec_layer_number=config.get('dec_layer_number', None),
        decoder_sa_type=config.get('decoder_sa_type', 'sa'),
        module_seq=config.get('decoder_module_seq', ['sa', 'ca', 'ffn']),
        embed_init_tgt=config.get('embed_init_tgt', True),
        use_detached_boxes_dec_out=config.get('use_detached_boxes_dec_out', False),
    )

    dn_labelbook_size = config.get('dn_labelbook_size', num_classes)

    model = DINO(
        backbone, transformer,
        num_classes=num_classes,
        num_queries=config['num_queries'],
        aux_loss=config.get('aux_loss', True),
        num_feature_levels=config.get('num_feature_levels', 4),
        nheads=config['nheads'],
        two_stage_type=config.get('two_stage_type', 'standard'),
        dec_pred_class_embed_share=config.get('dec_pred_class_embed_share', True),
        dec_pred_bbox_embed_share=config.get('dec_pred_bbox_embed_share', True),
        two_stage_class_embed_share=config.get('two_stage_class_embed_share', False),
        two_stage_bbox_embed_share=config.get('two_stage_bbox_embed_share', False),
        dn_number=config.get('dn_number', 100) if config.get('use_dn', True) else 0,
        dn_box_noise_scale=config.get('dn_box_noise_scale', 0.4),
        dn_label_noise_ratio=config.get('dn_label_noise_ratio', 0.5),
        dn_labelbook_size=dn_labelbook_size,
    )

    matcher = HungarianMatcher(
        cost_class=config.get('set_cost_class', 2.0),
        cost_bbox=config.get('set_cost_bbox', 5.0),
        cost_giou=config.get('set_cost_giou', 2.0),
        focal_alpha=config.get('focal_alpha', 0.25),
    )

    weight_dict = {
        'loss_ce': config.get('cls_loss_coef', 1.0),
        'loss_bbox': config.get('bbox_loss_coef', 5.0),
        'loss_giou': config.get('giou_loss_coef', 2.0),
    }
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    if config.get('use_dn', True):
        weight_dict['loss_ce_dn'] = config.get('cls_loss_coef', 1.0)
        weight_dict['loss_bbox_dn'] = config.get('bbox_loss_coef', 5.0)
        weight_dict['loss_giou_dn'] = config.get('giou_loss_coef', 2.0)
    clean_weight_dict = copy.deepcopy(weight_dict)

    if config.get('aux_loss', True):
        aux_weight_dict = {}
        for i in range(config['dec_layers'] - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if config.get('two_stage_type', 'standard') != 'no':
        interm_weight_dict = {}
        no_interm_box_loss = config.get('no_interm_box_loss', False)
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        interm_loss_coef = config.get('interm_loss_coef', 1.0)
        interm_weight_dict.update({
            k + '_interm': v * interm_loss_coef * _coeff_weight_dict[k]
            for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(
        num_classes, matcher=matcher, weight_dict=weight_dict,
        focal_alpha=config.get('focal_alpha', 0.25), losses=losses)
    criterion.to(device)

    postprocessors = {
        'bbox': PostProcess(
            num_select=config.get('num_select', 300),
            nms_iou_threshold=config.get('nms_iou_threshold', -1))}

    return model, criterion, postprocessors
