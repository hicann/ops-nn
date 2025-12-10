#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

def write_file(inputs, outputs):
    for input_name in inputs:
        inputs[input_name].tofile(os.path.join(input_name+".bin"))
    for output_name in outputs:
        outputs[output_name].tofile(os.path.join(output_name+".bin"))

def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    B, S, H, C = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    split_sizes = [Hi * Wi for (Hi, Wi) in value_spatial_shapes]
    v_per_level = torch.split(value, split_sizes, dim=1)
    grids = sampling_locations.mul(2).add(-1)

    sampled_per_level = []
    for lvl, (Hi, Wi) in enumerate(value_spatial_shapes):
        v = v_per_level[lvl]                                  # (B, Hi*Wi, H, C)
        v = v.permute(0, 2, 3, 1).reshape(B * H, C, Hi, Wi)   # (B*H, C, Hi, Wi)
        g = grids[:, :, :, lvl]                               # (B, Q, H, P, 2)
        g = g.permute(0, 2, 1, 3, 4).reshape(B * H, Q, P, 2)  # (B*H, Q, P, 2)
        sampled = F.grid_sample(
            v, g, mode="bilinear", padding_mode="zeros", align_corners=False
        )                                                     # (B*H, C, Q, P)
        sampled_per_level.append(sampled)
    sampled_all = torch.cat(sampled_per_level, dim=-1)
    attn = attention_weights.transpose(1, 2)                  # (B, H, Q, L, P)
    attn = attn.reshape(B * H, 1, Q, L * P)
    out = (sampled_all * attn).sum(dim=-1)
    out = out.reshape(B, H * C, Q).transpose(1, 2).contiguous()
    return out

if __name__ == "__main__":
    bs = int(sys.argv[1])
    num_heads = int(sys.argv[2])
    channels = int(sys.argv[3])
    num_levels = int(sys.argv[4])
    num_points = int(sys.argv[5])
    num_queries = int(sys.argv[6])

    cpu_shapes = torch.tensor([6, 4] * num_levels).reshape(num_levels, 2).int()
    cpu_shapes_numpy = cpu_shapes.numpy()
    num_keys = sum((H * W).item() for H, W in cpu_shapes)

    cpu_value = torch.rand(bs, num_keys, num_heads, channels) * 0.01
    cpu_value_numpy = cpu_value.float().numpy()
    cpu_sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2).float()
    cpu_sampling_locations_numpy = cpu_sampling_locations.numpy()
    cpu_attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
    cpu_attention_weights_numpy = cpu_attention_weights.float().numpy()

    cpu_offset = torch.cat((cpu_shapes.new_zeros((1, )), cpu_shapes.prod(1).cumsum(0)[:-1]))
    cpu_offset_nunmpy = cpu_offset.int().numpy()

    output_npu = multi_scale_deformable_attn_pytorch(cpu_value, cpu_shapes, cpu_sampling_locations, cpu_attention_weights).numpy()


    inputs_npu = {"value":cpu_value_numpy ,"value_spatial_shape": cpu_shapes_numpy,
                   "level_start_index":cpu_offset_nunmpy, "sample_loc": cpu_sampling_locations_numpy,
                  "attention_weight":cpu_attention_weights_numpy}
    outputs_npu = {"output":output_npu}
    write_file(inputs_npu, outputs_npu)