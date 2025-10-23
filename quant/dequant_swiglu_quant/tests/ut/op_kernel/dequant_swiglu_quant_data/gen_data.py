# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import torch
import tensorflow as tf


def do_golden(x, weight_scale, act_scale, bias, quant_scale, quant_offset,
              group_num, act_left, quant_mode):
    x = x.to(torch.float32)
    if bias is not None:
        x = torch.add(x, bias.to(torch.float32))
    x = torch.mul(x, weight_scale.to(torch.float32))
    x = torch.mul(x, act_scale.to(torch.float32))
    out = torch.chunk(x, 2, dim=-1)

    if act_left:
        x = out[0]
        y = out[1]
    else:
        x = out[1]
        y = out[0]

    x = torch.nn.functional.silu(x)
    x = torch.mul(x, y)

    scale_out = torch.zeros(x.shape[0], dtype=torch.float32)

    if quant_mode == "static":
        if quant_offset is not None:
            out = torch.add(x, quant_offset.to(torch.float32))
        else:
            out = x.clone()
    else:
        absx = torch.abs(x)
        maxx = torch.amax(absx, dim=-1)
        scale_out = maxx / 127
        max_values = 127 / maxx
        out = torch.mul(x, max_values.unsqueeze(-1))

    out = torch.clamp(out, -128, 127)
    out = torch.round(out)
    out = out.to(torch.int8)
    return out, scale_out


def gen_golden_data_simple(params_list):
    x_shape = params_list["x_shape"]
    data_x = torch.randint(-10, 10, x_shape, dtype=torch.int32)
    data_weight = torch.randn(x_shape[-1], dtype=torch.float32)
    data_act_scale = torch.randn((x_shape[0], 1), dtype=torch.float32)

    # Bias
    data_bias = None
    if params_list["bias_dtype"] is not None:
        if params_list["bias_dtype"] is torch.int32:
            data_bias = torch.randint(-10, 10, x_shape[-1], dtype=params_list["bias_dtype"])
        else:
            data_bias = torch.randn(x_shape[-1], dtype=params_list["bias_dtype"])

    data_quant_scale = torch.randn((1, x_shape[-1] // 2), dtype=params_list["quant_scale_dtype"])
    data_quant_offset = None
    data_group_index = torch.tensor([x_shape[0]], dtype=torch.int64)

    out_list = do_golden(
        data_x, data_weight, data_act_scale, data_bias,
        data_quant_scale, data_quant_offset, data_group_index,
        False, "dynamic"
    )

    # Save input data
    data_x_np = data_x.cpu().numpy()
    if params_list["x_dtype"] == torch.bfloat16:
        data_x_np = data_x_np.astype(tf.bfloat16.as_numpy_dtype)
    data_x_np.tofile("./input_x.bin")

    data_weight.cpu().numpy().tofile("./input_weight.bin")
    data_act_scale.cpu().numpy().tofile("./input_act_scale.bin")
    if data_bias is not None:
        data_bias.cpu().numpy().tofile("./input_bias.bin")
    data_quant_scale.cpu().numpy().tofile("./input_quant_scale.bin")
    data_group_index.cpu().numpy().tofile("./input_group_index.bin")

    # Save output data
    out_list[0].cpu().numpy().tofile("./output_y.bin")
    out_list[1].cpu().numpy().tofile("./output_scale.bin")


params_info = {
    "test_dequant_swiglu_quant_1":  {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float16,  "bias_dtype": None},
    "test_dequant_swiglu_quant_2":  {"x_shape": [128, 2048], "x_dtype": torch.bfloat16,"quant_scale_dtype": torch.float32,  "bias_dtype": None},
    "test_dequant_swiglu_quant_3":  {"x_shape": [128, 2048], "x_dtype": torch.bfloat16,"quant_scale_dtype": torch.bfloat16, "bias_dtype": None},
    "test_dequant_swiglu_quant_4":  {"x_shape": [128, 2048], "x_dtype": torch.bfloat16,"quant_scale_dtype": torch.float16,  "bias_dtype": None},
    "test_dequant_swiglu_quant_8":  {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float32,  "bias_dtype": torch.int32},
    "test_dequant_swiglu_quant_9":  {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.bfloat16, "bias_dtype": torch.int32},
    "test_dequant_swiglu_quant_10": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float16,  "bias_dtype": torch.int32},
    "test_dequant_swiglu_quant_11": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float32,  "bias_dtype": torch.float32},
    "test_dequant_swiglu_quant_12": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.bfloat16, "bias_dtype": torch.float32},
    "test_dequant_swiglu_quant_13": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float16,  "bias_dtype": torch.float32},
    "test_dequant_swiglu_quant_14": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float32,  "bias_dtype": torch.float16},
    "test_dequant_swiglu_quant_15": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.bfloat16, "bias_dtype": torch.float16},
    "test_dequant_swiglu_quant_16": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float16,  "bias_dtype": torch.float16},
    "test_dequant_swiglu_quant_17": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float32,  "bias_dtype": torch.bfloat16},
    "test_dequant_swiglu_quant_18": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.bfloat16, "bias_dtype": torch.bfloat16},
    "test_dequant_swiglu_quant_19": {"x_shape": [128, 2048], "x_dtype": torch.int32,   "quant_scale_dtype": torch.float16,  "bias_dtype": torch.bfloat16},
}


if __name__ == "__main__":
    test_name = sys.argv[1]
    if test_name not in params_info:
        raise ValueError(f"Unknown test name: {test_name}")
    params_list = params_info[test_name]
    gen_golden_data_simple(params_list)
