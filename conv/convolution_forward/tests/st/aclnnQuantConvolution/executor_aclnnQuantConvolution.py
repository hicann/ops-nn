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
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import torch
import numpy as np

from atk.common.log import Logger

logging = Logger().get_logger()

from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("aclnn_quant_convolution")
class QuantAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        import ctypes

        input_args, output_packages = super().init_by_input_data(input_data)
        input_args[11] = ctypes.c_int32(input_args[11].value)

        return input_args, output_packages

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name == "bias" or name == "scale":
            return AclFormat.ACL_FORMAT_ND
        return AclFormat.ACL_FORMAT_NCDHW

input_promote_type_dict = {
    'float32': {'float32': 'float32', 'float16': 'float32', 'double': 'double', 'bfloat16': 'float32', 'int8': 'float32', 'uint8': 'float32',
             'int16': 'float32', 'int32': 'float32', 'int64': 'float32', 'bool': 'float32', 'complex32': 'complex64',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'float16': {'float32': 'float32', 'float16': 'float16', 'double': 'double', 'bfloat16': 'float32', 'int8': 'float16', 'uint8': 'float16',
             'int16': 'float16', 'int32': 'float16', 'int64': 'float16', 'bool': 'float16', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'double': {'float32': 'double', 'float16': 'double', 'double': 'double', 'bfloat16': 'double', 'int8': 'double', 'uint8': 'double',
             'int16': 'double', 'int32': 'double', 'int64': 'double', 'bool': 'double', 'complex32': 'complex64',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'bfloat16': {'float32': 'float32', 'float16': 'float32', 'double': 'double', 'bfloat16': 'bfloat16', 'int8': 'bfloat16', 'uint8': 'bfloat16',
             'int16': 'bfloat16', 'int32': 'bfloat16', 'int64': 'bfloat16', 'bool': 'bfloat16', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'int8': {'float32': 'float32', 'float16': 'float16', 'double': 'double', 'bfloat16': 'bfloat16', 'int8': 'int8', 'uint8': 'int16',
             'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'int8', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'uint8': {'float32': 'float32', 'float16': 'float16', 'double': 'double', 'bfloat16': 'bfloat16', 'int8': 'int16', 'uint8': 'uint8',
             'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'uint8', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'int16': {'float32': 'float32', 'float16': 'float16', 'double': 'double', 'bfloat16': 'bfloat16', 'int8': 'int16', 'uint8': 'int16',
             'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'int16', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'uint16': {'uint16': 'uint16'},
    'int32': {'float32': 'float32', 'float16': 'float16', 'double': 'double', 'bfloat16': 'bfloat16', 'int8': 'int32', 'uint8': 'int32',
             'int16': 'int32', 'int32': 'int32', 'int64': 'int64', 'bool': 'int32', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'uint32': {'uint32': 'uint32'},
    'int64': {'float32': 'float32', 'float16': 'float16', 'double': 'double', 'bfloat16': 'bfloat16', 'int8': 'int64', 'uint8': 'int64',
             'int16': 'int64', 'int32': 'int64', 'int64': 'int64', 'bool': 'int64', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'uint64': {'uint64': 'uint64'},
    'bool': {'float32': 'float32', 'float16': 'float16', 'double': 'double', 'bfloat16': 'bfloat16', 'int8': 'int8', 'uint8': 'uint8',
             'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'bool': 'bool', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'complex32': {'float32': 'complex64', 'float16': 'complex32', 'double': 'complex64', 'bfloat16': 'complex32', 'int8': 'complex32', 'uint8': 'complex32',
             'int16': 'complex32', 'int32': 'complex32', 'int64': 'complex32', 'bool': 'complex32', 'complex32': 'complex32',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'complex64': {'float32': 'complex64', 'float16': 'complex64', 'double': 'complex64', 'bfloat16': 'complex64', 'int8': 'complex64', 'uint8': 'complex64',
             'int16': 'complex64', 'int32': 'complex64', 'int64': 'complex64', 'bool': 'complex64', 'complex32': 'complex64',
             'complex64': 'complex64', 'complex128': 'complex128'},
    'complex128': {'float32': 'complex128', 'float16': 'complex128', 'double': 'complex128', 'bfloat16': 'complex128', 'int8': 'complex128', 'uint8': 'complex128',
             'int16': 'complex128', 'int32': 'complex128', 'int64': 'complex128', 'bool': 'complex128', 'complex32': 'complex128',
             'complex64': 'complex128', 'complex128': 'complex128'},
}

dtype_dict = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8"
}

torch_dtype_dict = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8
}

cube_math_type_dict = {
    "KEEP_DTYPE" : 0,
    "ALLOW_FP32_DOWN_PRECISION" : 1,
    "USE_FP16" : 2,
    "USE_HF32" : 3
}

def _calculate_group(cin, cout, groups, c0):
    BLOCK_SIZE = 32
    cin_ori, cout_ori = cin // groups, cout // groups
    mag_factor0 = lcm(cin_ori, c0) // cin_ori
    mag_factor1 = lcm(cout_ori, BLOCK_SIZE) // cout_ori
    mag_factor = min(lcm(mag_factor0, mag_factor1), groups)

    cin_g = align(mag_factor * cin_ori, c0)
    cout_g = align(mag_factor * cout_ori, BLOCK_SIZE)

    group_dict = {
        "real_g": ceil_div(groups, mag_factor),
        "mag_factor": mag_factor,
        "cin_g": cin_g,
        "cin1_g": cin_g // c0,
        "cout_g": cout_g,
        "cout1_g": cout_g // BLOCK_SIZE,
        "groups": groups,
        "cin_ori": cin_ori,
        "cout_ori": cout_ori,
    }
    return group_dict


def lcm(a, b):
    return np.lcm(a, b)


def align(a, b):
    return ceil_div(a, b) * b


def ceil_div(a, b):
    return (a + b - 1) // b

def to_ndc1hwc0(data: np.ndarray, ori_format: str):
    ori_shape = data.shape
    n, c = ori_shape[ori_format.index("N")], ori_shape[ori_format.index("C")]
    h, w = ori_shape[ori_format.index("H")], ori_shape[ori_format.index("W")]
    d = ori_shape[ori_format.index("D")]
    c0 = 16
    c1 = ceil_div(c, c0)
    data = data.transpose(
        [
            ori_format.index("N"),
            ori_format.index("C"),
            ori_format.index("D"),
            ori_format.index("H"),
            ori_format.index("W"),
        ]
    )
    num_2_padding_in_cin = c1 * c0 - c
    zero_padding_array = np.zeros((n, num_2_padding_in_cin, d, h, w), dtype=data.dtype)
    data = np.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((n, c1, c0, d, h, w)).transpose(0, 3, 1, 4, 5, 2)
    return data


def to_fractal_z_3d(data: np.ndarray, ori_format: str, groups=1):
    BLOCK_SIZE = 32
    data_shape = data.shape
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    d = data_shape[ori_format.index("D")]
    fmap_c = c * groups
    out_c = n
    c0 = 16
    group_dict = _calculate_group(fmap_c, out_c, groups, c0)
    real_g = group_dict.get("real_g")
    cin1_g = group_dict.get("cin1_g")
    cout_g = group_dict.get("cout_g")
    mag_factor = group_dict.get("mag_factor")
    cout1_g = group_dict.get("cout1_g")
    weight_group = np.zeros((real_g, d, cin1_g, h, w, cout_g, c0), dtype=data.dtype)
    data = data.transpose(
        [
            ori_format.index("N"),
            ori_format.index("C"),
            ori_format.index("D"),
            ori_format.index("H"),
            ori_format.index("W"),
        ]
    )
    for g in range(groups):
        for ci in range(c):
            for co in range(n // groups):
                e = g % mag_factor
                dst_cin = e * c + ci
                dst_cout = e * (n // groups) + co
                src_cout = g * (n // groups) + co
                weight_group[
                    g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0
                ] = data[src_cout, ci, :, :, :]
    weight_group = weight_group.reshape(
        [real_g * d * cin1_g * h * w, cout1_g, BLOCK_SIZE, c0]
    )
    return weight_group


@register("aclnn_convolution_golden")
class AclnnConvolutionApi(BaseApi):
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(const aclTensor* input, const aclTensor* weight, const aclTensor* bias, const aclTensor *scale, const aclTensor *offset, const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool transposed, const aclIntArray* outputPadding, int64_t groups, int32_t offsetx, const char* roundMode, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input = input_data.kwargs["input"]
        weight = input_data.kwargs["weight"]
        bias = input_data.kwargs["bias"]
        scale = input_data.kwargs["scale"]

        stride = tuple(input_data.kwargs["stride"])
        padding = tuple(input_data.kwargs["padding"])
        dilation = tuple(input_data.kwargs["dilation"])
        offsetx = input_data.kwargs["offsetx"]
        cube_math_type = 0
        action_type = self.device

        transposed = False
        groups = 1

        def fp32_to_hf32_to_fp32(input_fp32):
            # Simulate npu hf32
            input1_hf32 = input_fp32.view(np.int32)
            input1_hf32 = np.right_shift(np.right_shift(input1_hf32, 1) + 1, 1)
            input1_hf32 = np.left_shift(input1_hf32, 2)
            input1_hf32 = input1_hf32.view(np.float32)
            return input1_hf32

        def option_tensor_none_zero_process(tmp_tensor, exit_tensor):
            if (tmp_tensor.size() == torch.Size([])) or (len(tmp_tensor) == 0):  # []ʱa.size()==torch.Size([]); [0,]ʱlen(a)=0;
                return None
            return exit_tensor

        def cast_aclnn_output_dtype(promote_dtype, cube_math_type, output_data):
            if cube_math_type == cube_math_type_dict["USE_FP16"]:
                output_data = output_data.to(torch.float16)
            elif (cube_math_type == cube_math_type_dict["ALLOW_FP32_DOWN_PRECISION"]) and ("float32" in str(promote_dtype)):
                output_data = output_data.to(torch.float32)
            elif (cube_math_type == cube_math_type_dict["USE_HF32"]) and ("float32" in str(promote_dtype)):
                output_data = output_data.to(torch.float32)
            else:
                output_data = output_data.to(promote_dtype)
                if ("bfloat16" in str(promote_dtype)):
                    output_data = output_data.to(torch.float32)
            if output_data.dtype == torch.bfloat16:
                output_data = output_data.to(torch.float16)
            return output_data

        def get_promote_dtype(dtype_input_list):
            promote_type = input_promote_type_dict[dtype_dict[dtype_input_list[0]]][dtype_dict[dtype_input_list[1]]]
            return torch_dtype_dict[promote_type]

        def cpu_golden_pre_precsion_per_cube_math_type(promote_dtype, cube_math_type, inputs, weight, bias):
            inputs = inputs.to(torch.float32)
            weight = weight.to(torch.float32)
            if bias is not None:
                bias = bias.to(torch.float32)
            if cube_math_type == 1:
                if promote_dtype == torch.float32:
                    inputs = torch.from_numpy(fp32_to_hf32_to_fp32(inputs.numpy()))
                    weight = torch.from_numpy(fp32_to_hf32_to_fp32(weight.numpy()))
                    if bias is not None:
                        bias = torch.from_numpy(fp32_to_hf32_to_fp32(bias.numpy()))
                else:
                    print("keep unchanged")
            elif cube_math_type == 2:
                if promote_dtype in (torch.float32, torch.float16):
                    inputs = inputs.to(torch.float16).to(torch.float32)
                    weight = weight.to(torch.float16).to(torch.float32)
                    if bias is not None:
                        bias = bias.to(torch.float16).to(torch.float32)
                else:
                    print("bf16 unsupported when cubemathtype is 2.")
            elif cube_math_type == 3:
                if promote_dtype == torch.float32:
                    inputs = torch.from_numpy(fp32_to_hf32_to_fp32(inputs.numpy()))
                    weight = torch.from_numpy(fp32_to_hf32_to_fp32(weight.numpy()))
                    if bias is not None:
                        bias = torch.from_numpy(fp32_to_hf32_to_fp32(bias.numpy()))
                elif promote_dtype == torch.float16:
                    print("keep unchanged")
                else:
                    print("bf16 unsupported when cubemathtype is 3.")
            else:
                print("keep unchanged")
            return inputs, weight, bias

        def cpu_golden_cal_dtype_per_percision_method(precision_method, inputs, weight, bias=None):
            if precision_method == 2:
                input_data = torch.tensor(inputs)
                weight_data = torch.tensor(weight)
                bias_data = torch.tensor(bias) if bias is not None else None
            else:
                input_data = torch.tensor(inputs, dtype=torch.float64)
                weight_data = torch.tensor(weight, dtype=torch.float64)
                bias_data = torch.tensor(bias, dtype=torch.float64) if bias is not None else None
            return input_data, weight_data, bias_data

        if self.output is None:
            promote_dtype = get_promote_dtype([input.dtype, weight.dtype])
            if bias is not None:
                bias = bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4) if transposed else bias

            if action_type == 'cpu':
                intput_6hd = to_ndc1hwc0(input.to(torch.float).numpy(), "NCDHW")
                weight_fz3d = to_fractal_z_3d(weight.to(torch.float).numpy(), "NCDHW")

                output = torch.nn.functional.conv3d(
                    input.to(torch.float),
                    weight.to(torch.float),
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                bias_dtype = bias.dtype
                scale = scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(torch.float32)
                bias = bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(torch.float32)
                output = scale * output + bias
                output = output.to(torch.float16 if bias_dtype == torch.float32 else bias_dtype)

                return output
        else:
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
        return output