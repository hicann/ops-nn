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
import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

input_promote_type_dict = {
    torch.float32: {torch.float32: torch.float32, torch.float16: torch.float32, torch.bfloat16: torch.float32},
    torch.float16: {torch.float32: torch.float32, torch.float16: torch.float16, torch.bfloat16: torch.float32},
    torch.bfloat16: {torch.float32: torch.float32, torch.float16: torch.float32, torch.bfloat16: torch.bfloat16}
}
 
cube_math_type_dict = {
    "KEEP_DTYPE" : 0,
    "ALLOW_FP32_DOWN_PRECISION" : 1,
    "USE_FP16" : 2,
    "USE_HF32" : 3
}

@register("golden_conv")
class ConvGolden(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(ConvGolden, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None
        self.format_convtype_map = {
            "NCL" : 1,
            "NCHW" : 2,
            "NCDHW" : 3
        }

    def change_padding(self, padding):
        if self.conv_type * 2 == len(padding):
            return True
        return False

    def init_by_input_data(self, input_data):
        self.conv_type = self.format_convtype_map[input_data.kwargs["format"]]
        self.api_name += str(self.conv_type) + "d"

    def fp32_to_hf32_to_fp32(self, input_fp32):
        # Simulate npu hf32
        arr_int = input_fp32.view(np.int32)
        mask = ~0b111111111111
        arr_int &= mask
        result = arr_int.view(np.float32)
        return result

    def cpu_golden_pre_precsion_per_cube_math_type(self, promote_dtype, cube_math_type, inputs, weight, bias):
        # 不同的cubemathtype，golden需要和npu_off保持一致行为，需要做按照下面进行精度处理
        # cubeMathType=0：  保持原数据类型进行计算
        # cubeMathType=1：  FP32转HF32; BF16保持BF16
        # cubeMathType=2：  BF16类型/FP32类型，输入转成FP16
        # cubeMathType=3：  BF16保持BF16
        # print("in cpu_golden_pre_precsion_per_cube_math_type")
        inputs = inputs.to(torch.float32)
        weight = weight.to(torch.float32)
        if bias is not None:
            bias = bias.to(torch.float32)
        if cube_math_type == 1:
            if promote_dtype == torch.float32:
                inputs = torch.from_numpy(self.fp32_to_hf32_to_fp32(inputs.numpy()))
                weight = torch.from_numpy(self.fp32_to_hf32_to_fp32(weight.numpy()))
                if bias is not None:
                    bias = torch.from_numpy(self.fp32_to_hf32_to_fp32(bias.numpy()))
            else:
                pass
        elif cube_math_type == 2:
            if promote_dtype in (torch.float32, torch.float16, torch.bfloat16):
                inputs = inputs.to(torch.float16).to(torch.float32)
                weight = weight.to(torch.float16).to(torch.float32)
                if bias is not None:
                    bias = bias.to(torch.float16).to(torch.float32)
            else:
                pass
        elif cube_math_type == 3:
            if promote_dtype == torch.float32:
                inputs = torch.from_numpy(self.fp32_to_hf32_to_fp32(inputs.numpy()))
                weight = torch.from_numpy(self.fp32_to_hf32_to_fp32(weight.numpy()))
                if bias is not None:
                    bias = torch.from_numpy(self.fp32_to_hf32_to_fp32(bias.numpy()))
            elif promote_dtype == torch.float16:
                pass
            else:
                pass
        else:
            pass
        return inputs, weight, bias

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        bias = input_data.kwargs['bias'] if sum(input_data.kwargs['bias'].shape) != 0 else None

        padding = input_data.kwargs['padding']
        fmap = input_data.kwargs['input']
        change_padding_flag = self.change_padding(padding)
        new_padding = padding.copy()
        if change_padding_flag and self.conv_type == 2:
            new_padding = [padding[2], padding[3], padding[0], padding[1]]
        elif not change_padding_flag:
            new_padding = []
            for i in range(self.conv_type):
                new_padding.append(padding[self.conv_type - 1 - i])
                new_padding.append(padding[self.conv_type - 1 - i])
        fmap_pad = torch.nn.functional.pad(fmap, pad=new_padding)
        fmap = fmap.to(torch.float32) if self.device == 'cpu' else fmap

        if self.device == 'cpu':
            if not input_data.kwargs['transposed']:
                weight = input_data.kwargs['weight']
                cube_math_type = input_data.kwargs['cubeMathType']
                promote_dtype = input_promote_type_dict[input_data.kwargs['input'].dtype][input_data.kwargs['weight'].dtype]
                input_fmap, weight, bias = self.cpu_golden_pre_precsion_per_cube_math_type(promote_dtype, cube_math_type, fmap_pad, weight, bias)
                output = eval(self.api_name)(input_fmap,
                                            weight,
                                            bias,
                                            stride = input_data.kwargs['stride'],
                                            padding = 0,
                                            dilation = input_data.kwargs['dilation'],
                                            groups = input_data.kwargs['groups']
                                            )
                if cube_math_type == 2:
                    ouptut = output.cpu().to(torch.float16)
                return output.cpu().to(promote_dtype)
            else:
                output = torch.nn.functional.conv_transpose3d(fmap.float(), input_data.kwargs['weight'].float(),
                                                              bias.float() if bias != None else None,
                                                              stride=input_data.kwargs['stride'], padding=padding,
                                                              dilation=input_data.kwargs['dilation'],
                                                              groups=input_data.kwargs['groups'],
                                                              output_padding=input_data.kwargs['outputPadding']) 


        return output.cpu().to(input_data.kwargs['input'].dtype)

@register("aclnn_conv")
class exec_conv(AclnnBaseApi):
    def get_format(self, input_data:InputDataset, index= None, name=None):
        self.format_map = {
            "NCL" : AclFormat.ACL_FORMAT_NCL,
            "NCHW" : AclFormat.ACL_FORMAT_NCHW,
            "NCDHW" : AclFormat.ACL_FORMAT_NCDHW
        }
        # 设置aclnnConvolutionGetWorkspaceSize的tensor格式
        return self.format_map[input_data.kwargs["format"]]

    def init_by_input_data(self, input_data):
        # output未在aclnnConvolutionGetWorkspaceSize接口最后, 需要调整
        input_args, output_packages = super().init_by_input_data(input_data)
        if sum(input_args[2].pytensor.shape) == 0:
            input_args[2] = TensorPtr()
        input_args[9], input_args[10] = input_args[10], input_args[9]
        output_packages[:] = [input_args[9]]
        return input_args, output_packages