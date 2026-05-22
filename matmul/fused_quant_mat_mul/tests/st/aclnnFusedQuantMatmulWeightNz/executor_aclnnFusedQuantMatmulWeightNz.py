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

import torch
import torch_npu
import ctypes
import logging
import numpy as np
import random
from torch.nn.functional import gelu as torch_gelu

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat, Int64, AclTensorList, AclIntArray, AclTensor

logging = Logger().get_logger()


def _is_transpose_last_two_dims(tensor):
    if tensor.dim() < 2 or tensor.dim() > 6:
        return False

    dim1 = tensor.dim() - 1
    dim2 = tensor.dim() - 2
    shape = tensor.shape
    strides = tensor.stride()
    if strides[dim2] != 1 or strides[dim1] != shape[dim2]:
        return False

    tmp_nx_d = shape[dim1] * shape[dim2]
    for batch_dim in range(tensor.dim() - 3, -1, -1):
        if strides[batch_dim] != tmp_nx_d:
            return False
        tmp_nx_d *= shape[batch_dim]

    return not (shape[dim1] == 1 and shape[dim2] == 1)


@register("execute_fused_quantmatmul_weightnz")
class AclnnFusedQuantMatmulWeightNz(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AclnnFusedQuantMatmulWeightNz, self).__init__(task_result)
        self.x1 = None
        self.x2 = None
        self.x1scale = None
        self.x2scale = None
        self.yscale = None
        self.x1offset = None
        self.x2offset = None
        self.yoffset = None
        self.bias = None
        self.x3 = None
        self.fusedoptype = None
        self.groupsize = None
        self.out_dtype = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 值域截断点
        value_max = 127 if self.x1.dtype == torch.int8 else 7

        if self.device == "cpu":
            if self.bias is None:
                logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 0:  bias - 无")
                # out = x1 @ x2 ∗ x2scale * x1scale
                out = torch.matmul(self.x1, self.x2).to(torch.float32) * self.x2scale.to(torch.float32)
                out = out.to(torch.float32) * self.x1scale.unsqueeze(-1).to(torch.float32)

            elif self.bias.dtype == torch.int32:
                logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 1: bias - INT32")
                # out = (x1 @ x2 + bias) ∗ x2scale * x1scale
                out = (torch.matmul(self.x1, self.x2) + self.bias).to(torch.float32) * self.x2scale.to(torch.float32)
                out = out.to(torch.float32) * self.x1scale.unsqueeze(-1).to(torch.float32)

            elif self.bias.dtype in [torch.bfloat16, torch.float16, torch.float32]:
                logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 2: bias - BFLOAT16/FLOAT16/FLOAT32")
                # out = x1 @ x2 ∗ x2scale * x1scale + bias
                out = torch.matmul(self.x1, self.x2)
                out = out.to(torch.float32) * self.x2scale.to(torch.float32)
                out = out.to(torch.float32) * self.x1scale.unsqueeze(-1).to(torch.float32)
                out = out.to(torch.float32) + self.bias.to(torch.float32)

            else:
                logging.error("输入 dtype 组合无效.")
                raise ValueError

            # gelu计算
            if self.fusedoptype == "gelu_erf":
                logging.info(f"用例 id: {self.task_result.case_config.id} | 后处理 gelu_erf")
                out = torch_gelu(out.to(torch.float32), approximate='none') # erf
            elif self.fusedoptype == "gelu_tanh":
                logging.info(f"用例 id: {self.task_result.case_config.id} | 后处理 gelu_tanh")
                out = torch_gelu(out.to(torch.float32), approximate='tanh') # tanh

            return out.to(self.out_dtype)

        if self.device == "npu":
            pass

    def init_by_input_data(self, input_data: InputDataset):
        self.x1 = input_data.kwargs['x1'].clone()
        self.x2 = input_data.kwargs['x2'].clone()
        self.x1scale = input_data.kwargs['x1Scale'].clone()
        self.x2scale = input_data.kwargs['x2Scale'].clone()

        self.yscale = None
        self.x1offset = None
        self.x2offset = None
        self.yoffset = None
        self.x3 = None
        self.is_nz = None

        if input_data.kwargs['transposeX1']:
            self.x1 = self.x1.transpose(-2, -1)
        
        if input_data.kwargs['transposeX2']:
            self.x2 = self.x2.transpose(-2, -1)
        
        if len(input_data.kwargs['bias'].shape) != 0:
            self.bias = input_data.kwargs['bias'].clone()
        else:
            self.bias = None
        
        self.groupsize = 0
        self.fusedoptype = input_data.kwargs['fusedOpType']
        
        self.out_dtype = input_data.kwargs['out'].dtype


@register("execute_aclnn_fused_quantmatmul_weightnz")
class PyAclnnFusedQuantMatmulWeightNz(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset): 
        input_args = []  # 算子的入参列表
        output_packages = []  # 算子的出参数据包列表

        transpose_x1 = input_data.kwargs.pop("transposeX1")
        transpose_x2 = input_data.kwargs.pop("transposeX2")
        self.is_nz = input_data.kwargs.pop("isNz")
        input_data.kwargs.pop("out")

        if transpose_x1:
            input_data.kwargs['x1'] = input_data.kwargs['x1'].transpose(-2, -1)

        x1 = input_data.kwargs['x1']
        if x1.dtype == torch.int32:
            x1_npu = torch_npu.npu_convert_weight_to_int4pack(x1.contiguous().npu())
            input_data.kwargs['x1'] = x1_npu

        x2 = input_data.kwargs['x2']
        if x2.dtype == torch.int32:
            x2_npu = torch_npu.npu_convert_weight_to_int4pack(x2.contiguous().npu())
            if transpose_x2:
                x2_npu = x2_npu.transpose(-2, -1)
            input_data.kwargs['x2'] = x2_npu
        elif transpose_x2:
            input_data.kwargs['x2'] = x2.transpose(-2, -1)

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)

        input_args[4] = TensorPtr()
        input_args[5] = TensorPtr()
        input_args[6] = TensorPtr()
        input_args[7] = TensorPtr()

        if len(input_data.kwargs['bias'].shape) == 0:
            input_args[8] = TensorPtr()
        
        input_args[9] = TensorPtr()
        input_args[11] = ctypes.c_long(0) # groupSize
        
        input_args.extend(output_packages)
        return input_args, output_packages
    
    def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
        if name == "x2":
            b = input_data.kwargs['x1'].shape[:-2]
            x2 = input_data.kwargs['x2']
            k, n = x2.shape[-2:]
            transpose_x2 = _is_transpose_last_two_dims(x2)
            nz_k0_value_trans = 64 if x2.dtype == torch.int32 else 32

            mat2_nd_shape = torch.Size([*b, (k + nz_k0_value_trans - 1) // nz_k0_value_trans,
                                         (n + 16 - 1) // 16, 16, nz_k0_value_trans]) if transpose_x2 else \
                torch.Size([*b, (n + nz_k0_value_trans - 1) // nz_k0_value_trans,
                            (k + 16 - 1) // 16, 16, nz_k0_value_trans])
            return mat2_nd_shape
        elif name is not None:
            return input_data.kwargs[name].shape

    def get_storage_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        if name == "x2":
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        else:
            return AclFormat.ACL_FORMAT_ND
