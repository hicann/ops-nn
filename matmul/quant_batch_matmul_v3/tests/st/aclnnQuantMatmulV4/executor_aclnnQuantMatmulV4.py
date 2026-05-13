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


@register("execute_quant_matmul_v4")
class AclnnQuantMatmulV4(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AclnnQuantMatmulV4, self).__init__(task_result)
        self.x1 = None
        self.x2 = None
        self.scale = None
        self.offset = None
        self.pertoken = None
        self.bias = None
        self.out_dtype = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 值域截断点
        value_max = 127 if self.x1.dtype == torch.int8 else 7
        if self.bias == None and self.pertoken == None:
            logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 0: pertoken - 无  bias - 无")
            # out = x1 @ x2 ∗ scale + offset
            out = torch.matmul(self.x1, self.x2).to(self.scale.dtype) * self.scale
            if self.offset != None:
                out = out.to(self.offset.dtype) + self.offset

        elif self.pertoken == None and self.bias.dtype == torch.int32:
            logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 1: pertoken - 无  bias - INT32")
            # out = (x1 @ x2 + bias) ∗ scale + offset
            out = (torch.matmul(self.x1, self.x2).to(self.bias.dtype) + self.bias).to(self.scale.dtype) * self.scale
            if self.offset != None:
                out = out.to(self.offset.dtype) + self.offset

        elif self.pertoken == None and self.bias.dtype in [torch.bfloat16, torch.float32]:
            logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 2: pertoken - 无  bias - BFLOAT16/FLOAT32")
            # out = x1 @ x2 ∗ scale + bias
            out = torch.matmul(self.x1, self.x2)
            out = out.to(self.scale.dtype) * self.scale
            out = out.to(self.bias.dtype) + self.bias

        elif self.pertoken != None and self.bias == None:
            logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 3: pertoken - 有  bias - 无")
            # out = x1 @ x2 ∗ scale ∗ pertokenScaleOptional
            out = torch.matmul(self.x1, self.x2).to(self.scale.dtype) * self.scale 
            out = out.to(self.pertoken.dtype) * self.pertoken.unsqueeze(1) 

        elif self.pertoken != None and self.bias.dtype == torch.int32:
            logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 4: pertoken - 有  bias - INT32")
            # out=(x1 @ x2 + bias) ∗ scale ∗ pertokenScaleOptional
            out = (torch.matmul(self.x1, self.x2) + self.bias).to(self.scale.dtype) * self.scale
            out = out.to(self.pertoken.dtype) * self.pertoken.unsqueeze(1) 

        elif self.pertoken != None and self.bias.dtype in [torch.bfloat16, torch.float16, torch.float32]:
            logging.info(f"用例 id: {self.task_result.case_config.id} | 场景 5: pertoken - 有  bias - BFLOAT16/FLOAT16/FLOAT32")
            # out = x1 @ x2 ∗ scale ∗ pertokenScaleOptional + bias
            out = torch.matmul(self.x1, self.x2).to(self.scale.dtype) * self.scale 
            out = out.to(self.pertoken.dtype) * self.pertoken.unsqueeze(1) 
            out = out.to(self.bias.dtype) + self.bias

        else:
            logging.error("输入 dtype 组合无效.")
            raise ValueError

        return out.to(self.out_dtype)

    def init_by_input_data(self, input_data: InputDataset):
        self.x1 = input_data.kwargs['x1'].clone()
        self.x2 = input_data.kwargs['x2'].clone()
        self.scale = input_data.kwargs['Scale'].clone()

        if input_data.kwargs['transposeX1']:
            self.x1 = self.x1.transpose(-2, -1)
        
        if input_data.kwargs['transposeX2']:
            self.x2 = self.x2.transpose(-2, -1)

        if len(input_data.kwargs['Offset'].shape) != 0:
            self.offset = input_data.kwargs['Offset'].clone()
        else:
            self.offset = None

        if len(input_data.kwargs['pertokenScaleOptional'].shape) != 0:
            self.pertoken = input_data.kwargs['pertokenScaleOptional'].clone()
        else:
            self.pertoken = None
        
        if len(input_data.kwargs['bias'].shape) != 0:
            self.bias = input_data.kwargs['bias'].clone()
        else:
            self.bias = None
        
        self.out_dtype = input_data.kwargs['out'].dtype

@register("execute_aclnn_quant_matmul_v4")
class PyAclnnQuantMatmulV4(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset): 
        input_args = []  # 算子的入参列表
        output_packages = []  # 算子的出参数据包列表

        x1 = input_data.kwargs['x1']
        if x1.dtype == torch.int32:
            x1_npu = torch_npu.npu_convert_weight_to_int4pack(x1.npu())
            input_data.kwargs['x1'] = x1_npu

        x2 = input_data.kwargs['x2']
        if x2.dtype == torch.int32:
            x2_npu = torch_npu.npu_convert_weight_to_int4pack(x2.npu())
            input_data.kwargs['x2'] = x2_npu

        input_data.kwargs.pop("isNz")
        input_data.kwargs.pop("out")

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)

        if len(input_data.kwargs['Offset'].shape) == 0:
            input_args[3] = TensorPtr()

        if len(input_data.kwargs['pertokenScaleOptional'].shape) == 0:
            input_args[4] = TensorPtr()

        if len(input_data.kwargs['bias'].shape) == 0:
            input_args[5] = TensorPtr()
            
        input_args.extend(output_packages)
        return input_args, output_packages

