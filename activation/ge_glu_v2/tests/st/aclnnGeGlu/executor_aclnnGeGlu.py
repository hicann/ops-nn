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
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import torch.nn.functional as F
import numpy as np


@register("function_gegluv2")
class aclnnGeGluV3Discontinues(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取输入参数
        input_tensor = input_data.kwargs.get("input", None)
        dim = input_data.kwargs.get("dim", -1)
        approximate = input_data.kwargs.get("approximate", 1)

        def do_gelu(x, approximate):
            """实现gelu激活函数"""
            if approximate == 0:
                return F.gelu(x, approximate="none")
            else:
                return F.gelu(x, approximate="tanh")

        def process(input_x, dim, approximate):
            """核心处理函数：切分张量并计算GeLU（增加维度校验）"""
            if input_x.dim() == 0:
                gelu_output = do_gelu(input_x, approximate)
                output = input_x * gelu_output
                return output, gelu_output
            
            # 转换负索引为正索引
            rank = input_x.dim()
            dim = dim if dim >= 0 else rank + dim
            
            # **新增校验：确保切分维度长度为偶数**
            if input_x.size(dim) % 2 != 0:
                raise ValueError(f"Dimension {dim} must have even size, got {input_x.size(dim)}.")
            
            x, gate = input_x.chunk(2, dim=dim)

            gelu_output = do_gelu(gate, approximate)
            # 校验形状一致性（可选，增强鲁棒性）
            if x.shape != gelu_output.shape:
                raise RuntimeError(f"Tensor shapes must match: x={x.shape}, gelu_output={gelu_output.shape}")
            
            output = x * gelu_output
            return output, gelu_output

        def ge_glu_v2_golden(input_x, dim, approximate):
            """golden参考实现"""
            dtype = input_x.dtype
            input_x = input_x.to(torch.float32) if dtype != torch.float32 else input_x

            try:
                if input_x.dim() == 0:
                    gelu_output = torch.tensor([])
                    output = torch.tensor([])
                elif len(input_x) == 0:
                    gelu_output = torch.tensor([])
                    output = torch.tensor([])
                else:
                    output, gelu_output = process(input_x, dim, approximate)
            except ValueError as e:
                # 捕获维度非偶数的异常并提示
                raise

            # 转换回原始数据类型
            if dtype != torch.float32:
                output = output.to(dtype)
                gelu_output = gelu_output.to(dtype)

            return output, gelu_output

        # 执行算子
        try:
            out, out_gelu = ge_glu_v2_golden(input_tensor, dim, approximate)
        except Exception as e:
            print(f"Error executing ge_glu_v2_golden: {e}")
            print(f"Input shape: {input_tensor.shape}")
            print(f"dim: {dim}")
            print(f"approximate: {approximate}")
            raise
        
        return out, out_gelu