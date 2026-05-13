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
import random

# import numpy as np
import torch
import torch_npu
# import tensorflow as tf
# from tensorflow.python.ops import gen_nn_ops

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("aclnn_softmax_cross_entropy_with_logits")
class SoftmaxCrossEntropyWithLogitsApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_features = input_data.kwargs["features"]
        input_labels = input_data.kwargs["labels"]

        # print(f"input_features is {input_features}")
        # print(f"input_labels is {input_labels}")
        data_max = input_features.max(dim=-1, keepdim=True).values
        data_sub = input_features - data_max
        data_exp = torch.exp(data_sub)
        data_sum = data_exp.sum(dim=-1, keepdim=True)
        data_div = data_exp / data_sum  # softmax输出

        # log_softmax
        data_log = data_sub - torch.log(data_sum)

        # 交叉熵损失
        data_mul = data_log * input_labels
        data_muls = data_mul * (-1)
        loss = data_muls.sum(dim=-1, keepdim=True).reshape(-1)
        # print(f"loss is {loss.shape}")
        # 反向传播梯度
        backprop = data_div - input_labels
        # print("***************aaaaa**********")

        return loss, backprop

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* features, aclTensor* labels, aclTensor* loss, aclTensor* backprop, uint64_t* workspaceSize,aclOpExecutor** executor)"

# @register("aclnn_softmax_cross_entropy_with_logits")
# class SoftmaxCrossEntropyWithLogitsApi(BaseApi):

#     def __call__(self, input_data: InputDataset, with_output: bool = False):
#         input_features = input_data.kwargs["features"]
#         input_labels = input_data.kwargs["labels"]

#         loss,backprop = torch_npu.npu_softmax_cross_entropy_with_logits_lsy(input_features, input_labels)

#         return loss, backprop

#     def get_cpp_func_signature_type(self):
#         return "aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* features, aclTensor* labels, aclTensor* loss, aclTensor* backprop, uint64_t* workspaceSize,aclOpExecutor** executor)"

# @register("aclnn_softmax")
# class SoftmaxCrossApi(AclnnBaseApi):

#     def after_call(self, output_packages):

#         loss, backprop = super().after_call(output_packages)
#         backprop.zero_() 
#         return loss, backprop



# @register("aclnn_softmax_cross_entropy_with_logits")
# class SoftmaxCrossEntropyWithLogitsApi(BaseApi):
#     def SoftMax(self, input_features, input_labels, if_debug=False):
#         if input_features.device.type != "npu":
#             input_features = input_features.npu()
#         if input_labels.device.type != "npu":
#             input_labels = input_labels.npu()
#         data_max = input_features.max(dim=-1, keepdim=True).values
#         data_sub = input_features - data_max
#         data_exp = torch.exp(data_sub)
#         data_sum = data_exp.sum(dim=-1, keepdim=True)
#         data_div = data_exp / data_sum  

#         data_log = data_sub - torch.log(data_sum)

#         data_mul = data_log * input_labels
#         data_muls = data_mul * (-1)
#         # loss = data_muls.sum(dim=-1, keepdim=True).reshape(-1)
#         loss = torch_npu.npu_softmax_cross_entropy_with_logits(input_features, input_labels)
#         backprop = data_div - input_labels

#         return loss, backprop
#     def __call__(self, input_data: InputDataset, with_output: bool = False):
#         input_features = input_data.kwargs["features"]
#         input_labels = input_data.kwargs["labels"]
        
#         # if self.device in ["cpu"]:
#         #     return self.SoftMax(input_features, input_labels)
#         # if "golden" in self.name and self.device in ["npu"]:
#         #     return self.SoftMax(input_features, input_labels)
#         if self.device in ["npu"]:
#             return self.SoftMax(input_features, input_labels)

#     def get_cpp_func_signature_type(self):
#         return "aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* features, aclTensor* labels, aclTensor* loss, aclTensor* backprop, uint64_t* workspaceSize,aclOpExecutor** executor)"

@register("aclnn_softmax_cross_entropy_with_logits_boundary")
class SoftmaxCrossEntropyWithLogitsBoundaryApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_features = input_data.kwargs["features"]
        input_labels = input_data.kwargs["labels"]

        if self.device == "cpu":
            batch_size = input_features.shape[0]
            loss = torch.zeros(batch_size, dtype=input_features.dtype, device=input_features.device)
            # 生成backprop占位输出（形状与input_features完全一致）
            backprop = torch.zeros_like(input_features)

            return loss, backprop
        elif self.device == "pyaclnn":

            return loss, backprop

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* features, aclTensor* labels, aclTensor* loss, aclTensor* backprop, uint64_t* workspaceSize,aclOpExecutor** executor)"

@register("aclnn_softmax_cross_entropy_with_logits_discontinuous")
class SoftmaxCrossEntropyWithLogitsDiscontinuousApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_features = input_data.kwargs["features"]
        input_labels = input_data.kwargs["labels"]

        # print(f"input_features is {input_features}")
        # print(f"input_labels is {input_labels}")
        data_max = input_features.max(dim=-1, keepdim=True).values
        data_sub = input_features - data_max
        data_exp = torch.exp(data_sub)
        data_sum = data_exp.sum(dim=-1, keepdim=True)
        data_div = data_exp / data_sum  # softmax输出

        # log_softmax
        data_log = data_sub - torch.log(data_sum)

        # 交叉熵损失
        data_mul = data_log * input_labels
        data_muls = data_mul * (-1)
        loss = data_muls.sum(dim=-1, keepdim=True).reshape(-1)
        print(f"loss is {loss.shape}")
        # 反向传播梯度
        backprop = data_div - input_labels
        # print("***************aaaaa**********")

        return loss, backprop

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* features, aclTensor* labels, aclTensor* loss, aclTensor* backprop, uint64_t* workspaceSize,aclOpExecutor** executor)"

    def init_by_input_data(self, input_data: InputDataset):
        
        print("=====", input_data.kwargs['features'].shape, input_data.kwargs['labels'].shape)
        print("转置前input tensor", input_data.kwargs['features'])
        print("=====", input_data.kwargs["features"].is_contiguous())
        origin_shape = input_data.kwargs['features'].shape
        if len(input_data.kwargs['features'].shape) == 2:
            input_data.kwargs['features'].transpose_(0, 1)
            input_data.kwargs['features'] = input_data.kwargs['features'].reshape(origin_shape)
        if len(input_data.kwargs['labels'].shape) == 2:
            input_data.kwargs['labels'].transpose_(0, 1)
            input_data.kwargs['labels'] = input_data.kwargs['labels'].reshape(origin_shape)
        print(input_data.kwargs["features"].is_contiguous())
        print("转置后input tensor", input_data.kwargs['features'])
        print(input_data.kwargs['features'].shape, input_data.kwargs['labels'].shape)  


@register("aclnn_softmax_cross_entropy_with_logits_exp")
class SoftmaxCrossEntropyWithLogitsApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_features = input_data.kwargs.get("features", None)
        input_labels = input_data.kwargs.get("labels", None)

        # 如果输入的 tensor 为空，直接返回一个默认的损失值（例如0），或者跳过计算
        if input_features is None or input_labels is None or input_features.numel() == 0 or input_labels.numel() == 0:
            # 获取input_features的dtype
            dtype = input_features.dtype if input_features is not None else torch.float32  # 如果input_features为空，使用默认的float32类型
            
            # 默认损失（保持dtype一致）
            loss = torch.tensor(0.0, dtype=dtype)  # 设置损失为与input_features相同的dtype
            backprop = torch.zeros_like(input_features, dtype=dtype) if input_features is not None else torch.zeros_like(input_labels, dtype=dtype)
            
            print(f"loss is {loss.shape} (empty tensor case)")
            print(f"loss dtype: {loss.dtype}")
            
            return loss, backprop
        
        # print(f"input_features is {input_features}")
        # print(f"input_labels is {input_labels}")
        data_max = input_features.max(dim=-1, keepdim=True).values
        data_sub = input_features - data_max
        data_exp = torch.exp(data_sub)
        data_sum = data_exp.sum(dim=-1, keepdim=True)
        data_div = data_exp / data_sum  # softmax输出

        # log_softmax
        data_log = data_sub - torch.log(data_sum)

        # 交叉熵损失
        data_mul = data_log * input_labels
        data_muls = data_mul * (-1)
        loss = data_muls.sum(dim=-1, keepdim=True).reshape(-1)
        print(f"loss is {loss.shape}")
        
        # 反向传播梯度
        backprop = data_div - input_labels
           
        return loss, backprop

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor* features, aclTensor* labels, aclTensor* loss, aclTensor* backprop, uint64_t* workspaceSize,aclOpExecutor** executor)"