#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import ctypes
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


@register("execute_aclnnEmbeddingBag_atk")
class FunctionEmbeddingBagApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionEmbeddingBagApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None
        self.mode = None
        self.offsets = []
        self.indices_shape = []
        self.per_sample_weights = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = torch.ops.aten._embedding_bag(
            weight=input_data.kwargs["weight"],
            indices=input_data.kwargs["indices"],
            offsets=input_data.kwargs["offsets"],
            scale_grad_by_freq=input_data.kwargs["scaleGradByFreq"],
            mode=input_data.kwargs["mode"],
            sparse=input_data.kwargs["sparse"],
            per_sample_weights=input_data.kwargs["perSampleWeights"],
            include_last_offset=input_data.kwargs["includeLastOffset"],
            padding_idx=input_data.kwargs["paddingIdx"],
        )

        y = output[0]
        offset2bag = output[1]
        bagsize = output[2]
        maxindices = output[3]
        return y, offset2bag, bagsize, maxindices

    def init_by_input_data(self, input_data: InputDataset):
        cat_tensor = torch.tensor([0])
        if self.device == "pyaclnn":
            cat_tensor = cat_tensor.npu()
        mode = input_data.kwargs["mode"]
        if mode != 0:
            input_data.kwargs["perSampleWeights"] = None
        input_data.kwargs["includeLastOffset"] = False
        padding_idx = input_data.kwargs["paddingIdx"]
        if padding_idx < 0:
            input_data.kwargs["paddingIdx"] = 20000
        self.indices_shape = input_data.kwargs["indices"].shape
        if input_data.kwargs["indices"].dtype in [torch.uint8, torch.int8, torch.int16]:
            input_data.kwargs["indices"] = input_data.kwargs["indices"].to(torch.int32)
        input_data.kwargs["offsets"] = torch.unique(
            torch.cat((input_data.kwargs["offsets"], cat_tensor)), sorted=True
        ).to(input_data.kwargs["offsets"].dtype)


@register("pyaclnn_aclnnEmbeddingBag_atk")
class PyaclnnEmbeddingBagApi(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        torch.npu.synchronize()
        input_args = []
        output_packages = []  # 算子的出参数据包列表
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)
        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)  # 保存完整AclTensorStruct结构
        input_args.extend(output_packages)
        mode = input_data.kwargs["mode"]
        if mode != 0:
            AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
            null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
            null_tensor_ptr = ctypes.cast(
                null_void_ptr, AclTensorPtr
            )  # 把这个空指针类型转换为tensor指针类型
            input_args[6] = null_tensor_ptr  # 赋值给第2个参数
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output
