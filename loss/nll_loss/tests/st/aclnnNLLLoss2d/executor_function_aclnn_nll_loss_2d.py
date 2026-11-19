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
import torch
import torch.nn.functional as F
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


torch.use_deterministic_algorithms(True)


@register("ascend_function_torch_ops_aten_nll_loss2d_foraward")
class MethodTorchNnNLLLoss2d(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self_dtype = input_data.kwargs["self"].dtype
        input_data.kwargs["self"] = F.log_softmax(
            input_data.kwargs["self"].to(torch.float32), dim=-1
        ).to(self_dtype)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["self"]
        target = input_data.kwargs["target"].to(torch.int64)
        weight = input_data.kwargs["weight"]
        reduction = input_data.kwargs["reduction"]
        ignore_index = input_data.kwargs["ignore_index"]

        ori_grad_dtype = x.dtype
        not_fp32 = ori_grad_dtype != torch.float32

        if self.device == "cpu" and not_fp32:
            x = x.to(torch.float32)
            weight = weight.to(torch.float32)
        loss, total_weight = torch.ops.aten.nll_loss2d_forward(
            x, target, weight, reduction, ignore_index
        )
        if reduction == 0:
            total_weight = torch.tensor(0.0, dtype=ori_grad_dtype)
        output = (loss, total_weight)
        if self.device == "cpu" and not_fp32:
            output = (loss.to(ori_grad_dtype), total_weight.to(ori_grad_dtype))
        return output


@register("aclnn_function_nll_loss2d")
class aclnnfunctionExecutor(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super().__init__(task_result, backend)
        self.reduction = None

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        self.reduction = input_args[3]
        if self.reduction.value == 0:
            output_packages = input_args[-len(output_packages) : -1]
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        if self.reduction.value == 0:
            for output_pack in output_packages:
                output.append(self.acl_tensor_to_torch(output_pack))
            torch_type = getattr(
                torch, self.task_result.output_info_list[1].dtype.replace("torch.", "")
            )
            total_weight = torch.tensor(0.0, dtype=torch_type)
            output.append(total_weight)
        else:
            for output_pack in output_packages:
                output.append(self.acl_tensor_to_torch(output_pack))
        return output
