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
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("golden_add_relu")
class MethodTorchAddReluApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        out = None
        self_tensor = input_data.kwargs["self"]
        other_tensor = input_data.kwargs["other"]
        alpha = input_data.kwargs["alpha"]
        
        result_dtype = torch.result_type(self_tensor, other_tensor)

        # 判断是否为浮点类型（任意一个为浮点，则整体用 float）
        is_floating = result_dtype.is_floating_point

        if is_floating:
            # 统一转为 float32（等价于 float）
            self_tensor = self_tensor.float()
            other_tensor = other_tensor.float()
            # alpha = torch.tensor(alpha, dtype=result_dtype)
            # 使用 float 计算
            muls_res = torch.mul(other_tensor, alpha)
            out = torch.add(self_tensor, muls_res)
            out = torch.relu(out)
        else:
            # 两个都是整形，使用 result_dtype
            self_tensor = self_tensor.to(result_dtype)
            other_tensor = other_tensor.to(result_dtype)
            alpha = alpha
            # 使用 int64 计算
            muls_res = torch.mul(other_tensor, alpha)
            out = torch.add(self_tensor, muls_res)
            out = torch.relu(out)

        return out.to(result_dtype)

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND
