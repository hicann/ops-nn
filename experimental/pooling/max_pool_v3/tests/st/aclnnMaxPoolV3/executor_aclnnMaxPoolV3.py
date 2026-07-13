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

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


@register("aclnn_max_pool_v3")
class AclnnMaxPoolV3Api(BaseApi):
    """aclnnMaxPoolV3 算子的 PyTorch 参考实现（支持 NPU/CPU 双后端）"""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 提取输入参数
        x = input_data.kwargs["x"]
        ksize = input_data.kwargs["ksize"]  # [N, C, H, W]
        strides = input_data.kwargs["strides"]  # [N, C, H, W]
        pads = input_data.kwargs["pads"]  # [padT, padB, padL, padR]
        ceil_mode = input_data.kwargs["ceilMode"]
        ceil_mode = False if ceil_mode == 0 else True

        # 从 NCHW 四元组提取空间维度参数，适配 PyTorch max_pool2d
        kernel_size = (
            (ksize[2], ksize[3]) if isinstance(ksize, (list, tuple)) else (ksize, ksize)
        )
        stride = (
            (strides[2], strides[3])
            if isinstance(strides, (list, tuple))
            else (strides, strides)
        )

        # pads 格式 [padT, padB, padL, padR]
        # PyTorch max_pool2d 的 padding 参数为对称 padding，无法表示非对称 padding。
        # 对于非对称 padding，先对输入做显式 F.pad，再以 padding=0 调用 max_pool2d。
        if isinstance(pads, (list, tuple)):
            padT, padB, padL, padR = pads[0], pads[1], pads[2], pads[3]
        else:
            padT = padB = padL = padR = pads

        need_explicit_pad = (padT != padB) or (padL != padR)

        ori_dtype = x.dtype

        if self.device == "npu":
            torch.npu.set_compile_mode(jit_compile=False)
            device = f"npu:{self.device_id}"
            x_in = x.to(device)
            if need_explicit_pad:
                # F.pad 格式: [padL, padR, padT, padB]
                x_in = torch.nn.functional.pad(
                    x_in, [padL, padR, padT, padB], mode="constant", value=float("-inf")
                )
                pool_padding = 0
            else:
                pool_padding = (padT, padL)
            output = torch.nn.functional.max_pool2d(
                x_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=pool_padding,
                dilation=1,
                ceil_mode=ceil_mode,
            )
            return output.cpu().to(ori_dtype)

        if self.device == "cpu":
            x_in = x.to(torch.float32)
            if need_explicit_pad:
                x_in = torch.nn.functional.pad(
                    x_in, [padL, padR, padT, padB], mode="constant", value=float("-inf")
                )
                pool_padding = 0
            else:
                pool_padding = (padT, padL)
            output = torch.nn.functional.max_pool2d(
                x_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=pool_padding,
                dilation=1,
                ceil_mode=ceil_mode,
            )
            return output.to(ori_dtype)

        # 未知设备类型：显式报错，避免隐式返回 None 导致调用方出现晦涩的 TypeError
        raise ValueError(f"Unsupported device: {self.device}, expected 'npu' or 'cpu'")

    def get_format(self, input_data: InputDataset, index=None, name=None):
        """返回 NCHW 格式"""
        return AclFormat.ACL_FORMAT_NCHW
