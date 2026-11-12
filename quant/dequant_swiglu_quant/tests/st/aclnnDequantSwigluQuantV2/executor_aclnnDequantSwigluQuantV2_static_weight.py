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
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
import ctypes


@register("aclnn_dequantswigluquantinputprocess_static_weight")
class DequantSwigluQuantInputProcess(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    # 默认调用，可省略
    def init_by_input_data(self, input_data):
        """参数处理"""

        input_args, output_packages = super().init_by_input_data(input_data)
        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型
        if (
            input_data.kwargs["x"].dtype == torch.bfloat16
            or input_data.kwargs["x"].dtype == torch.float16
        ):
            input_args[1] = null_tensor_ptr  # weightScale
            input_args[2] = null_tensor_ptr  # activationScale
            input_args[3] = null_tensor_ptr  # bias
        input_args[6] = null_tensor_ptr  # groupIndex
        return input_args, output_packages

    def after_call(self, output_packages):
        output1, output2 = super().after_call(output_packages)
        output2.zero_()
        return output1, output2


@register("gdd_exec_aclnnDequantSwigluQuant_static_weight")
class DequantSwigluQuantApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = True):
        if self.device == "cpu" or self.device == "npu":
            x = input_data.kwargs["x"]
            quant_scale = input_data.kwargs["quant_scale"]
            quant_offset = input_data.kwargs["quant_offset"]
            activate_left = input_data.kwargs["activate_left"]
            quant_mode = input_data.kwargs["quant_mode"]

            if str(input_data.kwargs["x"].dtype) == "torch.int32":
                wight_scale = input_data.kwargs["weight_scale"]
                activate_scale = input_data.kwargs["activation_scale"]
                bias = input_data.kwargs["bias"]
                return self.golden_dequant_swiglu_quant_002(
                    x,
                    wight_scale,
                    activate_scale,
                    bias,
                    quant_scale,
                    quant_offset,
                    activate_left,
                    quant_mode,
                )
            else:
                return self.golden_dequant_swiglu_quant_001(
                    x, quant_scale, quant_offset, activate_left, quant_mode
                )

    def golden_dequant_swiglu_quant_002(
        self,
        a,
        wight_scale,
        activate_scale,
        bias,
        quant_scale,
        quant_offset,
        activate_left,
        quant_mode,
    ):
        scale_dim0 = 1
        for i in a.shape[:-1]:
            scale_dim0 = scale_dim0 * i

        x = a
        x = x.type(torch.float32)
        res = torch.mul(x, wight_scale)
        res = torch.mul(res, activate_scale)
        if bias is not None:
            res = torch.add(res, bias)
        out = torch.chunk(res, 2, dim=-1)
        if activate_left:
            self_tensor = out[0]
            other = out[1]
        else:
            self_tensor = out[1]
            other = out[0]
        output = F.silu(self_tensor) * other
        output = torch.div(output, quant_scale)
        if quant_mode == "static":
            output = torch.add(output, quant_offset)

        if quant_mode == "dynamic":
            abs = torch.abs(output)
            max_values = torch.amax(abs, dim=-1)
            max_values = max_values / 127
            output = output / max_values.unsqueeze(1)

        output = torch.clamp(output, -128, 127)

        output2 = torch.zeros((scale_dim0, 1), dtype=torch.float32)
        output2.zero_()

        return torch.round(output).to(torch.int8).cpu(), output2

    def golden_dequant_swiglu_quant_001(
        self, a, quant_scale, quant_offset, activate_left, quant_mode
    ):
        # x是fp16 bf16
        # 静态量化
        a = a.type(torch.float32)
        x_shape = a.shape
        scale_dim0 = 1
        for i in x_shape[:-1]:
            scale_dim0 = scale_dim0 * i
        x = torch.chunk(a, 2, dim=-1)
        if activate_left:
            self_tensor = x[0]
            other = x[1]
        else:
            self_tensor = x[1]
            other = x[0]
        output = F.silu(self_tensor) * other
        if quant_scale is not None:
            output = torch.div(output, quant_scale.type(torch.float32))
        if quant_mode == "static":
            output = torch.add(output, quant_offset.type(torch.float32))
        if quant_mode == "dynamic":
            abs = torch.abs(output)
            max_values = torch.amax(abs, dim=-1)
            max_values = max_values / 127
            output = output / max_values.unsqueeze(1)
        output = torch.clamp(output, -128, 127)
        output2 = torch.zeros((scale_dim0, 1), dtype=torch.float32)
        output2.zero_()
        return torch.round(output).to(torch.int8).cpu(), output2
