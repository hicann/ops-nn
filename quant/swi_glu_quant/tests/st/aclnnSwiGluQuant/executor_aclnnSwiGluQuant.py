# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

import numpy as np
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct


@register("function_SwiGluQuant_v1")
class TorchSwiGluQuant(BaseApi):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def generate_swiglu_input_data(self, input_data: InputDataset):
        # 增加.cpu()，将NPU/GPU上的tensor拷贝到主机内存再转numpy
        data_x = input_data.kwargs["x"].to(torch.float).cpu().numpy().astype(np.float32)
        tiling_key = "0"
        if self.dtype == torch.float16:  # fp16
            tiling_key = "1"
        elif self.dtype == torch.bfloat16:  # bf16
            tiling_key = "2"
        elif self.dtype == torch.float32:  # fp32
            tiling_key = "3"
        a, b = np.split(data_x, 2, axis=-1)
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        return a, b, tiling_key

    def do_swiglu(self, a, b):
        if self.activateLeft:
            sigmoid_mul_a = self.sigmoid(a) * a
            swiglu_y = sigmoid_mul_a * b
        else:
            sigmoid_mul_a = self.sigmoid(b) * b
            swiglu_y = sigmoid_mul_a * a
        return swiglu_y

    def generate_quant_input_data(self, tiling_key, input_data: InputDataset):
        scales, offsets, group_index = None, None, None
        # 全部加上.cpu()，防止张量在NPU/GPU设备直接转numpy报错
        offsets = input_data.kwargs["offsetsOptional"].cpu().numpy().astype(np.float32)
        scales = (
            input_data.kwargs["smoothScalesOptional"].cpu().numpy().astype(np.float32)
        )
        group_index = (
            input_data.kwargs["groupIndexOptional"].cpu().numpy().astype(np.int32)
        )

        if self.group_length >= 1:
            if self.quant_mode == "dynamic":  # 有group index场景, 动态量化
                tiling_key += "06"
            else:
                if (
                    self.static_mode == "per_tensor"
                ):  # 有group index场景, 静态per_tensor量化
                    tiling_key += "04"
                else:  # 有group index场景, 静态per_channel量化
                    tiling_key += "05"

        return scales, offsets, group_index, tiling_key

    def do_quant_06(self, swiglu_y, scales, group_index):
        ori_shape = swiglu_y.shape
        prefix_shape = swiglu_y.shape[:-1]
        prefix_dims = int(np.prod(prefix_shape))
        last_shape = swiglu_y.shape[-1]

        y_tmp = swiglu_y.copy()
        swiglu_y_tmp = swiglu_y.copy()
        y_tmp = y_tmp.reshape(prefix_dims, last_shape)
        swiglu_y_tmp = swiglu_y_tmp.reshape(prefix_dims, last_shape)
        y_tmp[0 : group_index[0], :] = (
            swiglu_y_tmp[0 : group_index[0], :] * scales[0, :]
        )
        for i in range(self.group_length - 1):
            y_tmp[group_index[i] : group_index[i + 1], :] = (
                swiglu_y_tmp[group_index[i] : group_index[i + 1], :] * scales[i + 1, :]
            )

        y_max = np.amax(
            np.abs(y_tmp), axis=-1, keepdims=True
        )  # 动态量化依赖于每一行的最大值来调整缩放因子
        int_scale = 127
        if self.dst_type == 29:
            int_scale = 7
        dynamic_scale = int_scale / y_max

        quant_y_tmp = np.round(
            np.clip(y_tmp * dynamic_scale, -int_scale - 1, int_scale)
        )  # 使用 numpy 进行乘法和四舍五入

        quant_y = quant_y_tmp.astype(np.int8)  # 转换为 int8 类型
        dynamic_scale = dynamic_scale[..., 0]
        dynamic_scale = dynamic_scale.reshape(prefix_shape)
        quant_y = quant_y.reshape(ori_shape)
        return quant_y, dynamic_scale

    def do_quant_no6(self, swiglu_y, scales, group_index, offsets):
        ori_shape = swiglu_y.shape
        prefix_shape = swiglu_y.shape[:-1]
        prefix_dims = int(np.prod(prefix_shape))
        last_shape = swiglu_y.shape[-1]
        swiglu_y_tmp = swiglu_y.copy()

        y_tmp = swiglu_y.copy()
        y_tmp = y_tmp.reshape(prefix_dims, last_shape)
        swiglu_y_tmp = swiglu_y_tmp.reshape(prefix_dims, last_shape)

        if self.static_mode == "per_tensor":
            y_tmp[0 : group_index[0], :] = (
                swiglu_y_tmp[0 : group_index[0], :] * scales[0]
            )
            if self.group_length > 1:
                for i in range(self.group_length - 1):
                    y_tmp[group_index[i] : group_index[i + 1], :] = (
                        swiglu_y_tmp[group_index[i] : group_index[i + 1], :]
                        * scales[i + 1]
                    )
            y_tmp[0 : group_index[0], :] = y_tmp[0 : group_index[0], :] + offsets[0]
            if self.group_length > 1:
                for i in range(self.group_length - 1):
                    y_tmp[group_index[i] : group_index[i + 1], :] = (
                        y_tmp[group_index[i] : group_index[i + 1], :] + offsets[i + 1]
                    )

        else:
            y_tmp[0 : group_index[0], :] = (
                swiglu_y_tmp[0 : group_index[0], :] * scales[0, :] + offsets[0, :]
            )
            for i in range(self.group_length - 1):
                y_tmp[group_index[i] : group_index[i + 1], :] = (
                    swiglu_y_tmp[group_index[i] : group_index[i + 1], :]
                    * scales[i + 1, :]
                    + offsets[i + 1, :]
                )

        int_scale = 127
        if self.dst_type == 29:
            int_scale = 7
            y_tmp = np.round(y_tmp)
            y_tmp = np.clip(y_tmp, -8, 7)
        else:
            y_tmp = np.round(y_tmp)
            y_tmp = np.clip(y_tmp, -128, 127)

        dynamic_scale = int_scale / y_tmp

        quant_y = y_tmp.astype(np.int8)  # 转换为 int8 类型
        dynamic_scale = dynamic_scale[..., 0]
        dynamic_scale = dynamic_scale.reshape(prefix_shape)
        quant_y = quant_y.reshape(ori_shape)
        return quant_y, dynamic_scale

    def do_quant(self, swiglu_y, scales, offsets, group_index, tiling_key):
        quant_y = 0
        dynamic_scale = 0
        if (
            tiling_key == "206" or tiling_key == "106" or tiling_key == "306"
        ):  # BF16, 有group index场景, 动态量化
            quant_y, dynamic_scale = self.do_quant_06(swiglu_y, scales, group_index)
        else:
            quant_y, dynamic_scale = self.do_quant_no6(
                swiglu_y, scales, group_index, offsets
            )
        return quant_y, dynamic_scale

    def init_by_input_data(self, input_data: InputDataset):
        x = input_data.kwargs["x"]
        if 0 not in input_data.kwargs["x"].shape:
            activateLeft = input_data.kwargs["activateLeft"]
            quantModeOptional = input_data.kwargs["quantModeOptional"]
            groupIndexOptional = input_data.kwargs["groupIndexOptional"]
            staticMode = input_data.kwargs["staticMode"]

            self.dtype = x.dtype
            self.activateLeft = activateLeft
            self.row_len = np.prod(x.shape[:-1])
            self.col_len = x.shape[-1]
            self.group_length = np.prod(groupIndexOptional.shape)
            self.quant_mode = quantModeOptional
            self.static_mode = staticMode
            self.dst_type = 2

            group_index = None
            np.random.seed(0)
            group_index = np.random.choice(
                range(1, self.row_len - 1), int(self.group_length - 1), replace=False
            ).astype(np.int32)
            group_index = np.sort(group_index)
            group_index = np.append(group_index, self.row_len).astype(np.int32)

            # elif self.device == "cpu":

            if self.device == "cpu" or self.device == "gpu":
                input_data.kwargs["groupIndexOptional"] = torch.from_numpy(
                    group_index
                ).to(torch.int32)
            if self.device == "pyaclnn" or self.device == "npu":
                input_data.kwargs["groupIndexOptional"] = (
                    torch.from_numpy(group_index).to(torch.int32).npu()
                )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        if self.device == "cpu" or self.device == "gpu":
            # 1. 首先计算swiglu
            if 0 in input_data.kwargs["x"].shape:
                return []
            input_a, input_b, tiling_key = self.generate_swiglu_input_data(input_data)

            swiglu_y = self.do_swiglu(input_a, input_b)
            # 2. 执行quant操作
            scales, offsets, group_index, tiling_key = self.generate_quant_input_data(
                tiling_key, input_data
            )

            quant_y, dynamic_scale = self.do_quant(
                swiglu_y, scales, offsets, group_index, tiling_key
            )

            sp = dynamic_scale.shape
            scale = np.zeros(sp, dtype=np.float32)
            output = torch.from_numpy(quant_y), torch.from_numpy(scale)
        return output


@register("aclnn_swi_glu_quant_v1")
class FunctionaclnnSwiGluQuant(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)

    def get_cpp_func_signature_type(self):
        # return "aclnnStatus aclnnSwiGluQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *smoothScalesOptional, const aclTensor *offsetsOptional, const aclTensor *groupIndexOptional, bool activateLeft, char *quantModeOptional, int64_t groupListType, int64_t dstType, const aclTensor *yOut, const aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor)"
        return "aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *smoothScalesOptional, const aclTensor *offsetsOptional, const aclTensor *groupIndexOptional, bool activateLeft, char *quantModeOptional, const aclTensor *yOut, const aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor)"

    def init_by_input_data(self, input_data: InputDataset):
        self.dst_type = 2
        if input_data.kwargs["quantModeOptional"] == "dynamic":
            input_args, output_packages = super().init_by_input_data(input_data)
            from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

            input_args[2] = TensorPtr()
            return input_args, output_packages
        return super().init_by_input_data(input_data)

    def tensor_int8_to_int4s(self, tensor_int8):
        M, N = tensor_int8.shape
        split_size = tensor_int8.size(0) // 2
        upper_half, lower_half = torch.split(tensor_int8, split_size, 0)
        # upper_half, lower_half = torch.split(tensor_int8, 2, 0)
        tensor_array = upper_half.reshape(1, upper_half.numel())[0]
        int8_array = tensor_array[0 : upper_half.numel()].view(torch.int8).cpu().numpy()
        masks = np.array([0xF << (i * 4) for i in range(2)], dtype=np.uint8)
        shifted = (int8_array[:, None] & masks) >> np.arange(0, 8, 4)
        sign_extended = np.where(shifted & 0x8, shifted - 16, shifted)
        out = torch.tensor(sign_extended.astype(np.int32)).to(torch.int8).reshape(M, N)
        return out

    def after_call(self, output_packages: AclTensorStruct):
        output = []
        if self.dst_type == 29:
            output.append(
                self.tensor_int8_to_int4s(self.acl_tensor_to_torch(output_packages[0]))
            )
        else:
            output.append(self.acl_tensor_to_torch(output_packages[0]))
        output.append(torch.zeros_like(self.acl_tensor_to_torch(output_packages[1])))
        return output
