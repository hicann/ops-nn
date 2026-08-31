# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------
# thnn_fused_lstm_cell 执行插件（ST 用例 CPU golden）
# 接口（aclnnThnnFusedLstmCell，见 docs/aclnnThnnFusedLstmCell.md）：
#   gates = inputGates + hiddenGates + b_ih + b_hh
#   [i, f, g, o] = split(gates, 4, dim=-1)   # 内核偏移：i@0, f@H, g@2H, o@3H
#   i_out=sigmoid(i), f_out=sigmoid(f), g_out=tanh(g), o_out=sigmoid(o)
#   storage = concat([i_out, f_out, g_out, o_out], dim=-1)
#   cy = f_out*cx + i_out*g_out
#   hy = o_out*tanh(cy)
# 输出：(hy, cy, storage)
#
# 执行路径：
#   - CPU（golden）：PyTorch 原生算子，低精度输入先转 fp32 计算再回转（与 NPU 内核
#     cast-in/fp32 计算/cast-out 行为一致，保证混合容差比对有意义）
#   - NPU：纯 aclnn 算子，NPU 路径使用 `atk aclnn` 命令（pyaclnn 后端按
#     aclnn_name=ThnnFusedLstmCell 调用两段式 aclnn API，aclnn_api_type=aclnn_function）
# -------------------------------------------------------------------------
import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("thnn_fused_lstm_cell_api")
class ThnnFusedLstmCellApi(BaseApi):
    def __call__(self, input_data, with_output: bool = False):
        # 输入按 JSON 顺序/名称还原：inputGates, hiddenGates, cx, inputBias, hiddenBias
        inputs = self._extract_inputs(input_data)
        if inputs is None:
            raise ValueError(
                "无法从 input_data 中解析算子输入（args 与 kwargs 均未匹配）"
            )

        input_gates, hidden_gates, cx, input_bias, hidden_bias = inputs

        # ---- CPU golden ----
        # 低精度先转 fp32 计算（与 NPU 内核 cast-in 行为对齐）
        orig_dtype = input_gates.dtype
        compute_dtype = torch.float32
        ig = input_gates.to(compute_dtype)
        hg = hidden_gates.to(compute_dtype)
        c0 = cx.to(compute_dtype)
        b_ih = input_bias.to(compute_dtype) if input_bias is not None else None
        b_hh = hidden_bias.to(compute_dtype) if hidden_bias is not None else None

        # b = b_ih + b_hh（无偏置则为 0）
        if b_ih is None and b_hh is None:
            bias = 0.0
        elif b_ih is not None and b_hh is not None:
            bias = (b_ih + b_hh).unsqueeze(0)
        else:
            raise ValueError(
                "inputBias 与 hiddenBias 必须同存同缺（aclnn 入口 C++ 校验）"
            )

        gates = ig + hg + bias
        # 内核切分顺序 [i, f, g, o]（i@0, f@H, g@2H, o@3H，见 op_kernel FillTaskInfo）
        gates_i, gates_f, gates_g, gates_o = gates.chunk(4, dim=-1)

        i_out = torch.sigmoid(gates_i)
        f_out = torch.sigmoid(gates_f)
        g_out = torch.tanh(gates_g)
        o_out = torch.sigmoid(gates_o)

        storage = torch.cat([i_out, f_out, g_out, o_out], dim=-1)
        cy = f_out * c0 + i_out * g_out
        hy = o_out * torch.tanh(cy)

        # 结果转回输入 dtype（C++ 校验要求输出 dtype 与输入一致）
        hy = hy.to(orig_dtype)
        cy = cy.to(orig_dtype)
        storage = storage.to(orig_dtype)
        return (hy, cy, storage)

    @staticmethod
    def _extract_inputs(input_data):
        """按名称从 kwargs 优先、args 兜底提取 5 个输入，返回元组或 None。"""
        if getattr(input_data, "kwargs", None):
            names = ["inputGates", "hiddenGates", "cx", "inputBias", "hiddenBias"]
            if all(n in input_data.kwargs for n in names):
                return tuple(input_data.kwargs[n] for n in names)
        args = getattr(input_data, "args", None)
        if args is not None and len(args) >= 5:
            return tuple(args[0:5])
        return None
