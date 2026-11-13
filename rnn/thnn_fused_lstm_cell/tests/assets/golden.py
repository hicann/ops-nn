#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK TestSpec — ThnnFusedLstmCell（kernel / aclnn / torch.ops.aten e2e 通路）。

三方交叉校验（`--compare cross_check`，或由 tolerance 自动路由）：
  - **golden**：golden()——四门手写公式链，按收到的输入精度计算并原精度返回
    （框架提升精度调用时即以提升后精度产出真值，不做降精度回投；
    bfloat16 numpy 输入经 uint16 位桥位级无损转 torch.bfloat16）；
  - **third_party**：ThirdPartyImpl——GPU server 原生
    ``aten::_thnn_fused_lstm_cell``（真实竞品参考，fp32/fp16 调用失败打印
    日志并报错不回退；bf16 为 aten CUDA 未注册 dtype，按“退化仅限 API
    不支持的 dtype”口径回退 torch 手写组合），CPU server 回退 aten 数学
    语义的 torch 手写组合；由 xpu_server 在独立子进程执行（golden/竞品框架
    不得进 aclnn 进程，同进程会 SIGSEGV）；
  - **判据**：内置 CrossCheckComparison——大值域 mare/mere/rmse 三比值
    （NPU 误差 / torch 误差，比值上限由 tolerance 声明的 level 预设决定）
    + 小值域 ErrorCount 比值（≤2.0）+ NaN/Inf 三方交叉（NPU 与 torch 一致，
    或 golden 非有限且 NPU 与 golden 一致）。
"""

# kernel 模式键 = op_name(thnn_fused_lstm_cell)；GE op 类型名键 = CamelCase
# (ThnnFusedLstmCell，图引擎用例生成侧的 op_type 命名)；aclnn 模式键 =
# api_name(aclnnThnnFusedLstmCell)；torch e2e 模式键 =
# api_name(torch.ops.aten._thnn_fused_lstm_cell)
__spec__ = {
    "thnn_fused_lstm_cell": "ThnnFusedLstmCellKernelSpec",
    "ThnnFusedLstmCell": "ThnnFusedLstmCellKernelSpec",
    "aclnnThnnFusedLstmCell": "ThnnFusedLstmCellKernelSpec",
    "torch.ops.aten._thnn_fused_lstm_cell": "TorchThnnFusedLstmCellTestSpec",
}

# 三方标准 cross_check（计算类浮点输出）；比对器按 level 预设判据执行，
# 仅消费 standard/level（及三比值/small_value 系覆盖键），不消费 rtol/atol 数值键。
# 算子 spec 数值预算（max_relative 口径对账声明，冻结于算子 spec 的
# numerical_tolerance）：fp32 rtol 1e-6/atol 1e-7、fp16 rtol 1e-3/atol 1e-5、
# bf16 rtol 1.6e-2/atol 1e-3。
# level=L1（mare 5.0 / mere 1.5 / rmse 1.5）= OA 三方报告历史实跑口径
# （spec 只冻结 rtol/atol，level 非 spec 冻结值）
_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _np_to_torch(a):
    """numpy/torch.Tensor（含 bfloat16 扩展 dtype 数组 / torch.bfloat16）→ 同 dtype 的
    torch.Tensor——仅做形态转换（numpy→torch），不做数值精度 cast（bfloat16 经
    uint16 位桥位级无损转 torch.bfloat16）。
    aclnn 模式 golden 收 torch.Tensor；kernel 模式收 numpy.ndarray——双形态兼容。"""
    import numpy as np
    import torch

    if isinstance(a, torch.Tensor):
        return a
    if getattr(a.dtype, "name", None) == "bfloat16":
        return torch.from_numpy(np.ascontiguousarray(a).view(np.uint16).copy()).view(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(a))


def _torch_to_np(t, dt):
    """torch.Tensor → numpy/torch 同形态结果——仅做形态转换（torch→numpy），
    不做数值降精度 cast（计算已在收到的精度内完成，t 的 dtype 即结果精度；
    torch.bfloat16 结果按 fp32 精确扩位回 numpy——bf16→fp32 无损，保持真值）。"""
    import torch

    if isinstance(
        dt, torch.dtype
    ):  # aclnn 模式: 输入是 torch.Tensor → 返回 torch.Tensor
        return t.detach()
    if t.dtype == torch.bfloat16:
        return t.detach().float().numpy().copy()
    return t.detach().numpy().copy()


def thnn_fused_lstm_cell_golden(
    input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None, *attrs, **kwargs
):
    del attrs, kwargs
    import torch

    ig = _np_to_torch(input_gates)
    hg = _np_to_torch(hidden_gates)
    c0 = _np_to_torch(cx)
    B, H4 = ig.shape
    H = H4 // 4
    tdt = ig.dtype  # 计算全程保持收到的输入精度（无降精度 cast；缺省 bias 同 dtype 防止隐式 type-promote）
    bi = (
        _np_to_torch(input_bias)
        if input_bias is not None
        else torch.zeros(H4, dtype=tdt)
    )
    bh = (
        _np_to_torch(hidden_bias)
        if hidden_bias is not None
        else torch.zeros(H4, dtype=tdt)
    )
    gates = ig + hg + bi + bh
    i = torch.sigmoid(gates[:, 0:H])
    f = torch.sigmoid(gates[:, H : 2 * H])
    g = torch.tanh(gates[:, 2 * H : 3 * H])
    o = torch.sigmoid(gates[:, 3 * H : 4 * H])
    cy = f * c0 + i * g
    hy = o * torch.tanh(cy)
    storage = torch.cat([i, f, g, o], dim=1)
    dt = input_gates.dtype  # 原输入 dtype/形态：numpy→numpy、torch→torch（非数值 cast）
    return [_torch_to_np(hy, dt), _torch_to_np(cy, dt), _torch_to_np(storage, dt)]


class ThnnFusedLstmCellKernelSpec:
    class ThirdPartyImpl:
        """torch 第三方参考（xpu_server 独立子进程执行）。

        双路参考：
          1. **GPU server**：调用原生 ``aten::_thnn_fused_lstm_cell``（本算子
             对标的真实竞品实现，CUDA 侧存在；fp32/fp16 调用失败打印日志并
             抛出异常，不回退手写公式）；bf16 为 aten CUDA 未注册 dtype——按
             “退化仅限 API 不支持的 dtype”口径打印日志后回退 torch 手写组合
             （fp32 内算、出口按 dtype 回写，与 CPU 路同数学）；
          2. **CPU server**：aten 无 CPU kernel（实测 NotImplementedError），按
             aten CUDA 数学语义用 torch 原语手写组合（fp32 内算、出口按 dtype 回写，
             对齐 PyTorch accscalar_t 口径）。
              注意：不能用 torch.nn.LSTMCell——其 CPU 核 NaN 整行扩散、±Inf 产生
              NaN（实测），与 IEEE 逐元素语义相悖。
        参数名与算子 def 声明一致（按名绑参）：inputGates/hiddenGates/cx/inputBias/hiddenBias。
        """

        def __call__(
            self, inputGates, hiddenGates, cx, inputBias=None, hiddenBias=None, **kwargs
        ):
            del kwargs
            import torch

            if inputGates.is_cuda:
                try:
                    hy, cy, storage = torch.ops.aten._thnn_fused_lstm_cell(
                        inputGates, hiddenGates, cx, inputBias, hiddenBias
                    )
                    return [hy, cy, storage]
                except (NotImplementedError, RuntimeError) as e:
                    if inputGates.dtype != torch.bfloat16:
                        print(
                            f"[ThirdPartyImpl] aten::_thnn_fused_lstm_cell 调用失败，"
                            f"不回退手写公式，直接报错: dtype={inputGates.dtype}, "
                            f"{type(e).__name__}: {e}"
                        )
                        raise
                    print(
                        f"[ThirdPartyImpl] aten CUDA 无 bf16 注册"
                        f"（{type(e).__name__}: {e}），按 API 不支持 dtype 的"
                        f"退化口径回退 torch 手写组合（fp32 内算）"
                    )
            dt = inputGates.dtype
            ig, hg, c0 = inputGates.float(), hiddenGates.float(), cx.float()
            H4 = ig.shape[1]
            H = H4 // 4
            bi = (
                inputBias.float()
                if inputBias is not None
                else torch.zeros(H4, device=ig.device)
            )
            bh = (
                hiddenBias.float()
                if hiddenBias is not None
                else torch.zeros(H4, device=ig.device)
            )
            gates = ig + hg + bi + bh
            i = torch.sigmoid(gates[:, 0:H])
            f = torch.sigmoid(gates[:, H : 2 * H])
            g = torch.tanh(gates[:, 2 * H : 3 * H])
            o = torch.sigmoid(gates[:, 3 * H : 4 * H])
            cy = f * c0 + i * g
            hy = o * torch.tanh(cy)
            storage = torch.cat([i, f, g, o], dim=1)
            return [hy.to(dt), cy.to(dt), storage.to(dt)]

    golden = staticmethod(thnn_fused_lstm_cell_golden)
    third_party = {"torch": ThirdPartyImpl}
    tolerance = _TOLERANCE
    # 不定义 compare 方法：比对口径由 tolerance 声明的 cross_check 标准驱动
    # （内置 CrossCheckComparison 三方判据）；自定义 compare 会被比对框架在
    # kernel/aclnn 路采信，静默把声明的三方标准降为两方比对。两方模式
    # （--compare close/mixed）由框架内置比对承接，容差同源于 tolerance 声明。


class TorchThnnFusedLstmCellTestSpec:
    """torch 通路 spec（CSV api_name = torch.ops.aten._thnn_fused_lstm_cell）。

    aten 在 CPU 无 kernel（NotImplementedError，实测）——golden 不能直调 aten，
    复用 KernelSpec 同一手写数学链（按收到的精度计算独立真值，
    torch.Tensor in/out）；三方参考经 snake_case 适配委托 KernelSpec 的
    ThirdPartyImpl
    （GPU 原生 aten，失败即报错不回退 / CPU 手写组合）——本通路按 aten 参数名绑参，与
    kernel/aclnn 通路的 camelCase（算子 def 声明）不同。
    图模式（torch.compile）由框架通用 GraphNetwork 包裹注册 API 执行。
    """

    class ThirdPartyImpl:
        """torch 通路三方参考：参数名同 aten 声明（snake_case），双路逻辑委托复用。"""

        def __call__(
            self,
            input_gates,
            hidden_gates,
            cx,
            input_bias=None,
            hidden_bias=None,
            **kwargs,
        ):
            del kwargs
            return ThnnFusedLstmCellKernelSpec.ThirdPartyImpl()(
                inputGates=input_gates,
                hiddenGates=hidden_gates,
                cx=cx,
                inputBias=input_bias,
                hiddenBias=hidden_bias,
            )

    golden = staticmethod(thnn_fused_lstm_cell_golden)
    third_party = {"torch": ThirdPartyImpl}
    tolerance = _TOLERANCE
