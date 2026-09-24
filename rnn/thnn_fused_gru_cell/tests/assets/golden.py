#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 本文件与 ops-test-kit 仓 thnn_fused_gru_cell.py 保持一致（spec.yaml math_semantics
# oracle，golden_invariant_check.py 闭环）；两仓同步修改，勿单侧变更。
"""thnn_fused_gru_cell — TTK TestSpec（golden 参考实现，独立于被测 kernel）。

语义真值：docs/thnn_fused_gru_cell/spec.yaml ``math_semantics``（对标竞品 oracle
``torch.ops.aten._thnn_fused_gru_cell``，即 PyTorch v2.8.0 CUDA kernel
gru_cell_forward 的逐步骤形式）。本文件不 import / 不复用被测算子工程
（op_host / op_kernel / op_graph）的任何产物，仅以 numpy 直译 spec 公式。

数学语义（门序 [r, z, n]，bias (3H,) 沿 batch 维广播；门控 (B,3H)、隐状态 (B,H)）：

    rg = 1 / (1 + exp(-(gi_r + gh_r + b1_r + b2_r)))     # 重置门（r/z 段两路 bias 直接相加）
    zg = 1 / (1 + exp(-(gi_z + gh_z + b1_z + b2_z)))     # 更新门
    ng = tanh(gi_n + b1_n + rg * (gh_n + b2_n))          # b1_n 在 reset 乘法外、b2_n 在乘法内
    hy = ng + zg * (hx - ng)                             # 增量形式（对齐 aten 求值形式，
                                                         #   非 (1-z)*n + z*hx 合并形式）
    workspace = [rg | zg | ng | hx | gh_n+b2_n]          # (B, 5H) 五段布局

dtype 语义（spec dtype_policy：same_as_first_input + accumulator_dtype: float32）：

    float32  → fp32 直算；
    float16 / bfloat16 → 输入位桥保真提升 fp32（精确、无舍入）计算，终态单次
    round-to-nearest-even 舍回原 dtype（对齐 aten acc_type<scalar_t, is_cuda=true>）。
    numpy 侧 bf16 以 en_dtypes/ml_dtypes 扩展 dtype 到达，astype 往返即位桥
    （实测 ml_dtypes/torch/位级 RNE 三者逐位一致）；torch 侧 bf16 经
    .to(float32) 精确提升、.to(bfloat16) 单次 RNE 舍回。
    kernel 模式 Promote（cross_check 采集）送达的 fp64 提升输入按到达 dtype 直算。

双模式入参（同一 golden 服务 kernel/GEIR 与 aclnn，框架按位置传张量）：

    kernel/GEIR（numpy.ndarray）：op def.cpp 输入序 input_gates, hidden_gates, hx,
        input_bias, hidden_bias（可选 bias 缺位为 None）；
    aclnn（torch.Tensor）：C 头文件 aclnnThnnFusedGruCellGetWorkspaceSize 张量序
        （含输出位）—— 5 个输入 + hyOut/workspaceOut 两个输出占位（忽略，返回新张量）。
    可选 bias 缺位（None / aclnn 空指针）≡ 全零 bias（spec shape_constraints）。

竞品标杆（reference_oracle）只经 third_party 属性声明，由 TTK 在独立子进程 /
远端 XPU 隔离执行——golden 进程内绝不调用 torch.ops.aten._thnn_fused_gru_cell
（aclnn 进程已初始化 aclrt，同进程调用竞品 oracle 会 SIGSEGV）。torch 在本文件中
仅用于 aclnn 模式张量 <-> numpy 的数据摆渡（延迟 import；from_numpy / .numpy() /
.to，非计算 API），与 TTK 自身 torch_to_numpy_tensor 的摆渡方式同构。

tolerance：计算类算子（Broadcast / FusedComposite 数值计算范式）浮点输出 →
cross_check（$TTK/ttk/test_spec/validator.py 2.1 枚举）。如实标注：远程 XPU
（torch 第三方服务）不可用时无法按 cross_check 三方比对执行，按 binary_equal
回退执行——本算子含 sigmoid/tanh 超越函数且 fp16/bf16 经 fp32 中间精度舍回
（spec numerical_tolerance 已注明 bitwise 不可行），binary_equal 回退仅供无 XPU
环境观察，正常精度门禁以 cross_check 为准。

SPEC-ORACLE-1 闭环：本 golden 随附 docs/thnn_fused_gru_cell/develop/
golden_invariant_check.py（随机合法输入实测 spec math_semantics 全部值级不变量，
全部 PASS 方可作为 oracle）。
"""

import numpy as np

__spec__ = {
    # kernel / GEIR 模式：CSV op_name（snake_case，REG_OP 名）
    "thnn_fused_gru_cell": "ThnnFusedGruCellTestSpec",
    # aclnn 模式：CSV api_name（驼峰，aclnn<OpType>）
    "aclnnThnnFusedGruCell": "ThnnFusedGruCellTestSpec",
}


# --------------------------- 数学核心（numpy 直译 spec 公式） ---------------------------


def _gru_cell_reference(ig, hg, hx, b1, b2):
    """计算域内逐行直译 spec.yaml math_semantics.formula（对齐 aten gru_cell_forward）。

    ig/hg: (B, 3H)；hx: (B, H)；b1/b2: (3H,)——均为同一计算域 dtype。
    返回 (hy (B,H), workspace (B,5H))。
    """
    H = hx.shape[-1]
    # 门控预激活统一视为 2-D (B,3H)（合法 rank-2 输入为恒等变换；顺带规整视图 stride；
    # H=0（spec 合法退化维）时 -1 无法唯一推断，显式按首维 B 重排）
    ig = ig.reshape(ig.shape[0], 3 * H)
    hg = hg.reshape(hg.shape[0], 3 * H)
    hx = hx.reshape(ig.shape[0], H)
    # 沿最后一维按门序 [r, z, n] 三等分；bias 同步分段（(3H,) → 三段 (H,)，沿 batch 维广播）
    gi_r, gi_z, gi_n = ig[:, 0:H], ig[:, H : 2 * H], ig[:, 2 * H : 3 * H]
    gh_r, gh_z, gh_n = hg[:, 0:H], hg[:, H : 2 * H], hg[:, 2 * H : 3 * H]
    b1_r, b1_z, b1_n = b1[0:H], b1[H : 2 * H], b1[2 * H : 3 * H]
    b2_r, b2_z, b2_n = b2[0:H], b2[H : 2 * H], b2[2 * H : 3 * H]
    with np.errstate(over="ignore", invalid="ignore"):
        # 重置门 / 更新门（numpy 左结合 ((gi+gh)+b1)+b2；sigmoid 饱和：+inf→1、-inf→0、NaN 传播）
        rg = 1.0 / (1.0 + np.exp(-(gi_r + gh_r + b1_r + b2_r)))
        zg = 1.0 / (1.0 + np.exp(-(gi_z + gh_z + b1_z + b2_z)))
        # 新门：input_bias 的 n 段（b1_n）在 reset 乘法外，hidden_bias 的 n 段（b2_n）在乘法内
        ng = np.tanh(gi_n + b1_n + rg * (gh_n + b2_n))
        # 新隐状态：增量形式（spec 红线：不可改写为 (1-z)*n + z*hx 合并形式）
        hy = ng + zg * (hx - ng)
        # 反向复用 workspace：五段布局 [rg | zg | ng | hx | gh_n+b2_n]，(B, 5H)
        ws = np.concatenate([rg, zg, ng, hx, gh_n + b2_n], axis=-1)
    return hy, ws


def _compute_numpy(ig, hg, hx, b1, b2):
    """numpy 入口：按输入 dtype 选择计算域，终态舍回输入 dtype。返回 (hy, ws)。

    计算域：float64 直算（kernel Promote 提升送达的输入）；float32 直算；
    float16/bfloat16 → fp32 中间精度（位桥保真精确提升，终态单次 RNE 舍回）。
    """
    in_dtype = (
        ig.dtype
    )  # dtype 派发锚点 = input_gates（spec dtype_policy.same_as_first_input）
    dom = np.float64 if in_dtype == np.float64 else np.float32
    igf = np.asarray(ig, dtype=dom)
    hgf = np.asarray(hg, dtype=dom)
    hxf = np.asarray(hx, dtype=dom)
    three_h = igf.shape[-1]
    # 可选 bias 缺位（None）≡ 全零（aclnn 空指针路径）
    b1f = (
        np.asarray(b1, dtype=dom).reshape(-1)
        if b1 is not None
        else np.zeros(three_h, dtype=dom)
    )
    b2f = (
        np.asarray(b2, dtype=dom).reshape(-1)
        if b2 is not None
        else np.zeros(three_h, dtype=dom)
    )
    hy, ws = _gru_cell_reference(igf, hgf, hxf, b1f, b2f)
    # 终态舍回：fp16/bf16 单次 RNE（astype 即 RNE，实测与 torch/位级 RNE 逐位一致）；
    # fp32/fp64 为 no-op
    return hy.astype(in_dtype, copy=False), ws.astype(in_dtype, copy=False)


# ------------------- torch 数据摆渡（仅 aclnn 模式；延迟 import，不做计算） -------------------


def _is_torch_tensor(x):
    """duck-typing 识别 torch.Tensor（避免仅为探测而 import torch）。"""
    return x is not None and str(type(x).__module__).startswith("torch")


def _torch_to_numpy(t):
    """torch.Tensor → numpy.ndarray（纯数据摆渡）。bf16 经 fp32 精确位桥（numpy 无原生 bf16）。"""
    import torch

    if t is None:
        return None
    if t.dtype == torch.bfloat16:
        return t.detach().to(torch.float32).cpu().numpy()
    return t.detach().cpu().numpy()


def _numpy_to_torch(arr, torch_dtype):
    """numpy.ndarray → torch.Tensor（纯数据摆渡）。fp32 → fp16/bf16 由 torch 单次 RNE 舍入。"""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t if t.dtype == torch_dtype else t.to(torch_dtype)


def _golden_torch(ig_t, hg_t, hx_t, b1_t, b2_t):
    """aclnn 模式：torch 张量摆渡到 numpy 计算核心，结果摆渡回 torch（dtype 保持输入 dtype）。"""
    src_dtype = ig_t.dtype  # 派发锚点：inputGates
    ig = _torch_to_numpy(ig_t)  # bf16 → fp32（精确）；fp16/fp32 保持原 dtype
    hg = _torch_to_numpy(hg_t)
    hx = _torch_to_numpy(hx_t)
    b1 = _torch_to_numpy(b1_t)
    b2 = _torch_to_numpy(b2_t)
    hy, ws = _compute_numpy(ig, hg, hx, b1, b2)
    # bf16：numpy 侧为 fp32 → torch 单次 RNE 舍回 bfloat16；fp16/fp32 dtype 已一致直通
    return [_numpy_to_torch(hy, src_dtype), _numpy_to_torch(ws, src_dtype)]


class ThnnFusedGruCellTestSpec:
    """thnn_fused_gru_cell 测试规范（kernel/GEIR numpy 与 aclnn torch 双模式 golden）。"""

    def golden(
        input_gates,
        hidden_gates,
        hx,
        input_bias=None,
        hidden_bias=None,
        hy_out=None,
        workspace_out=None,
        **kwargs,
    ):
        """CPU 参考真值（框架按位置传张量，函数形式不做按名绑定）。

        kernel/GEIR 模式：def.cpp 输入序 5 个 numpy 输入（可选 bias 缺位 None）；
        aclnn 模式：C 头文件张量序 5 个 torch 输入 + hyOut/workspaceOut 输出占位（忽略）。
        返回 [hy, workspace]（numpy 路径为 numpy.ndarray，aclnn 路径为 torch.Tensor）。
        **kwargs 吸收框架元信息（input_dtypes / tensor_dtypes / short_soc_version 等）。
        """
        if _is_torch_tensor(input_gates):
            return _golden_torch(input_gates, hidden_gates, hx, input_bias, hidden_bias)
        hy, ws = _compute_numpy(input_gates, hidden_gates, hx, input_bias, hidden_bias)
        return [hy, ws]

    # 竞品标杆（spec.yaml reference_oracle：framework torch / api aten::_thnn_fused_gru_cell）。
    # 仅声明，由 TTK 在独立子进程 / 远端 XPU 隔离执行；golden 进程内不调用该 API。
    third_party = {"torch": "torch.ops.aten._thnn_fused_gru_cell"}

    # 计算类算子浮点输出 → cross_check（$TTK validator 2.1 枚举）。
    # 如实标注：远程 XPU 不可用时无法按 cross_check 三方比对执行，按 binary_equal 回退
    # （本算子超越函数 + fp32 中间精度舍回，spec 已注明 bitwise 不可行，详见模块 docstring）。
    tolerance = {
        "float32": {"standard": "cross_check"},
        "float16": {"standard": "cross_check"},
        "bfloat16": {"standard": "cross_check"},
    }
