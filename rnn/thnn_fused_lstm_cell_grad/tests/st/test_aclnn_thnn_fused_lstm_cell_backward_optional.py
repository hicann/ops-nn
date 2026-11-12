# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""可选梯度（gradHyOptional / gradCOptional）缺省场景的设备侧数值回归。

op_api 单测框架只能调用第一段接口（TestGetWorkspaceSize），无法执行算子、也读不到出参，
因此“缺失梯度按零补位”“两者缺失时真正写零”两类新增数值行为必须在本用例中回归：
经私有 API 走通 aten → torch_npu → aclnnThnnFusedLstmCellBackward 两段接口，
同步回 Host 后逐元素比对。

覆盖点：
  1. 两侧均缺省：先把出参缓冲区填非零哨兵再释放，由 NPU 缓存分配器复用给出参；
     算子必须把三个输出（hasBias=false 时为两个）整体写成 0。若置零路径未执行/漏写，
     出参会残留哨兵而被检出；
  2. 单侧缺省（另一侧为非零梯度，避免全零假通过）：
     结果与“显式传入同 shape 全零梯度”的调用逐位一致，并与 CPU 公式 golden 一致；
  3. hasBias=false：gradBiasOut 按接口约定不写入，不参与数值比对（仅校验 shape 契约）；
  4. 连续 / 非连续输入：非连续张量经 l0op::Contiguous 归一后结果与连续输入一致；
  5. dtype 覆盖 FLOAT16 / FLOAT32，shape 覆盖最小形态与常规形态。

golden 说明（与 op_kernel/thnn_fused_lstm_cell_grad.h 计算式一一对应）：
  i/f/g/o 取自 storage 的四段（前向已激活），dhy/dc 缺省按零参与计算
  grad_c  = (1 - tanh(cy)^2) * (o * dhy) + dc
  dc_prev = grad_c * f
  dgates  = [grad_c * g * (1-i) * i, grad_c * cx * (1-f) * f,
             grad_c * i * (1 - g^2), (dhy * tanh(cy)) * (1-o) * o]
  db      = dgates 在 batch 维求和
"""

import argparse

import torch
import torch_npu  # noqa: F401  pylint: disable=unused-import

# 哨兵值：出参被正确置零时必须被覆盖；未写入时会原样残留
SENTINEL = 3.25


def _make_inputs(dtype, batch, hidden, seed):
    generator = torch.Generator().manual_seed(seed)
    # 与内核一致：i/f/o 为 sigmoid 输出、g 为 tanh 输出，取值落在 (0,1)/(-1,1)
    storage = torch.rand(batch, 4 * hidden, generator=generator)
    storage[:, 2 * hidden : 3 * hidden] = storage[:, 2 * hidden : 3 * hidden] * 2 - 1
    cx = torch.randn(batch, hidden, generator=generator)
    cy = torch.randn(batch, hidden, generator=generator)
    grad_hy = torch.randn(batch, hidden, generator=generator)
    grad_c = torch.randn(batch, hidden, generator=generator)
    return [t.to(dtype) for t in (cx, cy, storage, grad_hy, grad_c)]


def _golden(grad_hy, grad_c, cx, cy, storage, has_bias):
    cx = cx.to(torch.float32)
    cy = cy.to(torch.float32)
    storage = storage.to(torch.float32)
    hidden = cx.shape[-1]
    i, f, g, o = (
        storage[:, index * hidden : (index + 1) * hidden] for index in range(4)
    )
    dhy = torch.zeros_like(cx) if grad_hy is None else grad_hy.to(torch.float32)
    dc = torch.zeros_like(cx) if grad_c is None else grad_c.to(torch.float32)

    tanh_cy = torch.tanh(cy)
    grad_c_value = (1 - tanh_cy * tanh_cy) * (o * dhy) + dc
    dc_prev = grad_c_value * f
    dgates = torch.cat(
        [
            grad_c_value * g * (1 - i) * i,
            grad_c_value * cx * (1 - f) * f,
            grad_c_value * i * (1 - g * g),
            (dhy * tanh_cy) * (1 - o) * o,
        ],
        dim=-1,
    )
    db = dgates.sum(dim=0) if has_bias else None
    return dgates, dc_prev, db


def _run(grad_hy, grad_c, cx, cy, storage, has_bias):
    """入参已位于 Device 侧，调用公开 aten 接口（内部走 aclnn 两段式并同步）。"""
    outputs = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_c, cx, cy, storage, has_bias
    )
    return [output.cpu().to(torch.float32) for output in outputs]


def _cpu(tensor):
    return None if tensor is None else tensor.cpu()


def _poison(shape, dtype, device):
    """填哨兵后释放：同 shape/dtype 的出参缓冲区会被缓存分配器复用（实测确定性复用）。"""
    buffer = torch.full(shape, SENTINEL, dtype=dtype, device=device)
    del buffer


def _tolerance(dtype):
    if dtype == torch.float16:
        return {"rtol": 1e-2, "atol": 1e-3}
    return {"rtol": 1e-4, "atol": 1e-5}


def _check_one(
    name, grad_hy, grad_c, cx, cy, storage, has_bias, device, non_contiguous=False
):
    batch, hidden = cx.shape
    if non_contiguous:
        # 在 Device 侧做步长切片，构造非连续输入（触发 l0op::Contiguous 归一分支）
        cx = _stride_view(cx, device)
        cy = _stride_view(cy, device)
        storage = _stride_view(storage, device)
        grad_hy = None if grad_hy is None else _stride_view(grad_hy, device)
        grad_c = None if grad_c is None else _stride_view(grad_c, device)
    else:
        cx, cy, storage = cx.to(device), cy.to(device), storage.to(device)
        grad_hy = None if grad_hy is None else grad_hy.to(device)
        grad_c = None if grad_c is None else grad_c.to(device)

    if grad_hy is None and grad_c is None:
        # 出参缓冲区预置哨兵，验证置零路径确实写入了 0（而非“恰好是 0”）
        _poison((batch, 4 * hidden), cx.dtype, device)
        _poison((batch, hidden), cx.dtype, device)
        if has_bias:
            _poison((4 * hidden,), cx.dtype, device)

        npu = _run(grad_hy, grad_c, cx, cy, storage, has_bias)
        written = npu if has_bias else npu[:2]
        for index, output in enumerate(written):
            if not bool((output == 0).all()):
                raise AssertionError(
                    f"{name}: 输出{index}存在非零元素（哨兵未被覆盖，应全零）"
                )
        if not has_bias:
            if tuple(npu[2].shape) != (4 * hidden,):
                raise AssertionError(
                    f"{name}: hasBias=false 时 gradBiasOut shape 异常 {tuple(npu[2].shape)}"
                )
        print(
            f"PASS {name}: 两侧缺省 -> 已写出零（哨兵被覆盖），hasBias={has_bias}",
            flush=True,
        )
        return

    npu = _run(grad_hy, grad_c, cx, cy, storage, has_bias)

    # 单侧缺省：与显式零梯度的调用结果逐位一致（另一侧为非零梯度）
    if grad_hy is None or grad_c is None:
        explicit = _run(
            torch.zeros_like(cx) if grad_hy is None else grad_hy,
            torch.zeros_like(cx) if grad_c is None else grad_c,
            cx,
            cy,
            storage,
            has_bias,
        )
        compare_len = len(explicit) if has_bias else len(explicit) - 1
        for index in range(compare_len):
            lhs, rhs = npu[index], explicit[index]
            if not torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8)):
                raise AssertionError(
                    f"{name}: 缺省与显式零梯度结果不一致（输出{index}）"
                )

    golden = _golden(
        _cpu(grad_hy), _cpu(grad_c), cx.cpu(), cy.cpu(), storage.cpu(), has_bias
    )
    tol = _tolerance(cx.dtype)
    for index, (actual, expect) in enumerate(zip(npu, golden)):
        if expect is None:
            continue
        torch.testing.assert_close(
            actual, expect, **tol, msg=f"{name}: 输出{index}与 golden 不一致"
        )
    print(
        f"PASS {name}: 与 golden 一致，缺省侧与显式零梯度逐位一致，hasBias={has_bias}",
        flush=True,
    )


def _stride_view(tensor, device):
    """生成与 tensor 等 shape、步长非连续的 Device 张量（每两个元素取一个）。"""
    source = torch.randn(
        tensor.shape[:-1] + (tensor.shape[-1] * 2,), dtype=tensor.dtype, device=device
    )
    return source[..., ::2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    cases = [
        ("fp32_both_present", torch.float32, 2, 3, True, True, True, False),
        ("fp32_grad_hy_none", torch.float32, 2, 3, False, True, True, False),
        ("fp32_grad_c_none", torch.float32, 2, 3, True, False, True, False),
        ("fp32_both_none", torch.float32, 2, 3, False, False, True, False),
        ("fp32_grad_hy_none_no_bias", torch.float32, 4, 5, False, True, False, False),
        (
            "fp32_grad_hy_none_non_contiguous",
            torch.float32,
            2,
            3,
            False,
            True,
            True,
            True,
        ),
        ("fp16_both_present", torch.float16, 2, 3, True, True, True, False),
        ("fp16_grad_hy_none", torch.float16, 2, 3, False, True, True, False),
        ("fp16_both_none", torch.float16, 2, 3, False, False, True, False),
        ("fp16_both_none_min_shape", torch.float16, 1, 1, False, False, True, False),
        ("fp16_both_none_no_bias", torch.float16, 2, 3, False, False, False, False),
    ]
    for index, (
        name,
        dtype,
        batch,
        hidden,
        with_hy,
        with_c,
        has_bias,
        non_contiguous,
    ) in enumerate(cases):
        cx, cy, storage, grad_hy, grad_c = _make_inputs(
            dtype, batch, hidden, 1000 + index
        )
        _check_one(
            name,
            grad_hy if with_hy else None,
            grad_c if with_c else None,
            cx,
            cy,
            storage,
            has_bias,
            args.device,
            non_contiguous,
        )
    print(f"ALL PASS ({len(cases)} cases)", flush=True)


if __name__ == "__main__":
    main()
