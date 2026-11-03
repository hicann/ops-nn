# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK references for Torch LSTM forward and the direct aclnnLSTM interface.

Torch entries compare y, h_n and c_n using CPU FP64 and same-dtype CUDA.
The ACLNN entry compares CPU FP64, same-dtype CUDA and NPU, including the seven
saved-state lists when train=True. Its CUDA reference is an explicit PyTorch
recurrence, not torch._VF.lstm/cuDNN, which does not expose these saved states.
The interfaces register separate TestSpec classes with their own contracts.
"""

import contextlib
import warnings

import numpy as np
import torch

__spec__ = {
    "npu_vf_testkits.lstm": "VfLstmTestSpec",
    "torch._VF.lstm": "VfLstmTestSpec",
    "aclnnLSTM": "AclnnLstmTestSpec",
}


def _to_torch(value):
    """numpy -> torch 桥接：库用例（TTK aclnn schema）经 function-form golden 传入
    numpy 数组，repo ST 用例传入 torch 张量；参考实现统一按 torch 计算。
    bf16（ml_dtypes）经 int16 位桥无损转 torch.bfloat16。标量原样透传。"""
    if value is None or isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_to_torch(v) for v in value)
    if isinstance(value, np.ndarray):
        name = getattr(value.dtype, "name", str(value.dtype))
        if "bfloat16" in name:
            return torch.from_numpy(np.ascontiguousarray(value).view(np.int16)).view(
                torch.bfloat16
            )
        return torch.from_numpy(value)
    return value


def _wide(value):
    return value.detach().to(device="cpu", dtype=torch.float64)


def _fp64(args):
    x, hx, params, *attributes = args
    return (_wide(x), [_wide(t) for t in hx], [_wide(t) for t in params], *attributes)


def _backend(args):
    # CUDA ATen accepts H=0; cuDNN rejects it. Do not change device, dtype or TF32 policy.
    if args[0].device.type == "cuda" and args[1][0].shape[-1] == 0:
        warnings.warn(
            "H=0 reference uses CUDA ATen with cuDNN disabled",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.backends.cudnn.flags(
            enabled=False,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        )
    return contextlib.nullcontext()


def _forward(args):
    if len(args) != 9:
        raise ValueError("Expected the nine arguments of aten::lstm.input")
    if float(args[5]) != 0.0:
        raise ValueError(
            "Cross-backend dropout masks are not synchronized; dropout must be zero"
        )
    return tuple(torch._VF.lstm(*args))


def forward_golden(*args, **kwargs):
    return _forward(_fp64(args))


class _ForwardCuda:
    def __call__(
        self,
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        **kwargs,
    ):
        if input.device.type != "cuda":
            raise ValueError("The third-party reference must run on CUDA")
        args = (
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )
        with _backend(args):
            return _forward(args)


class VfLstmTestSpec:
    golden = staticmethod(forward_golden)
    third_party = {"torch": _ForwardCuda}
    tolerance = {
        name: {"standard": "cross_check", "level": "L2"}
        for name in ("float32", "float16", "bfloat16")
    }


def _check_aclnn_options(batch_sizes, num_layers, dropout, bidirectional):
    if batch_sizes is not None:
        raise NotImplementedError("This reference supports non-packed inputs only")
    if bidirectional:
        raise NotImplementedError("This reference supports unidirectional LSTM only")
    if dropout != 0:
        raise NotImplementedError("This reference requires dropout=0")
    if (
        not isinstance(num_layers, int)
        or isinstance(num_layers, bool)
        or num_layers < 1
    ):
        raise ValueError("numLayers must be a positive integer")


def _prepare_aclnn(input, hx, params, has_bias, num_layers, batch_first, cpu_fp64=True):
    if input.ndim != 3:
        raise ValueError("input must be rank 3")

    def convert(value):
        if cpu_fp64:
            return _wide(value)
        if value.device != input.device or value.dtype != input.dtype:
            raise ValueError("CUDA inputs must share the input device and dtype")
        return value.detach()

    x = convert(input).transpose(0, 1) if batch_first else convert(input)
    t, b, width = x.shape
    stride = 4 if has_bias else 2
    if len(params) != stride * num_layers:
        raise ValueError("params length must be numLayers * (4 if hasBias else 2)")
    if params[1].ndim != 2:
        raise ValueError("Recurrent weights must have shape (4H, H)")
    hidden = params[1].shape[1]
    weights = [convert(p) for p in params]
    for layer in range(num_layers):
        expected = [(4 * hidden, width if layer == 0 else hidden), (4 * hidden, hidden)]
        if has_bias:
            expected += [(4 * hidden,), (4 * hidden,)]
        for tensor, shape in zip(
            weights[layer * stride : (layer + 1) * stride], expected
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"Layer {layer}: parameter shape {tuple(tensor.shape)} != {shape}"
                )
    if hx is None:
        states = [x.new_zeros((num_layers, b, hidden)) for _ in range(2)]
    else:
        if len(hx) != 2 or any(tuple(v.shape) != (num_layers, b, hidden) for v in hx):
            raise ValueError("hx must contain h0 and c0 with shape (numLayers, B, H)")
        states = [convert(v) for v in hx]
    return x, states, weights


def _aclnn_forward_reference(
    input, params, hx, has_bias, num_layers, batch_first, cpu_fp64=True
):
    x, (h0, c0), weights = _prepare_aclnn(
        input, hx, params, has_bias, num_layers, batch_first, cpu_fp64
    )
    # T=0（空序列）：y 与 saved-state 为空，hy/cy 取初始状态（与算子 A5 空输入语义一致）。
    # 结构须与常规路径同构：7 个 saved 组，每组 num_layers 个 (T,B,h) 空张量。
    if x.shape[0] == 0:
        hidden = h0.shape[-1]
        empty_y = x.new_empty((0, x.shape[1], hidden))
        if batch_first:
            empty_y = empty_y.transpose(0, 1).contiguous()
        saved = [
            [x.new_empty((0, x.shape[1], hidden)) for _ in range(num_layers)]
            for _ in range(7)
        ]
        return (empty_y, h0, c0, *saved)
    stride = 4 if has_bias else 2
    # List order: i, candidate j, f, o, h, c, tanh(c); members are time-first.
    saved = [[] for _ in range(7)]
    hn, cn = [], []
    for layer in range(num_layers):
        wih, whh = weights[layer * stride : layer * stride + 2]
        ht, ct = h0[layer], c0[layer]
        steps = [[] for _ in range(7)]
        for xt in x:
            gates = xt @ wih.T + ht @ whh.T
            if has_bias:
                gates = (
                    gates + weights[layer * stride + 2] + weights[layer * stride + 3]
                )
            ai, af, ag, ao = gates.chunk(4, dim=-1)
            it, ft, jt, ot = ai.sigmoid(), af.sigmoid(), ag.tanh(), ao.sigmoid()
            ct = ft * ct + it * jt
            tc = ct.tanh()
            ht = ot * tc
            for bucket, value in zip(steps, (it, jt, ft, ot, ht, ct, tc)):
                bucket.append(value)
        for bucket, values in zip(saved, steps):
            bucket.append(torch.stack(values))
        x = saved[4][-1]
        hn.append(ht)
        cn.append(ct)
    output = x.transpose(0, 1).contiguous() if batch_first else x
    return (output, torch.stack(hn), torch.stack(cn), *saved)


class AclnnLstmGolden:
    def __call__(
        self,
        input,
        params,
        hx,
        batchSizes,
        hasBias,
        numLayers,
        dropout,
        train,
        bidirectional,
        batchFirst,
    ):
        input = _to_torch(input)
        params = _to_torch(params)
        hx = _to_torch(hx)
        batchSizes = _to_torch(batchSizes)
        _check_aclnn_options(batchSizes, numLayers, dropout, bidirectional)
        result = _aclnn_forward_reference(
            input, params, hx, hasBias, numLayers, batchFirst
        )
        return result if train else result[:3]


def aclnn_lstm_golden(
    input,
    params,
    hx,
    batchSizes,
    hasBias,
    numLayers,
    dropout,
    train,
    bidirectional,
    batchFirst,
    output,
    hy,
    cy,
    iOut,
    jOut,
    fOut,
    oOut,
    hOut,
    cOut,
    tanhCOut,
    **kwargs,
):
    """TTK function-form ABI includes output buffers in C header order."""
    result = AclnnLstmGolden()(
        input,
        params,
        hx,
        batchSizes,
        hasBias,
        numLayers,
        dropout,
        train,
        bidirectional,
        batchFirst,
    )
    # TTK aclnn 比对按输出槽位 1:1 对应（实测仅比较 Top-10 槽位，TensorList 槽位
    # 只取首个成员，不展开）。因此返回 3 个标量输出 + 7 个 TensorList 的 layer 0
    # 成员；layer>=1 的 saved-state 超出比对覆盖，属覆盖缺口。
    if len(result) == 3:
        return result
    return (*result[:3], *[group[0] for group in result[3:]])


class _AclnnForwardCuda:
    def __call__(
        self,
        input,
        params,
        hx,
        batchSizes,
        hasBias,
        numLayers,
        dropout,
        train,
        bidirectional,
        batchFirst,
        output=None,
        hy=None,
        cy=None,
        iOut=None,
        jOut=None,
        fOut=None,
        oOut=None,
        hOut=None,
        cOut=None,
        tanhCOut=None,
        **kwargs,
    ):
        if input.device.type != "cuda":
            raise ValueError("The third-party reference must run on CUDA")
        _check_aclnn_options(batchSizes, numLayers, dropout, bidirectional)
        result = _aclnn_forward_reference(
            input, params, hx, hasBias, numLayers, batchFirst, cpu_fp64=False
        )
        result = result if train else result[:3]
        if len(result) == 3:
            return result
        return (*result[:3], *[group[0] for group in result[3:]])


class AclnnLstmTestSpec:
    golden = staticmethod(aclnn_lstm_golden)
    third_party = {"torch": _AclnnForwardCuda}
    tolerance = {
        dtype: {"standard": "cross_check", "level": "L2"}
        for dtype in ("float16", "bfloat16", "float32")
    }
