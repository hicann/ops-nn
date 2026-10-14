# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU FP64 golden and same-dtype PyTorch composition for aclnnLSTM.

The returned TensorLists follow the ACLNN header, not torch._VF.lstm's
three-output schema. The third-party reference uses explicit PyTorch operations,
including the seven caches. TTK places its inputs on the reference device.
"""

import torch

__spec__ = {"aclnnLSTM": "AclnnLstmTestSpec"}


def wide(value):
    return value.detach().to(device="cpu", dtype=torch.float64)


def check_options(batch_sizes, num_layers, dropout, bidirectional):
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


def check_reference_inputs(input, *groups):
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("The PyTorch reference requires FP32, FP16 or BF16 inputs")

    def check(value):
        if value is None:
            return
        if isinstance(value, (tuple, list)):
            for tensor in value:
                check(tensor)
        elif not isinstance(value, torch.Tensor):
            raise TypeError("Expected a tensor or tensor list")
        elif value.dtype != input.dtype or value.device != input.device:
            raise ValueError("Reference inputs must share the input dtype and device")

    for group in groups:
        check(group)


def prepare(input, hx, params, has_bias, num_layers, batch_first, *, promote=True):
    if input.ndim != 3:
        raise ValueError("input must be rank 3")
    convert = wide if promote else lambda value: value.detach()
    x = convert(input).transpose(0, 1) if batch_first else convert(input)
    t, b, width = x.shape
    if t == 0:
        raise NotImplementedError("T=0 is outside this reference's supported domain")
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


def forward_reference(
    input, params, hx, has_bias, num_layers, batch_first, *, promote=True
):
    x, (h0, c0), weights = prepare(
        input, hx, params, has_bias, num_layers, batch_first, promote=promote
    )
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
        check_options(batchSizes, numLayers, dropout, bidirectional)
        result = forward_reference(input, params, hx, hasBias, numLayers, batchFirst)
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
    # TTK compares flattened output buffers, including each TensorList member.
    return (*result[:3], *(tensor for group in result[3:] for tensor in group))


class TorchRefImpl:
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
        **kwargs,
    ):
        check_reference_inputs(input, params, hx)
        check_options(batchSizes, numLayers, dropout, bidirectional)
        result = forward_reference(
            input, params, hx, hasBias, numLayers, batchFirst, promote=False
        )
        if not train:
            return result[:3]
        return (*result[:3], *(tensor for group in result[3:] for tensor in group))


class AclnnLstmTestSpec:
    golden = staticmethod(aclnn_lstm_golden)
    third_party = {"torch": TorchRefImpl}
    tolerance = {
        name: {"standard": "cross_check", "level": "L2"}
        for name in ("float32", "float16", "bfloat16")
    }
