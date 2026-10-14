# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU FP64 BPTT reference consuming aclnnLstmBackward's supplied caches.

The third-party reference composes PyTorch operations on the same caches.
TTK places its inputs on the configured reference device.
It preserves their dtype and device; it does not call native cuDNN backward.
"""

import torch

__spec__ = {"aclnnLstmBackward": "AclnnLstmBackwardTestSpec"}


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


class AclnnLstmBackwardGolden:
    def __call__(
        self,
        input,
        hx,
        params,
        dy,
        dh,
        dc,
        i,
        g,
        f,
        o,
        h,
        c,
        tanhc,
        batchSizesOptional,
        hasBias,
        numLayers,
        dropout,
        train,
        bidirectional,
        batchFirst,
        outputMask,
        *,
        promote=True,
    ):
        check_options(batchSizesOptional, numLayers, dropout, bidirectional)
        if not train:
            raise ValueError("Backward reference requires train=True")
        if outputMask is not None and (len(outputMask) != 4 or not all(outputMask)):
            raise NotImplementedError("This reference requires all gradient outputs")
        x, (h0, c0), weights = prepare(
            input, hx, params, hasBias, numLayers, batchFirst, promote=promote
        )
        t, b, _ = x.shape
        hidden = h0.shape[-1]
        saved = (i, g, f, o, h, c, tanhc)
        for name, values in zip(("i", "g", "f", "o", "h", "c", "tanhc"), saved):
            if len(values) != numLayers or any(
                tuple(v.shape) != (t, b, hidden) for v in values
            ):
                raise ValueError(
                    f"{name} must contain numLayers time-first (T, B, H) tensors"
                )
        convert = wide if promote else lambda value: value.detach()
        si, sg, sf, so, sh, sc, st = [[convert(v) for v in values] for values in saved]

        def upstream(value, shape, name):
            if value is None:
                return x.new_zeros(shape)
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(shape)}")
            return convert(value)

        out_shape = (b, t, hidden) if batchFirst else (t, b, hidden)
        grad = upstream(dy, out_shape, "dy")
        if batchFirst:
            grad = grad.transpose(0, 1)
        dhn = upstream(dh, h0.shape, "dh")
        dcn = upstream(dc, c0.shape, "dc")
        dh0, dc0 = torch.zeros_like(h0), torch.zeros_like(c0)
        dparams = [torch.zeros_like(p) for p in weights]
        stride = 4 if hasBias else 2
        for layer in reversed(range(numLayers)):
            layer_x = x if layer == 0 else sh[layer - 1]
            dx = torch.zeros_like(layer_x)
            index = layer * stride
            wih, whh = weights[index : index + 2]
            dht, dct = dhn[layer], dcn[layer]
            for step in reversed(range(t)):
                hp = h0[layer] if step == 0 else sh[layer][step - 1]
                cp = c0[layer] if step == 0 else sc[layer][step - 1]
                it, gt, ft, ot, tc = (v[layer][step] for v in (si, sg, sf, so, st))
                dht = dht + grad[step]
                dct = dct + dht * ot * (1 - tc * tc)
                dai = dct * gt * it * (1 - it)
                daf = dct * cp * ft * (1 - ft)
                dag = dct * it * (1 - gt * gt)
                dao = dht * tc * ot * (1 - ot)
                da = torch.cat((dai, daf, dag, dao), dim=-1)
                dx[step] = da @ wih
                dht = da @ whh
                dct = dct * ft
                dparams[index] += da.T @ layer_x[step]
                dparams[index + 1] += da.T @ hp
                if hasBias:
                    db = da.sum(dim=0)
                    dparams[index + 2] += db
                    dparams[index + 3] += db
            dh0[layer], dc0[layer] = dht, dct
            grad = dx
        dx = grad.transpose(0, 1).contiguous() if batchFirst else grad
        return dx, dh0, dc0, dparams


def aclnn_lstm_backward_golden(
    input,
    hx,
    params,
    dy,
    dh,
    dc,
    i,
    g,
    f,
    o,
    h,
    c,
    tanhc,
    batchSizesOptional,
    hasBias,
    numLayers,
    dropout,
    train,
    bidirectional,
    batchFirst,
    outputMask,
    dxOut,
    dhPrevOut,
    dcPrevOut,
    dparamsOut,
    **kwargs,
):
    """TTK function-form ABI includes output buffers in C header order."""
    dx, dh0, dc0, dparams = AclnnLstmBackwardGolden()(
        input,
        hx,
        params,
        dy,
        dh,
        dc,
        i,
        g,
        f,
        o,
        h,
        c,
        tanhc,
        batchSizesOptional,
        hasBias,
        numLayers,
        dropout,
        train,
        bidirectional,
        batchFirst,
        outputMask,
    )
    return dx, dh0, dc0, *dparams


def prepare_saved_inputs(
    input,
    hx,
    params,
    dy,
    dh,
    dc,
    i,
    g,
    f,
    o,
    h,
    c,
    tanhc,
    batchSizesOptional,
    hasBias,
    numLayers,
    dropout,
    train,
    bidirectional,
    batchFirst,
    outputMask,
    dxOut,
    dhPrevOut,
    dcPrevOut,
    dparamsOut,
    **kwargs,
):
    """Populate only saved-state inputs; keep randomized x/weights/upstreams intact."""
    check_options(batchSizesOptional, numLayers, dropout, bidirectional)
    if not train:
        raise ValueError("Backward input preparation requires train=True")
    result = forward_reference(input, params, hx, hasBias, numLayers, batchFirst)
    with torch.no_grad():
        for name, targets, values in zip(
            ("i", "g", "f", "o", "h", "c", "tanhc"),
            (i, g, f, o, h, c, tanhc),
            result[3:],
        ):
            if len(targets) != len(values):
                raise ValueError(f"{name}: cache list length mismatch")
            for target, value in zip(targets, values):
                if target.shape != value.shape:
                    raise ValueError(f"{name}: cache shape mismatch")
                target.copy_(value.to(device=target.device, dtype=target.dtype))


class TorchRefImpl:
    def __call__(
        self,
        input,
        hx,
        params,
        dy,
        dh,
        dc,
        i,
        g,
        f,
        o,
        h,
        c,
        tanhc,
        batchSizesOptional,
        hasBias,
        numLayers,
        dropout,
        train,
        bidirectional,
        batchFirst,
        outputMask,
        **kwargs,
    ):
        check_reference_inputs(input, hx, params, dy, dh, dc, i, g, f, o, h, c, tanhc)
        dx, dh0, dc0, dparams = AclnnLstmBackwardGolden()(
            input,
            hx,
            params,
            dy,
            dh,
            dc,
            i,
            g,
            f,
            o,
            h,
            c,
            tanhc,
            batchSizesOptional,
            hasBias,
            numLayers,
            dropout,
            train,
            bidirectional,
            batchFirst,
            outputMask,
            promote=False,
        )
        return dx, dh0, dc0, *dparams


class AclnnLstmBackwardTestSpec:
    golden = staticmethod(aclnn_lstm_backward_golden)
    third_party = {"torch": TorchRefImpl}
    customize_inputs = staticmethod(prepare_saved_inputs)
    tolerance = {
        name: {"standard": "cross_check", "level": "L2"}
        for name in ("float32", "float16", "bfloat16")
    }
