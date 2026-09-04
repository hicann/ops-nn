#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""avg_pool3_d_grad 双标杆 golden（kernel/aclnn/e2e 路径统一）。

Kernel 路径的 golden 收到 numpy.ndarray，转 torch tensor 后用 torch.ops.aten.avg_pool3d_backward 计算，结果转回 numpy；
ACLNN/E2E 路径的 golden 直接收到 torch.Tensor（已在设备上），无需转换。
当空间尺寸小于 kernel 时，torch 会抛异常，golden 自动回退到 numpy 参考实现。
"""

__spec__ = {
    "avg_pool3_d_grad": "AvgPool3DGradKernelTestSpec",
    "aclnnAvgPool3dBackward": "AclnnAvgPool3dBackwardTestSpec",
    "torch.ops.aten.avg_pool3d_backward": "TorchAvgPool3dBackwardTestSpec",
    "torch.ops.cann_ops_nn.avg_pool3d_backward": "TorchAvgPool3dBackwardTestSpec",
}

import numpy as np


def _pool_spatial_norm(values, nd):
    v = [int(x) for x in values]
    if len(v) == 1:
        return [v[0]] * nd
    if len(v) >= nd:
        return v[-nd:]
    raise ValueError("pool attr length %d < %d" % (len(v), nd))


def _norm3(v, data_format):
    v = [int(x) for x in v]
    if len(v) == 1:
        return [v[0], v[0], v[0]]
    if len(v) >= 5:
        return [v[2], v[3], v[4]] if data_format == "NCDHW" else [v[1], v[2], v[3]]
    return v[:3]


def _norm6(p):
    p = [int(x) for x in p]
    if len(p) == 6:
        return p
    if len(p) == 3:
        return [p[0], p[0], p[1], p[1], p[2], p[2]]
    return [p[0]] * 6


def _default_stride(stride, kernel_size):
    if not stride:
        return kernel_size
    return stride


def _needs_numpy_pool_backward(self_shape, kernel_size, pool_nd):
    spatial = [int(x) for x in self_shape[-pool_nd:]]
    kernel = _pool_spatial_norm(kernel_size, pool_nd)
    return any(size < k for size, k in zip(spatial, kernel))


def _avg_pool_backward_np(
    grad,
    self_shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    pool_nd,
):
    import numpy as np

    grad = np.asarray(grad, dtype=np.float64)
    self_shape = tuple(int(x) for x in self_shape)
    pool_nd = int(pool_nd)
    batch_nd = grad.ndim - pool_nd
    kernel = _pool_spatial_norm(kernel_size, pool_nd)
    stride = _pool_spatial_norm(stride, pool_nd)
    pad = _pool_spatial_norm(padding, pool_nd)
    spatial_in = self_shape[-pool_nd:]
    out_dims = tuple(int(x) for x in grad.shape[-pool_nd:])

    divisor = (
        np.full(out_dims, float(int(divisor_override)))
        if divisor_override
        else np.ones(out_dims, dtype=np.float64)
    )
    if not divisor_override:
        for i in range(pool_nd):
            st = np.arange(out_dims[i]) * int(stride[i]) - int(pad[i])
            in_i, k_i, p_i = int(spatial_in[i]), int(kernel[i]), int(pad[i])
            if count_include_pad:
                cnt = np.clip(np.minimum(st + k_i, in_i + p_i) - st, 0, None)
            else:
                cnt = np.clip(np.minimum(st + k_i, in_i) - np.maximum(st, 0), 0, None)
            divisor *= cnt.reshape((1,) * i + (out_dims[i],) + (1,) * (pool_nd - 1 - i))

    out_ok = np.ones(out_dims, dtype=bool) if divisor_override else (divisor > 0)
    state = grad / np.where(out_ok, divisor, 1.0)
    invalid = np.flatnonzero(~out_ok)
    if invalid.size:
        state[
            (slice(None),) * batch_nd + tuple(np.unravel_index(invalid, out_dims))
        ] = 0.0

    for i in range(pool_nd):
        ax = -(pool_nd - i)
        state = np.moveaxis(state, ax, -1)
        o_num = state.shape[-1]
        x = np.arange(int(spatial_in[i]))
        lo = (x + int(pad[i]) - int(kernel[i])) // int(stride[i]) + 1
        hi = (x + int(pad[i])) // int(stride[i]) + 1
        lo = np.clip(lo, 0, o_num)
        hi = np.clip(hi, 0, o_num)
        pre = np.concatenate(
            [np.zeros(state.shape[:-1] + (1,)), np.cumsum(state, axis=-1)], axis=-1
        )
        out = pre[..., hi] - pre[..., lo]
        state = np.where(hi > lo, out, 0.0)
        state = np.moveaxis(state, -1, ax)

    return state.astype(np.float32)


def _avg_pool_backward_3d_checked(
    grad, self_t, kernel, stride, padding, ceil_mode, count_include_pad, divisor
):
    import torch

    if _needs_numpy_pool_backward(self_t.shape, kernel, 3):
        np_ref = _avg_pool_backward_np(
            grad.detach().cpu().numpy(),
            tuple(self_t.shape),
            kernel,
            stride,
            padding,
            bool(ceil_mode),
            bool(count_include_pad),
            int(divisor) if divisor else 0,
            3,
        )
        return torch.from_numpy(np_ref).to(grad.dtype)
    return torch.ops.aten.avg_pool3d_backward(
        grad,
        self_t,
        [int(x) for x in kernel][-3:],
        [int(x) for x in stride][-3:],
        [int(x) for x in padding][-3:],
        bool(ceil_mode),
        bool(count_include_pad),
        divisor,
    )


def _is_arch35(soc):
    s = str(soc).lower()
    return any(k in s for k in ("ascend950"))


def _avg_pool3d_grad_compute(
    grads_f32,
    orig_shape,
    k,
    s,
    p,
    ceil_mode,
    count_include_pad,
    divisor_override,
    data_format,
):
    import torch

    ndim = len(orig_shape)
    kd, kh, kw = k
    sd, sh, sw = s

    if data_format == "NCDHW":
        d, h, w = orig_shape[-3], orig_shape[-2], orig_shape[-1]
    else:
        if ndim == 5:
            d, h, w = orig_shape[1], orig_shape[2], orig_shape[3]
        else:
            d, h, w = orig_shape[0], orig_shape[1], orig_shape[2]

    if data_format == "NCDHW":
        grads_t = torch.from_numpy(grads_f32)
        grads_t = grads_t.unsqueeze(0) if ndim == 4 else grads_t
    else:
        grads_t = torch.from_numpy(grads_f32)
        if ndim == 5:
            grads_t = grads_t.permute(0, 4, 1, 2, 3)
        else:
            grads_t = grads_t.permute(3, 0, 1, 2).unsqueeze(0)

    n, c = grads_t.shape[0], grads_t.shape[1]
    self_t = torch.zeros((n, c, d, h, w), dtype=torch.float32)
    divisor = None if divisor_override == 0 else int(divisor_override)
    grad_input = _avg_pool_backward_3d_checked(
        grads_t.to(torch.float32),
        self_t,
        [kd, kh, kw],
        [sd, sh, sw],
        [p[0], p[2], p[4]],
        bool(ceil_mode),
        bool(count_include_pad),
        divisor,
    )
    grad_input = grad_input.detach().cpu().numpy()
    if data_format == "NCDHW":
        grad_input = grad_input if ndim == 5 else grad_input[0]
    else:
        if ndim == 5:
            grad_input = np.transpose(grad_input, (0, 2, 3, 4, 1))
        else:
            grad_input = grad_input[0].transpose(1, 2, 3, 0)
    return grad_input


class AvgPool3DGradKernelTestSpec:
    """avg_pool3_d_grad kernel 测试规范（kernel/geir 流程，numpy 入参）

    Parameters follow avg_pool3_d_grad_def.cpp:
    orig_input_shape, grads, ksize, strides, pads, ceil_mode,
    count_include_pad, divisor_override, data_format.
    """

    def golden(
        orig_input_shape,
        grads,
        *,
        ksize,
        strides,
        pads,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=0,
        data_format="NDHWC",
        **kwargs,
    ):
        if _is_arch35(kwargs.get("short_soc_version", "")):
            return AvgPool3DGradKernelTestSpec._golden_arch35(
                orig_input_shape,
                grads,
                ksize=ksize,
                strides=strides,
                pads=pads,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
                data_format=data_format,
                **kwargs,
            )

        import tensorflow as tf

        tf.compat.v1.disable_eager_execution()
        if len(grads.shape) != 6:
            raise RuntimeError(
                "avgpool3dgrad testcase golden supports NDC1HWC0 input only!"
            )
        n_index = data_format.index("N")
        d_index = data_format.index("D")
        h_index = data_format.index("H")
        w_index = data_format.index("W")
        c_index = data_format.index("C")
        stride_h, stride_w, stride_d = (
            strides[h_index],
            strides[w_index],
            strides[d_index],
        )
        filter_h, filter_w, filter_d = ksize[h_index], ksize[w_index], ksize[d_index]
        GN, GD, GC, GH, GW, C0 = grads.shape
        IN, ID, IH, IW, IC = (
            orig_input_shape[n_index],
            orig_input_shape[d_index],
            orig_input_shape[h_index],
            orig_input_shape[w_index],
            orig_input_shape[c_index],
        )
        IC = (IC + 15) // 16 * 16
        padding = "VALID" if all(i == 0 for i in pads) else "SAME"
        output_backprop = grads.transpose(0, 1, 3, 4, 2, 5).reshape(
            GN, GD, GH, GW, GC * C0
        )
        grads_tensor = tf.compat.v1.placeholder(
            grads.dtype, shape=output_backprop.shape
        )
        grads_tensor = tf.compat.v1.cast(grads_tensor, tf.float32)
        res = tf.compat.v1.raw_ops.AvgPool3DGrad(
            orig_input_shape=[GN, ID, IH, IW, GC * C0],
            grad=grads_tensor,
            ksize=[1, filter_d, filter_h, filter_w, 1],
            strides=[1, stride_d, stride_h, stride_w, 1],
            padding=padding,
            data_format="NDHWC",
            name="avg_pool3d_grad",
        )
        res = tf.compat.v1.cast(res, tf.float16)
        feed_dict = {grads_tensor: output_backprop}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(res, feed_dict=feed_dict)
        res = (
            out.reshape((IN, ID, IH, IW, IC // C0, C0))
            .transpose(0, 1, 4, 2, 3, 5)
            .copy()
            .astype(np.float16)
        )
        return res

    def _golden_arch35(
        orig_input_shape,
        grads,
        *,
        ksize,
        strides,
        pads,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=0,
        data_format="NDHWC",
        **kwargs,
    ):
        orig_shape = [int(v) for v in orig_input_shape]
        k = _norm3(ksize, data_format)
        s = _norm3(_default_stride(strides, ksize), data_format)
        p = _norm6(pads)
        grads_f32 = np.asarray(grads, dtype=np.float32)
        is_sym = p[0] == p[1] and p[2] == p[3] and p[4] == p[5]
        if not is_sym:
            raise RuntimeError(
                "arch35 torch golden only supports symmetric pads; got pads=%s" % p
            )
        grad_input = _avg_pool3d_grad_compute(
            grads_f32,
            orig_shape,
            k,
            s,
            p,
            ceil_mode,
            count_include_pad,
            divisor_override,
            data_format,
        )
        out_dtype = str(kwargs.get("output_dtypes", ["float32"])[0]).lower()
        if out_dtype == "float16":
            return grad_input.astype(np.float16)
        if "bfloat16" in out_dtype:
            try:
                from ml_dtypes import bfloat16

                return grad_input.astype(bfloat16)
            except ImportError:
                return grad_input.astype(np.float32)
        return grad_input.astype(np.float32)

    class ThirdPartyImpl:
        def __init__(self, orig_input_shape, grads, **kwargs):
            self.out_dtype = str(kwargs.get("output_dtypes", ["float32"])[0]).lower()
            self.orig_shape = [int(v) for v in orig_input_shape]
            import torch

            self._torch = torch

        def __call__(
            self,
            orig_input_shape,
            grads,
            *,
            ksize,
            strides,
            pads,
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=0,
            data_format="NDHWC",
            **kwargs,
        ):
            torch = self._torch
            k = _norm3(ksize, data_format)
            s = _norm3(_default_stride(strides, ksize), data_format)
            p = _norm6(pads)
            if isinstance(grads, torch.Tensor):
                grads_t = grads.detach().clone()
            else:
                grads_t = torch.from_numpy(np.asarray(grads))
            orig_shape = self.orig_shape
            ndim = len(orig_shape)
            if data_format == "NCDHW":
                d, h, w = orig_shape[-3], orig_shape[-2], orig_shape[-1]
                grads_t = grads_t.unsqueeze(0) if ndim == 4 else grads_t
            else:
                if ndim == 5:
                    d, h, w = orig_shape[1], orig_shape[2], orig_shape[3]
                    grads_t = grads_t.permute(0, 4, 1, 2, 3)
                else:
                    d, h, w = orig_shape[0], orig_shape[1], orig_shape[2]
                    grads_t = grads_t.permute(3, 0, 1, 2).unsqueeze(0)
            n, c = grads_t.shape[0], grads_t.shape[1]
            self_t = torch.zeros(
                (n, c, d, h, w), dtype=grads_t.dtype, device=grads_t.device
            )
            divisor = None if divisor_override == 0 else int(divisor_override)
            grad_input = _avg_pool_backward_3d_checked(
                grads_t,
                self_t,
                k,
                s,
                [p[0], p[2], p[4]],
                bool(ceil_mode),
                bool(count_include_pad),
                divisor,
            )
            if data_format == "NCDHW":
                grad_input = grad_input if ndim == 5 else grad_input[0]
            else:
                if ndim == 5:
                    grad_input = grad_input.permute(0, 2, 3, 4, 1)
                else:
                    grad_input = grad_input[0].permute(1, 2, 3, 0)
            if self.out_dtype == "float16":
                return [grad_input.to(torch.float16)]
            if "bfloat16" in self.out_dtype:
                return [grad_input.to(torch.bfloat16)]
            return [grad_input.to(torch.float32)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class AclnnAvgPool3dBackwardTestSpec:
    """aclnnAvgPool3dBackward 测试规范（aclnn 流程，torch 入参，已在设备上）

    Parameters follow aclnnAvgPool3dBackwardGetWorkspaceSize:
    gradOutput, self, kernelSize, stride, padding, ceilMode,
    countIncludePad, divisorOverride, output.
    """

    def golden(
        grad_output,
        self_t,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        output_t=None,
        **kwargs,
    ):
        import torch

        k = _norm3(kernel_size, "NCDHW")
        s = _norm3(_default_stride(stride, kernel_size), "NCDHW")
        p = _norm6(padding)
        divisor = None if (divisor_override or 0) == 0 else int(divisor_override)
        go = (
            grad_output.float()
            if isinstance(grad_output, torch.Tensor)
            else torch.from_numpy(np.asarray(grad_output, dtype=np.float32))
        )
        st = (
            self_t.float()
            if isinstance(self_t, torch.Tensor)
            else torch.from_numpy(np.asarray(self_t, dtype=np.float32))
        )
        ndim = go.ndim
        if ndim == 4:
            go = go.unsqueeze(0)
            st = st.unsqueeze(0)
        res = _avg_pool_backward_3d_checked(
            go,
            st,
            list(k),
            list(s),
            [p[0], p[2], p[4]],
            bool(ceil_mode),
            bool(count_include_pad),
            divisor,
        )
        if ndim == 4:
            res = res[0]
        return [
            res.to(
                grad_output.dtype
                if isinstance(grad_output, torch.Tensor)
                else np.float32
            )
        ]

    class ThirdPartyImpl:
        def __init__(self, grad_output, **kwargs):
            self.out_dtype = grad_output.dtype

        def __call__(
            self,
            grad_output,
            self_t,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            **kwargs,
        ):
            k = _norm3(kernel_size, "NCDHW")
            s = _norm3(_default_stride(stride, kernel_size), "NCDHW")
            p = _norm6(padding)
            divisor = None if (divisor_override or 0) == 0 else int(divisor_override)
            go = grad_output
            st = self_t
            ndim = go.ndim
            if ndim == 4:
                go = go.unsqueeze(0)
                st = st.unsqueeze(0)
            res = _avg_pool_backward_3d_checked(
                go,
                st,
                list(k),
                list(s),
                [p[0], p[2], p[4]],
                bool(ceil_mode),
                bool(count_include_pad),
                divisor,
            )
            if ndim == 4:
                res = res[0]
            return [res.to(self.out_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }


class TorchAvgPool3dBackwardTestSpec:
    """torch e2e 测试规范（torch 入参，已在设备上）

    Parameters follow the torch dispatcher schema:
    avg_pool3d_backward(grad_output, self, kernel_size, stride, padding,
    ceil_mode, count_include_pad, divisor_override) -> Tensor.
    """

    def golden(
        grad_output,
        self_t,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        **kwargs,
    ):
        k = _norm3(kernel_size, "NCDHW")
        s = _norm3(_default_stride(stride, kernel_size), "NCDHW")
        p = _norm6(padding)
        divisor = None if (divisor_override or 0) == 0 else int(divisor_override)
        res = _avg_pool_backward_3d_checked(
            grad_output.float(),
            self_t.float(),
            list(k),
            list(s),
            [p[0], p[2], p[4]],
            bool(ceil_mode),
            bool(count_include_pad),
            divisor,
        )
        return [res.to(grad_output.dtype)]

    class ThirdPartyImpl:
        def __init__(self, grad_output, **kwargs):
            self.out_dtype = grad_output.dtype

        def __call__(
            self,
            grad_output,
            self_t,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            **kwargs,
        ):
            k = _norm3(kernel_size, "NCDHW")
            s = _norm3(_default_stride(stride, kernel_size), "NCDHW")
            p = _norm6(padding)
            divisor = None if (divisor_override or 0) == 0 else int(divisor_override)
            res = _avg_pool_backward_3d_checked(
                grad_output,
                self_t,
                list(k),
                list(s),
                [p[0], p[2], p[4]],
                bool(ceil_mode),
                bool(count_include_pad),
                divisor,
            )
            return [res.to(self.out_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
