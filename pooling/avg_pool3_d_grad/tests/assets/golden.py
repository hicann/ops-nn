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

import numpy as np

__golden__ = {
    "kernel": {"avg_pool3_d_grad": "avg_pool3_d_grad_golden"},
    "aclnn": {
        "aclnnAvgPool3dBackward": "_aclnn_avg_pool3d_backward_golden",
    },
    "e2e": {
        "torch.ops.aten.avg_pool3d_backward": "_aclnn_avg_pool3d_backward_golden",
        "torch.ops.cann_ops_nn.avg_pool3d_backward": "_aclnn_avg_pool3d_backward_golden",
    },
}


def _acl_torch_f32(ref):
    """按输入形态转成 fp32 torch Tensor（兼容 torch.Tensor / numpy 数组）。"""
    import torch

    if isinstance(ref, torch.Tensor):
        return ref.detach().float()
    return torch.from_numpy(np.asarray(ref, dtype=np.float32))


def _acl_back_dtype(res, ref):
    """把 fp32 结果转回参考 dtype（torch 同 dtype；numpy 保 dtype）。"""
    import torch

    arr = res.detach().float().cpu()
    if isinstance(ref, torch.Tensor):
        return arr.to(ref.dtype)
    return arr.numpy().astype(ref.dtype)


def _acl_norm3(values, ndim3):
    """长度 1 广播到 ndim3(=3 或 2)，其余长度原样（torch 侧仅收 3 或 2）。

    标量（int/bool/None）按单值广播，避免上游错位传参时出现难查的
    TypeError/TypeError("bool object is not iterable")。
    """
    if values is None:
        return [0] * ndim3
    if isinstance(values, (bool, int)):
        return [int(values)] * ndim3
    v = list(values)
    if len(v) == 1:
        return [v[0]] * ndim3
    return v


def _pool_spatial_norm(values, nd):
    """归一化到 nd 个池化空间维（长度 1 广播，超长取末尾 nd 个）。"""
    v = [int(x) for x in values]
    if len(v) == 1:
        return [v[0]] * nd
    if len(v) >= nd:
        return v[-nd:]
    raise ValueError("pool attr length %d < %d" % (len(v), nd))


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
    """numpy 向量版，语义对齐 torch.ops.aten.avg_pool{2,3}d_backward（对称 padding）。"""
    import numpy as np

    grad = np.asarray(grad, dtype=np.float64)
    self_shape = tuple(int(x) for x in self_shape)
    pool_nd = int(pool_nd)
    batch_nd = grad.ndim - pool_nd
    if batch_nd < 0 or len(self_shape) != grad.ndim:
        raise ValueError(
            "grad/self shape mismatch: %s vs %s" % (grad.shape, self_shape)
        )
    kernel = _pool_spatial_norm(kernel_size, pool_nd)
    stride = _pool_spatial_norm(stride, pool_nd)
    pad = _pool_spatial_norm(padding, pool_nd)
    spatial_in = self_shape[-pool_nd:]
    out_dims = tuple(int(x) for x in grad.shape[-pool_nd:])

    # divisor：divisor_override 时全图取该值，否则为各轴逐输出计数乘积
    # （include_pad 计到 pad 右沿，否则只数落在输入内的坐标）。
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

    # 逐轴前缀和窗口收拢：输出空间逐轴变回输入空间。
    for i in range(pool_nd):
        ax = -(pool_nd - i)
        state = np.moveaxis(state, ax, -1)  # (..., O_i)
        o_num = state.shape[-1]
        x = np.arange(int(spatial_in[i]))
        lo = (x + int(pad[i]) - int(kernel[i])) // int(stride[i]) + 1
        hi = (x + int(pad[i])) // int(stride[i]) + 1  # 前缀结束索引，范围 [0, O_i]
        lo = np.clip(lo, 0, o_num)
        hi = np.clip(hi, 0, o_num)
        pre = np.concatenate(
            [np.zeros(state.shape[:-1] + (1,)), np.cumsum(state, axis=-1)], axis=-1
        )  # (..., O_i + 1)
        out = pre[..., hi] - pre[..., lo]  # (..., I_i)，无覆盖输出的坐标 hi==lo
        state = np.where(hi > lo, out, 0.0)
        state = np.moveaxis(state, -1, ax)

    return state.astype(np.float32)


def _avg_pool_backward_3d_checked(
    grad,
    self,
    kernel,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor,
):
    """3D avg-pool backward：空间尺寸不足时 torch 抛
    'input image smaller than kernel size'，提前检测改走 numpy 参考实现。
    """
    import torch

    if _needs_numpy_pool_backward(self.shape, kernel, 3):
        np_ref = _avg_pool_backward_np(
            grad.float().detach().cpu().numpy(),
            tuple(self.shape),
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
        self,
        [int(x) for x in kernel][-3:],
        [int(x) for x in stride][-3:],
        [int(x) for x in padding][-3:],
        bool(ceil_mode),
        bool(count_include_pad),
        divisor,
    )


def _aclnn_avg_pool3d_backward_golden(
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
    """aclnnAvgPool3dBackward 的 torch 参考（0→None 对标 torch 语义）。

    位置参数顺序与 aclnnAvgPool3dBackward 头文件实参顺序一致（output 在最后）：
    (gradOutput, self, kernelSize, stride, padding,
     ceilMode, countIncludePad, divisorOverride, output)
    """

    k = _acl_norm3(kernel_size, 3)
    s = _acl_norm3(stride, 3)
    p = _acl_norm3(padding, 3)
    divisor = None if (divisor_override or 0) == 0 else int(divisor_override)

    go = _acl_torch_f32(grad_output)
    st = _acl_torch_f32(self_t)
    ndim = go.ndim
    if ndim == 4:  # (C,D,H,W) -> (1,C,D,H,W)
        go = go.unsqueeze(0)
        st = st.unsqueeze(0)
    res = _avg_pool_backward_3d_checked(
        go,
        st,
        list(k),
        list(s),
        list(p),
        bool(ceil_mode),
        bool(count_include_pad),
        divisor,
    )
    if ndim == 4:
        res = res[0]
    return _acl_back_dtype(res, grad_output)


def _aclnn_avg_pool2d_backward_golden(
    grad_output,
    self_t,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    cube_math_type=None,
    output_t=None,
    **kwargs,
):
    """aclnnAvgPool2dBackward 的 torch 参考（0→None 对标 torch 语义）。

    位置参数顺序与 aclnnAvgPool2dBackward 头文件实参顺序一致（output 在最后）：
    (gradOutput, self, kernelSize, stride, padding, ceilMode,
     countIncludePad, divisorOverride, cubeMathType, output)
    """
    import torch

    k = _acl_norm3(kernel_size, 2)
    s = _acl_norm3(stride, 2)
    p = _acl_norm3(padding, 2)
    divisor = None if (divisor_override or 0) == 0 else int(divisor_override)

    go = _acl_torch_f32(grad_output)
    st = _acl_torch_f32(self_t)
    res = torch.ops.aten.avg_pool2d_backward(
        go,
        st,
        list(k),
        list(s),
        list(p),
        bool(ceil_mode),
        bool(count_include_pad),
        divisor,
    )
    return _acl_back_dtype(res, grad_output)


def avg_pool3_d_grad_golden(
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
    """
    Golden function for avg_pool3_d_grad.
    All the parameters (names and order) follow @avg_pool3_d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    if _is_arch35(kwargs.get("short_soc_version", "")):
        return avg_pool3_d_grad_golden_arch35(
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
            "avgpool3dgrad testcase golden function supports NDC1HWC0 input only!"
        )
    # Collect shape info
    n_index = data_format.index("N")
    d_index = data_format.index("D")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    c_index = data_format.index("C")
    stride_h, stride_w, stride_d = strides[h_index], strides[w_index], strides[d_index]
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
    if all(i == 0 for i in pads):
        padding = "VALID"
    else:
        padding = "SAME"

    # grads to NDHWC
    output_backprop = grads.transpose(0, 1, 3, 4, 2, 5).reshape(GN, GD, GH, GW, GC * C0)
    grads_tensor = tf.compat.v1.placeholder(grads.dtype, shape=output_backprop.shape)
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


def _is_arch35(soc):
    """Return True when the target SoC belongs to the arch35 (Ascend950/Ascend350 or Ascend910B/910_93) series."""
    s = str(soc).lower()
    return any(k in s for k in ("ascend950"))


def _parse_pads(pads):
    """Normalize pads to (d_front, d_back, h_top, h_bottom, w_left, w_right)."""
    if len(pads) == 6:
        return tuple(int(p) for p in pads)
    if len(pads) == 3:
        return (
            int(pads[0]),
            int(pads[0]),
            int(pads[1]),
            int(pads[1]),
            int(pads[2]),
            int(pads[2]),
        )
    val = int(pads[0])
    return (val, val, val, val, val, val)


def _norm3(v, data_format):
    """Normalize ksize/strides of length 1/3/5 to [kD, kH, kW]."""
    v = [int(x) for x in v]
    if len(v) == 1:
        return [v[0], v[0], v[0]]
    if len(v) >= 5:  # [N,C,D,H,W] (NCDHW) or [N,D,H,W,C] (NDHWC)
        return [v[2], v[3], v[4]] if data_format == "NCDHW" else [v[1], v[2], v[3]]
    return v[:3]


def _norm6(p):
    """Normalize pads of length 1/3/6 to [dL, dR, hT, hB, wL, wR] (symmetric forms expanded)."""
    p = [int(x) for x in p]
    if len(p) == 6:
        return p
    if len(p) == 3:
        return [p[0], p[0], p[1], p[1], p[2], p[2]]
    if len(p) != 1:
        raise ValueError(
            "pads must have length 1/3/6, got length %d (values=%s); "
            "silently using only the first value hid real misalignment before."
            % (len(p), p)
        )
    return [p[0]] * 6


def avg_pool3_d_grad_golden_arch35(
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
    """
    arch35 golden for avg_pool3_d_grad.

    Supports 4-dim (no batch) and 5-dim tensors in both NCDHW and NDHWC layouts,
    matching the arch35 tiling which accepts 4/5-dim tensors. ksize/strides accept
    length 1/3/5, pads accept length 1/3/6.

    Padding resolution:
      Only symmetric pads are supported. ``torch.ops.aten.avg_pool3d_backward``
      only accepts symmetric padding; non-symmetric pads raise an error.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    orig_shape = [int(v) for v in orig_input_shape]
    ndim = len(orig_shape)
    k = _norm3(ksize, data_format)
    s = _norm3(strides, data_format)
    p = _norm6(pads)
    kd, kh, kw = k
    sd, sh, sw = s

    if data_format == "NCDHW":
        d, h, w = orig_shape[-3], orig_shape[-2], orig_shape[-1]
    else:  # NDHWC
        if ndim == 5:
            d, h, w = orig_shape[1], orig_shape[2], orig_shape[3]
        else:  # NDHWC 4D: [D,H,W,C]
            d, h, w = orig_shape[0], orig_shape[1], orig_shape[2]

    # --- DEBUG LOG ---
    np_arr = np.asarray(grads)
    import sys as _sys

    print(
        "[GOLDEN-DBG] testcase=%s orig_shape=%s ksize=%s strides=%s pads=%s fmt=%s "
        "ceil=%s cip=%s div=%s d/h/w=%d/%d/%d"
        % (
            kwargs.get("testcase_name", "?"),
            orig_shape,
            k,
            s,
            p,
            data_format,
            ceil_mode,
            count_include_pad,
            divisor_override,
            d,
            h,
            w,
        ),
        file=_sys.stderr,
    )
    print(
        "[GOLDEN-DBG] grads shape=%s dtype=%s inf=%d nan=%d min=%s max=%s"
        % (
            np_arr.shape,
            np_arr.dtype,
            int(np.isinf(np_arr).sum()),
            int(np.isnan(np_arr).sum()),
            np.nanmin(np_arr) if np_arr.size else "?",
            np.nanmax(np_arr) if np_arr.size else "?",
        ),
        file=_sys.stderr,
    )

    # torch.from_numpy cannot take ml_dtypes bfloat16 arrays; lift to float32 for the torch graph.
    grads_f32 = np.asarray(grads, dtype=np.float32)

    is_sym = p[0] == p[1] and p[2] == p[3] and p[4] == p[5]
    if not is_sym:
        raise RuntimeError(
            "arch35 torch golden only supports symmetric pads; got pads=%s" % p
        )
    # Normalize grads to torch NCDHW 5D layout.
    if data_format == "NCDHW":
        grads_t = torch.from_numpy(grads_f32)
        grads_t = grads_t.unsqueeze(0) if ndim == 4 else grads_t
    else:  # NDHWC
        grads_t = torch.from_numpy(grads_f32)
        grads_t = (
            grads_t.permute(0, 4, 1, 2, 3)
            if ndim == 5
            else grads_t.permute(3, 0, 1, 2).unsqueeze(0)
        )
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
    grad_input = grad_input.detach().numpy()
    if data_format == "NCDHW":
        grad_input = grad_input if ndim == 5 else grad_input[0]
    else:  # NDHWC
        grad_input = (
            np.transpose(grad_input, (0, 2, 3, 4, 1))
            if ndim == 5
            else grad_input[0].transpose(1, 2, 3, 0)
        )
    res = grad_input

    res = np.asarray(res)
    import sys as _sys

    print(
        "[GOLDEN-DBG] out shape=%s dtype=%s inf=%d nan=%d min=%s max=%s"
        % (
            res.shape,
            res.dtype,
            int(np.isinf(res).sum()),
            int(np.isnan(res).sum()),
            np.nanmin(res) if res.size else "?",
            np.nanmax(res) if res.size else "?",
        ),
        file=_sys.stderr,
    )

    out_dtype = str(kwargs.get("output_dtypes", ["float32"])[0]).lower()
    if out_dtype == "float16":
        return res.astype(np.float16)
    if "bfloat16" in out_dtype:
        try:
            from ml_dtypes import bfloat16

            return res.astype(bfloat16)
        except ImportError:
            return res.astype(np.float32)
    return res.astype(np.float32)


# ---------------------------------------------------------------------------
# E2E TestSpec (torch). Merged from avg_pool3_d_grad_graph_spec.py.
# __spec__ is scanned by TestSpecManager; __golden__ above by CustomPluginManager,
# the two loaders are independent, so both can live in this file.
# ---------------------------------------------------------------------------
# import torch  # noqa: E402
# import torch.nn as nn  # noqa: E402

# try:
#     import cann_ops_nn  # noqa: F401  # ensure torch.ops.cann_ops_nn.avg_pool3d_backward is registered
# except ImportError:
#     pass

# __spec__ = {
#     "torch.ops.aten.avg_pool3d_backward": "AvgPool3DGradTestSpec",
#     "torch.ops.cann_ops_nn.avg_pool3d_backward": "CannOpsNnAvgPool3dBackwardTestSpec",
# }


# class AvgPool3DGradTestSpec:
#     @staticmethod
#     def golden(
#         grad_output,
#         self_tensor,
#         kernel_size,
#         stride,
#         padding,
#         ceil_mode,
#         count_include_pad,
#         divisor_override,
#         **kwargs,
#     ):
#         orig_dtype = grad_output.dtype
#         # 对标 torch：divisor_override=0 表示默认（不覆盖），传给 aten 用 None
#         divisor = None if (divisor_override or 0) == 0 else int(divisor_override)
#         out = _avg_pool_backward_3d_checked(
#             grad_output.float(),
#             self_tensor.float(),
#             kernel_size,
#             stride,
#             padding,
#             ceil_mode,
#             count_include_pad,
#             divisor,
#         )
#         return [out.to(orig_dtype)]


# class CannOpsNnAvgPool3dBackwardGraph(nn.Module):
#     def forward(
#         self,
#         grad_output,
#         self_tensor,
#         kernel_size,
#         stride,
#         padding,
#         ceil_mode,
#         count_include_pad,
#         divisor_override,
#     ):
#         return torch.ops.cann_ops_nn.avg_pool3d_backward(
#             grad_output,
#             self_tensor,
#             list(kernel_size),
#             list(stride),
#             list(padding),
#             ceil_mode,
#             count_include_pad,
#             divisor_override,
#         )


# class CannOpsNnAvgPool3dBackwardTestSpec:
#     torch_graph = CannOpsNnAvgPool3dBackwardGraph

#     @staticmethod
#     def golden(
#         grad_output,
#         self_tensor,
#         kernel_size,
#         stride,
#         padding,
#         ceil_mode,
#         count_include_pad,
#         divisor_override,
#         **kwargs,
#     ):
#         orig_dtype = grad_output.dtype
#         # 对标 torch：divisor_override=0 表示默认（不覆盖），传给 aten 用 None
#         divisor = None if (divisor_override or 0) == 0 else int(divisor_override)
#         out = _avg_pool_backward_3d_checked(
#             grad_output.float(),
#             self_tensor.float(),
#             kernel_size,
#             stride,
#             padding,
#             ceil_mode,
#             count_include_pad,
#             divisor,
#         )
#         return [out.to(orig_dtype)]
