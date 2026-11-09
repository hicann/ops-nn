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
"""E2E 层 golden 实现。

委托 ACLNN golden 做核心计算。
对应 torch API: torch_npu.npu_quant_matmul
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import qbmmv3_aclnn_golden as _aclnn


# ============================================================================
# helpers
# ============================================================================

# torch ScalarType enum → dtype 字符串
# npu_quant_matmul 的 output_dtype 参数使用 torch ScalarType int（不是 CANN dtype int）
_SCALAR_TYPE_TO_STR = {
    1: "int8",
    3: "int32",
    5: "float16",
    6: "float32",
    15: "bfloat16",
}


def _output_dtype_to_str(output_dtype):
    """将 output_dtype 参数（int 或 torch.dtype）转换为 dtype 字符串。

    int 值使用 torch ScalarType enum（不是 CANN dtype enum）。
    """
    if output_dtype is None:
        return None
    if isinstance(output_dtype, int):
        return _SCALAR_TYPE_TO_STR.get(output_dtype, str(output_dtype))
    # torch.dtype
    return _util.dtype_to_str(output_dtype)


# ACL_FLOAT4_E2M1：x1_dtype/x2_dtype 属性声明值（1 字节载体，2 个 fp4/字节）
_ACL_FLOAT4_E2M1 = 296
_FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _decode_fp4_nibble(code):
    val = _FP4_LUT[code & 0x7]
    return np.where(code & 0x8, -val, val).astype(np.float32)


def _decode_fp4_carrier(arr, dtype_code, axis, expect_dim):
    """uint8 载体 fp4 解码为 float32（x1_dtype/x2_dtype 属性=296 时生效）。

    两种载体约定（按 scale 逻辑维自动判定）：
      - 压缩 2 元素/字节（奇数打包维用例的 [L] 路线）：axis 维长 x2 == expect_dim，
        低半字节为前一个元素；
      - 1 元素/字节（无法压缩的奇数逻辑维）：直接按字节低半字节解码。
    """
    if arr is None or dtype_code != _ACL_FLOAT4_E2M1 or arr.dtype != np.uint8:
        return arr
    ax = axis % arr.ndim
    if expect_dim is not None and arr.shape[ax] * 2 == expect_dim:
        lo = _decode_fp4_nibble(arr & 0x0F)
        hi = _decode_fp4_nibble((arr >> 4) & 0x0F)
        stacked = np.stack([lo, hi], axis=ax + 1)
        new_shape = arr.shape[:ax] + (arr.shape[ax] * 2,) + arr.shape[ax + 1 :]
        return stacked.reshape(new_shape)
    if expect_dim == arr.shape[ax]:
        return _decode_fp4_nibble(arr & 0x0F)
    raise ValueError(
        f"FP4 carrier axis {arr.shape[ax]} cannot represent {expect_dim} elements"
    )


def _fp4_axes_and_k(x1, x2, a, b, a_dtype, b_dtype, pertoken, scale):
    """Resolve packed/unpacked K from both operands and MX group counts.

    For small K where both encodings fit the same scale group, the public
    two-elements-per-byte API convention takes precedence.
    """
    a_byte = a_dtype == _ACL_FLOAT4_E2M1 and a.dtype == np.uint8
    b_byte = b_dtype == _ACL_FLOAT4_E2M1 and b.dtype == np.uint8
    if not (a_byte or b_byte):
        return None
    if min(a.ndim, b.ndim) < 2:
        raise ValueError("FP4 matmul inputs must have at least two dimensions")

    def axis(tensor):
        strides = (
            tensor.stride()
            if hasattr(tensor, "stride")
            else tuple(s // tensor.itemsize for s in tensor.strides)
        )
        return -2 if strides[-2] == 1 and strides[-1] != 1 else -1

    aa, ba = axis(x1), axis(x2)
    a_k, b_k = a.shape[-1], b.shape[-2]
    candidates = ({a_k, 2 * a_k} if a_byte and aa == -1 else {a_k}) & (
        {b_k, 2 * b_k} if b_byte and ba == -2 else {b_k}
    )
    for tensor, kg_axis in ((pertoken, -2), (scale, -3)):
        if tensor is not None and tensor.ndim >= 3 and tensor.shape[-1] == 2:
            kg = tensor.shape[kg_axis]
            candidates = {k for k in candidates if (k + 63) // 64 == kg}
    if not candidates:
        raise ValueError("FP4 carrier K axes do not match the MX scale group counts")
    return aa, ba, max(candidates)


# ============================================================================
# TestSpec class
# ============================================================================


class TorchNpuQuantMatmulTestSpec:
    """torch_npu.npu_quant_matmul E2E golden。

    参数签名对齐 torch_npu.npu_quant_matmul schema:
      (x1, x2, scale, *, offset, pertoken_scale, bias,
       output_dtype, x1_dtype, x2_dtype, pertoken_scale_dtype,
       scale_dtype, group_sizes, y_scale) -> Tensor

    Golden 按逻辑视图计算；stride 仅用于确定 FP4 字节载体的打包轴。
    output_dtype 使用 torch ScalarType；x1_dtype/x2_dtype 声明 FP4 载体语义。
    """

    @staticmethod
    def golden(
        x1,
        x2,
        scale,
        *,
        offset=None,
        pertoken_scale=None,
        bias=None,
        output_dtype=None,
        x1_dtype=None,
        x2_dtype=None,
        pertoken_scale_dtype=None,
        scale_dtype=None,
        group_sizes=None,
        y_scale=None,
        fp4_packed_lastdim=None,
        **kwargs,
    ):
        # 1. 方向契约：e2e 通路交付的是逻辑视图值（转置语义已体现在视图数值排布中，
        #    strides 仅供框架在设备侧做派发）；而 kernel compute 采用「原始数组 +
        #    transpose 标志」约定，若此处再按 strides 推断标志传入会形成双重转置
        #    （K 维错位/广播失败）。故计算侧固定按非转置处理，M/N 从逻辑视图形状取。
        transpose_x1 = False
        transpose_x2 = False

        # 2. output_dtype → dtype 字符串
        out_dtype = _output_dtype_to_str(output_dtype)

        # 3. group_sizes → group_size int
        group_size = _util.pack_group_sizes(group_sizes)

        # 4. 委托 ACLNN _kernel_qbmm（绕过 golden() 的 _out_dtype fallback）
        # E2E 模式无 out tensor，output dtype 从 output_dtype 参数获取
        x1_np, x2_np, scale_np, offset_np, pertoken_np, bias_np = _util.to_numpy(
            x1, x2, scale, offset, pertoken_scale, bias
        )

        int4_carrier = False
        # int4 packed carrier：x1/x2 为 int32、每个 int32 沿 K 打包 8 个有符号 int4。
        # e2e 表未携带 convert_x_to_int4 属性，但 NPU 仍按 packed int4 解释，
        # golden 必须同步解包，否则会退化成 int32 matmul。
        if (
            x1_np is not None
            and x2_np is not None
            and x1_np.dtype == np.int32
            and x2_np.dtype == np.int32
            and x1_np.ndim == 2
            and x2_np.ndim == 2
        ):

            def _unpack_int4_carrier(arr):
                u = np.ascontiguousarray(arr).view(np.uint8).reshape(arr.shape + (4,))
                nib = np.empty(arr.shape + (8,), dtype=np.int16)
                for j in range(4):
                    byte = u[..., j].astype(np.int16)
                    nib[..., 2 * j] = byte & 0x0F
                    nib[..., 2 * j + 1] = (byte >> 4) & 0x0F
                return np.where(nib >= 8, nib - 16, nib).astype(np.int8)

            x1_np = _unpack_int4_carrier(x1_np).reshape(x1_np.shape[0], -1)
            x2_np = _unpack_int4_carrier(x2_np)
            x2_np = x2_np.transpose(0, 2, 1).reshape(x2_np.shape[0] * 8, x2_np.shape[1])
            int4_carrier = True

        # uint8 载体 e8m0（属性 scale_dtype/pertoken_scale_dtype=293）：
        # 解码为 float32 数值，并向 kernel golden 传 scale_dtype 提示，保持 MX 量化模式。
        scale_hint = None
        if scale_np is not None and int(scale_dtype or 0) == 293:
            scale_np = _util.e8m0_to_float(scale_np)
            scale_hint = "float8_e8m0"
        if pertoken_np is not None and int(pertoken_scale_dtype or 0) == 293:
            pertoken_np = _util.e8m0_to_float(pertoken_np)

        # fp4 uint8 载体解码（x1_dtype/x2_dtype=296）。
        # fp4_packed_lastdim 是图腿新增的显式末维打包约定；
        # 其余场景沿用上游的轴推理逻辑。
        if fp4_packed_lastdim:

            def _decode_fp4_packed(arr, dtype_code):
                if (
                    arr is None
                    or dtype_code != _ACL_FLOAT4_E2M1
                    or arr.dtype != np.uint8
                ):
                    return arr
                ax = arr.ndim - 1
                lo = _decode_fp4_nibble(arr & 0x0F)
                hi = _decode_fp4_nibble((arr >> 4) & 0x0F)
                stacked = np.stack([lo, hi], axis=ax + 1)
                new_shape = arr.shape[:ax] + (arr.shape[ax] * 2,)
                return stacked.reshape(new_shape)

            x1_np = _decode_fp4_packed(x1_np, x1_dtype)
            x2_np = _decode_fp4_packed(x2_np, x2_dtype)
        else:
            # 逻辑 M/N 从 pertoken/scale 获取；压缩轴由 _fp4_axes_and_k 统一推断。
            if pertoken_np is None or pertoken_np.ndim < 2:
                m_expect = None
            elif pertoken_np.ndim >= 3:
                m_expect = pertoken_np.shape[-3]
            else:
                m_expect = pertoken_np.shape[-2]
            if scale_np is None or scale_np.ndim < 2:
                n_expect = None
            elif scale_np.ndim >= 3:
                n_expect = scale_np.shape[-2]
            else:
                n_expect = scale_np.shape[-1]

            packed = _fp4_axes_and_k(
                x1, x2, x1_np, x2_np, x1_dtype, x2_dtype, pertoken_np, scale_np
            )
            if packed is not None:
                a_axis, b_axis, logical_k = packed
                x1_np = _decode_fp4_carrier(
                    x1_np, x1_dtype, a_axis, m_expect if a_axis == -2 else logical_k
                )
                x2_np = _decode_fp4_carrier(
                    x2_np, x2_dtype, b_axis, n_expect if b_axis == -1 else logical_k
                )

        deq_scale, u64_offset = _util.decode_u64_scale(scale_np)
        if u64_offset is not None and offset_np is None:
            offset_np = u64_offset

        # 从 x1/x2 shape 推断 output shape（用于 determine_quant_mode 的 M/N）
        if transpose_x1:
            m = x1_np.shape[-1]
        else:
            m = x1_np.shape[-2]
        if transpose_x2:
            n = x2_np.shape[-2]
        else:
            n = x2_np.shape[-1]
        out_shapes = ((m, n),) if x1_np.ndim >= 2 else ()

        return _aclnn._kernel_qbmm(
            x1_np,
            x2_np,
            deq_scale,
            offset_np,
            bias_np,
            pertoken_np,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            group_size=group_size,
            out_dtype=out_dtype,
            out_shapes=out_shapes,
            scale_dtype_hint=scale_hint,
            force_scale_gen=int4_carrier,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        scale,
        *,
        offset=None,
        pertoken_scale=None,
        bias=None,
        output_dtype=None,
        x1_dtype=None,
        x2_dtype=None,
        pertoken_scale_dtype=None,
        scale_dtype=None,
        group_sizes=None,
        y_scale=None,
        **kwargs,
    ):
        """E2E 模式的 customize_inputs（in-place 修改）。

        与 ACLNN customize_inputs 逻辑一致：
        1. E8M0 NaN 清洗
        2. UINT64 scale 合理化重新生成

        注：e2e 路径设备侧由 ttk 依 view 逻辑值做 format cast（NZ 排布由框架生成），
        无需改写 storage，golden 直接消费 view 值即可。
        """
        _aclnn._customize_inputs_impl(
            x1, x2, scale, offset, bias, pertoken_scale, **kwargs
        )


__spec__ = {
    "torch_npu.npu_quant_matmul": "TorchNpuQuantMatmulTestSpec",
}
