#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------

"""AscendAntiQuant TTK TestSpec（golden / third_party / tolerance）。

算子语义（per-tensor 反量化，scale/offset 为标量属性）：
    sqrt_mode == False : y = cast((x + offset) * scale)
    sqrt_mode == True  : y = cast((x + offset) * scale * scale)
    dtype = 0 -> float32（缺省，与 OpDef/GEIR 原型一致）, dtype = 1 -> float16

输入 x 支持 int8 / float8_e5m2 / float8_e4m3fn / hifloat8，统一解码到
float64 后按上式求真值，最后一次性 cast 到输出 dtype（不提前舍入）。

golden 仅依赖 numpy 与扩展 dtype 包（ml_dtypes / en_dtypes），不 import
tensorflow：golden.py 会被原样同步到 torch-only 的远端 GPU 端点执行，且
TTK worker 进程已加载 CANN/TBE，进程内 import tensorflow 会因符号冲突崩溃。
"""

import numpy as np

__spec__ = {
    "ascend_anti_quant": "AscendAntiQuantTestSpec",
    "AscendAntiQuant": "AscendAntiQuantTestSpec",
}

# dtype 别名 -> 规范名；未命中时原样返回。
_DTYPE_ALIASES = {
    "f8_e5m2": "float8_e5m2",
    "fp8_e5m2": "float8_e5m2",
    "f8_e4m3fn": "float8_e4m3fn",
    "fp8_e4m3fn": "float8_e4m3fn",
    "hif8": "hifloat8",
    "hifp8": "hifloat8",
}

# 未携带语义的裸字节存储（需要声明 dtype 才能还原）。
_RAW_DTYPE_NAMES = ("void8", "void", "uint8")


class AscendAntiQuantTestSpec:
    """AscendAntiQuant 的 golden / 三方实现 / 精度判据声明。"""

    @staticmethod
    def _dtype_hint(dtype_hint):
        """TTK 传入的 input_dtypes 可能是 tuple/list/单值，取第一个。"""
        if dtype_hint is None:
            return ""
        if isinstance(dtype_hint, (tuple, list)):
            dtype_hint = dtype_hint[0] if dtype_hint else ""
        return str(dtype_hint)

    @staticmethod
    def _canonical_dtype(name):
        """dtype 别名归一化（TTK/CSV 可能用 fp8_e4m3fn、hif8 等写法）。"""
        name = str(name).lower()
        return _DTYPE_ALIASES.get(name, name)

    @staticmethod
    def _decode_to_float64(x, dtype_hint=None):
        """量化输入统一解码为 float64（声明 dtype 为准 + 严格校验）。

        量化输入的位布局互不兼容（hifloat8 / float8_e5m2 / float8_e4m3fn），
        且 kind == "V" 并不代表 dtype 丢失（ml_dtypes 的 float8_e4m3fn、
        en_dtypes 的 hifloat8 注册的 kind 都是 V），因此不能按 dtype.kind 猜。
        规则：
        - 以 kwargs['input_dtypes'] 声明的 dtype 为准；
        - 数组自身 dtype 可识别时，必须与声明一致，否则直接报错；
        - 裸 void/uint8 才允许按声明 dtype 解位模式；无声明时拒绝猜测。
        """
        arr = np.asarray(x)
        declared = AscendAntiQuantTestSpec._canonical_dtype(
            AscendAntiQuantTestSpec._dtype_hint(dtype_hint)
        )
        name = AscendAntiQuantTestSpec._canonical_dtype(
            getattr(arr.dtype, "name", arr.dtype)
        )
        is_raw = name in _RAW_DTYPE_NAMES

        if declared:
            if not is_raw and name != declared:
                raise TypeError(
                    f"input array dtype {name!r} != declared dtype {declared!r}; "
                    "check CSV input_dtypes"
                )
            tag = declared
        elif is_raw:
            raise TypeError(
                "raw (void/uint8) quantized input requires declared dtype; "
                "TTK passes it via kwargs['input_dtypes']"
            )
        else:
            tag = name

        if tag == "int8":
            if is_raw:
                return np.ascontiguousarray(arr).view(np.int8).astype(np.float64)
            return arr.astype(np.float64)

        raw = np.ascontiguousarray(arr).view(np.uint8)
        if tag == "float8_e5m2":
            from ml_dtypes import float8_e5m2

            return raw.view(float8_e5m2).astype(np.float64)
        if tag == "float8_e4m3fn":
            from ml_dtypes import float8_e4m3fn

            return raw.view(float8_e4m3fn).astype(np.float64)
        if tag == "hifloat8":
            from en_dtypes import hifloat8

            return raw.view(hifloat8).astype(np.float64)
        raise TypeError(f"unsupported quantized input dtype {tag!r}")

    @staticmethod
    def golden(
        x,
        *,
        scale=1.0,
        offset=0.0,
        dtype=0,
        sqrt_mode=False,
        input_dtypes=None,
        **kwargs,
    ):
        x_fp64 = AscendAntiQuantTestSpec._decode_to_float64(x, input_dtypes)

        s = float(scale * scale if sqrt_mode else scale)
        off = float(offset)

        y = (x_fp64 + off) * s

        return [y.astype(np.float16 if dtype == 1 else np.float32)]

    class TorchImpl:
        """torch 三方竞品实现：(x + offset) * scale。

        按算子规格在 float32 上计算（模拟真实竞品 GPU 实现）：fp32 加法后
        逐次乘 scale；sqrt_mode 为两次独立的 fp32 乘法，而非先算
        s_eff=scale*scale 再乘——两种舍入路径在 fp32 下有 1-ulp 级差异。
        golden 保持 float64 真值作基准。
        """

        def __init__(
            self, *, scale=1.0, offset=0.0, dtype=0, sqrt_mode=False, **kwargs
        ):
            self.scale = float(scale)
            self.offset = float(offset)
            self.dtype = int(dtype)
            self.sqrt_mode = bool(sqrt_mode)

        def __call__(self, x, **kwargs):
            import torch

            s = float(np.float32(self.scale))
            y = x.to(torch.float32) + self.offset
            y = y * s
            if self.sqrt_mode:
                y = y * s
            if self.dtype == 1:
                y = y.to(torch.float16)
            else:
                y = y.to(torch.float32)
            return [y]

    third_party = {
        "torch": TorchImpl,
    }

    tolerance = {
        "float32": {"standard": "cross_check"},
        "float16": {"standard": "cross_check"},
    }
