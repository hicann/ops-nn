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
"""
matmul golden 公共入口（总 util / 纯汇总门面）。

matmul 类算子的 golden 资产只需引用本模块，即可调用全部公共功能函数。
本模块不定义任何函数，只做 5 个功能模块的统一 re-export：

  matmul_dtype_util      dtype 常量、名称解析与 cast、精度截断
  matmul_rearrange_util  ND/NZ 排布（transdata）、transpose 等结构变换
  matmul_torch_util      torch <-> numpy 桥接（torch_to_numpy / to_numpy /
                         write_back）
  matmul_quant_util      量化参数编解码、scale 生成/规整、weight pack/unpack
  matmul_compare_util    isclose / numeric ULP 数据对比

依赖方向为单向（入口 -> 功能模块；功能模块之间仅允许叶子方向依赖），
功能模块不反向引用本模块，因此任何模块都可在任意顺序下被模块级导入，
无需函数级延迟导入。
"""

from matmul_dtype_util import (
    np_fp4_e2m1,
    np_fp4_e1m2,
    np_hif8,
    np_fp8_e8m0,
    np_bfloat16,
    np_fp8_e4m3,
    np_fp8_e5m2,
    np_int4,
    DTypeTable,
    dtype_to_str,
    dtype_to_np,
    cast_output_dtype,
    get_intermediate_dtype,
    truncate,
    hf32_truncate_np,
    e8m0_to_float,
)
from matmul_rearrange_util import (
    transpose_last_two_dims,
    gen_axes_for_transpose,
    detect_transpose_from_strides,
    nz_to_nd,
    get_nz_n0,
    nd_to_nz,
    overwrite_storage,
    nz_storage_to_nd,
    prepare_nz_input,
    nz_input_to_nd,
)
from matmul_torch_util import (
    torch_to_numpy,
    to_numpy,
    write_back,
)
from matmul_quant_util import (
    ceil_div,
    scale_generate,
    f32_2_s9,
    needs_scale_generate,
    sanitize_e8m0_scale,
    sanitize_e8m0_carrier,
    u64_to_deq_scale,
    u64_to_offset,
    pack_u64_scale,
    generate_u64_scale,
    decode_u64_scale,
    unpack_groupsize,
    get_effective_group_size,
    pack_group_sizes,
    pack_int32_from_int4,
    pack_int32_from_fp4,
    unpack_int32_to_int4,
    unpack_float_to_fp4,
    FP4_E2M1_TABLE,
)
from matmul_compare_util import (
    isclose_compare,
    numeric_ulp_compare,
)


__all__ = [
    # dtype
    "np_fp4_e2m1",
    "np_fp4_e1m2",
    "np_hif8",
    "np_fp8_e8m0",
    "np_bfloat16",
    "np_fp8_e4m3",
    "np_fp8_e5m2",
    "np_int4",
    "DTypeTable",
    "dtype_to_str",
    "dtype_to_np",
    "cast_output_dtype",
    "get_intermediate_dtype",
    "truncate",
    "hf32_truncate_np",
    "e8m0_to_float",
    # rearrange
    "transpose_last_two_dims",
    "gen_axes_for_transpose",
    "detect_transpose_from_strides",
    "nz_to_nd",
    "get_nz_n0",
    "nd_to_nz",
    "overwrite_storage",
    "nz_storage_to_nd",
    "prepare_nz_input",
    "nz_input_to_nd",
    # quant
    "ceil_div",
    "scale_generate",
    "f32_2_s9",
    "needs_scale_generate",
    "sanitize_e8m0_scale",
    "sanitize_e8m0_carrier",
    "u64_to_deq_scale",
    "u64_to_offset",
    "pack_u64_scale",
    "generate_u64_scale",
    "decode_u64_scale",
    "unpack_groupsize",
    "get_effective_group_size",
    "pack_group_sizes",
    "pack_int32_from_int4",
    "pack_int32_from_fp4",
    "unpack_int32_to_int4",
    "unpack_float_to_fp4",
    "FP4_E2M1_TABLE",
    # compare
    "isclose_compare",
    "numeric_ulp_compare",
    # torch bridge
    "torch_to_numpy",
    "write_back",
    "to_numpy",
]
