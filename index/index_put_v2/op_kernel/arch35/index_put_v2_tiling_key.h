/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_tiling_key.h
 * \brief index tiling key declare
 */

#ifndef INDEX_PUT_V2_TILING_KEY_H
#define INDEX_PUT_V2_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
// Binary build copies op_kernel under tbe/ascendc.
#if __has_include("../../index/arch35/index_tiling_data.h")
#include "../../index/arch35/index_tiling_data.h"
#else
#include "../../../index/op_kernel/arch35/index_tiling_data.h"
#endif
#include "index_put_v2_tiling_data.h"

namespace IndexPutV2 {

#define INDEX_PUT_V2_TPL_B8 1    // 8位数据类型
#define INDEX_PUT_V2_TPL_B16 2   // 16位数据类型
#define INDEX_PUT_V2_TPL_B32 4   // 32位数据类型
#define INDEX_PUT_V2_TPL_B64 8   // 64位数据类型
#define INDEX_PUT_V2_TPL_B128 16 // 128位数据类型

// 模板参数声明
ASCENDC_TPL_ARGS_DECL(IndexPutV2,
    ASCENDC_TPL_DTYPE_DECL(X_DTYPE, 0, INDEX_PUT_V2_TPL_B8, INDEX_PUT_V2_TPL_B16,
        INDEX_PUT_V2_TPL_B32, INDEX_PUT_V2_TPL_B64, INDEX_PUT_V2_TPL_B128),
    ASCENDC_TPL_DTYPE_DECL(FULL_LOAD_TYPE, 0),
    ASCENDC_TPL_BOOL_DECL(IS_PERF, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_SIMD, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_NOCON, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_ACCUMULATE, 0, 1),
    ASCENDC_TPL_BOOL_DECL(IS_OVERLENGTH, 0, 1)
);

// 模板选择器
ASCENDC_TPL_SEL(
    // index_tiling_arch35.cpp: 普通 SIMT 模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DTYPE_SEL(X_DTYPE, INDEX_PUT_V2_TPL_B8, INDEX_PUT_V2_TPL_B16, INDEX_PUT_V2_TPL_B32,
            INDEX_PUT_V2_TPL_B64, INDEX_PUT_V2_TPL_B128),
        ASCENDC_TPL_DTYPE_SEL(FULL_LOAD_TYPE, 0),
        ASCENDC_TPL_BOOL_SEL(IS_PERF, 0),
        ASCENDC_TPL_BOOL_SEL(IS_SIMD, 0),
        ASCENDC_TPL_BOOL_SEL(IS_NOCON, 0),
        ASCENDC_TPL_BOOL_SEL(IS_ACCUMULATE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_OVERLENGTH, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(Index::IndexSimtTilingData)
    ),

    // index_tiling_no_continuous.cpp: 非连续模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DTYPE_SEL(X_DTYPE, 0),
        ASCENDC_TPL_DTYPE_SEL(FULL_LOAD_TYPE, 0),
        ASCENDC_TPL_BOOL_SEL(IS_PERF, 0),
        ASCENDC_TPL_BOOL_SEL(IS_SIMD, 0),
        ASCENDC_TPL_BOOL_SEL(IS_NOCON, 1),
        ASCENDC_TPL_BOOL_SEL(IS_ACCUMULATE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_OVERLENGTH, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(Index::IndexNonContinuousTilingData)
    ),

    // index_put_v2_simd_tiling.cpp: SIMD 模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DTYPE_SEL(X_DTYPE, 0),
        ASCENDC_TPL_DTYPE_SEL(FULL_LOAD_TYPE, 0),
        ASCENDC_TPL_BOOL_SEL(IS_PERF, 0),
        ASCENDC_TPL_BOOL_SEL(IS_SIMD, 1),
        ASCENDC_TPL_BOOL_SEL(IS_NOCON, 0),
        ASCENDC_TPL_BOOL_SEL(IS_ACCUMULATE, 0, 1),
        ASCENDC_TPL_BOOL_SEL(IS_OVERLENGTH, 0),
        ASCENDC_TPL_TILING_STRUCT_SEL(IndexPutV2SimdTilingData)
    )
);

} // namespace IndexPutV2

#endif // INDEX_PUT_V2_TILING_KEY_H
