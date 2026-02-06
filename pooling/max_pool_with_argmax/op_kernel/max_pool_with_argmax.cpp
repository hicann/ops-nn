/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file max_pool_with_argmax.cpp
 * \brief max_pool_with_argmax implied
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "arch35/max_pool_with_argmax_nhwc_big_c.h"
#include "arch35/max_pool_with_argmax_nhwc_small_c.h"
#include "arch35/max_pool_with_argmax_simt.h"
#include "arch35/max_pool_with_argmax_struct_common.h"

#define MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT32_NCHW 500001
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT32_NHWC 500002
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT64_NCHW 500011
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT64_NHWC 500012

#define MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT32_NCHW_NANPROP 500101
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT32_NHWC_NANPROP 500102
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT64_NCHW_NANPROP 500111
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT64_NHWC_NANPROP 500112

#define MAX_POOL_WITH_ARGMAX_NHWC_SMALL_C 700001
#define MAX_POOL_WITH_ARGMAX_NHWC_SMALL_C_PAD 700002
#define MAX_POOL_WITH_ARGMAX_NHWC_SMALL_C_NANPROP 700011
#define MAX_POOL_WITH_ARGMAX_NHWC_SMALL_C_PAD_NANPROP 700012

#define MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C 800001
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_PAD 800002
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_NANPROP 800011
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_PAD_NANPROP 800012

constexpr uint32_t PAD_DISABLE = 0;
constexpr uint32_t PAD_ENABLE = 1;
constexpr uint32_t NCHW = 0;
constexpr uint32_t NHWC = 1;
constexpr uint32_t NANPROP_FALSE = 0;
constexpr uint32_t NANPROP_TRUE = 1;

extern "C" __global__ __aicore__ void max_pool_with_argmax(GM_ADDR x, GM_ADDR y, GM_ADDR argmax, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    TPipe pipeBase;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
    if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 800001", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxNHWC::MaxPoolWithArgmaxNhwCKernel<DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE, NANPROP_FALSE> op(&pipeBase, &tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_PAD)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 800002", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxNHWC::MaxPoolWithArgmaxNhwCKernel<DTYPE_X, DTYPE_ARGMAX, PAD_ENABLE, NANPROP_FALSE> op(&pipeBase, &tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_NANPROP)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 800011", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxNHWC::MaxPoolWithArgmaxNhwCKernel<DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE, NANPROP_TRUE> op(&pipeBase, &tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_PAD_NANPROP)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 800012", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxNHWC::MaxPoolWithArgmaxNhwCKernel<DTYPE_X, DTYPE_ARGMAX, PAD_ENABLE, NANPROP_TRUE> op(&pipeBase, &tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_NHWC_SMALL_C)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 700001", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSmallCNameSpace::MaxPoolWithArgmaxSmallC<DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE, NANPROP_FALSE> op(
            &pipeBase, &tilingDataIn);
        op.Init(x, y, argmax);
        op.MaxPoolWithArgmaxSmallCProcess();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_NHWC_SMALL_C_PAD)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 700002", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSmallCNameSpace::MaxPoolWithArgmaxSmallC<DTYPE_X, DTYPE_ARGMAX, PAD_ENABLE, NANPROP_FALSE> op(
            &pipeBase, &tilingDataIn);
        op.Init(x, y, argmax);
        op.MaxPoolWithArgmaxSmallCProcess();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_NHWC_SMALL_C_NANPROP)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 700011", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSmallCNameSpace::MaxPoolWithArgmaxSmallC<DTYPE_X, DTYPE_ARGMAX, PAD_DISABLE, NANPROP_TRUE> op(
            &pipeBase, &tilingDataIn);
        op.Init(x, y, argmax);
        op.MaxPoolWithArgmaxSmallCProcess();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_NHWC_SMALL_C_PAD_NANPROP)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 700012", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSmallCNameSpace::MaxPoolWithArgmaxSmallC<DTYPE_X, DTYPE_ARGMAX, PAD_ENABLE, NANPROP_TRUE> op(
            &pipeBase, &tilingDataIn);
        op.Init(x, y, argmax);
        op.MaxPoolWithArgmaxSmallCProcess();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT32_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500001", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSimtNamespace::MaxPoolWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NCHW, false, NANPROP_FALSE> op(&tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT32_NHWC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500002", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSimtNamespace::MaxPoolWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NHWC, false, NANPROP_FALSE> op(&tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT32_NCHW_NANPROP)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500101", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSimtNamespace::MaxPoolWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NCHW, false, NANPROP_TRUE> op(&tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT32_NHWC_NANPROP)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500102", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSimtNamespace::MaxPoolWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NHWC, false, NANPROP_TRUE> op(&tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT64_NCHW)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500011", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSimtNamespace::MaxPoolWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NCHW, true, NANPROP_FALSE> op(&tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT64_NHWC)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500012", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSimtNamespace::MaxPoolWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NHWC, true, NANPROP_FALSE> op(&tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT64_NCHW_NANPROP)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500111", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSimtNamespace::MaxPoolWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NCHW, true, NANPROP_TRUE> op(&tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_SIMT_INT64_NHWC_NANPROP)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 500112", MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData);
        GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData, tilingDataIn, tiling);
        MaxPoolWithArgmaxSimtNamespace::MaxPoolWithArgmaxSimt<DTYPE_X, DTYPE_ARGMAX, NHWC, true, NANPROP_TRUE> op(&tilingDataIn);
        op.Init(x, y, argmax);
        op.Process();
    }
}