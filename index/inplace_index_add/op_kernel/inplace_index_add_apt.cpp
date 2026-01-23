/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_add.cpp
 * \brief inplace_index_add
 */
#include "arch35/inplace_index_add_simt.h"
#include "arch35/inplace_index_add_determinstic.h"
#include "arch35/inplace_index_add_determinstic_notquant.h"
#include "arch35/inplace_index_add_simd.h"
#include "arch35/inplace_index_add_simd_sort.h"
#include "arch35/inplace_index_add_simt_sort.h"

using namespace InplaceIndexAdd;

#define IIA_UINT32_FP32 100000
#define IIA_UINT32_FP16 100001
#define IIA_UINT32_INT8 100002
#define IIA_UINT32_INT32 100003
#define IIA_UINT32_UINT8 100004
#define IIA_UINT32_INT16 100006
#define IIA_UINT32_BF16 100027
#define IIA_UINT32_INT64 100009
#define IIA_UINT32_BOOL 100012

#define IIA_UINT64_FP32 100100
#define IIA_UINT64_FP16 100101
#define IIA_UINT64_INT8 100102
#define IIA_UINT64_INT32 100103
#define IIA_UINT64_UINT8 100104
#define IIA_UINT64_INT16 100106
#define IIA_UINT64_BF16 100127
#define IIA_UINT64_INT64 100109
#define IIA_UINT64_BOOL 100112

#define IIA_WITH_ALPHA_UINT32_FP32 101000
#define IIA_WITH_ALPHA_UINT32_FP16 101001
#define IIA_WITH_ALPHA_UINT32_INT8 101002
#define IIA_WITH_ALPHA_UINT32_INT32 101003
#define IIA_WITH_ALPHA_UINT32_UINT8 101004
#define IIA_WITH_ALPHA_UINT32_INT16 101006
#define IIA_WITH_ALPHA_UINT32_BF16 101027
#define IIA_WITH_ALPHA_UINT32_INT64 101009
#define IIA_WITH_ALPHA_UINT32_BOOL 101012

#define IIA_WITH_ALPHA_UINT64_FP32 101100
#define IIA_WITH_ALPHA_UINT64_FP16 101101
#define IIA_WITH_ALPHA_UINT64_INT8 101102
#define IIA_WITH_ALPHA_UINT64_INT32 101103
#define IIA_WITH_ALPHA_UINT64_UINT8 101104
#define IIA_WITH_ALPHA_UINT64_INT16 101106
#define IIA_WITH_ALPHA_UINT64_BF16 101127
#define IIA_WITH_ALPHA_UINT64_INT64 101109
#define IIA_WITH_ALPHA_UINT64_BOOL 101112

#define DETERMINSTIC_FLOAT32  300000
#define DETERMINSTIC_FLOAT16  300001
#define DETERMINSTIC_BFLOAT16 300002
#define TEMPLATE_SIMD_SORT    200000
#define TEMPLATE_SIMD         400000
#define TEMPLATE_SIMT_SORT_UINT32_NO_ALPHA  500000
#define TEMPLATE_SIMT_SORT_UINT32_ALPHA     500001
#define TEMPLATE_SIMT_SORT_UINT64_NO_ALPHA  500010
#define TEMPLATE_SIMT_SORT_UINT64_ALPHA     500011

extern "C" __global__ __aicore__ void inplace_index_add(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha,
                                                        GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(IIA_UINT32_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<float, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<float, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<half, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<half, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
        
    } else if (TILING_KEY_IS(IIA_UINT32_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint32_t, int32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint32_t, int32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint32_t, int32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint32_t, int32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint32_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint32_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT32_BOOL)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, half, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, half, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<float, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<float, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<half, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<half, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint64_t, int32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint64_t, int32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint64_t, int32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint64_t, int32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint64_t, uint32_t, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint64_t, uint32_t, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_UINT64_BOOL)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, half, false, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, half, false, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<float, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<float, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<half, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<half, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint32_t, int32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint32_t, int32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint32_t, int32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint32_t, int32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint32_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint32_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT32_BOOL)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, half, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint32_t, half, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_FP32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<float, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<float, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<half, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<half, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_INT8)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint64_t, int32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int8_t, DTYPE_INDICES, uint64_t, int32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_INT32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int32_t, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_UINT8)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_INT16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint64_t, int32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int16_t, DTYPE_INDICES, uint64_t, int32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<bfloat16_t, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_INT64)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint64_t, uint32_t, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<int64_t, DTYPE_INDICES, uint64_t, uint32_t, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(IIA_WITH_ALPHA_UINT64_BOOL)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtTilingData, tilingData, tiling);
        if (tilingData.indicesStride == 1) {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, half, true, true> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddSimt<uint8_t, DTYPE_INDICES, uint64_t, half, true, false> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(DETERMINSTIC_BFLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddDeterminsticTilingData, tilingData, tiling);
        if (tilingData.isSplitPreAxis == 1 || tilingData.isSplitAfterAxis == 1) {
            InplaceIndexAddDeterminsticNotQuant<bfloat16_t, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddDeterminstic<bfloat16_t, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(DETERMINSTIC_FLOAT16)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddDeterminsticTilingData, tilingData, tiling);
        if (tilingData.isSplitPreAxis == 1 || tilingData.isSplitAfterAxis == 1) {
            InplaceIndexAddDeterminsticNotQuant<half, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddDeterminstic<half, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(DETERMINSTIC_FLOAT32)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddDeterminsticTilingData, tilingData, tiling);
        if (tilingData.isSplitPreAxis == 1 || tilingData.isSplitAfterAxis == 1) {
            InplaceIndexAddDeterminsticNotQuant<float, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        } else {
            InplaceIndexAddDeterminstic<float, DTYPE_INDICES> op(tilingData, pipe);
            op.Init(var, indices, updates, alpha, userWS);
            op.Process();
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMD_SORT)) {
        if constexpr (std::is_same<int64_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimdSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                InplaceIndexAddSimdSort<DTYPE_VAR, DTYPE_INDICES, true> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            } else {
                InplaceIndexAddSimdSort<DTYPE_VAR, DTYPE_INDICES, false> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMD)) {
        if constexpr (std::is_same<int64_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimdTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                InplaceIndexAddSimd<DTYPE_VAR, DTYPE_INDICES, DTYPE_VAR, true> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            } else {
                InplaceIndexAddSimd<DTYPE_VAR, DTYPE_INDICES, DTYPE_VAR, false> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMT_SORT_UINT32_NO_ALPHA)) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value ||
            std::is_same<int16_t, DTYPE_VAR>::value || std::is_same<bool, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                InplaceIndexAddSimtSort<DTYPE_VAR, DTYPE_INDICES, uint32_t, false, true> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            } else {
                InplaceIndexAddSimtSort<DTYPE_VAR, DTYPE_INDICES, uint32_t, false, false> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMT_SORT_UINT32_ALPHA)) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value ||
            std::is_same<int16_t, DTYPE_VAR>::value || std::is_same<bool, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                InplaceIndexAddSimtSort<DTYPE_VAR, DTYPE_INDICES, uint32_t, true, true> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            } else {
                InplaceIndexAddSimtSort<DTYPE_VAR, DTYPE_INDICES, uint32_t, true, false> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            }
        }
    } else if (TILING_KEY_IS(TEMPLATE_SIMT_SORT_UINT64_NO_ALPHA)) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value ||
            std::is_same<int16_t, DTYPE_VAR>::value || std::is_same<bool, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                InplaceIndexAddSimtSort<DTYPE_VAR, DTYPE_INDICES, uint64_t, false, true> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            } else {
                InplaceIndexAddSimtSort<DTYPE_VAR, DTYPE_INDICES, uint64_t, false, false> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            }
        } 
    } else if (TILING_KEY_IS(TEMPLATE_SIMT_SORT_UINT64_ALPHA)) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<uint8_t, DTYPE_VAR>::value ||
            std::is_same<int16_t, DTYPE_VAR>::value || std::is_same<bool, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddSimtSortTilingData, tilingData, tiling);
            if (tilingData.indicesStride == 1) {
                InplaceIndexAddSimtSort<DTYPE_VAR, DTYPE_INDICES, uint64_t, true, true> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            } else {
                InplaceIndexAddSimtSort<DTYPE_VAR, DTYPE_INDICES, uint64_t, true, false> op(tilingData, pipe);
                op.Init(var, indices, updates, alpha, userWS);
                op.Process();
            }
        } 
    }
}