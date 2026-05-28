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
 * \file scatter_nd_max_apt.cpp
 * \brief scatter_nd_max kernel
 */


#include "../scatter_nd_common/arch35/scatter_nd_common_simd_sort.h"
#include "../scatter_nd_common/arch35/scatter_nd_common_simt.h"
#include "../scatter_nd_common/arch35/scatter_nd_max_tiling_key.h"
#include "../scatter_nd_common/arch35/scatter_nd_common_base.h"

using namespace AscendC;
using namespace ScatterNdCommon;


template <uint64_t TEMPLATE_MODE, uint64_t CAST_MODE, uint64_t ADDR_MODE>
__global__ __aicore__ void scatter_nd_max(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); //
    REGISTER_TILING_DEFAULT(ScatterNdCommonSimdSortTilingData);
    if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD_SORT && CAST_MODE == CAST_0 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_VAR>::value || std::is_same<uint64_t, DTYPE_VAR>::value ||
            std::is_same<int64_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimdSortTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimdSort<DTYPE_VAR, DTYPE_INDICES, DTYPE_INDICES, DTYPE_INDICES, CAST_0, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD_SORT && CAST_MODE == CAST_1 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_VAR>::value || std::is_same<uint64_t, DTYPE_VAR>::value ||
            std::is_same<int64_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimdSortTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimdSort<DTYPE_VAR, DTYPE_INDICES, int16_t, DTYPE_INDICES, CAST_1, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD_SORT && CAST_MODE == CAST_2 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_VAR>::value || std::is_same<uint64_t, DTYPE_VAR>::value ||
            std::is_same<int64_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimdSortTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimdSort<DTYPE_VAR, DTYPE_INDICES, int32_t, DTYPE_INDICES, CAST_2, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD_SORT && CAST_MODE == CAST_3 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_VAR>::value || std::is_same<uint64_t, DTYPE_VAR>::value ||
            std::is_same<int64_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimdSortTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimdSort<DTYPE_VAR, DTYPE_INDICES, int16_t, DTYPE_INDICES, CAST_3, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD_SORT && CAST_MODE == CAST_4 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_VAR>::value || std::is_same<uint64_t, DTYPE_VAR>::value ||
            std::is_same<int64_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimdSortTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimdSort<DTYPE_VAR, DTYPE_INDICES, uint8_t, DTYPE_INDICES, CAST_4, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD_SORT && CAST_MODE == CAST_5 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_VAR>::value || std::is_same<uint64_t, DTYPE_VAR>::value ||
            std::is_same<int64_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimdSortTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimdSort<DTYPE_VAR, DTYPE_INDICES, uint8_t, DTYPE_INDICES, CAST_5, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD_SORT && CAST_MODE == CAST_0 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        if constexpr (
            std::is_same<uint32_t, DTYPE_VAR>::value || std::is_same<uint64_t, DTYPE_VAR>::value ||
            std::is_same<int64_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimdSortTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimdSort<DTYPE_VAR, DTYPE_INDICES, int64_t, int64_t, CAST_0, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && CAST_MODE == CAST_0 && ADDR_MODE == TPL_MODE_ADDR_INT32) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<int16_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimtTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimt<DTYPE_VAR, DTYPE_INDICES, uint32_t, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && CAST_MODE == CAST_0 && ADDR_MODE == TPL_MODE_ADDR_INT64) {
        if constexpr (std::is_same<int8_t, DTYPE_VAR>::value || std::is_same<int16_t, DTYPE_VAR>::value) {
            return;
        } else {
            GET_TILING_DATA_WITH_STRUCT(ScatterNdCommonSimtTilingData, tilingData, tiling);
            ScatterNdCommon::ScatterNdCommonSimt<DTYPE_VAR, DTYPE_INDICES, uint64_t, MODE_MAX> op(tilingData, pipe);
            op.Init(x, indices, updates, y);
            op.Process();
        }
    }

}
