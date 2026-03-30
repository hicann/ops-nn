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
 * \file index_fill_apt.cpp
 * \brief
 */
#include "arch35/index_fill_simt.h"
#include "arch35/index_fill_simd.h"
#include "arch35/index_fill_tiling_key.h"
#include "arch35/index_fill_struct.h"

using namespace IndexFill;

template <uint64_t TEMPLATE_MODE, uint64_t DYTPE_MODE>
__global__ __aicore__ void index_fill(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    AscendC::TPipe tpipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(IndexFillTilingData);
    if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_EMPTY) {
        return;
    }

    using T = typename IndexFill::ComputeTypeGet<DTYPE_X>::type;
    if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD && DYTPE_MODE == TPL_MODE_DTYPE_B32) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD && DYTPE_MODE == TPL_MODE_DTYPE_B32", IndexFillSimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexFillSimdTilingData, tilingData, tiling);
        IndexFillSimdImpl<T, DTYPE_INDICES, uint32_t> op(tilingData, tpipe);
        op.Init(x, indices, value, y, workspace);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD && DYTPE_MODE == TPL_MODE_DTYPE_B64) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMD && DYTPE_MODE == TPL_MODE_DTYPE_B64", IndexFillSimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexFillSimdTilingData, tilingData, tiling);
        IndexFillSimdImpl<T, DTYPE_INDICES, uint64_t> op(tilingData, tpipe);
        op.Init(x, indices, value, y, workspace);
        op.Process();
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B32) {
        // simt为全功能模板
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B32", IndexFillSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexFillSimtTilingData, tilingData, tiling);
        IndexFillSimtImpl<T, DTYPE_INDICES, uint32_t> op(&tilingData, &tpipe);
        op.Init(x, y, indices, value);
        op.Process((__gm__ T*)y);
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B64) {
        // simt为全功能模板
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_TEMPLATE_SIMT && DYTPE_MODE == TPL_MODE_DTYPE_B64", IndexFillSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(IndexFillSimtTilingData, tilingData, tiling);
        IndexFillSimtImpl<T, DTYPE_INDICES, uint64_t> op(&tilingData, &tpipe);
        op.Init(x, y, indices, value);
        op.Process((__gm__ T*)y);
    }
}
