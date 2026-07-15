/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant.cpp
 * \brief
 */

#include "arch35/swiglu_group_quant_perf.h"
#include "arch35/swiglu_mx_quant_perf.h"
#include "arch35/swiglu_mxfp4_quant_perf.h"
#include "arch35/swiglu_dynamic_group_quant_hifp8.h"
#include "arch35/swiglu_static_group_quant_hifp8.h"
#define BLOCK_QUANT_TILING_KEY 1000
#define BLOCK_QUANT_YORIGIN_TILING_KEY 1100
#define MX_QUANT_TILING_KEY 2000
#define MX_QUANT_YORIGIN_TILING_KEY 2100
#define MXFP4_QUANT_TILING_KEY 3000
#define DYNAMIC_HIFP8_QUANT_TILING_KEY 4000
#define STATIC_HIFP8_QUANT_TILING_KEY 4100
using namespace AscendC;

extern "C" __global__ __aicore__ void swiglu_group_quant(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale,
    GM_ADDR y, GM_ADDR yScale, GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    TPipe pipe;
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if (TILING_KEY_IS(BLOCK_QUANT_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantTilingData, tilingData, tiling);
        SwigluGroupQuant::SwigluGroupQuantPerf<DTYPE_X, DTYPE_Y, DTYPE_Y_SCALE, false> op;
        op.Init(x, weight, groupIndex, y, yScale, yOrigin, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(BLOCK_QUANT_YORIGIN_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantTilingData, tilingData, tiling);
        SwigluGroupQuant::SwigluGroupQuantPerf<DTYPE_X, DTYPE_Y, DTYPE_Y_SCALE, true> op;
        op.Init(x, weight, groupIndex, y, yScale, yOrigin, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(MX_QUANT_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantTilingData, tilingData, tiling);
        SwigluGroupQuant::SwigluMxQuantPerf<DTYPE_X, DTYPE_Y, DTYPE_Y_SCALE, false> op;
        op.Init(x, weight, groupIndex, y, yScale, yOrigin, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(MX_QUANT_YORIGIN_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantTilingData, tilingData, tiling);
        SwigluGroupQuant::SwigluMxQuantPerf<DTYPE_X, DTYPE_Y, DTYPE_Y_SCALE, true> op;
        op.Init(x, weight, groupIndex, y, yScale, yOrigin, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(MXFP4_QUANT_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantTilingData, tilingData, tiling);
        SwigluGroupQuant::SwigluMxFp4QuantPerf<DTYPE_X, DTYPE_Y, DTYPE_Y_SCALE> op;
        op.Init(x, weight, groupIndex, y, yScale, userWs, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(DYNAMIC_HIFP8_QUANT_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantHifp8TilingData, hifp8TilingData, tiling);
        SwigluGroupQuantDynamicHifp8Ops::SwigluGroupQuantDynamicHifp8Kernel<DTYPE_X> op;
        op.Init(x, weight, groupIndex, y, yScale, yOrigin, userWs, &hifp8TilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(STATIC_HIFP8_QUANT_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantHifp8TilingData, hifp8TilingData, tiling);
        SwigluGroupQuantStaticHifp8Ops::SwigluGroupQuantStaticHifp8Kernel<DTYPE_X> op;
        op.Init(x, weight, groupIndex, scale, y, yOrigin, userWs, &hifp8TilingData, &pipe);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
