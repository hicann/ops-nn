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
 * \file swiglu_group_quant_mx_dispatch.h
 * \brief MX launch helpers for the single-axis SwigluGroupQuant kernel.
 */

#ifndef SWIGLU_GROUP_QUANT_MX_DISPATCH_H
#define SWIGLU_GROUP_QUANT_MX_DISPATCH_H

#include "arch35/swiglu_mx_quant_perf.h"
#include "../swiglu_group_quant_with_dual_axis/arch35/swiglu_group_quant_mx_kernel.h"

using namespace AscendC;

template <typename T0, typename T1, typename T2, bool outputOrigin>
__aicore__ inline void RunMxQuant(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale,
                                  GM_ADDR yOrigin, GM_ADDR userWs, const SwigluGroupQuantTilingData* tilingData,
                                  TPipe* pipe)
{
    SwigluGroupQuant::SwigluMxQuantPerf<T0, T1, T2, outputOrigin> op;
    op.Init(x, weight, groupIndex, y, yScale, yOrigin, userWs, tilingData, pipe);
    op.Process();
}

__aicore__ inline SwigluGroupQuantMxTilingData MakeMxQuantExtendTiling(
    const SwigluGroupQuantMxExtendTilingData& tilingData)
{
    SwigluGroupQuantMxTilingData sharedTiling = {};
    sharedTiling.usedCoreNum = tilingData.usedCoreNum;
    sharedTiling.activateLeft = 1;
    sharedTiling.dimM = tilingData.dimM;
    sharedTiling.dimN = tilingData.dimN;
    sharedTiling.numGroups = 1;
    sharedTiling.dimNBlockNum = tilingData.dimNBlockNum;
    sharedTiling.dimNTail = tilingData.dimNTail;
    return sharedTiling;
}

template <typename xDtype, typename yDtype, uint64_t mode, bool hasAttrs, bool hasClamp>
__aicore__ inline void RunMxQuantExtendShared(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale,
                                              GM_ADDR yOrigin, SwigluGroupQuantMxTilingData* sharedTiling,
                                              const SwigluGroupQuantMxExtendTilingData& tilingData, TPipe* pipe)
{
    // Reuse the dual-axis kernel's axis-1 path so single-axis y/yScale remain identical to dual-axis y1/mxScale1.
    SwigluGroupQuantMx::SwigluGroupQuantMxKernel<xDtype, yDtype, mode, AscendC::RoundMode::CAST_RINT, TPL_SCALE_ALG_1,
                                                 TPL_NO_GROUP_INDEX, hasAttrs, hasClamp,
                                                 SwigluGroupQuantMx::SingleAxisPolicy>
        op(sharedTiling, pipe, tilingData.clampLimit, tilingData.alpha, tilingData.bias, tilingData.flags,
           tilingData.weightType);
    op.Init(x, weight, groupIndex, y, yScale, nullptr, nullptr, yOrigin);
    op.Process();
}

template <typename xDtype, typename yDtype, uint64_t mode>
__aicore__ inline void DispatchMxQuantExtendShared(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y,
                                                   GM_ADDR yScale, GM_ADDR yOrigin,
                                                   SwigluGroupQuantMxTilingData* sharedTiling,
                                                   const SwigluGroupQuantMxExtendTilingData& tilingData, TPipe* pipe)
{
    const bool hasClamp = (tilingData.flags & MX_HAS_CLAMP) != 0;
    const bool hasAttrs = hasClamp || tilingData.alpha != 1.0F || tilingData.bias != 0.0F;
    if (!hasAttrs) {
        RunMxQuantExtendShared<xDtype, yDtype, mode, false, false>(x, weight, groupIndex, y, yScale, yOrigin,
                                                                   sharedTiling, tilingData, pipe);
    } else if (hasClamp) {
        RunMxQuantExtendShared<xDtype, yDtype, mode, true, true>(x, weight, groupIndex, y, yScale, yOrigin,
                                                                 sharedTiling, tilingData, pipe);
    } else {
        RunMxQuantExtendShared<xDtype, yDtype, mode, true, false>(x, weight, groupIndex, y, yScale, yOrigin,
                                                                  sharedTiling, tilingData, pipe);
    }
}

template <typename xDtype, typename yDtype>
__aicore__ inline void RunMxQuantExtend(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale,
                                        GM_ADDR yOrigin, const SwigluGroupQuantMxExtendTilingData& tilingData,
                                        TPipe* pipe)
{
    constexpr bool supportedX = IsSameType<xDtype, half>::value || IsSameType<xDtype, bfloat16_t>::value;
    constexpr bool supportedY = IsSameType<yDtype, fp8_e4m3fn_t>::value || IsSameType<yDtype, fp8_e5m2_t>::value;
    if constexpr (supportedX && supportedY) {
        auto sharedTiling = MakeMxQuantExtendTiling(tilingData);
        if (sharedTiling.dimNBlockNum < sharedTiling.usedCoreNum) {
            DispatchMxQuantExtendShared<xDtype, yDtype, TPL_MODE_ROTATE>(x, weight, groupIndex, y, yScale, yOrigin,
                                                                         &sharedTiling, tilingData, pipe);
        } else {
            DispatchMxQuantExtendShared<xDtype, yDtype, TPL_MODE_BLOCK>(x, weight, groupIndex, y, yScale, yOrigin,
                                                                        &sharedTiling, tilingData, pipe);
        }
    }
}

#endif // SWIGLU_GROUP_QUANT_MX_DISPATCH_H
