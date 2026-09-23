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
 * \file swiglu_group_quant_with_dual_axis.cpp
 * \brief Kernel entry of SwigluGroupQuantWithDualAxis.
 */

#include "arch35/swiglu_group_quant_with_dual_axis_tiling_data.h"
#include "arch35/swiglu_group_quant_with_dual_axis_tiling_key.h"
#include "arch35/swiglu_group_quant_mx_kernel.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AscendC;

namespace {
template <uint64_t isGroupIdx>
__aicore__ inline SwigluGroupQuantMxTilingData MakeMxTilingData(
    const SwigluGroupQuantWithDualAxisTilingData& tilingData)
{
    constexpr int64_t tileCols = SwigluGroupQuantMx::ONCE_ROW_LEN;
    SwigluGroupQuantMxTilingData mxTilingData = {};
    mxTilingData.usedCoreNum = tilingData.usedCoreCount;
    mxTilingData.activateLeft = 1;
    mxTilingData.dimM = tilingData.t;
    mxTilingData.dimN = tilingData.h;
    mxTilingData.numGroups = isGroupIdx == TPL_GROUP_INDEX ? tilingData.groupCount : 1;
    mxTilingData.dimNBlockNum = tilingData.h / tileCols + (tilingData.h % tileCols != 0);
    mxTilingData.dimNTail = tilingData.h % tileCols == 0 ? tileCols : tilingData.h % tileCols;
    return mxTilingData;
}

template <uint64_t mode, uint64_t isGroupIdx, bool hasAttrs, bool hasClamp>
__aicore__ inline void RunMxQuant(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y1, GM_ADDR mxScale1,
                                  GM_ADDR y2, GM_ADDR mxScale2, GM_ADDR yOrigin,
                                  SwigluGroupQuantMxTilingData* tilingData,
                                  const SwigluGroupQuantWithDualAxisTilingData& groupTiling, TPipe* pipe)
{
    SwigluGroupQuantMx::SwigluGroupQuantMxKernel<DTYPE_X, DTYPE_Y1, mode, AscendC::RoundMode::CAST_RINT,
                                                 TPL_SCALE_ALG_1, isGroupIdx, hasAttrs, hasClamp,
                                                 SwigluGroupQuantMx::DualAxisPolicy>
        op(tilingData, pipe, groupTiling.clampLimit, groupTiling.alpha, groupTiling.bias, groupTiling.flags,
           groupTiling.weightType);
    op.Init(x, weight, groupIndex, y1, mxScale1, y2, mxScale2, yOrigin);
    op.Process();
}

} // namespace

template <uint64_t MODE, uint64_t IS_GROUP, uint64_t HAS_ATTRS, uint64_t HAS_CLAMP>
__global__ __aicore__ void swiglu_group_quant_with_dual_axis(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y1,
                                                             GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2,
                                                             GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SwigluGroupQuantWithDualAxisTilingData);
    (void)workspace;
    GET_TILING_DATA_WITH_STRUCT(SwigluGroupQuantWithDualAxisTilingData, tilingData, tiling);

    const int64_t originalOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    TPipe pipe;
    auto mxTilingData = MakeMxTilingData<IS_GROUP>(tilingData);
    RunMxQuant<MODE, IS_GROUP, HAS_ATTRS != 0, HAS_CLAMP != 0>(x, weight, groupIndex, y1, mxScale1, y2, mxScale2,
                                                               yOrigin, &mxTilingData, tilingData, &pipe);
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(originalOverflowMode);
}
