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
 * \file max_pool3d_grad_with_argmax_scatter_tiling.cpp
 * \brief
 */
#include <iostream>
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace optiling {
ge::graphStatus MaxPool3DGradWithArgmaxScatterTiling::GetShapeAttrsInfo()
{
    MaxPool3DGradWithArgmaxTilingBase::GetShapeAttrsInfo();
    maxPoolGradParams.ncRound = 0UL;
    maxPoolGradParams.ncRoundTail = 0UL;
    maxPoolGradParams.totalRound = 0UL;
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradWithArgmaxScatterTiling::IsCapable()
{
    return true;
}

bool MaxPool3DGradWithArgmaxScatterTiling::SetScatterTilingParams()
{
    const uint64_t doDim = maxPoolGradParams.doDim;
    const uint64_t hoDim = maxPoolGradParams.hoDim;
    const uint64_t woDim = maxPoolGradParams.woDim;
    const uint64_t xDtypeSize = maxPoolGradParams.xDtypeSize;
    const uint64_t indexDtypeSize = maxPoolGradParams.indexDtypeSize;

    uint64_t ncPreCore = Ops::Base::CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.totalCoreNum);
    maxPoolGradParams.usedCoreNum = Ops::Base::CeilDiv(maxPoolGradParams.ncDim, ncPreCore);

    // Scatter main tiling cal
    // 1. All Tensor full size, cut nc between cores, without cut in one core
    uint64_t noCutSize = ncPreCore * doDim * hoDim * woDim * (xDtypeSize + indexDtypeSize);
    if (noCutSize <= (maxPoolGradParams.maxUbSize - BLOCK_SIZE)) {
        maxPoolGradParams.baseNc = ncPreCore;
        maxPoolGradParams.baseDo = doDim;
        maxPoolGradParams.baseHo = hoDim;
        maxPoolGradParams.baseWo = woDim;
        maxPoolGradParams.ubCutAxis = TILING_UB_NO_CUT;
        return true;
    }

    // 2. Cut nc
    uint64_t perNcSize = 1UL * doDim * hoDim * woDim * (xDtypeSize + indexDtypeSize);
    if (perNcSize <= (maxPoolGradParams.maxUbSize - BLOCK_SIZE)) {
        uint64_t baseNc = (maxPoolGradParams.maxUbSize - BLOCK_SIZE) / perNcSize;
        if (baseNc > MAX_BLOCK_COUNT) { // Use NC for blockCount.
            baseNc = MAX_BLOCK_COUNT;
        }
        maxPoolGradParams.baseNc = baseNc;
        maxPoolGradParams.baseDo = doDim;
        maxPoolGradParams.baseHo = hoDim;
        maxPoolGradParams.baseWo = woDim;
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_NC;
        return true;
    }
    maxPoolGradParams.baseNc = 1UL;

    // 3. Cut do
    uint64_t perDoSize = 1UL * 1UL * hoDim * woDim * (xDtypeSize + indexDtypeSize);
    if (perDoSize <= (maxPoolGradParams.maxUbSize - BLOCK_SIZE)) {
        maxPoolGradParams.baseDo = (maxPoolGradParams.maxUbSize - BLOCK_SIZE) / perDoSize;
        maxPoolGradParams.baseHo = hoDim;
        maxPoolGradParams.baseWo = woDim;
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_DO;
        return true;
    }
    maxPoolGradParams.baseDo = 1UL;

    // 4. Cut ho
    uint64_t perHoSize = 1UL * 1UL * 1UL * woDim * (xDtypeSize + indexDtypeSize);
    if (perHoSize <= (maxPoolGradParams.maxUbSize - BLOCK_SIZE)) {
        maxPoolGradParams.baseHo = (maxPoolGradParams.maxUbSize - BLOCK_SIZE) / perHoSize;
        maxPoolGradParams.baseWo = woDim;
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_HO;
        return true;
    }
    maxPoolGradParams.baseHo = 1UL;

    // 5. Cut wo
    uint64_t perWoSize = 1UL * 1UL * 1UL * 1UL * (xDtypeSize + indexDtypeSize);
    if (perWoSize <= (maxPoolGradParams.maxUbSize - BLOCK_SIZE)) {
        maxPoolGradParams.baseWo = (maxPoolGradParams.maxUbSize - BLOCK_SIZE) / perWoSize;
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_WO;
        return true;
    }
    maxPoolGradParams.baseWo = 1UL;

    return false;
}

void MaxPool3DGradWithArgmaxScatterTiling::SetOtherTilingParams()
{
    SetCntTailTilingParams();
    maxPoolGradParams.ncRound = Ops::Base::CeilDiv(maxPoolGradParams.ncCnt, maxPoolGradParams.usedCoreNum);
    maxPoolGradParams.preCoreNum = maxPoolGradParams.ncCnt % maxPoolGradParams.usedCoreNum;
    maxPoolGradParams.ncRoundTail =
        maxPoolGradParams.preCoreNum == 0UL ? maxPoolGradParams.ncRound : maxPoolGradParams.ncRound - 1UL;
    maxPoolGradParams.totalRound =
        maxPoolGradParams.ncRound * maxPoolGradParams.doCnt * maxPoolGradParams.hoCnt * maxPoolGradParams.woCnt;
    if (maxPoolGradParams.xDtypeSize != DTYPE_LEN_B32 && maxPoolGradParams.isOverLap) {
        maxPoolGradParams.workspaceSize = maxPoolGradParams.ncDim * maxPoolGradParams.diDim * maxPoolGradParams.hiDim *
                                          maxPoolGradParams.wiDim * sizeof(float);
    } else {
        maxPoolGradParams.workspaceSize = 0UL;
    }
}

void MaxPool3DGradWithArgmaxScatterTiling::SetScatterTilingData()
{
    tilingData.set_ncRound(maxPoolGradParams.ncRound);
    tilingData.set_ncRoundTail(maxPoolGradParams.ncRoundTail);
    tilingData.set_totalRound(maxPoolGradParams.totalRound);
    tilingData.set_preCoreNum(maxPoolGradParams.preCoreNum);
}

void MaxPool3DGradWithArgmaxScatterTiling::PrintScatterTilingData()
{
    OP_LOGI(
        context_->GetNodeName(), "TilingData ncRound: %lu, ncRoundTail: %lu, totalRound: %lu.",
        tilingData.get_ncRound(), tilingData.get_ncRoundTail(), tilingData.get_totalRound());
}

ge::graphStatus MaxPool3DGradWithArgmaxScatterTiling::DoOpTiling()
{
    bool res = SetScatterTilingParams();
    OP_CHECK_IF(
        !res, OP_LOGE(context_->GetNodeName(), "Scatter cal tiling params failed."), return ge::GRAPH_FAILED);
    maxPoolGradParams.tilingType = TILING_TYPE_SCATTER;
    SetOtherTilingParams();
    SetBaseTilingData();
    SetScatterTilingData();
    PrintTilingData();
    PrintScatterTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", MaxPool3DGradWithArgmaxScatterTiling, 6);
} // namespace optiling
