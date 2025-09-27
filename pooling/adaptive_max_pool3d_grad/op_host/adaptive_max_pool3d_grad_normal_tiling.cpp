/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_max_pool3d_grad_normal_tiling.cpp
 * \brief
 */
#include "adaptive_max_pool3d_grad_tiling.h"
#include <iostream>

constexpr int FACTOR_TWO = 2;
constexpr int FACTOR_SEVEN = 7;

namespace optiling {
ge::graphStatus AdaptiveMaxPool3DGradNormalTiling::GetShapeAttrsInfo()
{
    auto ret = AdaptiveMaxPool3DGradTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveMaxPool3DGradNormalTiling::IsCapable()
{
    uint64_t normalTensorSizeMin = CalUBTotalSize(1UL, 1UL, 1UL);
    if (normalTensorSizeMin <= maxPoolGradParams.maxUbSize) {
        return true;
    }
    return false;
}

uint64_t AdaptiveMaxPool3DGradNormalTiling::CalUBTotalSize(uint64_t baseDo, uint64_t baseHo, uint64_t baseWo)
{
    const uint64_t vl = maxPoolGradParams.vl;
    uint64_t doHoWo = baseDo * baseHo * baseWo;
    uint64_t b32BlockAlignNum = static_cast<uint64_t>(BLOCK_SIZE) / static_cast<uint64_t>(DTYPE_LEN_B32);
    uint64_t dtypeBlockAlignNum = BLOCK_SIZE / maxPoolGradParams.xDtypeSize;
    uint64_t doHoWoB32Align = Ops::Base::CeilDiv(doHoWo, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t doHoWoDtypeAlign = Ops::Base::CeilDiv(doHoWo, dtypeBlockAlignNum) * dtypeBlockAlignNum;
    uint64_t maxkWB32Align = Ops::Base::CeilDiv(maxPoolGradParams.kwMax, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t maxkWAlignDtype = Ops::Base::CeilDiv(maxPoolGradParams.kwMax, dtypeBlockAlignNum) * dtypeBlockAlignNum;
    uint64_t vlMaskNumAlign = Ops::Base::CeilDiv(vl / 8, static_cast<uint64_t>(BLOCK_SIZE)) * BLOCK_SIZE;

    uint64_t normalDiHiWiTensorSize = 0UL;
    uint64_t normalDoHoWoTensorSize = static_cast<uint64_t>(FACTOR_TWO) * vl * doHoWoDtypeAlign *
                                      maxPoolGradParams.xDtypeSize; // gradQue&gradTransposeBuf
    if (maxPoolGradParams.isOverLap) {
        normalDiHiWiTensorSize = static_cast<uint64_t>(FACTOR_TWO) * vl * maxPoolGradParams.kdMax *
                                 maxPoolGradParams.khMax * maxkWB32Align * DTYPE_LEN_B32; // yQue&yTransposeBuf
    } else {
        normalDiHiWiTensorSize = static_cast<uint64_t>(FACTOR_TWO) * vl * maxPoolGradParams.kdMax *
                                 maxPoolGradParams.khMax * maxkWAlignDtype *
                                 maxPoolGradParams.xDtypeSize; // yQue&yTransposeBuf
    }

    normalDoHoWoTensorSize += static_cast<uint64_t>(FACTOR_SEVEN) * vl * doHoWoB32Align *
                              DTYPE_LEN_B32; // indicesQue&indicesTransposeBuf&indicesFloatBuf
                                             // indicesDBuf&indicesHBuf&indicesWBuf
                                             // tempBuf
    normalDoHoWoTensorSize += static_cast<uint64_t>(FACTOR_TWO) * vl * maxPoolGradParams.kdMax *
                              maxPoolGradParams.khMax * maxPoolGradParams.kwMax *
                              DTYPE_LEN_B32; // kernelIdxBuf&tempGradBuf

    normalDoHoWoTensorSize += vlMaskNumAlign * maxPoolGradParams.kdMax * maxPoolGradParams.khMax *
                              maxPoolGradParams.kwMax * DTYPE_LEN_B8; // maskBuf

    return normalDiHiWiTensorSize + normalDoHoWoTensorSize + SELECT_RESERVED_UB_SIZE;
}

uint64_t AdaptiveMaxPool3DGradNormalTiling::CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd)
{
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0UL;
    baseXoEnd = baseXoEnd + 1UL;
    while (baseXoEnd - baseXoStart > 1UL) {
        baseXoMid = (baseXoStart + baseXoEnd) / static_cast<uint64_t>(FACTOR_TWO);
        tmpTotalSize = CalUBTotalSize(maxPoolGradParams.baseDo, maxPoolGradParams.baseHo, baseXoMid);
        if (tmpTotalSize <= maxPoolGradParams.maxUbSize) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

bool AdaptiveMaxPool3DGradNormalTiling::SetNormalParamsUB()
{
    uint64_t noCutSize =
        CalUBTotalSize(maxPoolGradParams.baseDo, maxPoolGradParams.baseHo, maxPoolGradParams.singleCoreWo);
    if (noCutSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        maxPoolGradParams.ubCutAxis = TILING_UB_NO_CUT;
        return true;
    }
    // 3. Cut d&h&w
    uint64_t perWoSize = CalUBTotalSize(maxPoolGradParams.baseDo, maxPoolGradParams.baseHo, 1UL);
    if (perWoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseWo = CalBestBaseSize(1UL, maxPoolGradParams.singleCoreWo);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_WO;
        return true;
    }
    OP_LOGE(context_->GetNodeName(), "Normal set tiling failed.");
    return false;
}

bool AdaptiveMaxPool3DGradNormalTiling::SetNormalTilingParams()
{
    const uint64_t ncDim = maxPoolGradParams.ncDim;
    const uint64_t doDim = maxPoolGradParams.doDim;
    const uint64_t hoDim = maxPoolGradParams.hoDim;
    const uint64_t woDim = maxPoolGradParams.woDim;
    const uint64_t totalCoreNum = maxPoolGradParams.totalCoreNum;
    const uint64_t vl = maxPoolGradParams.vl;
    maxPoolGradParams.singleCoreDo = doDim;
    maxPoolGradParams.singleCoreHo = hoDim;
    maxPoolGradParams.singleCoreWo = woDim;
    bool isDOverlap = (maxPoolGradParams.diDim % maxPoolGradParams.doDim) != 0UL;
    maxPoolGradParams.baseDo = 1UL;
    maxPoolGradParams.baseHo = 1UL;

    // Normal tiling cal begin
    // 1. Cut nc between core
    maxPoolGradParams.singleCoreNc = vl;
    maxPoolGradParams.baseNc = vl <= ncDim ? vl : ncDim;
    uint64_t ncCnt = Ops::Base::CeilDiv(ncDim, maxPoolGradParams.singleCoreNc);
    if (ncCnt >= totalCoreNum) {
        return SetNormalParamsUB();
    }
    // 2. Cut nc&do between core
    uint64_t doCntNeed = Ops::Base::CeilDiv(totalCoreNum, ncCnt);
    // 2.1 Dim no overlap
    if (!isDOverlap && (0UL != doCntNeed)) {
        uint64_t singleCoreDo = doDim / doCntNeed;
        maxPoolGradParams.singleCoreDo = singleCoreDo < 1UL ? 1UL : singleCoreDo;
    }

    // 2.2 D dim overlap, can cut Gcd count of d Dim between core
    if (isDOverlap && (0UL != doCntNeed)) {
        maxPoolGradParams.singleCoreDo = doDim / maxPoolGradParams.dGcd;
    }
    return SetNormalParamsUB();
}

void AdaptiveMaxPool3DGradNormalTiling::SetOtherTilingParams()
{
    maxPoolGradParams.ncCnt = Ops::Base::CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.singleCoreNc);
    maxPoolGradParams.doCnt = Ops::Base::CeilDiv(maxPoolGradParams.doDim, maxPoolGradParams.singleCoreDo);
    maxPoolGradParams.hoCnt = Ops::Base::CeilDiv(maxPoolGradParams.hoDim, maxPoolGradParams.singleCoreHo);
    maxPoolGradParams.woCnt = Ops::Base::CeilDiv(maxPoolGradParams.woDim, maxPoolGradParams.singleCoreWo);
    maxPoolGradParams.ncTail = maxPoolGradParams.ncDim - (maxPoolGradParams.ncCnt - 1UL) * maxPoolGradParams.baseNc;
    maxPoolGradParams.doTail =
        maxPoolGradParams.doDim - (maxPoolGradParams.doCnt - 1UL) * maxPoolGradParams.singleCoreDo;
    maxPoolGradParams.hoTail =
        maxPoolGradParams.hoDim - (maxPoolGradParams.hoCnt - 1UL) * maxPoolGradParams.singleCoreHo;
    maxPoolGradParams.woTail =
        maxPoolGradParams.woDim - (maxPoolGradParams.woCnt - 1UL) * maxPoolGradParams.singleCoreWo;
    maxPoolGradParams.totalCnt =
        maxPoolGradParams.ncCnt * maxPoolGradParams.doCnt * maxPoolGradParams.hoCnt * maxPoolGradParams.woCnt;
    maxPoolGradParams.usedCoreNum = std::min(maxPoolGradParams.totalCnt, maxPoolGradParams.totalCoreNum);
    if (maxPoolGradParams.xDtypeSize != DTYPE_LEN_B32 && maxPoolGradParams.isOverLap) {
        maxPoolGradParams.workspaceSize = maxPoolGradParams.ncDim * maxPoolGradParams.diDim * maxPoolGradParams.hiDim *
                                          maxPoolGradParams.wiDim * sizeof(float);
    } else {
        maxPoolGradParams.workspaceSize = 0UL;
    }
}

void AdaptiveMaxPool3DGradNormalTiling::SetNormalTilingData()
{
    tilingData.set_singleCoreNc(maxPoolGradParams.singleCoreNc);
    tilingData.set_singleCoreDo(maxPoolGradParams.singleCoreDo);
    tilingData.set_singleCoreHo(maxPoolGradParams.singleCoreHo);
    tilingData.set_singleCoreWo(maxPoolGradParams.singleCoreWo);
}

void AdaptiveMaxPool3DGradNormalTiling::PrintNormalTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "TilingData singleCoreNc: %lu, singleCoreDo: %lu, singleCoreHo: %lu, singleCoreWo: %lu.",
        tilingData.get_singleCoreNc(), tilingData.get_singleCoreDo(), tilingData.get_singleCoreHo(),
        tilingData.get_singleCoreWo());
}

ge::graphStatus AdaptiveMaxPool3DGradNormalTiling::DoOpTiling()
{
    bool res = SetNormalTilingParams();
    OP_CHECK_IF(!res, OP_LOGE(context_->GetNodeName(), "Normal cal tiling params failed."), return ge::GRAPH_FAILED);
    maxPoolGradParams.tilingType = TILING_TYPE_NORMAL;
    SetOtherTilingParams();
    SetBaseTilingData();
    SetNormalTilingData();
    PrintTilingData();
    PrintNormalTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("AdaptiveMaxPool3DGrad", AdaptiveMaxPool3DGradNormalTiling, 0);
} // namespace optiling
