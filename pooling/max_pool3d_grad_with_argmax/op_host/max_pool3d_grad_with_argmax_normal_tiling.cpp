/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file max_pool3d_grad_with_argmax_normal_tiling.cpp
 * \brief
 */
#include <iostream>
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace optiling {
ge::graphStatus MaxPool3DGradWithArgmaxNormalTiling::GetShapeAttrsInfo()
{
    auto ret = MaxPool3DGradWithArgmaxTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradWithArgmaxNormalTiling::IsCapable()
{
    if ((maxPoolGradParams.dilationD != 1UL) || (maxPoolGradParams.dilationH != 1UL) ||
        (maxPoolGradParams.dilationW != 1UL)) {
        return false;
    }

    uint64_t normalTensorSizeMin = CalUBTotalSize(1UL, 1UL, 1UL, maxPoolGradParams.isOverLap);
    if (normalTensorSizeMin <= maxPoolGradParams.maxUbSize) {
        return true;
    }
    return false;
}

uint64_t MaxPool3DGradWithArgmaxNormalTiling::CalUBTotalSize(
    uint64_t baseDo, uint64_t baseHo, uint64_t baseWo, bool isCoreOverLap)
{
    const uint64_t vl = maxPoolGradParams.vl;
    uint64_t baseDi = (baseDo - 1UL) * maxPoolGradParams.sd + maxPoolGradParams.kd;
    uint64_t baseHi = (baseHo - 1UL) * maxPoolGradParams.sh + maxPoolGradParams.kh;
    uint64_t baseWi = (baseWo - 1UL) * maxPoolGradParams.sw + maxPoolGradParams.kw;
    uint64_t diHiWi = baseDi * baseHi * baseWi;
    uint64_t doHoWo = baseDo * baseHo * baseWo;
    uint64_t b8BlockAlignNum = static_cast<uint64_t>(BLOCK_SIZE / DTYPE_LEN_B8);
    uint64_t b32BlockAlignNum = static_cast<uint64_t>(BLOCK_SIZE / DTYPE_LEN_B32);
    uint64_t dtypeBlockAlignNum = BLOCK_SIZE / maxPoolGradParams.xDtypeSize;
    uint64_t wiB32Align = Ops::Base::CeilDiv(baseWi, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t wiDtypeAlign = Ops::Base::CeilDiv(baseWi, dtypeBlockAlignNum) * dtypeBlockAlignNum;
    uint64_t diHiWiB32Align = baseDi * baseHi * wiB32Align;
    uint64_t diHiWiDtypeAlign = baseDi * baseHi * wiDtypeAlign;
    uint64_t doHoWoB32Align = Ops::Base::CeilDiv(doHoWo, b32BlockAlignNum) * b32BlockAlignNum;
    uint64_t doHoWoDtypeAlign = Ops::Base::CeilDiv(doHoWo, dtypeBlockAlignNum) * dtypeBlockAlignNum;

    uint64_t normalDiHiWiTensorSize;
    if (isCoreOverLap) {
        normalDiHiWiTensorSize = 1UL * vl * diHiWi * DTYPE_LEN_B32 +        // xIndexBuf
                                 2UL * vl * diHiWiB32Align * DTYPE_LEN_B32; // yQue&yTransposeBuf
    } else {
        normalDiHiWiTensorSize = 1UL * vl * diHiWi * DTYPE_LEN_B32 +                         // xIndexBuf
                                 1UL * vl * diHiWiDtypeAlign * DTYPE_LEN_B32 +               // yTransposeBuf
                                 1UL * vl * diHiWiDtypeAlign * maxPoolGradParams.xDtypeSize; // yQue
    }
    uint64_t normalDoHoWoTensorSize =
        Ops::Base::CeilDiv(1 * vl * doHoWo, b8BlockAlignNum) * b8BlockAlignNum * DTYPE_LEN_B8 + // maskBuf
        2 * vl * doHoWoB32Align * DTYPE_LEN_B32 +                 // argmaxQue&argmaxTransposeBuf
        2 * vl * doHoWoDtypeAlign * maxPoolGradParams.xDtypeSize; // gradQue&gradTransposeBuf
    return normalDiHiWiTensorSize + normalDoHoWoTensorSize + SELECT_RESERVED_UB_SIZE;
}

void MaxPool3DGradWithArgmaxNormalTiling::CalcBaseNc()
{
    maxPoolGradParams.vl = NUM_PER_REP_B32;
    if (maxPoolGradParams.ncDim >= maxPoolGradParams.vl * maxPoolGradParams.totalCoreNum) {
        if (CalUBTotalSize(
                maxPoolGradParams.doDim, maxPoolGradParams.hoDim, maxPoolGradParams.woDim,
                maxPoolGradParams.isOverLap) <= maxPoolGradParams.maxUbSize) {
            return;
        }
    } else if (maxPoolGradParams.ncDim >= maxPoolGradParams.vl) {
        if (!maxPoolGradParams.isOverLap) {
            return;
        }
    }
    const uint64_t minVL = 16;
    if (maxPoolGradParams.ncDim <= maxPoolGradParams.vl - minVL) {
        maxPoolGradParams.vl = Ops::Base::CeilAlign(maxPoolGradParams.ncDim, minVL);
    }
    if (CalUBTotalSize(
            maxPoolGradParams.doDim, maxPoolGradParams.hoDim, maxPoolGradParams.woDim, maxPoolGradParams.isOverLap) >
        maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.vl = minVL;
    }
    return;
}

/* *
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
uint64_t MaxPool3DGradWithArgmaxNormalTiling::CalBestBaseSize(
    uint64_t baseXoStart, uint64_t baseXoEnd, const uint32_t ubCutAxis)
{
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1UL;
    while (baseXoEnd - baseXoStart > 1UL) {
        baseXoMid = (baseXoStart + baseXoEnd) / 2UL;
        if (ubCutAxis == TILING_UB_CUT_DO) {
            tmpTotalSize = CalUBTotalSize(
                baseXoMid, maxPoolGradParams.baseHo, maxPoolGradParams.baseWo, maxPoolGradParams.isOverLap);
        } else if (ubCutAxis == TILING_UB_CUT_HO) {
            tmpTotalSize = CalUBTotalSize(
                maxPoolGradParams.baseDo, baseXoMid, maxPoolGradParams.baseWo, maxPoolGradParams.isOverLap);
        } else if (ubCutAxis == TILING_UB_CUT_WO) {
            tmpTotalSize = CalUBTotalSize(
                maxPoolGradParams.baseDo, maxPoolGradParams.baseHo, baseXoMid, maxPoolGradParams.isOverLap);
        }
        if (tmpTotalSize <= maxPoolGradParams.maxUbSize) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

bool MaxPool3DGradWithArgmaxNormalTiling::SetNormalParamsNotCutUB(const uint32_t ubCutAxis, bool isCoreOverLap)
{
    uint64_t noCutSize = CalUBTotalSize(
        maxPoolGradParams.singleCoreDo, maxPoolGradParams.singleCoreHo, maxPoolGradParams.singleCoreWo, isCoreOverLap);
    if (noCutSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseDo = maxPoolGradParams.singleCoreDo;
        maxPoolGradParams.baseHo = maxPoolGradParams.singleCoreHo;
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        maxPoolGradParams.ubCutAxis = ubCutAxis;
        return true;
    }
    return false;
}

bool MaxPool3DGradWithArgmaxNormalTiling::SetNormalParamsCutUB()
{
    // 1. Cut d
    uint64_t perDoSize = CalUBTotalSize(
        1UL, maxPoolGradParams.singleCoreHo, maxPoolGradParams.singleCoreWo, maxPoolGradParams.isOverLap);
    if (perDoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseHo = maxPoolGradParams.singleCoreHo;
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        // Cal best baseDo
        maxPoolGradParams.baseDo = CalBestBaseSize(1UL, maxPoolGradParams.singleCoreDo, TILING_UB_CUT_DO);
        uint64_t baseDoCnt = Ops::Base::CeilDiv(maxPoolGradParams.singleCoreDo, maxPoolGradParams.baseDo);
        maxPoolGradParams.baseDo = Ops::Base::CeilDiv(maxPoolGradParams.singleCoreDo, baseDoCnt);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_DO;
        return true;
    }

    // 2. Cut d&h
    uint64_t perHoSize = CalUBTotalSize(1UL, 1UL, maxPoolGradParams.singleCoreWo, maxPoolGradParams.isOverLap);
    if (perHoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseDo = 1UL;
        maxPoolGradParams.baseWo = maxPoolGradParams.singleCoreWo;
        // Cal best baseHo
        maxPoolGradParams.baseHo = CalBestBaseSize(1UL, maxPoolGradParams.singleCoreHo, TILING_UB_CUT_HO);
        uint64_t baseHoCnt = Ops::Base::CeilDiv(maxPoolGradParams.singleCoreHo, maxPoolGradParams.baseHo);
        maxPoolGradParams.baseHo = Ops::Base::CeilDiv(maxPoolGradParams.singleCoreHo, baseHoCnt);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_HO;
        return true;
    }

    // 3. Cut d&h&w
    uint64_t perWoSize = CalUBTotalSize(1UL, 1UL, 1UL, maxPoolGradParams.isOverLap);
    if (perWoSize <= maxPoolGradParams.maxUbSize) {
        maxPoolGradParams.baseDo = 1UL;
        maxPoolGradParams.baseHo = 1UL;
        // Cal best baseWo
        maxPoolGradParams.baseWo = CalBestBaseSize(1UL, maxPoolGradParams.singleCoreWo, TILING_UB_CUT_WO);
        uint64_t baseWoCnt = Ops::Base::CeilDiv(maxPoolGradParams.singleCoreWo, maxPoolGradParams.baseWo);
        maxPoolGradParams.baseWo = Ops::Base::CeilDiv(maxPoolGradParams.singleCoreWo, baseWoCnt);
        maxPoolGradParams.ubCutAxis = TILING_UB_CUT_WO;
        return true;
    }

    return false;
}

bool MaxPool3DGradWithArgmaxNormalTiling::SetNormalTilingParams()
{
    const uint64_t ncDim = maxPoolGradParams.ncDim;
    const uint64_t doDim = maxPoolGradParams.doDim;
    const uint64_t hoDim = maxPoolGradParams.hoDim;
    const uint64_t woDim = maxPoolGradParams.woDim;
    const uint64_t totalCoreNum = maxPoolGradParams.totalCoreNum;

    CalcBaseNc();
    const uint64_t vl = maxPoolGradParams.vl;
    maxPoolGradParams.singleCoreDo = doDim;
    maxPoolGradParams.singleCoreHo = hoDim;
    maxPoolGradParams.singleCoreWo = woDim;

    bool fixOverlapOutput = (context_->GetDeterministic() == 1) && maxPoolGradParams.isOverLap;
    OP_LOGI(context_->GetNodeName(), "GetDeterministic state: %u", context_->GetDeterministic());
    // Normal tiling cal begin
    // 1. Cut nc between core
    maxPoolGradParams.singleCoreNc = vl;
    maxPoolGradParams.baseNc = vl;
    uint64_t ncCnt = Ops::Base::CeilDiv(ncDim, maxPoolGradParams.singleCoreNc);
    if (SetNormalParamsNotCutUB(TILING_UB_NO_CUT, false)) {
        maxPoolGradParams.isOverLap = false;
        return true;
    }
    if (ncCnt >= totalCoreNum) {
        return SetNormalParamsCutUB();
    }

    // 2. Cut nc&do between core
    uint64_t doCntNeed = Ops::Base::CeilDiv(totalCoreNum, ncCnt); // need bigger than this
    if (doCntNeed <= doDim) {
        maxPoolGradParams.singleCoreHo = hoDim;
        maxPoolGradParams.singleCoreWo = woDim;
        // 2.1 Dim no overlap
        if (!fixOverlapOutput && (0UL != doCntNeed)) {
            maxPoolGradParams.singleCoreDo = Ops::Base::CeilDiv(doDim, doCntNeed);
            return SetNormalParamsCutUB();
        }
        // 2.2 Dim overlap, cut nc
        maxPoolGradParams.singleCoreDo = doDim;
        return SetNormalParamsCutUB();
    }
    maxPoolGradParams.singleCoreDo = 1UL;
    uint64_t doCnt = Ops::Base::CeilDiv(doDim, maxPoolGradParams.singleCoreDo);

    // 3. Cut nc&do&ho between core
    uint64_t hoCntNeed = Ops::Base::CeilDiv(totalCoreNum, ncCnt * doCnt); // Need bigger than this
    if (SetNormalParamsNotCutUB(TILING_UB_CUT_DO, maxPoolGradParams.isOverLap)) {
        return true;
    }
    if (hoCntNeed <= hoDim) {
        maxPoolGradParams.singleCoreWo = woDim;
        // 3.1 Dim no overlap
        if (!fixOverlapOutput && (0UL != hoCntNeed)) {
            maxPoolGradParams.singleCoreHo = Ops::Base::CeilDiv(hoDim, hoCntNeed);
            return SetNormalParamsCutUB();
        }
        // 3.2 Cut nc
        maxPoolGradParams.singleCoreHo = hoDim;
        return SetNormalParamsCutUB();
    }
    maxPoolGradParams.singleCoreHo = 1UL;
    uint64_t hoCnt = hoDim;

    // 4. Cut nc&do&ho&wo between core
    uint64_t woCntNeed = Ops::Base::CeilDiv(totalCoreNum, ncCnt * doCnt * hoCnt); // Need bigger than this
    if (SetNormalParamsNotCutUB(TILING_UB_CUT_HO, maxPoolGradParams.isOverLap)) {
        return true;
    }
    if (!fixOverlapOutput) {
        // 4.1 Dim no overlap
        if (woCntNeed <= woDim && (0UL != woCntNeed)) {
            maxPoolGradParams.singleCoreWo = Ops::Base::CeilDiv(woDim, woCntNeed);
        } else {
            maxPoolGradParams.singleCoreWo = 1UL;
        }
        return SetNormalParamsCutUB();
    } else {
        // 4.2 Cut nc
        maxPoolGradParams.singleCoreWo = woDim;
        return SetNormalParamsCutUB();
    }

    OP_LOGE(context_->GetNodeName(), "Normal set tiling failed.");
    return false;
}

void MaxPool3DGradWithArgmaxNormalTiling::SetOtherTilingParams()
{
    maxPoolGradParams.ncCnt = Ops::Base::CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.singleCoreNc);
    maxPoolGradParams.doCnt = Ops::Base::CeilDiv(maxPoolGradParams.doDim, maxPoolGradParams.singleCoreDo);
    maxPoolGradParams.hoCnt = Ops::Base::CeilDiv(maxPoolGradParams.hoDim, maxPoolGradParams.singleCoreHo);
    maxPoolGradParams.woCnt = Ops::Base::CeilDiv(maxPoolGradParams.woDim, maxPoolGradParams.singleCoreWo);
    maxPoolGradParams.ncTail =
        maxPoolGradParams.ncDim - (maxPoolGradParams.ncCnt - 1UL) * maxPoolGradParams.singleCoreNc;
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

void MaxPool3DGradWithArgmaxNormalTiling::SetNormalTilingData()
{
    tilingData.set_singleCoreNc(maxPoolGradParams.singleCoreNc);
    tilingData.set_singleCoreDo(maxPoolGradParams.singleCoreDo);
    tilingData.set_singleCoreHo(maxPoolGradParams.singleCoreHo);
    tilingData.set_singleCoreWo(maxPoolGradParams.singleCoreWo);
}

void MaxPool3DGradWithArgmaxNormalTiling::PrintNormalTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "TilingData singleCoreNc: %lu, singleCoreDo: %lu, singleCoreHo: %lu, singleCoreWo: %lu.",
        tilingData.get_singleCoreNc(), tilingData.get_singleCoreDo(), tilingData.get_singleCoreHo(),
        tilingData.get_singleCoreWo());
}

ge::graphStatus MaxPool3DGradWithArgmaxNormalTiling::DoOpTiling()
{
    bool res = SetNormalTilingParams();
    OP_CHECK_IF(
        !res, OP_LOGE(context_->GetNodeName(), "Normal cal tiling params failed."), return ge::GRAPH_FAILED);
    maxPoolGradParams.tilingType = TILING_TYPE_NORMAL;
    SetOtherTilingParams();
    SetBaseTilingData();
    SetNormalTilingData();
    PrintTilingData();
    PrintNormalTilingData();
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", MaxPool3DGradWithArgmaxNormalTiling, 2);
} // namespace optiling
