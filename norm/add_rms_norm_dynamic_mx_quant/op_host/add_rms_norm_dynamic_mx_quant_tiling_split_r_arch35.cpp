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
 * \file add_rms_norm_dynamic_mx_quant_tiling_split_r_arch35.cpp
 * \brief
 */
#include "add_rms_norm_dynamic_mx_quant_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

using namespace optiling::add_rms_norm_dynamic_mx_quant;

namespace optiling {
using namespace NormCheck;

constexpr uint32_t RETAINED_SIZE_256 = 256;

int64_t AddRmsNormDynamicMxQuantSplitRTiling::GetPowerSplit(uint64_t numN)
{
    uint64_t ubLoops = Ops::Base::CeilDiv(numCol_, numN);
    int64_t powerSplit = ubLoops == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubLoops)));
    powerSplit = (static_cast<uint64_t>(powerSplit) == ubLoops) ? powerSplit / NUM_TWO : powerSplit;
    if (ubLoops == 1) {
        powerSplit = 1;
    }
    return powerSplit;
}

int64_t AddRmsNormDynamicMxQuantSplitRTiling::GetCacheId(int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + 1)) - 1;
}

uint64_t AddRmsNormDynamicMxQuantSplitRTiling::GetMaxBaseN(uint64_t initialN)
{
    uint64_t vlfp32 = vecLengthFP32_;
    uint64_t binaryAddElemtMaxLen = vlfp32 * vlfp32 * NUM_TWO * NUM_TWO;

    // Fixed buffers (not dependent on baseN)
    uint64_t rstdBufSize = Ops::Base::CeilAlign(baseM_ * FP32_SIZE, ubBlockSize_) * DOUBLE_BUFFER;
    uint64_t binaryAddBufSize = Ops::Base::CeilAlign(vlfp32 * FP32_SIZE * NUM_TWO, ubBlockSize_);

    int64_t powerSplit = GetPowerSplit(initialN);
    uint64_t cacheBufSize = static_cast<uint64_t>(GetCacheId(powerSplit - 1) + 1) * ubBlockSize_;

    while (powerSplit > 1 && NUM_TWO * initialN <= binaryAddElemtMaxLen) {
        uint64_t candidateN = NUM_TWO * initialN;
        uint64_t xInputBuf = NUM_TWO * candidateN * xDtypeSize_ * DOUBLE_BUFFER;
        uint64_t gammaBetaBuf = (1 + betaFlag_) * candidateN * gammaDtypeSize_ * DOUBLE_BUFFER;
        uint64_t yBuf = 0;
        if (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype_) != 0) {
            yBuf = Ops::Base::CeilAlign(candidateN * FP8_SIZE, ubBlockSize_) * DOUBLE_BUFFER;
        } else if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0) {
            yBuf = Ops::Base::CeilAlign(candidateN / NUM_TWO, ubBlockSize_) * DOUBLE_BUFFER;
        }
        uint64_t xOutBuf = candidateN * xDtypeSize_ * DOUBLE_BUFFER;
        uint64_t mxScaleBuf = Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(candidateN, static_cast<uint64_t>(MX_BLOCK_SIZE_32)) * sizeof(uint8_t), ubBlockSize_);

        uint64_t xFp32Buf = Ops::Base::CeilAlign(candidateN * FP32_SIZE, ubBlockSize_);
        uint64_t yTmpBuf = Ops::Base::CeilAlign(candidateN * xDtypeSize_, ubBlockSize_);
        uint64_t maxExpBuf = Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(candidateN, static_cast<uint64_t>(MX_BLOCK_SIZE_32)) * sizeof(uint16_t), ubBlockSize_);
        uint64_t halfScaleBuf = maxExpBuf;

        uint64_t totalNDependent =
            xInputBuf + gammaBetaBuf + yBuf + xOutBuf + mxScaleBuf + xFp32Buf + yTmpBuf + maxExpBuf + halfScaleBuf;

        if (totalNDependent > maxUbSize_ - RETAINED_SIZE_256 - (rstdBufSize + cacheBufSize + binaryAddBufSize)) {
            break;
        }

        initialN = candidateN;
        powerSplit = GetPowerSplit(initialN);
        cacheBufSize = static_cast<uint64_t>(GetCacheId(powerSplit - 1) + 1) * ubBlockSize_;
    }
    return initialN;
}

bool AddRmsNormDynamicMxQuantSplitRTiling::IsCapable()
{
    if (Y_SUPPORT_DTYPE_SET.count(yDtype_) == 0) {
        return false;
    }
    return numCol_ >= baseN_;
}

ge::graphStatus AddRmsNormDynamicMxQuantSplitRTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DoOpTiling for SplitR.");

    // Multi-core split on A axis (same as full-load)
    mPerCore_ = Ops::Base::CeilDiv(numRow_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(numRow_, mPerCore_);
    mLastCore_ = numRow_ - (usedCoreNum_ - 1) * mPerCore_;
    blockFactor_ = mPerCore_;

    // Maximize baseN within UB constraints , baseN_ is a multiple of 64
    baseN_ = GetMaxBaseN(baseN_);
    baseNBlockSize_ = baseN_ / MX_BLOCK_SIZE_32;

    // R-axis split
    nUbLoops_ = Ops::Base::CeilDiv(numCol_, baseN_);

    binAddQuotient_ = baseN_ == 0 ? 1 : (1UL << (ULONG_BIT_LEN - 1 - __builtin_clzl(baseN_)));
    binAddQuotient_ = (binAddQuotient_ == baseN_) ? binAddQuotient_ / NUM_TWO : binAddQuotient_;
    powerSplit_ = GetPowerSplit(baseN_);
    mainFoldCount_ = powerSplit_ * baseN_ > numCol_ ? 0 : (numCol_ - powerSplit_ * baseN_) / baseN_;

    foldTail_ = numCol_ % baseN_;

    OP_LOGI(
        context_->GetNodeName(),
        "SplitR: baseN=%lu, baseM=%lu, nUbLoops=%lu, powerSplit=%lu, "
        "mainFoldCount=%lu, foldTail=%lu, blockFactor=%lu.",
        baseN_, baseM_, nUbLoops_, powerSplit_, mainFoldCount_, foldTail_, blockFactor_);

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormDynamicMxQuantSplitRTiling::SetTilingData()
{
    // AddRmsNorm fields
    tilingData.numCol = numCol_;
    tilingData.numColAlign = numColAlign_;
    tilingData.blockFactor = blockFactor_;
    tilingData.mLastCore = mLastCore_;
    // SplitR specific fields
    tilingData.baseN = baseN_;
    tilingData.baseM = baseM_;
    tilingData.baseNBlockSize = baseNBlockSize_;
    tilingData.nUbLoops = nUbLoops_;
    tilingData.binAddQuotient = binAddQuotient_;
    tilingData.powerSplit = powerSplit_;
    tilingData.mainFoldCount = mainFoldCount_;
    tilingData.foldTail = foldTail_;
    tilingData.epsilon = epsilon_;
    tilingData.avgFactor = avgFactor_;
    // DynamicMxQuant fields
    tilingData.roundMode = roundMode_;
    tilingData.mxBlockSize = mxBlockSize_;
    tilingData.scaleAlg = scaleAlg_;
    tilingData.mxScaleSize = mxScaleSize_;
    // Flags
    tilingData.betaFlag = betaFlag_;
    tilingData.rstdFlag = rstdFlag_;
}

void AddRmsNormDynamicMxQuantSplitRTiling::PrintTilingData()
{
    OP_LOGI(
        context_->GetNodeName(),
        "TilingData numCol: %lu, numColAlign: %lu, "
        "blockFactor: %lu, mLastCore: %lu, baseN: %lu, baseM: %lu, "
        "nUbLoops: %lu, binAddQuotient: %lu, powerSplit: %lu, "
        "mainFoldCount: %lu, foldTail: %lu, epsilon: %f, avgFactor: %f.",
        tilingData.numCol, tilingData.numColAlign, tilingData.blockFactor, tilingData.mLastCore,
        tilingData.baseN, tilingData.baseM, tilingData.nUbLoops, tilingData.binAddQuotient, tilingData.powerSplit,
        tilingData.mainFoldCount, tilingData.foldTail, tilingData.epsilon, tilingData.avgFactor);
    OP_LOGI(
        context_->GetNodeName(),
        "TilingData roundMode: %lu, mxBlockSize: %lu, scaleAlg: %ld, "
        "mxScaleSize: %lu, betaFlag: %u, rstdFlag: %u.",
        tilingData.roundMode, tilingData.mxBlockSize, tilingData.scaleAlg,
        tilingData.mxScaleSize, tilingData.betaFlag, tilingData.rstdFlag);
}

ge::graphStatus AddRmsNormDynamicMxQuantSplitRTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantSplitRTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        sizeof(tilingData) > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(tilingData),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormDynamicMxQuantSplitRTiling::GetTilingKey() const
{
    AddRmsNormDynamicMxQuantTilingKey tilingKey;
    tilingKey.SetComputeMode(ComputeMode::SPLIT_R);
    if (Y_SUPPORT_DTYPE_FP8_SET.count(yDtype_) != 0) {
        tilingKey.SetYDataType(YDataType::FP8);
    } else if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0) {
        tilingKey.SetYDataType(YDataType::FP4);
    }
    return tilingKey.GetTilingKey();
}

REGISTER_OPS_TILING_TEMPLATE(AddRmsNormDynamicMxQuant, AddRmsNormDynamicMxQuantSplitRTiling, ARND_SPLIT_R_PRIORITY);
} // namespace optiling
