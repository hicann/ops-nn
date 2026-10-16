/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../add_rms_norm_hi_f4_quant_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

using namespace optiling;
using namespace NormCheck;

namespace optiling {

constexpr uint32_t RETAINED_SIZE_256 = 256;

int64_t AddRmsNormHiF4QuantSplitDTiling::GetPowerSplit(uint64_t numN)
{
    uint64_t ubLoops = Ops::Base::CeilDiv(numCol_, numN);
    int64_t powerSplit = ubLoops == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubLoops)));
    powerSplit = (static_cast<uint64_t>(powerSplit) == ubLoops) ? powerSplit / NUM_TWO : powerSplit;
    if (ubLoops == 1) {
        powerSplit = 1;
    }
    return powerSplit;
}

int64_t AddRmsNormHiF4QuantSplitDTiling::GetCacheId(int64_t idx) { return __builtin_popcountll(idx ^ (idx + 1)) - 1; }

uint64_t AddRmsNormHiF4QuantSplitDTiling::CalUBTotalSize(uint64_t candidateN, uint64_t powerSplit)
{
    uint64_t vlfp32 = vecLengthFP32_;
    uint64_t cacheBufSize = static_cast<uint64_t>(GetCacheId(powerSplit - 1) + 1) * ubBlockSize_;
    uint64_t binaryAddBufSize = Ops::Base::CeilAlign(vlfp32 * FP32_SIZE * NUM_TWO, ubBlockSize_);
    uint64_t rstdBufSize = Ops::Base::CeilAlign(baseM_ * FP32_SIZE, ubBlockSize_) * DOUBLE_BUFFER;

    uint64_t xInputBuf = NUM_TWO * candidateN * xDtypeSize_ * DOUBLE_BUFFER;
    uint64_t gammaBetaBuf = (1 + betaFlag_) * Ops::Base::CeilAlign(candidateN * gammaDtypeSize_, ubBlockSize_) *
                            DOUBLE_BUFFER;
    uint64_t yBuf = Ops::Base::CeilAlign(candidateN / NUM_TWO, ubBlockSize_) * DOUBLE_BUFFER;
    uint64_t xOutBuf = Ops::Base::CeilAlign(candidateN * xDtypeSize_, ubBlockSize_) * DOUBLE_BUFFER;
    uint64_t yTmpBuf = Ops::Base::CeilAlign(candidateN * xDtypeSize_, ubBlockSize_);

    int64_t ubFactorBlockNum = static_cast<int64_t>(candidateN / mxBlockSize_);
    int64_t totalGroupsPerRow = ubFactorBlockNum;
    int64_t alignTotalGroups = Ops::Base::CeilAlign(totalGroupsPerRow, static_cast<int64_t>(HIF4_BLOCK_SIZE));
    uint64_t scaleBufSize = Ops::Base::CeilAlign(static_cast<uint64_t>(alignTotalGroups) * sizeof(float),
                                                 ubBlockSize_) *
                            DOUBLE_BUFFER;

    uint64_t xFp32Buf = Ops::Base::CeilAlign(candidateN * FP32_SIZE, ubBlockSize_);

    uint64_t bufE8Size = Ops::Base::CeilAlign(static_cast<uint64_t>(totalGroupsPerRow),
                                              static_cast<uint64_t>(HIF4_BLOCK_SIZE)) *
                         sizeof(int16_t);
    uint64_t buf1E8Size = Ops::Base::CeilAlign(static_cast<uint64_t>(totalGroupsPerRow),
                                               static_cast<uint64_t>(HIF4_BLOCK_SIZE)) *
                          sizeof(int32_t);
    uint64_t buf1E16Size = Ops::Base::CeilAlign(static_cast<uint64_t>(totalGroupsPerRow),
                                                static_cast<uint64_t>(HIF4_BLOCK_SIZE)) *
                           sizeof(int16_t);
    uint64_t maxPer4Size = Ops::Base::CeilAlign(static_cast<uint64_t>(totalGroupsPerRow),
                                                static_cast<uint64_t>(DQ_HIFP4_MAXPER4_ALIGN)) *
                           16 * sizeof(int16_t);
    uint64_t maxPer8Size = Ops::Base::CeilAlign(static_cast<uint64_t>(totalGroupsPerRow),
                                                static_cast<uint64_t>(DQ_HIFP4_MAXPER8_ALIGN)) *
                           8 * sizeof(int16_t);
    uint64_t tmpScaleSize = Ops::Base::CeilAlign(static_cast<uint64_t>(totalGroupsPerRow),
                                                 static_cast<uint64_t>(DQ_HIFP4_SCALETMP_ALIGN)) *
                            32 * sizeof(int16_t);

    uint64_t totalNDependent = xInputBuf + gammaBetaBuf + yBuf + xOutBuf + yTmpBuf + scaleBufSize + xFp32Buf +
                               cacheBufSize + binaryAddBufSize + rstdBufSize + bufE8Size + buf1E8Size + buf1E16Size +
                               maxPer4Size + maxPer8Size + tmpScaleSize;

    return totalNDependent;
}

uint64_t AddRmsNormHiF4QuantSplitDTiling::GetMaxBaseN(uint64_t initialN)
{
    uint64_t vlfp32 = vecLengthFP32_;
    uint64_t binaryAddElemtMaxLen = vlfp32 * vlfp32 * NUM_TWO * NUM_TWO;

    int64_t powerSplit = GetPowerSplit(initialN);
    while (powerSplit > 1 && NUM_TWO * initialN <= binaryAddElemtMaxLen) {
        uint64_t candidateN = NUM_TWO * initialN;
        uint64_t newPowerSplit = GetPowerSplit(candidateN);
        uint64_t totalSize = CalUBTotalSize(candidateN, newPowerSplit);
        if (totalSize > maxUbSize_ - RETAINED_SIZE_256) {
            break;
        }
        initialN = candidateN;
        powerSplit = newPowerSplit;
    }
    return initialN;
}

bool AddRmsNormHiF4QuantSplitDTiling::IsCapable() { return true; }

ge::graphStatus AddRmsNormHiF4QuantSplitDTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DoOpTiling for SplitD.");

    mPerCore_ = Ops::Base::CeilDiv(numRow_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(numRow_, mPerCore_);
    mLastCore_ = numRow_ - (usedCoreNum_ - 1) * mPerCore_;
    blockFactor_ = mPerCore_;

    baseN_ = GetMaxBaseN(HIF4_BLOCK_SIZE);
    baseNBlockSize_ = baseN_ / HIF4_BLOCK_SIZE;
    nUbLoops_ = Ops::Base::CeilDiv(numCol_, baseN_);

    binAddQuotient_ = baseN_ == 0 ? 1 : (1UL << (ULONG_BIT_LEN - 1 - __builtin_clzl(baseN_)));
    binAddQuotient_ = (binAddQuotient_ == baseN_) ? binAddQuotient_ / NUM_TWO : binAddQuotient_;
    powerSplit_ = GetPowerSplit(baseN_);
    mainFoldCount_ = powerSplit_ * baseN_ > numCol_ ? 0 : (numCol_ - powerSplit_ * baseN_) / baseN_;
    foldTail_ = numCol_ % baseN_;

    OP_LOGI(context_->GetNodeName(),
            "SplitD: baseN=%lu, baseM=%lu, nUbLoops=%lu, powerSplit=%lu, "
            "mainFoldCount=%lu, foldTail=%lu, blockFactor=%lu.",
            baseN_, baseM_, nUbLoops_, powerSplit_, mainFoldCount_, foldTail_, blockFactor_);

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormHiF4QuantSplitDTiling::SetTilingData()
{
    tilingData.numCol = numCol_;
    tilingData.numColAlign = numColAlign_;
    tilingData.blockFactor = blockFactor_;
    tilingData.mLastCore = mLastCore_;
    tilingData.baseN = baseN_;
    tilingData.baseNBlockSize = baseNBlockSize_;
    tilingData.baseM = baseM_;
    tilingData.nUbLoops = nUbLoops_;
    tilingData.binAddQuotient = binAddQuotient_;
    tilingData.powerSplit = powerSplit_;
    tilingData.mainFoldCount = mainFoldCount_;
    tilingData.foldTail = foldTail_;
    tilingData.epsilon = epsilon_;
    tilingData.avgFactor = avgFactor_;
    tilingData.mxBlockSize = mxBlockSize_;
    tilingData.mxScaleSize = mxScaleSize_;
    tilingData.betaFlag = betaFlag_;
    tilingData.rstdFlag = rstdFlag_;
    tilingData.scaleAlg = scaleAlg_;

    int64_t ubFactorBlockNum = static_cast<int64_t>(baseN_ / HIF4_BLOCK_SIZE);
    int64_t totalGroupsPerRow = ubFactorBlockNum;
    int64_t ubFormer = Ops::Base::CeilAlign(totalGroupsPerRow, HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM);
    int64_t innerLoopEle = Ops::Base::CeilDiv(ubFormer, HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM);
    int64_t max8LoopEle = Ops::Base::CeilDiv(ubFormer, HIFP4_MAXPER8_REG_ONELOOP);
    int64_t scaleLoopEle = Ops::Base::CeilDiv(ubFormer, HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM);
    int64_t outLoopEle = Ops::Base::CeilDiv(ubFormer, HIFP4_OUTPUT_REG_ONELOOP);

    tilingData.innerLoopEle = innerLoopEle;
    tilingData.max8LoopEle = max8LoopEle;
    tilingData.scaleLoopEle = scaleLoopEle;
    tilingData.outLoopEle = outLoopEle;
}

void AddRmsNormHiF4QuantSplitDTiling::PrintTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "SplitD TilingData numCol=%lu, baseN=%lu, baseM=%lu, nUbLoops=%lu, "
            "binAddQuot=%lu, powerSplit=%lu, mainFoldCount=%lu, foldTail=%lu, "
            "blockFactor=%lu, mLastCore=%lu, mxScaleSize=%lu.",
            tilingData.numCol, tilingData.baseN, tilingData.baseM, tilingData.nUbLoops, tilingData.binAddQuotient,
            tilingData.powerSplit, tilingData.mainFoldCount, tilingData.foldTail, tilingData.blockFactor,
            tilingData.mLastCore, tilingData.mxScaleSize);
}

ge::graphStatus AddRmsNormHiF4QuantSplitDTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AddRmsNormHiF4QuantSplitDTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "SplitD usedCoreNum is %lu.", usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(tilingData) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(tilingData), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormHiF4QuantSplitDTiling::GetTilingKey() const { return 201; }

REGISTER_OPS_TILING_TEMPLATE(AddRmsNormHiF4Quant, AddRmsNormHiF4QuantSplitDTiling, ARND_SPLIT_D_PRIORITY);
} // namespace optiling
