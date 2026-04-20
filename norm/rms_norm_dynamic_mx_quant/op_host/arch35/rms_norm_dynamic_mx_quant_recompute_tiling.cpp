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
 * \file rms_norm_dynamic_mx_quant_recompute_tiling.cpp
 * \brief RmsNormDynamicMxQuant Recompute tiling implementation
 */

#include "rms_norm_dynamic_mx_quant_tiling.h"

using namespace ge;
using namespace optiling::rms_norm_dynamic_mx_quant;

namespace optiling {

// ============================================================
// helpers
// ============================================================
int64_t RmsNormDynamicMxQuantRecomputeTiling::GetUbSize(int64_t baseN)
{
    constexpr int64_t RESERVED_UB_SIZE = 1024;
    int64_t mxBlockSize = MX_BLOCK_SIZE;

    int64_t xBufSize = baseN * FP16_BYTES * DOUBLE_BUFFER;
    int64_t xFoldBufSize = xBufSize;
    int64_t xFp32TmpBufSize = baseN * FP32_BYTES;
    int64_t binaryAddBufSize = vlFp32_ * CONST_TWO * FP32_BYTES;

    int64_t powerSplit = GetPowerSplit(baseN, numN_);
    int64_t cacheId = GetCacheId(powerSplit - 1);
    int64_t cacheBufSize = (cacheId + 1) * ubBlockSize_;

    int64_t rstdOutBufSize = baseM_ * FP32_BYTES * DOUBLE_BUFFER;
    int64_t gammaAlignedSize = baseN * gammaDtypeSize_ * DOUBLE_BUFFER;
    int64_t gammaBetaUbSize = hasInputBeta_ ? gammaAlignedSize * CONST_TWO : gammaAlignedSize;

    int64_t normOutBufSize = baseN * FP16_BYTES;
    int64_t scaleBufSize = static_cast<int64_t>(baseN / mxBlockSize) * FP16_BYTES;
    int64_t maxExpBufSize = scaleBufSize;
    int64_t maxHalfScaleBufSize = scaleBufSize;
    int64_t scaleOutBufSize = scaleBufSize * DOUBLE_BUFFER;

    int64_t yOutBufSize = CONST_FOUR * baseN * DOUBLE_BUFFER;
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) != 0) {
        yOutBufSize = baseN / CONST_TWO * DOUBLE_BUFFER;
    }

    return RESERVED_UB_SIZE + xBufSize + xFoldBufSize + xFp32TmpBufSize + binaryAddBufSize + cacheBufSize +
           rstdOutBufSize + gammaBetaUbSize + normOutBufSize + maxExpBufSize + maxHalfScaleBufSize + scaleOutBufSize +
           yOutBufSize;
}

int64_t RmsNormDynamicMxQuantRecomputeTiling::GetPowerSplit(int64_t baseN, int64_t numN) const
{
    int64_t ubLoops = Ops::Base::CeilDiv(numN, baseN);
    int64_t powerSplit = ubLoops == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubLoops)));
    powerSplit = (powerSplit == ubLoops) ? powerSplit / CONST_TWO : powerSplit;
    if (ubLoops == 1) {
        powerSplit = 1;
    }
    return powerSplit;
}

int64_t RmsNormDynamicMxQuantRecomputeTiling::GetCacheId(int64_t idx) const
{
    return __builtin_popcountll(idx ^ (idx + 1)) - 1;
}

// ============================================================
// IsCapable: recompute 至少需要 numN >= baseN(64) 才能做一次 ub 间二分
// ============================================================
bool RmsNormDynamicMxQuantRecomputeTiling::IsCapable()
{
    return numN_ > baseN_;
}

// ============================================================
// CalcMxQuantParams: 计算 mx quant 相关参数
// ============================================================
void RmsNormDynamicMxQuantRecomputeTiling::CalcMxQuantParams(MxQuantParams& params)
{
    params.mxBlockSize = MX_BLOCK_SIZE;
    int64_t nMxblockAligned = Ops::Base::CeilAlign(numN_, MX_BLOCK_SIZE_DOUBLE);
    params.nMxblockNumAlignedTwo = nMxblockAligned / MX_BLOCK_SIZE;
    params.needPadN = (nMxblockAligned == numN_) ? 0 : 1;
}

// ============================================================
// ExpandBaseN: 尽量扩大 baseN（保持 2 的幂次）
// ============================================================
ge::graphStatus RmsNormDynamicMxQuantRecomputeTiling::ExpandBaseN()
{
    while (baseN_ * CONST_TWO <= numN_ && GetUbSize(baseN_ * CONST_TWO) <= ubSize_) {
        baseN_ *= CONST_TWO;
        OP_LOGD(
            context_->GetNodeName(), "baseN*2: %ld, GetUbSize: %ld, ubSize_: %ld.", baseN_ * 2, GetUbSize(baseN_ * 2),
            ubSize_);
    }
    OP_LOGI(context_->GetNodeName(), "baseN: %ld, GetUbSize: %ld, ubSize_: %ld.", baseN_, GetUbSize(baseN_), ubSize_);

    OP_CHECK_IF(
        GetUbSize(baseN_) > ubSize_,
        OP_LOGE(
            context_->GetNodeName(),
            "recompute_template: input shape (%ld, %ld) too large, UB required %ld > UB size %ld.", numM_, numN_,
            GetUbSize(baseN_), ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// CalcCoreSplit: 分核计算
// ============================================================
void RmsNormDynamicMxQuantRecomputeTiling::CalcCoreSplit(CoreSplitResult& result)
{
    result.mPerCore = numM_ / usedCoreNum_;
    result.mTailCores = numM_ - usedCoreNum_ * result.mPerCore;
}

// ============================================================
// CalcNSplit: R 轴切分计算
// ============================================================
void RmsNormDynamicMxQuantRecomputeTiling::CalcNSplit(NSplitResult& result)
{
    int64_t nMxblockAligned = Ops::Base::CeilAlign(numN_, MX_BLOCK_SIZE_DOUBLE);
    result.nUbLoops = Ops::Base::CeilDiv(numN_, baseN_);
    result.baseNTail = numN_ - (result.nUbLoops - 1) * baseN_;
    result.baseNTailAligned = nMxblockAligned - (result.nUbLoops - 1) * baseN_;

    result.binAddQuotient = baseN_ <= 1 ? 0 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(baseN_ - 1)));
    if (result.binAddQuotient == baseN_) {
        result.binAddQuotient /= CONST_TWO;
    }
    if (baseN_ <= CONST_TWO) {
        result.binAddQuotient = (baseN_ == CONST_TWO) ? CONST_ONE : CONST_ZERO;
    }

    result.powerSplit = GetPowerSplit(baseN_, numN_);
    result.mainFoldCount =
        (static_cast<int64_t>(result.powerSplit) * baseN_ > numN_) ? 0 : (numN_ - result.powerSplit * baseN_) / baseN_;
    result.foldTail = numN_ % baseN_;
    result.resultCacheId = GetCacheId(result.powerSplit - 1);
}

// ============================================================
// FillTilingData: 填充 tiling data
// ============================================================
void RmsNormDynamicMxQuantRecomputeTiling::FillTilingData(
    const MxQuantParams& mxParams, const CoreSplitResult& coreSplit, const NSplitResult& nSplit)
{
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.mPerCore = coreSplit.mPerCore;
    tilingData_.mTailCores = coreSplit.mTailCores;
    tilingData_.numM = numM_;
    tilingData_.numN = numN_;
    tilingData_.baseM = baseM_;
    tilingData_.baseN = baseN_;
    tilingData_.baseNTail = nSplit.baseNTail;
    tilingData_.baseNTailAligned = nSplit.baseNTailAligned;
    tilingData_.nUbLoops = nSplit.nUbLoops;
    tilingData_.binAddQuotient = nSplit.binAddQuotient;
    tilingData_.powerSplit = nSplit.powerSplit;
    tilingData_.mainFoldCount = nSplit.mainFoldCount;
    tilingData_.foldTail = nSplit.foldTail;
    tilingData_.resultCacheId = nSplit.resultCacheId;
    tilingData_.mxBlockSize = mxParams.mxBlockSize;
    tilingData_.nMxblockNumAlignedTwo = mxParams.nMxblockNumAlignedTwo;
    tilingData_.needPadN = mxParams.needPadN;
    tilingData_.scaleAlg = scaleAlg_;
    tilingData_.roundMode = roundMode_;
    tilingData_.hasInputBeta = hasInputBeta_;
    tilingData_.hasOutputRstd = hasOutputRstd_;
    tilingData_.epsilon = epsilon_;
    tilingData_.avgFactor = avgFactor_;
}

// ============================================================
// DoOpTiling
// ============================================================
ge::graphStatus RmsNormDynamicMxQuantRecomputeTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormDynamicMxQuantRecomputeTiling DoOpTiling.");

    MxQuantParams mxParams;
    CalcMxQuantParams(mxParams);

    auto ret = ExpandBaseN();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    usedCoreNum_ = std::min(numM_, totalCoreNum_);
    if (usedCoreNum_ == 0) {
        OP_LOGD(
            context_->GetNodeName(),
            "DoOpTiling failed, usedCoreNum must not be 0, input shape: (%ld, %ld), total core num: %ld.", numM_, numN_,
            totalCoreNum_);
        return ge::GRAPH_FAILED;
    }

    CoreSplitResult coreSplit;
    CalcCoreSplit(coreSplit);

    NSplitResult nSplit;
    CalcNSplit(nSplit);

    FillTilingData(mxParams, coreSplit, nSplit);

    OP_LOGI(
        context_->GetNodeName(),
        "Recompute Tiling: usedCoreNum: %ld, numM: %ld, numN: %ld, baseM: %ld, baseN: %ld, mPerCore: %ld, "
        "mTailCores: %ld, nUbLoops: %ld, binAddQuotient: %ld, powerSplit: %ld, mainFoldCount: %ld, foldTail: %ld.",
        usedCoreNum_, numM_, numN_, baseM_, baseN_, coreSplit.mPerCore, coreSplit.mTailCores, nSplit.nUbLoops,
        nSplit.binAddQuotient, nSplit.powerSplit, nSplit.mainFoldCount, nSplit.foldTail);

    return ge::GRAPH_SUCCESS;
}

uint64_t RmsNormDynamicMxQuantRecomputeTiling::GetTilingKey() const
{
    RmsNormDynamicMxQuantTilingKey tilingKey;
    tilingKey.SetComputeMode(ComputeMode::RECOMPUTE);
    if (IsOptimizeCondition()) {
        tilingKey.SetOptimizeMode(OptimizeMode::OPTIMIZE);
    } else {
        tilingKey.SetOptimizeMode(OptimizeMode::NORMAL);
    }
    return tilingKey.GetTilingKey();
}

ge::graphStatus RmsNormDynamicMxQuantRecomputeTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    size_t tilingDataSize = sizeof(tilingData_);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    OP_CHECK_IF(ret != EOK, OP_LOGE(context_->GetNodeName(), "memcpy_s failed."), return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormDynamicMxQuant, RmsNormDynamicMxQuantRecomputeTiling, TEMPLATE_RECOMPUTE_PRIORITY);

} // namespace optiling
