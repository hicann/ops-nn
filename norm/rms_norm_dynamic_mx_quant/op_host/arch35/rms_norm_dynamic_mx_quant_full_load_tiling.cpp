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
 * \file rms_norm_dynamic_mx_quant_full_load_general_tiling.cpp
 * \brief RmsNormDynamicMxQuant FullLoad General tiling implementation
 */

#include "rms_norm_dynamic_mx_quant_tiling.h"

using namespace ge;
using namespace optiling::rms_norm_dynamic_mx_quant;

namespace optiling {

bool RmsNormDynamicMxQuantFullLoadTiling::IsCapable()
{
    int64_t binAddFoldThreshold = DOUBLE_BUFFER * vlFp32_ * vlFp32_ * CONST_TWO;
    if (numN_ > binAddFoldThreshold) {
        OP_LOGD(context_->GetNodeName(),
                "FullLoad IsCapable false: numN=%ld >= binAddFoldThreshold=%ld, "
                "binary add rounds increase, recommend Recompute mode.",
                numN_, binAddFoldThreshold);
        return false;
    }
    return true;
}

ge::graphStatus RmsNormDynamicMxQuantFullLoadTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormDynamicMxQuantFullLoadTiling DoOpTiling.");

    // 分核
    usedCoreNum_ = std::min(numM_, totalCoreNum_);
    if (usedCoreNum_ == 0) {
        OP_LOGD(context_->GetNodeName(),
                "DoOpTiling failed, usedCoreNum must not be 0, input shape: (%ld, %ld), total core num: %ld.", numM_,
                numN_, totalCoreNum_);
        return ge::GRAPH_FAILED;
    }
    int64_t mPerCore = numM_ / usedCoreNum_;
    int64_t mTailCores = numM_ - usedCoreNum_ * mPerCore;
    int64_t mPerTailCore = mTailCores > 0 ? (mPerCore + 1) : mPerCore;

    // mx quant
    int64_t mxBlockSize = MX_BLOCK_SIZE;
    int64_t nMxblockAligned = Ops::Base::CeilAlign(numN_, MX_BLOCK_SIZE_DOUBLE);
    int64_t nMxblockNumAlignedTwo = nMxblockAligned / mxBlockSize;
    int64_t needPadN = nMxblockAligned == numN_ ? 0 : 1;

    // bin add
    int64_t numNUbAligned = Ops::Base::CeilAlign(numN_, ubBlockB16Num_);
    int64_t binAddFoldPoint = FindNearestPower2(numNUbAligned);
    int64_t binAddVls = Ops::Base::CeilDiv(binAddFoldPoint, vlFp32_);
    int64_t binAddOutBufLen = Ops::Base::CeilAlign(binAddVls, ubBlockFp32Num_);

    // ub切分
    constexpr int64_t RESERVED_UB_SIZE = 1024;
    int64_t gammaAlignedSize = Ops::Base::CeilAlign(numN_, vlFp32_) * gammaDtypeSize_;
    int64_t gammaBetaUbSize = hasInputBeta_ ? gammaAlignedSize * CONST_TWO : gammaAlignedSize;

    // Match every InitBuffer allocation, including complete unmasked load tiles.
    // This total is monotonic in m, so a binary search retains the largest tile
    // that fits instead of relying on a linear estimate plus a fixed reserve.
    const auto getUbSize = [&](int64_t m) {
        int64_t xSize = m * nMxblockAligned * FP16_BYTES;
        int64_t normSize = Ops::Base::CeilAlign(m * nMxblockAligned, vlFp32_ * CONST_FOUR) * FP16_BYTES;
        int64_t reduceTmpCount = m * binAddOutBufLen;
        if (numN_ > vlFp32_ * CONST_TWO) {
            reduceTmpCount = (m - 1) * binAddOutBufLen + Ops::Base::CeilAlign(binAddOutBufLen, vlFp32_);
        }
        int64_t reduceTmpSize = reduceTmpCount * FP32_BYTES;
        int64_t rstdSize = Ops::Base::CeilAlign(m * FP32_BYTES, ubBlockSize_);
        int64_t reduceOutSize = Ops::Base::CeilAlign(m, vlFp32_) * FP32_BYTES;
        int64_t scaleOutSize = Ops::Base::CeilAlign(m * nMxblockNumAlignedTwo * FP16_BYTES, ubBlockSize_);
        int64_t scaleScratchSize = Ops::Base::CeilAlign(m * nMxblockNumAlignedTwo, vlFp32_ * CONST_TWO) * FP16_BYTES;
        int64_t ySize = Ops::Base::CeilAlign(m * nMxblockAligned / CONST_TWO, vlFp32_ * CONST_TWO);
        if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) == 0) {
            ySize = Ops::Base::CeilAlign(m, CONST_FOUR) * nMxblockAligned;
        }
        return RESERVED_UB_SIZE + gammaBetaUbSize + xSize * DOUBLE_BUFFER + normSize + reduceTmpSize +
               rstdSize * DOUBLE_BUFFER + reduceOutSize + scaleOutSize * DOUBLE_BUFFER + scaleScratchSize * CONST_TWO +
               ySize * DOUBLE_BUFFER;
    };
    int64_t mUbFactorMax = 0;
    int64_t lower = 1;
    int64_t upper = std::min(mPerTailCore, ubSize_ / nMxblockAligned / FP16_BYTES);
    while (lower <= upper) {
        int64_t middle = lower + (upper - lower) / CONST_TWO;
        if (getUbSize(middle) <= ubSize_) {
            mUbFactorMax = middle;
            lower = middle + 1;
        } else {
            upper = middle - 1;
        }
    }

    OP_CHECK_IF(
        mUbFactorMax < 1,
        OP_LOGI(context_->GetNodeName(), "fused input shape (%ld, %ld) too large, full_load_template out of ub[%ld].",
                numM_, numN_, ubSize_),
        return ge::GRAPH_PARAM_INVALID);

    int64_t mUbFactor = std::min(mPerTailCore, mUbFactorMax);

    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.mTailCores = mTailCores;
    tilingData_.numM = numM_;
    tilingData_.numN = numN_;
    tilingData_.mPerCore = mPerCore;
    tilingData_.mUbFactor = mUbFactor;
    tilingData_.binAddFoldPoint = binAddFoldPoint;
    tilingData_.mxBlockSize = mxBlockSize;
    tilingData_.nMxblockAligned = nMxblockAligned;
    tilingData_.nMxblockNumAlignedTwo = nMxblockNumAlignedTwo;
    tilingData_.needPadN = needPadN;
    tilingData_.scaleAlg = scaleAlg_;
    tilingData_.roundMode = roundMode_;
    tilingData_.hasInputBeta = hasInputBeta_;
    tilingData_.hasOutputRstd = hasOutputRstd_;
    tilingData_.epsilon = epsilon_;
    tilingData_.avgFactor = avgFactor_;

    return ge::GRAPH_SUCCESS;
}

uint64_t RmsNormDynamicMxQuantFullLoadTiling::GetTilingKey() const
{
    RmsNormDynamicMxQuantTilingKey tilingKey;
    tilingKey.SetComputeMode(ComputeMode::FULL_LOAD);
    if (IsOptimizeCondition()) {
        tilingKey.SetOptimizeMode(OptimizeMode::OPTIMIZE);
    } else {
        tilingKey.SetOptimizeMode(OptimizeMode::NORMAL);
    }
    return tilingKey.GetTilingKey();
}

ge::graphStatus RmsNormDynamicMxQuantFullLoadTiling::PostTiling()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    context_->SetBlockDim(usedCoreNum_);

    size_t tilingDataSize = sizeof(tilingData_);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<const void*>(&tilingData_), tilingDataSize);
    OP_CHECK_IF(ret != EOK, OP_LOGE(context_->GetNodeName(), "memcpy_s failed."), return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(RmsNormDynamicMxQuant, RmsNormDynamicMxQuantFullLoadTiling, TEMPLATE_FULL_LOAD_PRIORITY);

} // namespace optiling
