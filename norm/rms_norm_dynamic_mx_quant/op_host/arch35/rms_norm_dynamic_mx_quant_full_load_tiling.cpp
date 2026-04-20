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
    constexpr int64_t BIN_ADD_FOLD_THRESHOLD = 16384;
    if (numN_ > BIN_ADD_FOLD_THRESHOLD) {
        OP_LOGD(
            context_->GetNodeName(),
            "FullLoad IsCapable false: numN=%ld >= binAddFoldThreshold=%ld, "
            "binary add rounds increase, recommend Recompute mode.",
            numN_, BIN_ADD_FOLD_THRESHOLD);
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
        OP_LOGD(
            context_->GetNodeName(),
            "DoOpTiling failed, usedCoreNum must not be 0, input shape: (%ld, %ld), total core num: %ld.", numM_, numN_,
            totalCoreNum_);
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
    constexpr int64_t RESERVED_UB_SIZE = 1024; // reserved for ub block align
    int64_t gammaAlignedSize = Ops::Base::CeilAlign(numN_ * gammaDtypeSize_, ubBlockSize_);
    int64_t gammaBetaUbSize = hasInputBeta_ ? gammaAlignedSize * CONST_TWO : gammaAlignedSize;

    int64_t mUbFactorMax = 0;
    if (Y_SUPPORT_DTYPE_FP4_SET.count(yDtype_) == 0) { // fp8量化
        mUbFactorMax =
            (ubSize_ - gammaBetaUbSize - RESERVED_UB_SIZE - CONST_THREE * nMxblockAligned * DOUBLE_BUFFER) /
            (nMxblockAligned * FP16_BYTES * CONST_THREE + binAddOutBufLen * FP32_BYTES + FP32_BYTES * CONST_THREE +
             nMxblockAligned * DOUBLE_BUFFER + nMxblockNumAlignedTwo * FP16_BYTES * CONST_FOUR);
    } else { // fp4量化
        mUbFactorMax =
            (ubSize_ - gammaBetaUbSize - RESERVED_UB_SIZE) /
            (nMxblockAligned * FP16_BYTES * CONST_THREE + binAddOutBufLen * FP32_BYTES + FP32_BYTES * CONST_THREE +
             nMxblockAligned / CONST_TWO * DOUBLE_BUFFER + nMxblockNumAlignedTwo * FP16_BYTES * CONST_FOUR);
    }

    OP_CHECK_IF(
        mUbFactorMax < 1,
        OP_LOGI(
            context_->GetNodeName(), "fused input shape (%ld, %ld) too large, full_load_template out of ub[%ld].",
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

REGISTER_OPS_TILING_TEMPLATE(RmsNormDynamicMxQuant, RmsNormDynamicMxQuantFullLoadTiling, TEMPLATE_FULL_LOAD_PRIORITY);

} // namespace optiling
