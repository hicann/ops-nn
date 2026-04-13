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

#include "rms_norm_dynamic_mx_quant_tiling_arch35.h"

using namespace ge;

namespace optiling {

bool RmsNormDynamicMxQuantFullLoadTiling::IsCapable()
{
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
        OP_LOGE(
            context_->GetNodeName(), "fused input shape (%ld, %ld) too large, full_load_template out of ub[%ld].",
            numM_, numN_, ubSize_),
        return ge::GRAPH_FAILED);

    int64_t mUbFactor = std::min(mPerTailCore, mUbFactorMax);

    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_mTailCores(mTailCores);
    tilingData_.set_numM(numM_);
    tilingData_.set_numN(numN_);
    tilingData_.set_numNUbAligned(numNUbAligned);
    tilingData_.set_mPerCore(mPerCore);
    tilingData_.set_mUbFactor(mUbFactor);
    tilingData_.set_binAddFoldPoint(binAddFoldPoint);
    tilingData_.set_mxBlockSize(mxBlockSize);
    tilingData_.set_nMxblockAligned(nMxblockAligned);
    tilingData_.set_nMxblockNumAlignedTwo(nMxblockNumAlignedTwo);
    tilingData_.set_needPadN(needPadN);
    tilingData_.set_scaleAlg(scaleAlg_);
    tilingData_.set_roundMode(roundMode_);
    tilingData_.set_hasInputBeta(hasInputBeta_);
    tilingData_.set_hasOutputRstd(hasOutputRstd_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_avgFactor(avgFactor_);

    return ge::GRAPH_SUCCESS;
}

uint64_t RmsNormDynamicMxQuantFullLoadTiling::GetTilingKey() const
{
    if (IsOptimizeCondition()) {
        return TILING_KEY_FULL_LOAD_OPTIMIZE;
    }
    return TILINGKEY_FULL_LOAD_GENERAL;
}

ge::graphStatus RmsNormDynamicMxQuantFullLoadTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(
    RmsNormDynamicMxQuant, RmsNormDynamicMxQuantFullLoadTiling, TEMPLATE_FULL_LOAD_GENERAL_PRIORITY);

} // namespace optiling
