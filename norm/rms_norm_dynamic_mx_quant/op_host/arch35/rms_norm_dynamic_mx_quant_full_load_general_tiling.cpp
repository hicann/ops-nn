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
 * \file rms_norm_dynamic_mx_quant_full_load_general_tiling.cpp
 * \brief RmsNormDynamicMxQuant FullLoad General tiling implementation
 */

#include "rms_norm_dynamic_mx_quant_tiling_arch35.h"
using namespace ge;

namespace optiling {

bool RmsNormDynamicMxQuantFullLoadGeneralTiling::IsCapable()
{
    // numN_ <= FULL_LOAD_THRESHOLD;
    return true;
}

ge::graphStatus RmsNormDynamicMxQuantFullLoadGeneralTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter RmsNormDynamicMxQuantFullLoadGeneralTiling DoOpTiling.");

    // mx quant

    int64_t fp32_bytes = sizeof(float);
    int64_t fp16_bytes = fp32_bytes / NUM_TWO;
    int64_t blockNumB16 = blockSize_ / fp16_bytes;
    int64_t numNUbAligned = (numN_ + blockNumB16 - 1) / blockNumB16 * blockNumB16;
    int64_t colFlodFactor =  1UL << (ULONG_BIT_LEN - 1 - __builtin_clzl(numNUbAligned));
    if(colFlodFactor == numNUbAligned){
        colFlodFactor /= NUM_TWO;
    }
    int64_t binAddVls = (colFlodFactor + vlFp32_ -1)/ vlFp32_;
    int64_t binAddTmpLen =(binAddVls + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;

    int64_t mxBlockSize = MX_BLOCK_SIZE; // numN_ < blockSize ? numN_ : MX_BLOCK_SIZE;
    int64_t totalNMxBlockAligned = (numN_ + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE * MX_BLOCK_SIZE;
    int64_t needPadN = totalNMxBlockAligned == numN_ ? 0 : 1;
    int64_t totalNMxBlockNum = totalNMxBlockAligned / MX_BLOCK_SIZE;
    int64_t totalNMxBlockNumAlignedTwo = (totalNMxBlockNum + 1) / NUM_TWO * NUM_TWO;
    int64_t needPadScale = totalNMxBlockNumAlignedTwo == totalNMxBlockNum ? 0 : 1;

    // 分核
    usedCoreNum_ = std::min(numM_, totalCoreNum_);
    if(usedCoreNum_ == 0){
        return ge::GRAPH_FAILED;
    }
    int64_t mPerCore = numM_ / usedCoreNum_;
    int64_t mTailCores = numM_ - usedCoreNum_ * mPerCore;
    int64_t mPerTailCore = mTailCores > 0 ? (mPerCore + 1) : mPerCore;

    // ub切分
    int64_t gammaAlignedSize = (numN_ * gammaDtypeSize_ + blockSize_ - 1) / blockSize_ * blockSize_; 
    int64_t gammaBetaUbSize = hasInputBeta_ ? gammaAlignedSize * 2 : gammaAlignedSize;

    int64_t mUbFactorMax = 
        (ubSize_ - gammaBetaUbSize - BLOCKALIGNRESERVE - NUM_THREE * totalNMxBlockAligned * DOUBLE_BUFFER) /
        (totalNMxBlockAligned * DOUBLE_BUFFER * fp16_bytes + binAddTmpLen * fp32_bytes + fp32_bytes * NUM_THREE + 
        totalNMxBlockAligned * fp16_bytes + totalNMxBlockAligned * DOUBLE_BUFFER +
        totalNMxBlockNumAlignedTwo * fp16_bytes * (NUM_FOUR + needPadScale) + totalNMxBlockAligned);

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
    tilingData_.set_colFlodFactor(colFlodFactor);
    tilingData_.set_mxBlockSize(mxBlockSize);
    tilingData_.set_totalNMxBlockAligned(totalNMxBlockAligned);
    tilingData_.set_totalNMxBlockNumAlignedTwo(totalNMxBlockNumAlignedTwo);
    tilingData_.set_totalNMxBlockNum(totalNMxBlockNum);
    tilingData_.set_needPadN(needPadN);
    tilingData_.set_needPadScale(needPadScale);
    tilingData_.set_scaleAlg(scaleAlg_);
    tilingData_.set_hasInputBeta(hasInputBeta_);
    tilingData_.set_hasOutputRstd(hasOutputRstd_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_avgFactor(avgFactor_);

    OP_LOGI(
        context_->GetNodeName(),
        "FullLoadGeneral Tiling: usedCoreNum: %ld, numM: %ld, numN: %ld, "
        "mPerCore: %ld, mTailCores: %ld, needPadN: %ld, mUbFactor: %ld, mUbFactorMax: %ld",
        usedCoreNum_, numM_, numN_, mPerCore, mTailCores, needPadN, mUbFactor, mUbFactorMax);

    return ge::GRAPH_SUCCESS;
}

uint64_t RmsNormDynamicMxQuantFullLoadGeneralTiling::GetTilingKey() const
{
    return TILINGKEY_FULL_LOAD_GENERAL;
}

ge::graphStatus RmsNormDynamicMxQuantFullLoadGeneralTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_; //+ Ops::Base::CeilAlign(mxScaleSize_, WORKSPACE_ALIGN_SIZE);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(
    RmsNormDynamicMxQuant, RmsNormDynamicMxQuantFullLoadGeneralTiling, TEMPLATE_FULL_LOAD_GENERAL_PRIORITY);

} // namespace optiling
