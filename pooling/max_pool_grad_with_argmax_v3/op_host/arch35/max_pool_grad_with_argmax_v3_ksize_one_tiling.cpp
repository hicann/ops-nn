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
 * \file max_pool_grad_with_argmax_v3_ksize_one_tiling.cpp
 * \brief
 */
#include "max_pool_grad_with_argmax_v3_ksize_one_tiling.h"
#include "platform/platform_info.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling
{
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 1024;
static constexpr int64_t EXTRA_BUFFER_SIZE = 256;
static constexpr int64_t KSIZE_ONE_TILING_KEY = 800;
static constexpr int64_t T3_INT64 = 10;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CACHE_LINE_SIZE = 128;
static constexpr int64_t MIN_DATA_SIZE   = 1024;
static constexpr int64_t ALIGN_LENTH     = 512;

void MaxPoolGradWithArgmaxV3KsizeOneTiling::InitializationVars()
{
    baseData_.vRegSize = Ops::Base::GetVRegSize(context_);
    baseData_.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    baseData_.inputBytes = inputData.inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    baseData_.indexBytes = inputData.indexDtype == ge::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    baseData_.availableUb = hardwareData.ubSize - UB_RESVERVED_SIZE;
    baseData_.totalCoreNum = hardwareData.coreNum;
    baseData_.coreUsedForBestPerformance = baseData_.totalCoreNum;

    int64_t oneBlockNumT1 = baseData_.ubBlockSize / baseData_.inputBytes;
    int64_t oneBlockNumT2 = baseData_.ubBlockSize / baseData_.indexBytes;

    baseData_.maxDataNumInOneBlock = std::max(oneBlockNumT1, oneBlockNumT2);

    baseData_.proDataNumInOneBeatT2 = baseData_.vRegSize / baseData_.ubBlockSize * oneBlockNumT2;
    baseData_.moveDataNumCacheLineT2 = CACHE_LINE_SIZE / baseData_.indexBytes;

    baseData_.isPad = 0;
    if (inputData.hPad != 0 || inputData.wPad != 0) {
        baseData_.isPad = 1;
    }

    baseData_.hProBatchSize = 1;
    if (inputData.hKernel > inputData.hStride) {
        baseData_.hProBatchSize = Ops::Base::CeilDiv(inputData.hKernel, inputData.hStride);
    }

    baseData_.wProBatchSize = 1;
    if (inputData.wKernel > inputData.wStride) {
        baseData_.wProBatchSize = Ops::Base::CeilDiv(inputData.wKernel, inputData.wStride);
    }

    baseData_.isOverlap = 0;
    if (baseData_.wProBatchSize != 1 || baseData_.hProBatchSize != 1) {
        baseData_.isOverlap = 1;
    }
}

bool MaxPoolGradWithArgmaxV3KsizeOneTiling::IsCapable()
{
    if (inputData.hStride != 1 || inputData.wStride != 1 ||
        inputData.hKernel != 1 || inputData.wKernel != 1) {
        return false;
    }

    InitializationVars();
    return true;
}

uint64_t MaxPoolGradWithArgmaxV3KsizeOneTiling::GetTilingKey() const
{
    uint64_t tilingKey = KSIZE_ONE_TILING_KEY;
    return tilingKey;
}

void MaxPoolGradWithArgmaxV3KsizeOneTiling::DoUBTiling()
{
    ubFactor_ = baseData_.availableUb / BUFFER_NUM / baseData_.inputBytes;
    int64_t alignSize = ALIGN_LENTH / baseData_.inputBytes;
    int64_t coreData = Ops::Base::CeilDiv(inputData.gradShapeSize, baseData_.totalCoreNum);
    coreData = Ops::Base::CeilAlign(coreData, alignSize);
    coreData = std::max(coreData, MIN_DATA_SIZE);
    usedCoreNum_ = Ops::Base::CeilDiv(inputData.gradShapeSize, coreData);
    // 512字节对齐
    blockFactor_ = coreData; 
    tailBlockFactor_ = inputData.gradShapeSize - (usedCoreNum_ - 1) * blockFactor_;
    coreLoop_ = Ops::Base::CeilDiv(blockFactor_, ubFactor_);
    tailUbFactor_ = blockFactor_ - (coreLoop_ - 1) * ubFactor_;
    tailCoreLoop_ = Ops::Base::CeilDiv(tailBlockFactor_, ubFactor_);
    tailCoreTailUbFactor_ = tailBlockFactor_ - (tailCoreLoop_ - 1) * ubFactor_;
}


void MaxPoolGradWithArgmaxV3KsizeOneTiling::PrintBaseData() const
{
    OP_LOGD("MaxPoolGradWithArgmaxV3KsizeOne", "[MaxPoolGradWithArgmaxV3KsizeOne] PrintBaseData start running");

    std::ostringstream info;
    info << "baseData_.vRegSize: " << baseData_.vRegSize << std::endl;
    info << "baseData_.ubBlockSize: " << baseData_.ubBlockSize << std::endl;

    info << "baseData_.inputBytes: " << baseData_.inputBytes << std::endl;
    info << "baseData_.indexBytes: " << baseData_.indexBytes << std::endl;
    info << "baseData_.availableUb: " << baseData_.availableUb << std::endl;
    info << "baseData_.maxDataNumInOneBlock: " << baseData_.maxDataNumInOneBlock << std::endl;
    info << "baseData_.proDataNumInOneBeatT2: " << baseData_.proDataNumInOneBeatT2 << std::endl;
    info << "baseData_.totalCoreNum: " << baseData_.totalCoreNum << std::endl;
    info << "baseData_.coreUsedForBestPerformance: " << baseData_.coreUsedForBestPerformance << std::endl;

    info << "baseData_.isPad: " << baseData_.isPad << std::endl;
    info << "baseData_.isOverlap: " << baseData_.isOverlap << std::endl;
    info << "baseData_.hProBatchSize: " << baseData_.hProBatchSize << std::endl;
    info << "baseData_.wProBatchSize: " << baseData_.wProBatchSize << std::endl;
    info << "baseData_.moveDataNumCacheLineT2: " << baseData_.moveDataNumCacheLineT2 << std::endl;

    OP_LOGI("MaxPoolGradWithArgmaxV3NHWC", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxV3KsizeOneTiling::PrintTilingData() const
{
    OP_LOGD("MaxPoolGradWithArgmaxV3KsizeOne", "[MaxPoolGradWithArgmaxV3KsizeOneTiling] PrintTilingData start running");

    std::ostringstream info;
    info << "usedCoreNum: " << usedCoreNum_ << std::endl;
    info << "blockFactor: " << blockFactor_ << std::endl;
    info << "tailBlockFactor: " << tailBlockFactor_ << std::endl;
    info << "coreLoop: " << coreLoop_ << std::endl;
    info << "tailCoreLoop: " << tailCoreLoop_ << std::endl;
    info << "ubFactor: " << ubFactor_ << std::endl;
    info << "tailUbFactor: " << tailUbFactor_ << std::endl;
    info << "tailCoreTailUbFactor: " << tailCoreTailUbFactor_ << std::endl;

    OP_LOGI("MaxPoolGradWithArgmaxV3KsizeOne", "%s", info.str().c_str());
}

void MaxPoolGradWithArgmaxV3KsizeOneTiling::SetTilingData()
{
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_tailBlockFactor(tailBlockFactor_);
    tilingData_.set_coreLoop(coreLoop_);
    tilingData_.set_tailCoreLoop(tailCoreLoop_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_tailUbFactor(tailUbFactor_);
    tilingData_.set_tailCoreTailUbFactor(tailCoreTailUbFactor_);
    tilingData_.set_tilingKey(GetTilingKey());
}

ge::graphStatus MaxPoolGradWithArgmaxV3KsizeOneTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    PrintBaseData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradWithArgmaxV3KsizeOneTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolGradWithArgmaxV3, MaxPoolGradWithArgmaxV3KsizeOneTiling, 0);

}  // namespace optiling
