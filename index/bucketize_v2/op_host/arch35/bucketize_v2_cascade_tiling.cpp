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
 * \file bucketize_v2_cascade_tiling.cpp
 * \brief 
 */

#include "bucketize_v2_cascade_tiling.h"
#include "../../op_kernel/arch35/bucketize_v2_struct.h"
#include "../../op_kernel/arch35/bucketize_v2_tiling_key.h"
#include "tiling/tiling_api.h"

namespace optiling {

static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t MIN_BLOCKFACTOR = 1024;
static constexpr int64_t INDICE_B16_WRITE_B64_ONCE_REG_NUM = 4;
static constexpr int64_t COMMON_WRITE_ONCE_REG_NUM = 2;
static constexpr int64_t B64_LOAD_ONCE_REG_NUM = 2;
static constexpr int64_t B16 = 2;
static constexpr int64_t SIMT_DCACHE_SIZE = 32 * 1024;
static constexpr uint64_t TEMPLATE_MODE = 2;

using namespace BucketizeV2;
bool BucketizeV2CascadeTiling::IsCapable()
{
    return true;
}

uint64_t BucketizeV2CascadeTiling::GetTilingKey() const
{
    OP_LOGI("BucketizeV2::GetTilingKey begin");
    uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, right_);
    OP_LOGI(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

void BucketizeV2CascadeTiling::DoBlockTiling()
{
    blockFactor_ = Ops::Base::CeilDiv(xSize_, coreNum_);
    int64_t minBlockElement = MIN_BLOCKFACTOR / xDtypeSize_;
    blockFactor_ = Ops::Base::CeilAlign(blockFactor_, minBlockElement);
    usedCoreNum_ = Ops::Base::CeilDiv(xSize_, blockFactor_);
    blockTail_ = xSize_ - (usedCoreNum_ - 1) * blockFactor_;
}

void BucketizeV2CascadeTiling::DoUBTiling()
{
    int64_t inputUbAlignFactor = xDtypeSize_ == sizeof(int64_t) ? B64_LOAD_ONCE_REG_NUM * vRegLength_ : vRegLength_;
    int64_t onceWriteVregNum = yDtypeSize_ == sizeof(int64_t) && xDtypeSize_ <= B16 ? INDICE_B16_WRITE_B64_ONCE_REG_NUM : COMMON_WRITE_ONCE_REG_NUM;
    int64_t midOutputUbAlignFactor = onceWriteVregNum * vRegLength_;
    int64_t ubAvailable = static_cast<int64_t>(ubSize_) - SIMT_DCACHE_SIZE - DOUBLE_BUFFER * (inputUbAlignFactor + ubBlockSize_) - midOutputUbAlignFactor - ubBlockSize_;
    ubFactor_ = ubAvailable / (DOUBLE_BUFFER * xDtypeSize_ + (DOUBLE_BUFFER + 1) * yDtypeSize_ + boundDtypeSize_);
    
    inUbSize_ = Ops::Base::CeilAlign(ubFactor_ * xDtypeSize_, inputUbAlignFactor);
    outUbSize_ = Ops::Base::CeilAlign(ubFactor_ * yDtypeSize_, ubBlockSize_);
    midOutUbSize_ = Ops::Base::CeilAlign(ubFactor_ * yDtypeSize_, midOutputUbAlignFactor);
    boundBufSize_ = Ops::Base::CeilAlign(ubFactor_ * boundDtypeSize_, ubBlockSize_);
    int64_t maxBoundSampleSize = boundBufSize_ / boundDtypeSize_ - 1;
    sampleRatio_ = Ops::Base::CeilDiv(boundSize_, maxBoundSampleSize);
    sampleBoundSize_ = Ops::Base::CeilDiv(boundSize_, sampleRatio_) + 1;
    outerMaxIter_ = GetUpLog2(sampleBoundSize_);
    innerMaxIter_ = GetUpLog2(sampleRatio_ + 1);
}

void BucketizeV2CascadeTiling::SetTilingData()
{
    BucketizeV2CascadeTilingData* tilingData =
    context_->GetTilingData<BucketizeV2CascadeTilingData>();

    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->ubFactor = ubFactor_;
    tilingData->boundBufSize = boundBufSize_;
    tilingData->inUbSize = inUbSize_;
    tilingData->outUbSize = outUbSize_;
    tilingData->midOutUbSize = midOutUbSize_;
    tilingData->boundSize = boundSize_;
    tilingData->sampleBoundSize = sampleBoundSize_;
    tilingData->sampleRatio = sampleRatio_;
    tilingData->outerMaxIter = outerMaxIter_;
    tilingData->innerMaxIter = innerMaxIter_;
}

ge::graphStatus BucketizeV2CascadeTiling::DoOpTiling()
{
    ubBlockSize_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    vRegLength_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_));
    DoBlockTiling();
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BucketizeV2CascadeTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - SIMT_DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

void BucketizeV2CascadeTiling::DumpTilingInfo()
{
    BucketizeV2CascadeTilingData* tilingData =
    context_->GetTilingData<BucketizeV2CascadeTilingData>();
    std::string str;
    str += " usedCoreNum:" + std::to_string(tilingData->usedCoreNum);
    str += " blockFactor:" + std::to_string(tilingData->blockFactor);
    str += " blockTail:" + std::to_string(tilingData->blockTail);
    str += " ubFactor:" + std::to_string(tilingData->ubFactor);
    str += " boundBufSize:" + std::to_string(tilingData->boundBufSize);
    str += " inUbSize:" + std::to_string(tilingData->inUbSize);
    str += " outUbSize:" + std::to_string(tilingData->outUbSize);
    str += " midOutUbSize:" + std::to_string(tilingData->midOutUbSize);
    str += " boundSize:" + std::to_string(tilingData->boundSize);
    str += " sampleBoundSize:" + std::to_string(tilingData->sampleBoundSize);
    str += " sampleRatio:" + std::to_string(tilingData->sampleRatio);
    str += " outerMaxIter:" + std::to_string(tilingData->outerMaxIter);
    str += " innerMaxIter:" + std::to_string(tilingData->innerMaxIter);
    OP_LOGI(context_->GetNodeName(), "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(BucketizeV2, BucketizeV2CascadeTiling, 20);

} // namespace optiling
