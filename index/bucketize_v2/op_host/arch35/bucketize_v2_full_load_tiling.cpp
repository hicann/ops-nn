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
 * \file bucketize_v2_full_load_tiling.cpp
 * \brief 
 */

#include "bucketize_v2_full_load_tiling.h"
#include "../../op_kernel/arch35/bucketize_v2_struct.h"
#include "../../op_kernel/arch35/bucketize_v2_tiling_key.h"
#include "tiling/tiling_api.h"

namespace optiling {

static constexpr int64_t MIN_INPUT_OUT_SIZE= 16 * 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t MIN_BLOCKFACTOR = 1024;
static constexpr int64_t INDICE_B16_WRITE_B64_ONCE_REG_NUM = 4;
static constexpr int64_t COMMON_WRITE_ONCE_REG_NUM = 2;
static constexpr int64_t B64_LOAD_ONCE_REG_NUM = 2;
static constexpr int64_t B16 = 2;
static constexpr uint64_t TEMPLATE_MODE = 1;

using namespace BucketizeV2;
bool BucketizeV2FullLoadTiling::IsCapable()
{
    if (boundDtypeSize_ <= B16 && boundSize_ > static_cast<int64_t>(std::numeric_limits<uint16_t>::max()) + 1) {
        return false;
    }
    if (boundSize_ * boundDtypeSize_ + BUFFER_NUM * DOUBLE_BUFFER * MIN_INPUT_OUT_SIZE < ubSize_) {
        return true;
    }
    return false;
}

uint64_t BucketizeV2FullLoadTiling::GetTilingKey() const
{
    OP_LOGI("BucketizeV2::GetTilingKey begin");
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, right_);
    OP_LOGI(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

void BucketizeV2FullLoadTiling::DoBlockTiling()
{
    blockFactor_ = Ops::Base::CeilDiv(xSize_, coreNum_);
    int64_t minBlockElement = MIN_BLOCKFACTOR / xDtypeSize_;
    blockFactor_ = Ops::Base::CeilAlign(blockFactor_, minBlockElement);
    usedCoreNum_ = Ops::Base::CeilDiv(xSize_, blockFactor_);
    blockTail_ = xSize_ - (usedCoreNum_ - 1) * blockFactor_;
}

void BucketizeV2FullLoadTiling::DoUBTiling()
{
    boundBufSize_ = Ops::Base::CeilAlign(boundSize_ * boundDtypeSize_, ubBlockSize_);
    int64_t ubAvailable = static_cast<int64_t>(ubSize_) - boundBufSize_;
    int64_t inputUbAlignFactor = xDtypeSize_ == sizeof(int64_t) ? B64_LOAD_ONCE_REG_NUM * vRegLength_ : vRegLength_;
    int64_t onceStoreVregNum = yDtypeSize_ == sizeof(int64_t) && xDtypeSize_ <= sizeof(int16_t) ? INDICE_B16_WRITE_B64_ONCE_REG_NUM : COMMON_WRITE_ONCE_REG_NUM;
    int64_t outputUbAlignFactor = onceStoreVregNum * vRegLength_;
    ubFactor_ = (ubAvailable / DOUBLE_BUFFER -  inputUbAlignFactor - outputUbAlignFactor) / (xDtypeSize_ + yDtypeSize_);
    inUbSize_ = Ops::Base::CeilAlign(ubFactor_ * xDtypeSize_, inputUbAlignFactor);
    outUbSize_ = Ops::Base::CeilAlign(ubFactor_ * yDtypeSize_, outputUbAlignFactor);
    maxIter_ = GetUpLog2(boundSize_);
}

void BucketizeV2FullLoadTiling::SetTilingData()
{
    BucketizeV2FullLoadTilingData* tilingData =
    context_->GetTilingData<BucketizeV2FullLoadTilingData>();

    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->ubFactor = ubFactor_;
    tilingData->boundBufSize = boundBufSize_;
    tilingData->inUbSize = inUbSize_;
    tilingData->outUbSize = outUbSize_;
    tilingData->boundSize = boundSize_;
    tilingData->maxIter = maxIter_;
}

ge::graphStatus BucketizeV2FullLoadTiling::DoOpTiling()
{
    ubBlockSize_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    vRegLength_ = static_cast<int64_t>(Ops::Base::GetVRegSize(context_));
    DoBlockTiling();
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BucketizeV2FullLoadTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

void BucketizeV2FullLoadTiling::DumpTilingInfo()
{
    BucketizeV2FullLoadTilingData* tilingData =
    context_->GetTilingData<BucketizeV2FullLoadTilingData>();
    std::string str;
    str += " usedCoreNum:" + std::to_string(tilingData->usedCoreNum);
    str += " blockFactor:" + std::to_string(tilingData->blockFactor);
    str += " blockTail:" + std::to_string(tilingData->blockTail);
    str += " ubFactor:" + std::to_string(tilingData->ubFactor);
    str += " boundBufSize:" + std::to_string(tilingData->boundBufSize);
    str += " inUbSize:" + std::to_string(tilingData->inUbSize);
    str += " outUbSize:" + std::to_string(tilingData->outUbSize);
    str += " boundSize:" + std::to_string(tilingData->boundSize);
    str += " maxIter:" + std::to_string(tilingData->maxIter);
    OP_LOGI(context_->GetNodeName(), "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(BucketizeV2, BucketizeV2FullLoadTiling, 10);

} // namespace optiling
