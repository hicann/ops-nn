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
 * \file bucketize_v2_simt_tiling.cpp
 * \brief
 */

#include "bucketize_v2_simt_tiling.h"
#include "../../op_kernel/arch35/bucketize_v2_struct.h"
#include "../../op_kernel/arch35/bucketize_v2_tiling_key.h"
#include "tiling/tiling_api.h"

namespace optiling {

static constexpr int64_t B16 = 2;
static constexpr int64_t B32 = 4;
static constexpr int64_t B64 = 8;
static constexpr uint64_t TEMPLATE_MODE = 3;
static constexpr int64_t THREAD_NUM = 2048;
static constexpr int64_t B32_VEC_BUND_THREHOLD = 16;
static constexpr int64_t B64_VEC_BUND_THREHOLD = 8;

using namespace BucketizeV2;
bool BucketizeV2SimtTiling::IsCapable()
{
    if (boundSize_ == 0) {
        return true;
    }
    if (boundDtypeSize_ <= B16 || (boundDtypeSize_ == B32 && boundSize_ <= B32_VEC_BUND_THREHOLD) ||
        (boundDtypeSize_ == B64 && boundSize_ <= B64_VEC_BUND_THREHOLD)) {
        return false;
    }
    return true;
}

uint64_t BucketizeV2SimtTiling::GetTilingKey() const
{
    OP_LOGI("BucketizeV2::GetTilingKey begin");
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, right_, IsUsedInt64_);
    OP_LOGI(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

void BucketizeV2SimtTiling::DoBlockTiling()
{
    int64_t blockNum = Ops::Base::CeilDiv(xSize_, THREAD_NUM);
    usedCoreNum_ = blockNum > coreNum_ ? coreNum_ : blockNum;
    maxIter_ = GetUpLog2(boundSize_);
    IsUsedInt64_ = boundSize_ >= std::numeric_limits<int32_t>::max();
}

void BucketizeV2SimtTiling::SetTilingData()
{
    BucketizeV2SimtTilingData* tilingData = context_->GetTilingData<BucketizeV2SimtTilingData>();
    tilingData->boundSize = boundSize_;
    tilingData->xSize = xSize_;
    tilingData->maxIter = maxIter_;
}

ge::graphStatus BucketizeV2SimtTiling::DoOpTiling()
{
    DoBlockTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BucketizeV2SimtTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

void BucketizeV2SimtTiling::DumpTilingInfo()
{
    BucketizeV2SimtTilingData* tilingData = context_->GetTilingData<BucketizeV2SimtTilingData>();
    std::string str;
    str += " boundSize:" + std::to_string(tilingData->boundSize);
    str += " xSize:" + std::to_string(tilingData->xSize);
    str += " maxIter:" + std::to_string(tilingData->maxIter);
    OP_LOGI(context_->GetNodeName(), "%s", str.c_str());
}

REGISTER_OPS_TILING_TEMPLATE(BucketizeV2, BucketizeV2SimtTiling, 5);

} // namespace optiling
