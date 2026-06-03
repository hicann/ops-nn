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
 * \file scatter_nd_common_simt_tiling.cpp
 * \brief scatter_nd_common_simt_tiling
 */

#include "scatter_nd_common_simt_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

using namespace ScatterNdCommon;

static constexpr uint64_t RESERVE_SIZE = 256;
static constexpr int64_t DCACHE_SIZE = 131072;  // 128k
static constexpr int64_t TEMPLATE_MODE_SIMT = 8;
static constexpr uint64_t DB_BUFFER = 2;
static constexpr uint64_t MIN_TILING_SIZE = 128;


bool ScatterNdCommonSimtTiling::IsCapable()
{
    return true;
}

uint64_t ScatterNdCommonSimtTiling::GetTilingKey() const
{
    OP_LOGD("ScatterNdCommonSimtTiling::GetTilingKey begin");
    uint64_t addrMode = 1;
    if (indiceShapeSize_ < UINT32_MAX && updateShapeSize_ < UINT32_MAX && outputShapeSize_ < UINT32_MAX) {
        addrMode = 0;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE_SIMT, indiceCastMode_, addrMode);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

void ScatterNdCommonSimtTiling::SetTilingData()
{
    ScatterNdCommon::ScatterNdCommonSimtTilingData *tilingData = 
        context_->GetTilingData<ScatterNdCommon::ScatterNdCommonSimtTilingData>();

    for (int32_t i = 0; i < OPTILING_MAX_RANK_COUNT; i++) {
        tilingData->strideList[i] = strideList_[i];
    }
    for (int32_t i = 0; i < OPTILING_MAX_SHAPE_RANK; i++) {
        tilingData->outPutShape[i] = outPutShape_[i];
    }
    tilingData->blockNum = blockNum_;
    tilingData->rankSize = rankSize_;
    tilingData->blockTilingSize = blockTilingSize_;
    tilingData->tailBlockTilingSize = tailBlockTilingSize_;
    tilingData->ubTilingSize = ubTilingSize_;
    tilingData->sliceSize = sliceSize_;
    tilingData->varInAxis = varInAxis_;
}


ge::graphStatus ScatterNdCommonSimtTiling::BlockTiling()
{
    auto typeSize = ge::GetSizeByDataType(updateDtype_);
    OP_CHECK_IF(
        typeSize <= 0,
        OP_LOGE(context_, "update dtypeSize must be greater than 0, dtypeSize: %ld", typeSize),
        return ge::GRAPH_FAILED);
    alignFactor_ = Ops::Base::GetUbBlockSize(context_) / typeSize;
    auto blockFactor = Ops::Base::CeilDiv(updateShapeSize_, totalCoreNum_);
    auto blockAlignFactor = Ops::Base::CeilDiv(blockFactor, alignFactor_) * alignFactor_;
    blockTilingSize_ = std::max(static_cast<uint64_t>(blockAlignFactor), MIN_TILING_SIZE);
    blockNum_ = Ops::Base::CeilDiv(updateShapeSize_, blockTilingSize_);
    tailBlockTilingSize_ = updateShapeSize_ - blockTilingSize_ * (blockNum_ - 1UL);
    OP_LOGD(context_->GetNodeName(),
               "updateShapeSize = %lld, blockFactor = %lld, blockAlignFactor = %lld,"
               "blockTilingSize = %d, tailBlockTilingSize = %d", updateShapeSize_,
               blockFactor, blockAlignFactor, blockTilingSize_, tailBlockTilingSize_);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus ScatterNdCommonSimtTiling::UbTiling()
{
    if (indiceShapeSize_ == static_cast<uint64_t>(0) || updateShapeSize_ == static_cast<uint64_t>(0)) {
        return ge::GRAPH_SUCCESS;
    }
    // halfUbSize for double buffer
    auto halfUbSize = (ubSize_ - DCACHE_SIZE - RESERVE_SIZE) / DB_BUFFER;
    auto indiceNum = indiceShapeSize_ / rankSize_;
    sliceSize_ = updateShapeSize_ / indiceNum;
    OP_CHECK_IF(sliceSize_ == static_cast<uint64_t>(0),
                    OP_LOGE(context_->GetNodeName(), "sliceSize %lu is zero. please check.", sliceSize_),
                    return ge::GRAPH_FAILED);
    auto updateTypeSize = ge::GetSizeByDataType(updateDtype_);
    OP_CHECK_IF(
        updateTypeSize <= 0,
        OP_LOGE(context_, "updateTypeSize must be greater than 0, updateTypeSize: %ld", updateTypeSize),
        return ge::GRAPH_FAILED);
    indiceDtype_ = context_->GetInputDesc(INPUT_IDX_INDICES)->GetDataType();
    auto indiceTypeSize = ge::GetSizeByDataType(indiceDtype_);
    // sliceUb : the required size of UB for one scatter operation;
    auto sliceUb = sliceSize_ * updateTypeSize + rankSize_ * indiceTypeSize;
    sliceUb = Ops::Base::CeilDiv(sliceUb, static_cast<uint64_t>(alignFactor_)) * alignFactor_;
    if (sliceUb > halfUbSize) {
        // for scatter operator. At least  rank size index need to be move in UB.
        ubTilingSize_ = (halfUbSize - rankSize_ * indiceTypeSize) / updateTypeSize;
    } else {
        // calculate the size of updates that need to be move in UB
        auto maxIndiceCnt = halfUbSize / sliceUb;
        ubTilingSize_ = maxIndiceCnt * sliceSize_;
    }
    OP_LOGD(context_->GetNodeName(), "sliceUb = %lu, halfUbSize = %u, ubTilingSize = %u", sliceUb, halfUbSize,
                ubTilingSize_);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus ScatterNdCommonSimtTiling::DoOpTiling()
{
    ge::graphStatus res = BlockTiling();
    if (res == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    res = UbTiling();
    if (res == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    SetStride();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdCommonSimtTiling::PostTiling()
{
    OP_LOGD("ScatterNdCommonSimtTiling::PostTiling begin");
    context_->SetBlockDim(blockNum_);
    if (indiceShapeSize_ == 0 || updateShapeSize_ == 0) {
        // 输入为空tensor时，设置blockNum为1，在kernel中直接返回
        context_->SetBlockDim(1);
    }

    auto res = context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "SetLocalMemorySize ubSize = %lu failed.", ubSize_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


std::string ScatterNdCommonSimtTiling::TilingDataToString()
{
    ScatterNdCommon::ScatterNdCommonSimtTilingData* tilingData =
        context_->GetTilingData<ScatterNdCommon::ScatterNdCommonSimtTilingData>();
    std::string str = " blockNum:" + std::to_string(tilingData->blockNum);
    str += " rankSize:" + std::to_string(tilingData->rankSize);
    str += " blockTilingSize:" + std::to_string(tilingData->blockTilingSize);
    str += " tailBlockTilingSize:" + std::to_string(tilingData->tailBlockTilingSize);
    str += " ubTilingSize:" + std::to_string(tilingData->ubTilingSize);
    str += " sliceSize:" + std::to_string(tilingData->sliceSize);
    str += " varInAxis:" + std::to_string(tilingData->varInAxis);
    for (int32_t i = 0; i < OPTILING_MAX_SHAPE_RANK; i++) {
        str += " outPutShape[i]:" + std::to_string(tilingData->outPutShape[i]);
    }
    for (int32_t i = 0; i < OPTILING_MAX_RANK_COUNT; i++) {
        str += " strideList[i]:" + std::to_string(tilingData->strideList[i]);
    }
    return str;
}

void ScatterNdCommonSimtTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "Tiling info is: %s", TilingDataToString().c_str());
}

} // namespace optiling