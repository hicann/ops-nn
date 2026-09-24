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
 * \file index_tiling_nocon_broadcast.cpp
 * \brief Non-continuous view x indices broadcast tiling implementation for Index operator (priority 5)
 */

#include <algorithm>
#include <string>
#include "log/log.h"
#include "op_common/op_host/util/const_util.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/index_tiling_key.h"
#include "index_tiling.h"
#include "index_tiling_nocon_broadcast.h"

using namespace Index;
namespace optiling {
#ifdef DAVID_FPGA
constexpr uint32_t NOCON_BC_MAX_THREAD = 128;
constexpr uint32_t NOCON_BC_LIMIT_THREAD = 64;
#else
constexpr uint32_t NOCON_BC_MAX_THREAD = 512;
constexpr uint32_t NOCON_BC_LIMIT_THREAD = 256;
#endif
constexpr uint32_t NOCON_BC_DCACHE_SIZE = 32 * 1024;
constexpr uint32_t NOCON_BC_MAX_SUPPORT_DIM_NUM = 4;
static constexpr int64_t NOCON_BC_IN_X_IDX = 0;
static constexpr int64_t NOCON_BC_IN_INDEXSIZE_IDX = 1;
static constexpr int64_t NOCON_BC_IN_INDEX_IDX = 3;
static constexpr int64_t NOCON_BC_OUT_Y_IDX = 0;

bool IndexNoConBroadcastTiling::IsCapable()
{
    // take over only when indices need broadcast;
    if (!IndicesNeedBroadcast(context_, paramIndicesIdx_, static_cast<uint32_t>(tensorNum_))) {
        return false;
    }
    if (inputDimNum_ > NOCON_BC_MAX_SUPPORT_DIM_NUM || broadcastDimNum_ > NOCON_BC_MAX_SUPPORT_DIM_NUM ||
        tensorNum_ > static_cast<int64_t>(NOCON_BC_MAX_SUPPORT_DIM_NUM)) {
        return false;
    }
    if (!IsContinuous(xShape_, xStride_)) {
        return true;
    }
    for (int64_t j = 0; j < tensorNum_; ++j) {
        if (!IsContinuous(indexShapesVec_[j], indexStridesVec_[j])) {
            return true;
        }
    }
    return false;
}

ge::graphStatus IndexNoConBroadcastTiling::ComputeBroadcastInfo()
{
    uint32_t checkedNum = std::min(static_cast<int64_t>(NOCON_BC_MAX_SUPPORT_DIM_NUM), tensorNum_);
    for (uint32_t j = 0; j < checkedNum; ++j) {
        broadcastDimNum_ = std::max(broadcastDimNum_, static_cast<uint32_t>(indexShapesVec_[j].GetDimNum()));
    }
    if (broadcastDimNum_ > NOCON_BC_MAX_SUPPORT_DIM_NUM) {
        return ge::GRAPH_SUCCESS;
    }
    for (uint32_t d = 0; d < broadcastDimNum_; ++d) {
        int64_t mergedDim = 1;
        for (uint32_t j = 0; j < checkedNum; ++j) {
            int64_t aligned = static_cast<int64_t>(d) + static_cast<int64_t>(indexShapesVec_[j].GetDimNum()) -
                              static_cast<int64_t>(broadcastDimNum_);
            int64_t curDim = (aligned >= 0) ? indexShapesVec_[j].GetDim(aligned) : 1;
            if (curDim != 1) {
                if (mergedDim == 1) {
                    mergedDim = curDim;
                } else if (curDim != mergedDim) {
                    OP_LOGE(context_->GetNodeName(), "indices shapes are not broadcastable at dim %u: %ld vs %ld", d,
                            curDim, mergedDim);
                    return ge::GRAPH_FAILED;
                }
            }
        }
        broadcastShape_[d] = mergedDim;
    }

    indexSize_ = 1;
    for (uint32_t d = 0; d < broadcastDimNum_; ++d) {
        indexSize_ *= static_cast<uint64_t>(broadcastShape_[d]);
    }

    for (uint32_t j = 0; j < checkedNum; ++j) {
        int64_t rankJ = static_cast<int64_t>(indexShapesVec_[j].GetDimNum());
        for (uint32_t d = 0; d < broadcastDimNum_; ++d) {
            int64_t aligned = static_cast<int64_t>(d) + rankJ - static_cast<int64_t>(broadcastDimNum_);
            if (aligned < 0 || indexShapesVec_[j].GetDim(aligned) == 1) {
                indexBcStride_[j][d] = 0;
            } else {
                indexBcStride_[j][d] = indexStridesVec_[j][aligned];
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexNoConBroadcastTiling::GetShapeAttrsInfo()
{
    const char* opType = context_->GetNodeType();
    OP_CHECK_NULL_WITH_CONTEXT(context_, opType);
    OP_LOGD("IndexNoConBroadcastTiling", "tiling for %s", opType);
    isIndexPut_ = false;

    tilingData_ = context_->GetTilingData<IndexNoConBroadcastTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    OP_CHECK_IF(
        memset_s(tilingData_, sizeof(IndexNoConBroadcastTilingData), 0, sizeof(IndexNoConBroadcastTilingData)) != EOK,
        OP_LOGE(context_->GetNodeName(), "set tiling data error"), return ge::GRAPH_FAILED);

    paramIndexedSizesIdx_ = NOCON_BC_IN_INDEXSIZE_IDX;
    paramIndicesIdx_ = NOCON_BC_IN_INDEX_IDX;

    auto xDesc = context_->GetRequiredInputDesc(NOCON_BC_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();
    OP_CHECK_IF(ParamTypeIsInvalid(xDtype_),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "x", Ops::Base::ToString(xDtype_).c_str(),
                    "should be in [DT_FLOAT, DT_FLOAT16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8, DT_INT32, DT_INT64]"),
                return ge::GRAPH_FAILED);

    const std::set<ge::DataType> supportedIndexDtypes = {ge::DT_INT32, ge::DT_INT64};
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);
    auto indiceInstanceInfo = computeNodeInfo->GetInputInstanceInfo(paramIndicesIdx_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indiceInstanceInfo);
    tensorNum_ = indiceInstanceInfo->GetInstanceNum();
    OP_LOGI("IndexNoConBroadcast", "tensor Num: %ld", tensorNum_);
    int64_t checkedNum = std::min(static_cast<int64_t>(NOCON_BC_MAX_SUPPORT_DIM_NUM), tensorNum_);
    for (int64_t i = 0; i < checkedNum; ++i) {
        auto indexDesc = context_->GetDynamicInputDesc(paramIndicesIdx_, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, indexDesc);
        ge::DataType curIndexDtype = indexDesc->GetDataType();
        OP_CHECK_IF(supportedIndexDtypes.count(curIndexDtype) == 0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "index",
                                                          Ops::Base::ToString(curIndexDtype).c_str(),
                                                          "should be in [DT_INT32, DT_INT64]"),
                    return ge::GRAPH_FAILED;);
    }
    auto yDesc = context_->GetOutputDesc(NOCON_BC_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF(
        yDtype != xDtype_,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x, y",
            (Ops::Base::ToString(xDtype_) + ", " + Ops::Base::ToString(yDtype)).c_str(), "should have same dtype"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetTensorInfo(xShape_, xStride_, NOCON_BC_IN_X_IDX, false) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "get x tensor info failed"), return ge::GRAPH_FAILED);
    inputDimNum_ = xShape_.GetDimNum();
    auto const inShape = context_->GetInputShape(NOCON_BC_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    inputLength_ = inShape->GetShape().GetShapeSize();
    auto const indexedSizes = context_->GetInputShape(paramIndexedSizesIdx_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexedSizes);
    indexedSizesNum_ = indexedSizes->GetShape().GetDim(0);

    for (int64_t j = 0; j < checkedNum; ++j) {
        auto curShape = context_->GetDynamicInputShape(paramIndicesIdx_, j);
        OP_CHECK_NULL_WITH_CONTEXT(context_, curShape);
        indexShapesVec_[j] = curShape->GetShape();
        GetIndexStrideInfo(indexShapesVec_[j], indexStridesVec_[j], paramIndicesIdx_, j);
    }

    OP_CHECK_IF(GetTensorInfo(yShape_, yStride_, NOCON_BC_OUT_Y_IDX, true) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "get y tensor info failed"), return ge::GRAPH_FAILED);
    auto const outputSize = context_->GetOutputShape(NOCON_BC_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputSize);
    outputLength_ = outputSize->GetStorageShape().GetShapeSize();

    gert::Shape indexFlag;
    if (Ops::Base::GetConstIntToShape(context_, paramIndexedSizesIdx_, indexFlag)) {
        uint32_t idN = 0;
        for (uint32_t i = 0; i < inputDimNum_ && idN < NOCON_BC_MAX_SUPPORT_DIM_NUM; ++i) {
            if (i < indexFlag.GetDimNum() && indexFlag[i] != 0) {
                indexInputShape_[idN] = xShape_[i];
                idN++;
            }
        }
    }

    if (ComputeBroadcastInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD("IndexNoConBroadcast", "indices number: %ld, broadcast dim num: %u, index size(numel B): %lu", tensorNum_,
            broadcastDimNum_, indexSize_);
    return ge::GRAPH_SUCCESS;
}

void IndexNoConBroadcastTiling::SetTilingData()
{
    for (int64_t i = 0; i < static_cast<int64_t>(inputDimNum_); i++) {
        tilingData_->xShape[i] = xShape_[i];
        tilingData_->xStride[i] = xStride_[i];
    }
    for (uint32_t i = 0; i < broadcastDimNum_; i++) {
        tilingData_->broadcastShape[i] = broadcastShape_[i];
        tilingData_->yStride[i] = yStride_[i];
        for (uint32_t j = 0; j < NOCON_BC_MAX_SUPPORT_DIM_NUM; j++) {
            tilingData_->indexBcStride[j][i] = indexBcStride_[j][i];
        }
    }
    for (uint32_t j = 0; j < NOCON_BC_MAX_SUPPORT_DIM_NUM; j++) {
        tilingData_->indexInputShape[j] = indexInputShape_[j];
    }

    tilingData_->indexSize = indexSize_;
    tilingData_->indexedDimNum = tensorNum_;
    tilingData_->indexedSizesNum = indexedSizesNum_;
    tilingData_->inputDimNum = inputDimNum_;
    tilingData_->inputLength = inputLength_;
    tilingData_->outputLength = outputLength_;
    tilingData_->broadcastDimNum = broadcastDimNum_;
    tilingData_->accumulateMode = 0;
    tilingData_->valueDimNum = 0;
}

ge::graphStatus IndexNoConBroadcastTiling::DoOpTiling()
{
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexNoConBroadcastTiling::GetTilingKey() const
{
    auto idxDesc = context_->GetInputDesc(paramIndicesIdx_);
    uint32_t isOverlength = 0;
    if (idxDesc != nullptr) {
        isOverlength = inputLength_ > UINT32_MAX || outputLength_ > UINT32_MAX ||
                       idxDesc->GetDataType() == ge::DT_INT64;
    } else {
        isOverlength = inputLength_ > UINT32_MAX || outputLength_ > UINT32_MAX;
    }
    uint32_t xDtype = GenXDtype();
    return GET_TPL_TILING_KEY(xDtype, 0, 0, 0, 1, 0, isOverlength, 1);
}

ge::graphStatus IndexNoConBroadcastTiling::PostTiling()
{
    uint64_t usedThread = tensorNum_ >= static_cast<int64_t>(NOCON_BC_MAX_SUPPORT_DIM_NUM) ? NOCON_BC_LIMIT_THREAD :
                                                                                             NOCON_BC_MAX_THREAD;
    context_->SetBlockDim(std::min(Ops::Base::CeilDiv(outputLength_, usedThread), coreNum_));
    context_->SetLocalMemorySize(NOCON_BC_DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Index, IndexNoConBroadcastTiling, 5);
} // namespace optiling
