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
 * \file index_tiling_broadcast.cpp
 * \brief Broadcast tiling implementation for Index operator (indices broadcast scenario)
 */

#include <algorithm>
#include "op_common/op_host/util/const_util.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/index_tiling_key.h"
#include "index_tiling.h"
#include "index_tiling_broadcast.h"

using namespace Index;
namespace optiling {
constexpr size_t BC_X_IDX = 0;
constexpr size_t BC_INDEXED_SIZES_IDX = 1;
constexpr size_t BC_INDICES_IDX = 3;
constexpr uint32_t BC_DCACHE_SIZE = 128 * 1024;
constexpr uint32_t BC_MAX_DIM = 8;
constexpr uint32_t BC_LIMIT_DIM = 5;
#ifdef DAVID_FPGA
constexpr uint32_t BC_MAX_THREAD = 128;
constexpr uint32_t BC_LIMIT_THREAD = 64;
#else
constexpr uint32_t BC_MAX_THREAD = 512;
constexpr uint32_t BC_LIMIT_THREAD = 256;
#endif

static bool IsRowMajorContinuous(const gert::Shape& shape, const gert::Stride& stride)
{
    int64_t validStride = 1;
    for (int64_t i = static_cast<int64_t>(shape.GetDimNum()) - 1; i >= 0; i--) {
        if (shape[i] == 1) {
            continue;
        }
        if (validStride != stride[i]) {
            return false;
        }
        validStride *= shape[i];
    }
    return true;
}

static bool IsXContinuous(gert::TilingContext* context)
{
    if (!context->InputIsView(BC_X_IDX)) {
        return true;
    }
    auto* xStride = context->GetInputStride(BC_X_IDX);
    if (xStride == nullptr || xStride->GetDimNum() == 0) {
        return true;
    }
    auto xShape = context->GetInputShape(BC_X_IDX);
    if (xShape == nullptr) {
        return false;
    }
    return IsRowMajorContinuous(xShape->GetShape(), *xStride);
}

static bool IsIndicesContinuous(gert::TilingContext* context, uint32_t indicesNum)
{
    if (!context->InputIsView(BC_INDICES_IDX)) {
        return true;
    }
    for (uint32_t j = 0; j < indicesNum; ++j) {
        auto* indexStride = context->GetDynamicInputStride(BC_INDICES_IDX, j);
        if (indexStride == nullptr || indexStride->GetDimNum() == 0) {
            continue;
        }
        auto indexShape = context->GetDynamicInputShape(BC_INDICES_IDX, j);
        if (indexShape == nullptr) {
            return false;
        }
        if (!IsRowMajorContinuous(indexShape->GetShape(), *indexStride)) {
            return false;
        }
    }
    return true;
}

bool IndexBroadcastTiling::IsCapable()
{
    uint32_t indicesNum = 0;
    for (size_t i = 0; i < BC_MAX_DIM; ++i) {
        if (context_->GetDynamicInputTensor(BC_INDICES_IDX, i) == nullptr) {
            indicesNum = i;
            break;
        }
    }
    if (context_->GetDynamicInputTensor(BC_INDICES_IDX, 0) != nullptr && indicesNum == 0) {
        indicesNum = BC_MAX_DIM;
    }
    if (!IndicesNeedBroadcast(context_, BC_INDICES_IDX, indicesNum)) {
        return false;
    }
    if (!IsXContinuous(context_) || !IsIndicesContinuous(context_, indicesNum)) {
        return false;
    }
    return true;
}

uint32_t IndexBroadcastTiling::ParamsDtypeImprove(uint32_t lastDimSize, uint32_t dataTypeBytes)
{
    uint32_t lastAxisByte = lastDimSize * dataTypeBytes;
    if ((dataTypeBytes < DTYPE_SIZE_B128) && ((lastAxisByte % DTYPE_SIZE_B128) == 0)) {
        OP_LOGD("IndexBroadcast", "ParamsDtypeImprove lastAxisByte %u, improve to DTYPE_SIZE_B128", lastAxisByte);
        return DTYPE_SIZE_B128 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B64) && ((lastAxisByte % DTYPE_SIZE_B64) == 0)) {
        OP_LOGD("IndexBroadcast", "ParamsDtypeImprove lastAxisByte %u, improve to DTYPE_SIZE_B64", lastAxisByte);
        return DTYPE_SIZE_B64 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B32) && ((lastAxisByte % DTYPE_SIZE_B32) == 0)) {
        OP_LOGD("IndexBroadcast", "ParamsDtypeImprove lastAxisByte %u, improve to DTYPE_SIZE_B32", lastAxisByte);
        return DTYPE_SIZE_B32 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B16) && ((lastAxisByte % DTYPE_SIZE_B16) == 0)) {
        OP_LOGD("IndexBroadcast", "ParamsDtypeImprove lastAxisByte %u, improve to DTYPE_SIZE_B16", lastAxisByte);
        return DTYPE_SIZE_B16 / dataTypeBytes;
    }
    return 0;
}

uint64_t IndexBroadcastTiling::UpdateTilingData()
{
    gert::Shape indexFlag;
    if (!Ops::Base::GetConstIntToShape(context_, BC_INDEXED_SIZES_IDX, indexFlag)) {
        return 0;
    }
    bool nonTailIndex = indexFlag.GetDimNum() < static_cast<size_t>(inputDimNum_);
    if (!nonTailIndex) {
        nonTailIndex = nonTailIndex || (!indexFlag[inputDimNum_ - 1]);
    }
    if (nonTailIndex) {
        uint64_t dataTypeBytes = GenXDtype();
        if (dataTypeBytes == 0UL) {
            auto firstInput = context_->GetInputDesc(0);
            if (firstInput != nullptr) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    "IndexBroadcast", "x", Ops::Base::ToString(firstInput->GetDataType()).c_str(),
                    "x dtype not supported, must be in [DT_INT64, DT_INT32, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, "
                    "DT_UINT8, DT_BOOL, DT_COMPLEX64]");
            }
            return 0;
        }
        uint64_t factor = ParamsDtypeImprove(inputShapes_[inputDimNum_ - 1], dataTypeBytes);
        if (factor) {
            inputLength_ /= factor;
            outputLength_ /= factor;
            inputShapes_[inputDimNum_ - 1] /= factor;
            return factor;
        }
    }
    return 1;
}

ge::graphStatus IndexBroadcastTiling::ComputeBroadcastInfo()
{
    gert::Shape indexShapes[BC_MAX_DIM];
    for (uint32_t j = 0; j < indicesNum_; ++j) {
        auto curShape = context_->GetDynamicInputShape(BC_INDICES_IDX, j);
        OP_CHECK_NULL_WITH_CONTEXT(context_, curShape);
        indexShapes[j] = curShape->GetStorageShape();
        broadcastDimNum_ = std::max(broadcastDimNum_, static_cast<uint32_t>(indexShapes[j].GetDimNum()));
    }
    OP_CHECK_IF(
        broadcastDimNum_ > BC_MAX_DIM,
        OP_LOGE(context_->GetNodeName(), "broadcast dim num %u exceeds max support %u", broadcastDimNum_, BC_MAX_DIM),
        return ge::GRAPH_FAILED);

    for (uint32_t d = 0; d < broadcastDimNum_; ++d) {
        uint64_t mergedDim = 1;
        for (uint32_t j = 0; j < indicesNum_; ++j) {
            int64_t dimIdx = static_cast<int64_t>(indexShapes[j].GetDimNum()) - broadcastDimNum_ + d;
            uint64_t curDim = (dimIdx >= 0) ? static_cast<uint64_t>(indexShapes[j].GetDim(dimIdx)) : 1;
            if (curDim != 1) {
                if (mergedDim == 1) {
                    mergedDim = curDim;
                } else if (curDim != mergedDim) {
                    OP_LOGE(context_->GetNodeName(), "indices shapes are not broadcastable at dim %u: %lu vs %lu", d,
                            curDim, mergedDim);
                    return ge::GRAPH_FAILED;
                }
            }
        }
        broadcastShape_[d] = mergedDim;
    }

    indexSize_ = 1;
    for (uint32_t d = 0; d < broadcastDimNum_; ++d) {
        indexSize_ *= broadcastShape_[d];
    }

    for (uint32_t j = 0; j < indicesNum_; ++j) {
        int64_t curDimNum = static_cast<int64_t>(indexShapes[j].GetDimNum());
        uint64_t curStride = 1;
        for (int64_t d = static_cast<int64_t>(broadcastDimNum_) - 1; d >= 0; --d) {
            int64_t dimIdx = curDimNum - static_cast<int64_t>(broadcastDimNum_) + d;
            if (dimIdx >= 0) {
                uint64_t dimVal = static_cast<uint64_t>(indexShapes[j].GetDim(dimIdx));
                indexBcStride_[j][d] = (dimVal == broadcastShape_[d]) ? curStride : 0;
                curStride *= dimVal;
            } else {
                indexBcStride_[j][d] = 0;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexBroadcastTiling::GetShapeAttrsInfo()
{
    OP_LOGD("Index", "Tiling4BroadcastIndex rt2.0 is running.");

    tilingData_ = context_->GetTilingData<IndexBroadcastTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    OP_CHECK_IF(memset_s(tilingData_, sizeof(IndexBroadcastTilingData), 0, sizeof(IndexBroadcastTilingData)) != EOK,
                OP_LOGE(context_->GetNodeName(), "set tiling data error"), return ge::GRAPH_FAILED);

    auto const inShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto const inShapeVal = inShape->GetStorageShape();
    inputLength_ = inShapeVal.GetShapeSize();
    inputDimNum_ = inShapeVal.GetDimNum();
    for (size_t i = 0; i < static_cast<size_t>(inputDimNum_); ++i) {
        inputShapes_[i] = inShapeVal.GetDim(i);
    }
    OP_LOGD("IndexBroadcast", "input dim Num: %u", inputDimNum_);
    auto const outputSize = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputSize);
    outputLength_ = outputSize->GetStorageShape().GetShapeSize();
    OP_LOGD("IndexBroadcast", "outputLength_: %lu", outputLength_);
    int32_t indicesNum = 0;
    for (size_t i = 0; i < BC_MAX_DIM; ++i) {
        if (context_->GetDynamicInputTensor(BC_INDICES_IDX, i) == nullptr) {
            indicesNum = i;
            break;
        }
    }
    if (context_->GetDynamicInputTensor(BC_INDICES_IDX, 0) != nullptr && indicesNum == 0) {
        indicesNum = BC_MAX_DIM;
    }
    indicesNum_ = static_cast<uint32_t>(indicesNum);
    auto const indexedSizes = context_->GetInputShape(BC_INDEXED_SIZES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexedSizes);
    indexedSizesNum_ = static_cast<uint32_t>(indexedSizes->GetStorageShape().GetDim(0));
    OP_LOGI("IndexBroadcast", "input indexed_sizes size: %u", indexedSizesNum_);

    if (ComputeBroadcastInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD("IndexBroadcast", "indices number: %u, broadcast dim num: %u, index size(numel B): %lu", indicesNum_,
            broadcastDimNum_, indexSize_);

    factor4Index_ = UpdateTilingData();
    if (factor4Index_ == 0UL) {
        OP_LOGE("IndexBroadcast", "UpdateTilingData failed");
        return ge::GRAPH_FAILED;
    }
    xDtype_ = GenXDtype(factor4Index_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexBroadcastTiling::DoOpTiling()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    for (uint32_t i = 0; i < BC_MAX_DIM; i++) {
        tilingData_->inputShape[i] = inputShapes_[i];
        tilingData_->broadcastShape[i] = broadcastShape_[i];
        for (uint32_t j = 0; j < BC_MAX_DIM; j++) {
            tilingData_->indexBcStride[j][i] = indexBcStride_[j][i];
        }
    }
    tilingData_->indexedSizesNum = indexedSizesNum_;
    tilingData_->indexedDimNum = indicesNum_;
    tilingData_->indexSize = indexSize_;
    tilingData_->broadcastDimNum = broadcastDimNum_;
    tilingData_->inputDimNum = inputDimNum_;
    tilingData_->inputLength = inputLength_;
    tilingData_->outputLength = outputLength_;
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexBroadcastTiling::GetTilingKey() const
{
    uint32_t isOverlength = inputLength_ > UINT32_MAX || outputLength_ > UINT32_MAX;
    return GET_TPL_TILING_KEY(xDtype_, INDEX_NOT_FULL_LOAD, 0, 0, 0, 0, isOverlength, 1);
}

ge::graphStatus IndexBroadcastTiling::PostTiling()
{
    uint64_t usedThread = indicesNum_ >= BC_LIMIT_DIM ? BC_LIMIT_THREAD : BC_MAX_THREAD;
    context_->SetBlockDim(std::min(Ops::Base::CeilDiv(outputLength_, usedThread), coreNum_));
    context_->SetLocalMemorySize(ubSize_ - BC_DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Index, IndexBroadcastTiling, 7);
} // namespace optiling
