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
 * \file index_tiling_arch35.cpp
 * \brief SIMT tiling implementation for Index/IndexPutV2 operators
 */

#include <string_view>
#include "op_common/op_host/util/const_util.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/index_tiling_key.h"
#include "index_tiling.h"
#include "index_tiling_arch35.h"

using namespace Index;
namespace optiling {
constexpr size_t INDEXED_SIZES_IDX = 1;
constexpr size_t INDICES_IDX = 3;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t MAX_DIM = 8;
constexpr uint32_t LIMIT_DIM = 5;
#ifdef DAVID_FPGA
constexpr uint32_t MAX_THREAD = 128;
constexpr uint32_t LIMIT_THREAD = 64;
#else
constexpr uint32_t MAX_THREAD = 512;
constexpr uint32_t LIMIT_THREAD = 256;
#endif
constexpr int32_t DIM_2 = 2;

template <typename T>
ge::graphStatus InitTilingData(gert::TilingContext* context, T*& tilingData)
{
    tilingData = context->GetTilingData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(memset_s(tilingData, sizeof(T), 0, sizeof(T)) != EOK,
                OP_LOGE(context->GetNodeName(), "set tiling data error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint32_t IndexSimtTiling::ParamsDtypeImprove(uint32_t lastDimSize, uint32_t dataTypeBytes)
{
    uint32_t lastAxisByte = lastDimSize * dataTypeBytes;
    // for accumulate mode, atomic add is not supported for int4 type
    if ((dataTypeBytes < DTYPE_SIZE_B128) && ((lastAxisByte % DTYPE_SIZE_B128) == 0)) {
        OP_LOGD("IndexSimt", "ParamsDtypeImprove lastAxisByte %u, improve to DTYPE_SIZE_B128", lastAxisByte);
        return DTYPE_SIZE_B128 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B64) && ((lastAxisByte % DTYPE_SIZE_B64) == 0)) {
        OP_LOGD("IndexSimt", "ParamsDtypeImprove lastAxisByte %u, improve to DTYPE_SIZE_B64", lastAxisByte);
        return DTYPE_SIZE_B64 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B32) && ((lastAxisByte % DTYPE_SIZE_B32) == 0)) {
        OP_LOGD("IndexSimt", "ParamsDtypeImprove lastAxisByte %u, improve to DTYPE_SIZE_B32", lastAxisByte);
        return DTYPE_SIZE_B32 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B16) && ((lastAxisByte % DTYPE_SIZE_B16) == 0)) {
        OP_LOGD("IndexSimt", "ParamsDtypeImprove lastAxisByte %u, improve to DTYPE_SIZE_B16", lastAxisByte);
        return DTYPE_SIZE_B16 / dataTypeBytes;
    }
    return 0;
}

uint64_t IndexSimtTiling::UpdateTilingData()
{
    const int64_t paramIndexedSizesIdx = isIndexPut_ ? INDEXED_SIZES_IDX + 1 : INDEXED_SIZES_IDX;
    gert::Shape indexFlag;
    if (!Ops::Base::GetConstIntToShape(context_, paramIndexedSizesIdx, indexFlag)) {
        return 0;
    }
    bool nonTailIndex = indexFlag.GetDimNum() < static_cast<size_t>(inputDimNum_);
    if (!nonTailIndex) {
        nonTailIndex = nonTailIndex || (!indexFlag[inputDimNum_ - 1]);
    }
    if (!accumulateMode_ && nonTailIndex) {
        uint64_t dataTypeBytes = GenXDtype();
        if (dataTypeBytes == 0UL) {
            OP_LOGE("IndexSimt", "GenXDtype failed");
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

ge::graphStatus IndexSimtTiling::GetParamsShapeInfo()
{
    OP_LOGD("Index", "Tiling4SimtIndex rt2.0 is running.");

    // get input shape & input shape's dim num
    auto const inShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inShape);
    auto const inShapeVal = inShape->GetStorageShape();
    inputLength_ = inShapeVal.GetShapeSize();
    const size_t inputRank = inShapeVal.GetDimNum();
    inputDimNum_ = inputRank;
    for (size_t i = 0; i < inputRank; ++i) {
        inputShapes_[i] = inShapeVal.GetDim(i);
    }
    OP_LOGD("IndexSimt", "input dim Num: %u", inputDimNum_);
    // get output size
    auto const outputSize = isIndexPut_ ? context_->GetInputShape(1) : context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputSize);
    auto const outputSizeSal = outputSize->GetStorageShape();
    outputLength_ = outputSizeSal.GetShapeSize();
    OP_LOGD("IndexSimt", "outputLength_: %lu", outputLength_);
    // get the size of indices of indexed dim
    auto paramIndicesIdx = isIndexPut_ ? INDICES_IDX + 1 : INDICES_IDX;
    int32_t indicesNum = 0;
    for (size_t i = 0; i < MAX_DIM; ++i) {
        auto curTensor = context_->GetDynamicInputTensor(paramIndicesIdx, i);
        if (curTensor == nullptr) {
            // the num of dims that are indexed
            indicesNum = i;
            break;
        }
    }
    if (context_->GetDynamicInputTensor(paramIndicesIdx, 0) != nullptr && indicesNum == 0) {
        indicesNum = MAX_DIM;
    }
    if (!isIndexPut_ && inputDimNum_ == static_cast<uint32_t>(DIM_2) && indicesNum == DIM_2) {
        isPerfTemplate_ = true;
    }
    indicesNum_ = static_cast<uint32_t>(indicesNum);
    const int64_t paramIndexedSizesIdx = isIndexPut_ ? INDEXED_SIZES_IDX + 1 : INDEXED_SIZES_IDX;
    auto const indexedSizes = context_->GetInputShape(paramIndexedSizesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexedSizes);
    auto const indexedSizesShape = indexedSizes->GetStorageShape();
    auto const indexedSizesNum = indexedSizesShape.GetDim(0);
    OP_LOGI("IndexSimt", "input indexed_sizes size: %ld", indexedSizesNum);
    indexedSizesNum_ = static_cast<uint32_t>(indexedSizesNum);
    auto curIndexShape = context_->GetDynamicInputShape(paramIndicesIdx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, curIndexShape);
    auto const indexSizeVal = curIndexShape->GetStorageShape();
    indexLength_ = indexSizeVal.GetShapeSize();
    // the length of index tensor since they shoud be the same for every dim
    OP_LOGD("IndexSimt", "indices number: %d, index length: %lu", indicesNum, indexLength_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexSimtTiling::GetShapeAttrsInfo()
{
    const char* op_type = context_->GetNodeType();
    OP_CHECK_NULL_WITH_CONTEXT(context_, op_type);
    OP_LOGD("IndexSimtTiling", "tiling for %s", op_type);
    isIndexPut_ = std::string_view(op_type) == "IndexPutV2";
    if (isIndexPut_) {
        auto const attrs = context_->GetAttrs();
        auto* accumuMode = attrs->GetAttrPointer<bool>(0);
        if (accumuMode != nullptr && *accumuMode) {
            accumulateMode_ = true;
            OP_LOGD("IndexPutV2", "accumulate mode enabled.");
        }
    }

    auto getResult = GetParamsShapeInfo();
    if (getResult != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    factor4Index_ = (!isIndexPut_ && isPerfTemplate_) ? 1 : UpdateTilingData();
    if (factor4Index_ == 0UL) {
        OP_LOGE("IndexSimt", "UpdateTilingData failed");
        return ge::GRAPH_FAILED;
    }
    xDtype_ = GenXDtype(factor4Index_);

    return isPerfTemplate_ ? InitTilingData(context_, perfTilingData_) : InitTilingData(context_, tilingData_);
}

uint64_t IndexSimtTiling::GetTilingKey() const
{
    uint32_t isOverlength = inputLength_ > UINT32_MAX || outputLength_ > UINT32_MAX;
    return GET_TPL_TILING_KEY(xDtype_, 0, isPerfTemplate_, 0, 0, accumulateMode_, isOverlength);
}

ge::graphStatus IndexSimtTiling::PostTiling()
{
    uint64_t usedThread = indexedDimNum_ >= LIMIT_DIM ? LIMIT_THREAD : MAX_THREAD;
    if (isPerfTemplate_) {
        OP_CHECK_NULL_WITH_CONTEXT(context_, perfTilingData_);
        usedThread = MAX_THREAD;
        perfTilingData_->outputLength = outputLength_;
        perfTilingData_->inputShape0 = inputShapes_[0];
        perfTilingData_->inputShape1 = inputShapes_[1];
    } else {
        OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
        for (int i = 0; i < MAX_DIM; i++) {
            tilingData_->inputShape[i] = inputShapes_[i];
        }
        tilingData_->indexedSizesNum = indexedSizesNum_;
        tilingData_->indexedDimNum = indicesNum_;
        tilingData_->indexSize = indexLength_;
        tilingData_->accumulateMode = accumulateMode_ ? 1 : 0;
        tilingData_->inputDimNum = inputDimNum_;
        tilingData_->inputLength = inputLength_;
        tilingData_->outputLength = outputLength_;
    }
    context_->SetBlockDim(std::min(Ops::Base::CeilDiv(outputLength_, usedThread), coreNum_));
    context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Index, IndexSimtTiling, 30);
REGISTER_OPS_TILING_TEMPLATE(IndexPutV2, IndexSimtTiling, 30);
} // namespace optiling
