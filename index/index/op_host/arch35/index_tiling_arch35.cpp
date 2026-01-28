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
 * \file index_tiling.cc
 * \brief ac index tiling cpp
 */

#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "op_common/op_host/util/const_util.h"
#include "util/math_util.h"
#include "index_tiling.h"
#include "index_tiling_arch35.h"

using namespace AscendC;

namespace optiling {
// tilingkey
constexpr uint64_t DTYPE_BOOL = 11;
constexpr uint64_t DTYPE_INT8 = 1;
constexpr uint64_t DTYPE_F16 = 2;
constexpr uint64_t DTYPE_BF16 = 3;
constexpr uint64_t DTYPE_INT32 = 4;
constexpr uint64_t DTYPE_F32 = 5;
constexpr uint64_t DTYPE_INT64 = 8;

constexpr uint64_t INPUT_DTYPE_B8 = 1;
constexpr uint64_t INPUT_DTYPE_B16 = 2;
constexpr uint64_t INPUT_DTYPE_B32 = 4;
constexpr uint64_t INPUT_DTYPE_B64 = 8;
constexpr uint64_t INPUT_DTYPE_B128 = 16;

constexpr uint32_t IDX_TYPE_TILING_KEY_WEIGHT = 100;
constexpr size_t INDEXED_SIZES_IDX = 1;
constexpr size_t INDICES_IDX = 3;
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t MAX_DIM = 8;
constexpr uint32_t LIMIT_DIM = 5;
#ifdef DAVID_FPGA
constexpr uint32_t MAX_THREAD = 128;
constexpr uint32_t LIMIT_THREAD = 64;
#else
constexpr uint32_t MAX_THREAD = 512;
constexpr uint32_t LIMIT_THREAD = 256;
#endif
constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;
constexpr int32_t DIM_2 = 2;
constexpr uint32_t INDEX_PERF_TILING_KEY = 1000;
constexpr uint32_t INDEX_SUPPORT_INT64_TILING_KEY = 10000;

uint64_t GetDataTypeInByte(gert::TilingContext* context)
{
    auto paramsDtype = context->GetInputDesc(0)->GetDataType();
    uint64_t tilingKey{0};
    if (paramsDtype == ge::DT_INT64) {
        tilingKey = INPUT_DTYPE_B64;
    } else if ((paramsDtype == ge::DT_INT32) || (paramsDtype == ge::DT_FLOAT)) {
        tilingKey = INPUT_DTYPE_B32;
    } else if ((paramsDtype == ge::DT_FLOAT16) || (paramsDtype == ge::DT_BF16)) {
        tilingKey = INPUT_DTYPE_B16;
    } else if ((paramsDtype == ge::DT_INT8) || (paramsDtype == ge::DT_UINT8) || (paramsDtype == ge::DT_BOOL)) {
        tilingKey = INPUT_DTYPE_B8;
    } else {
        OP_LOGE("IndexSimt", "input x dtype bytes error!");
    }
    return tilingKey;
}

uint32_t IndexSimtTiling::ParamsDtypeImprove(uint32_t lastDimSize, uint32_t dataTypeBytes)
{
    uint32_t lastAxisByte = lastDimSize * dataTypeBytes;
    // for accumulate mode, atomic add is not supported for int4 type
    if ((dataTypeBytes < INPUT_DTYPE_B128) && ((lastAxisByte % INPUT_DTYPE_B128) == 0)) {
        OP_LOGD("IndexSimt", "ParamsDtypeImprove lastAxisByte %u, improve to INPUT_DTYPE_B128", lastAxisByte);
        return INPUT_DTYPE_B128 / dataTypeBytes;
    }

    if ((dataTypeBytes < INPUT_DTYPE_B64) && ((lastAxisByte % INPUT_DTYPE_B64) == 0)) {
        OP_LOGD("IndexSimt", "ParamsDtypeImprove lastAxisByte %u, improve to INPUT_DTYPE_B64", lastAxisByte);
        return INPUT_DTYPE_B64 / dataTypeBytes;
    }

    if ((dataTypeBytes < INPUT_DTYPE_B32) && ((lastAxisByte % INPUT_DTYPE_B32) == 0)) {
        OP_LOGD("IndexSimt", "ParamsDtypeImprove lastAxisByte %u, improve to INPUT_DTYPE_B32", lastAxisByte);
        return INPUT_DTYPE_B32 / dataTypeBytes;
    }

    if ((dataTypeBytes < INPUT_DTYPE_B16) && ((lastAxisByte % INPUT_DTYPE_B16) == 0)) {
        OP_LOGD("IndexSimt", "ParamsDtypeImprove lastAxisByte %u, improve to INPUT_DTYPE_B16", lastAxisByte);
        return INPUT_DTYPE_B16 / dataTypeBytes;
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
        uint64_t dataTypeBytes = GetDataTypeInByte(context_);
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
        PerfTilingData_.set_outputLength(outputLength_);
        PerfTilingData_.set_inputShape0(inputShapes_[0]);
        PerfTilingData_.set_inputShape1(inputShapes_[1]);
    }
    const int64_t paramIndexedSizesIdx = isIndexPut_ ? INDEXED_SIZES_IDX + 1 : INDEXED_SIZES_IDX;
    auto const indexedSizes = context_->GetInputShape(paramIndexedSizesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexedSizes);
    auto const indexedSizesShape = indexedSizes->GetStorageShape();
    auto const indexedSizesNum = indexedSizesShape.GetDim(0);
    OP_LOGI("IndexSimt", "input indexed_sizes size: %ld", indexedSizesNum);
    tilingData_.set_indexedSizesNum(indexedSizesNum);
    tilingData_.set_indexedDimNum(indicesNum);
    auto curIndexShape = context_->GetDynamicInputShape(paramIndicesIdx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, curIndexShape);
    auto const indexSizeVal = curIndexShape->GetStorageShape();
    uint64_t indexLength = indexSizeVal.GetShapeSize();
    // the length of index tensor since they shoud be the same for every dim
    tilingData_.set_indexSize(indexLength);
    OP_LOGD("IndexSimt", "indices number: %d, index length: %lu", indicesNum, indexLength);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexSimtTiling::GetShapeAttrsInfo()
{
    if (!isIndexPut_) {
        accumulateMode_ = false;
        tilingData_.set_accumulateMode(0);
    } else {
        // attr
        auto const attrs = context_->GetAttrs();
        auto* accumuMode = attrs->GetAttrPointer<bool>(0);
        if (*accumuMode) {
            tilingData_.set_accumulateMode(1);
            accumulateMode_ = true;
            OP_LOGD("IndexPutV2", "accumulate mode enabled.");
        } else {
            accumulateMode_ = false;
            tilingData_.set_accumulateMode(0);
        }
    }

    auto getResult = GetParamsShapeInfo();
    if (getResult != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return GenerateTilingKey();
}

ge::graphStatus IndexSimtTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexSimtTiling::GenerateTilingKey()
{
    if (!isIndexPut_) {
        return GenIndexTilingKey();
    }
    auto firstInput = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, firstInput);
    auto paramsDtype = firstInput->GetDataType();
    uint64_t tilingKey;
    if (paramsDtype == ge::DT_INT64) {
        tilingKey = DTYPE_INT64;
    } else if (paramsDtype == ge::DT_INT32) {
        tilingKey = DTYPE_INT32;
    } else if (paramsDtype == ge::DT_FLOAT) {
        tilingKey = DTYPE_F32;
    } else if (paramsDtype == ge::DT_FLOAT16) {
        tilingKey = DTYPE_F16;
    } else if (paramsDtype == ge::DT_BF16) {
        tilingKey = DTYPE_BF16;
    } else if (paramsDtype == ge::DT_INT8) {
        tilingKey = DTYPE_INT8;
    } else if (paramsDtype == ge::DT_BOOL) {
        tilingKey = DTYPE_BOOL;
    } else if (paramsDtype == ge::DT_UINT8) {
        tilingKey = 0;
    } else {
        OP_LOGE("IndexSimt", "input x dtype error!");
        return ge::GRAPH_FAILED;
    }
    auto factor = UpdateTilingData();
    if (factor == 0) {
        OP_LOGE("IndexSimt", "get index flag failed!");
    }
    if ((tilingKey == 0 || tilingKey == DTYPE_BOOL) && factor != 1) {
        tilingKey_ = factor;
    } else {
        tilingKey_ = tilingKey * factor;
    }
    OP_LOGD("IndexSimt", "tiling key is %u", tilingKey_);
    auto idxInput = context_->GetInputDesc(isIndexPut_ ? INDICES_IDX + 1 : INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idxInput);
    auto idxDtype = idxInput->GetDataType();
    if (idxDtype == ge::DT_INT64) {
        tilingKey_ += IDX_TYPE_TILING_KEY_WEIGHT;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexSimtTiling::GenIndexTilingKey()
{
    if (isPerfTemplate_) {
        tilingKey_ = INDEX_PERF_TILING_KEY;
        tilingKey_ += GetDataTypeInByte(context_);
        if (inputLength_ > INT32_MAX || outputLength_ > INT32_MAX) {
            tilingKey_ += INDEX_SUPPORT_INT64_TILING_KEY;
        }
        return ge::GRAPH_SUCCESS;
    }
    auto firstInput = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, firstInput);
    auto paramsDtype = firstInput->GetDataType();
    uint64_t tilingKey;
    if (paramsDtype == ge::DT_INT64) {
        tilingKey = DTYPE_INT64;
    } else if (paramsDtype == ge::DT_INT32 || paramsDtype == ge::DT_FLOAT) {
        tilingKey = DTYPE_INT32;
    } else if (paramsDtype == ge::DT_FLOAT16 || paramsDtype == ge::DT_BF16) {
        tilingKey = DTYPE_F16;
    } else if (paramsDtype == ge::DT_INT8 || paramsDtype == ge::DT_BOOL || paramsDtype == ge::DT_UINT8) {
        tilingKey = DTYPE_INT8;
    } else {
        OP_LOGE("IndexSimt", "input x dtype error!");
        return ge::GRAPH_FAILED;
    }
    auto factor = UpdateTilingData();
    if (factor == 0UL) {
        OP_LOGE("IndexSimt", "get index flag failed!");
    }
    tilingKey_ = static_cast<uint32_t>(tilingKey * factor);
    if (inputLength_ > INT32_MAX || outputLength_ > INT32_MAX) {
        tilingKey_ += INDEX_SUPPORT_INT64_TILING_KEY;
    }
    OP_LOGI("IndexSimt", "tiling key is %u", tilingKey_);
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexSimtTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus IndexSimtTiling::GetWorkspaceSize()
{
    size_t* workspace = context_->GetWorkspaceSizes(1);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexSimtTiling::PostTiling()
{
    tilingData_.set_inputShape(inputShapes_);
    tilingData_.set_inputDimNum(inputDimNum_);
    tilingData_.set_inputLength(inputLength_);
    tilingData_.set_outputLength(outputLength_);
    uint64_t usedThread = indexedDimNum_ >= LIMIT_DIM ? LIMIT_THREAD : MAX_THREAD;
    if (isPerfTemplate_) {
        usedThread = MAX_THREAD;
        PerfTilingData_.SaveToBuffer(
            context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(PerfTilingData_.GetDataSize());
    } else {
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    }
    context_->SetBlockDim(std::min(Ops::Base::CeilDiv(outputLength_, usedThread), coreNum_));
    context_->SetLocalMemorySize(DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}
REGISTER_OPS_TILING_TEMPLATE(Index, IndexSimtTiling, 30);
} // namespace optiling
