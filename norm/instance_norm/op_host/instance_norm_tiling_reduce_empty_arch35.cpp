/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file instance_norm_tiling_reduce_empty_arch35.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include "instance_norm_tiling.h"

using namespace ge;

namespace optiling {
constexpr uint64_t TILINGKEY_REDUCE_EMPTY = 50000;

constexpr static int32_t FP32_BYTE = 4;
constexpr static int32_t FP16_BYTE = 2;
constexpr static int64_t BUFFER_NUM = 1;
constexpr static int64_t REDUCE_EMPTY_CASE_ALLOWED_R = 0;
constexpr static int64_t SINGLE_AIV_CORE_THREFOLD_BYTES = 32L * 1024L;

void InstanceNormReduceEmptyTiling::Reset(gert::TilingContext* context)
{
    InstanceNormRegbaseTilingBase::Reset(context);
    blockNum_ = 0;
}

bool InstanceNormReduceEmptyTiling::IsCapable()
{
    if (r != REDUCE_EMPTY_CASE_ALLOWED_R) {
        OP_LOGI(context_->GetNodeName(), "InstanceNormReduceEmptyTiling not support the shape info.");
        return false;
    }

    OP_LOGI(context_->GetNodeName(), "InstanceNormReduceEmptyTiling IsCapable: IsCapable is true !");
    return true;
}

ge::graphStatus InstanceNormReduceEmptyTiling::DoOpTiling()
{
    // core num
    int64_t totalLength = a1 * a0;
    int64_t aivNum = aicoreParams_.blockDim;
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;

    int64_t elemSize = FP32_BYTE;
    int64_t perLoopMaxIndicesElements =
        (static_cast<int64_t>(aicoreParams_.ubSize)) / static_cast<int64_t>(elemSize) / BUFFER_NUM;

    if (meanDataType == ge::DT_FLOAT16 || meanDataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
        perLoopMaxIndicesElements = (static_cast<int64_t>(aicoreParams_.ubSize)) / static_cast<int64_t>(elemSize) /
                                    BUFFER_NUM;
    }
    // 核间切分计算
    blockNum_ = Ops::Base::CeilDiv(totalLength * elemSize, SINGLE_AIV_CORE_THREFOLD_BYTES);
    if (blockNum_ > aivNum) {
        blockNum_ = aivNum;
    }
    perCoreElements = Ops::Base::CeilDiv(totalLength, blockNum_);
    lastCoreElements = totalLength - (blockNum_ - 1) * perCoreElements;

    td_.perCoreElements = perCoreElements;
    td_.lastCoreElements = lastCoreElements;

    int64_t perCoreLoops = 1;
    int64_t perCorePerLoopElements = perCoreElements;
    int64_t perCoreLastLoopElements = perCoreElements;

    OP_LOGI(context_->GetNodeName(), "InstanceNormReduceEmptyTiling DoOpTiling: class member ubBlockSize is %lu !", ubBlockSize);
    // 核内切分计算（头核）
    perCorePerLoopElements = Ops::Base::FloorAlign(std::min(perLoopMaxIndicesElements, perCoreElements), ubBlockSize / elemSize);
    perCoreLoops = Ops::Base::CeilDiv(perCoreElements, perCorePerLoopElements);
    perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    td_.perCoreLoops = perCoreLoops;
    td_.perCorePerLoopElements = perCorePerLoopElements;
    td_.perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = 1;
    int64_t lastCorePerLoopElements = lastCoreElements;
    int64_t lastCoreLastLoopElements = lastCoreElements;

    // 尾核核内切分计算
    lastCorePerLoopElements = Ops::Base::FloorAlign(std::min(perLoopMaxIndicesElements, lastCoreElements), ubBlockSize / elemSize);
    lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, lastCorePerLoopElements);
    lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    td_.lastCoreLoops = lastCoreLoops;
    td_.lastCorePerLoopElements = lastCorePerLoopElements;
    td_.lastCoreLastLoopElements = lastCoreLastLoopElements;

    OP_LOGI(context_->GetNodeName(), "InstanceNormReduceEmptyTiling DoOpTiling: blockNum is %lu, perCoreElements is %lu,"
            " lastCoreElements is %lu, perCoreLoops is %lu, perCorePerLoopElements is %lu, perCoreLastLoopElements is %lu,"
            " lastCoreLoops is %lu, lastCorePerLoopElements is %lu, lastCoreLastLoopElements is %lu !",
            blockNum_, perCoreElements, lastCoreElements, perCoreLoops, perCorePerLoopElements,
            perCoreLastLoopElements, lastCoreLoops, lastCorePerLoopElements, lastCoreLastLoopElements);

    return ge::GRAPH_SUCCESS;
}

uint64_t InstanceNormReduceEmptyTiling::GetTilingKey() const
{
    return TILINGKEY_REDUCE_EMPTY;
}

ge::graphStatus InstanceNormReduceEmptyTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
        OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(td_),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&td_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(td_)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InstanceNorm, InstanceNormReduceEmptyTiling, IN_REDUCE_EMPTY_PRIORITY);
} // namespace optiling