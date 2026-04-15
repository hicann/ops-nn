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
 * \file add_rms_norm_dynamic_mx_quant_tiling_reduce_empty_arch35.cpp
 * \brief
 */
#include "add_rms_norm_dynamic_mx_quant_tiling.h"

namespace optiling {

constexpr uint64_t TILING_KEY_REDUCE_EMPTY = 300;
constexpr static int64_t BUFFER_NUM = 1;
constexpr static int64_t SINGLE_AIV_CORE_THRESHOLD_BYTES = 32L * 1024L;
constexpr static uint64_t DEFAULT_NUM = 0;

bool AddRmsNormDynamicMxQuantReduceEmptyTiling::IsCapable()
{
    if (numCol_ != 0 && numRow_ != 0) {
        OP_LOGI(context_->GetNodeName(), "ReduceEmptyTiling not applicable: numCol=%lu != 0 or numRow=%lu != 0.", numCol_, numRow_);
        return false;
    }
    OP_LOGI(context_->GetNodeName(), "ReduceEmptyTiling IsCapable: true (numCol=%lu, numRow=%lu).", numCol_, numRow_);
    return true;
}

ge::graphStatus AddRmsNormDynamicMxQuantReduceEmptyTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter ReduceEmpty DoOpTiling.");

    td_.rstdFlag = rstdFlag_;
    td_.numRow = numRow_;

    if (rstdFlag_ == 0 || numRow_ == 0) {
        usedCoreNum_ = 1;
        td_.perCoreElements = DEFAULT_NUM;
        td_.lastCoreElements = DEFAULT_NUM;
        td_.perCoreLoops = DEFAULT_NUM;
        td_.perCorePerLoopElements = DEFAULT_NUM;
        td_.perCoreLastLoopElements = DEFAULT_NUM;
        td_.lastCoreLoops = DEFAULT_NUM;
        td_.lastCorePerLoopElements = DEFAULT_NUM;
        td_.lastCoreLastLoopElements = DEFAULT_NUM;
        return ge::GRAPH_SUCCESS;
    }

    // rstd output: numRow elements of FP32
    int64_t totalLength = static_cast<int64_t>(numRow_);
    int64_t elemSize = FP32_SIZE;
    int64_t aivNum = static_cast<int64_t>(totalCoreNum_);
    int64_t perLoopMaxElements =
        static_cast<int64_t>(maxUbSize_) / elemSize / BUFFER_NUM;

    // Core splitting
    int64_t blockNum = Ops::Base::CeilDiv(totalLength * elemSize, SINGLE_AIV_CORE_THRESHOLD_BYTES);
    if (blockNum > aivNum) {
        blockNum = aivNum;
    }

    usedCoreNum_ = static_cast<uint64_t>(blockNum);

    int64_t perCoreElements = Ops::Base::CeilDiv(totalLength, blockNum);
    int64_t lastCoreElements = totalLength - (blockNum - 1) * perCoreElements;

    td_.perCoreElements = static_cast<uint64_t>(perCoreElements);
    td_.lastCoreElements = static_cast<uint64_t>(lastCoreElements);

    // Intra-core loop splitting (non-last core)
    int64_t ubAlignElems = static_cast<int64_t>(ubBlockSize_) / elemSize;
    int64_t perCorePerLoopElements = Ops::Base::FloorAlign(
        std::min(perLoopMaxElements, perCoreElements), ubAlignElems);

    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, perCorePerLoopElements);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    td_.perCoreLoops = static_cast<uint64_t>(perCoreLoops);
    td_.perCorePerLoopElements = static_cast<uint64_t>(perCorePerLoopElements);
    td_.perCoreLastLoopElements = static_cast<uint64_t>(perCoreLastLoopElements);

    // Intra-core loop splitting (last core)
    int64_t lastCorePerLoopElements = Ops::Base::FloorAlign(
        std::min(perLoopMaxElements, lastCoreElements), ubAlignElems);
    if (lastCorePerLoopElements < 1) {
        lastCorePerLoopElements = lastCoreElements;
    }
    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, lastCorePerLoopElements);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    td_.lastCoreLoops = static_cast<uint64_t>(lastCoreLoops);
    td_.lastCorePerLoopElements = static_cast<uint64_t>(lastCorePerLoopElements);
    td_.lastCoreLastLoopElements = static_cast<uint64_t>(lastCoreLastLoopElements);

    OP_LOGI(context_->GetNodeName(),
            "ReduceEmpty DoOpTiling: blockNum=%lu, perCoreElements=%lu, lastCoreElements=%lu, "
            "perCoreLoops=%lu, perCorePerLoopElements=%lu, perCoreLastLoopElements=%lu, "
            "lastCoreLoops=%lu, lastCorePerLoopElements=%lu, lastCoreLastLoopElements=%lu, rstdFlag=%u.",
            usedCoreNum_, td_.perCoreElements, td_.lastCoreElements,
            td_.perCoreLoops, td_.perCorePerLoopElements, td_.perCoreLastLoopElements,
            td_.lastCoreLoops, td_.lastCorePerLoopElements, td_.lastCoreLastLoopElements, td_.rstdFlag);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicMxQuantReduceEmptyTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormDynamicMxQuantReduceEmptyTiling::GetTilingKey() const
{
    return TILING_KEY_REDUCE_EMPTY;
}

ge::graphStatus AddRmsNormDynamicMxQuantReduceEmptyTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "ReduceEmpty PostTiling: usedCoreNum=%lu.", usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(td_) > rawTilingData->GetCapacity(),
        OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                sizeof(td_), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&td_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(td_)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AddRmsNormDynamicMxQuant, AddRmsNormDynamicMxQuantReduceEmptyTiling, ARND_REDUCE_EMPTY_PRIORITY);
} // namespace optiling
