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
 * \file rms_norm_dynamic_mx_quant_reduce_empty_tiling_arch35.cpp
 * \brief RmsNormDynamicMxQuant ReduceEmpty tiling implementation
 */
#include "rms_norm_dynamic_mx_quant_tiling_arch35.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

constexpr static int64_t BUFFER_NUM = 1;
constexpr static int64_t SINGLE_AIV_CORE_THRESHOLD_BYTES = 32L * 1024L;
constexpr static uint64_t DEFAULT_NUM = 0;

bool RmsNormDynamicMxQuantReduceEmptyTiling::IsCapable()
{
    if (numN_ != 0 && numM_ != 0) {
        OP_LOGI(context_->GetNodeName(), "ReduceEmptyTiling not applicable: numN=%ld != 0 and numM=%ld != 0.", numN_, numM_);
        return false;
    }
    OP_LOGI(context_->GetNodeName(), "ReduceEmptyTiling IsCapable: true (numN=%ld, numM=%ld).", numN_, numM_);
    return true;
}

ge::graphStatus RmsNormDynamicMxQuantReduceEmptyTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter ReduceEmpty DoOpTiling.");

    td_.set_hasOutputRstd(static_cast<uint64_t>(hasOutputRstd_));
    td_.set_numM(static_cast<uint64_t>(numM_));

    if (hasOutputRstd_ == 0 || numM_ == 0) {
        usedCoreNum_ = 1;
        td_.set_perCoreElements(DEFAULT_NUM);
        td_.set_lastCoreElements(DEFAULT_NUM);
        td_.set_perCoreLoops(DEFAULT_NUM);
        td_.set_perCorePerLoopElements(DEFAULT_NUM);
        td_.set_perCoreLastLoopElements(DEFAULT_NUM);
        td_.set_lastCoreLoops(DEFAULT_NUM);
        td_.set_lastCorePerLoopElements(DEFAULT_NUM);
        td_.set_lastCoreLastLoopElements(DEFAULT_NUM);
        return ge::GRAPH_SUCCESS;
    }

    // rstd output: numM elements of FP32
    int64_t totalLength = numM_;
    int64_t elemSize = FP32_BYTES;
    int64_t aivNum = totalCoreNum_;
    int64_t perLoopMaxElements = static_cast<int64_t>(ubSize_) / elemSize / BUFFER_NUM;

    // Core splitting
    int64_t blockNum = Ops::Base::CeilDiv(totalLength * elemSize, SINGLE_AIV_CORE_THRESHOLD_BYTES);
    if (blockNum > aivNum) {
        blockNum = aivNum;
    }

    usedCoreNum_ = static_cast<uint64_t>(blockNum);

    int64_t perCoreElements = Ops::Base::CeilDiv(totalLength, blockNum);
    int64_t lastCoreElements = totalLength - (blockNum - 1) * perCoreElements;

    td_.set_perCoreElements(static_cast<uint64_t>(perCoreElements));
    td_.set_lastCoreElements(static_cast<uint64_t>(lastCoreElements));

    // Intra-core loop splitting (non-last core)
    int64_t ubAlignElems = ubBlockSize_ / elemSize;
    int64_t perCorePerLoopElements = Ops::Base::FloorAlign(
        std::min(perLoopMaxElements, perCoreElements), ubAlignElems);

    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, perCorePerLoopElements);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    td_.set_perCoreLoops(static_cast<uint64_t>(perCoreLoops));
    td_.set_perCorePerLoopElements(static_cast<uint64_t>(perCorePerLoopElements));
    td_.set_perCoreLastLoopElements(static_cast<uint64_t>(perCoreLastLoopElements));

    // Intra-core loop splitting (last core)
    int64_t lastCorePerLoopElements = Ops::Base::FloorAlign(
        std::min(perLoopMaxElements, lastCoreElements), ubAlignElems);
    if (lastCorePerLoopElements < 1) {
        lastCorePerLoopElements = lastCoreElements;
    }
    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, lastCorePerLoopElements);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    td_.set_lastCoreLoops(static_cast<uint64_t>(lastCoreLoops));
    td_.set_lastCorePerLoopElements(static_cast<uint64_t>(lastCorePerLoopElements));
    td_.set_lastCoreLastLoopElements(static_cast<uint64_t>(lastCoreLastLoopElements));

    OP_LOGI(context_->GetNodeName(),
            "ReduceEmpty DoOpTiling: blockNum=%lu, perCoreElements=%lu, lastCoreElements=%lu, "
            "perCoreLoops=%lu, perCorePerLoopElements=%lu, perCoreLastLoopElements=%lu, "
            "lastCoreLoops=%lu, lastCorePerLoopElements=%lu, lastCoreLastLoopElements=%lu, hasOutputRstd=%lu.",
            usedCoreNum_, td_.get_perCoreElements(), td_.get_lastCoreElements(),
            td_.get_perCoreLoops(), td_.get_perCorePerLoopElements(), td_.get_perCoreLastLoopElements(),
            td_.get_lastCoreLoops(), td_.get_lastCorePerLoopElements(), td_.get_lastCoreLastLoopElements(),
            td_.get_hasOutputRstd());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RmsNormDynamicMxQuantReduceEmptyTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t RmsNormDynamicMxQuantReduceEmptyTiling::GetTilingKey() const
{
    return TILINGKEY_REDUCE_EMPTY;
}

ge::graphStatus RmsNormDynamicMxQuantReduceEmptyTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "ReduceEmpty PostTiling: usedCoreNum=%lu.", usedCoreNum_);
    context_->SetBlockDim(usedCoreNum_);

    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(
    RmsNormDynamicMxQuant, RmsNormDynamicMxQuantReduceEmptyTiling, TEMPLATE_REDUCE_EMPTY_PRIORITY);

} // namespace optiling
