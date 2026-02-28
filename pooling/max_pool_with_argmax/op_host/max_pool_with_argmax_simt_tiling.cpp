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
 * \file max_pool_with_argmax_simt_tiling.cpp
 * \brief
 */
#include <cctype>
#include <algorithm>
#include "op_host/tiling_templates_registry.h"
#include "max_pool_with_argmax_simt_tiling.h"
#include "../op_kernel/arch35/max_pool_with_argmax_struct_common.h"

using namespace AscendC;
using namespace ge;
namespace optiling {
static constexpr int64_t NAN_BASE = 100;

ge::graphStatus MaxPoolWithArgmaxTilingSIMT::DoOpTiling()
{
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData* tilingData =
        context_->GetTilingData<MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData>();
    tilingData->nDim = inputData.nInput;
    tilingData->cDim = inputData.cInput;
    tilingData->hInDim = inputData.inputShape[H_IDX_];
    tilingData->wInDim = inputData.inputShape[W_IDX_];
    tilingData->hOutDim = inputData.outShape[H_IDX_];
    tilingData->wOutDim = inputData.outShape[W_IDX_];
    tilingData->kSizeH = inputData.kernelSize[H_IDX_];
    tilingData->kSizeW = inputData.kernelSize[W_IDX_];
    tilingData->stridesH = inputData.stride[H_IDX_];
    tilingData->stridesW = inputData.stride[W_IDX_];
    tilingData->padH = inputData.pad[H_IDX_];
    tilingData->padW = inputData.pad[W_IDX_];
    tilingData->includeBatchInIndex = inputData.includeBatchInIndex;

    outputDataCount = tilingData->nDim * tilingData->cDim * tilingData->hOutDim * tilingData->wOutDim;
    int64_t threads = std::min(outputDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outputDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(coreNum));
    tilingData->threadNums = threads;
    tilingData->blockNums = blockNum;

    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPoolWithArgmaxTilingSIMT::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (inputData.inputFormat == ge::Format::FORMAT_NHWC && outputDataCount <= MAX_INT32) {
        tilingKey = SIMT_NHWC_TILING_KEY_INT32; // 500002
    } else if (inputData.inputFormat == ge::Format::FORMAT_NCHW && outputDataCount <= MAX_INT32) {
        tilingKey = SIMT_NCHW_TILING_KEY_INT32; // 500001
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC && outputDataCount > MAX_INT32) {
        tilingKey = SIMT_NHWC_TILING_KEY_INT64; // 500012
    } else if (inputData.inputFormat == ge::Format::FORMAT_NCHW && outputDataCount > MAX_INT32) {
        tilingKey = SIMT_NCHW_TILING_KEY_INT64; // 500011
    }

    return tilingKey + NAN_BASE * static_cast<uint64_t>(inputData.nanProp);
}

ge::graphStatus MaxPoolWithArgmaxTilingSIMT::PostTiling()
{
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData* tilingData =
        context_->GetTilingData<MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData>();
    context_->SetBlockDim(tilingData->blockNums);
    return ge::GRAPH_SUCCESS;
}

void MaxPoolWithArgmaxTilingSIMT::DumpTilingInfo()
{
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData* tilingData =
        context_->GetTilingData<MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData>();
    std::ostringstream info;
    info << "nDim: " << tilingData->nDim;
    info << ", hInDim: " << tilingData->hInDim;
    info << ", wInDim: " << tilingData->wInDim;
    info << ", cDim: " << tilingData->cDim;
    info << ", hOutDim: " << tilingData->hOutDim;
    info << ", wOutDim: " << tilingData->wOutDim;
    info << ", kSizeH: " << tilingData->kSizeH;
    info << ", kSizeW: " << tilingData->kSizeW;
    info << ", stridesH: " << tilingData->stridesH;
    info << ", stridesW: " << tilingData->stridesW;
    info << ", padH: " << tilingData->padH;
    info << ", padW: " << tilingData->padH;
    info << ", includeBatchInIndex: " << tilingData->includeBatchInIndex;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}


REGISTER_TILING_TEMPLATE("MaxPoolWithArgmax", MaxPoolWithArgmaxTilingSIMT, 100);
} // namespace optiling