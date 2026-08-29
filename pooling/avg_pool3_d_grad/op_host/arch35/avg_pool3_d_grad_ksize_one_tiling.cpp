/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_ksize_one_tiling.cpp
 * \brief KsizeOne scheme tiling for 3D average pooling backward (arch35).
 *        When kD=kH=kW=1 and sD=sH=sW=1: grads and output are identical shape,
 *        backward is simply: read grad, Div by divisor, write to output.
 *        No SyncAll, no zero-fill, no stride scatter. Flat 1D tiling.
 */

#include <algorithm>

#include "op_host/tiling_templates_registry.h"
#include "avg_pool3_d_grad_ksize_one_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling {
using namespace AvgPool3DGrad;

static constexpr int64_t UB_RESERVED_BYTES = 1024;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t INPUT_OUTPUT_COEXIST = 2;
static constexpr int64_t MIN_BYTES_PER_CORE = 2048;

bool AvgPool3DGradKsizeOneTiling::IsCapable()
{
    if (inputData.kernelSize[D_DIM] != 1 || inputData.kernelSize[H_DIM] != 1 || inputData.kernelSize[W_DIM] != 1) {
        return false;
    }
    if (inputData.stride[D_DIM] != 1 || inputData.stride[H_DIM] != 1 || inputData.stride[W_DIM] != 1) {
        return false;
    }
    return true;
}

void AvgPool3DGradKsizeOneTiling::CalcDivisor()
{
    divisor_ = (inputData.divisorOverride == 0L) ? 1 : inputData.divisorOverride;
}

uint64_t AvgPool3DGradKsizeOneTiling::GetTilingKey() const
{
    uint32_t hasDivisor = (divisor_ != 1) ? TPL_HAS_DIVISOR : TPL_NO_DIVISOR;
    return GET_TPL_TILING_KEY(TPL_KSIZE_ONE_KERNEL, TPL_NCDHW_FORMAT, TPL_INT64, TPL_NO_PAD, TPL_NO_CHECK_RANGE,
                              TPL_NO_COUNT_PAD, hasDivisor);
}

ge::graphStatus AvgPool3DGradKsizeOneTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

void AvgPool3DGradKsizeOneTiling::DoBlockTiling()
{
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context_) / inputData.dtypeSize;
    int64_t availableUbElements = static_cast<int64_t>(ubSize - UB_RESERVED_BYTES) / inputData.dtypeSize;
    availableUbElements = availableUbElements / DOUBLE_BUFFER / INPUT_OUTPUT_COEXIST;
    availableUbElements -= ubBlockSize;
    totalElements_ = inputData.batches * inputData.outShape[D_DIM] * inputData.outShape[H_DIM] *
                     inputData.outShape[W_DIM] * inputData.channels;

    int64_t minElementsPerCore = MIN_BYTES_PER_CORE / inputData.dtypeSize;
    if (minElementsPerCore < 1) {
        minElementsPerCore = 1;
    }
    usedCoreNum_ = std::min(static_cast<int64_t>(coreNum), std::max(int64_t(1), totalElements_ / minElementsPerCore));
    elementsPerCore_ = totalElements_ / usedCoreNum_;
    tailCoreElements_ = totalElements_ - elementsPerCore_ * usedCoreNum_;
    int64_t ubElements = std::min(elementsPerCore_ + 1, availableUbElements);
    ubBufferSize_ = Ops::Base::CeilAlign(std::max(ubElements, minElementsPerCore), ubBlockSize);
}

ge::graphStatus AvgPool3DGradKsizeOneTiling::SetTilingData()
{
    AvgPool3DGradKsizeOneTilingData* tilingData = context_->GetTilingData<AvgPool3DGradKsizeOneTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    tilingData->ubBufferSize = ubBufferSize_;
    tilingData->elementsPerCore = elementsPerCore_;
    tilingData->tailCoreElements = tailCoreElements_;
    tilingData->totalElements = totalElements_;
    tilingData->divisor = divisor_;
    tilingData->usedCoreNum = usedCoreNum_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3DGradKsizeOneTiling::DoOpTiling()
{
    DoBlockTiling();
    CalcDivisor();
    return SetTilingData();
}

ge::graphStatus AvgPool3DGradKsizeOneTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

void AvgPool3DGradKsizeOneTiling::DumpTilingInfo()
{
    OP_LOGI(context_,
            "KsizeOne tiling: ubBufferSize=%ld elementsPerCore=%ld tailCoreElements=%ld "
            "totalElements=%ld divisor=%ld usedCoreNum=%ld",
            ubBufferSize_, elementsPerCore_, tailCoreElements_, totalElements_, divisor_, usedCoreNum_);
}

REGISTER_OPS_TILING_TEMPLATE(AvgPool3DGrad, AvgPool3DGradKsizeOneTiling, 0);

} // namespace optiling
