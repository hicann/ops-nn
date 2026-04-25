/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_grad_tiling_simt.cpp.cpp
 * \brief
 */
#include <cctype>
#include <algorithm>
#include "error_util.h"
#include "max_pool_grad_tiling_simt.h"
#include "op_host/tiling_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../op_kernel/arch35/max_pool_grad_struct.h"

using namespace ge;
using namespace PoolGradNameSpace;
namespace optiling {
static constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;

bool MaxPoolGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus MaxPoolGradTiling::DoOpTiling()
{
    tilingData_->nDim = inputData.nX;
    tilingData_->cDim = inputData.cX;
    tilingData_->hInDim = inputData.hX;
    tilingData_->wInDim = inputData.wX;
    tilingData_->hOutDim = inputData.hGrad;
    tilingData_->wOutDim = inputData.wGrad;
    tilingData_->kSizeH = inputData.hKernel;
    tilingData_->kSizeW = inputData.wKernel;
    tilingData_->stridesH = inputData.hStride;
    tilingData_->stridesW = inputData.wStride;
    tilingData_->padH = inputData.hPad;
    tilingData_->padW = inputData.wPad;
    tilingData_->dilationH = inputData.hDilation;
    tilingData_->dilationW = inputData.wDilation;
    tilingData_->ceilMode = inputData.ceilMode;

    outputDataCount = tilingData_->nDim * tilingData_->cDim * tilingData_->hOutDim * tilingData_->wOutDim;
    int64_t threads = std::min(outputDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outputDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(hardwareData.coreNum));
    context_->SetBlockDim(blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPoolGradTiling::GetTilingKey() const
{
    uint32_t kernelMode = TPL_SIMT_KERNEL;
    uint32_t format = (inputData.inputFormat == ge::Format::FORMAT_NHWC) ? TPL_NHWC_FORMAT : TPL_NCHW_FORMAT;
    int64_t planeSize = inputData.hX * inputData.wX;
    uint32_t indicesDtype = (planeSize > static_cast<int64_t>(MAX_INT32)) ? TPL_INT64 : TPL_INT32;
    uint32_t isCheckRange = TPL_NO_CHECK_RANGE;
    return GET_TPL_TILING_KEY(kernelMode, format, indicesDtype, isCheckRange);
}

ge::graphStatus MaxPoolGradTiling::PostTiling()
{
    int64_t outDataCount = inputData.nX * inputData.cX * inputData.hX * inputData.wX;
    int64_t threads = std::min(outDataCount, MAX_THREAD_NUM);
    int64_t blockNum = Ops::Base::CeilDiv(outDataCount, threads);
    blockNum = std::min(blockNum, static_cast<int64_t>(hardwareData.coreNum));
    context_->SetBlockDim(blockNum);
    context_->SetLocalMemorySize(hardwareData.ubSize - DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradTiling::GetWorkspaceSize()
{
    auto gradShape = context_->GetInputShape(1); // grad shape = [n, c, hOut, wOut]
    OPS_CHECK_NULL_WITH_CONTEXT(context_, gradShape);
    auto gradShapeInfo = Ops::Base::EnsureNotScalar(gradShape->GetStorageShape());
    int64_t argmaxCount = gradShapeInfo.GetShapeSize(); // argmax数量 = n * c * hOut * wOut
    int64_t planeSize = inputData.hX * inputData.wX;
    size_t indicesSize = (planeSize > static_cast<int64_t>(MAX_INT32)) ? INT64_SIZE : INT32_SIZE;
    size_t argmaxSize = argmaxCount * indicesSize;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = argmaxSize + WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

void MaxPoolGradTiling::DumpTilingInfo()
{
    std::string str;
    str += " nDim:" + std::to_string(tilingData_->nDim);
    str += " cDim:" + std::to_string(tilingData_->cDim);
    str += " hInDim:" + std::to_string(tilingData_->hInDim);
    str += " wInDim:" + std::to_string(tilingData_->wInDim);
    str += " hOutDim:" + std::to_string(tilingData_->hOutDim);
    str += " wOutDim:" + std::to_string(tilingData_->wOutDim);
    str += " kSizeH:" + std::to_string(tilingData_->kSizeH);
    str += " kSizeW:" + std::to_string(tilingData_->kSizeW);
    str += " stridesH:" + std::to_string(tilingData_->stridesH);
    str += " stridesW:" + std::to_string(tilingData_->stridesW);
    str += " padH:" + std::to_string(tilingData_->padH);
    str += " padW:" + std::to_string(tilingData_->padW);
    str += " dilationH:" + std::to_string(tilingData_->dilationH);
    str += " dilationW:" + std::to_string(tilingData_->dilationW);
    str += " ceilMode:" + std::to_string(tilingData_->ceilMode);
    OP_LOGI(context_, "%s", str.c_str());
}
REGISTER_OPS_TILING_TEMPLATE(MaxPoolGrad, MaxPoolGradTiling, 100);
} // namespace optiling