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
 * \file adaptive_avg_pool2d_base_tiling.cpp
 * \brief
 */

#include "adaptive_avg_pool2d_base_tiling.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace std;

namespace optiling {


ge::graphStatus AdaptivePool2dBaseTiling::GetAndCheckDataFormat()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool2dBaseTiling::CheckNpuArch()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "GetShapeAttrsInfo begin 950, arch:%d.", npuArch);

    if (npuArch != NpuArch::DAV_3510) {
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool2dBaseTiling::CheckOutDims()
{
    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto outShape = outputShape->GetStorageShape();
    uint64_t nOutDim = 0;
    uint64_t cOutDim = 0;
    uint64_t hOutDim = 0;
    uint64_t wOutDim = 0;
    if (outShape.GetDimNum() == DIM_NUM_FOUR) {
        nOutDim = outShape.GetDim(DIM_N);
        cOutDim = outShape.GetDim(DIM_C);
        hOutDim = outShape.GetDim(DIM_H);
        wOutDim = outShape.GetDim(DIM_W);
    } else if (outShape.GetDimNum() == DIM_NUM_THREE) {
        nOutDim = 1;
        cOutDim = outShape.GetDim(DIM_C - 1);
        hOutDim = outShape.GetDim(DIM_H - 1);
        wOutDim = outShape.GetDim(DIM_W - 1);
    } else {
        OP_LOGE(nodeName, "The dim of outShape should be 3 or 4");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(nOutDim != input_.nIn || cOutDim != input_.cIn || hOutDim != input_.hOut || wOutDim != input_.wOut,
        OP_LOGE(nodeName, "Invalid shape. Maybe out tensor is error."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool2dBaseTiling::GetRealOutDims(const int64_t* outputSize, const gert::Shape& xShape, 
                                                        size_t output_size_len, size_t input_dim_num)
{
    std::vector<int> realOutDims = {};
    if (output_size_len  == OUTPUT_DIM_MAX) {
        for (size_t i = 0; i < OUTPUT_DIM_MAX; i++) {
            realOutDims.push_back(outputSize[i]);
        }
    } else if (output_size_len  == ONE_DIM) {
        for (size_t i = 0; i < OUTPUT_DIM_MAX; i++) {
            realOutDims.push_back(outputSize[0]);
        }
    } else {
        for (size_t i = 0; i < OUTPUT_DIM_MAX; i++) {
            realOutDims.push_back(xShape.GetDim(i + input_dim_num - OUTPUT_DIM_MAX));
        }
    }
    OP_CHECK_IF(realOutDims[0] <= 0 || realOutDims[1] <= 0,
        OP_LOGE(nodeName, "the value of outputsize should > 0"), return ge::GRAPH_FAILED);
    input_.hOut = realOutDims[0];
    input_.wOut = realOutDims[1];

    ge::graphStatus getCheckOutDimsResult = CheckOutDims();
    if (getCheckOutDimsResult != ge::GRAPH_SUCCESS) {
        return getCheckOutDimsResult;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool2dBaseTiling::GetShapeAttrsInfo()
{
    ge::graphStatus checkNpuArchResult = CheckNpuArch();
    if (checkNpuArchResult != ge::GRAPH_SUCCESS) {
        return checkNpuArchResult;
    }
    OP_LOGD(nodeName, "GetShapeAttrsInfo begin.");
    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputXDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto xDtype = inputXDesc->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
        OP_LOGE(nodeName, "The data type of x only supports float, float16, bfloat16"), return ge::GRAPH_FAILED);
    input_.xDtype = xDtype;
    gert::Shape xShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    if (xShape.GetDimNum() == DIM_NUM_FOUR) {
        input_.nIn = xShape.GetDim(DIM_N);
        input_.cIn = xShape.GetDim(DIM_C);
        input_.hIn = xShape.GetDim(DIM_H);
        input_.wIn = xShape.GetDim(DIM_W);
    } else if (xShape.GetDimNum() == DIM_NUM_THREE) {
        input_.nIn = 1;
        input_.cIn = xShape.GetDim(DIM_C - 1);
        input_.hIn = xShape.GetDim(DIM_H - 1);
        input_.wIn = xShape.GetDim(DIM_W - 1);
    } else {
        OP_LOGE(nodeName, "The dim of xShape should be 3 or 4");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(input_.nIn < 1 || input_.cIn < 1 || input_.hIn < 1 || input_.wIn < 1,
        OP_LOGE(nodeName, "Invalid shape. Maybe empty tensor."), return ge::GRAPH_FAILED);
    auto attrPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
    auto outputSizePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputSizePtr);
    size_t output_size_len = outputSizePtr->GetSize();
    size_t input_dim_num = xShape.GetDimNum();
    OP_CHECK_IF((output_size_len != OUTPUT_DIM_MAX && output_size_len != ONE_DIM && output_size_len != NONE_DIM), 
        OP_LOGE(nodeName, "the size of outputsize only support 0, 1, or 2"),
        return ge::GRAPH_FAILED);
    const int64_t* outputSize = static_cast<const int64_t*>(outputSizePtr->GetData());
    ge::graphStatus getRealOutDimsResult = GetRealOutDims(outputSize, xShape, output_size_len, input_dim_num);
    if (getRealOutDimsResult != ge::GRAPH_SUCCESS) {
        return getRealOutDimsResult;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool2dBaseTiling::GetPlatformInfo()
{
     auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const AdaptivePool2dCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        input_.coreNum = compileInfoPtr->coreNum;
        input_.ubSize = compileInfoPtr->ubSizePlatForm;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        input_.coreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        input_.ubSize = ubSize;
    }

    return ge::GRAPH_SUCCESS;
}

bool AdaptivePool2dBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus AdaptivePool2dBaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool2dBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool2dBaseTiling::GetWorkspaceSize()
{
    auto sys_workspace = SYS_WORKSPACE_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sys_workspace);
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptivePool2dBaseTiling::CalKernelSizeOneDimMax(uint64_t inSize, uint64_t outSize)
{
    // kernel sizemax
    uint64_t kernelSize = 1;
    outSize = outSize == 0 ? 1 : outSize;
    if (outSize > KERNEL_CALC_COUNT_THERSHOLD) {
        return (inSize + outSize - 1) / outSize + 1;
    }
    for (uint64_t i = 0; i < outSize; i++) {
        auto kernelLeft = (i * inSize) / outSize;
        auto kernelRight = ((i + 1) * inSize + outSize - 1) / outSize;
        auto kernelCurrent = kernelRight - kernelLeft;
        kernelSize = kernelCurrent > kernelSize ? kernelCurrent : kernelSize;
    }
    return kernelSize;
}
 
ge::graphStatus AdaptivePool2dBaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptivePool2dBaseTiling::GetTilingKey() const
{
    return 0;
}

void AdaptivePool2dBaseTiling::DumpTilingInfo()
{
    return;
}
}