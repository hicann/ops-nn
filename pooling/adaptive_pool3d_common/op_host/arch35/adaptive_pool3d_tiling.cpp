/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file adaptive_pool3d_tiling_base.cpp
 * \brief
 */

#include "adaptive_pool3d_tiling.h"

using Ops::NN::Optiling::TilingRegistry;
using namespace std;

namespace optiling {

ge::graphStatus AdaptivePool3dBaseTiling::GetAndCheckIndicesDtype()
{
    auto outputIndicesDesc = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputIndicesDesc);
    auto indicesDtype = outputIndicesDesc->GetDataType();
    auto opNodeName = context_->GetNodeName() == nullptr ? "AdaptivePool3d" : context_->GetNodeName();
    if (indicesDtype != ge::DT_INT32 && indicesDtype != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(
            opNodeName, "indices", ge::TypeUtils::DataTypeToSerialString(indicesDtype).c_str(),
            "[DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }
    if (input_.dIn * input_.hIn * input_.wIn > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) &&
        indicesDtype != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opNodeName, "indices", ge::TypeUtils::DataTypeToSerialString(indicesDtype).c_str(),
            "when D*H*W exceeds INT32_MAX, dtype must be DT_INT64");
        return ge::GRAPH_FAILED;
    }
    input_.indicesDtype = indicesDtype;

    int64_t attrDtype = DTYPE_INT32;
    auto attrPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
    const int* attrDtypePtr = attrPtr->GetAttrPointer<int>(1);
    if (attrDtypePtr != nullptr) {
        attrDtype = *attrDtypePtr;
    }
    if (attrDtype != DTYPE_INT32 && attrDtype != DTYPE_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(
            opNodeName, "indices_attr", std::to_string(attrDtype).c_str(), "[DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool3dBaseTiling::GetAndCheckDataFormat()
{
    auto curAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, curAttrs);
    std::string inputFormatStr("NDHWC");
    const char* inputFormat = curAttrs->GetAttrPointer<char>(1);
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }
    if (inputFormatStr == "NCDHW") {
        input_.dataFormat = ge::Format::FORMAT_NCDHW;
    } else if (inputFormatStr == "NDHWC") {
        input_.dataFormat = ge::Format::FORMAT_NDHWC;
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_, "AdaptivePool3d only support NCDHW and NDHWC, not support format %s", inputFormatStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptivePool3dBaseTiling::CalKernelSizeOneDimMax(uint64_t inSize, uint64_t outSize)
{
    // 计算kernel size的max值
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

ge::graphStatus AdaptivePool3dBaseTiling::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    bool checkErrFlag = false;
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    auto nodeName = context_->GetNodeName() == nullptr ? "AdaptivePool3d" : context_->GetNodeName();
    OP_LOGD(nodeName, "GetShapeAttrsInfo begin 950, arch:%d.", npuArch);

    if (npuArch != NpuArch::DAV_3510) {
        return ge::GRAPH_PARAM_INVALID;
    }
    OP_LOGD(nodeName, "GetShapeAttrsInfo begin.");

    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputXDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXDesc);
    auto xDtype = inputXDesc->GetDataType();
    checkErrFlag = xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16;
    if (checkErrFlag) {
        OP_LOGE_FOR_INVALID_DTYPE(
            nodeName, "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(), "[DT_FLOAT, DT_FLOAT16, DT_BF16]");
        return ge::GRAPH_FAILED;
    }
    input_.xDtype = xDtype;

    gert::Shape xShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    if (xShape.GetDimNum() == DIM_NUM_FIVE) {
        input_.nIn = xShape.GetDim(DIM_N);
        input_.cIn = xShape.GetDim(DIM_C);
        input_.dIn = xShape.GetDim(DIM_D);
        input_.hIn = xShape.GetDim(DIM_H);
        input_.wIn = xShape.GetDim(DIM_W);
    } else if (xShape.GetDimNum() == DIM_NUM_FOUR) {
        input_.nIn = 1;
        input_.cIn = xShape.GetDim(DIM_C - 1);
        input_.dIn = xShape.GetDim(DIM_D - 1);
        input_.hIn = xShape.GetDim(DIM_H - 1);
        input_.wIn = xShape.GetDim(DIM_W - 1);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x", std::to_string(xShape.GetDimNum()).c_str(), "4 or 5");
        return ge::GRAPH_FAILED;
    }
    checkErrFlag = input_.nIn < 1 || input_.cIn < 1 || input_.dIn < 1 || input_.hIn < 1 || input_.wIn < 1;
    if (checkErrFlag) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x", "empty tensor", "each dimension must be >= 1");
        return ge::GRAPH_FAILED;
    }

    auto attrPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
    auto outputSizePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputSizePtr);
    size_t output_size_len = outputSizePtr->GetSize();
    size_t input_dim_num = xShape.GetDimNum();
    checkErrFlag = output_size_len != OUTPUT_DIM_MAX && output_size_len != ONE_DIM && output_size_len != NONE_DIM;
    if (checkErrFlag) {
        OP_LOGE_FOR_INVALID_LISTSIZE(nodeName, "output_size", std::to_string(output_size_len).c_str(), "0, 1, or 3");
        return ge::GRAPH_FAILED;
    }
    const int64_t* outputSize = static_cast<const int64_t*>(outputSizePtr->GetData());
    std::vector<int> realOutDims = {};
    if (output_size_len == OUTPUT_DIM_MAX) {
        for (size_t i = 0; i < OUTPUT_DIM_MAX; i++) {
            realOutDims.push_back(outputSize[i]);
        }
    } else if (output_size_len == ONE_DIM) {
        for (size_t i = 0; i < OUTPUT_DIM_MAX; i++) {
            realOutDims.push_back(outputSize[0]);
        }
    } else {
        for (size_t i = 0; i < OUTPUT_DIM_MAX; i++) {
            realOutDims.push_back(xShape.GetDim(i + input_dim_num - OUTPUT_DIM_MAX));
        }
    }
    checkErrFlag = realOutDims[0] <= 0 || realOutDims[1] <= 0 || realOutDims[OUTPUTSIZE_DIMW] <= 0;
    if (checkErrFlag) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            nodeName, "outputSize[0], outputSize[1], outputSize[2]",
            (std::to_string(outputSize[0]) + ", " + std::to_string(outputSize[1]) + ", " +
             std::to_string(outputSize[OUTPUTSIZE_DIMW]))
                .c_str(),
            "each dimension must be > 0");
        return ge::GRAPH_FAILED;
    }
    input_.dOut = realOutDims[0];
    input_.hOut = realOutDims[1];
    input_.wOut = realOutDims[OUTPUTSIZE_DIMW];

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool3dBaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const AdaptivePool3dCompileInfo*>(context_->GetCompileInfo());
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

bool AdaptivePool3dBaseTiling::IsCapable() { return true; }

ge::graphStatus AdaptivePool3dBaseTiling::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaptivePool3dBaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaptivePool3dBaseTiling::GetWorkspaceSize()
{
    auto sys_workspace = SYS_WORKSPACE_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sys_workspace);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptivePool3dBaseTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t AdaptivePool3dBaseTiling::GetTilingKey() const { return 0; }

void AdaptivePool3dBaseTiling::DumpTilingInfo() { return; }
} // namespace optiling