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
 * \file max_pool_with_argmax_v3_tiling_base.cpp
 * \brief
 */

#include "tiling_base/tiling_templates_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "max_pool_with_argmax_v3_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

const int INPUT_IDX_X = 0;
const int NCHW_DIMS = 4;
const int CHW_DIMS = 3;
const int KERNEL_POS = 0;
const int STRIDE_POS = 1;
const int PADDING_POS = 2;
const int DTYPE_POS = 3;
const int DILATION_POS = 4;
const int CEIL_POS = 5;
const int FORMAT_POS = 6;
const int WS_SYS_SIZE = 16 * 1024 * 1024;
static const int MP_MAX_2D_DIM_ZERO = 0;
static const int MP_MAX_2D_DIM_ONE = 1;
static const int MP_MAX_2D_DIM_TWO = 2;
static const int MP_MAX_2D_DIM_THREE = 3;
static const int64_t MP_MAX_2D_TYPE_INT32 = 3;
static const int64_t MP_MAX_2D_TYPE_INT64 = 9;

ge::graphStatus MaxPoolWithArgmaxV3BaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MaxPoolWithArgmaxV3CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context_, "compile info is null"),
            return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;

        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNum();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    }
    OP_CHECK_IF(coreNum == 0, CUBE_INNER_ERR_REPORT(context_, "coreNum is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxV3BaseTiling::GetShapeAttrsInfo()
{
    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());

    OP_CHECK_IF(
        !(inputShape.GetDimNum() == NCHW_DIMS || inputShape.GetDimNum() == CHW_DIMS),
        OP_LOGE(
            context_->GetNodeName(), "MaxPoolWithArgmaxV3: input shape dim = %zu, should be equal 3 or 4",
            inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputShape.GetShapeSize() <= 0,
        OP_LOGE(
            context_->GetNodeName(), "MaxPoolWithArgmaxV3: input shape size %ld less than zero failed",
            inputShape.GetShapeSize()),
        return ge::GRAPH_FAILED);
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(), "MaxPoolWithArgmaxV3: invalid dtype");
        return ge::GRAPH_FAILED;
    }

    auto outX = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = Ops::Base::EnsureNotScalar(outX->GetStorageShape());
    auto indicesX = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesX);
    auto indicesShape = Ops::Base::EnsureNotScalar(indicesX->GetStorageShape());
    if (indicesShape != outShape) {
        OP_LOGE(context_->GetNodeName(), "MaxPoolWithArgmaxV3: indices shape and values shape is different");
        return ge::GRAPH_FAILED;
    }
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    std::string inputFormatStr("NCHW");
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }
    if (inputShape.GetDimNum() == CHW_DIMS) {
        if (inputFormatStr == "NCHW") {
            inputFormatStr = "CHW";
        } else {
            inputFormatStr = "HWC";
        }
    }
    int h_dim = MP_MAX_2D_DIM_TWO, w_dim = MP_MAX_2D_DIM_THREE;
    if (inputFormatStr == "NCHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCHW;
        inputData.batches = inputShape.GetDim(MP_MAX_2D_DIM_ZERO) * inputShape.GetDim(MP_MAX_2D_DIM_ONE);
        inputData.nInput = inputShape.GetDim(MP_MAX_2D_DIM_ZERO);
        inputData.cInput = inputShape.GetDim(MP_MAX_2D_DIM_ONE);
    } else if (inputFormatStr == "NHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NHWC;
        h_dim = MP_MAX_2D_DIM_ONE;
        w_dim = MP_MAX_2D_DIM_TWO;
        inputData.batches = inputShape.GetDim(MP_MAX_2D_DIM_ZERO) * inputShape.GetDim(MP_MAX_2D_DIM_THREE);
        inputData.nInput = inputShape.GetDim(MP_MAX_2D_DIM_ZERO);
        inputData.cInput = inputShape.GetDim(MP_MAX_2D_DIM_THREE);
    } else if (inputFormatStr == "CHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCHW;
        h_dim -= 1;
        w_dim -= 1;
        inputData.batches = 1 * inputShape.GetDim(MP_MAX_2D_DIM_ZERO);
        inputData.nInput = 1;
        inputData.cInput = inputShape.GetDim(MP_MAX_2D_DIM_ZERO);
    } else if (inputFormatStr == "HWC") {
        inputData.inputFormat = ge::Format::FORMAT_NHWC;
        h_dim = MP_MAX_2D_DIM_ONE - 1;
        w_dim = MP_MAX_2D_DIM_TWO - 1;
        inputData.batches = 1 * inputShape.GetDim(MP_MAX_2D_DIM_TWO);
        inputData.nInput = 1;
        inputData.cInput = inputShape.GetDim(MP_MAX_2D_DIM_TWO);
    } else {
        OP_LOGE(context_->GetNodeName(), "MaxPoolWithArgmaxV3: not support format %s", inputFormatStr.c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        outShape.GetDim(h_dim) < 1 || outShape.GetDim(w_dim) < 1,
        OP_LOGE(
            context_->GetNodeName(), "MaxPoolWithArgmaxV3: output shape [%ld, %ld] not support", outShape.GetDim(h_dim),
            outShape.GetDim(w_dim)),
        return ge::GRAPH_FAILED);

    inputData.inputShape =
        array<uint64_t, HW_DIMS>{uint64_t(inputShape.GetDim(h_dim)), uint64_t(inputShape.GetDim(w_dim))};
    inputData.outShape = array<uint64_t, HW_DIMS>{uint64_t(outShape.GetDim(h_dim)), uint64_t(outShape.GetDim(w_dim))};

    int32_t hValue = 0;
    int32_t wValue = 0;
    const gert::TypedContinuousVector<int64_t>* kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    hValue = *(kernelSize->GetData());
    wValue = *(kernelSize->GetData() + 1);
    inputData.kernelSize = array<uint64_t, HW_DIMS>{uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        hValue <= 0 || wValue <= 0,
        OP_LOGE(context_->GetNodeName(), "MaxPoolWithArgmaxV3: not support kernel shape [%d, %d]", hValue, wValue),
        return ge::GRAPH_FAILED);

    int32_t khValue = hValue;
    int32_t kwValue = wValue;
    const gert::TypedContinuousVector<int64_t>* stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stride);
    hValue = *(stride->GetData());
    wValue = *(stride->GetData() + 1);
    inputData.stride = array<uint64_t, HW_DIMS>{uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        hValue <= 0 || wValue <= 0,
        OP_LOGE(context_->GetNodeName(), "MaxPoolWithArgmaxV3: not support stride shape [%d, %d]", hValue, wValue),
        return ge::GRAPH_FAILED);

    const gert::TypedContinuousVector<int64_t>* padding = runtimeAttrs->GetListInt(PADDING_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padding);
    hValue = *(padding->GetData());
    wValue = *(padding->GetData() + 1);
    inputData.pad = array<uint64_t, HW_DIMS>{uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        hValue > khValue / 2 || wValue > kwValue / 2,
        OP_LOGE(
            context_->GetNodeName(), "MaxPoolWithArgmaxV3: not support pad shape [%d, %d] kernel shape [%d, %d]",
            hValue, wValue, khValue, kwValue),
        return ge::GRAPH_FAILED);

    inputData.dilation = array<uint64_t, HW_DIMS>{1, 1};
    hValue = 1;
    wValue = 1;
    const gert::TypedContinuousVector<int64_t>* dilation = runtimeAttrs->GetListInt(DILATION_POS);
    if (dilation != nullptr) {
        hValue = *(dilation->GetData());
        wValue = *(dilation->GetData() + 1);
        inputData.dilation = array<uint64_t, HW_DIMS>{uint64_t(hValue), uint64_t(wValue)};
        OP_CHECK_IF(
            hValue <= 0 || wValue <= 0,
            OP_LOGE(
                context_->GetNodeName(), "MaxPoolWithArgmaxV3: not support dilation shape [%d, %d]", hValue, wValue),
            return ge::GRAPH_FAILED);
    }

    inputData.ceilMode = false;
    const bool* ceilModePtr = runtimeAttrs->GetAttrPointer<bool>(CEIL_POS);
    if (ceilModePtr != nullptr) {
        inputData.ceilMode = *ceilModePtr;
    }

    int indexDtype = 3;
    const int* indexDtypePtr = runtimeAttrs->GetAttrPointer<int>(DTYPE_POS);
    if (indexDtypePtr != nullptr) {
        indexDtype = *indexDtypePtr;
    }
    switch (indexDtype) {
        case MP_MAX_2D_TYPE_INT32:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
        case MP_MAX_2D_TYPE_INT64:
            inputData.indexDtype = ge::DataType::DT_INT64;
            break;
        default:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
    }
    return ge::GRAPH_SUCCESS;
}

bool MaxPoolWithArgmaxV3BaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus MaxPoolWithArgmaxV3BaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxV3BaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPoolWithArgmaxV3BaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus MaxPoolWithArgmaxV3BaseTiling::GetWorkspaceSize()
{
    auto sys_workspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxV3BaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling