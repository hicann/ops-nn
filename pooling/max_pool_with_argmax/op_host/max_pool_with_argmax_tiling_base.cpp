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
 * \file max_pool_with_argmax_tiling_base.cpp
 * \brief
 */

#include "max_pool_with_argmax_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

const int INPUT_IDX_X = 0;
const int NCHW_DIMS = 4;
const int KERNEL_POS = 0;
const int STRIDE_POS = 1;
const int PADDING_POS = 2;
const int DTYPE_POS = 3;
const int INCLUDE_BATCH_IN_INDEX_POS = 4;
const int DATA_FORMAT_POS = 5;
const int NAN_PROP_POS = 6;
const int WS_SYS_SIZE = 16 * 1024 * 1024;
static const int MP_MAX_2D_DIM_ZERO = 0;
static const int MP_MAX_2D_DIM_ONE = 1;
static const int MP_MAX_2D_DIM_TWO = 2;
static const int MP_MAX_2D_DIM_THREE = 3;
static const int64_t MP_MAX_2D_TYPE_INT32 = 3;
static const int64_t MP_MAX_2D_TYPE_INT64 = 9;
static const int64_t DIGIT_ONE = 1;
static const uint32_t DIGIT_TWO = 2;
static const gert::Shape g_vec_1_shape = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

static ge::graphStatus CheckOutPutShapeForValid(gert::TilingContext* context, const InputInfo& inputData)
{
    int64_t expectedH = (inputData.inputShape[H_DIM] - inputData.kernelSize[H_DIM] + inputData.stride[H_DIM]) /
                        inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] - inputData.kernelSize[W_DIM] + inputData.stride[W_DIM]) /
                        inputData.stride[W_DIM];
    if (static_cast<int64_t>(inputData.outShape[H_DIM]) != expectedH ||
        static_cast<int64_t>(inputData.outShape[W_DIM]) != expectedW) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "y[h_dim, w_dim]",
            ("[" + std::to_string(static_cast<int64_t>(inputData.outShape[H_DIM])) + ", " +
             std::to_string(static_cast<int64_t>(inputData.outShape[W_DIM])) + "]")
                .c_str(),
            ("y[h_dim, w_dim] must be equal to the theoretical value [" + std::to_string(expectedH) + ", " +
             std::to_string(expectedW) + "]")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShapeForSame(gert::TilingContext* context, const InputInfo& inputData)
{
    int64_t expectedH = (inputData.inputShape[H_DIM] + inputData.stride[H_DIM] - 1) / inputData.stride[H_DIM];
    int64_t expectedW = (inputData.inputShape[W_DIM] + inputData.stride[W_DIM] - 1) / inputData.stride[W_DIM];
    if (static_cast<int64_t>(inputData.outShape[H_DIM]) != expectedH ||
        static_cast<int64_t>(inputData.outShape[W_DIM]) != expectedW) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "y[h_dim, w_dim]",
            ("[" + std::to_string(static_cast<int64_t>(inputData.outShape[H_DIM])) + ", " +
             std::to_string(static_cast<int64_t>(inputData.outShape[W_DIM])) + "]")
                .c_str(),
            ("y[h_dim, w_dim] must be equal to the theoretical value [" + std::to_string(expectedH) + ", " +
             std::to_string(expectedW) + "]")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutPutShape(gert::TilingContext* context, const InputInfo& inputData,
                                        const string& padModeStr)
{
    if (padModeStr == "VALID") {
        return CheckOutPutShapeForValid(context, inputData);
    }
    return CheckOutPutShapeForSame(context, inputData);
}

ge::graphStatus MaxPoolWithArgmaxBaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const MaxPoolWithArgmaxCompileInfo*>(context_->GetCompileInfo());
        OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context_, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;

        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    }
    OP_TILING_CHECK(coreNum == 0, CUBE_INNER_ERR_REPORT(context_, "coreNum is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxBaseTiling::GetShapeAttrsInfo()
{
    auto inputX = context_->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());

    OP_TILING_CHECK(
        inputShape.GetDimNum() != NCHW_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(inputShape.GetDimNum()).c_str(), "4"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputShape.GetShapeSize() <= 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "x",
                                                  std::to_string(inputShape.GetShapeSize()).c_str(), "greater than 0"),
                    return ge::GRAPH_FAILED);
    auto inputDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", Ops::Base::ToString(dtype).c_str(),
                                  "DT_FLOAT, DT_FLOAT16, or DT_BF16");
        return ge::GRAPH_FAILED;
    }

    auto outX = context_->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = EnsureNotScalar(outX->GetStorageShape());
    auto indicesX = context_->GetOutputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, indicesX);
    auto indicesShape = EnsureNotScalar(indicesX->GetStorageShape());
    if (indicesShape != outShape) {
        std::string errorMsg = Ops::Base::ToString(outShape) + " and " + Ops::Base::ToString(indicesShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "y and argmax", errorMsg.c_str(),
                                               "the shapes of y and argmax must be the same");
        return ge::GRAPH_FAILED;
    }
    auto runtimeAttrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    std::string inputFormatStr("NHWC");
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(DATA_FORMAT_POS);
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }
    int32_t c_dim = 1;
    int32_t h_dim = MP_MAX_2D_DIM_TWO;
    int32_t w_dim = MP_MAX_2D_DIM_THREE;
    if (inputFormatStr == "NCHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCHW;
        inputData.nInput = inputShape.GetDim(MP_MAX_2D_DIM_ZERO);
        inputData.cInput = inputShape.GetDim(MP_MAX_2D_DIM_ONE);
    } else if (inputFormatStr == "NHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NHWC;
        h_dim = MP_MAX_2D_DIM_ONE;
        w_dim = MP_MAX_2D_DIM_TWO;
        c_dim = MP_MAX_2D_DIM_THREE;
        inputData.nInput = inputShape.GetDim(MP_MAX_2D_DIM_ZERO);
        inputData.cInput = inputShape.GetDim(MP_MAX_2D_DIM_THREE);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "data_format", inputFormatStr.c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(outShape.GetDim(h_dim) < 1 || outShape.GetDim(w_dim) < 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y[h_dim, w_dim]",
                                                          ("[" + std::to_string(outShape.GetDim(h_dim)) + ", " +
                                                           std::to_string(outShape.GetDim(w_dim)) + "]")
                                                              .c_str(),
                                                          "h_dim and w_dim shape of y should be greater than 0"),
                    return ge::GRAPH_FAILED);

    inputData.inputShape = array<uint64_t, HW_DIMS>{uint64_t(inputShape.GetDim(h_dim)),
                                                    uint64_t(inputShape.GetDim(w_dim))};
    inputData.outShape = array<uint64_t, HW_DIMS>{uint64_t(outShape.GetDim(h_dim)), uint64_t(outShape.GetDim(w_dim))};

    int32_t nValue = 0;
    int32_t cValue = 0;
    int32_t hValue = 0;
    int32_t wValue = 0;
    const gert::TypedContinuousVector<int64_t>* kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    nValue = *(kernelSize->GetData());
    cValue = *(kernelSize->GetData() + c_dim);
    OP_TILING_CHECK((nValue != DIGIT_ONE) || (cValue != DIGIT_ONE),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "ksize[n_dim, c_dim]",
                        ("[" + std::to_string(nValue) + ", " + std::to_string(cValue) + "]").c_str(),
                        "n_dim and c_dim of ksize must be 1"),
                    return ge::GRAPH_FAILED);
    hValue = *(kernelSize->GetData() + h_dim);
    wValue = *(kernelSize->GetData() + w_dim);
    inputData.kernelSize = array<uint64_t, HW_DIMS>{uint64_t(hValue), uint64_t(wValue)};
    OP_TILING_CHECK(hValue <= 0 || wValue <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "ksize[h_dim, w_dim]",
                        ("[" + std::to_string(hValue) + ", " + std::to_string(wValue) + "]").c_str(),
                        "h_dim and w_dim of ksize should be greater than 0"),
                    return ge::GRAPH_FAILED);

    const gert::TypedContinuousVector<int64_t>* stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, stride);
    nValue = *(stride->GetData());
    cValue = *(stride->GetData() + c_dim);
    OP_TILING_CHECK((nValue != DIGIT_ONE) || (cValue != DIGIT_ONE),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "strides[n_dim, c_dim]",
                        ("[" + std::to_string(nValue) + ", " + std::to_string(cValue) + "]").c_str(),
                        "n_dim and c_dim of strides must be 1"),
                    return ge::GRAPH_FAILED);
    hValue = *(stride->GetData() + h_dim);
    wValue = *(stride->GetData() + w_dim);
    inputData.stride = array<uint64_t, HW_DIMS>{uint64_t(hValue), uint64_t(wValue)};
    OP_TILING_CHECK(hValue <= 0 || wValue <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "strides[h_dim, w_dim]",
                        ("[" + std::to_string(hValue) + ", " + std::to_string(wValue) + "]").c_str(),
                        "h_dim and w_dim of strides should be greater than 0"),
                    return ge::GRAPH_FAILED);

    auto padModePtr = runtimeAttrs->GetStr(PADDING_POS);
    string padMode(padModePtr);
    OP_TILING_CHECK(padMode != "VALID" && padMode != "SAME",
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "padding", padMode.c_str(), "VALID or SAME"),
                    return ge::GRAPH_FAILED);
    inputData.isPad = 0;
    if (padMode == "VALID") {
        inputData.pad = {0, 0}; // top, left
    } else {
        uint64_t hActualNeed = (inputData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
        uint64_t wActualNeed = (inputData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];

        uint64_t hPadNeed = (hActualNeed > inputData.inputShape[H_DIM]) ? (hActualNeed - inputData.inputShape[H_DIM]) :
                                                                          0;
        uint64_t wPadNeed = (wActualNeed > inputData.inputShape[W_DIM]) ? (wActualNeed - inputData.inputShape[W_DIM]) :
                                                                          0;
        if (hPadNeed != 0 || wPadNeed != 0) {
            inputData.isPad = 1;
        }

        inputData.pad = {hPadNeed / DIGIT_TWO, wPadNeed / DIGIT_TWO}; // top, left
    }

    ge::graphStatus res = CheckOutPutShape(context_, inputData, padMode);
    if (res != ge::GRAPH_SUCCESS) {
        return res;
    }

    int indexDtype = MP_MAX_2D_TYPE_INT64;
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
            inputData.indexDtype = ge::DataType::DT_INT64;
            break;
    }

    const bool* includeBatchInIndex = runtimeAttrs->GetAttrPointer<bool>(INCLUDE_BATCH_IN_INDEX_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, includeBatchInIndex);
    inputData.includeBatchInIndex = static_cast<int64_t>(*includeBatchInIndex);
    // only support false now
    OP_TILING_CHECK(inputData.includeBatchInIndex != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "include_batch_in_index",
                                                          std::to_string(inputData.includeBatchInIndex).c_str(),
                                                          "include_batch_in_index only support 0(false) now"),
                    return ge::GRAPH_FAILED);

    const bool* nanProp = runtimeAttrs->GetAttrPointer<bool>(NAN_PROP_POS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, nanProp);
    inputData.nanProp = static_cast<int64_t>(*nanProp);

    return ge::GRAPH_SUCCESS;
}

bool MaxPoolWithArgmaxBaseTiling::IsCapable() { return true; }

ge::graphStatus MaxPoolWithArgmaxBaseTiling::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MaxPoolWithArgmaxBaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t MaxPoolWithArgmaxBaseTiling::GetTilingKey() const { return 0; }

ge::graphStatus MaxPoolWithArgmaxBaseTiling::GetWorkspaceSize()
{
    auto sys_workspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolWithArgmaxBaseTiling::PostTiling() { return ge::GRAPH_SUCCESS; }
} // namespace optiling