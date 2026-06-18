
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
 * \file max_pool3d_with_argmax_v2_ksize_one_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <string>
#include "log/log.h"
#include "util/math_util.h"
#include "error_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool3d_with_argmax_v2_ksize_one_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"

namespace optiling {
static constexpr int64_t KSIZE_ONE_TILING_KEY = 700001;
static constexpr int64_t FLOAT16_OR_BF16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESERVED_SIZE = 1024;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t NCDHW_DIMS = 5;
static constexpr int64_t KERNEL_POS = 0;
static constexpr int64_t STRIDE_POS = 1;
static constexpr int64_t PADDING_POS = 2;
static constexpr int64_t DILATION_POS = 3;
static constexpr int64_t CEIL_POS = 4;
static constexpr int64_t FORMAT_POS = 5;
static constexpr int64_t DTYPE_POS = 6;
static const int32_t MP_MAX_3D_DIM_ZERO = 0;
static const int32_t MP_MAX_3D_DIM_ONE = 1;
static const int32_t MP_MAX_3D_DIM_TWO = 2;
static const int32_t MP_MAX_3D_DIM_THREE = 3;
static const int32_t MP_MAX_3D_DIM_FOUR = 4;
static const int64_t MP_MAX_3D_TYPE_INT32 = 3;
static const int64_t MP_MAX_3D_TYPE_INT64 = 9;
static constexpr int64_t ALIGN_LENGTH = 512;
static constexpr int64_t MIN_DATA_SIZE = 1024;

static const gert::Shape g_vec_1_shape = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus MaxPool3DWithArgmaxV2KsizeOneTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const MaxPool3DWithArgmaxV2CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = ubSizePlatform;
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2KsizeOneTiling::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_PARAM_INVALID;
    }

    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    OP_CHECK_IF(
        inputShape.GetDimNum() != NCDHW_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
            std::to_string(inputShape.GetDimNum()).c_str(), std::to_string(NCDHW_DIMS).c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputShape.GetShapeSize() <= 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "x",
            std::to_string(inputShape.GetShapeSize()).c_str(), "greater than 0"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(dtype).c_str(),
            "BFLOAT16, FLOAT16 or FLOAT32");
        return ge::GRAPH_FAILED;
    }

    auto outX = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = EnsureNotScalar(outX->GetStorageShape());
    auto indicesX = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesX);
    auto indicesShape = EnsureNotScalar(indicesX->GetStorageShape());
    if (indicesShape != outShape) {
        std::string shapeMsg = Ops::Base::ToString(indicesShape) + " and " + Ops::Base::ToString(outShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "indicesShape and outShape", shapeMsg.c_str(),
            "The shape of indices must be the same as the shape of out");
        return ge::GRAPH_FAILED;
    }

    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    std::string inputFormatStr("NCDHW");
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }

    int d_dim = MP_MAX_3D_DIM_TWO;
    int h_dim = MP_MAX_3D_DIM_THREE;
    int w_dim = MP_MAX_3D_DIM_FOUR;

    if (inputFormatStr == "NCDHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCDHW;
        inputData.nInput = inputShape.GetDim(MP_MAX_3D_DIM_ZERO);
        inputData.cInput = inputShape.GetDim(MP_MAX_3D_DIM_ONE);
        inputData.batches = inputData.nInput * inputData.cInput;
    } else {
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(
        outShape.GetDim(d_dim) < 1 || outShape.GetDim(h_dim) < 1 || outShape.GetDim(w_dim) < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "outShape", 
                            ("{" + std::to_string(outShape.GetDim(d_dim)) + ", " +
                            std::to_string(outShape.GetDim(h_dim)) + ", " +
                            std::to_string(outShape.GetDim(w_dim)) + "}").c_str(),
                            "all axes of out must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);

    inputData.inputShape = array<uint64_t, DHW_DIMS>{
        uint64_t(inputShape.GetDim(d_dim)), uint64_t(inputShape.GetDim(h_dim)), uint64_t(inputShape.GetDim(w_dim))};
    inputData.outShape = array<uint64_t, DHW_DIMS>{
        uint64_t(outShape.GetDim(d_dim)), uint64_t(outShape.GetDim(h_dim)), uint64_t(outShape.GetDim(w_dim))};

    auto outputIndicesDesc = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputIndicesDesc);
    auto indicesDtype = outputIndicesDesc->GetDataType();
    OP_CHECK_IF(
        inputShape.GetDim(d_dim) * inputShape.GetDim(h_dim) * inputShape.GetDim(w_dim) >
                static_cast<int64_t>(std::numeric_limits<int32_t>::max()) &&
            indicesDtype != ge::DT_INT64,
        OP_LOGI(
            context_->GetNodeName(),
            "The value of D*H*W exceeds INT32_MAX, the data type of indices should be DT_INT64"),
        return ge::GRAPH_FAILED);

    int32_t dValue = 0;
    int32_t hValue = 0;
    int32_t wValue = 0;
    const gert::TypedContinuousVector<int64_t>* kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    dValue = *(kernelSize->GetData());
    hValue = *(kernelSize->GetData() + 1);
    wValue = *(kernelSize->GetData() + MP_MAX_3D_DIM_TWO);
    inputData.kernelSize = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        dValue <= 0 || hValue <= 0 || wValue <= 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "ksize", 
            ("{" + std::to_string(dValue) + ", " + std::to_string(hValue) + ", " + std::to_string(wValue) + "}").c_str(),
            "all values of ksize must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);

    int32_t kdValue = dValue;
    int32_t khValue = hValue;
    int32_t kwValue = wValue;
    const gert::TypedContinuousVector<int64_t>* stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stride);
    dValue = *(stride->GetData());
    hValue = *(stride->GetData() + 1);
    wValue = *(stride->GetData() + MP_MAX_3D_DIM_TWO);
    inputData.stride = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        hValue <= 0 || wValue <= 0 || dValue <= 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "strides", 
            ("{" + std::to_string(dValue) + ", " + std::to_string(hValue) + ", " + std::to_string(wValue) + "}").c_str(),
            "all values of strides must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);

    const gert::TypedContinuousVector<int64_t>* padding = runtimeAttrs->GetListInt(PADDING_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padding);
    dValue = *(padding->GetData());
    hValue = *(padding->GetData() + 1);
    wValue = *(padding->GetData() + MP_MAX_3D_DIM_TWO);
    inputData.pad = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        hValue > khValue / MP_MAX_3D_DIM_TWO || wValue > kwValue / MP_MAX_3D_DIM_TWO ||
            dValue > kdValue / MP_MAX_3D_DIM_TWO,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "pads", 
            ("pad {" + std::to_string(dValue) + ", " + std::to_string(hValue) + ", " +
            std::to_string(wValue) + "}, kernel_size {" + std::to_string(kdValue) + ", " +
            std::to_string(khValue) + ", " + std::to_string(kwValue) + "}").c_str(),
            "each value of pads must be less than or equal to half of the corresponding kernel_size value"),
        return ge::GRAPH_FAILED);

    inputData.dilation = array<uint64_t, DHW_DIMS>{1, 1, 1};
    dValue = 1;
    hValue = 1;
    wValue = 1;
    const gert::TypedContinuousVector<int64_t>* dilation = runtimeAttrs->GetListInt(DILATION_POS);
    if (dilation != nullptr) {
        dValue = *(dilation->GetData());
        hValue = *(dilation->GetData() + 1);
        wValue = *(dilation->GetData() + MP_MAX_3D_DIM_TWO);
        inputData.dilation = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
        OP_CHECK_IF(
            dValue <= 0 || hValue <= 0 || wValue <= 0,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "dilation", 
                ("{" + std::to_string(dValue) + ", " + std::to_string(hValue) + ", " +
                std::to_string(wValue) + "}").c_str(),
                "all values of dilation must be greater than or equal to 1"),
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
        case MP_MAX_3D_TYPE_INT32:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
        case MP_MAX_3D_TYPE_INT64:
            inputData.indexDtype = ge::DataType::DT_INT64;
            break;
        default:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
    }

    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DWithArgmaxV2KsizeOneTiling::IsCapable()
{
    if (inputData.kernelSize[D_DIM] != 1 || inputData.kernelSize[H_DIM] != 1 || inputData.kernelSize[W_DIM] != 1) {
        return false;
    }
    if (inputData.stride[D_DIM] != 1 || inputData.stride[H_DIM] != 1 || inputData.stride[W_DIM] != 1) {
        return false;
    }
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;
    }
    InitializationVars();
    return true;
}

uint64_t MaxPool3DWithArgmaxV2KsizeOneTiling::GetTilingKey() const { return KSIZE_ONE_TILING_KEY; }

void MaxPool3DWithArgmaxV2KsizeOneTiling::InitializationVars()
{
    inputBytes_ = (dtype == ge::DataType::DT_FLOAT) ? FLOAT32_SIZE : FLOAT16_OR_BF16_SIZE;
    indexBytes_ = inputData.indexDtype == ge::DataType::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    ncTotal_ = static_cast<int64_t>(inputData.nInput) * static_cast<int64_t>(inputData.cInput);
}

void MaxPool3DWithArgmaxV2KsizeOneTiling::DoTiling()
{
    int64_t blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t dOut = static_cast<int64_t>(inputData.outShape[D_DIM]);
    int64_t hOut = static_cast<int64_t>(inputData.outShape[H_DIM]);
    int64_t wOut = static_cast<int64_t>(inputData.outShape[W_DIM]);
    int64_t totalElems = ncTotal_ * dOut * hOut * wOut;

    int64_t availableUb = static_cast<int64_t>(ubSize) - UB_RESERVED_SIZE;
    ubFactor_ = availableUb / DOUBLE_BUFFER / (inputBytes_ + indexBytes_);
    int64_t alignElems = blockSize / inputBytes_;
    ubFactor_ = (ubFactor_ / alignElems) * alignElems;

    int64_t alignSize = ALIGN_LENGTH / inputBytes_;
    int64_t coreData = Ops::Base::CeilDiv(totalElems, static_cast<int64_t>(coreNum));
    coreData = Ops::Base::CeilAlign(coreData, alignSize);
    coreData = std::max(coreData, MIN_DATA_SIZE);
    usedCoreNum_ = Ops::Base::CeilDiv(totalElems, coreData);

    blockFactor_ = coreData;
    tailBlockFactor_ = totalElems - (usedCoreNum_ - 1) * blockFactor_;
    coreLoop_ = Ops::Base::CeilDiv(blockFactor_, ubFactor_);
    tailUbFactor_ = blockFactor_ - (coreLoop_ - 1) * ubFactor_;
    tailCoreLoop_ = Ops::Base::CeilDiv(tailBlockFactor_, ubFactor_);
    tailCoreTailUbFactor_ = tailBlockFactor_ - (tailCoreLoop_ - 1) * ubFactor_;
    inputBufferSize_ = Ops::Base::CeilAlign(ubFactor_ * inputBytes_, blockSize);
    argmaxBufferSize_ = Ops::Base::CeilAlign(ubFactor_ * indexBytes_, blockSize);
}

void MaxPool3DWithArgmaxV2KsizeOneTiling::SetTilingData()
{
    tilingData_->dInput = static_cast<int64_t>(inputData.inputShape[D_DIM]);
    tilingData_->hInput = static_cast<int64_t>(inputData.inputShape[H_DIM]);
    tilingData_->wInput = static_cast<int64_t>(inputData.inputShape[W_DIM]);
    tilingData_->dOutput = static_cast<int64_t>(inputData.outShape[D_DIM]);
    tilingData_->hOutput = static_cast<int64_t>(inputData.outShape[H_DIM]);
    tilingData_->wOutput = static_cast<int64_t>(inputData.outShape[W_DIM]);
    tilingData_->padD = static_cast<int64_t>(inputData.pad[D_DIM]);
    tilingData_->padH = static_cast<int64_t>(inputData.pad[H_DIM]);
    tilingData_->padW = static_cast<int64_t>(inputData.pad[W_DIM]);
    tilingData_->ncTotal = ncTotal_;
    tilingData_->usedCoreNum = usedCoreNum_;
    tilingData_->blockFactor = blockFactor_;
    tilingData_->tailBlockFactor = tailBlockFactor_;
    tilingData_->coreLoop = coreLoop_;
    tilingData_->tailCoreLoop = tailCoreLoop_;
    tilingData_->ubFactor = ubFactor_;
    tilingData_->tailUbFactor = tailUbFactor_;
    tilingData_->tailCoreTailUbFactor = tailCoreTailUbFactor_;
    tilingData_->inputBufferSize = inputBufferSize_;
    tilingData_->argmaxBufferSize = argmaxBufferSize_;
}

void MaxPool3DWithArgmaxV2KsizeOneTiling::PrintBaseData() const
{
    OP_LOGD("MaxPool3DWithArgmaxV2KsizeOne", "[MaxPool3DWithArgmaxV2KsizeOneTiling] PrintBaseData start running");

    std::ostringstream info;
    info << "inputBytes: " << inputBytes_ << std::endl;
    info << "indexBytes: " << indexBytes_ << std::endl;
    info << "ncTotal: " << ncTotal_ << std::endl;
    info << "inputShape: [" << inputData.inputShape[D_DIM] << ", " << inputData.inputShape[H_DIM] << ", "
         << inputData.inputShape[W_DIM] << "]" << std::endl;
    info << "outShape: [" << inputData.outShape[D_DIM] << ", " << inputData.outShape[H_DIM] << ", "
         << inputData.outShape[W_DIM] << "]" << std::endl;
    info << "kernelSize: [" << inputData.kernelSize[D_DIM] << ", " << inputData.kernelSize[H_DIM] << ", "
         << inputData.kernelSize[W_DIM] << "]" << std::endl;
    info << "stride: [" << inputData.stride[D_DIM] << ", " << inputData.stride[H_DIM] << ", " << inputData.stride[W_DIM]
         << "]" << std::endl;
    info << "pad: [" << inputData.pad[D_DIM] << ", " << inputData.pad[H_DIM] << ", " << inputData.pad[W_DIM] << "]"
         << std::endl;
    info << "batches: " << inputData.batches << std::endl;
    info << "coreNum: " << coreNum << std::endl;
    info << "ubSize: " << ubSize << std::endl;

    OP_LOGI("MaxPool3DWithArgmaxV2KsizeOne", "%s", info.str().c_str());
}

void MaxPool3DWithArgmaxV2KsizeOneTiling::PrintTilingData() const
{
    OP_LOGD("MaxPool3DWithArgmaxV2KsizeOne", "[MaxPool3DWithArgmaxV2KsizeOneTiling] PrintTilingData start running");

    std::ostringstream info;
    info << "usedCoreNum: " << usedCoreNum_ << std::endl;
    info << "blockFactor: " << blockFactor_ << std::endl;
    info << "tailBlockFactor: " << tailBlockFactor_ << std::endl;
    info << "coreLoop: " << coreLoop_ << std::endl;
    info << "tailCoreLoop: " << tailCoreLoop_ << std::endl;
    info << "ubFactor: " << ubFactor_ << std::endl;
    info << "tailUbFactor: " << tailUbFactor_ << std::endl;
    info << "tailCoreTailUbFactor: " << tailCoreTailUbFactor_ << std::endl;
    info << "inputBufferSize: " << inputBufferSize_ << std::endl;
    info << "argmaxBufferSize: " << argmaxBufferSize_ << std::endl;

    OP_LOGI("MaxPool3DWithArgmaxV2KsizeOne", "%s", info.str().c_str());
}

ge::graphStatus MaxPool3DWithArgmaxV2KsizeOneTiling::DoOpTiling()
{
    DoTiling();
    SetTilingData();
    PrintBaseData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2KsizeOneTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2KsizeOneTiling, 0);

} // namespace optiling