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
 * \file max_pool3d_with_argmax_v2_tiling_nc_transpose_.cpp
 * \brief NC transpose tiling for arch35: optimized for large NC + small kernel scenarios
 */

#include <cctype>
#include <algorithm>
#include "log/log.h"
#include "util/math_util.h"
#include "error_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool3d_with_argmax_v2_tiling_nc_transpose.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"

namespace optiling
{
static constexpr int64_t FLOAT16_OR_BF16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t TRANS_ALIGN = 16;
static constexpr int64_t INT16_MAX_LIMIT = 32767;

static constexpr int64_t STRIDE_POS = 1;
static constexpr int64_t NCDHW_DIMS = 5;
static constexpr int64_t KERNEL_POS = 0;
static constexpr int64_t CEIL_POS = 4;
static constexpr int64_t FORMAT_POS = 5;
static constexpr int64_t DTYPE_POS = 6;
static constexpr int64_t PADDING_POS = 2;
static constexpr int64_t DILATION_POS = 3;

static const int64_t MP_MAX_3D_TYPE_INT32 = 3;
static const int64_t MP_MAX_3D_TYPE_INT64 = 9;
static const int32_t MP_MAX_3D_DIM_ZERO = 0;
static const int32_t MP_MAX_3D_DIM_ONE = 1;
static const int32_t MP_MAX_3D_DIM_TWO = 2;
static const int32_t MP_MAX_3D_DIM_THREE = 3;
static const int32_t MP_MAX_3D_DIM_FOUR = 4;

static constexpr int64_t NC_TRANSPOSE_NO_PAD_KEY = 500001;
// Kernel volume threshold: only enable NC transpose for small kernels
static constexpr int64_t MAX_KERNEL_VOLUME = 128;
// Minimum NC to justify the transpose overhead
static constexpr int64_t MIN_NC_FOR_TRANSPOSE = 48 * 64;
static constexpr int64_t DILATION_THRESHOLD = 1;
static constexpr int64_t NC_TRANSPOSE_MIN_OUT_SPATIAL = 32;

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const MaxPool3DWithArgmaxV2CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context_, "compile info is null"),
                        return ge::GRAPH_FAILED);
        ubSize = compileInfoPtr->ubSize;
        coreNum = compileInfoPtr->coreNum;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = ubSizePlatform;
    }
    OP_CHECK_IF(coreNum == 0, CUBE_INNER_ERR_REPORT(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::ValidatePlatformAndInput()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_PARAM_INVALID;
    }

    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = inputX->GetStorageShape();
    OP_CHECK_IF(inputShape.GetDimNum() != NCDHW_DIMS,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x",
                    std::to_string(inputShape.GetDimNum()).c_str(), std::to_string(NCDHW_DIMS).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputShape.GetShapeSize() <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE(context_->GetNodeName(), "x",
                    std::to_string(inputShape.GetShapeSize()).c_str(), "greater than 0"),
                return ge::GRAPH_FAILED);

    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
                    ge::TypeUtils::DataTypeToSerialString(dtype).c_str(),
                    "BFLOAT16, FLOAT16 or FLOAT32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::ParseShapeAndFormat()
{
    auto inputX = context_->GetInputShape(0);
    auto inputShape = inputX->GetStorageShape();
    auto outX = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = outX->GetStorageShape();

    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    std::string inputFormatStr("NCDHW");
    const char* inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }
    if (inputFormatStr != "NCDHW") {
        return ge::GRAPH_PARAM_INVALID;
    }

    inputData.inputFormat = ge::Format::FORMAT_NCDHW;
    inputData.batches = inputShape.GetDim(MP_MAX_3D_DIM_ZERO) * inputShape.GetDim(MP_MAX_3D_DIM_ONE);
    inputData.nInput = inputShape.GetDim(MP_MAX_3D_DIM_ZERO);
    inputData.cInput = inputShape.GetDim(MP_MAX_3D_DIM_ONE);

    int d_dim = MP_MAX_3D_DIM_TWO;
    int h_dim = MP_MAX_3D_DIM_THREE;
    int w_dim = MP_MAX_3D_DIM_FOUR;
    inputData.inputShape =
        array<uint64_t, DHW_DIMS>{uint64_t(inputShape.GetDim(d_dim)), uint64_t(inputShape.GetDim(h_dim)),
                                  uint64_t(inputShape.GetDim(w_dim))};
    inputData.outShape =
        array<uint64_t, DHW_DIMS>{uint64_t(outShape.GetDim(d_dim)), uint64_t(outShape.GetDim(h_dim)),
                                  uint64_t(outShape.GetDim(w_dim))};
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::GetShapeAttrsInfo()
{
    ge::graphStatus ret = ValidatePlatformAndInput();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ParseShapeAndFormat();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    ret = ParseKernelStrideAttrs(runtimeAttrs);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ParsePadDilationAttrs(runtimeAttrs);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ParseCeilModeAndDtype(runtimeAttrs);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::ParseKernelStrideAttrs(const gert::RuntimeAttrs* runtimeAttrs)
{
    int32_t dValue = 0;
    int32_t hValue = 0;
    int32_t wValue = 0;
    const gert::TypedContinuousVector<int64_t>* kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    OP_CHECK_IF(
        kernelSize->GetSize() < 3,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ksize",
            std::to_string(kernelSize->GetSize()).c_str(),
            "ksize list size must be greater than or equal to 3"),
        return ge::GRAPH_FAILED);
    const int64_t* kData = kernelSize->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, kData);
    dValue = *kData;
    hValue = *(kData + 1);
    wValue = *(kData + MP_MAX_3D_DIM_TWO);
    inputData.kernelSize = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        dValue <= 0 || hValue <= 0 || wValue <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ksize",
            ("{" + std::to_string(dValue) + ", " + std::to_string(hValue) + ", " +
             std::to_string(wValue) + "}").c_str(),
            "all values of ksize must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);

    const gert::TypedContinuousVector<int64_t>* stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stride);
    dValue = *(stride->GetData());
    hValue = *(stride->GetData() + 1);
    wValue = *(stride->GetData() + MP_MAX_3D_DIM_TWO);
    inputData.stride = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        dValue <= 0 || hValue <= 0 || wValue <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "strides",
            ("{" + std::to_string(dValue) + ", " + std::to_string(hValue) + ", " +
             std::to_string(wValue) + "}").c_str(),
            "all values of strides must be greater than or equal to 1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::ParsePadDilationAttrs(const gert::RuntimeAttrs* runtimeAttrs)
{
    int32_t dValue = 0;
    int32_t hValue = 0;
    int32_t wValue = 0;
    const gert::TypedContinuousVector<int64_t>* padding = runtimeAttrs->GetListInt(PADDING_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padding);
    dValue = *(padding->GetData());
    hValue = *(padding->GetData() + 1);
    wValue = *(padding->GetData() + MP_MAX_3D_DIM_TWO);
    inputData.pad = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};

    inputData.dilation = array<uint64_t, DHW_DIMS>{1, 1, 1};
    const gert::TypedContinuousVector<int64_t>* dilation = runtimeAttrs->GetListInt(DILATION_POS);
    if (dilation != nullptr) {
        dValue = *(dilation->GetData());
        hValue = *(dilation->GetData() + 1);
        wValue = *(dilation->GetData() + MP_MAX_3D_DIM_TWO);
        inputData.dilation = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::ParseCeilModeAndDtype(const gert::RuntimeAttrs* runtimeAttrs)
{
    int indexdtype = 3;
    const int* indexDtypePtr = runtimeAttrs->GetAttrPointer<int>(DTYPE_POS);
    if (indexDtypePtr != nullptr) {
        indexdtype = *indexDtypePtr;
    }
    switch (indexdtype) {
        case MP_MAX_3D_TYPE_INT64:
            inputData.indexDtype = ge::DataType::DT_INT64;
            break;
        case MP_MAX_3D_TYPE_INT32:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;     
        default:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
    }

    inputData.ceilMode = false;
    const bool* ceilModePtr = runtimeAttrs->GetAttrPointer<bool>(CEIL_POS);
    if (ceilModePtr != nullptr) {
        inputData.ceilMode = *ceilModePtr;
    }
    return ge::GRAPH_SUCCESS;
}

void MaxPool3DWithArgmaxV2NcTransposeTiling::InitializationVars()
{
    baseData_.inputBytes = dtype == ge::DataType::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_OR_BF16_SIZE;
    baseData_.indexBytes = inputData.indexDtype == ge::DataType::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    baseData_.availableUb = ubSize;
    baseData_.totalCoreNum = coreNum;
    baseData_.oneBlockNumT1 = Ops::Base::GetUbBlockSize(context_) / baseData_.inputBytes;
    baseData_.oneBlockNumT2 = Ops::Base::GetUbBlockSize(context_) / baseData_.indexBytes;
    baseData_.vlNumT1 = Ops::Base::GetVRegSize(context_) / baseData_.inputBytes;

    baseData_.dInput = inputData.inputShape[D_DIM];
    baseData_.hInput = inputData.inputShape[H_DIM];
    baseData_.wInput = inputData.inputShape[W_DIM];
    baseData_.padFront = inputData.pad[D_DIM];
    baseData_.padTop = inputData.pad[H_DIM];
    baseData_.padLeft = inputData.pad[W_DIM];   
    baseData_.dStride = inputData.stride[D_DIM];
    baseData_.hStride = inputData.stride[H_DIM];
    baseData_.wStride = inputData.stride[W_DIM];
    baseData_.dKernel = inputData.kernelSize[D_DIM];
    baseData_.hKernel = inputData.kernelSize[H_DIM];
    baseData_.wKernel = inputData.kernelSize[W_DIM];
    baseData_.dOutput = inputData.outShape[D_DIM];
    baseData_.hOutput = inputData.outShape[H_DIM];
    baseData_.wOutput = inputData.outShape[W_DIM];
    baseData_.highAxisTotal = inputData.batches;
    baseData_.isPad = 0;
    if (baseData_.padTop != 0 || baseData_.padFront != 0 || baseData_.padLeft != 0) {
        baseData_.isPad = 1;
    }
    if (inputData.ceilMode && baseData_.isPad == 0) {
        if (((baseData_.dOutput - 1) * baseData_.dStride + baseData_.dKernel) != baseData_.dInput ||
            ((baseData_.hOutput - 1) * baseData_.hStride + baseData_.hKernel) != baseData_.hInput ||
            ((baseData_.wOutput - 1) * baseData_.wStride + baseData_.wKernel) != baseData_.wInput) {
            baseData_.isPad = 1;
        }
    }
}

bool MaxPool3DWithArgmaxV2NcTransposeTiling::IsCapable()
{
    // Only for NCDHW format, no dilation
    if (inputData.dilation[D_DIM] > DILATION_THRESHOLD || inputData.dilation[H_DIM] > DILATION_THRESHOLD ||
        inputData.dilation[W_DIM] > DILATION_THRESHOLD || inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;
    }

    InitializationVars();

    // Pad path disabled: fallback to other templates when padding exists
    if (baseData_.isPad == 1) {
        return false;
    }

    // Argmax is computed as int32 internally; if D*H*W exceeds INT32_MAX, int32 overflows
    int64_t inputDHW = baseData_.dInput * baseData_.hInput * baseData_.wInput;
    if (inputDHW > INT32_MAX) {
        return false;
    }

    // Only for small kernel + large NC scenarios
    int64_t kernelVolume = baseData_.dKernel * baseData_.hKernel * baseData_.wKernel;
    if (kernelVolume > MAX_KERNEL_VOLUME) {
        return false;
    }
    if (baseData_.highAxisTotal < MIN_NC_FOR_TRANSPOSE) {
        return false;
    }

    // Output spatial size too small: transpose overhead dominates effective compute
    int64_t outSpatialSize = baseData_.dOutput * baseData_.hOutput * baseData_.wOutput;
    if (outSpatialSize < NC_TRANSPOSE_MIN_OUT_SPATIAL) {
        return false;
    }

    // ncInner is fixed to vlNumT1 (the vector register length)
    splitData_.ncInner = baseData_.vlNumT1;
    splitData_.dOutputInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = 1;
    DoBufferCalculate();
    if (splitData_.totalBufferSize > baseData_.availableUb) {
        return false;
    }

    // Delta value range check: delta is stored as IDX_T (int16_t for fp16/bf16, int32_t for float).
    // For fp16/bf16, IDX_T is int16_t with max value 32767. If max delta exceeds this, overflow occurs.
    int64_t maxDelta = (baseData_.dKernel - 1) * baseData_.hInput * baseData_.wInput +
                       (baseData_.hKernel - 1) * baseData_.wInput + (baseData_.wKernel - 1);
    if (baseData_.inputBytes == FLOAT16_OR_BF16_SIZE && maxDelta > INT16_MAX_LIMIT) {
        return false;
    }
    return true;
}

uint64_t MaxPool3DWithArgmaxV2NcTransposeTiling::GetTilingKey() const
{
    return NC_TRANSPOSE_NO_PAD_KEY;
}

void MaxPool3DWithArgmaxV2NcTransposeTiling::DoBufferCalculate()
{
    splitData_.dInputInner = (splitData_.dOutputInner - 1) * baseData_.dStride + baseData_.dKernel;
    splitData_.hInputInner = (splitData_.hOutputInner - 1) * baseData_.hStride + baseData_.hKernel;
    splitData_.wInputInner = (splitData_.wOutputInner - 1) * baseData_.wStride + baseData_.wKernel;

    int64_t wInputInnerAligned = Ops::Base::CeilAlign(splitData_.wInputInner, baseData_.oneBlockNumT1);
    int64_t ncInnerAligned = Ops::Base::CeilAlign(splitData_.ncInner, static_cast<int64_t>(TRANS_ALIGN));

    splitData_.inputBufferSize =
        splitData_.ncInner * splitData_.dInputInner * splitData_.hInputInner * wInputInnerAligned * baseData_.inputBytes;
    int64_t outputSpatialSize = splitData_.dOutputInner * splitData_.hOutputInner * splitData_.wOutputInner;
    int64_t outputSpatialAligned = Ops::Base::CeilAlign(outputSpatialSize, static_cast<int64_t>(TRANS_ALIGN));
    int64_t inputTransSize =
        splitData_.dInputInner * splitData_.hInputInner * wInputInnerAligned * ncInnerAligned * baseData_.inputBytes;
    int64_t outputTransSize = ncInnerAligned * outputSpatialAligned * (baseData_.inputBytes + baseData_.indexBytes);
    splitData_.transBufferSize = std::max(inputTransSize, outputTransSize);
    splitData_.maxValueBufferSize = outputSpatialAligned * ncInnerAligned * baseData_.inputBytes;
    splitData_.argmaxBufferSize =
        outputSpatialAligned * ncInnerAligned * std::max(INT32_SIZE, baseData_.indexBytes);
    int64_t kernelVolume = baseData_.dKernel * baseData_.hKernel * baseData_.wKernel;
    // delta 表：kernel 内只取 kernelVolume * vlIDX_ 个 IDX_T（fp16→int16, float→int32）
    splitData_.deltaIndexBufferSize = kernelVolume * baseData_.vlNumT1 * baseData_.inputBytes;

    // int64 Cast 临时 buffer：仅当 T2=int64 时需要，用于 Cast(int32→int64) 后搬出
    if (baseData_.indexBytes == INT64_SIZE) {
        splitData_.castBufferSize = outputSpatialAligned * INT64_SIZE;
    } else {
        splitData_.castBufferSize = 0;
    }

    int64_t totalSize = splitData_.inputBufferSize + splitData_.transBufferSize +
                        splitData_.maxValueBufferSize + splitData_.argmaxBufferSize +
                        splitData_.deltaIndexBufferSize + splitData_.castBufferSize;
    splitData_.totalBufferSize = totalSize;
}

bool MaxPool3DWithArgmaxV2NcTransposeTiling::IsMeetTargetCoreNum() const
{
    int64_t NUM_AIV = coreNum;
    int64_t tmpDOutputOuter = Ops::Base::CeilDiv(baseData_.dOutput, splitData_.dOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(baseData_.hOutput, splitData_.hOutputInner);
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(baseData_.wOutput, splitData_.wOutputInner);
    int64_t tmpNCOutputOuter = Ops::Base::CeilDiv(baseData_.highAxisTotal, splitData_.ncInner);
    return tmpWOutputOuter * tmpHOutputOuter * tmpDOutputOuter * tmpNCOutputOuter >= NUM_AIV;
}

bool MaxPool3DWithArgmaxV2NcTransposeTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData_.totalBufferSize <= baseData_.availableUb;
}

void MaxPool3DWithArgmaxV2NcTransposeTiling::BinarySearch(int64_t start, int64_t end, int64_t* value)
{
    int64_t bestSplit = 1;
    int64_t left = start;
    int64_t right = end;  

    while (left <= right) {
        int64_t midValue = left + (right - left) / DOUBLE;
        *value = midValue;
        if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
            bestSplit = midValue;
            left = midValue + 1;
        } else {
            right = midValue - 1;
        }
    }
    *value = bestSplit;
}

bool MaxPool3DWithArgmaxV2NcTransposeTiling::TrySplitNC()
{
    splitData_.ncInner = baseData_.vlNumT1;
    splitData_.dOutputInner = baseData_.dOutput;
    splitData_.hOutputInner = baseData_.hOutput;
    splitData_.wOutputInner = baseData_.wOutput;
    if (IsMeetUBSize()) {
        return true;
    }
    return false;
}

bool MaxPool3DWithArgmaxV2NcTransposeTiling::TrySplitD()
{
    splitData_.ncInner = baseData_.vlNumT1;
    splitData_.dOutputInner = 1;
    splitData_.hOutputInner = baseData_.hOutput;
    splitData_.wOutputInner = baseData_.wOutput;
    if (IsMeetUBSize()) {
        BinarySearch(1, baseData_.dOutput, &splitData_.dOutputInner);
        return true;
    }
    return false;
}

bool MaxPool3DWithArgmaxV2NcTransposeTiling::TrySplitH()
{
    splitData_.ncInner = baseData_.vlNumT1;
    splitData_.dOutputInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = baseData_.wOutput;
    // Try with relaxed core count: maximize spatial tile
    if (IsMeetUBSize()) {
        // Binary search for max hOutputInner (ignore core count)
        int64_t left = 1;
        int64_t right = baseData_.hOutput;
        int64_t bestH = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / DOUBLE;
            splitData_.hOutputInner = mid;
            if (IsMeetUBSize()) {
                bestH = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData_.hOutputInner = bestH;
        return true;
    }
    return false;
}

bool MaxPool3DWithArgmaxV2NcTransposeTiling::TrySplitW()
{
    splitData_.ncInner = baseData_.vlNumT1;
    splitData_.dOutputInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = 1;
    // Relax core count constraint: maximize wOutputInner to reduce CopyIn/CopyOut overhead
    if (IsMeetUBSize()) {
        // Binary search for max wOutputInner that fits in UB (ignore core count target)
        int64_t left = 1;
        int64_t right = baseData_.wOutput;
        int64_t bestW = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / DOUBLE;
            splitData_.wOutputInner = mid;
            if (IsMeetUBSize()) {
                bestW = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData_.wOutputInner = bestW;
        return true;
    }
    return false;
}

void MaxPool3DWithArgmaxV2NcTransposeTiling::SearchBestTiling()
{
    if (TrySplitNC()) {
        return;
    }
    if (TrySplitD()) {
        return;
    }
    if (TrySplitH()) {
        return;
    }
    if (TrySplitW()) {
        return;
    }
}

void MaxPool3DWithArgmaxV2NcTransposeTiling::DoUBTiling()
{
    SearchBestTiling();
    // 除0校验：Inner 值必须 > 0
    OP_CHECK_IF(splitData_.wOutputInner <= 0 || splitData_.hOutputInner <= 0 ||
                splitData_.dOutputInner <= 0 || splitData_.ncInner <= 0,
        CUBE_INNER_ERR_REPORT(context_, "NcTranspose: invalid inner values after SearchBestTiling, "
            "wInner=%ld, hInner=%ld, dInner=%ld, ncInner=%ld",
            splitData_.wOutputInner, splitData_.hOutputInner,
            splitData_.dOutputInner, splitData_.ncInner),
        return);
    DoBufferCalculate();

    splitData_.wOutputOuter = Ops::Base::CeilDiv(baseData_.wOutput, splitData_.wOutputInner);
    int64_t tempWTail = baseData_.wOutput % splitData_.wOutputInner;
    splitData_.wOutputTail = tempWTail == 0 ? splitData_.wOutputInner : tempWTail;

    splitData_.hOutputOuter = Ops::Base::CeilDiv(baseData_.hOutput, splitData_.hOutputInner);
    int64_t tempHTail = baseData_.hOutput % splitData_.hOutputInner;
    splitData_.hOutputTail = tempHTail == 0 ? splitData_.hOutputInner : tempHTail;

    splitData_.dOutputOuter = Ops::Base::CeilDiv(baseData_.dOutput, splitData_.dOutputInner);
    int64_t tempDTail = baseData_.dOutput % splitData_.dOutputInner;
    splitData_.dOutputTail = tempDTail == 0 ? splitData_.dOutputInner : tempDTail;

    splitData_.ncOuter = Ops::Base::CeilDiv(baseData_.highAxisTotal, splitData_.ncInner);
    int64_t tempNcTail = baseData_.highAxisTotal % splitData_.ncInner;
    splitData_.ncTail = tempNcTail == 0 ? splitData_.ncInner : tempNcTail;
}

void MaxPool3DWithArgmaxV2NcTransposeTiling::DoBlockTiling()
{
    splitData_.totalBaseBlockNum =
        splitData_.ncOuter * splitData_.dOutputOuter * splitData_.hOutputOuter * splitData_.wOutputOuter;
    splitData_.normalCoreProcessNum =
        Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, baseData_.totalCoreNum);
    splitData_.usedCoreNum =
        Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, splitData_.normalCoreProcessNum);
    splitData_.tailCoreProcessNum =
        splitData_.totalBaseBlockNum - splitData_.normalCoreProcessNum * (splitData_.usedCoreNum - 1);
}

void MaxPool3DWithArgmaxV2NcTransposeTiling::SetTilingData()
{
    tilingData_->dInput = baseData_.dInput;
    tilingData_->hInput = baseData_.hInput;
    tilingData_->wInput = baseData_.wInput;
    tilingData_->dOutputInner = splitData_.dOutputInner;
    tilingData_->dOutputTail = splitData_.dOutputTail;
    tilingData_->dOutputOuter = splitData_.dOutputOuter;
    tilingData_->hOutputInner = splitData_.hOutputInner;
    tilingData_->hOutputTail = splitData_.hOutputTail;
    tilingData_->hOutputOuter = splitData_.hOutputOuter;
    tilingData_->wOutputInner = splitData_.wOutputInner;
    tilingData_->wOutputTail = splitData_.wOutputTail;
    tilingData_->wOutputOuter = splitData_.wOutputOuter;    
    tilingData_->dKernel = baseData_.dKernel;
    tilingData_->hKernel = baseData_.hKernel;
    tilingData_->wKernel = baseData_.wKernel;
    tilingData_->dStride = baseData_.dStride;
    tilingData_->hStride = baseData_.hStride;
    tilingData_->wStride = baseData_.wStride;
    tilingData_->dOutput = baseData_.dOutput;
    tilingData_->hOutput = baseData_.hOutput;
    tilingData_->wOutput = baseData_.wOutput;
    tilingData_->ncInner = splitData_.ncInner;
    tilingData_->ncTail = splitData_.ncTail;
    tilingData_->ncOuter = splitData_.ncOuter; 
    tilingData_->padFront = baseData_.padFront;
    tilingData_->padTop = baseData_.padTop;
    tilingData_->padLeft = baseData_.padLeft;     
    tilingData_->normalCoreProcessNum = splitData_.normalCoreProcessNum;
    tilingData_->tailCoreProcessNum = splitData_.tailCoreProcessNum;
    tilingData_->usedCoreNum = splitData_.usedCoreNum;
    tilingData_->inputBufferSize = splitData_.inputBufferSize;
    tilingData_->transBufferSize = splitData_.transBufferSize;
    tilingData_->maxValueBufferSize = splitData_.maxValueBufferSize;
    tilingData_->argmaxBufferSize = splitData_.argmaxBufferSize;
    tilingData_->deltaIndexBufferSize = splitData_.deltaIndexBufferSize;
    tilingData_->castBufferSize = splitData_.castBufferSize;
    tilingData_->isPad = baseData_.isPad;

    // 传递给 kernel 的除数必须为正
    OP_CHECK_IF(splitData_.dOutputOuter <= 0 || splitData_.hOutputOuter <= 0 ||
                splitData_.wOutputOuter <= 0,
        CUBE_INNER_ERR_REPORT(context_, "NcTranspose: outer values must be positive, "
            "dOuter=%ld, hOuter=%ld, wOuter=%ld",
            splitData_.dOutputOuter, splitData_.hOutputOuter, splitData_.wOutputOuter),
        return);
}

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    OP_LOGI(context_->GetNodeName(),
            "NcTranspose tiling: ncInner=%ld, ncTail=%ld, ncOuter=%ld, "
            "dOut[%ld/%ld/%ld], hOut[%ld/%ld/%ld], wOut[%ld/%ld/%ld], "
            "usedCore=%ld, normalProc=%ld, tailProc=%ld, isPad=%ld, "
            "buf[input=%ld trans=%ld max=%ld argmax=%ld index=%ld total=%ld] availableUb=%ld",
            splitData_.ncInner, splitData_.ncTail, splitData_.ncOuter,
            splitData_.dOutputInner, splitData_.dOutputTail, splitData_.dOutputOuter,
            splitData_.hOutputInner, splitData_.hOutputTail, splitData_.hOutputOuter,
            splitData_.wOutputInner, splitData_.wOutputTail, splitData_.wOutputOuter,
            splitData_.usedCoreNum, splitData_.normalCoreProcessNum, splitData_.tailCoreProcessNum,
            baseData_.isPad,
            splitData_.inputBufferSize, splitData_.transBufferSize, splitData_.maxValueBufferSize,
            splitData_.argmaxBufferSize, splitData_.deltaIndexBufferSize, splitData_.totalBufferSize,
            baseData_.availableUb);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NcTransposeTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2NcTransposeTiling, 1);

}  // namespace optiling