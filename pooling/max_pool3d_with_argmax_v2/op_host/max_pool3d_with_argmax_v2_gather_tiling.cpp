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
 * \file max_pool_with_argmax_v3_gather_tiling.cpp
 * \brief
 */

#include <cctype>
#include <algorithm>
#include "log/log.h"
#include "util/math_util.h"
#include "error_util.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "max_pool3d_with_argmax_v2_gather_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling_base/tiling_util.h"
#include <iostream>
namespace optiling
{
static constexpr int64_t FLOAT16_OR_BF16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;
static constexpr int64_t UB_RESVERVED_SIZE = 0;
static constexpr int64_t HELPER_BUFFER_SIZE = 1024;
static constexpr int64_t NO_PADDING_TILING_KEY = 400001;
static constexpr int64_t PADDING_TILING_KEY = 400002;
static constexpr int64_t MAX_BANDWIDTH_COEFFICIENTS = 2;
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t CACHE_LINE_SIZE = 128;
static constexpr int64_t MIN_OUTPUT_THRESHOLD = 16;
static constexpr int64_t DILATION_THRESHOLD = 1;

static constexpr int64_t NCDHW_DIMS = 5;
static constexpr int64_t NUM_64 = 64;
static constexpr int64_t INPUT_IDX_X = 0;
static constexpr int64_t KERNEL_POS = 0;
static constexpr int64_t STRIDE_POS = 1;
static constexpr int64_t PADDING_POS = 2;
static constexpr int64_t DTYPE_POS = 6;
static constexpr int64_t DILATION_POS = 3;
static constexpr int64_t CEIL_POS = 4;
static constexpr int64_t FORMAT_POS = 5;

static const int32_t MP_MAX_3D_DIM_ZERO = 0;
static const int32_t MP_MAX_3D_DIM_ONE = 1;
static const int32_t MP_MAX_3D_DIM_TWO = 2;
static const int32_t MP_MAX_3D_DIM_THREE = 3;
static const int32_t MP_MAX_3D_DIM_FOUR = 4;
static const int64_t MP_MAX_3D_TYPE_INT32 = 3;
static const int64_t MP_MAX_3D_TYPE_INT64 = 9;

static const gert::Shape g_vec_1_shape = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus MaxPool3DWithArgmaxV2GatherTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MaxPool3DWithArgmaxV2CompileInfo*>(context_->GetCompileInfo());
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
    OP_CHECK_IF(
        coreNum == 0, OP_LOGE(context_->GetNodeName(), "coreNum is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2GatherTiling::GetShapeAttrsInfo() 
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)){
        return ge::GRAPH_PARAM_INVALID;
    }
    
    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    OP_CHECK_IF(inputShape.GetDimNum() != NCDHW_DIMS,
                    OP_LOGE(context_->GetNodeName(),
                                                    "MaxPool3DWithArgmaxV2: input shape dim = %zu, should be equal 5",
                                                    inputShape.GetDimNum()),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputShape.GetShapeSize() <= 0,
                    OP_LOGE(context_->GetNodeName(),
                                                    "MaxPool3DWithArgmaxV2: input shape size %ld less than zero failed",
                                                    inputShape.GetShapeSize()),
                    return ge::GRAPH_FAILED);
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc); 
    dtype = inputDesc->GetDataType(); 
    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName(), "MaxPool3DWithArgmaxV2: invalid dtype");
        return ge::GRAPH_FAILED;
    }

    auto outX = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = EnsureNotScalar(outX->GetStorageShape());
    auto indicesX = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesX);
    auto indicesShape = EnsureNotScalar(indicesX->GetStorageShape());
    if (indicesShape != outShape) {
        OP_LOGE(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: indices shape and values shape is different");
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
        inputData.batches = inputShape.GetDim(MP_MAX_3D_DIM_ZERO) * inputShape.GetDim(MP_MAX_3D_DIM_ONE);
        inputData.nInput = inputShape.GetDim(MP_MAX_3D_DIM_ZERO);
        inputData.cInput = inputShape.GetDim(MP_MAX_3D_DIM_ONE);
    } else {
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(outShape.GetDim(d_dim) < 1 || outShape.GetDim(h_dim) < 1 || outShape.GetDim(w_dim) < 1 ,
                    OP_LOGE(context_->GetNodeName(),
                                                    "MaxPool3DWithArgmaxV2: output shape [%ld, %ld, %ld] not support",
                                                    outShape.GetDim(d_dim), outShape.GetDim(h_dim), outShape.GetDim(w_dim)),
                    return ge::GRAPH_FAILED);

    inputData.inputShape =
        array<uint64_t, DHW_DIMS>{uint64_t(inputShape.GetDim(d_dim)), uint64_t(inputShape.GetDim(h_dim)), uint64_t(inputShape.GetDim(w_dim))};
    inputData.outShape = 
        array<uint64_t, DHW_DIMS>{uint64_t(outShape.GetDim(d_dim)), uint64_t(outShape.GetDim(h_dim)), uint64_t(outShape.GetDim(w_dim))};
    int32_t dValue = 0;
    int32_t hValue = 0;
    int32_t wValue = 0;
    const gert::TypedContinuousVector<int64_t>* kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
    dValue = *(kernelSize->GetData());
    hValue = *(kernelSize->GetData() + 1);
    wValue = *(kernelSize->GetData() + 2);
    inputData.kernelSize = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        dValue <= 0 || hValue <= 0 || wValue <= 0,
        OP_LOGE(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: not support kernel shape [%d,%d, %d]", dValue, hValue, wValue),
        return ge::GRAPH_FAILED);

    int32_t kdValue = dValue;
    int32_t khValue = hValue;
    int32_t kwValue = wValue;
    const gert::TypedContinuousVector<int64_t>* stride = runtimeAttrs->GetListInt(STRIDE_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stride);
    dValue = *(stride->GetData());
    hValue = *(stride->GetData() + 1);
    wValue = *(stride->GetData() + 2);
    inputData.stride = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        hValue <= 0 || wValue <= 0 || dValue <=0,
        OP_LOGE(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: not support stride shape [%d, %d, %d]", dValue, hValue, wValue),
        return ge::GRAPH_FAILED);
    
    const gert::TypedContinuousVector<int64_t>* padding = runtimeAttrs->GetListInt(PADDING_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padding);
    dValue = *(padding->GetData());
    hValue = *(padding->GetData() + 1);
    wValue = *(padding->GetData() + 2);
    inputData.pad = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
    OP_CHECK_IF(
        hValue > khValue / 2 || wValue > kwValue / 2 || dValue > kdValue / 2,
        OP_LOGE(context_->GetNodeName(),
                                        "MaxPool3DWithArgmaxV2: not support pad shape [%d, %d, %d] kernel shape [%d, %d, %d]",
                                        dValue, hValue, wValue, kdValue, khValue, kwValue),
        return ge::GRAPH_FAILED);

    inputData.dilation = array<uint64_t, DHW_DIMS>{1, 1, 1};
    dValue = 1;
    hValue = 1;
    wValue = 1;
    const gert::TypedContinuousVector<int64_t>* dilation = runtimeAttrs->GetListInt(DILATION_POS);
    if (dilation != nullptr) {
        dValue = *(dilation->GetData());
        hValue = *(dilation->GetData() + 1);
        wValue = *(dilation->GetData() + 2);
        inputData.dilation = array<uint64_t, DHW_DIMS>{uint64_t(dValue), uint64_t(hValue), uint64_t(wValue)};
        OP_CHECK_IF(
            dValue <= 0 || hValue <= 0 || wValue <= 0,
            OP_LOGE(context_->GetNodeName(),
                                            "MaxPool3DWithArgmaxV2: not support dilation shape [%d, %d, %d]", dValue, hValue, wValue),
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

void MaxPool3DWithArgmaxV2GatherTiling::InitializationVars()
{
    baseData_.inputBytes = dtype == ge::DataType::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_OR_BF16_SIZE;
    baseData_.indexBytes = inputData.indexDtype == ge::DataType::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    baseData_.availableUb = ubSize - UB_RESVERVED_SIZE;
    baseData_.totalCoreNum = coreNum;
    baseData_.coreUsedForBestPerformance = baseData_.inputBytes == FLOAT32_SIZE
                                              ? baseData_.totalCoreNum / MAX_BANDWIDTH_COEFFICIENTS
                                              : baseData_.totalCoreNum;
    baseData_.coreUsedForBestPerformance = NUM_64;
    baseData_.padFront = inputData.pad[D_DIM];
    baseData_.padTop = inputData.pad[H_DIM];
    baseData_.padLeft = inputData.pad[W_DIM];
    baseData_.dInput = inputData.inputShape[D_DIM];
    baseData_.hInput = inputData.inputShape[H_DIM];
    baseData_.wInput = inputData.inputShape[W_DIM];
    baseData_.dOutput = inputData.outShape[D_DIM];
    baseData_.hOutput = inputData.outShape[H_DIM];
    baseData_.wOutput = inputData.outShape[W_DIM];
    baseData_.dStride = inputData.stride[D_DIM];
    baseData_.hStride = inputData.stride[H_DIM];
    baseData_.wStride = inputData.stride[W_DIM];
    baseData_.dKernel = inputData.kernelSize[D_DIM];
    baseData_.hKernel = inputData.kernelSize[H_DIM];
    baseData_.wKernel = inputData.kernelSize[W_DIM];
    baseData_.highAxisTotal = inputData.batches; 
    baseData_.dDilation = inputData.dilation[D_DIM];
    baseData_.hDilation = inputData.dilation[H_DIM];
    baseData_.wDilation = inputData.dilation[W_DIM];
    baseData_.isPad = 0;
    if (baseData_.padTop != 0 || baseData_.padFront != 0 || baseData_.padLeft != 0) {
        baseData_.isPad = 1;
    }
    if (inputData.ceilMode && baseData_.isPad == 0) {
        if (((baseData_.wOutput - 1) * baseData_.wStride + baseData_.wKernel) != baseData_.wInput ||
            ((baseData_.hOutput - 1) * baseData_.hStride + baseData_.hKernel) != baseData_.hInput ||
            ((baseData_.dOutput - 1) * baseData_.dStride + baseData_.dKernel) != baseData_.dInput ) {
            baseData_.isPad = 1;
        }
    }

    baseData_.oneBlockNumT1 = Ops::Base::GetUbBlockSize(context_) / baseData_.inputBytes;
    baseData_.oneBlockNumT2 = Ops::Base::GetUbBlockSize(context_) / baseData_.indexBytes;
}

bool MaxPool3DWithArgmaxV2GatherTiling::IsCapable()
{
    if (inputData.dilation[D_DIM] > DILATION_THRESHOLD || inputData.dilation[H_DIM] > DILATION_THRESHOLD 
        || inputData.dilation[W_DIM] > DILATION_THRESHOLD ||inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;    
    }

    InitializationVars();
    if (baseData_.wKernel * baseData_.inputBytes >= CACHE_LINE_SIZE) {
        return false;
    }

    splitData_.dOutputInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = 1;
    splitData_.highAxisInner = 1;
    DoBufferCalculate();
    int64_t dRate = (baseData_.dOutput == 1 || (baseData_.dStride / baseData_.dKernel) < 1) 
                    ? 1 
                    : (baseData_.dStride / baseData_.dKernel);
    int64_t hRate = (baseData_.hOutput == 1 || (baseData_.hStride / baseData_.hKernel) < 1) 
                    ? 1 
                    : (baseData_.hStride / baseData_.hKernel);
    int64_t wRate = (baseData_.wOutput == 1 || (baseData_.wStride / baseData_.wKernel) < 1) 
                    ? 1 
                    : (baseData_.wStride / baseData_.wKernel);
    return splitData_.totalBufferSize <= baseData_.availableUb / (MIN_OUTPUT_THRESHOLD * dRate * hRate * wRate);    
}

uint64_t MaxPool3DWithArgmaxV2GatherTiling::GetTilingKey() const
{
    uint64_t tilingKey = NO_PADDING_TILING_KEY;
    if (baseData_.isPad == 1) {
        tilingKey = PADDING_TILING_KEY;
    }
    return tilingKey;
}

void MaxPool3DWithArgmaxV2GatherTiling::DoBufferCalculate()
{
    splitData_.dInputInner =
        (splitData_.dOutputInner - 1) * baseData_.dStride + (baseData_.dKernel - 1) * baseData_.dDilation + 1;
    splitData_.hInputInner =
        (splitData_.hOutputInner - 1) * baseData_.hStride + (baseData_.hKernel - 1) * baseData_.hDilation + 1;
    splitData_.wInputInner =
        (splitData_.wOutputInner - 1) * baseData_.wStride + (baseData_.wKernel - 1) * baseData_.wDilation + 1;
    int64_t maxDataNumInOneBlock = std::max(baseData_.oneBlockNumT1, baseData_.oneBlockNumT2);
    int64_t wInputInnerAligned = Ops::Base::CeilAlign(splitData_.wInputInner, baseData_.oneBlockNumT1);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData_.wOutputInner, maxDataNumInOneBlock);
    int64_t inputBufferSize = 
        splitData_.highAxisInner  * splitData_.dInputInner * splitData_.hInputInner * wInputInnerAligned * baseData_.inputBytes;
    splitData_.inputBufferSize = inputBufferSize;

    if (baseData_.isPad == 1) {
        inputBufferSize *= DOUBLE; 
    }
    int64_t outputDataSize = splitData_.highAxisInner * splitData_.dOutputInner * splitData_.hOutputInner * wOutputInnerAligned;
    splitData_.maxValueBufferSize = outputDataSize * baseData_.inputBytes;
    splitData_.argmaxBufferSize = outputDataSize * baseData_.indexBytes;

    int64_t tmpTotalBufferSize =
        inputBufferSize + splitData_.maxValueBufferSize + splitData_.argmaxBufferSize + HELPER_BUFFER_SIZE;
    
    splitData_.totalBufferSize = tmpTotalBufferSize * DOUBLE;
    if (baseData_.isPad == 1) {
        splitData_.totalBufferSize -= splitData_.inputBufferSize;
    }
}

bool MaxPool3DWithArgmaxV2GatherTiling::IsMeetTargetCoreNum() const
{  
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(baseData_.wOutput, splitData_.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(baseData_.hOutput, splitData_.hOutputInner);
    int64_t tmpDOutputOuter = Ops::Base::CeilDiv(baseData_.dOutput, splitData_.dOutputInner);
    int64_t tmpNCOutputOuter = Ops::Base::CeilDiv(baseData_.highAxisTotal, splitData_.highAxisInner);
    return tmpWOutputOuter * tmpHOutputOuter * tmpDOutputOuter * tmpNCOutputOuter  >= baseData_.coreUsedForBestPerformance;
}

bool MaxPool3DWithArgmaxV2GatherTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    return splitData_.totalBufferSize <= baseData_.availableUb;
}

void MaxPool3DWithArgmaxV2GatherTiling::BinarySearch(int64_t start, int64_t end, int64_t* value)
{
    int64_t left = start;
    int64_t right = end;
    int64_t bestSplit = 1;

    while (left <= right) {
        int64_t mid = left + (right - left) / DOUBLE;
        *value = mid;
        if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
            bestSplit = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    *value = bestSplit;
}

bool MaxPool3DWithArgmaxV2GatherTiling::TrySplitNC()
{
    splitData_.dOutputInner = baseData_.dOutput; 
    splitData_.hOutputInner = baseData_.hOutput; 
    splitData_.wOutputInner = baseData_.wOutput;
    splitData_.highAxisInner = Ops::Base::CeilDiv(baseData_.highAxisTotal, baseData_.coreUsedForBestPerformance);
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        return true;
    }
    splitData_.highAxisInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.highAxisTotal, &splitData_.highAxisInner);
        return true;
    } else {
        return false;
    }
}

bool MaxPool3DWithArgmaxV2GatherTiling::TrySplitD()
{
    splitData_.highAxisInner = 1;
    splitData_.dOutputInner = 1;
    splitData_.hOutputInner = baseData_.hOutput;
    splitData_.wOutputInner = baseData_.wOutput;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.dOutput, &splitData_.dOutputInner);
        return true;
    } else {
        return false;
    }
}

bool MaxPool3DWithArgmaxV2GatherTiling::TrySplitH()
{
    splitData_.highAxisInner = 1;
    splitData_.dOutputInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = baseData_.wOutput;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.hOutput, &splitData_.hOutputInner);
        return true;
    } else {
        return false;
    }
}

bool MaxPool3DWithArgmaxV2GatherTiling::TrySplitW()
{
    splitData_.highAxisInner = 1;
    splitData_.dOutputInner = 1;
    splitData_.hOutputInner = 1;
    splitData_.wOutputInner = 1;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        BinarySearch(1, baseData_.wOutput, &splitData_.wOutputInner);
        return true;
    } else {
        return false;
    }
}

void MaxPool3DWithArgmaxV2GatherTiling::SearchBestTiling()
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

void MaxPool3DWithArgmaxV2GatherTiling::DoUBTiling()
{
    SearchBestTiling();
    DoBufferCalculate();
    splitData_.wOutputOuter = Ops::Base::CeilDiv(baseData_.wOutput, splitData_.wOutputInner);
    int64_t tempWOutputTail = baseData_.wOutput % splitData_.wOutputInner;
    splitData_.wOutputTail = tempWOutputTail == 0 ? splitData_.wOutputInner : tempWOutputTail;

    splitData_.hOutputOuter = Ops::Base::CeilDiv(baseData_.hOutput, splitData_.hOutputInner);
    int64_t tempHOutputTail = baseData_.hOutput % splitData_.hOutputInner;
    splitData_.hOutputTail = tempHOutputTail == 0 ? splitData_.hOutputInner : tempHOutputTail;

    splitData_.dOutputOuter = Ops::Base::CeilDiv(baseData_.dOutput, splitData_.dOutputInner);
    int64_t tempDOutputTail = baseData_.dOutput % splitData_.dOutputInner;
    splitData_.dOutputTail = tempDOutputTail == 0 ? splitData_.dOutputInner : tempDOutputTail;

    splitData_.highAxisOuter = Ops::Base::CeilDiv(baseData_.highAxisTotal, splitData_.highAxisInner);
    int64_t tempNOutputTail = baseData_.highAxisTotal % splitData_.highAxisInner;
    splitData_.highAxisTail = tempNOutputTail == 0 ? splitData_.highAxisInner : tempNOutputTail;
}

void MaxPool3DWithArgmaxV2GatherTiling::DoBlockTiling()
{
    splitData_.totalBaseBlockNum = splitData_.highAxisOuter * splitData_.dOutputOuter * splitData_.hOutputOuter * splitData_.wOutputOuter ;
    splitData_.normalCoreProcessNum = Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, baseData_.totalCoreNum);
    splitData_.usedCoreNum = Ops::Base::CeilDiv(splitData_.totalBaseBlockNum, splitData_.normalCoreProcessNum);
    splitData_.tailCoreProcessNum =
        splitData_.totalBaseBlockNum - splitData_.normalCoreProcessNum * (splitData_.usedCoreNum - 1);
}

void MaxPool3DWithArgmaxV2GatherTiling::PrintBaseData() const
{
    OP_LOGI("PrintBaseData", "%s", baseData_.ToString().c_str());
}

void MaxPool3DWithArgmaxV2GatherTiling::PrintSplitData() const
{
    OP_LOGI("PrintSplitData", "%s", splitData_.ToString().c_str());
}

void MaxPool3DWithArgmaxV2GatherTiling::SetTilingData()
{
    tilingData_->dInput = baseData_.dInput;
    tilingData_->hInput = baseData_.hInput;
    tilingData_->wInput = baseData_.wInput;
    tilingData_->dOutput = baseData_.dOutput;
    tilingData_->hOutput = baseData_.hOutput;
    tilingData_->wOutput = baseData_.wOutput;
    tilingData_->dKernel = baseData_.dKernel;
    tilingData_->hKernel = baseData_.hKernel;
    tilingData_->wKernel = baseData_.wKernel;
    tilingData_->dStride = baseData_.dStride;
    tilingData_->hStride = baseData_.hStride;
    tilingData_->wStride = baseData_.wStride;
    tilingData_->padFront = baseData_.padFront;
    tilingData_->padTop = baseData_.padTop;
    tilingData_->padLeft = baseData_.padLeft;
    tilingData_->highAxisInner = splitData_.highAxisInner;
    tilingData_->highAxisTail = splitData_.highAxisTail;
    tilingData_->highAxisOuter = splitData_.highAxisOuter;
    tilingData_->dOutputInner = splitData_.dOutputInner;
    tilingData_->dOutputTail = splitData_.dOutputTail;
    tilingData_->dOutputOuter = splitData_.dOutputOuter;
    tilingData_->hOutputInner = splitData_.hOutputInner;
    tilingData_->hOutputTail = splitData_.hOutputTail;
    tilingData_->hOutputOuter = splitData_.hOutputOuter;
    tilingData_->wOutputInner = splitData_.wOutputInner;
    tilingData_->wOutputTail = splitData_.wOutputTail;
    tilingData_->wOutputOuter = splitData_.wOutputOuter;
    tilingData_->normalCoreProcessNum = splitData_.normalCoreProcessNum;
    tilingData_->tailCoreProcessNum = splitData_.tailCoreProcessNum;
    tilingData_->usedCoreNum = splitData_.usedCoreNum;
    tilingData_->inputBufferSize = splitData_.inputBufferSize;
    tilingData_->maxValueBufferSize = splitData_.maxValueBufferSize;
    tilingData_->argmaxBufferSize = splitData_.argmaxBufferSize;
    tilingData_->isPad = baseData_.isPad;
    tilingData_->dDilation = baseData_.dDilation;
    tilingData_->hDilation = baseData_.hDilation;
    tilingData_->wDilation = baseData_.wDilation;
}

ge::graphStatus MaxPool3DWithArgmaxV2GatherTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    PrintBaseData();
    PrintSplitData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2GatherTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_->usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2GatherTiling, 0);

}  // namespace optiling
