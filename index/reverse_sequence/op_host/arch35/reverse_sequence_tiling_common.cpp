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
 * \file reverse_sequence_tiling_common.cpp
 * \brief
 */

#include "reverse_sequence_tiling_common.h"
#include "reverse_sequence_tiling.h"

using namespace AscendC;
using namespace ge;

 
namespace optiling
{
static constexpr int64_t BATCH_AXIS_IDX = 1;  
static constexpr int64_t SEQ_AXIS_IDX = 0;
static constexpr int64_t X_IDX = 0;
static constexpr int64_t SEQ_LEN_IDX = 1;
static constexpr int64_t X_MIN_DIM_CNT = 2;
static constexpr int64_t SEQ_LENGTHS_DIM_CNT = 1;
static constexpr int64_t Y_INDEX = 0;
static constexpr int64_t ONE_DIM = 1;
static constexpr int64_t TWO_DIM = 2;
static constexpr int64_t THREE_DIM = 3;
static constexpr int64_t TYPE_SB = 0;
static constexpr int64_t TYPE_BS = 1;
static constexpr int64_t TYPE_BSA = 2;
static constexpr int64_t TYPE_ABS = 3;
static constexpr int64_t TYPE_SBA = 4;
static constexpr int64_t TYPE_ASB = 5;
static constexpr int64_t TYPE_SAB = 6;
static constexpr int64_t TYPE_BAS = 7;
static constexpr int64_t TYPE_OTHER = 8;

static const std::set<ge::DataType> X_DTYPES = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8,
                                                   ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_INT64,
                                                   ge::DT_BOOL, ge::DT_DOUBLE, ge::DT_COMPLEX64};
static const std::set<ge::DataType> SEQ_LENGTHS_DTYPES = {ge::DT_INT32, ge::DT_INT64};

static ge::graphStatus CheckDTypeParams(gert::TilingContext* context, ReverseInputInfo& inputData)
{
    OP_LOGD("CheckDTypeParams begin");
    auto xDescPtr = context->GetInputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDescPtr);
    ge::DataType xDtype = xDescPtr->GetDataType();
    OP_CHECK_IF(X_DTYPES.find(xDtype) == X_DTYPES.end(),
        OP_LOGE(context->GetNodeName(), "Input x dtype not supports %d", static_cast<int32_t>(xDtype)), return ge::GRAPH_FAILED);
    
    inputData.xDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_CHECK_IF(inputData.xDtypeSize <= 0, 
        OP_LOGE(context->GetNodeName(), "Get xDtypeSize[%ld] failed.", inputData.xDtypeSize), return ge::GRAPH_FAILED);
    
    auto seqLengthsDescPtr = context->GetInputDesc(SEQ_LEN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqLengthsDescPtr);
    auto seqLengthsDtype = seqLengthsDescPtr->GetDataType();
    OP_CHECK_IF(SEQ_LENGTHS_DTYPES.find(seqLengthsDtype) == SEQ_LENGTHS_DTYPES.end(),
        OP_LOGE(context->GetNodeName(), "Input seqLengths dtype not supports %d.", static_cast<int32_t>(seqLengthsDtype)), return ge::GRAPH_FAILED);
    
    auto yDescPtr = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDescPtr);
    ge::DataType yDtype = yDescPtr->GetDataType();
    OP_CHECK_IF(yDtype != xDtype, 
        OP_LOGE(context->GetNodeName(), "The dtype of y and x must be the same."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static void ComputeCombineType(int64_t rank, ReverseInputInfo& inputData)
{
    if (rank == TWO_DIM) {
        if (inputData.seqAxis == 0 && inputData.batchAxis == 1) {
            inputData.comBineType = TYPE_SB; // SB
        } else {
            inputData.comBineType = TYPE_BS; // BS
        }
    } else if (rank == THREE_DIM) {
        if (inputData.batchAxis == inputData.seqAxis - 1) {
           if (inputData.batchAxis == 0) {
                inputData.comBineType = TYPE_BSA; // BSA
            } else {
                inputData.comBineType = TYPE_ABS; // ABS
            }
        } else if (inputData.batchAxis == inputData.seqAxis + 1) {
            if (inputData.seqAxis == 0) {
                inputData.comBineType = TYPE_SBA; // SBA
            } else {
                inputData.comBineType = TYPE_ASB; // ASB
            }
        } else if (inputData.seqAxis > inputData.batchAxis) {
            inputData.comBineType = TYPE_SAB; // SAB
        } else {
            inputData.comBineType = TYPE_BAS; // BAS
        }
    } else {
        inputData.comBineType = TYPE_OTHER;
    }
}

static void ComputeComBineAxis(const gert::Shape& xShape, int64_t rank, ReverseInputInfo& inputData)
{
    // 合轴，A0 S/B A1 S/B A2
    int64_t firstAxis = std::min(inputData.seqAxis, inputData.batchAxis);
    int64_t secondAxis = std::max(inputData.seqAxis, inputData.batchAxis);

    int64_t i = 0;
    int64_t j = 0;
    if (firstAxis != 0) {
        int64_t firstDim = 1;
        for (; i < firstAxis; i++) {
            firstDim *= xShape.GetDim(i);
        }
        inputData.inputDim[j++] = firstDim;
        inputData.comBineDims++;
    }
    inputData.inputDim[j++] = xShape.GetDim(i++);
    inputData.comBineDims++;
    
    if (secondAxis == i) {
        inputData.inputDim[j++] = xShape.GetDim(i++);
        inputData.comBineDims++;
    } else {
        int64_t secondDim = 1;
        for (; i < secondAxis; ++i) {
            secondDim *= xShape.GetDim(i);
        }
        inputData.inputDim[j++] = secondDim;
        inputData.comBineDims++;
        inputData.inputDim[j++] = xShape.GetDim(i++);
        inputData.comBineDims++;
    }

    if (secondAxis != rank - 1) {
        int64_t afterDim = 1;
        for (; i < rank; ++i) {
            afterDim *= xShape.GetDim(i);
        }
        inputData.inputDim[j] = afterDim;
        inputData.comBineDims++;
    }
}

static void ComputeAfterAxisSize(const gert::Shape& xShape, int64_t rank, ReverseInputInfo& inputData)
{
    for (int64_t i = inputData.seqAxis + 1; i < rank; ++i) {
        inputData.reverseSize *= xShape.GetDim(i);
    }

    for (int64_t i = inputData.batchAxis + 1; i < rank; ++i) {
        inputData.batchSize *= xShape.GetDim(i);
    }
}

static ge::graphStatus CheckAttrParams(gert::TilingContext* context, ReverseInputInfo& inputData)
{
    OP_LOGD("CheckAttrParams begin");
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto* seqAxisPtr = attrs->GetAttrPointer<int64_t>(SEQ_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqAxisPtr);
    inputData.seqAxis = static_cast<int64_t>(*seqAxisPtr);

    const auto* batchAxisPtr = attrs->GetAttrPointer<int64_t>(BATCH_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, batchAxisPtr);
    inputData.batchAxis = static_cast<int64_t>(*batchAxisPtr);
    OP_LOGD(context, "CheckShapeParams inputData.batchAxis=%ld", inputData.batchAxis);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeParams(gert::TilingContext* context, ReverseInputInfo& inputData)
{
    OP_LOGD("CheckShapeParams begin");
    auto xShapePtr = context->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    inputData.xShapeSize = xShape.GetShapeSize();
    int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(rank < X_MIN_DIM_CNT,
        OP_LOGE(context->GetNodeName(), "Input x dim count[%ld] must >= %ld.", rank, X_MIN_DIM_CNT), return ge::GRAPH_FAILED);
    
    inputData.seqAxis = inputData.seqAxis < 0 ? inputData.seqAxis + rank : inputData.seqAxis;
    inputData.batchAxis = inputData.batchAxis < 0 ? inputData.batchAxis + rank : inputData.batchAxis;
    OP_LOGD(context, "CheckShapeParams inputData.batchAxis=%ld", inputData.batchAxis);
    
    OP_CHECK_IF(inputData.seqAxis == inputData.batchAxis,
        OP_LOGE(context->GetNodeName(), "batchAxis == seqAxis == %ld.", inputData.seqAxis), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputData.batchAxis < 0 || inputData.batchAxis >= rank,
        OP_LOGE(context->GetNodeName(), "Invalid batchAxis: %ld.", inputData.batchAxis), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputData.seqAxis < 0 || inputData.seqAxis >= rank,
        OP_LOGE(context->GetNodeName(), "Invalid seqAxis: %ld.", inputData.seqAxis), return ge::GRAPH_FAILED);
    
    inputData.batchDim = xShape.GetDim(inputData.batchAxis);
    inputData.seqDim = xShape.GetDim(inputData.seqAxis);

    auto seqLenShapePtr = context->GetInputShape(SEQ_LEN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqLenShapePtr);
    auto seqLenShape = seqLenShapePtr->GetStorageShape();
    OP_CHECK_IF(seqLenShape.GetDimNum() != SEQ_LENGTHS_DIM_CNT,
        OP_LOGE(context->GetNodeName(), "seq_length must be 1-dim, not:%d.", static_cast<int16_t>(seqLenShape.GetDimNum())), return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(static_cast<int64_t>(seqLenShape.GetDim(0)) != inputData.batchDim,
        OP_LOGE(context->GetNodeName(), "Length of seq_length(%ld) != input.dims(batchAxis):%ld.",
         static_cast<int64_t>(seqLenShape.GetDim(0)), inputData.batchDim), return ge::GRAPH_FAILED);

    OP_CHECK_IF(xShape.GetShapeSize() == 0 || seqLenShape.GetShapeSize() == 0,
        OP_LOGE(context->GetNodeName(), "Input x or seqLengths not support empty tensor."), return ge::GRAPH_FAILED);

    auto yShapePtr = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(yShape != xShape, 
        OP_LOGE(context->GetNodeName(), "The shape of y and x must be the same."), return ge::GRAPH_FAILED);

    ComputeAfterAxisSize(xShape, rank, inputData);
    ComputeComBineAxis(xShape, rank, inputData);
    ComputeCombineType(rank, inputData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetReverseSequenceShapeAttrsInfo(gert::TilingContext* context, ReverseInputInfo& inputData)
{
    OP_LOGD("GetReverseSequenceShapeAttrsInfo begin");
    OP_CHECK_IF(CheckDTypeParams(context, inputData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CheckDTypeParams failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckAttrParams(context, inputData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CheckAttrParams failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShapeParams(context, inputData) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CheckShapeParams failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetReverseSequencePlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint64_t& coreNum)
{
    OP_LOGD("GetReverseSequencePlatformInfo begin");
    auto compileInfo = reinterpret_cast<const ReverseSequenceCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    coreNum = compileInfo->coreNum;
    ubSize = compileInfo->ubSize;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling