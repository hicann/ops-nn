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
 * \file sparse_fill_empty_rows_infershape.cpp
 * \brief
 */

#include <string>

#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kIndicesIdx = 0U;
constexpr size_t kValuesIdx = 1U;
constexpr size_t kDenseShapeIdx = 2U;
constexpr size_t kDefaultValueIdx = 3U;
constexpr size_t kYIndicesIdx = 0U;
constexpr size_t kYValuesIdx = 1U;
constexpr size_t kEmptyRowIdx = 2U;
constexpr size_t kReverseIdxMap = 3U;
constexpr size_t kIndicesRank = 2U;
constexpr size_t kValuesRank = 1U;
constexpr int64_t kMinNonEmptyElements = 1;
} // namespace

static ge::graphStatus CheckInputDtype(gert::InferShapeRangeContext* context, size_t index, ge::DataType expectType,
                                       const char* inputName)
{
    const gert::CompileTimeTensorDesc* inputDesc = context->GetInputDesc(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    if (inputDesc->GetDataType() != expectType) {
        const std::string actualType = ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType());
        const std::string expectedType = ge::TypeUtils::DataTypeToSerialString(expectType);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), inputName, actualType.c_str(), expectedType.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSparseFillEmptyRowsInputs(gert::InferShapeRangeContext* context)
{
    if (CheckInputDtype(context, kIndicesIdx, ge::DT_INT64, "indices") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDtype(context, kDenseShapeIdx, ge::DT_INT64, "dense_shape") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::CompileTimeTensorDesc* valuesDesc = context->GetInputDesc(kValuesIdx);
    const gert::CompileTimeTensorDesc* defaultValueDesc = context->GetInputDesc(kDefaultValueIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, valuesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, defaultValueDesc);
    if (valuesDesc->GetDataType() != defaultValueDesc->GetDataType()) {
        const std::string actualTypes = ge::TypeUtils::DataTypeToSerialString(valuesDesc->GetDataType()) + ", " +
                                        ge::TypeUtils::DataTypeToSerialString(defaultValueDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "values, default_value", actualTypes.c_str(),
                                               "values and default_value must have the same dtype");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRangeSparseFillEmptyRows(gert::InferShapeRangeContext* context)
{
    if (CheckSparseFillEmptyRowsInputs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto indicesShapeRange = context->GetInputShapeRange(kIndicesIdx);
    auto denseShapeTensorRange = context->GetInputTensorRange(kDenseShapeIdx);
    auto yIndicesShapeRange = context->GetOutputShapeRange(kYIndicesIdx);
    auto yValuesShapeRange = context->GetOutputShapeRange(kYValuesIdx);
    auto yEmptyRowIndicatorRange = context->GetOutputShapeRange(kEmptyRowIdx);
    auto yReverseIndexMapRange = context->GetOutputShapeRange(kReverseIdxMap);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapeRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, denseShapeTensorRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, yIndicesShapeRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, yValuesShapeRange);
    OP_CHECK_NULL_WITH_CONTEXT(context, denseShapeTensorRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapeRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShapeRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, yIndicesShapeRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, yIndicesShapeRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, yValuesShapeRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, yValuesShapeRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, yEmptyRowIndicatorRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, yEmptyRowIndicatorRange->GetMin());
    OP_CHECK_NULL_WITH_CONTEXT(context, yReverseIndexMapRange->GetMax());
    OP_CHECK_NULL_WITH_CONTEXT(context, yReverseIndexMapRange->GetMin());

    const gert::Shape* indicesMaxShape = indicesShapeRange->GetMax();
    if (indicesMaxShape->GetDimNum() != kIndicesRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "indices",
                                     std::to_string(indicesMaxShape->GetDimNum()).c_str(), "2");
        return ge::GRAPH_FAILED;
    }

    auto denseShapeTensor = denseShapeTensorRange->GetMax();
    auto shapeData = denseShapeTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeData);
    auto shapeSize = denseShapeTensor->GetShapeSize();
    if (denseShapeTensor->GetOriginShape().GetDimNum() != kValuesRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "dense_shape",
                                     std::to_string(denseShapeTensor->GetOriginShape().GetDimNum()).c_str(), "1");
        return ge::GRAPH_FAILED;
    }
    if (shapeSize <= 0 || static_cast<size_t>(shapeSize) > gert::Shape::kMaxDimNum) {
        const std::string reason = "the number of elements must be in range [1, " +
                                   std::to_string(gert::Shape::kMaxDimNum) + "]";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "dense_shape",
                                                  std::to_string(shapeSize).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    gert::Shape denseShape;
    denseShape.SetDimNum(static_cast<size_t>(shapeSize));
    for (size_t i = 0U; i < static_cast<size_t>(shapeSize); ++i) {
        if (shapeData[i] < 0) {
            const std::string paramName = "dense_shape[" + std::to_string(i) + "]";
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), paramName.c_str(), std::to_string(shapeData[i]).c_str(),
                                      ">= 0");
            return ge::GRAPH_FAILED;
        }
        denseShape.SetDim(i, shapeData[i]);
    }

    auto elementsNum = denseShape.GetShapeSize();
    if (elementsNum == gert::Shape::kInvalidDimValue) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "dense_shape", Shape2String(denseShape).c_str(),
                                              "the product of its elements overflows int64");
        return ge::GRAPH_FAILED;
    }
    const int64_t minElementsNum = (elementsNum == 0) ? 0 : kMinNonEmptyElements;
    const int64_t indicesMinNum = indicesShapeRange->GetMin()->GetDim(0);
    const int64_t indicesMaxNum = indicesMaxShape->GetDim(0);
    const int64_t indicesRankDim = indicesMaxShape->GetDim(1);

    yIndicesShapeRange->GetMax()->SetDimNum(kIndicesRank);
    yIndicesShapeRange->GetMax()->SetDim(0, elementsNum);
    yIndicesShapeRange->GetMax()->SetDim(1, indicesRankDim);
    yIndicesShapeRange->GetMin()->SetDimNum(kIndicesRank);
    yIndicesShapeRange->GetMin()->SetDim(0, minElementsNum);
    yIndicesShapeRange->GetMin()->SetDim(1, indicesRankDim);

    yValuesShapeRange->GetMax()->SetDimNum(kValuesRank);
    yValuesShapeRange->GetMax()->SetDim(0, elementsNum);
    yValuesShapeRange->GetMin()->SetDimNum(kValuesRank);
    yValuesShapeRange->GetMin()->SetDim(0, minElementsNum);

    yEmptyRowIndicatorRange->GetMax()->SetDimNum(kValuesRank);
    yEmptyRowIndicatorRange->GetMax()->SetDim(0, denseShape.GetDim(0));
    yEmptyRowIndicatorRange->GetMin()->SetDimNum(kValuesRank);
    yEmptyRowIndicatorRange->GetMin()->SetDim(0, denseShape.GetDim(0));

    yReverseIndexMapRange->GetMax()->SetDimNum(kValuesRank);
    yReverseIndexMapRange->GetMax()->SetDim(0, indicesMaxNum);
    yReverseIndexMapRange->GetMin()->SetDimNum(kValuesRank);
    yReverseIndexMapRange->GetMin()->SetDim(0, indicesMinNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFillEmptyRows)
    .InferShapeRange(InferShapeRangeSparseFillEmptyRows)
    .InputsDataDependency({kDenseShapeIdx});
} // namespace ops
