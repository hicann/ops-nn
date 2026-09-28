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
 * \file wts_arq_infershape.cpp
 * \brief WtsARQ shape/dtype inference.
 *
 * Rules (spec.yaml shape_constraints):
 *   - rank(w_min) == rank(w_max) == rank(w), rank <= 8
 *   - shape(w_min) == shape(w_max)
 *   - each dim of w_min/w_max is 1 or equals the w dim (restricted broadcast)
 *   - y.shape = w.shape, y.dtype = w.dtype
 *   - total elements of w <= 2^31 (SHAPE_SIZE_LIMIT, checked for static shapes)
 *   - attr num_bits must be 8
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"

#include <sstream>
#include <string>

using namespace ge;

namespace ops {

static constexpr size_t INPUT_W_IDX = 0;
static constexpr size_t INPUT_W_MIN_IDX = 1;
static constexpr size_t INPUT_W_MAX_IDX = 2;
static constexpr size_t OUTPUT_Y_IDX = 0;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2;
static constexpr int64_t MAX_SUPPORTED_RANK = 8;
static constexpr int64_t SHAPE_SIZE_LIMIT = 1LL << 31;
static constexpr size_t ATTR_NUM_BITS_IDX = 0;
static constexpr int64_t SUPPORTED_NUM_BITS = 8;

static ge::graphStatus ValidateNumBits(const char* nodeName, const gert::RuntimeAttrs* attrs)
{
    if (attrs == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t* numBitsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_BITS_IDX);
    const int64_t numBits = (numBitsPtr != nullptr) ? *numBitsPtr : SUPPORTED_NUM_BITS;
    if (numBits != SUPPORTED_NUM_BITS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "num_bits", std::to_string(numBits).c_str(),
                                              "The value of num_bits must be 8");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static bool IsUnknownRank(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}

static bool HasUnknownDim(const gert::Shape& shape)
{
    if (IsUnknownRank(shape)) {
        return true;
    }
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

static std::string ShapeToString(const gert::Shape& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << shape.GetDim(i);
    }
    oss << "]";
    return oss.str();
}

static ge::graphStatus InferShapeForWtsARQ(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to infer shape for WtsARQ.");
    if (ValidateNumBits(context->GetNodeName(), context->GetAttrs()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* wShape = context->GetInputShape(INPUT_W_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    const gert::Shape* wMinShape = context->GetInputShape(INPUT_W_MIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wMinShape);
    const gert::Shape* wMaxShape = context->GetInputShape(INPUT_W_MAX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wMaxShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // Rank/shape relations are checked on everything that is decidable: an unknown
    // rank (-2) or unknown dim (-1) only defers the relations it participates in.
    // The previous early return on any unknown rank skipped checks that are fully
    // decidable from the remaining inputs (e.g. w=[-2] with w_min=[2,2]/w_max=[3,3],
    // or a known rank-9 w with an unknown-rank sibling).
    const gert::Shape* shapes[3] = {wShape, wMinShape, wMaxShape};
    const char* shapeNames[3] = {"w", "w_min", "w_max"};
    bool unknownRank[3] = {false, false, false};
    size_t ranks[3] = {0, 0, 0};
    for (size_t i = 0; i < 3; ++i) {
        unknownRank[i] = IsUnknownRank(*shapes[i]);
        ranks[i] = shapes[i]->GetDimNum();
    }

    // 1) per-input rank limit for every input whose rank is known
    for (size_t i = 0; i < 3; ++i) {
        if (!unknownRank[i] && ranks[i] > static_cast<size_t>(MAX_SUPPORTED_RANK)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), shapeNames[i],
                                                     std::to_string(ranks[i]).c_str(), "The rank of w must be <= 8");
            return ge::GRAPH_FAILED;
        }
    }
    // 2) rank equality across every pair whose ranks are both known
    int firstKnown = -1;
    for (size_t i = 0; i < 3; ++i) {
        if (unknownRank[i]) {
            continue;
        }
        if (firstKnown < 0) {
            firstKnown = static_cast<int>(i);
            continue;
        }
        if (ranks[i] != ranks[static_cast<size_t>(firstKnown)]) {
            const std::string rankStr = std::string(shapeNames[firstKnown]) + "=" +
                                        std::to_string(ranks[static_cast<size_t>(firstKnown)]) + ", " +
                                        std::string(shapeNames[i]) + "=" + std::to_string(ranks[i]);
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "w, w_min, w_max", rankStr.c_str(),
                                                      "The rank of w_min and w_max must be same as w");
            return ge::GRAPH_FAILED;
        }
    }

    // 3) w_min vs w_max shape equality on axes known on both sides
    if (!unknownRank[1] && !unknownRank[2]) {
        for (size_t i = 0; i < ranks[1]; ++i) {
            const int64_t minDim = wMinShape->GetDim(i);
            const int64_t maxDim = wMaxShape->GetDim(i);
            if (minDim >= 0 && maxDim >= 0 && minDim != maxDim) {
                const std::string shapeStr = ShapeToString(*wMinShape) + ", " + ShapeToString(*wMaxShape);
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "w_min, w_max", shapeStr.c_str(),
                                                       "The shape of w_min must be same as w_max");
                return ge::GRAPH_FAILED;
            }
        }
    }
    // 4) restricted broadcast against w on axes known on both sides
    if (!unknownRank[0]) {
        for (size_t i = 0; i < ranks[0]; ++i) {
            const int64_t wDim = wShape->GetDim(i);
            if (wDim < 0) {
                continue;
            }
            if (!unknownRank[1]) {
                const int64_t minDim = wMinShape->GetDim(i);
                if (minDim >= 0 && minDim != wDim && minDim != 1) {
                    const std::string shapeStr = ShapeToString(*wShape) + ", " + ShapeToString(*wMinShape);
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "w, w_min", shapeStr.c_str(),
                        "Each dim of w_min and w_max must be the same as w or equal to 1");
                    return ge::GRAPH_FAILED;
                }
            }
            if (!unknownRank[2]) {
                const int64_t maxDim = wMaxShape->GetDim(i);
                if (maxDim >= 0 && maxDim != wDim && maxDim != 1) {
                    const std::string shapeStr = ShapeToString(*wShape) + ", " + ShapeToString(*wMaxShape);
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "w, w_max", shapeStr.c_str(),
                        "Each dim of w_min and w_max must be the same as w or equal to 1");
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    // 5) SHAPE_SIZE_LIMIT check only when w is fully static
    if (!unknownRank[0] && !HasUnknownDim(*wShape)) {
        int64_t wElems = 1;
        for (size_t i = 0; i < ranks[0]; ++i) {
            const int64_t dim = wShape->GetDim(i);
            if (dim == 0) {
                wElems = 0;
                break;
            }
            if (wElems > SHAPE_SIZE_LIMIT / dim) {
                wElems = SHAPE_SIZE_LIMIT + 1;
                break;
            }
            wElems *= dim;
        }
        if (wElems > SHAPE_SIZE_LIMIT) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "w", std::to_string(wElems).c_str(),
                                                      "The shape size of w must be smaller than or equal to 2^31");
            return ge::GRAPH_FAILED;
        }
    }

    *yShape = *wShape;
    OP_LOGD(context->GetNodeName(), "End to infer shape for WtsARQ.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForWtsARQ(gert::InferDataTypeContext* context)
{
    if (ValidateNumBits(context->GetNodeName(), context->GetAttrs()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType wDtype = context->GetInputDataType(INPUT_W_IDX);
    const ge::DataType wMinDtype = context->GetInputDataType(INPUT_W_MIN_IDX);
    const ge::DataType wMaxDtype = context->GetInputDataType(INPUT_W_MAX_IDX);

    if (wDtype != ge::DT_FLOAT16 && wDtype != ge::DT_FLOAT) {
        const std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(wDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "w", dtypeStr.c_str(), "DT_FLOAT16, DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    if (wDtype != wMinDtype) {
        const std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(wDtype) + ", " +
                                     ge::TypeUtils::DataTypeToSerialString(wMinDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "w, w_min", dtypeStr.c_str(),
                                               "The type of w_min must be same as w");
        return ge::GRAPH_FAILED;
    }
    if (wDtype != wMaxDtype) {
        const std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(wDtype) + ", " +
                                     ge::TypeUtils::DataTypeToSerialString(wMaxDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "w, w_max", dtypeStr.c_str(),
                                               "The type of w_max must be same as w");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(OUTPUT_Y_IDX, wDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(WtsARQ).InferShape(InferShapeForWtsARQ).InferDataType(InferDataTypeForWtsARQ);
} // namespace ops
