/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {

constexpr size_t KERNEL_SIZE_INDEX = 0;
constexpr size_t STRIDE_INDEX = 1;
constexpr size_t PADDING_INDEX = 2;
constexpr size_t DILATION_INDEX = 3;
constexpr size_t CEIL_MODE_INDEX = 4;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_RANK = -2;

struct SpatialPair {
    int64_t h;
    int64_t w;
};

bool ParsePair(const gert::TypedContinuousVector<int64_t>* values, bool allowEmpty, const SpatialPair& emptyValue,
               SpatialPair& result)
{
    if (values == nullptr) {
        return false;
    }
    const size_t size = values->GetSize();
    if (size == 0) {
        if (!allowEmpty) {
            return false;
        }
        result = emptyValue;
        return true;
    }
    if (size != 1 && size != 2) {
        return false;
    }
    const int64_t* data = values->GetData();
    result.h = data[0];
    result.w = size == 1 ? data[0] : data[1];
    return true;
}

bool MulNoOverflow(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool ComputeOutputDim(int64_t input, int64_t kernel, int64_t stride, int64_t padding, int64_t dilation, bool ceilMode,
                      int64_t& output)
{
    if (input == UNKNOWN_DIM) {
        output = UNKNOWN_DIM;
        return true;
    }
    if (input <= 0) {
        return false;
    }
    // dilation has already been checked to be 1. Form input - kernel first to avoid
    // overflowing input + 2 * padding for otherwise representable dimensions.
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    const int64_t difference = input - effectiveKernel;
    const int64_t totalPadding = 2 * padding;
    if (difference > std::numeric_limits<int64_t>::max() - totalPadding) {
        return false;
    }
    const int64_t numerator = difference + totalPadding;
    // C++ truncates towards zero. Adjust explicitly to implement mathematical
    // ceil/floor, including a valid ceil-mode window larger than the input.
    int64_t quotient = numerator / stride;
    const int64_t remainder = numerator % stride;
    if (ceilMode && remainder > 0) {
        ++quotient;
    } else if (!ceilMode && remainder < 0) {
        --quotient;
    }
    if (quotient < 0 || quotient == std::numeric_limits<int64_t>::max()) {
        return false;
    }
    output = quotient + 1;
    if (input > std::numeric_limits<int64_t>::max() - padding) {
        return false;
    }
    // Compare quotient * stride >= input + padding without multiplying.
    const int64_t rightEdge = input + padding;
    const int64_t firstPaddingWindow = rightEdge / stride + (rightEdge % stride != 0);
    if (ceilMode && quotient >= firstPaddingWindow) {
        --output;
    }
    return output > 0;
}

ge::graphStatus InferShapeMaxPool2dWithMask(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* indicesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    const size_t rank = inputShape->GetDimNum();
    if (rank == 1 && inputShape->GetDim(0) == UNKNOWN_RANK) {
        *outShape = *inputShape;
        *indicesShape = *inputShape;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(rank != 3 && rank != 4,
                OP_LOGE(context, "MaxPool2dWithMask input rank must be 3 or 4, but got %zu", rank),
                return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* kernelValues = attrs->GetListInt(KERNEL_SIZE_INDEX);
    const auto* strideValues = attrs->GetListInt(STRIDE_INDEX);
    const auto* paddingValues = attrs->GetListInt(PADDING_INDEX);
    const auto* dilationValues = attrs->GetListInt(DILATION_INDEX);

    SpatialPair kernel{};
    SpatialPair stride{};
    SpatialPair padding{};
    SpatialPair dilation{};
    OP_CHECK_IF(!ParsePair(kernelValues, false, {}, kernel) || kernel.h <= 0 || kernel.w <= 0,
                OP_LOGE(context, "kernelSize must contain one or two positive values"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ParsePair(strideValues, true, kernel, stride) || stride.h <= 0 || stride.w <= 0,
                OP_LOGE(context, "stride must be empty or contain one or two positive values"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ParsePair(paddingValues, false, {}, padding) || padding.h < 0 || padding.w < 0 ||
                    padding.h > kernel.h / 2 || padding.w > kernel.w / 2,
                OP_LOGE(context, "padding must satisfy 0 <= padding[d] <= floor(kernelSize[d] / 2)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ParsePair(dilationValues, false, {}, dilation) || dilation.h != 1 || dilation.w != 1,
                OP_LOGE(context, "dilation must contain one or two values equal to 1"), return ge::GRAPH_FAILED);

    const bool* ceilModeAttr = attrs->GetBool(CEIL_MODE_INDEX);
    const bool ceilMode = ceilModeAttr == nullptr ? false : *ceilModeAttr;
    const size_t hIndex = rank - 2;
    const size_t wIndex = rank - 1;
    int64_t outputHeight = 0;
    int64_t outputWidth = 0;
    OP_CHECK_IF(!ComputeOutputDim(inputShape->GetDim(hIndex), kernel.h, stride.h, padding.h, dilation.h, ceilMode,
                                  outputHeight) ||
                    !ComputeOutputDim(inputShape->GetDim(wIndex), kernel.w, stride.w, padding.w, dilation.w, ceilMode,
                                      outputWidth),
                OP_LOGE(context, "failed to infer a positive output shape"), return ge::GRAPH_FAILED);

    *outShape = *inputShape;
    outShape->SetDim(hIndex, outputHeight);
    outShape->SetDim(wIndex, outputWidth);

    int64_t kernelElements = 0;
    OP_CHECK_IF(!MulNoOverflow(kernel.h, kernel.w, kernelElements),
                OP_LOGE(context, "kernelSize product overflows int64"), return ge::GRAPH_FAILED);

    int64_t maskWidth = UNKNOWN_DIM;
    if (outputHeight != UNKNOWN_DIM && outputWidth != UNKNOWN_DIM) {
        int64_t outputElements = 0;
        OP_CHECK_IF(!MulNoOverflow(outputHeight, outputWidth, outputElements) ||
                        outputElements > std::numeric_limits<int64_t>::max() - 15,
                    OP_LOGE(context, "output spatial size overflows int64"), return ge::GRAPH_FAILED);
        const int64_t maskBlocks = (outputElements + 15) / 16 + 1;
        OP_CHECK_IF(!MulNoOverflow(maskBlocks, 32, maskWidth), OP_LOGE(context, "indices mask width overflows int64"),
                    return ge::GRAPH_FAILED);
        if (kernelElements == 1) {
            int64_t indexWidth = 0;
            OP_CHECK_IF(!MulNoOverflow(outputElements / 8 + (outputElements % 8 != 0), 32, indexWidth),
                        OP_LOGE(context, "int32 indices width overflows int64"), return ge::GRAPH_FAILED);
            maskWidth = std::max(maskWidth, indexWidth);
        }
    }

    indicesShape->SetDimNum(rank);
    if (rank == 4) {
        indicesShape->SetDim(0, inputShape->GetDim(0));
        indicesShape->SetDim(1, inputShape->GetDim(1));
        indicesShape->SetDim(2, kernelElements);
        indicesShape->SetDim(3, maskWidth);
    } else {
        indicesShape->SetDim(0, inputShape->GetDim(0));
        indicesShape->SetDim(1, kernelElements);
        indicesShape->SetDim(2, maskWidth);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_INFERSHAPE(MaxPool2dWithMask).InferShape(InferShapeMaxPool2dWithMask);

} // namespace ops
