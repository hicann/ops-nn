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
 * \file foreach_flat_regbase_validator.h
 * \brief Reusable validation helpers. Operators opt in through their own validator function.
 */
#ifndef FOREACH_FLAT_REGBASE_VALIDATOR_H
#define FOREACH_FLAT_REGBASE_VALIDATOR_H

#include <cstdint>
#include <limits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/arch35/foreach_flat_tiling_data.h"

namespace optiling {
namespace ForeachFlatRegbaseValidation {

using DtypePredicate = bool (*)(ge::DataType);
constexpr int32_t MAX_DIM_NUM = 8;

inline ge::graphStatus ValidateShape(gert::TilingContext* context, const gert::Shape& shape, const char* valueName,
                                     uint32_t tensorIndex)
{
    if (shape.GetDimNum() > MAX_DIM_NUM) {
        OP_LOGE(context, "%s[%u] supports at most %d dimensions", valueName, tensorIndex, MAX_DIM_NUM);
        return ge::GRAPH_FAILED;
    }
    for (size_t dimIndex = 0; dimIndex < shape.GetDimNum(); ++dimIndex) {
        if (shape.GetDim(dimIndex) < 0) {
            OP_LOGE(context, "%s[%u] contains a negative dimension", valueName, tensorIndex);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

template <uint32_t INPUT_FLOW_COUNT>
ge::graphStatus ValidateHomogeneous(gert::TilingContext* context, DtypePredicate acceptsDtype)
{
    static_assert(INPUT_FLOW_COUNT > 0U, "a flat foreach validator needs at least one tensor-list input");
    auto computeNodeInfo = context->GetComputeNodeInfo();
    if (computeNodeInfo == nullptr || acceptsDtype == nullptr) {
        OP_LOGE(context, "invalid compute-node metadata or dtype predicate");
        return ge::GRAPH_FAILED;
    }

    auto firstListInfo = computeNodeInfo->GetInputInstanceInfo(0);
    if (firstListInfo == nullptr) {
        OP_LOGE(context, "input tensor-list metadata is null");
        return ge::GRAPH_FAILED;
    }
    uint64_t tensorCount = firstListInfo->GetInstanceNum();
    if (tensorCount == 0U || tensorCount > FOREACH_FLAT_MAX_TENSOR_COUNT) {
        OP_LOGE(context, "tensor-list size must be in [1, %u], but got %lu", FOREACH_FLAT_MAX_TENSOR_COUNT,
                tensorCount);
        return ge::GRAPH_FAILED;
    }

    for (uint32_t flow = 1; flow < INPUT_FLOW_COUNT; ++flow) {
        auto listInfo = computeNodeInfo->GetInputInstanceInfo(flow);
        if (listInfo == nullptr || listInfo->GetInstanceNum() != tensorCount) {
            OP_LOGE(context, "all tensor-list inputs must have the same list size");
            return ge::GRAPH_FAILED;
        }
    }
    if (context->GetComputeNodeOutputNum() != tensorCount) {
        OP_LOGE(context, "input and output tensor lists must have the same list size");
        return ge::GRAPH_FAILED;
    }

    ge::DataType listDtype = ge::DT_UNDEFINED;
    int64_t totalElements = 0;
    for (uint32_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex) {
        auto firstDesc = context->GetDynamicInputDesc(0, tensorIndex);
        auto firstShape = context->GetDynamicInputShape(0, tensorIndex);
        if (firstDesc == nullptr || firstShape == nullptr) {
            OP_LOGE(context, "x[%u] descriptor or shape is null", tensorIndex);
            return ge::GRAPH_FAILED;
        }

        ge::DataType currentDtype = firstDesc->GetDataType();
        if (!acceptsDtype(currentDtype)) {
            OP_LOGE(context, "x[%u] has an unsupported dtype", tensorIndex);
            return ge::GRAPH_FAILED;
        }
        if (listDtype == ge::DT_UNDEFINED) {
            listDtype = currentDtype;
        } else if (currentDtype != listDtype) {
            OP_LOGE(context, "all tensors in a tensor list must have the same dtype");
            return ge::GRAPH_FAILED;
        }

        const gert::Shape& storageShape = firstShape->GetStorageShape();
        if (ValidateShape(context, storageShape, "x", tensorIndex) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        int64_t dataCount = storageShape.GetShapeSize();
        if (dataCount < 0 || totalElements > std::numeric_limits<int64_t>::max() - dataCount) {
            OP_LOGE(context, "invalid or overflowing tensor-list element count");
            return ge::GRAPH_FAILED;
        }
        totalElements += dataCount;

        for (uint32_t flow = 1; flow < INPUT_FLOW_COUNT; ++flow) {
            auto inputDesc = context->GetDynamicInputDesc(flow, tensorIndex);
            auto inputShape = context->GetDynamicInputShape(flow, tensorIndex);
            if (inputDesc == nullptr || inputShape == nullptr || inputDesc->GetDataType() != listDtype ||
                inputShape->GetStorageShape() != storageShape) {
                OP_LOGE(context, "same-position input tensors must have identical dtype and storage shape");
                return ge::GRAPH_FAILED;
            }
        }

        auto outputDesc = context->GetOutputDesc(tensorIndex);
        auto outputShape = context->GetOutputShape(tensorIndex);
        if (outputDesc == nullptr || outputShape == nullptr || outputDesc->GetDataType() != listDtype) {
            OP_LOGE(context, "same-position output must match the first input dtype");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& outputStorageShape = outputShape->GetStorageShape();
        if (ValidateShape(context, outputStorageShape, "y", tensorIndex) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        // Match the Membase contract: the output storage may be represented differently
        // (notably scalar () versus (1,)), but it must have enough flat capacity for x.
        if (outputStorageShape.GetShapeSize() < dataCount) {
            OP_LOGE(context, "y[%u] storage must contain at least as many elements as x[%u]", tensorIndex, tensorIndex);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline bool IsFloatFamily(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16;
}

} // namespace ForeachFlatRegbaseValidation
} // namespace optiling

#endif // FOREACH_FLAT_REGBASE_VALIDATOR_H
