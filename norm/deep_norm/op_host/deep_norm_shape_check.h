/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deep_norm_shape_check.h
 * \brief Shared DeepNorm input/output shape legality checks used by both the
 *        arch22 and arch35 tiling. Keeping a single copy avoids the two
 *        implementations drifting (arch35 previously dropped these guards).
 */
#ifndef OPS_NORM_DEEP_NORM_SHAPE_CHECK_H
#define OPS_NORM_DEEP_NORM_SHAPE_CHECK_H

#include <string>
#include <utility>
#include "exe_graph/runtime/tiling_context.h"
#include "log/log.h"

namespace optiling {

constexpr size_t DEEP_NORM_MAX_DIM_X = 8;
constexpr size_t DEEP_NORM_MIN_DIM_X = 2;
constexpr size_t DEEP_NORM_MAX_DIM_GAMMA = 7;
constexpr size_t DEEP_NORM_MIN_DIM_GAMMA = 1;

constexpr int32_t DEEP_NORM_INPUT_X_INDEX = 0;
constexpr int32_t DEEP_NORM_INPUT_GX_INDEX = 1;
constexpr int32_t DEEP_NORM_INPUT_BETA_INDEX = 2;
constexpr int32_t DEEP_NORM_INPUT_GAMMA_INDEX = 3;
constexpr int32_t DEEP_NORM_OUTPUT_MEAN_INDEX = 0;
constexpr int32_t DEEP_NORM_OUTPUT_RSTD_INDEX = 1;
constexpr int32_t DEEP_NORM_OUTPUT_Y_INDEX = 2;

inline ge::graphStatus CheckDeepNormShapeDim(const gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(DEEP_NORM_INPUT_X_INDEX);
    const gert::StorageShape* gxShape = context->GetInputShape(DEEP_NORM_INPUT_GX_INDEX);
    const gert::StorageShape* betaShape = context->GetInputShape(DEEP_NORM_INPUT_BETA_INDEX);
    const gert::StorageShape* gammaShape = context->GetInputShape(DEEP_NORM_INPUT_GAMMA_INDEX);
    const gert::StorageShape* meanShape = context->GetOutputShape(DEEP_NORM_OUTPUT_MEAN_INDEX);
    const gert::StorageShape* rstdShape = context->GetOutputShape(DEEP_NORM_OUTPUT_RSTD_INDEX);
    const gert::StorageShape* yShape = context->GetOutputShape(DEEP_NORM_OUTPUT_Y_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gxDimNum = gxShape->GetStorageShape().GetDimNum();
    size_t betaDimNum = betaShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t meanDimNum = meanShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();
    size_t yDimNum = yShape->GetStorageShape().GetDimNum();

    // Check shape dim range
    OP_CHECK_IF((xDimNum > DEEP_NORM_MAX_DIM_X) || (xDimNum < DEEP_NORM_MIN_DIM_X),
                OP_LOGE(context, "Input x shape invaild, dim num should in range[%lu, %lu].", DEEP_NORM_MIN_DIM_X,
                        DEEP_NORM_MAX_DIM_X),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((gammaDimNum > DEEP_NORM_MAX_DIM_GAMMA) || (gammaDimNum < DEEP_NORM_MIN_DIM_GAMMA),
                OP_LOGE(context, "Input gamma shape invaild, dim num should in range[%lu, %lu].",
                        DEEP_NORM_MIN_DIM_GAMMA, DEEP_NORM_MAX_DIM_GAMMA),
                return ge::GRAPH_FAILED);
    // Check shape dim relationship
    OP_CHECK_IF(gxDimNum != xDimNum, OP_LOGE(context, "Input gx shape invaild, dim num is not equal x dim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((yDimNum != xDimNum) || (meanDimNum != xDimNum) || (rstdDimNum != xDimNum),
                OP_LOGE(context, "Output y/mean/rstd shape invaild, dim num is not equal x dim."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaDimNum != gammaDimNum,
                OP_LOGE(context, "Input beta shape invaild, dim num is not equal gamma dim."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum <= gammaDimNum, OP_LOGE(context, "x dim num should not be smaller than gamma dim num."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckDeepNormShapeValue(const gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(DEEP_NORM_INPUT_X_INDEX);
    const gert::StorageShape* gxShape = context->GetInputShape(DEEP_NORM_INPUT_GX_INDEX);
    const gert::StorageShape* betaShape = context->GetInputShape(DEEP_NORM_INPUT_BETA_INDEX);
    const gert::StorageShape* gammaShape = context->GetInputShape(DEEP_NORM_INPUT_GAMMA_INDEX);
    const gert::StorageShape* meanShape = context->GetOutputShape(DEEP_NORM_OUTPUT_MEAN_INDEX);
    const gert::StorageShape* rstdShape = context->GetOutputShape(DEEP_NORM_OUTPUT_RSTD_INDEX);
    const gert::StorageShape* yShape = context->GetOutputShape(DEEP_NORM_OUTPUT_Y_INDEX);

    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();

    // Check shape value
    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_CHECK_IF(xShape->GetStorageShape().GetDim(i) == 0, OP_LOGE(context, "Input x shape can not be 0."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gxShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i),
                    OP_LOGE(context, "Input gx shape invaild, shape is not equal x shape."), return ge::GRAPH_FAILED);
        OP_CHECK_IF((yShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)),
                    OP_LOGE(context, "Input y shape invaild, shape is not equal x shape."), return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < xDimNum - gammaDimNum; i++) {
        OP_CHECK_IF((rstdShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)) ||
                        (meanShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)),
                    OP_LOGE(context, "Output rstd/mean shape invaild, shape is not equal x first few dim."),
                    return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_CHECK_IF(
            (gammaShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(xDimNum - gammaDimNum + i)) ||
                (betaShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(xDimNum - gammaDimNum + i)),
            OP_LOGE(context, "Input gamma shape invaild, gamma shape is not equal x last few dim."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// x/gx/beta/gamma/y 的 dtype 必须同为 DT_FLOAT16 / DT_FLOAT / DT_BF16;gx/beta/gamma/y 需与 x 一致。
// mean/rstd 固定为 DT_FLOAT,已由算子 IR 注册约束,这里不再重复校验。
inline ge::graphStatus CheckDeepNormDtype(const gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(DEEP_NORM_INPUT_X_INDEX);
    auto gxDesc = context->GetInputDesc(DEEP_NORM_INPUT_GX_INDEX);
    auto betaDesc = context->GetInputDesc(DEEP_NORM_INPUT_BETA_INDEX);
    auto gammaDesc = context->GetInputDesc(DEEP_NORM_INPUT_GAMMA_INDEX);
    auto yDesc = context->GetOutputDesc(DEEP_NORM_OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gxDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT && xDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(xDtype),
                                                      "The dtype of x must be DT_FLOAT16, DT_FLOAT, or DT_BF16"),
                return ge::GRAPH_FAILED);

    const std::pair<const char*, ge::DataType> others[] = {
        {"gx", gxDesc->GetDataType()},
        {"beta", betaDesc->GetDataType()},
        {"gamma", gammaDesc->GetDataType()},
        {"y", yDesc->GetDataType()},
    };
    for (const auto& item : others) {
        OP_CHECK_IF(item.second != xDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context->GetNodeName(), std::string("x, ") + item.first,
                        Ops::Base::ToString(xDtype) + ", " + Ops::Base::ToString(item.second),
                        "The dtypes of x, gx, beta, gamma, y must be the same"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

#endif // OPS_NORM_DEEP_NORM_SHAPE_CHECK_H
