/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_tiling.cpp
 * \brief
 */
#include <sstream>
#include <cctype>
#include "scatter_nd_tiling_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
static constexpr size_t SC_IN_INDICES_IDX = 0;
static constexpr size_t SC_IN_X_IDX = 1;
static constexpr size_t SC_OUT_Y_IDX = 0;

static bool CheckScatterNdTensorShape(
    const gert::TilingContext* context, const CalcShapeInfo& calcShapeInfo, const int64_t indicesLastDim)
{
    const int64_t indicesDims = calcShapeInfo.indicesShape.GetDimNum();
    const int64_t updatesDims = calcShapeInfo.varShape.GetDimNum();
    const int64_t outputDims = calcShapeInfo.outShape.GetDimNum();

    OP_CHECK_IF(
        indicesDims <= 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "indices", std::to_string(indicesDims).c_str(),
            "The shape dim of indices must be within the range (2,8)"),
        return false);

    OP_CHECK_IF(
        outputDims - indicesLastDim != updatesDims - indicesDims + 1,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "update and output",
            (std::to_string(outputDims) + ", " + std::to_string(updatesDims)).c_str(),
            ("The shape dim of output after the " + std::to_string(indicesLastDim) + "th axis must be " +
             "equal to the shape dim of update after the " + std::to_string(indicesDims - 1) + "th axis")
                .c_str()),
        return false);

    for (int64_t i = 0; i < indicesDims - 1; i++) {
        OP_CHECK_IF(
            calcShapeInfo.indicesShape.GetDim(i) != calcShapeInfo.varShape.GetDim(i),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "indices and updates",
                (std::to_string(calcShapeInfo.indicesShape.GetDim(i)) + " and " +
                 std::to_string(calcShapeInfo.varShape.GetDim(i)))
                    .c_str(),
                ("The first " + std::to_string(indicesDims - 1) + " axes of indices must equal to that of update")
                    .c_str()),
            return false);
    }

    for (int64_t i = 0; i < updatesDims - indicesDims + 1; i++) {
        OP_CHECK_IF(
            calcShapeInfo.varShape.GetDim(indicesDims - 1 + i) != calcShapeInfo.outShape[indicesLastDim + i],
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "output and updates",
                (std::to_string(calcShapeInfo.outShape[indicesLastDim + i]) + " and " +
                 std::to_string(calcShapeInfo.varShape.GetDim(indicesDims - 1 + i)))
                    .c_str(),
                ("axis " + std::to_string(indicesLastDim + i) + " of output must be equal to the axis " +
                 std::to_string(indicesDims - 1 + i) + " of update")
                    .c_str()),
            return false);
    }
    return true;
}

static ge::graphStatus TilingPrepare4ScatterNd(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ScatterNdCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4ScatterNd GRAPH_SUCESS.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ScatterNd(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "ScatterNdTiling running begin");
    auto compileInfo = reinterpret_cast<const ScatterNdCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    CalcShapeInfo calcShapeInfo;

    const gert::StorageShape* indicesStorageShape = context->GetInputShape(SC_IN_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesStorageShape);
    calcShapeInfo.indicesShape = Ops::Base::EnsureNotScalar(indicesStorageShape->GetStorageShape());

    const gert::StorageShape* xStorageShape = context->GetInputShape(SC_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    calcShapeInfo.varShape = Ops::Base::EnsureNotScalar(xStorageShape->GetStorageShape());

    const gert::StorageShape* yStorageShape = context->GetOutputShape(SC_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);
    calcShapeInfo.outShape = Ops::Base::EnsureNotScalar(yStorageShape->GetStorageShape());

    int64_t indicesLastDim = calcShapeInfo.indicesShape.GetDim(calcShapeInfo.indicesShape.GetDimNum() - 1);
    if (!CheckScatterNdTensorShape(context, calcShapeInfo, indicesLastDim)) {
        return false;
    }

    OP_LOGD(context->GetNodeName(), "ScatterNdTiling is ascendc. runing Smit tiling.");
    return TilingScatterNd(context);
}

// register tiling interface of the Tiling4ScatterNd op.
IMPL_OP_OPTILING(ScatterNd).Tiling(Tiling4ScatterNd).TilingParse<ScatterNdCompileInfo>(TilingPrepare4ScatterNd);
} // namespace optiling
