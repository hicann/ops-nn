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


static bool CheckScatterNdTensorShape(const gert::TilingContext* context, const CalcShapeInfo& calcShapeInfo,
                                      const int64_t indicesLastDim) {
  const int64_t indicesDims = calcShapeInfo.indicesShape.GetDimNum();
  const int64_t updatesDims = calcShapeInfo.varShape.GetDimNum();
  const int64_t outputDims = calcShapeInfo.outShape.GetDimNum();

  OP_CHECK_IF(indicesDims <= 1,
                  OP_LOGE(
                      context->GetNodeName(), "the ndim of indices is less than 1 or equal to 1, indicesDims = %ld.",
                      indicesDims),
                  return false);

  OP_CHECK_IF(
      outputDims - indicesLastDim != updatesDims - indicesDims + 1,
      OP_LOGE(context->GetNodeName(),
                                      "output's shape and updates'shape are not equal in some dimensions "
                                      "outputDims - indicesLastDim = %ld, updatesDims - indicesDims + 1 = %ld.",
                                      outputDims - indicesLastDim, updatesDims - indicesDims + 1),
      return false);

  for (int64_t i = 0; i < indicesDims - 1; i++) {
    OP_CHECK_IF(calcShapeInfo.indicesShape.GetDim(i) != calcShapeInfo.varShape.GetDim(i),
                    OP_LOGE(
                        context->GetNodeName(),
                        "indices's shape and updates'shape are not equal in some dimensions, "
                        "calcShapeInfo.indicesShape.GetDim(i) = %ld, calcShapeInfo.varShape.GetDim(i) = %ld.",
                        calcShapeInfo.indicesShape.GetDim(i), calcShapeInfo.varShape.GetDim(i)),
                    return false);
  }

  for (int64_t i = 0; i < updatesDims - indicesDims + 1; i++) {
    OP_CHECK_IF(
        calcShapeInfo.varShape.GetDim(indicesDims - 1 + i) != calcShapeInfo.outShape[indicesLastDim + i],
        OP_LOGE(context->GetNodeName(),
                                        "output's shape and updates'shape are not equal in some dimensions, "
                                        "calcShapeInfo.varShape.GetDim(indicesDims - 1 + i) = %ld, "
                                        "calcShapeInfo.outShape[indicesLastDim + i] = %ld.",
                                        calcShapeInfo.varShape.GetDim(indicesDims - 1 + i),
                                        calcShapeInfo.outShape[indicesLastDim + i]),
        return false);
  }
  return true;
}

static ge::graphStatus TilingPrepare4ScatterNd(gert::TilingParseContext* context) {
  auto compileInfo = context->GetCompiledInfo<ScatterNdCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

  OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4ScatterNd GRAPH_SUCESS.");
  return ge::GRAPH_SUCCESS;
  
}

static ge::graphStatus Tiling4ScatterNd(gert::TilingContext* context) {
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
  OP_CHECK_IF(!CheckScatterNdTensorShape(context, calcShapeInfo, indicesLastDim),
                  OP_LOGE(context->GetNodeName(), "CheckScatterNdTensorShape is failed!"),
                  return false);

  OP_LOGD(context->GetNodeName(), "ScatterNdTiling is ascendc. runing Smit tiling.");
  return TilingScatterNd(context);
}

// register tiling interface of the Tiling4ScatterNd op.
IMPL_OP_OPTILING(ScatterNd).Tiling(Tiling4ScatterNd).TilingParse<ScatterNdCompileInfo>(TilingPrepare4ScatterNd);
}  // namespace optiling
