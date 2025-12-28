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
 * \file grouped_dynamic_mx_quant.cc
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/shape_util.h"

using namespace ge;
namespace ops {
constexpr size_t INDEX_ATTR_DST_TYPE = 1;
constexpr size_t INDEX_ATTR_BLOCK_SIZE = 2;
constexpr size_t SCALE_DIM_NUM = 3;
static const int32_t DTYPE_FLOAT8_E5M2 = 35;
static const int32_t DTYPE_FLOAT8_E4M3FN = 36;

template <typename T>
static std::string Shape2String(const T& shape) {
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

graphStatus InferShapeForGroupedDynamicMxQuant(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForGroupedDynamicMxQuant");
  const gert::Shape* xShape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
  const gert::Shape* groupIdxShape = context->GetInputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, groupIdxShape);

  gert::Shape* yShape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
  *yShape = *xShape;

  gert::Shape* scaleShape = context->GetOutputShape(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

  auto attrsPtr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
  
  const int32_t *blockSize = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_BLOCK_SIZE);
  OP_CHECK_NULL_WITH_CONTEXT(context, blockSize);
  OP_CHECK_IF(static_cast<int64_t>(*blockSize) != 32,
           OP_LOGE(context->GetNodeName(), "blockSize is invalid, must be 32"),
           return ge::GRAPH_FAILED);
  size_t xShapeSize = xShape->GetDimNum();
  size_t groupIdxShapeSize = groupIdxShape->GetDimNum();
  OP_CHECK_IF(groupIdxShapeSize != 1,
            OP_LOGE(context->GetNodeName(),
                "group_index's shape must be 1D, but is %lu", groupIdxShapeSize),
            return ge::GRAPH_FAILED);
  int64_t groupIdxDim0 = groupIdxShape->GetDim(0);
  OP_CHECK_IF(groupIdxDim0 == 0,
            OP_LOGE(context->GetNodeName(), "group_index does not support empty tensor"),
            return ge::GRAPH_FAILED);

  // dynamic -2 (input x)
  if (Ops::Base::IsUnknownRank(*xShape)) {
      OP_LOGD(context->GetNodeName(), "input x is UnknownRank, set outputs' shape to -2");
      *scaleShape = *xShape;
      return ge::GRAPH_SUCCESS;
  } else {
      OP_CHECK_IF(xShapeSize != 2,
        OP_LOGE(context->GetNodeName(),
            "input x is not UnknownRank, shape must be 2D, but is %lu", xShapeSize),
        return ge::GRAPH_FAILED);
  }

  int64_t dim0Size = (xShape->GetDim(0) / (static_cast<int64_t>(*blockSize) * 2) + groupIdxDim0); // 不带起始0
  // dynamic -2 or -1 (groupIdxShape), dynamic -1 (input x)
  if (Ops::Base::IsUnknownRank(*groupIdxShape) || groupIdxDim0 == -1|| xShape->GetDim(0) == -1) {
    dim0Size = -1;
  }

  scaleShape->SetDimNum(SCALE_DIM_NUM);
  scaleShape->SetDim(0, dim0Size);
  scaleShape->SetDim(1, (xShape->GetDim(1) == -1) ? -1 : xShape->GetDim(1));
  scaleShape->SetDim(2, 2); // mxscale's third dimensions only support 2
  OP_LOGD(context->GetNodeName(), "mxscale shape is :%s after infershape.", Shape2String(*scaleShape).c_str());
  OP_LOGD(context->GetNodeName(), "End to do InferShapeForGroupedDynamicMxQuant");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGroupedDynamicMxQuant(gert::InferDataTypeContext *context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForGroupedDynamicMxQuant");
  auto attrsPtr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
  ge::DataType yDtype = ge::DT_FLOAT8_E5M2;
  const int32_t *pDstDtype = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
  if (pDstDtype != nullptr) {
      int32_t dstDtype = *pDstDtype;
      OP_CHECK_IF(dstDtype != DTYPE_FLOAT8_E5M2 && dstDtype != DTYPE_FLOAT8_E4M3FN,
               OP_LOGE(context->GetNodeName(),
                                                   "attr dst_type only support 35(FLOAT8_E5M2) and 36(FLOAT8_E4M3FN)"),
               return ge::GRAPH_FAILED);
      yDtype = static_cast<ge::DataType>(dstDtype);
  }
  context->SetOutputDataType(0, yDtype);
  context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
  OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForGroupedDynamicMxQuant");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedDynamicMxQuant)
    .InferShape(InferShapeForGroupedDynamicMxQuant)
    .InferDataType(InferDataTypeForGroupedDynamicMxQuant);
}  // namespace ops