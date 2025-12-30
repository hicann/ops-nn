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
 * \file scatter_nd_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "util/const_util.h"

using namespace ge;
namespace ops {

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

inline bool IsConstTensor(const gert::Tensor* input_tensor) {
  if (input_tensor != nullptr) {
    if (input_tensor->GetAddr() == nullptr) {
      // empty tensor
      return input_tensor->GetShapeSize() == 0;
    }
    return true;
  }
  return false;
}

static constexpr size_t SC_OUT_VAR_IDX = 0;

// ---------------ScatterNd Ops START-----------------
static constexpr int64_t SC_IN_SHAPE_IDX = 2;

static ge::graphStatus Infershape4Nd(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do ScatterNdInfershape.");

  const gert::Tensor* shape_tensor = context->GetInputTensor(SC_IN_SHAPE_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor);
  gert::Shape* output_shape = context->GetOutputShape(SC_OUT_VAR_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

  ge::DataType shape_dtype = shape_tensor->GetDataType();
  OP_CHECK_IF(shape_dtype != ge::DT_INT32 && shape_dtype != ge::DT_INT64,
           OP_LOGE(
               context->GetNodeName(),
               "[", context->GetNodeName(), "], has wrong dtype [", ge::TypeUtils::DataTypeToSerialString(shape_dtype).c_str(), "], it should be in ", "[int32, int64]"),
           return ge::GRAPH_FAILED);

  if (IsConstTensor(shape_tensor)) {
    OP_CHECK_IF(!Ops::Base::GetConstIntToShape(context, SC_IN_SHAPE_IDX, *output_shape),
             OP_LOGE(context->GetNodeName(), "Infershape4ScatterNd is failed!"),
             return ge::GRAPH_FAILED);
  } else {
    output_shape->SetDimNum(0);
    output_shape->AppendDim(UNKNOWN_DIM_VALUE_);
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterNd).InferShape(Infershape4Nd).InputsDataDependency({SC_IN_SHAPE_IDX});
// ---------------ScatterNd Ops END-----------------

}  // namespace ops
