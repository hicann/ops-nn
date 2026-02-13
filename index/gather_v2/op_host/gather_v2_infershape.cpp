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
 * \file gather_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/const_util.h"
#include <string>
#include <sstream>

using namespace ge;
namespace ops {
const size_t INPUT_IDX_X = 0;
const size_t INPUT_IDX_INDICES = 1;
const size_t INPUT_IDX_AXIS = 2;

struct GatherInfo {
  int64_t axis;
  int64_t index_batch_dims;
  int64_t x_real_dim_cnt;
  int64_t rank_indices;
};

template <typename T>
static std::string ConcatString(const T& arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
static std::string ConcatString(const T& arg, const Ts& ... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}

static std::string OtherErrMsg(const std::string& error_detail) {
  std::string msg = error_detail;
  return msg;
}

static bool CheckAndUpdateAxis(const gert::InferShapeContext* context, int64_t& batch_dims, int64_t& axes_data,
                               const GatherInfo& gather_info) {
  int64_t x_real_dim_cnt = gather_info.x_real_dim_cnt;
  int64_t index_batch_dims = gather_info.index_batch_dims;
  int64_t rank_indices = gather_info.rank_indices;
  if (batch_dims < -rank_indices || (batch_dims > rank_indices && rank_indices != 0)) {
      std::string err_msg = OtherErrMsg(ConcatString("Expected batch_dims in the range [", std::to_string(-rank_indices), ",",
                                                     std::to_string(rank_indices), "), but got", std::to_string(batch_dims)));
      OP_LOGE(context->GetNodeName(), "%s", err_msg.c_str());
      return false;
    }
  if (batch_dims < 0) {
    batch_dims = batch_dims + rank_indices;
  }
  OP_CHECK_IF(batch_dims >= x_real_dim_cnt,
           OP_LOGE(context->GetNodeName(), "batch_dims must be less than rank x)"),
           return false);
  auto x_shape = context->GetInputShape(0);
  auto indies_shape = context->GetInputShape(1);
  for (int i = 0; i < batch_dims; i++) {
    if (x_shape->GetDim(i) != indies_shape->GetDim(i) && x_shape->GetDim(i) > 0 && indies_shape->GetDim(i) > 0) {
      OP_LOGE(context->GetNodeName(), "x_shape not equal indies_shape");
      return false;
    }
  }
  if (index_batch_dims == 1 && (batch_dims != 0)) {
    axes_data = batch_dims;
  }
  OP_CHECK_IF(axes_data > 0 && (x_real_dim_cnt < axes_data + 1),
           OP_LOGE(context->GetNodeName(), "axis is invalid"), return false);
  if (axes_data < 0) {
    OP_CHECK_IF(x_real_dim_cnt < -axes_data,
             OP_LOGE(context->GetNodeName(), "axis is invalid"), return false);

    axes_data = x_real_dim_cnt + axes_data;
    OP_CHECK_IF(
        batch_dims > axes_data,
        OP_LOGE(context->GetNodeName(), "batch_dims must be less than or equal to axis"),
        return false);
  }
  return true;
}

static ge::graphStatus GatherCommonInfer(gert::InferShapeContext* context, const gert::Shape* x_shape,
                                         const gert::Shape* indies_shape, gert::Shape* out_shape,
                                         GatherInfo& gather_info) {
  OP_LOGD(context->GetNodeName(), "GatherCommonInfer infershape impl is begin");
  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const auto* batchdims = attrs->GetAttrPointer<int64_t>(gather_info.index_batch_dims);
  OP_CHECK_NULL_WITH_CONTEXT(context, batchdims);
  int64_t batch_dims = *batchdims;
  int64_t axis = gather_info.axis;
  int64_t x_real_dim_cnt = gather_info.x_real_dim_cnt;
  int64_t rank_indices = gather_info.rank_indices;

  gert::Shape real_x_shape = gert::Shape();
  gert::Shape real_indices_shape = gert::Shape();
  real_x_shape.SetDimNum(0);
  real_indices_shape.SetDimNum(0);
  for (int64_t i = 0; i < x_real_dim_cnt; i++) {
    real_x_shape.AppendDim(x_shape->GetDim(i));
  }
  for (int64_t i = 0; i < rank_indices; i++) {
    real_indices_shape.AppendDim(indies_shape->GetDim(i));
  }

  // Adapt the scene of scalar input, set it's shape to (1,)
  if (x_shape->IsScalar()) {
    OP_LOGD(context->GetNodeName(), "GatherCommonInfer x_shape is scalar, set it's shape to (1,)");
    real_x_shape.AppendDim(1);
    x_real_dim_cnt = 1;
    gather_info.x_real_dim_cnt = 1;
  }

  OP_CHECK_IF(x_real_dim_cnt < 1,
           OP_LOGE("x_real_dim_cnt", "x_real_dim_cnt must be more than 1"),
           return ge::GRAPH_FAILED);

  OP_CHECK_IF(!CheckAndUpdateAxis(context, batch_dims, axis, gather_info),
           OP_LOGE(context->GetNodeName(), "check axis and batchdim failed"),
           return ge::GRAPH_FAILED);

  for (int64_t i = 0; i < axis; i++) {
    out_shape->AppendDim(real_x_shape.GetDim(i));
  }
  // real dim cnt has no existing meaning .Original shape has replace its meaning now
  for (int64_t i = batch_dims; i < rank_indices; i++) {
    out_shape->AppendDim(real_indices_shape.GetDim(i));
  }

  for (int64_t i = axis + 1; i < x_real_dim_cnt; i++) {
    out_shape->AppendDim(real_x_shape.GetDim(i));
  }
  OP_LOGD(context->GetNodeName(), "infershape is success");
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForGatherV2(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "infershape is begin");
  auto x_shape = context->GetInputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  int64_t x_real_dim_cnt = x_shape->GetDimNum();
  auto indies_shape = context->GetInputShape(INPUT_IDX_INDICES);
  OP_CHECK_NULL_WITH_CONTEXT(context, indies_shape);
  int64_t rank_indices = indies_shape->GetDimNum();

  auto axes_tensor = context->GetInputTensor(INPUT_IDX_AXIS);
  OP_CHECK_NULL_WITH_CONTEXT(context, axes_tensor);
  auto out_shape = context->GetOutputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  out_shape->SetDimNum(0);
  auto axes_size = static_cast<int32_t>(axes_tensor->GetShapeSize());
  OP_LOGD(context->GetNodeName(), "axes_size is %d", axes_size);
  OP_CHECK_IF(axes_size < 1, OP_LOGE(context->GetNodeName(), "axes size must big than 0!"),
           return ge::GRAPH_FAILED);
  int64_t axis = 0;
  OP_CHECK_IF(!Ops::Base::GetConstInt(context, INPUT_IDX_AXIS, axis),
           OP_LOGE(context->GetNodeName(), "get axis failed"), return ge::GRAPH_FAILED);

  GatherInfo gatherinfo = {axis, 0, x_real_dim_cnt, rank_indices};
  return GatherCommonInfer(context, x_shape, indies_shape, out_shape, gatherinfo);
}

static ge::graphStatus InferShapeForGather(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "infershape is begin");
  auto x_shape = context->GetInputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  int64_t x_real_dim_cnt = x_shape->GetDimNum();
  auto indies_shape = context->GetInputShape(INPUT_IDX_INDICES);
  OP_CHECK_NULL_WITH_CONTEXT(context, indies_shape);
  int64_t rank_indices = indies_shape->GetDimNum();

  auto out_shape = context->GetOutputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  out_shape->SetDimNum(0);
  GatherInfo gatherinfo = {0, 1, x_real_dim_cnt, rank_indices};

  return GatherCommonInfer(context, x_shape, indies_shape, out_shape, gatherinfo);
}

IMPL_OP_INFERSHAPE(GatherV2).InferShape(InferShapeForGatherV2).InputsDataDependency({INPUT_IDX_AXIS});
IMPL_OP_INFERSHAPE(Gather).InferShape(InferShapeForGather);
}  // namespace ops