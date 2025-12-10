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
 * \file multi_scale_deformable_attn_function.cpp
 * \brief
 */

#include "multi_scale_deformable_attn_function.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
using namespace op;
namespace l0op {

OP_TYPE_REGISTER(MultiScaleDeformableAttnFunction);
static const int64_t VALUE_INDEX_0 = 0;
static const int64_t VALUE_INDEX_3 = 3;
static const int64_t LOCATION_INDEX_1 = 1;
static const int64_t LOCATION_INDEX_3 = 3;
static const int64_t LOCATION_INDEX_5 = 5;
static const int64_t VALUE_INDEX_1 = 1;
static const int64_t VALUE_INDEX_2 = 2;
static const int64_t NUMQUERIES_MIN = 32;

const std::tuple<aclTensor*> MultiScaleDeformableAttnFunction(const aclTensor *value,
                                                              const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                                                              const aclTensor *location, const aclTensor *attnWeight,
                                                              aclOpExecutor *executor) {
  L0_DFX(MultiScaleDeformableAttnFunction, value, spatialShape, levelStartIndex, location, attnWeight);
  // shape推导
  Shape outputShape;
  auto valueShape = value->GetViewShape();
  auto locationShape = location->GetViewShape();

  uint64_t numQueries = locationShape.GetDim(1);
  uint64_t nqIndex = LOCATION_INDEX_1;
  uint64_t nhIndex = VALUE_INDEX_2;
  if (numQueries < NUMQUERIES_MIN) {
    nqIndex = LOCATION_INDEX_5;
    nhIndex = VALUE_INDEX_1;
  }
  outputShape.AppendDim(valueShape.GetDim(VALUE_INDEX_0));

  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
    nqIndex = LOCATION_INDEX_3;
    nhIndex = VALUE_INDEX_1;
    outputShape.AppendDim(valueShape.GetDim(nhIndex) * valueShape.GetDim(VALUE_INDEX_3));
    outputShape.AppendDim(locationShape.GetDim(nqIndex));
  } else {
    outputShape.AppendDim(locationShape.GetDim(nqIndex));
    outputShape.AppendDim(valueShape.GetDim(nhIndex) * valueShape.GetDim(VALUE_INDEX_3));
  }

  // 创建输出Tensor
  auto output = executor->AllocTensor(outputShape, value->GetDataType());

  // 调用device的MultiScaleDeformableAttnFunction算子
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(MultiScaleDeformableAttnFunction,
                                         OP_INPUT(value, spatialShape, levelStartIndex, location, attnWeight),
                                         OP_OUTPUT(output));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MultiScaleDeformableAttnFunctionAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    std::tuple<aclTensor*>(nullptr));
  return std::tie(output);
}
} // l0op
