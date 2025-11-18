/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file celu_v2.cpp
 * \brief
 */

#include "softmax_cross_entropy_with_logits.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SoftmaxCrossEntropyWithLogits);

std::tuple<aclTensor*, aclTensor*> SoftmaxCrossEntropyWithLogits(const aclTensor *features, const aclTensor *labels, aclOpExecutor *executor) {
  L0_DFX(SoftmaxCrossEntropyWithLogits, features, labels);
  // 获取 features 的 shape
  const auto& featShape = features->GetViewShape();
  size_t dimNum = featShape.GetDimNum();
  op::Shape lossShape({featShape.GetDim(0)});
  // 分配输出 Tensor
  auto loss = executor->AllocTensor(lossShape, features->GetDataType());
  auto backprop = executor->AllocTensor(featShape, features->GetDataType());  // backprop 仍保持原 shape
  if (loss == nullptr || backprop == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc loss or backprop tensor failed.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }
  // 注册 AICORE 执行任务
  ADD_TO_LAUNCHER_LIST_AICORE(
    SoftmaxCrossEntropyWithLogits, OP_INPUT(features, labels), OP_OUTPUT(loss, backprop));
  return std::tuple<aclTensor*, aclTensor*>(loss, backprop);
}

}  // namespace l0op