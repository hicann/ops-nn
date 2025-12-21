/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "weightQuantBatchmatmul.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(WeightQuantBatchmatmul);

const aclTensor *WeightQuantBatchmatmul(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                        const aclTensor *diagonalMatrix, const aclTensor *deqOffset,
                                        const aclTensor *deqScale, const bool adjX1, const bool adjX2,
                                        aclOpExecutor *executor) {
  L0_DFX(WeightQuantBatchmatmul, x1, x2, bias, diagonalMatrix,
         deqOffset, deqScale, adjX1, adjX2);
  auto weightQuantBmmOut = executor->AllocTensor(DataType::DT_FLOAT16, Format::FORMAT_ND, Format::FORMAT_ND);
  OP_CHECK(
      weightQuantBmmOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "weightQuantBmmOut AllocTensor failed."),
      return nullptr);

  auto ret = INFER_SHAPE(WeightQuantBatchmatmul,
                         OP_INPUT(x1, x2, diagonalMatrix, deqOffset, deqOffset, bias),
                         OP_OUTPUT(weightQuantBmmOut),
                         OP_ATTR(adjX1, adjX2, -1L, 0L));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
    return nullptr;
  }
  ret = ADD_TO_LAUNCHER_LIST_AICORE(WeightQuantBatchmatmul,
                                    OP_INPUT(x1, x2, diagonalMatrix, deqOffset, deqScale, bias),
                                    OP_OUTPUT(weightQuantBmmOut),
                                    OP_ATTR(adjX1, adjX2, -1L, 0L));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return nullptr;
  }
  return weightQuantBmmOut;
};
}  // namespace l0op
