/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_nd.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(GatherNd);

const aclTensor *GatherNd(const aclTensor *self, const aclTensor *indices, aclOpExecutor *executor,
                          bool negativeIndexSupport) {
  L0_DFX(GatherNd, self, indices, negativeIndexSupport);
  // 根据算子语义，推导算子输出shape
  op::Shape outShape;
  int64_t rankSelf = self->GetViewShape().GetDimNum();
  int64_t rankIndices = indices->GetViewShape().GetDimNum();
  int64_t indicesLastElement = indices->GetViewShape().GetDim(rankIndices - 1);

  for (int64_t i = 0; i < rankIndices - 1; ++i) {
    outShape.AppendDim(indices->GetViewShape().GetDim(i));
  }
  for (int64_t i = indicesLastElement; i < rankSelf; ++i) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }

  // 根据推导出的输出shape申请输出tensor
  auto gatherNdOut = executor->AllocTensor(outShape, self->GetDataType(), op::Format::FORMAT_ND);
  auto retAicore =
    ADD_TO_LAUNCHER_LIST_AICORE(GatherNd,
                                OP_INPUT(self, indices),
                                OP_OUTPUT(gatherNdOut),
                                OP_ATTR(negativeIndexSupport));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                       "GatherNd add to aicore launch list failed.");
  return gatherNdOut;
}
} // l0op