/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "embedding.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Embedding);
const aclTensor* EmbeddingAiCore(
    const aclTensor* self, const aclTensor* indices, aclTensor* embeddingOut, aclOpExecutor* executor)
{
    L0_DFX(EmbeddingAiCore, self, indices);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Embedding, OP_INPUT(self, indices), OP_OUTPUT(embeddingOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EmbeddingAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return embeddingOut;
}

const aclTensor *Embedding(const aclTensor *self, const aclTensor *indices, aclOpExecutor *executor) {
  size_t selfDim = self->GetViewShape().GetDimNum() > 0 ? self->GetViewShape().GetDimNum() : 1;
  size_t axis = 0;
  size_t batchDims = 0;
  // 根据算子语义，推导算子输出shape
  op::Shape outShape;
  for (size_t i = batchDims; i < indices->GetViewShape().GetDimNum(); i++) {
    outShape.AppendDim(indices->GetViewShape().GetDim(i));
  }
  for (size_t i = axis + 1; i < selfDim; i++) {
    outShape.AppendDim(self->GetViewShape().GetDim(i));
  }

  // 当self是零维tensor时，上述推导公式不再适用，不管一维indices中有多少个0，out始终是零维tensor
  if (self->GetViewShape().GetDimNum() == 0) {
    outShape = self->GetViewShape();
  }
  // 根据推导出的输出shape申请输出tensor
  auto embeddingOut = executor->AllocTensor(outShape, self->GetDataType(), op::Format::FORMAT_ND);
  return EmbeddingAiCore(self, indices, embeddingOut, executor);
}
} // l0op
