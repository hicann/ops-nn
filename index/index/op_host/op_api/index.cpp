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
 * \file index.cpp
 * \brief
 */

#include "index.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Index);

const aclTensor *IndexAiCore(const aclTensor *self,
                             const aclTensor *indexedSizes,
                             const aclTensor *indexedStrides,
                             const op::Shape outPutShape,
                             const aclTensorList*indices,
                             aclOpExecutor *executor) {
  L0_DFX(IndexAiCore, self, indexedSizes, indexedStrides, indices);
  // 根据推导出的输出shape申请输出tensor
  // 第一个参数是输出shape，第二个参数是输出的dtype
  auto out = executor->AllocTensor(outPutShape, self->GetDataType());

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Index,
                                         OP_INPUT(self, indexedSizes, indexedStrides, indices),
                                         OP_OUTPUT(out));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

const aclTensor *IndexAiCpu(const aclTensor *self,
                            const aclTensor *indexedSizes,
                            const aclTensor *indexedStrides,
                            const op::Shape outPutShape,
                            const aclTensorList*indices,
                            aclOpExecutor *executor) {
  L0_DFX(IndexAiCpu, self, indexedSizes, indexedStrides, indices);
  // 根据推导出的输出shape申请输出tensor
  // 第一个参数是输出shape，第二个参数是输出的dtype
  auto out = executor->AllocTensor(outPutShape, self->GetDataType());
  static internal::AicpuTaskSpace space("Index");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Index, OP_ATTR_NAMES(),
                                        OP_INPUT(self, indexedSizes, indexedStrides, indices),
                                        OP_OUTPUT(out));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}
}  // namespace l0op