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
 * \file index_put.cpp
 * \brief
 */

#include "index_put.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(IndexPut);

const aclTensor *IndexPut(const aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values,
                          const aclTensor *masks, const bool accumulate, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(IndexPut, selfRef, values, masks, indices, accumulate, out);
  static internal::AicpuTaskSpace space("IndexPut");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(IndexPut, OP_ATTR_NAMES({"accumulate"}),
                                        OP_INPUT(selfRef, values, masks, indices),
                                        OP_OUTPUT(out),
                                        OP_ATTR(accumulate));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexPut ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}
}  // namespace l0op

