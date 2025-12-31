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
 * \file index_put_with_sort_v2.cpp
 * \brief
 */
#include "index_put_with_sort_v2.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
using namespace op;

namespace l0op {

OP_TYPE_REGISTER(IndexPutWithSortV2);

const aclTensor *IndexPutWithSortV2(const aclTensor *self, const aclTensor *linearIndex, const aclTensor *posIdx,
  const aclTensor *values, const aclIntArray* indexed_sizes, const bool accumulate, 
  aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(IndexPutWithSortV2, self, linearIndex, posIdx, values, indexed_sizes, accumulate);
  
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexPutWithSortV2,
                                         OP_INPUT(self, linearIndex, posIdx, values),
                                         OP_OUTPUT(out),
                                         OP_ATTR(indexed_sizes, accumulate));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                       "IndexPutWithSortV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}
}  // namespace l0op

