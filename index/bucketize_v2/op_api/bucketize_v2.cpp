/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bucketize_v2.cpp
 * \brief
 */
#include "bucketize_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(BucketizeV2);
// AICORE算子kernel
const aclTensor *BucketizeV2(const aclTensor *self, const aclTensor *boundaries, const bool outInt32, 
                            const bool right, aclOpExecutor *executor) {
  L0_DFX(BucketizeV2, self, boundaries, outInt32, right);
  op::DataType outDtype = outInt32 ? op::DataType::DT_INT32 : op::DataType::DT_INT64; 
  auto bucketizeOut = executor->AllocTensor(self->GetViewShape(), outDtype);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BucketizeV2,
                                         OP_INPUT(self, boundaries),
                                         OP_OUTPUT(bucketizeOut),
                                         OP_ATTR(outInt32),
                                         OP_ATTR(right));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BucketizeV2 ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return bucketizeOut;
}
}  // namespace l0op
