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
 * \file lp_norm_update_v2.cpp
 * \brief
 */
#include "lp_norm_update_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LpNormUpdateV2);

const aclTensor *LpNormUpdateV2(const aclTensor *x,
                                float p,
                                float epsilon,
                                aclOpExecutor *executor) {
  L0_DFX(LpNormUpdateV2, x, p, epsilon);
  auto out = executor->AllocTensor(x->GetViewShape(), x->GetDataType());

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LpNormUpdateV2,
                                         OP_INPUT(x),
                                         OP_OUTPUT(out),
                                         OP_ATTR(p, epsilon));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LpNormUpdateV2 ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}
}