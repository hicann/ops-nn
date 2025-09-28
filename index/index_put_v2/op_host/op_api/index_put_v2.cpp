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
 * \file index_put_v2.cpp
 * \brief
 */
#include "index_put_v2.h"
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

OP_TYPE_REGISTER(IndexPutV2);

const aclTensor *IndexPutV2(const aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values,
                            const aclTensor *masks, const bool accumulate, aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(IndexPutV2, selfRef, values, masks, indices, out, accumulate);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexPutV2,
                                         OP_INPUT(selfRef, values, masks, masks, indices),
                                         OP_OUTPUT(out),
                                         OP_ATTR(accumulate));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexPutV2 ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}
}  // namespace l0op

