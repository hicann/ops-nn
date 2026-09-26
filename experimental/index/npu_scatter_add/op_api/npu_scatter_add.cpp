/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_scatter_add.cpp
 * \brief NpuScatterAdd Level0 实现：ADD_TO_LAUNCHER_LIST_AICORE 下发 AICORE kernel
 */
#include "npu_scatter_add.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(NpuScatterAdd);
// AICORE算子kernel
const aclTensor* NpuScatterAddAiCore(const aclTensor* x, const aclTensor* y, const aclTensor* s,
                                     const aclTensor* indices, const aclTensor* sortIdx, const aclTensor* validTokenNum,
                                     const bool useHighPrecision, aclOpExecutor* executor)
{
    L0_DFX(NpuScatterAddAiCore, x, y, s, indices, sortIdx, validTokenNum);
    auto npuScatterAddOut = const_cast<aclTensor*>(y);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(NpuScatterAdd, OP_INPUT(x, y, s, indices, sortIdx, validTokenNum),
                                           OP_OUTPUT(npuScatterAddOut), OP_ATTR(useHighPrecision));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "NpuScatterAddAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return npuScatterAddOut;
}
} // namespace l0op
