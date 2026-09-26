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
 * \file npu_scatter_add_bwd.cpp
 * \brief NpuScatterAddBwd Level0 实现：ADD_TO_LAUNCHER_LIST_AICORE 下发 AICORE kernel
 */
#include "npu_scatter_add_bwd.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(NpuScatterAddBwd);
// AICORE算子kernel
std::tuple<aclTensor*, aclTensor*> NpuScatterAddBwdAiCore(const aclTensor* yGrad, const aclTensor* x,
                                                          const aclTensor* s, const aclTensor* indices,
                                                          const aclTensor* xGrad, const aclTensor* sGrad,
                                                          aclOpExecutor* executor)
{
    L0_DFX(NpuScatterAddBwdAiCore, yGrad, x, s, indices);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(NpuScatterAddBwd, OP_INPUT(yGrad, x, s, indices), OP_OUTPUT(xGrad, sGrad));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "NpuScatterAddBwdAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return std::make_tuple(nullptr, nullptr));
    return std::make_tuple(const_cast<aclTensor*>(xGrad), const_cast<aclTensor*>(sGrad));
}
} // namespace l0op
