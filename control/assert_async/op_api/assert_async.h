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
 * \file assert_async.h
 * \brief AssertAsync L0 interface.
 */

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_ASSERT_ASYNC_H_
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_ASSERT_ASYNC_H_

#include <string>

#include "opdev/op_executor.h"

namespace l0op {
/**
 * \brief Assert inputCondition (numel == 1) is non-zero; on failure the device prints assertMsg and traps.
 * \param inputCondition Single-element input tensor.
 * \param assertMsg Message printed on failure.
 * \param executor Op executor.
 * \return ACLNN_SUCCESS on success; ACLNN_ERR_INNER_NULLPTR if ADD_TO_LAUNCHER_LIST_AICORE fails.
 */
aclnnStatus AssertAsync(const aclTensor* inputCondition, const std::string& assertMsg, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_ASSERT_ASYNC_H_
