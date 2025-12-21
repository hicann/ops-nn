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
 * \file mse_loss_grad.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_MSE_LOSS_GRAD_OP_H_
#define OP_API_INC_LEVEL0_OP_MSE_LOSS_GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *MseLossGradV1(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor, const aclTensor *out);
const aclTensor *MseLossGradV2(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor, const aclTensor *out);
const aclTensor *MseLossGrad(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor);
}

#endif  // OP_API_INC_LEVEL0_OP_MSE_LOSS_GRAD_OP_H_

