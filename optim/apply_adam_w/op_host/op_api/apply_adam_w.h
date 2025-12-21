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
 * \file apply_adam_w.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_APPLY_ADAM_W_H_
#define OP_API_INC_LEVEL0_OP_APPLY_ADAM_W_H_

#include "opdev/op_executor.h"

namespace l0op {
std::tuple<const aclTensor*, const aclTensor*, const aclTensor*> ApplyAdamW(const aclTensor* varRef,
                                                                            const aclTensor* mRef,
                                                                            const aclTensor* vRef,
                                                                            const aclTensor* beta1Power,
                                                                            const aclTensor* beta2Power,
                                                                            const aclTensor* lr,
                                                                            const aclTensor* weightDecay,
                                                                            const aclTensor* beta1,
                                                                            const aclTensor* beta2,
                                                                            const aclTensor* eps,
                                                                            const aclTensor* grad,
                                                                            const aclTensor* maxGradNormOptional,
                                                                            bool amsgrad, bool maximize,
                                                                            aclOpExecutor *executor);
}

#endif
