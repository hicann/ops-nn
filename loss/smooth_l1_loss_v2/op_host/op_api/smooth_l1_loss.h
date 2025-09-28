/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_INC_LEVEL0_OP_SMOOTH_L1_LOSS_OP_H_
#define OP_API_OP_API_INC_LEVEL0_OP_SMOOTH_L1_LOSS_OP_H_
#include "opdev/fast_vector.h"
#include "opdev/op_executor.h"
namespace l0op {
const aclTensor *SmoothL1Loss(const aclTensor *self, const aclTensor *target, const std::string &reduction, float beta,
                              aclOpExecutor *executor);
}
#endif  // OP_API_OP_API_INC_LEVEL0_OP_SMOOTH_L1_LOSS_OP_H_
