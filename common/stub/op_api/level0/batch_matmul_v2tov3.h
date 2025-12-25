/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_MATMUL_V2TOV3_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_MATMUL_V2TOV3_H_

#include "opdev/op_executor.h"
#include "matmul/common/op_host/op_api/matmul_util.h"

namespace l0op {

bool BmmCheckHitV3Shape(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const bool adjX1,
                        const bool adjX2, op::Format self_format, op::Format mat2_format, const bool enableFp16Bf16InFp32Out);
}

#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_BATCH_MATMUL_V2TOV3_H_