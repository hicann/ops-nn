/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_COMPRESS_H
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_COMPRESS_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* MatmulCompress(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* compressIndex, bool transposeX1,
    bool transposeX2, int32_t compressK, int32_t compressN, aclOpExecutor* executor);
}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_COMPRESS_H