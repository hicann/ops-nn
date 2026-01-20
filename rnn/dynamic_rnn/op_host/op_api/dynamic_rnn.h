/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_OP_DYNAMICRNN_OP_H_
#define OP_API_INC_LEVEL0_OP_DYNAMICRNN_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*> DynamicRNN(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias, 
    const aclTensor *initHOptional,
    const aclTensor *initCOptional,
    const aclTensor *seqLengthOptional,
    const char *direction,
    bool train,
    aclTensor *yOutDirec,
    aclTensor *iOutDirec,
    aclTensor *jOutDirec,
    aclTensor *fOutDirec,
    aclTensor *oOutDirec,
    aclTensor *hOutDirec,
    aclTensor *cOutDirec,
    aclTensor *tanhCOutDirec,
    aclOpExecutor* executor);
}
#endif
