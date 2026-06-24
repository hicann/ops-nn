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
 * \file gru.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_GRU_H_
#define OP_API_INC_LEVEL0_OP_GRU_H_

#include "opdev/op_executor.h"

namespace l0op{
const std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*> Gru(
    const aclTensor* input, const aclTensor* weightInput, const aclTensor* weightHidden,
    const aclTensor* biasInput, const aclTensor* biasHidden,
    const aclTensor* seqLengthOptional,
    const aclTensor* initHOptional,
    const char* direction,
    bool train,
    aclTensor* yOut,
    aclTensor* outputHOut,
    aclTensor* rOut,
    aclTensor* zOut,
    aclTensor* nOut,
    aclTensor* nHOut,
    aclOpExecutor* executor);
}

#endif//OP_API_INC_LEVEL0_OP_GRU_H_