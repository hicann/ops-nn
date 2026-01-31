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
 * \file embedding_dense_grad.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_EMBEDDING_DENSE_GRAD_H_
#define OP_API_INC_LEVEL0_EMBEDDING_DENSE_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* EmbeddingDenseGrad(
    const aclTensor* grad, const aclTensor* indices, uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq,
    aclOpExecutor* executor);

const aclTensor* EmbeddingDenseGradV2(
    const aclTensor* grad, const aclTensor* sortIndices, const aclTensor* posIdx, const aclTensor* out, uint64_t numWeights,
    uint64_t paddingIdx, bool scaleGradByFreq, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_EMBEDDING_DENSE_GRAD_H_
