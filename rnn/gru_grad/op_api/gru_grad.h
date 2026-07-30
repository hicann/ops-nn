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
 * \file gru_grad.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_GRU_GRAD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_GRU_GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

constexpr size_t OUT_DX_INDEX = 0;
constexpr size_t OUT_DH_PREV_INDEX = 1;
constexpr size_t OUT_DW_INPUT_INDEX = 2;
constexpr size_t OUT_DW_HIDDEN_INDEX = 3;
constexpr size_t OUT_DB_INPUT_INDEX = 4;
constexpr size_t OUT_DB_HIDDEN_INDEX = 5;
constexpr size_t GRU_GRAD_OUT_NUM = 6;

const std::array<const aclTensor*, GRU_GRAD_OUT_NUM> GruGrad(
    const aclTensor* input, const aclTensor* weightInput, const aclTensor* weightHidden, const aclTensor* initHOptional,
    const aclTensor* outputH, const aclTensor* resetGate, const aclTensor* updateGate, const aclTensor* newGate,
    const aclTensor* hn, const aclTensor* dy, const aclTensor* dh, const aclTensor* seqLengthOptional,
    const char* direction, bool hasBias, bool batchFirst, aclOpExecutor* executor);

} // namespace l0op
#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_GRU_GRAD_OP_H_
