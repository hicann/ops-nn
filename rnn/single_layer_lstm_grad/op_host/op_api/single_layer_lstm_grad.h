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
 * \file single_layer_lstm_grad.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_SINGLE_LAYER_LSTM_GRAD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_SINGLE_LAYER_LSTM_GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 5> SingleLayerLstmGrad(
    const aclTensor *x,
    const aclTensor *w,
    const aclTensor *b,
    const aclTensor *y,
    const aclTensor *initH,
    const aclTensor *initC,
    const aclTensor *h,
    const aclTensor *c,
    const aclTensor *dy,
    const aclTensor *dh,
    const aclTensor *dc,
    const aclTensor *i,
    const aclTensor *j,
    const aclTensor *f,
    const aclTensor *o,
    const aclTensor *tanhc,
    const aclTensor *seqLength,
    const char *direction,
    const char *gateOrder,
    aclOpExecutor *executor);

}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_SINGLE_LAYER_LSTM_GRAD_OP_H_
