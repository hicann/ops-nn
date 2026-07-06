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
 * \file fused_sgd.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_FUSED_SGD_H_
#define OP_API_INC_LEVEL0_OP_FUSED_SGD_H_

#include "opdev/op_executor.h"

namespace l0op {
std::tuple<const aclTensorList*, const aclTensorList*, const aclTensorList*> FusedSgd(
    const aclTensorList* paramsRef, const aclTensorList* gradsRef, const aclTensorList* momentumBufferListOptionalRef,
    const aclTensor* gradScaleOptional, float weightDecay, float momentum, float lr, float dampening, bool nesterov,
    bool maximize, bool isFirstStep, aclOpExecutor* executor);
}

#endif
