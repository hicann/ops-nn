/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file softsign_grad_proto.h
 * \brief SoftsignGrad 算子 IR 原型定义（图模式适配）
 *
 * 数学公式: output = gradients / (1 + |features|)^2
 */
#ifndef OPS_OP_PROTO_INC_SOFTSIGN_GRAD_H_
#define OPS_OP_PROTO_INC_SOFTSIGN_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes gradients for the Softsign activation.
 *
 * output = gradients / (1 + abs(features))^2
 *
 * @par Inputs:
 * @li gradients: A tensor of type float16, float32, or bfloat16. Upstream gradients.
 * @li features: A tensor of type float16, float32, or bfloat16. Forward input features.
 *               Must have the same shape and dtype as gradients.
 *
 * @par Outputs:
 * output: A tensor with the same shape and dtype as gradients.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow/PyTorch Softsign backward operation.
 */
REG_OP(SoftsignGrad)
    .INPUT(gradients, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(SoftsignGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SOFTSIGN_GRAD_H_
