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
 * \file softplus_v2_proto.h
 * \brief SoftplusV2 operator GE IR registration
 */
#ifndef OPS_OP_PROTO_INC_SOFTPLUSV2_H_
#define OPS_OP_PROTO_INC_SOFTPLUSV2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes Softplus activation: y = (1/beta) * ln(1 + exp(beta * x)) when beta*x <= threshold, else y = x.
 * @par Inputs:
 * One input:
 * @li x: A Tensor. Must be one of the following types: float32, float16, bfloat16.
 *
 * @par Attributes:
 * @li beta: An optional float. Scaling factor, defaults to 1.0.
 * @li threshold: An optional float. Linearization threshold, defaults to 20.0.
 *
 * @par Outputs:
 * y: A Tensor with the same type and shape as x.
 */
REG_OP(SoftplusV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(beta, Float, 1.0)
    .ATTR(threshold, Float, 20.0)
    .OP_END_FACTORY_REG(SoftplusV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SOFTPLUSV2_H_
