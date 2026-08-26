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
 * \file group_norm_silu_quant_proto.h
 * \brief GroupNorm + SiLU followed by per-tensor / per-channel int8 quantization.
 */

#ifndef NORM_GROUP_NORM_SILU_QUANT_PROTO_H_
#define NORM_GROUP_NORM_SILU_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Computes GroupNorm on x, optionally applies SiLU, then quantizes the result to int8.
 *   y = round(silu(group_norm(x, num_groups, gamma, beta, eps)) / quantScale), clamped to [-128, 127].
 *
 * @par Inputs:
 * @li x: Required tensor of type float16 or bfloat16. Shape is (N, C, *), dim0 = N, dim1 = C.
 * @li gamma: Optional 1-D tensor, same dtype as x, element count = C.
 * @li beta: Optional 1-D tensor, same dtype as x, element count = C.
 * @li quantScale: Required 1-D float32 tensor. Element count must be 1 (per-tensor) or C (per-channel).
 *
 * @par Attributes:
 * @li num_groups: Required int. C must be divisible by num_groups.
 * @li eps: Optional float. Defaults to 1e-5. Must be greater than 0.
 * @li activate_silu: Optional bool. Whether to apply SiLU. Defaults to true.
 *
 * @par Outputs:
 * @li yOut: Quantized output, int8, same shape as x.
 * @li meanOut: Group mean, same dtype as x, shape (N, num_groups).
 * @li rstdOut: Reciprocal of group standard deviation, same dtype as x, shape (N, num_groups).
 *
 * @par Third-party framework compatibility
 * It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
 */
#ifndef OPS_PROTO_DEF_GROUPNORMSILUQUANT
#define OPS_PROTO_DEF_GROUPNORMSILUQUANT
REG_OP(GroupNormSiluQuant)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(beta, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(quantScale, TensorType({DT_FLOAT}))
    .OUTPUT(yOut, TensorType({DT_INT8}))
    .OUTPUT(meanOut, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(rstdOut, TensorType({DT_BF16, DT_FLOAT16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(eps, Float, 0.00001f)
    .ATTR(activate_silu, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSiluQuant)
#endif // OPS_PROTO_DEF_GROUPNORMSILUQUANT

} // namespace ge

#endif // NORM_GROUP_NORM_SILU_QUANT_PROTO_H_
