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
 * \file l2_normalize_grad_proto.h
 * \brief
 */
#ifndef OPS_NORM_L2_NORMALIZE_GRAD_PROTO_H_
#define OPS_NORM_L2_NORMALIZE_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief L2NormalizeGrad: gradient of L2Normalize (y = x / max(sqrt(sum(x^2, dim)), eps)).
 *  dx = (dy - y * sum(y * dy, dim)) / max(sqrt(sum(x^2, dim)), eps)

 * @par Inputs
 * Three inputs, including:
 * @li x: The forward input.
 *     A Tensor. Support dtype: [float32, float16], support format: [ND].
 * @li y: The forward output (= normalized x). Same shape/dtype as x.
 *     A Tensor. Support dtype: [float32, float16], support format: [ND].
 * @li dy: The gradient returned backward. Same shape/dtype as x.
 *     A Tensor. Support dtype: [float32, float16], support format: [ND].

 * @par Attributes
 * @li dim: An optional list-int attribute, the normalization/reduction axes. Defaults to {}.
 *     Each element must be within [-x.dim(), x.dim()-1]; a negative value is equivalent to
 *     "value + x.dim()". Duplicated/unordered elements are folded, de-duplicated and sorted,
 *     so {1, 1} and {1, -2} (rank 3) are equivalent to {1}. An empty attribute means "no
 *     reduction": dx degenerates to the element-wise form. The resolved axes must form a
 *     contiguous range (e.g. {1}, {1, 2}); a set with non-adjacent axis indices such as {0, 2} is not supported.
 *     The length of the attribute must not exceed 20.
 * @li eps: An optional float attribute, the denominator floor. Defaults to 1e-4.

 * @par Outputs
 * @li dx: The gradient of input "x". Has the same type and shape as "x".
 *     A Tensor. Support dtype: [float32, float16], support format: [ND].

 * @par Third-party framework compatibility
 * Compatible with the L2 scenario of the PyTorch operator NormalizeGrad.
 */
#ifndef OPS_PROTO_DEF_L2NORMALIZEGRAD
#define OPS_PROTO_DEF_L2NORMALIZEGRAD
REG_OP(L2NormalizeGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(dim, ListInt, {})
    .ATTR(eps, Float, 0.0001f)
    .OP_END_FACTORY_REG(L2NormalizeGrad)
#endif // OPS_PROTO_DEF_L2NORMALIZEGRAD

} // namespace ge
#endif // OPS_NORM_L2_NORMALIZE_GRAD_PROTO_H_
