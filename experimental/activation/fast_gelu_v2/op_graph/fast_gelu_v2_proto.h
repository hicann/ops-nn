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
 * \file fast_gelu_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FAST_GELU_V2_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FAST_GELU_V2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The FastGeluV2 activation function is 
*        FastGeluV2(x) = x * (sgn(x) * [-0.1444 * (clip(|0.7071 * x|, max=1.769) - 1.769)^2 + 0.5] + 0.5),
*        where sgn(x) function is (x+0.000000000001)/|(x+0.000000000001)|.

* @par Inputs:
* One input, including:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bfloat16, float16, float32

* @par Outputs:
* y: A Tensor. Has the same type as "x".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator FastGeluV2
*/
REG_OP(FastGeluV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(FastGeluV2)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_FAST_GELU_V2_OPS_H_
