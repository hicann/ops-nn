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
 * \file dual_level_quant_batch_matmul_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_DUAL_LEVEL_QUANT_BATCH_MATMUL_PROTO_H_
#define OPS_MATMUL_DUAL_LEVEL_QUANT_BATCH_MATMUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The fusion operator of antiquant function and matmul.

* @par Inputs:
* @li x1: A matrix tensor. Format supports ND. The type supports DT_FLOAT4_E2M1. Shape is [M, K].
* @li x2: A matrix tensor. Format supports NZ. The type supports DT_FLOAT4_E2M1. Shape is [N, K].
* @li x1_level0_scale: A matrix tensor. Format supports ND. The type supports DT_FLOAT.
* First-level quantization parameter for x1. Shape is [M, Ceil(K/level0_group_size)].
* @li x1_level1_scale: A matrix tensor. Format supports ND. The type supports DT_FLOAT8_E8M0.
* Second-level quantization parameter for x1. Shape is [M, Ceil(K/(2*level1_group_size)), 2].
* @li x2_level0_scale: A matrix tensor. Format supports ND. The type supports DT_FLOAT.
* First-level quantization parameter for x2. Shape is [Ceil(K/level0_group_size), N].
* @li x2_level1_scale: A matrix tensor. Format supports ND. The type support DT_FLOAT8_E8M0.
* Second-level quantization parameter for x2. Shape is [N, Ceil(K/(2*level1_group_size)), 2].
* @li bias: An Optional tensor. Shape support (N). Format support ND. The type support DT_FLOAT.

* @par Attributes:
* @li dtype: An int. Declare the output dtype, supports 1(float16), 27(bfloat16).
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to [K, M]. Now only support false.
* Default value is false.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [K, N] to [N, K] and
* "x2_level1_scale" from [Ceil(K/(2*level1_group_size)), N, 2] to [N, Ceil(K/(2*level1_group_size)), 2]. 
* Now only support true. Default value is true.
* @li level0_group_size: An int. First-level quantization parameter. Size supports 512.
* Default value is 512.
* @li level1_group_size: An int. Second-level quantization parameter. Size supports 32.
* Default value is 32.

* @par Outputs:
* y: A matrix tensor. The format support ND. The type support float16, bfloat16.
*/
REG_OP(DualLevelQuantBatchMatmul)
    .INPUT(x1, TensorType({DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_FLOAT4_E2M1}))
    .INPUT(x1_level0_scale, TensorType({DT_FLOAT}))
    .INPUT(x1_level1_scale, TensorType({DT_FLOAT8_E8M0}))
    .INPUT(x2_level0_scale, TensorType({DT_FLOAT}))
    .INPUT(x2_level1_scale, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, true)
    .ATTR(level0_group_size, Int, 512)
    .ATTR(level1_group_size, Int, 32)
    .OP_END_FACTORY_REG(DualLevelQuantBatchMatmul)
} // namespace ge

#endif // OPS_MATMUL_DUAL_LEVEL_QUANT_BATCH_MATMUL_PROTO_H_