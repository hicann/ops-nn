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
 * \file nn_quantize.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FLAT_QUANT_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FLAT_QUANT_H_
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Flatness matters for LLM quantization. \n
 *
 * @par Inputs:
 * @li x: A tensor. The original data input. 3-D with shape [K, M, N]. Must be one of the following types:
 * float16, bfloat16. The format support ND.
 * @li kronecker_p1: A tensor. Input calculation matrix 1. 2-D with shape [M, M]. The value of M is same as input "x".
 * Must be one of the following types: float16, bfloat16. Has the same type as input "x".
 * The format support ND.
 * @li kronecker_p2: A tensor. Input calculation matrix 2. 2-D with shape [N, N]. The value of N is same as input "x".
 * Must be one of the following types: float16, bfloat16. Has the same type as input "x".
 * The format support ND. \n
 *
 * @par Outputs:
 * @li out: When the output data type is int4,the shape is consistent with the input parameter x.When the output data
 type is int32,the shape is [K,M,N/8].When the output data type is float4_e2m1,the shape is [K,M*N].
 * @li quant_scale: When the output data type is int4 or int32,the shape is [K].When the output data type is
 float4_e2m1,the shape is [K,ceilDiv(M*N,64),2].
 * The values of K,M,and N are same as that of "x". \n

 * @par Attributes:
 * clip_ratio: An optional float. Used to control the quantization cropping ratio. Defaults to 1. \n
 * dst_dtype: An optional int. Used to control the quantization dst_type. Defaults to 29. \n
 */
REG_OP(FlatQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(kronecker_p1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(kronecker_p2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(out, TensorType({DT_INT4, DT_FLOAT4_E2M1}))
    .OUTPUT(quant_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .ATTR(clip_ratio, Float, 1)
    .ATTR(dst_dtype, Int, DT_INT4)
    .OP_END_FACTORY_REG(FlatQuant)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_FLAT_QUANT_H_
