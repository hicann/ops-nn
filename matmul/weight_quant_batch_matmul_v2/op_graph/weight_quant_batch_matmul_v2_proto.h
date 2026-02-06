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
 * \file weight_quant_batch_matmul_v2_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_WEIGHT_QUANT_BATCH_MATMUL_V2_PROTO_H_
#define OPS_MATMUL_WEIGHT_QUANT_BATCH_MATMUL_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The fusion operator of antiquant function and matmul.

* @par Inputs:
* @li x: A matrix tensor. Shape supports (m,k)/(k,m), Format supports ND.
* The type support float16, bfloat16. The m value must be in [1, 2147483647]. The k value
* must be at least 1.
* @li weight: A matrix tensor of quantized weight. Shape supports (n,k)/(k,n),
* Format supports ND/NZ. The type support int8, int4, int32, float8_e4m3fn, hifloat8, float4_e2m1. \n
* Format must be ND when the type is float8_e4m3fn or hifloat8.
* For Ascend 950 AI Processor, transpose_weight must be false when format is NZ. \n
* The k, n value must be at least 1.
* The k value must be even when type is int4/float4_e2m1 and transpose_weight
* is true, and the n value must be even when type is int4/float4_e2m1 and transpose_weight
* is false. When type is int32, the input is int4-packed data,
* and shape supports (n, k/8)/(k, n/8).
* @li antiquant_scale: A tensor for antiquant scale.
* For different anti quantization modes, the per_tensor mode's shape supports
* (1)/(1,1), the per_channel mode's shape supports (n,)/(1,n)/(n,1), the per_group
* modes' shape supports (ceil(k/antiquant_group_size),n)/
* (n,ceil(k/antiquant_group_size)). Format supports ND. The type support
* float16, bfloat16, uint64 and int64. \n
* When the type is float16 or bfloat16, it should be the same with x. When the type is uint64 or
* int64, there are the following constraints: \n
* 1. antiquant_scale only support the per_channel mode. \n
* 2. x's type should be float16. \n
* 3. transpose_x should be false and transpose_weight should be true. \n
* 4. the m value must be in [1, 96] and the k, n value must be 64 aligned. \n
* 5. weight's type should be int8 and format should be ND. \n
* Additionally, the type supports float8_e8m0 when the weight dtype is float4_e2m1.
* @li antiquant_offset: An Optional tensor for antiquant offset. Shape, format
* and type should be the same with antiquant_scale if antiquant_scale's type is
* not uint64/int64. If antiquant_scale's type is uint64/int64, antiquant_offset should be int32.
* When weight dtype is float8_e4m3fn, hifloat8, float4_e2m1, antiquant_offset is ​excluded from selection.
* @li quant_scale: An Optional tensor for quantization parameters.
* Shape supports (1)/(1,n), format supports ND.
* The type support float32, uint64.
* This parameter must exist when type of output is int8.
* This parameter must not exist when type of antiquant_scale is uint64/int64.
* @li quant_offset: An Optional tensor for quantization parameters.
* Shape and format is the same with quant_scale.
* The type support float32.
* This parameter must not exist when type of antiquant_scale is uint64/int64.
* @li bias: An Optional tensor. Shape supports (n)/(1,n), Format supports ND.
* When type of x is float16, the type of bias should be float16. When type of x
* is bfloat16, the type of bias should be bfloat16. When type of x is bfloat16
* and weight is int4 or int8, the type of bias should be float32. \n
* Specifically, these optional inputs support the shape (0,). At this point,
* it means that the optional input doesn't exist.

* @par Attributes:
* @li transpose_x: A bool. x is transposed if true. Default: false.
* When transpose_x is true, x's shape is (k, m).
* @li transpose_weight: A bool. weight is transposed if true. Default: false.
* When transpose_weight is true, weight's shape is (n, k),
* antiquant_scale's shape should be (1)/(1,1)/(n,)/(n,1)/(n,ceil(k/antiquant_group_size)).
* When transpose_weight is false, weight's shape is (k, n),
* antiquant_scale's shape should be (1)/(1,1)/(n,)/(1,n)/(ceil(k/antiquant_group_size),n).
* @li antiquant_group_size: int, antiquant_group_size must in [0, k-1]
* and antiquant_group_size % 32 == 0. \n
* When antiquant_scale dtype is float8_e8m0, antiquant_group_size must be 32. \n
* When the antiquant_group_size is 0, it means that the per-group mode is not used. Default: 0. \n
* @li dtype: An int. Declare the output dtype, supports 1(float16), 2(int8), 27(bfloat16).
* Default: -1, if the input has quant_scale, the output dtype is int8,
* otherwise the output dtype is the same as the input x dtype.
* @li inner_precise: An int. Calculation mode, only supports 0(high precision), 1(high Performance). Default: 0

* @par Outputs:
* y: A matrix tensor. The type supports float16, bfloat16, int8. The format supports ND.
* The type should be int8 when quant_scale exist.
* The type should be the same with x when quant_scale not exits.

* @attention Constraints:
* @li It is not recommended to use weight NZ format on Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component,
* because its performance may not be better than ND format.
* @li All of these conditions must be met on Atlas Inference Series Product: weight type is int8,
* weight format is NZ, transpose_weight is true, x type is float16, antiquant_scale only support the per_channel mode,
* quant_scale not exists, quant_offset not exists, antiquant_group_size is 0.
* @li Per_channel mode: To improve performance, it is recommended to use the
* weight input after transposition. When the range of m is [65,96], it is
* recommended to use the antiquant_scale with data type UINT64/INT64.
*/
REG_OP(WeightQuantBatchMatmulV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(
        weight, TensorType(
                    {DT_INT8, DT_INT4, DT_INT32, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT, DT_UINT64}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(dtype, Int, -1)
    .ATTR(inner_precise, Int, 0)
    .OP_END_FACTORY_REG(WeightQuantBatchMatmulV2)
} // namespace ge

#endif // OPS_MATMUL_WEIGHT_QUANT_BATCH_MATMUL_V2_PROTO_H_