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
 * \file quant_matmul_activation_quant_proto.h
 * \brief
 */
#ifndef OPS_QUANT_MATMUL_ACTIVATION_QUANT_PROTO_H_
#define OPS_QUANT_MATMUL_ACTIVATION_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief The fusion operator of QuantBatchMatmul, Gelu and mxQuant.

* @par Inputs:
 * @li x1: A matrix tensor. Must be one of the following types: float8_e5m2, float8_e4m3fn, float4_e2m1 \n
           when the data type is float8_e5m2, float8_e4m3fn, float4_e2m1, the format only supports ND format. \n
           - In ND format, the shape ranges from 2D to 6D. When transpose_x2 is false, \n
            the shape is (batch,k,n), where batch is optional. \n
 * @li x2: A matrix tensor. Must be one of the following types: float8_e4m3fn, float8_e5m2, float4_e2m1. \n
           when the data type is float8_e4m3fn or float4_e2m1, the format supports NZ and ND format. \n
           when the data type is float8_e5m2, the format only supports ND format. \n
          - In ND format, the shape ranges from 2D to 6D. When transpose_x2 is false, \n
            the shape is (batch,k,n), where batch is optional. \n
          - In NZ (Ascend affinity) format, the shape ranges from 4D to 8D. \n
              - When transpose_x2 is true, the shape is (batch,x2_k1,x2_n1,x2_n0,x2_k0), where batch is optional, x2_k0
= 32, and x2_n0 = 16. \n
              - When transpose_x2 is false, the shape is (batch,x2_n1,x2_k1,x2_k0,x2_n0), where batch is optional, x2_k0
= 16, and x2_n0 = 32. \n
              - When x1 is ND format, k in the shape of x1 and the shape of x2 must meet the following requirement: \n
                    ceil(k / x2_k0) == x2_k1. \n
* @li bias: An optional matrix tensor. Must be float32, supports ND format.
            The shape is 1D (t,) or 3D (batch, 1, n),
            with t equal to n, where n is the same as that of x2.
* @li x1_scale: An optional matrix tensor. The type supports float8_e8m0, supports ND format. \n
                      - When the data type is float8_e8m0, the shape is 3D. When the shape of x1 is (m, k), scale is (m,
z, 2), when the shape of x1 is (k, m), scale is (z, m, 2), where z = ceil(k / 64) and k is the reduce axis of x1. \n
* @li x2_scale: A matrix tensor, quantization parameter,
             Must be one of the following types: float8_e8m0, supports ND format. \n
            - When the data type is float8_e8m0, the shape is 3D. When the shape of x2 is (n, k), scale is (n, z, 2),
             when the shape of x2 is (k, n), scale is (z, n, 2), where z = ceil(k / 64) and k is the reduce axis of x2.
\n


* @par Attributes:
* @li transpose_x1: A bool. If true, changes the shape of "x1" from [m, k] to [k, m] before multiplication.
* Default: false. Only supports false now.
* @li transpose_x2: A bool. If true, changes the shape of "x2" from [k, n] to [n, k] before multiplication.
* Default: false. Only supports false now.
* @li group_size: An optional Int. Indicating the ratio between pertoken_scale/scale and x1/x2 in group dequantization.
* If the value of pertoken_scale along the k-dimension is n, one value in pertoken_scale can be used to dequantize n
values in x1 along the k-dimension. \n
* The group_size is composed of the group_size_m, group_size_n, and group_size_k, total occupying 48 bits.
* 0-15 bits of group_size indicate group_size_k, 16-31 bits indicate group_size_n, 32-47 bits indicate group_size_m,
* 48-63 bits of group_size are noneffective. \n
* If any of group_size_m, group_size_n, group_size_k calculated by group_size is 0, recalculate it by
* input shape, eg: group_size_m = m / scale_m （m % scale_m must be 0). \n
* Final group_size_m, group_size_n, group_size_k should satisify following requirements: \n
* In mx quantification, the supported final [group_size_m, group_size_n, group_size_k] combinations are [1, 1, 32]. \n
* @li activation_type: A optional string. The gelu approximation algorithm to use: 'gelu_tanh' or 'gelu_erf', default is
'gelu_tanh'.
* @li y_dtype: A Int. Declare the output dtype.
* @li quant_dtype: An optional string. Declare the quant mode.Support mx, pertoken, pertensor and perchannel. Defaults
to "rint".
* @li round_mode: An optional string. Defaults to "rint".
* @li scale_alg: An optional int.The algorithm for the scale in quantization.Default to 0.
* Support MxFP8(OCP Microscaling Formats (Mx) Specification , count 0) or MxFP8(nvidia-cuBLAS , count 1).
* @li dst_type_max: An optional Float.Max_dtype takes the maximum value of the quant_data_type, or the provided
value.Defaults to 0.


* @par Outputs:
 * @li y: A matrix tensor. The data type is float8_e5m2, float8_e4m3fn, float4_e2m1. The format supports ND. \n
* @li
y_scale: An output tensor of type FLOAT8_E8M0. Shape needs to meet the following conditions: \n
* - rank(mxscale) = rank(x) + 1.
* - axis_change = -1.
* - mxscale.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2.
* - mxscale.shape[rank(x)] = 2.
* - Other dimensions match input x.
* mxscale tensor is padded with zeros to ensure its size along the quantized axis is even.


* @attention Constraints:
* @li The shape of bias should be 1D when the shape of out is 2D, 4D, 5D or 6D, and the shape of bias should be 1D or 3D
* when the out shape is 3D.
* @li Inputs does not support tensor with dimension size 0.
* @li When x2 is ND format, x1 should be ND format.
* @li In mx quantification, when x1 type and x2 type are both float8_e4m3fn/float8_e5m2 and x1_scale/x2_scale types are
float8_e8m0:
*      - x1 and x2 inner axis (the last dimension of view shape, independent of transpose_x1/transpose_x2) must be even.
*      - k must be greater than 2.
 *      - supported final [group_size_m, group_size_n, group_size_k] is [1, 1, 32].
 * @li In mx quantification, when x1 type and x2 type are both float4_e2m1 and x1_scale/x2_scale types are
 * float8_e8m0 (MXFP4 scenario):
 *      - x1 and x2 inner axis (the last dimension of view shape, independent of transpose_x1/transpose_x2) must be
even.
 *      - k must be greater than 2.
 *      - y's N dimension must be even.
 *      - supported final [group_size_m, group_size_n, group_size_k] is [1, 1, 32].
 *      - scale_alg only supports 0 and 2. When scale_alg is 2, dst_type_max supports 0.0 and 6.0-12.0.
 *      - round_mode supports "rint", "floor", "round".
 *      - When x2 is NZ format, x1 does not support transpose, i.e. transpose_x1 cannot be true.
 * @li Only weight supports ND and NZ format on Ascend 950 AI Processor. All other inputs and outputs only support
ND
format.
* @li The following are the supported data type combinations by platform.

* - Ascend 950 AI Processor:
*\n
| x1                        | x2            | bias         | x1_scale   | x2_scale   | out                       |
y_scale    | | :-----------------------: | :-----------: | :----------: | :---------: | :---------: |
:-----------------------: | :---------: | | float8_e4m3fn             | float8_e4m3fn | null/float32 | float8_e8m0 |
float8_e8m0 | float8_e4m3fn             | float8_e8m0 | | float8_e5m2               | float8_e5m2   | null/float32 |
float8_e8m0 | float8_e8m0 | float8_e5m2               | float8_e8m0 | | float8_e4m3fn             | float8_e5m2   |
null/float32 | float8_e8m0 | float8_e8m0 | float8_e4m3fn             | float8_e8m0 | | float8_e5m2               |
float8_e4m3fn | null/float32 | float8_e8m0 | float8_e8m0 | float8_e5m2               | float8_e8m0 | | float4_e2m1 |
float4_e2m1   | null/float32 | float8_e8m0 | float8_e8m0 | float4_e2m1               | float8_e8m0 |
*\n

* - Ascend 950 AI Processor with group_sizes scenarios, supported data type and shapes combinations:
*\n
| quantization      | x1 type                            | x1_scale type  | x1 shape      | x2 shape      | x2_scale
shape                           | x1_scale shape                        | group_size      |
|-------------------|------------------------------------|----------------|---------------|---------------|------------------------------------------|---------------------------------------|-----------------|
| mx                | float8_e4m3fn/float8_e5m2          | float8_e8m0    | (batch, m, k) | (batch, n, k) | (n, ceil(k /
64), 2)                     | (m, ceil(k / 64), 2)                  | [1, 1, 32]      | | mx                |
float8_e4m3fn/float8_e5m2          | float8_e8m0    | (batch, m, k) | (batch, k, n) | (ceil(k / 64), n, 2) | (m, ceil(k
/ 64), 2)                  | [1, 1, 32]      | | mx                | float8_e4m3fn/float8_e5m2          | float8_e8m0 |
(batch, k, m) | (batch, k, n) | (ceil(k / 64), n, 2)                     | (ceil(k / 64), m, 2)                  | [1,
1, 32]      | | mx                | float8_e4m3fn/float8_e5m2          | float8_e8m0    | (batch, k, m) | (batch, n, k)
| (n, ceil(k / 64), 2)                     | (ceil(k / 64), m, 2)                  | [1, 1, 32]      |
| mx                | float4_e2m1                       | float8_e8m0    | (batch, m, k) | (batch, k, n) | (ceil(k /
64), n, 2)                     | (m, ceil(k / 64), 2)                  | [1, 1, 32]      | | mx                |
float4_e2m1                       | float8_e8m0    | (batch, m, k) | (batch, n, k) | (n, ceil(k /
64), 2)                     | (m, ceil(k / 64), 2)                  | [1, 1, 32]      |

*\n
*/
REG_OP(QuantMatmulActivationQuant)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .ATTR(activation_type, String, "gelu_tanh")
    .ATTR(y_dtype, Int, DT_FLOAT8_E4M3FN)
    .ATTR(quant_mode, String, "mx")
    .ATTR(round_mode, String, "rint")
    .ATTR(scale_alg, Int, 0)
    .ATTR(dst_type_max, Float, 0.0)
    .OP_END_FACTORY_REG(QuantMatmulActivationQuant)
} // namespace ge

#endif // OPS_QUANT_MATMUL_ACTIVATION_QUANT_PROTO_H_
