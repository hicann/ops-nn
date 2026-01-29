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
 * \file quant_batch_matmul_v3_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_QUANT_BATCH_MATMUL_V3_PROTO_H_
#define OPS_MATMUL_QUANT_BATCH_MATMUL_V3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Quant Batch Matmul Calculation.

* @par Inputs:
* Six inputs, including:
* @li x1: A matrix tensor. Must be one of the following types: int8, int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1. \n
          when the data type is int8, supports ND and NZ formats. \n
          when the data type is int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2 or float4_e2m1, only supports ND format. \n
          - In ND format and Non-int4 type, the shape ranges from 2D to 6D. When transpose_x1 is false, the shape is (batch,m,k), where
          batch is optional; In int4 type, shape only supports 2D. \n
          - In NZ (Ascend affinity) format, the shape ranges from 4D to 8D. When tranpose_x1 is true, the shape is
          (batch,x1_m1,x1_k1,x1_k0,x1_m0), where batch is optional, x1_m0 = 32, and x1_k0 = 16. When transpose_x1 is false, the shape is
          (batch,x1_k1,x1_m1,x1_m0,x1_k0), where batch is optional, x1_m0 = 16, and x1_k0 = 32. \n
          When the data type is int4, float4_e1m2 or float4_e2m1, the last dim must be even. \n
* @li x2: A matrix tensor. Must be one of the following types: int8, int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1. \n
          when the data type is int8, supports ND and NZ formats. \n
          when the data type is int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2 or float4_e2m1, the format only supports ND. \n
          - In ND format and Non-int4 type, the shape ranges from 2D to 6D. When transpose_x2 is false, the shape is (batch,k,n), where
          batch is optional; In int4 type, shape only supports 2D. \n
          - In NZ (Ascend affinity) format, the shape ranges from 4D to 8D. \n
              - When tranpose_x2 is true, the shape is (batch,x2_k1,x2_n1,x2_n0,x2_k0), where batch is optional, x2_k0 = 32, and x2_n0 = 16. \n
              - When transpose_x2 is false, the shape is (batch,x2_n1,x2_k1,x2_k0,x2_n0), where batch is optional, x2_k0 = 16, and x2_n0 = 32. \n
              - When x1 is ND format, k in the shape of x1 and the shape of x2 must meet the following requirement: \n
                    ceil(k / x2_k0) == x2_k1. \n
              - When x1 is NZ format, the shape of x1 and x2 must meet the following requirement: \n
                  - when x1_k0 == x2_k0, x1_k1 == x2_k1, \n
                  - when x1_k0 > x2_k0, ceil((x2_k0 * x2_k1) / x1_k0) == x1_k1, \n
                  - when x1_k0 < x2_k0, ceil((x1_k0 * x1_k1) / x2_k0) == x2_k1. \n
          When the data type is int4, float4_e1m2 or float4_e2m1, the last dim must be even.
* @li scale: A matrix tensor, quantization parameter,
             Must be one of the following types: uint64, float32, int64, bfloat16, float8_e8m0, supports ND format. \n
            - When the data type is bfloat16, uint64 or int64,
             the shape is 1D (t,), with t equal to 1 or n, where n is the same as that of x2. \n
            - When the data type is float8_e8m0,
             the shape is 2D, when the shape of x2 is (n, k), scale is (n, z). Suppose scale_k = ceil(k / 32),
                - When scale_k is an odd number, z = scale_k + 1, \n
                - when scale_k is an even number, z = scale_k. \n
             where 32 is group_size_k(refer to group_size), z must be an even number and z can be equal to (ceil(k / 64) * 2), and k is the reduce axis of x2. \n
            - when the data type is float32,
             the dimension of shape should be 1D or same as that of x2. \n
               - When the quant mode of x2 is perchannel or pertensor, the shape is 1D (t,),
                 with t equal to 1 or n, where n is the same as that of x2. \n
               - When the quant mode of x2 is perblock, the dimension of shape is same as x2, when x2 is (batch, k, n),
                 scale is (batch, ceil(k / 128), ceil(n / 128)), where 128 is group_size_k, group_size_n(refer to group_size). \n
* @li offset: An optional matrix tensor, quantization parameter. Must be one of the following types: float32.
              supports ND format. The shape is 1D (t,), with t equal to 1 or n, where n is the same as that of x2.
* @li bias: An optional matrix tensor. Must be one of the following types: int32, bfloat16, float16, float32, supports ND format.
            The shape is 1D (t,) or 3D (batch, 1, n),
            with t equal to n, where n is the same as that of x2.
* @li pertoken_scale: An optional matrix tensor. The type supports float32, float8_e8m0, supports ND format. \n
                      - When the data type is float8_e8m0, the shape is 2D,
                        when the shape of x1 is (m, k), scale is (m, z), with z = (ceil(k / 64) * 2) which is same to
                        scale, and k is the reduce axis of x1.
                      - When the data type is float32, the dimension of shape should be 1D or same as that of x1. \n,
                        - When the quant mode of x1 is pertoken or pertensor, the shape is 1D (t,),
                          with t equal to 1 or m, where m is the same as that of x1. \n
                        - When the quant mode of x1 is perblock or pergroup, the dimension of shape is same as x1,
                          when x2 is (batch, m, k), for perblock, pertoken_scale is (batch, ceil(m / 128), ceil(k / 128)),
                          for pergroup, pertoken_scale is (batch, m, ceil(k / 128)),
                          where 128 is group_size_m, group_size_k(refer to group_size). \n

* @par Attributes:
* Four attributes, including:
* @li dtype: An Int. Declare the output type, supports 0(float32), 1(float16), 2(int8), 27(bfloat16),
* 34(hifloat8), 36(float8_e4m3fn). Default: 2(int8).
* @li transpose_x1: An optional bool. If true, changes the shape of "x1" from [m, k] to
* [k, m] before multiplication. Default: false.
* @li transpose_x2: An optional bool. If true, changes the shape of "x2" from [k, n] to
* [n, k] before multiplication. Default: false.
* @li group_size: An optional Int. Indicating the ratio between pertoken_scale/scale and x1/x2 in group dequantization.
* If the value of pertoken_scale along the k-dimension is n, one value in pertoken_scale can be used to dequantize n values in x1 along the k-dimension. \n
* The group_size is composed of the group_size_m, group_size_n, and group_size_k, total occupying 48 bits.
* 0-15 bits of group_size indicate group_size_k, 16-31 bits indicate group_size_n, 32-47 bits indicate group_size_m,
* 48-63 bits of group_size are noneffective. \n
* If any of group_size_m, group_size_n, group_size_k calculated by group_size is 0, recalculate it by
* input shape, eg: group_size_m = m / scale_m （m % scale_m must be 0). \n
* Final group_size_m, group_size_n, group_size_k should satisify following requirements: \n
* In mx quantification, the supported final [group_size_m, group_size_n, group_size_k] combinations are [1, 1, 32]. \n
* In pergroup-perblock && perblock-perblock quantification,
* the supported final [group_size_m, group_size_n, group_size_k] combinations are [1, 128, 128] and [128, 128, 128].
* For other input types, the group_size value must be 0. \n

* @par Outputs:
* One output, including:
* y: A matrix Tensor. Must be one of the following types: float16, int8, bfloat16, int32, float32, hifloat8, float8_e4m3fn.
     The format supports ND. The shape ranges from 2D to 6D,
     that is (batch, m, n), where batch is optional. Broadcasting can be performed on the batch dimension of x1 and x2.
     The output batch is the same as the batch after broadcasting, m is the same as that of x1, and n is the same as
     that of x2. \n

* @attention Constraints:
* @li The shape of bias should be 1D when the shape of out is 2D, 4D, 5D or 6D, and the shape of bias should be 1D or 3D
* when the out shape is 3D.
* @li The size of the last dimension of x1 and x2 cannot exceed 65535 only on the following computing platforms:
* Atlas A2 Training Series Product/Atlas A2 Inference Series Product and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
* The last dimension of x1 refers to m when transpose_x1 is true or k when transpose_x1 is false. 
* The last dimension of x2 refers to k when transpose_x2 is true or n when transpose_x2 is false.
* @li If input type of x1 and x2 is int4, transpose_x1 should be false, the size of the last dimension of x1 or x2 should
* be an even number.
* @li Input does not support tensor with dimension size 0.
* @li When input type of x1 and x2 is int4, x1 should be ND format.
* @li When y type is int8, x1 should be ND format.
* @li When x2 is ND format, x1 should be ND format.
* @li The following are the supported data type combinations by platform.

* - Atlas Inference Series Product:
*\n
| x1       | x2       | scale        | offset        | bias          | pertoken | out      |
| :------: | :------: | :----------: | :-----------: | :-----------: | :------: | :------: |
| int8     | int8     | uint64/int64 | null          | null/int32    | null     | float16  |
| int8     | int8     | uint64/int64 | null/float32  | null/int32    | null     | int8     |
*\n
* - Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product:
*\n
| x1       | x2       | scale            | offset        | bias                        | pertoken     | out      |
| :------: | :------: | :--------------: | :-----------: | :-------------------------: | :----------: | :------: |
| int8     | int8     | uint64/int64     | null          | null/int32                  | null         | float16  |
| int8     | int8     | uint64/int64     | null/float32  | null/int32                  | null         | int8     |
| int8     | int8     | float32/bfloat16 | null          | null/int32/bfloat16/float32 | null/float32 | bfloat16 |
| int8     | int8     | float32          | null          | null/int32/float16/float32  | float32      | float16  |
| int4     | int4     | uint64/int64     | null          | null/int32                  | null         | float16  |
| int8     | int8     | float32/bfloat16 | null          | null/int32                  | null         | int32    |
| int4     | int4     | float32/bfloat16 | null          | null/int32/bfloat16/float32 | float32      | bfloat16 |
| int4     | int4     | float32          | null          | null/int32/float16/float32  | float32      | float16  |
*\n
* - Ascend 950 AI Processor:
*\n
| x1                        | x2                        | scale                | offset        | bias                        | pertoken    | out                                    |
| :-----------------------: | :-----------------------: | :------------------: | :-----------: | :-------------------------: | :---------: | :------------------------------------: |
| int8                      | int8                      | uint64/int64         | null          | null/int32                  | null        | float16/bfloat16                       |
| int8                      | int8                      | uint64/int64         | null/float32  | null/int32                  | null        | int8                                   |
| int8                      | int8                      | float32/bfloat16     | null          | null/int32/float32/bfloat16 | null/float32| bfloat16                               |
| int8                      | int8                      | float32              | null          | null/int32/float32/float16  | float32     | float16                      |
| hifloat8                  | hifloat8                  | uint64/int64         | null          | null/float32                | null        | hifloat8/float16/bfloat16/float32      |
| hifloat8                  | hifloat8                  | float32              | null          | null/float32                | float32     | float16/bfloat16/float32               |
| float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | uint64/int64         | null          | null/float32                | null        | float8_e4m3fn/float16/bfloat16/float32 |
| float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32              | null          | null/float32                | float32     | float16/bfloat16/float32               |
| float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0          | null          | null/float32                | float8_e8m0 | float16/bfloat16/float32               |
| float4_e2m1/float4_e1m2   | float4_e2m1/float4_e1m2   | float8_e8m0          | null          | null/float32                | float8_e8m0 | float16/bfloat16/float32               |
*\n
* - Ascend 950 AI Processor, supported data type and quant mode combinations:
*\n
pertensor-perchannel && pertensor-pertensor:
*\n
  | x1 type                   | x2 type                   | pertoken type    | scale type       |
  | ------------------------- | ------------------------- | ---------------- | ---------------- |
  | int8                      | int8                      | null             | uint64/int64     |
  | int8                      | int8                      | null             | float32/bfloat16 |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | null             | uint64/int64     |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32          | float32          |
  | hifloat8                  | hifloat8                  | null             | uint64/int64     |
  | hifloat8                  | hifloat8                  | float32          | float32          |
*\n
pertoken-perchannel && pertoken-pertensor
*\n
  | x1 type                   | x2 type                   | pertoken type    | scale type       |
  | ------------------------- | ------------------------- | ---------------- | ---------------- |
  | int8                      | int8                      | float32          | float32/bfloat16 |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32          | float32          |
  | hifloat8                  | hifloat8                  | float32          | float32          |
*\n
pergroup-perblock && perblock-perblock:
*\n
  | x1 type                   | x2 type                   | pertoken type    | scale type       |
  | ------------------------- | ------------------------- | ---------------- | ---------------- |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32          | float32          |
  | hifloat8                  | hifloat8                  | float32          | float32          |
*\n
mx quant：
*\n
  | x1 type                   | x2 type                   | pertoken type    | scale type       |
  | ------------------------- | ------------------------- | ---------------- | ---------------- |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0      | float8_e8m0      |
  | float4_e2m1/float4_e1m2   | float4_e2m1/float4_e1m2   | float8_e8m0      | float8_e8m0      |
*\n
* - Ascend 950 AI Processor with group_sizes scenarios, supported data type and shapes combinations:
*\n
| quantization      | x1 type                            | scale type  | x1 shape      | x2 shape      | scale shape                           | pertoken shape                        | group_size      |
|-------------------|------------------------------------|-------------|---------------|---------------|---------------------------------------|---------------------------------------|-----------------|
| perblock-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, m, k) | (batch, k, n) | (batch, ceil(k / 128), ceil(n / 128)) | (batch, ceil(m / 128), ceil(k / 128)) | [128, 128, 128] |
| perblock-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, m, k) | (batch, n, k) | (batch, ceil(n / 128), ceil(k / 128)) | (batch, ceil(m / 128), ceil(k / 128)) | [128, 128, 128] |
| perblock-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, k, m) | (batch, k, n) | (batch, ceil(k / 128), ceil(n / 128)) | (batch, ceil(k / 128), ceil(m / 128)) | [128, 128, 128] |
| perblock-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, k, m) | (batch, n, k) | (batch, ceil(n / 128), ceil(k / 128)) | (batch, ceil(k / 128), ceil(m / 128)) | [128, 128, 128] |
| pergroup-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, m, k) | (batch, k, n) | (batch, ceil(k / 128), ceil(n / 128)) | (batch, m, ceil(k / 128))             | [1, 128, 128]   |
| pergroup-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, m, k) | (batch, n, k) | (batch, ceil(n / 128), ceil(k / 128)) | (batch, m, ceil(k / 128))             | [1, 128, 128]   |
| pergroup-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, k, m) | (batch, k, n) | (batch, ceil(k / 128), ceil(n / 128)) | (batch, ceil(k / 128), m)             | [1, 128, 128]   |
| pergroup-perblock | float8_e4m3fn/float8_e5m2/hifloat8 | float32     | (batch, k, m) | (batch, n, k) | (batch, ceil(n / 128), ceil(k / 128)) | (batch, ceil(k / 128), m)             | [1, 128, 128]   |
| mx                | float8_e4m3fn/float8_e5m2          | float8_e8m0 | (batch, m, k) | (batch, n, k) | (n, ceil(k / 64) * 2)                 | (m, ceil(k / 64) * 2)                 | [1, 1, 32]      |
| mx                | float4_e2m1/float4_e1m2            | float8_e8m0 | (batch, m, k) | (batch, n, k) | (n, ceil(k / 64) * 2)                 | (m, ceil(k / 64) * 2)                 | [1, 1, 32]      |

*\n
*/
REG_OP(QuantBatchMatmulV3)
    .INPUT(x1, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .INPUT(scale, TensorType({DT_UINT64, DT_FLOAT, DT_INT64, DT_BF16, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_INT32, DT_FLOAT, DT_HIFLOAT8, DT_FLOAT8_E4M3FN}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(QuantBatchMatmulV3)
} // namespace ge

#endif // OPS_MATMUL_QUANT_BATCH_MATMUL_V3_PROTO_H_