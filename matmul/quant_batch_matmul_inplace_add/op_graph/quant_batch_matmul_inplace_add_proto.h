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
 * \file quant_batch_matmul_inplace_add_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_QUANT_BATCH_MATMUL_INPLACE_ADD_PROTO_H_
#define OPS_MATMUL_QUANT_BATCH_MATMUL_INPLACE_ADD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Quant Batch Matmul Calculation With Inplace Add.
* @par Inputs:
* Five inputs, including:
* @li x1: A tensor. Format supports ND. Must be one of the following types: hifloat8, float8_e5m2, float8_e4m3fn,
* float4_e2m1.
* In ND format, the shape ranges from 2D to 6D. When transpose_x1 is false, the shape is (m,k), when
* transpose_x1 is true, the shape is (k,m).
* @li x2: A tensor. Must be one of the following types: hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1.
* In ND format, the shape ranges from 2D to 6D. When transpose_x2 is false, the shape is (m,k), when
* transpose_x1 is true, the shape is (k,m).
* @li x2_scale: A tensor, quantization parameter. Must be float32 or float8_e8m0 dtype, supports ND format.
* When the data type is float8_e8m0, the shape is 3D. When the shape of x2 is (n, k), scale is (n, z, 2),
* when the shape of x2 is (k, n), scale is (z, n, 2), where z = ceil(k / 64) and k is the reduce axis of x2.
* When the data type is float32, the shape is 1D and only per-tensor tt quantization is supported.
* @li y: A tensor. Format supports ND. Data type supports float32.
* @li x1_scale: A tensor, quantization parameter. Must be float32 or float8_e8m0 dtype, supports ND format.
* When the data type is float8_e8m0, the shape is 3D. When the shape of x1 is (m, k), scale is (m, z, 2),
* when the shape of x2 is (k, m), scale is (z, m, 2), where z = ceil(k / 64) and k is the reduce axis of x2.
* When the data type is float32, the shape is 1D and only per-tensor tt quantization is supported.

* @par Attributes:
* Three attributes, including:
* @li transpose_x1: An optional bool. If true, changes the shape of "x1" from [m, k] to
* [k, m] before multiplication. Default: false.
* @li transpose_x2: An optional bool. If true, changes the shape of "x2" from [k, n] to
* [n, k] before multiplication. Default: false.
* @li group_size: An optional Int. Indicating the ratio between x1_scale/x2_scale and x1/x2 in group dequantization.
Default: 0.
* If the value of x2_scale along the k-dimension is n, one value in x2_scale can be used to dequantize n values in x1
along the k-dimension.
* The group_size is composed of the group_size_m, group_size_n, and group_size_k, total occupying 48 bits.
* 0-15 bits of group_size indicate group_size_k, 16-31 bits indicate group_size_n, 32-47 bits indicate group_size_m,
* 48-63 bits of group_size are noneffective.
* In mx quantification, the supported [group_size_m, group_size_n, group_size_k] combinations are [1, 1, 32].

* @par Outputs:
* One output, including:
* y: A matrix tensor. The dtype must be float32. The format supports ND. The shape supports (m,n).

* @attention Constraints:
* The combination of transpose_x1/transpose_x2 is only supported true/false for now.

* @li The following are the supported data type combinations by platform.
*\n
*\n
* - Ascend 950 AI Processor with group_sizes scenarios, supported data type and shapes combinations:
*\n
| quant | x1 type | x2 type | x1_scale type | x2_scale type | y type |
|-------|---------|---------|---------------|---------------|--------|
| mxfp8 | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0 | float8_e8m0 | float32 |
| mxfp4 | float4_e2m1 | float4_e2m1 | float8_e8m0 | float8_e8m0 | float32 |
| tt | hifloat8 | hifloat8 | float32 | float32 | float32 |
*\n
| quant | x1 shape | x2 shape | x1_scale shape | x2_scale shape | y shape | group_size |
|-------|----------|----------|----------------|----------------|---------|------------|
| mxfp8/mxfp4 | (k,m) | (k,n) | (ceil(k/64),m,2) | (ceil(k/64),n,2) | (m,n) | [1, 1, 32] |
| tt | (k,m) | (n,k) | (1) | (1) | (m,n) | 0 |
*\n
*/
#ifndef OPS_PROTO_DEF_QUANTBATCHMATMULINPLACEADD
#define OPS_PROTO_DEF_QUANTBATCHMATMULINPLACEADD
REG_OP(QuantBatchMatmulInplaceAdd)
    .INPUT(x1, TensorType({DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(QuantBatchMatmulInplaceAdd)
#endif
} // namespace ge

#endif // OPS_MATMUL_QUANT_BATCH_MATMUL_INPLACE_ADD_PROTO_H_
