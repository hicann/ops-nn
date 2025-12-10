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
 * \file fusion_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief DequantBias. \n

* @par Inputs:
* @li x: A 2D tensor. Input tensor representing the inverse quantization operation.
* Supported format "ND". The shape is [M, N], and the data type supports int32.
* @li weight_scale: A 1D tensor. Indicates the weight of the multiplication on the N-dimensional input of the anti-quantization operation.
* The shape is [N], and the length is consistent with the N-dimensional length of x. The data type supports float32, bfloat16.
* @li activate_scale: A 1D tensor. The data type supports float32.
* Indicates the weight of the multiplication on the M dimension of the input for the anti-quantization operation.
* The shape is [M], with a length consistent with the M dimension of x, and the data type supports float32.
* Supported format "ND".
* @li bias: A 1D tensor. Indicates the weight of the addition on the N-dimensional input of the anti-quantization operation.
* The shape is [N], with a length consistent with the N-dimensional length of x.
* The data type supports float32, bfloat16, float16, int32. Supported format "ND".

* @par Attributes:
* output_dtype: An int attr. Indicates the data type of the output out. The value is [1, 27].
* A value of 1 indicates that the output type is float16, and a value of 27 indicates that the output type is bfloat16.
* When the weight_scale data type is float32, this parameter is set to 1; when the weight_scale data type is bfloat16,
* this parameter is set to 27.

* @par Outputs:
* y: A 2D tensor. The output tensor of the quantization operation.
* The shape is [M, N], and the data type supports float16, bfloat16. Supported format "ND". \n
*/
REG_OP(DequantBias)
    .INPUT(x, TensorType({DT_INT32}))
    .INPUT(weight_scale, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(activate_scale, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32, DT_INT32}))
    .REQUIRED_ATTR(output_dtype, Int)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(DequantBias)
}  // namespace ge


#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUSION_OPS_H_