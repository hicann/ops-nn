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
 * \file softmax_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_

#include "graph/operator_reg.h"
#include "nn_norm.h"

namespace ge {
/**
* @brief Applies the Softmax function to an n-dimensional input tensor
*  rescaling them. so that the elements of the n-dimensional output tensor lie
*  in the range [0,1] and sum to 1.

* @par Inputs:
* One input:
* x: A mutable input tensor, which can be floating point tensors with different precisions. Must be one of the following data types: float16, float32, bfloat16,
* double. Should be a variable tensor. The format must be ND. Shape support 1D ~ 8D. \n

* @par Attributes:
* @li axes: An optional list of int. Specifies on which dimensions of input x the Softmax operation is performed.
* Multi-axis reduction is supported. Defaults to "{-1}".
* In Ascend 910_95 AI Processor, only single-axis reduction is supported. \n
* @li half_to_float: An optional bool. 
* This parameter determines whether to convert the output data type to float32 when the input data type is float16.
* Defaults to "false".
* - If true and the input data type is float16, the output data type should be float32.
* - Otherwise, the output data type should be the same as the input data type. \n

* @par Outputs:
* y: A ND tensor. The output tensor represents the probability distribution of the input tensor after being processed by the Softmax function.
* Has the same dimensionality and shape as the "x" with values in the range [0, 1].
* Must be one of the following types: float16, float32, bfloat16, double. \n

* @par Third-party framework compatibility
*  Compatible with the TensorFlow operator Softmax.
*/
REG_OP(SoftmaxV2)
    .INPUT(x, TensorType({ DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .OUTPUT(y, TensorType({ DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT }))
    .ATTR(axes, ListInt, {-1})
    .ATTR(half_to_float, Bool, false)
    .OP_END_FACTORY_REG(SoftmaxV2)

}
#endif