 /**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_INDEX_FILL_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INDEX_FILL_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Fills the elements of the input tensor x with value val by selecting the indices in the order given in index.
* This is a non in-place replacement operator and does not affect the input tensor of the element.

* @par Inputs:
* Three inputs, including:
* @li x: A tensor that serves as the source; its duplicate is created and subsequently filled with the specified values.
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or Atlas A3 Training Series 
* Product/Atlas A3 Inference Series Product, a tensor of type float16, float32, bfloat16, int64, int32, bool can be supported. \n
* In Ascend 950 AI Processor, a tensor of type float16, float32, bfloat16, int64, int32, bool, int8, uint8, int16, double can be supported. \n
* @li indices: A tensor, which equivalent to a vector or scalar. indices of input tensor to fill in. Must be one of the following types:
*     int32, int64. \n
* @li val: The value to fill with. It's a scalar or a one-dimensional tensor with only one element. 
* In Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component or Atlas A3 Training Series 
* Product/Atlas A3 Inference Series Product, a tensor of type float16, float32, bfloat16, int64, int32, bool can be supported. \n
* In Ascend 950 AI Processor, a tensor of type float16, float32, bfloat16, int64, int32, bool, int8, uint8, int16, double can be supported. \n
* @par Attributes:
* dim: A required int. Used to select the dimension of the input tensor. \n

* @par Outputs:
* y: A tensor with the same shape as 'x'. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator index_fill. \n

* @attention Constraints:
* @li The input x and output y must have same shape.
* @li The data type of element in tensor x may be same with the data type of input val, or may be different. If the types diff, 
* the data type of val must be convertible to the data type of element in tensor x.
* @li The input indices must be either a 1D tensor (equivalent to vector) or a 0D tensor (equivalent to scalar).
*
*/

REG_OP(IndexFill)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT64, DT_INT32, DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_DOUBLE}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(val, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT64, DT_INT32, DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT64, DT_INT32, DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_DOUBLE}))
    .REQUIRED_ATTR(dim, Int)
    .OP_END_FACTORY_REG(IndexFill)
}

#endif  // OPS_BUILT_IN_OP_PROTO_INC_INDEX_FILL_H_