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
 * \file max_pool_grad_grad_with_argmax_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MAX_POOL_GRAD_GRAD_WITH_ARGMAX_H_
#define OPS_OP_PROTO_INC_MAX_POOL_GRAD_GRAD_WITH_ARGMAX_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes second-order gradients of the maxpooling function.

*@par Inputs:
*Three inputs, including:
  * @li x: A 4D Tensor. Supported type: RealNumberType() (double, float32, float16, int16, int32, int64, int8, uint16,
 uint32, uint64, uint8, bfloat16).
* Must set the format, supported format list ["NHWC"].
 * @li grad: A 4D Tensor. Supported type: same as x.
* Must set the format, supported format list ["NHWC"].
* @li argmax: A 4D Tensor. Supported type: int32, int64.
* Must set the format, supported format list ["NHWC"]. \n

*@par Outputs:
*y: A Tensor. Has the same dtype as input "x", shape same as argmax.

*@par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* A list that has length 4.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of the input tensor.
* A list that has length 4.
* @li padding: A required string, specifying the type of the padding algorithm to use.
* "SAME" or "VALID".

*@attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 and ksize[3] = 1.
* @li "strides" is a list that has length 4: strides[0] = 1 and strides[3] = 1.
* @li "padding": only supports "SAME" or "VALID".

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MaxPoolGradGradWithArgmax.
*/
REG_OP(MaxPoolGradGradWithArgmax)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(grad, TensorType::RealNumberType())
    .INPUT(argmax, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::RealNumberType())
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .OP_END_FACTORY_REG(MaxPoolGradGradWithArgmax)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MAX_POOL_GRAD_GRAD_WITH_ARGMAX_H_
