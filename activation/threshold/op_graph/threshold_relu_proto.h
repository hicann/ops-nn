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
 * \file threshold_relu_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_THREASHOLD_RELU_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_THREASHOLD_RELU_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Tests whether the input exceeds a threshold.

*@par Inputs:
* x: A ND Tensor with any format. Must be one of the following types: float16, float32, bfloat16. \n

*@par Attributes:
* threshold: A required float32. Defaults to "0.0". "x" is compared with "threshold", outputs "1" for inputs above
threshold; "0" otherwise. \n

*@par Outputs:
* y: A ND Tensor with any format. Has the same dtype as the input. Must be one of the following types: float16, float32,
bfloat16.
*@par Third-party framework compatibility
* Compatible with the Caffe operator Threshold.
*/

REG_OP(Threshold)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(threshold, Float, 0.0)
    .OP_END_FACTORY_REG(Threshold);
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_THREASHOLD_RELU_OPS_H_
