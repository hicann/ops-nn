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
 * \file confusion_softmax_grad_proto.h
 * \brief
 */
#ifndef OPS_NORM_CONFUSION_SOFTMAX_GRAD_PROTO_H_
#define OPS_NORM_CONFUSION_SOFTMAX_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Confuse mul, sum and sub.

*@par Inputs
*Two inputs, including:
* @li grad: A ND tensor. Must be one of the following data types: bfloat16, float16, float32.
* @li x: A ND tensor. Has the same shape and data type as "grad". \n

*@par Outputs
* y: A ND tensor.  Has the same shape and data type as "grad". \n

*@par Restrictions
*Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(ConfusionSoftmaxGrad)
  .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .OP_END_FACTORY_REG(ConfusionSoftmaxGrad)
} // namespace ge

#endif // OPS_NORM_CONFUSION_SOFTMAX_GRAD_PROTO_H_