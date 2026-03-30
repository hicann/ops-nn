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
 * \file softmax_grad_ext_proto.h
 * \brief
 */
#ifndef OP_NN_SOFTMAX_GRAD_EXT_PROTO_H_
#define OP_NN_SOFTMAX_GRAD_EXT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Function softmax gradients ext.

*@par Inputs:
* @li grad: A tensor dtype of bfloat16, float16, float32.
* Indicates the reverse input tensor.
* The format must be FRACTAL_NZ unless specified. Support 2D ~ 6D.
* In Ascend 950 AI Processor, the format must be ND.

* @li x1: A tensor dtype of bfloat16, float16, float32. Indicates the inverse gradient.
* The format must be FRACTAL_NZ unless specified, Support 2D ~ 6D.
* In Ascend 950 AI Processor, the format must be ND.
* Has the same type, format and shape as input grad.
* @li x2: A scalar dtype of bfloat16, float16, float32. Indicates the inverse scaling factor.
* The format must be ND. Has the same type as input grad. \n

*@par Attributes:
*@li axes: A int attr. The axis for reduce, Defaults to 1.
*@li keep_dims: A bool attr. Whether to keep the dimension number. Defaults to true.
* Currently, this parameter can only be set to true. If true, retains reduced dimensions with length 1. \n

*@par Outputs:
* y: A tensor dtype of bfloat16, float16, float32. Indicates the reverse output.
* Has the same type, format and shape as input grad. \n

*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
*/
REG_OP(SoftmaxGradExt)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(axes, Int, 1)
    .ATTR(keep_dims, Bool, true)
    .OP_END_FACTORY_REG(SoftmaxGradExt)

} // namespace ge

#endif // OP_NN_SOFTMAX_GRAD_EXT_PROTO_H_
