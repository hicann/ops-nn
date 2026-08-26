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
 * \file in_training_reduce_grad_proto.h
 * \brief INTrainingReduceGrad GE IR 算子注册
 */
#ifndef OPS_OP_PROTO_INC_IN_TRAINING_REDUCE_GRAD_H_
#define OPS_OP_PROTO_INC_IN_TRAINING_REDUCE_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Performs the backpropagation of InstanceNorm. \n

*@par Inputs:
* Seven inputs, including:
*@li dy: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
*@li x: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
*@li variance: A 4D tensor of type float32, for the variance of "x", format [NCHW, NHWC] and HW=1.
*@li mean: A 4D tensor of type float32, for the mean of "x", format [NCHW, NHWC] and HW=1.
*@li res_gamma: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1.
*@li res_beta: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1.
*@li gamma: A 4D tensor of type float32, format [NCHW, NHWC] and HW=1. \n

*@par Outputs:
*pd_x: A 4D tensor of type float16 or float32, for the offset of "x", format [NCHW, NHWC]. \n

*@attention Constraints:
* The preceding layer of this operator must be INTrainingUpdateGrad. \n
*/
REG_OP(INTrainingReduceGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(res_gamma, TensorType({DT_FLOAT}))
    .INPUT(res_beta, TensorType({DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingReduceGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IN_TRAINING_REDUCE_GRAD_H_
