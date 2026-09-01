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
 * \file arg_max_grad_proto.h
 * \brief
 */
#ifndef OPS_INDEX_ARG_MAX_GRAD_PROTO_H_
#define OPS_INDEX_ARG_MAX_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Returns the reverse tensor of the ArgMax operator of a tensor. \n

* @par Inputs:
* three input, including:
* var: A ND Tensor of type float16, float32, int32 or int8. \n
* indices: A ND Tensor of type int32. \n
* updates: A ND Tensor of type float16, float32, int32 or int8. \n

* @par Attributes:
* @li dimension: An integer of type int, specifying the axis information of the index with the maximum value.\n

* @par Outputs:
* y: A ND Tensor of type float16, float32, int32 or int8. \n
*
*@attention Constraints:
*@li indices: only support int32,and shape same to "updates"
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the dimension length of "x".
*@li y:A ND Tensor, the type and shape is same to "var" \n

*@par Third-party framework compatibility
* not support all scene like pytorch operator scatter
* exp:
* var.shape=[2,3,4,5], dim=2, the shape of indices and updates should be [2,3,5]
* not support the shape of indices and updates is [2,3,2,5] like pytorch operator scatter. \n

* @attention Constraints:
* The operator will not be enhanced in the future.
*/
#ifndef OPS_PROTO_DEF_ARGMAXGRAD
#define OPS_PROTO_DEF_ARGMAXGRAD
REG_OP(ArgMaxGrad)
    .INPUT(var, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(updates, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8}))
    .REQUIRED_ATTR(dimension, Int)
    .OP_END_FACTORY_REG(ArgMaxGrad)
#endif // OPS_PROTO_DEF_ARGMAXGRAD
} // namespace ge

#endif // OPS_INDEX_ARG_MAX_GRAD_PROTO_H_
