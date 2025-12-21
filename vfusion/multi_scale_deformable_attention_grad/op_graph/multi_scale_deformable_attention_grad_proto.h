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
 * \file multi_scale_deformable_attention_grad_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_MAT_MUL_V3_PROTO_H_
#define OPS_MATMUL_MAT_MUL_V3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief multi-scale deformable attention grad.
*
* @par Inputs:
* @li value: A Tensor. Must be one of the following types: float32.
* @li value_spatial_shapes: A Tensor. Must be one of the following types: int32.
* @li value_level_start_index: A Tensor. Must be one of the following types: int32.
* @li sampling_locations: A Tensor. Must be one of the following types: float32.
* @li attention_weights: A Tensor. Must be one of the following types: float32.
* @li grad_output: A Tensor. Must be one of the following types: float32.
*
* @par Outputs:
* grad_value: A Tensor. Must be one of the following types: float32.
* grad_sampling_locations: A Tensor. Must be one of the following types: float32.
* grad_attention_weights: A Tensor. Must be one of the following types: float32.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MultiScaleDeformableAttentionGrad)
    .INPUT(value, TensorType({DT_FLOAT}))
    .INPUT(value_spatial_shapes, TensorType({DT_INT32}))
    .INPUT(value_level_start_index, TensorType({DT_INT32}))
    .INPUT(sampling_locations, TensorType({DT_FLOAT}))
    .INPUT(attention_weights, TensorType({DT_FLOAT}))
    .INPUT(grad_output, TensorType({DT_FLOAT}))
    .OUTPUT(grad_value, TensorType({DT_FLOAT}))
    .OUTPUT(grad_sampling_locations, TensorType({DT_FLOAT}))
    .OUTPUT(grad_attention_weights, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(MultiScaleDeformableAttentionGrad)
} // namespace ge

#endif // OPS_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_PROTO_H