/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chamfer_distance_grad_proto.h
 * \brief
 */

#ifndef CHAMFER_DISTANCE_GRAD_PROTO_H
#define CHAMFER_DISTANCE_GRAD_PROTO_H
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Applies set operation along last dimension of 6 Tensor inputs. \n

* @par Inputs:
* @li xyz1: A Tensor. Must be one of the following types: float16, float32. Point set with shape (B, N, 2).
* @li xyz2: A Tensor. Must have the same type and shape as xyz1.
* @li idx1: A Tensor. Must be one of the following types: int32. min indices with set one with shape (B, N).
* @li idx2: A Tensor. Must be one of the following types: int32. min indices with set two with shape (B, N).
* @li grad_dist1: A Tensor. Must have the same type as xyz1. Grad of dist1 with shape (B, N).
* @li grad_dist2: A Tensor. Must have the same type as xyz1. Grad of dist2 with shape (B, N).
\n

* @par Outputs:
* @li grad_xyz1: A Tensor. Must be one of the following types: float16, bfloat16, float32. grad of xyz1 with shape (B,
N, 2).
* @li grad_xyz2: A Tensor. Must be one of the following types: float16, bfloat16, float32. grad of xyz2 with shape (B,
N, 2).
*/

REG_OP(ChamferDistanceGrad)
    .INPUT(xyz1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(xyz2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(idx1, TensorType({DT_INT32}))
    .INPUT(idx2, TensorType({DT_INT32}))
    .INPUT(grad_dist1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(grad_dist2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(grad_xyz1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(grad_xyz2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ChamferDistanceGrad)

} // namespace ge
#endif