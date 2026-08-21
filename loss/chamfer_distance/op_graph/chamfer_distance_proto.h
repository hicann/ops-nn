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
 * \file chamfer_distance_proto.h
 * \brief
 */
#ifndef OPS_LOSS_CHAMFER_DISTANCE_PROTO_H_
#define OPS_LOSS_CHAMFER_DISTANCE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the chamfer distance between two 2-D point sets. \n

* @par Inputs:
* @li xyz1: A Tensor. Must be one of the following types: float16, bfloat16, float32.
* Point set with shape (2, B, N), where xyz1[0] holds all x coordinates and xyz1[1] holds all y coordinates.
* @li xyz2: A Tensor. Must have the same type and shape as xyz1. \n

* @par Outputs:
* @li dist1: A Tensor. Must have the same type as xyz1. Minimum squared distance from each point of xyz1
* to xyz2 with shape (B, N).
* @li dist2: A Tensor. Must have the same type as xyz1. Minimum squared distance from each point of xyz2
* to xyz1 with shape (B, N).
* @li idx1: A Tensor of type int32. Index of the nearest point in xyz2 with shape (B, N).
* @li idx2: A Tensor of type int32. Index of the nearest point in xyz1 with shape (B, N). \n

* @par Third-party framework compatibility
* Compatible with the mmcv operator chamfer_distance.
*/
REG_OP(ChamferDistance)
    .INPUT(xyz1, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(xyz2, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(dist1, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(dist2, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(idx1, TensorType({DT_INT32}))
    .OUTPUT(idx2, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ChamferDistance)
} // namespace ge

#endif // OPS_LOSS_CHAMFER_DISTANCE_PROTO_H_
