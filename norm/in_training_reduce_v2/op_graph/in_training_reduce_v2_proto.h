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
 * \file in_training_reduce_v2_proto.h
 * \brief
 */
#ifndef OPS_NORM_IN_TRAINING_REDUCE_V2_PROTO_H_
#define OPS_NORM_IN_TRAINING_REDUCE_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief InstanceNorm training-forward spatial reduction, paired with INTrainingUpdateV2.
 * For each instance-channel (n, c), computes sum = Sigma x and square_sum = Sigma x^2
 * over the spatial dimensions, retaining each reduced dimension with size 1. These are raw sums
 * without division or affine transformation. Float16 inputs are promoted to float32
 * before multiplication and accumulation. This operator has no attributes.
 *
 * @par Inputs:
 * @li x: A float16 or float32 tensor. NCHW/NHWC inputs are 4D and reduce H and W.
 *        "Ascend 950PR&950DT系列产品" supports native 5D NDHWC [N,D,H,W,C], reducing D, H and W.
 *        On "Ascend 950PR&950DT系列产品", all dimensions must be positive.
 *        The input element count must not exceed 2^63-1 on these products.
 *        For NHWC/NDHWC, C*sizeof(x.dtype) must not exceed 2^40-1 bytes on these products.
 *
 * @par Outputs:
 * @li sum: A float32 tensor with logical shape [N,C,1,1] for NCHW input or
 *          [N,1,1,C] for NHWC input, or [N,1,1,1,C] for NDHWC input on "Ascend 950PR&950DT系列产品".
 *          On "Ascend 950PR&950DT系列产品" the storage format is ND.
 * @li square_sum: A float32 tensor with the same shape and storage format as sum.
 */

#ifndef OPS_PROTO_DEF_INTRAININGREDUCEV2
#define OPS_PROTO_DEF_INTRAININGREDUCEV2
REG_OP(INTrainingReduceV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingReduceV2)
#endif // OPS_PROTO_DEF_INTRAININGREDUCEV2
} // namespace ge

#endif // OPS_NORM_IN_TRAINING_REDUCE_V2_PROTO_H_
