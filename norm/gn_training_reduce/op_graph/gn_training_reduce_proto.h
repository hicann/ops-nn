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
 * \file gn_training_reduce_proto.h
 * \brief GNTrainingReduce operator prototype (GE IR) declaration.
 */

#ifndef GN_TRAINING_REDUCE_PROTO_H
#define GN_TRAINING_REDUCE_PROTO_H

// operator_reg.h: provides REG_OP, TensorType, INPUT, OUTPUT, ATTR, OP_END_FACTORY_REG.
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Performs reduced group normalization.

*@par Inputs:
*x: A Tensor of type float16 or float32, with format NCHW NHWC . \n

*@par Outputs:
*@li sum: A Tensor of type float32 for SUM reduced "x". shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for NHWC.
*@li square_sum: A Tensor of type float32 for SUMSQ reduced "x".shape is [N, G, 1, 1, 1] for NCHW, [N, 1, 1, G, 1] for
NHWC.

*@par Attributes:
*num_groups: An optional Int, specifying the num of groups, default to 2 . \n
*When the operator is used together with GNTrainingUpdate, its value must be the same as GNTrainingUpdate's
*num_groups. \n

*@attention Constraints:
* This operator is a GroupNorm fusion operator for updating the moving averages for training.
* This operator is used in conjunction with GNTrainingUpdate.
*/
REG_OP(GNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .ATTR(num_groups, Int, 2)
    .OP_END_FACTORY_REG(GNTrainingReduce)
} // namespace ge

#endif
