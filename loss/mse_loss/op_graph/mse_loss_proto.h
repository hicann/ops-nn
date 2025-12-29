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
 * \file nn_norm_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_

#include "graph/operator_reg.h"
namespace ge {

/**
 * @brief Computes mse loss.
 * @par Inputs:
 * two inputs, including:
 *  @li predict: An ND Tensor of dtype float16, float32 or bfloat16.
 *  @li label: An ND Tensor of dtype float16, float32 or bfloat16.\n
 *
 * @par Attributes:
 * reduction:An optional str from sum, none, mean, Defaults to "mean".\n
 *
 * @par Outputs:
 * y: when reduction=sum/mean, y is scale. when reduction=none, y has
 *    same type and shape as "predict".\n
 */
REG_OP(MseLoss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MseLoss)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_OPS_H_