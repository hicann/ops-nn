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
 * \file sync_batch_norm_backward_elemt_proto.h
 * \brief
 */
#ifndef SYNC_BATCH_NORM_BACKWARD_ELEMT_PROTO_H
#define SYNC_BATCH_NORM_BACKWARD_ELEMT_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief part of SyncBatchNormBackward .

*@par Inputs:
* Seven inputs, including:
*@li grad_output: A ND tensor(2D-8D). Forward output differential. Must be one of the following types: float16, bfloat16, float32.
*@li save_input: A ND tensor(2D-8D). The input of forward. Has the same shape, format and type as input "grad_output".
*@li mean: A ND tensor(2D-8D). Mean of saved forward input. Has the same shape, format and type as input "grad_output".
*@li invstd: A ND tensor(2D-8D). Reciprocal of the standard deviation of the saved forward input. Has the same shape, format and type as input "grad_output".
*@li weight: A ND tensor(2D-8D). The weight parameter. Has the same shape, format and type as input "grad_output".
*@li mean_dy: A ND tensor(2D-8D). A part of sum_dy. Has the same shape, format and type as input "grad_output".
*@li mean_dy_xmu: A ND tensor(2D-8D). A part of sum_dy_xmu. Has the same shape, format and type as input "grad_output". \n

*@par Outputs:
* grad_input: A ND tensor(2D-8D). The reverse gradient corresponding to the input in the forward calculation. Has the same shape, format and type as input "grad_output". \n
*/
REG_OP(SyncBatchNormBackwardElemt)
    .INPUT(grad_output, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(save_input, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(invstd, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(mean_dy, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(mean_dy_xmu, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(grad_input, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(SyncBatchNormBackwardElemt)
}  // namespace ge

#endif  // SYNC_BATCH_NORM_BACKWARD_ELEMT_PROTO_H
