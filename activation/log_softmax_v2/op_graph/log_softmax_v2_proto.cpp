/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log_softmax_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_LOG_SOFTMAX_V2_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_LOG_SOFTMAX_V2_OPS_H_

#include "graph/operator_reg.h"
#include "nn_norm.h"
namespace ge {
/**
*@brief Computes log softmax activations .

*@par Inputs:
*One input:
* logits: A ND tensor. Must be one of the following data types: double, bfloat16, float16, float32 . \n

*@par Attributes:
* axes: An optional list of ints. Multi-axis reduction is supported. Defaults to "{-1}" .
* In Ascend 910_95 AI Processor, only single-axis reduction is supported. \n

*@par Outputs:
* logsoftmax: A ND tensor. Has the same data type as "logits" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator LogSoftmax.
*/
REG_OP(LogSoftmaxV2)
    .INPUT(logits, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(logsoftmax, TensorType({DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(LogSoftmaxV2)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_LOG_SOFTMAX_V2_OPS_H_