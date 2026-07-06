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
 * \file threshold_relu_proto.h
 * \brief
 */
#ifndef OP_GRAPH_THREASHOLD_V2_OPS_H_
#define OP_GRAPH_THREASHOLD_V2_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Thresholds each element of the input Tensor: y = (x > threshold) ? x : value

* @par Inputs:
* Three inputs, including:
* @li x: A ND Tensor. Support 1D~8D.
* Must be one of the following types: float16, float32, int8, int32, uint8, int64, bfloat16. \n
* @li threshold: A Tensor which should have the shape (1,), the value to threshold at.
* Must be one of the following types: float16, float32, int8, int32, uint8, int64, bfloat16. \n
* @li value: A Tensor which should have the shape (1,), the value to replace with. default value is 0.
* Must be one of the following types: float16, float32, int8, int32, uint8, int64, bfloat16. \n

* @par Outputs:
* y: A Tensor which has the same shape, format and type as the input x. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator Threshold.
*/
REG_OP(ThresholdV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_BF16}))
    .INPUT(threshold, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_BF16}))
    .OPTIONAL_INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(ThresholdV2);
} // namespace ge

#endif // OP_GRAPH_THREASHOLD_V2_OPS_H_
