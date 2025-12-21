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
 * \file scatter_elements_v2_proto.cpp
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_SCATTER_ELEMENTS_V2_H_
#define OPS_OP_PROTO_INC_SCATTER_ELEMENTS_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Uses "updates" to update tensor "var" by "indices". \n

* @par Inputs:
* Three inputs, including:
* @li var: An ND Tensor . \n
* Must be one of the following types: float32, float16, int16, int32, int64, int8, uint8, bfloat16.
* @li indices: An ND Tensor of type int32 or int64
* @li updates: An ND Tensor . \n
* Must be one of the following types: float32, float16, int16, int32, int64, int8, uint8, bfloat16.

* @par Attributes:
* @li axis: An optional int. Defaults to 0.
* @li reduction: An optional string. Defaults to string "none" and can be
* "add" or "mul". \n

* @attention Constraints:
* @li In non-last axis scenarios, you are advised to convert x, indices, and updates to the last axes,
* use ScatterElementsV2 for calculation, and then convert them to the original axes.
* @li Only Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and
* Atlas A3 Training Series Product/Atlas A3 Inference Series Product support ScatterElementsV2. \n

* @par Outputs:
* var: A Tensor. Has the same type and format as input "var" . \n
*/
REG_OP(ScatterElementsV2)
    .INPUT(var, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BF16}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BF16}))
    .OUTPUT(var, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT8,DT_BF16}))
    .ATTR(axis, Int, 0)
    .ATTR(reduction, String, "none")
    .OP_END_FACTORY_REG(ScatterElementsV2) // namespace ge
}
#endif // OPS_OP_PROTO_INC_SCATTER_ELEMENTS_V2_H_
