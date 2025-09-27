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
 * \file foreach_lerp_scalar_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_LERP_SCALAR_H_
#define OPS_OP_PROTO_INC_FOREACH_LERP_SCALAR_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply lerp operation for each tensor in tensor list with tensors in another tensor list and
 * a scalar in manner of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors, the length cannot exceed 50,
 *         the dtype can be BFloat16, Float16 or Float32, and the format support ND.
 * @li x2: Another tensor list containing multiple tensors, must has the same length, dtype and format as input "x1".
 * @li weight: A scalar in form of tensor with only one element,
 *        the dtype can be Float32, and the format supports ND.
 * @par Outputs:
 *     y: A tensor list which store the tensors whose value are produced by lerp,
 *        has the same length, dtype and format as input "x1".
 */
REG_OP(ForeachLerpScalar)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLerpScalar)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_LERP_SCALAR_H_
