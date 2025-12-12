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
 * \file foreach_div_scalar_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_DIV_SCALAR_H_
#define OPS_OP_PROTO_INC_FOREACH_DIV_SCALAR_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply div operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scalar
 */
REG_OP(ForeachDivScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachDivScalar)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_DIV_SCALAR_H_