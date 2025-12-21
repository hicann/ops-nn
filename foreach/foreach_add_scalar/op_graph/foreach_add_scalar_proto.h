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
 * \file foreach_add_scalar_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_ADD_SCALAR_H_
#define OPS_OP_PROTO_INC_FOREACH_ADD_SCALAR_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply add operation for each tensor in tensor list with a scalar in manner of element-wise
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors, the length cannot exceed 50,
 *        the dtype can be BFloat16, Float16, Int32 or Float32, and the format support ND.
 * @li scalar: A scalar in form of tensor with only one element,
 *        the dtype can be Float16, Int32 or Float32, and the format supports ND.
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scalar,
 *        has the same length, dtype and format as input "x".
 */
REG_OP(ForeachAddScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddScalar)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_ADD_SCALAR_H_
