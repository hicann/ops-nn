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
 * \file foreach_pow_scalar_and_tensor_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_POW_SCALAR_AND_TENSOR_H_
#define OPS_OP_PROTO_INC_FOREACH_POW_SCALAR_AND_TENSOR_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply power operation for each tensor in tensor list with a scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalars in scalar list
 */
REG_OP(ForeachPowScalarAndTensor)
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowScalarAndTensor)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_POW_SCALAR_AND_TENSOR_H_