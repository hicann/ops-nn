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
 * \file foreach_addcmul_list_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_ADDCMUL_LIST_H_
#define OPS_OP_PROTO_INC_FOREACH_ADDCMUL_LIST_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply AddcMul operation for each tensor in tensor list with a scalar in scalar list
 *  or a tensor in manner of element-wise
 * @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar list or a tensor
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcMul with the scalars
 */
REG_OP(ForeachAddcmulList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcmulList)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_ADDCMUL_LIST_H_
