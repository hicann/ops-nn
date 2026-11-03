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
 * \file foreach_minimum_list_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_MINIMUM_LIST_H_
#define OPS_OP_PROTO_INC_FOREACH_MINIMUM_LIST_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply the element-wise minimum operation to corresponding tensors in two tensor lists.
 * The two tensor lists shall contain the same number of tensors with matching shapes.
 * @par Inputs:
 * Two inputs:
 * @li x1: The first tensor list
 * @li x2: The second tensor list
 * @par Outputs:
 * @li y: A tensor list containing the element-wise minimum of each corresponding x1 and x2 tensor
 */
REG_OP(ForeachMinimumList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMinimumList)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_MINIMUM_LIST_H_
