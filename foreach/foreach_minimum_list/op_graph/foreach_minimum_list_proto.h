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
 * \file foreach_minimum_list_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_MINIMUM_LIST_H_
#define OPS_OP_PROTO_INC_FOREACH_MINIMUM_LIST_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply minimum operation for each tensor in tensor list with a list of scalar in manner
 * of element-wise the number of tensors in tensor list shall be equal to the number of scalars
 * in scalar list
 * @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scalars in scalar list
 */
REG_OP(ForeachMinimumList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMinimumList)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_MINIMUM_LIST_H_