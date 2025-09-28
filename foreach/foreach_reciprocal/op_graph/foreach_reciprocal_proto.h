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
 * \file foreach_reciprocal_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_RECIPROCAL_H_
#define OPS_OP_PROTO_INC_FOREACH_RECIPROCAL_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief Apply reciprocal operation for each tensor in a tensor list in manner of element-wise
 * @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
 * @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the reciprocal value of the x
 */
REG_OP(ForeachReciprocal)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachReciprocal)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_RECIPROCAL_H_