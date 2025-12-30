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
 * \file masked_scatter_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_MASKED_SCATTER_H_
#define OPS_OP_PROTO_INC_MASKED_SCATTER_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief update the value of X with value according to mask.

* @par Inputs:
* three inputs, including:
*  @li x: A Tensor of dtype is float16 or float32 or float64 or
*      int64 or int32 or int16 or int8 or uint8 or bool or bfloat16.
*  @li mask: A Tensor of dtype is bool.
*  @li updates: A tensor with the same type as x. \n

* @par Outputs:
*  @li y: A tensor with the same type as x. \n
*/
REG_OP(MaskedScatter)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                          DT_INT16, DT_INT32, DT_INT64, DT_BOOL, DT_BF16}))
    .INPUT(mask, TensorType({DT_BOOL}))
    .INPUT(updates, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8,
                                DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_BOOL, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_INT16, DT_INT32, DT_INT64, DT_BOOL, DT_BF16}))
    .OP_END_FACTORY_REG(MaskedScatter)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MASKED_SCATTER_H_