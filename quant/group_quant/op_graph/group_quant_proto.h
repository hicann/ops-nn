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
 * \file nn_quantize.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Quantize feature map by group.
 * @par Inputs:
 * @li x: A Tensor. 2-D with shape [S, H]. Must be one of the following types:
 * float32, float16, bfloat16. The format support ND.
 * @li scale: A Tensor. Specifying the quantitation scale of x. 2-D with shape
 * [E, H], the second dim of scale shape is same as the second dim of x shape.
 * Must be one of the following types: float32, float16, bfloat16.
 * The format support ND.
 * @li group_index: A Tensor. Specifying the index of group. 1-D with shape
 * [E, ], the first dim of scale shape is same as the first dim of scale shape.
 * Must be one of the following types: int32, int64. The format support ND.
 * @li offset: A Tensor. Optional. Specifying the quantitation offset of x. 1-D
 * with shape [1, ] or 0-D with shape []. Must be one of the following types:
 * float32, float16, bfloat16. The dtype of offset should be same as scale.
 * The format support ND.
 * @par Outputs:
 * y: A 2-D Tensor. Shape is same as input x. The format support ND.
 * Must be one of the following types: int4, int8.
 * @par Attributes:
 * dst_type: An optional attribute of type int. Declare the output dtype.
 * Support DT_INT4, DT_INT8. Defaults to DT_INT8.
 * @attention Constraints:
 * @li If output y data type is INT4, the last dim of y shape should be
 * an even number.
 * @li Input group_index value should be in the range of [0, S] and be an
 * non-decreasing sequence. The last value of input group_index must be the
 * same as the first dim of x shape.
 */
REG_OP(GroupQuant)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT4, DT_INT8}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(GroupQuant)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
