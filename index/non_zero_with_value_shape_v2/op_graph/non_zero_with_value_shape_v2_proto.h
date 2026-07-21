/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_INDEX_NON_ZERO_WITH_VALUE_SHAPE_V2_PROTO_H_
#define OPS_NN_INDEX_NON_ZERO_WITH_VALUE_SHAPE_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Returns tensors with updated shapes from NonZeroWithValue.
 *
 * @par Inputs:
 * @li value: Value tensor from NonZeroWithValue.
 * @li index: Index tensor from NonZeroWithValue, supports DT_INT32.
 * @li count: Count tensor for non-zero elements, supports DT_INT32.
 *
 * @par Outputs:
 * @li value: Updated value tensor.
 * @li index: Updated index tensor, supports DT_INT32.
 */
REG_OP(NonZeroWithValueShapeV2)
    .INPUT(value, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                              DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .INPUT(index, TensorType({DT_INT32}))
    .INPUT(count, TensorType({DT_INT32}))
    .OUTPUT(value, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                               DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(index, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(NonZeroWithValueShapeV2)

} // namespace ge

#endif // OPS_NN_INDEX_NON_ZERO_WITH_VALUE_SHAPE_V2_PROTO_H_
