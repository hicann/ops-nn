/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_BUCKETIZE_PROTO_H_
#define OPS_NN_BUCKETIZE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Bucketize input tensor according to ascending boundaries.
 *
 * @par Inputs:
 * @li x: Input tensor, supports INT32, INT64, FLOAT, DOUBLE.
 *
 * @par Attributes:
 * @li boundaries: Ascending list of bucket boundaries.
 * @li dtype: Output type, supports DT_INT32 or DT_INT64, default DT_INT32.
 * @li right: Whether to use right insertion semantics, default true.
 *
 * @par Outputs:
 * @li y: Output tensor with same shape as x.
 */
REG_OP(Bucketize)
    .INPUT(x, TensorType({DT_INT32, DT_INT64, DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(boundaries, ListFloat)
    .ATTR(dtype, Type, DT_INT32)
    .ATTR(right, Bool, true)
    .OP_END_FACTORY_REG(Bucketize)

} // namespace ge

#endif // OPS_NN_BUCKETIZE_PROTO_H_
