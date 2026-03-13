/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_gradient_args_proto.h
 * \brief
 */

#ifndef INDEX_BROADCAST_GRADIENT_ARGS_PROTO_H_
#define INDEX_BROADCAST_GRADIENT_ARGS_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns the reduction indices for computing gradients of "x1" and "x2" with broadcast.
*
* @par Inputs:
* @li x1: A tensor. The type support int32 and int64. Its shape must be 1D. Format: ND.
* @li x2: A tensor. Its type is consistent with x1. Its shape must be 1D. Format: ND.

* @par Outputs:
* @li y1: A tensor. Reduction indices of x1. Its type is consistent with x1. Its shape must be 1D. Format: ND.
* @li y2: A tensor. Reduction indices of x2. Its type is consistent with x1. Its shape must be 1D. Format: ND.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BroadcastGradientArgs.
*/
REG_OP(BroadcastGradientArgs)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y1, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y2, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(BroadcastGradientArgs)

} // namespace ge

#endif // INDEX_BROADCAST_GRADIENT_ARGS_PROTO_H_