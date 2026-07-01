/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_GRAPH_GATHER_V2_PROTO_EXPERIMENTAL_H
#define OP_GRAPH_GATHER_V2_PROTO_EXPERIMENTAL_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(GatherV2)
    .INPUT(x, TensorType::ALL())
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .ATTR(batch_dims, Int, 0)
    .ATTR(negative_index_support, Bool, false)
    .OP_END_FACTORY_REG(GatherV2)
} // namespace ge

#endif // OP_GRAPH_GATHER_V2_PROTO_EXPERIMENTAL_H
