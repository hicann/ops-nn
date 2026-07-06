/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NN_EXPERIMENTAL_INDEX_EXPERIMENTAL_INDEX_PROTO_H_
#define OPS_NN_EXPERIMENTAL_INDEX_EXPERIMENTAL_INDEX_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(IndexAiCore)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8,
                          DT_BOOL}))
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT8,
                           DT_BOOL}))
    .OP_END_FACTORY_REG(IndexAiCore)
} // namespace ge
#endif // OPS_NN_EXPERIMENTAL_INDEX_EXPERIMENTAL_INDEX_PROTO_H_
