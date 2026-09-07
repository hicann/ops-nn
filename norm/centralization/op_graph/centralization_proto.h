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
 * \file centralization_proto.h
 * \brief Centralization proto definition.
 */
#ifndef OPS_NORM_CENTRALIZATION_PROTO_H_
#define OPS_NORM_CENTRALIZATION_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Computes Centralization. result = x - mean(x, axes).

 * @par Inputs:
 * x: An ND tensor of type float16 or float32.

 * @par Attributes:
 * axes: The dimensions to reduce. A list of unique dimensions in the range [-rank(x), rank(x)).
 * Negative dimensions are accepted. Defaults to {-1}.

 * @par Outputs:
 * y: A tensor with the same shape and type as x.

 */
#ifndef OPS_PROTO_DEF_CENTRALIZATION
#define OPS_PROTO_DEF_CENTRALIZATION
REG_OP(Centralization)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(axes, ListInt, {-1})
    .OP_END_FACTORY_REG(Centralization)
#endif // OPS_PROTO_DEF_CENTRALIZATION

} // namespace ge

#endif // OPS_NORM_CENTRALIZATION_PROTO_H_
