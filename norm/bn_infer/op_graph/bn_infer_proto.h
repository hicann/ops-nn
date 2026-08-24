/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_RUNTIME_BN_INFER_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_RUNTIME_BN_INFER_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(BNInfer)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(BNInfer)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_RUNTIME_BN_INFER_PROTO_H_
