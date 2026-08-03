/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUSED_PATCH_MLP_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUSED_PATCH_MLP_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(FusedPatchMlp)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(biases, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_layers, Int)
    .OP_END_FACTORY_REG(FusedPatchMlp)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_FUSED_PATCH_MLP_H_
