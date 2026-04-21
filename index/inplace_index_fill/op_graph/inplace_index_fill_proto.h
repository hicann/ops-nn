/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_INPLACE_INDEX_FILL_H_
#define OPS_BUILT_IN_OP_PROTO_INC_INPLACE_INDEX_FILL_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

#define INPLACE_INDEX_FILL_SUPPORT_TYPES {                                  \
    ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT8, \
    ge::DT_UINT8, ge::DT_INT16,  ge::DT_INT32,   ge::DT_INT64, ge::DT_BOOL, \
    ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT8, \
    ge::DT_UINT8, ge::DT_INT16,  ge::DT_INT32,   ge::DT_INT64, ge::DT_BOOL}
REG_OP(InplaceIndexFill)
    .INPUT(x, TensorType(INPLACE_INDEX_FILL_SUPPORT_TYPES))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(value, TensorType(INPLACE_INDEX_FILL_SUPPORT_TYPES))
    .OUTPUT(x, TensorType(INPLACE_INDEX_FILL_SUPPORT_TYPES))
    .REQUIRED_ATTR(dim, Int)
    .OP_END_FACTORY_REG(InplaceIndexFill)
}

#endif  // OPS_BUILT_IN_OP_PROTO_INC_INPLACE_INDEX_FILL_H_