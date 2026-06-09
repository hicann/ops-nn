/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file foreach_sub_list_inplace_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FOREACH_SUB_LIST_INPLACE_H_
#define OPS_OP_PROTO_INC_FOREACH_SUB_LIST_INPLACE_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 * @brief y(=x1) = x1 - alpha * x2, element-wise over two tensor lists, written back in place to x1.
 */
REG_OP(ForeachSubListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubListInplace)
} // namespace ge
#endif // OPS_OP_PROTO_INC_FOREACH_SUB_LIST_INPLACE_H_
