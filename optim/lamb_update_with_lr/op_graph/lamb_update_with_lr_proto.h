/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lamb_update_with_lr_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LAMB_UPDATE_WITH_LR_OPS_H_
#define OPS_OP_PROTO_INC_LAMB_UPDATE_WITH_LR_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief A fusion operator for bert lamb (trust-ratio weight update with clip). \n

*@par Inputs:
* Nine inputs (input_mul1, input_sub are full tensors; the rest are scalar):
*@li input_greater1. @li input_greater_realdiv. @li input_realdiv. @li input_mul0: lr.
*@li input_mul1: update. @li input_sub: param. @li greater_y: threshold. @li select_e: fallback.
*@li minimum_y: clip upper bound. \n

*@par Outputs:
*y: input_sub - clip(ratio) * input_mul0 * input_mul1. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambUpdateWithLr)
    .INPUT(input_greater1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_greater_realdiv, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_realdiv, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_mul0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_mul1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(input_sub, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(greater_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(select_e, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(minimum_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(LambUpdateWithLr)
} // namespace ge

#endif // OPS_OP_PROTO_INC_LAMB_UPDATE_WITH_LR_OPS_H_
