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
 * \file lamb_update_with_lr_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LAMB_UPDATE_WITH_LR_V2_OPS_H_
#define OPS_OP_PROTO_INC_LAMB_UPDATE_WITH_LR_V2_OPS_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief A fusion operator for bert lamb (trust-ratio weight update, no clip). \n

*@par Inputs:
* Seven inputs (x4, x5 are full tensors; the rest are scalar):
*@li x1: weight norm. @li x2: gradient norm. @li x3: learning rate. @li x4: update. @li x5: param.
*@li greater_y: threshold (0). @li select_e: fallback (1.0). \n

*@par Outputs:
*y: x5 - x3 * ratio * x4, ratio = where(x1>greater_y, where(x2>greater_y, x1/x2, select_e), select_e). \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(LambUpdateWithLrV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x4, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x5, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(greater_y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(select_e, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(LambUpdateWithLrV2)
} // namespace ge

#endif // OPS_OP_PROTO_INC_LAMB_UPDATE_WITH_LR_V2_OPS_H_
