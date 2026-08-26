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
 * \file multilabel_margin_loss_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MULTILABEL_MARGIN_LOSS_OPS_H_
#define OPS_OP_PROTO_INC_MULTILABEL_MARGIN_LOSS_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Computes the multilabel margin loss (multi-class multi-classification hinge loss) between
* input x and target label indices. \n

* @par Inputs:
* @li x: The prediction, a 1D or 2D tensor with shape (C) or (N, C). Support dtype: float32/float16/bfloat16.
* @li target: The label indices, with the SAME shape as x. Each row lists valid class indices in [0, C), the
* remainder filled with the -1 sentinel (read until the first -1). Support dtype: int32. \n

* @par Attributes:
* @li reduction: An optional attribute. Specifies the reduction applied to the output. Type is string.
*                One of "none"/"mean"/"sum". Defaults to "mean" . \n

* @par Outputs:
* @li y: The loss. If reduction is "none" and x is 2D (N, C), y is 1D with shape (N); if reduction is "none"
* and x is 1D, or reduction is "mean"/"sum", y is a scalar. Support dtype: float32/float16/bfloat16.
* @li is_target: A tensor with the SAME shape as target; is_target[.., i] = 1 when i is a label of that
* sample, else 0. Support dtype: int32. \n

* @par Third-party framework compatibility
* Compatible with pytorch MultiLabelMarginLoss operator
*/
#ifndef OPS_PROTO_DEF_MULTILABELMARGINLOSS
#define OPS_PROTO_DEF_MULTILABELMARGINLOSS
REG_OP(MultilabelMarginLoss)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(target, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(is_target, TensorType({DT_INT32}))
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(MultilabelMarginLoss)
#endif // OPS_PROTO_DEF_MULTILABELMARGINLOSS
} // namespace ge

#endif // OPS_OP_PROTO_INC_MULTILABEL_MARGIN_LOSS_OPS_H_
