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
 * \file cosine_embedding_loss_proto.h
 * \brief
 */
#ifndef COSINE_EMBEDDING_LOSS_OP_PROTO_H_
#define COSINE_EMBEDDING_LOSS_OP_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Measures the loss given input tensors x1, x2 and a label tensor target with
* values 1 or -1. \n

* @par Inputs:
* @li x1: A Tensor. Must be one of int32, float16, float32. Broadcasting with x2 is supported.
A2上支持8个，实际内部只支持3个
* @li x2: A Tensor. Must be one of int32, float16, float32. Broadcasting with x1 is supported.
* @li target: A Tensor of labels (+1 / -1). It broadcasts with the x1/x2 shape after axis-1 reduction. \n

* @par Attributes:
* @li margin: A Float. Should be a number from -1 to 1. Defaults to 0.
* @li reduction: A String. Specifies the reduction: "none", "mean", "sum". Defaults to "mean". \n

* @par Outputs:
* y: A Tensor of type float32. "none" returns the broadcast loss shape; "mean"/"sum" produce shape [1]. \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator CosineEmbeddingLoss.
*/
REG_OP(CosineEmbeddingLoss)
    .INPUT(x1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .ATTR(margin, Float, 0.0)
    .ATTR(reduction, String, "mean")
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(CosineEmbeddingLoss)
} // namespace ge
#endif // COSINE_EMBEDDING_LOSS_OP_PROTO_H_
