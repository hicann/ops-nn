/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lp_loss_proto.h
 * \brief
 */
#ifndef LP_LOSS_PROTO_H_
#define LP_LOSS_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Computes the lp loss between "predict" and "label". Only p=1 is supported, which computes the mean
* absolute error (MAE).

* @par Inputs:
* @li predict: An ND tensor of type float16, float32 or bfloat16. The predicted value.
* @li label: An ND tensor of type float16, float32 or bfloat16. The golden value. Its dtype and shape must be
* exactly the same as "predict": this operator does not perform dtype promotion or shape broadcasting between
* the two inputs. Dtypes or shapes outside the declared domain are rejected during tiling.
* @li Both inputs must be contiguous ND tensors. Non-contiguous layouts are not supported on this interface:
* the kernel applies the access view derived from "predict" to "label" as well.
* @li Dynamic shape and dynamic rank are supported: -1 marks an unknown dimension value and -2 marks an
* unknown rank. The supported rank range is 0 to 8.

* @par Attributes:
* @li p: A required int attribute that decides which loss to compute, now the p only can be 1 to compute l1_loss.
* @li reduction: An optional string which specifies the reduction to apply to the output. It can be "mean","sum" or
"none". Defaults to "mean".
* "none": no reduction will be applied.
* "mean": the sum of the output will be divided by the number of elements in the output.
* "sum": the output will be summed.

* @par Outputs:
*  y: An ND tensor with the same type as "predict". When "reduction" is "none", its shape is the same as
* "predict". When "reduction" is "mean" or "sum", it is a 0-dimensional tensor (a scalar).

* @par Third-party framework compatibility
* Compatible with the Pytorch operator torch.nn.L1Loss (that is, lp loss with p=1).
*/
REG_OP(LpLoss)
    .INPUT(predict, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(label, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(p, Int)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(LpLoss)
} // namespace ge

#endif
