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
 * \file global_lp_pool_proto.h
 * \brief GlobalLpPool operator proto header
 */

#ifndef GLOBAL_LP_POOL_PROTO_H_
#define GLOBAL_LP_POOL_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes GlobalLpPool, GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
 * values in the same channel.
 *
 * @par Inputs:
 * x: A 4D or 5D Tensor of type float16, float32 or bfloat16, with format ND. \n
 *
 * @par Attributes:
 * @li p: p value of the Lp norm used to pool over the input data. Must be one of the following types: float32. Defaults
 * to 2.0. \n
 *
 * @par Outputs:
 * y: A 4D or 5D Tensor. Has the same type and format as "x". \n
 * When x is a 4D Tensor, the shape of y is [x.shape[0],x.shape[1],1,1]. \n
 * When x is a 5D Tensor, the shape of y is [x.shape[0],x.shape[1],1,1,1].
 *
 * @par Third-party framework compatibility
 * Compatible with the ONNX operator GlobalLpPool.
 */
REG_OP(GlobalLpPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Float, 2.0)
    .OP_END_FACTORY_REG(GlobalLpPool)

} // namespace ge

#endif // GLOBAL_LP_POOL_PROTO_H_
