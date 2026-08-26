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
/**
 * @brief Performs batch normalization inference.
 *
 * @par Inputs:
 * @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
 * @li scale: A 1D tensor of type float32, for the scale factor, the shape is same as dim C of input x. \n
 * @li offset: A 1D tensor of type float32, for the offset, the shape is same as dim C of input x. \n
 * @li mean: A 1D tensor of type float32, for the mean, the shape is same as dim C of input x. \n
 * @li variance: A 1D tensor of type float32, for the variance, the shape is same as dim C of input x. \n
 *
 * @par Attributes:
 * epsilon: An optional float32, specifying the small value added to variance to
 * avoid dividing by zero. Defaults to "0.0001" . \n
 *
 * @par Outputs:
 * y: A 4D tensor of type float16 or float32 or bfloat16 for the normalized "x", with format NHWC or NCHW. \n
 *
 * @attention Constraints:
 * For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the
 * square root instruction.
 */
#ifndef OPS_PROTO_DEF_BNINFER
#define OPS_PROTO_DEF_BNINFER
REG_OP(BNInfer)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(BNInfer)
#endif // OPS_PROTO_DEF_BNINFER
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_RUNTIME_BN_INFER_PROTO_H_
