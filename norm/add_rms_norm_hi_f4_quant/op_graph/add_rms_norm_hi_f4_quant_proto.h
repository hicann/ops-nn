/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file add_rms_norm_hi_f4_quant_proto.h
 * \brief
 */
#ifndef NORM_ADD_RMS_NORM_HI_F4_QUANT_PROTO_H_
#define NORM_ADD_RMS_NORM_HI_F4_QUANT_PROTO_H_

#include "graph/operator_reg.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ge {
/**
* @brief Performs fused Add + RmsNorm + HiF4 quantization on input tensors.
* Adds x1 and x2, then applies RmsNorm normalization, followed by HiF4 quantization.

* @par Inputs:
* @li x1: An input tensor of type float16 or bfloat16.
* The shape supports at least 2 dimensions, and at most 7 dimensions.
* @li x2: An input tensor of type float16 or bfloat16.
* The shape and data type must be the same as x1.
* @li gamma: A weight tensor for normalization.
* The shape is 1D, matching the last dimension of x1.
* Data type must be the same as x1, or float32.
* @li beta: An optional bias tensor for normalization.
* If provided, shape and data type must match gamma.

* @par Attributes:
* @li epsilon: A float. A small value added to the denominator for numerical stability. Defaults to 1e-6.
* @li output_rstd: A bool. Whether to output the reciprocal of standard deviation. Defaults to false.
* @li scale_alg: An int. The HiF4 Eb/Ec threshold strategy. 0=mse_first, 1=unbiased. Defaults to 0.

* @par Outputs:
* @li y: Quantized output tensor of type HIFLOAT4. Same shape as x1.
* @li x: The sum of x1 and x2. Same shape and type as x1.
* @li scale: An output tensor of type HIFLOAT4_SCALE. \n
* - rank(scale) = rank(x1) + 1.
* - scale.shape[-2] = ceil(x1.shape[-1] / 64).
* - scale.shape[-1] = 1.
* - Other dimensions match input x1.
* @li rstd: The reciprocal of standard deviation. Type is float32.

* @attention Constraints:
* @li The input x1 last dimension must be a multiple of 64.
* @li The dimensions of x1 must be in the range [2, 7].
* @li HiF4 quantization uses a three-level scaling architecture with block size 64.
*/
REG_OP(AddRmsNormHiF4Quant)
    .INPUT(x1, TensorType({Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16}))
    .INPUT(x2, TensorType({Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16}))
    .INPUT(gamma, TensorType({Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16, Ops::NN::DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16, Ops::NN::DT_FLOAT}))
    .OUTPUT(y, TensorType({Ops::NN::DT_HIFLOAT4}))
    .OUTPUT(x, TensorType({Ops::NN::DT_FLOAT16, Ops::NN::DT_BF16}))
    .OUTPUT(scale, TensorType({Ops::NN::DT_HIFLOAT4_SCALE}))
    .OUTPUT(rstd, TensorType({Ops::NN::DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .ATTR(output_rstd, Bool, false)
    .ATTR(scale_alg, Int, 0)
    .OP_END_FACTORY_REG(AddRmsNormHiF4Quant)

} // namespace ge

#endif // NORM_ADD_RMS_NORM_HI_F4_QUANT_PROTO_H_
