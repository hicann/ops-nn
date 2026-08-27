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
 * \file bn_inference_proto.h
 * \brief BNInference graph prototype.
 */
#ifndef OPS_NORM_BN_INFERENCE_PROTO_H_
#define OPS_NORM_BN_INFERENCE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Performs inference batch normalization on a 4-D or 5-D tensor.
 *
 * @par Inputs:
 * @li x: The feature tensor. On Ascend 950PR/Ascend 950DT, NCHW/NHWC use four dimensions, NCDHW/NDHWC use five
 *     dimensions, and ND uses four or five dimensions. NCHW/NCDHW use channel axis 1, NHWC/NDHWC use the last axis,
 *     and ND storage follows its public origin format or uses axis 1 when the origin format is ND.
 * @li mean: A one-dimensional tensor of length C. On Ascend 950PR/Ascend 950DT, it is the original mean when mode is
 *     nonzero and the pre-folded alpha when mode is 0.
 * @li variance: A one-dimensional tensor of length C. On Ascend 950PR/Ascend 950DT, it is the original variance when
 *     mode is nonzero and the pre-folded beta when mode is 0.
 * @li momentum: The momentum tensor. On Ascend 950PR/Ascend 950DT, its shape is [], [1], or [C]. Its first element is
 *     used only when mode is nonzero and both optional inputs are absent.
 * @li scale: Optional one-dimensional tensor of length C. On Ascend 950PR/Ascend 950DT, scale-only is supported.
 * @li offset: Optional one-dimensional tensor of length C. On Ascend 950PR/Ascend 950DT, offset-only is supported
 *     when mode is nonzero; mode 0 requires scale when offset is present.
 *
 * @par Attributes:
 * @li epsilon: An optional float. The default value is 1e-5.
 * @li use_global_stats: An optional bool. The default value is true.
 * @li mode: An optional integer. The default value is 1. On Ascend 950PR/Ascend 950DT, 0 selects the pre-folded
 *     expression and every nonzero integer selects complete BNInference behavior. On other products, mode does not
 *     change the complete BNInference behavior.
 *
 * @par Outputs:
 * @li y: A tensor with the same shape, data type, and format as x.
 */
#ifndef OPS_PROTO_DEF_BNINFERENCE
#define OPS_PROTO_DEF_BNINFERENCE
// clang-format off
REG_OP(BNInference)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(momentum, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float,1e-5f)
    .ATTR(use_global_stats, Bool,true)
    .ATTR(mode, Int,1)
    .OP_END_FACTORY_REG(BNInference)
// clang-format on
#endif // OPS_PROTO_DEF_BNINFERENCE
} // namespace ge

#endif // OPS_NORM_BN_INFERENCE_PROTO_H_
