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
 * \file dynamic_qunat_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Dynamic Quant. Performs pre-token symmetric dynamic quantization on input tensors.
 * @par Inputs:
 * @li x: A Tensor. Type is:DT_FLOAT16 or DT_BF16. For Atlas A2 Training Series Product/Atlas 800I A2 Inference
 * Product/A200I A2 Box Heterogeneous Component and Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
 * Whose shape must be greater than 1. The data format support ND.
 * @li smooth_scales: An optional Tensor. Shape is the last dimension of x.
 * The data type can be FLOAT16 or BFLOAT16.
 * @li group_index: An optional Tensor. Specifying the index of group. 1-D with shape
 * [E, ], the first dim of scale shape is same as the first dim of scale shape.
 * Must be one of the following types: int32. The format support ND. \n
 * @par Attributes:
 * dst_type: An optional int32. Output y data type enum value.
 * Support DT_INT4, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8.
 * Defaults to DT_INT8. \n
 * @par Outputs:
 * @li y: A Tensor. Quantized output tensor, Shape is same as input x.
 * The format support ND. Type specified by dst_type, support INT4, INT8,
 * FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8.
 * @li scale: A Tensor. Scale used for quantization.
 * Type is DT_FLOAT32. The format support ND.
 */
REG_OP(DynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(DynamicQuant)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_