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
 * \file dynamic_qunat_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Dynamic Quant V2. Performs pre-token/per-tensor asymmetric dynamic quantization on input tensors.
 * @par Inputs:
 * @li x: A tensor. Type is:DT_FLOAT16 or DT_BF16. For Atlas A2 Training Series Product/Atlas 800I A2 Inference
 * Product/A200I A2 Box Heterogeneous Component and Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
 * Whose shape must be greater than 1. The data format support ND.
 * @li smooth_scales: An optional tensor.
 * When group_index is null, shape is 1 Dims. Dim[0] is the last dimension of x.
 * When group_index is not null, shape is 2 Dims. Dim[0] is the expert num(E). E must be not greater than 1024. Dim[1]
 * is the last dimension of x. The data type can be FLOAT16 or BFLOAT16. The data type must be the same as that of x.
 * The data format support ND.
 * @li group_index: An optional tensor. Specifying the index of group. 1-D with shape
 * [E, ].
 * The first dim of group_index shape is same as the first dim of smooth_scales shape.
 * Must be one of the following types: int32. The format support ND.
 * If group_index is not null, smooth_scales must be not null. \n
 * @par Attributes:
 * @li dst_type: An optional attribute of type int. Declare the output dtype.
 * Support DT_INT4, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8.
 * Defaults to DT_INT8.
 * @li is_symmetrical: An optional attribute of type bool. Select whether to be symmetrical.
 * Defaults to false.
 * @li quant_mode: An optional attribute of type string. Specifies the mode of quantization.
 * Support "pertoken", "pertensor", "perchannel". Defaults to "pertoken". \n
 * @par Outputs:
 * @li y: A tensor. Quantized output tensor, Shape is same as input x. If y dtype is int4, x last dim must be divisible
 * by 2. The format support ND. Type specified by dst_type, support INT4, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8.
 * @li scale: A tensor. Scale used for quantization.
 * When quant_mode is "pertoken", shape is the same as the shape of x after removing the last dimension.
 * When quant_mode is "pertensor", shape is (1,).
 * When quant_mode is "perchannel", shape is the same as the shape of x after removing the second last dimension.
 * Type is DT_FLOAT32. The format support ND.
 * @li offset: A tensor. Offset used for quantization. Shape is the same as the shape of scale.
 * Type is DT_FLOAT32. The format support ND. Shape is same as scale. \n
 * @attention Constraints:
 * Warning: E should to be be not greater than multiplication result of x Dims after removing the last dimension(S).
 * The value of group_index should to be increasing, ranging from 0 to S. The last value is should to be S. Otherwise
 * the result is meaningless.
 */
REG_OP(DynamicQuantV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(offset, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .ATTR(is_symmetrical, Bool, false)
    .ATTR(quant_mode, String, "pertoken")
    .OP_END_FACTORY_REG(DynamicQuantV2)
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_