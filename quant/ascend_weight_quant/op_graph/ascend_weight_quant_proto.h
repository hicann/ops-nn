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
 * \file ascend_weight_quant_proto.h
 * \brief
 */

#ifndef ASCEND_WEIGHT_QUANT_PROTO_H
#define ASCEND_WEIGHT_QUANT_PROTO_H

#include "graph/operator_reg.h"
namespace ge {

/**
 * @brief Quantizes the weight of int8.
 * @par Inputs:
 * @li x: A required tensor of the quantized weight. Must be of the following types: int8.
 * @li offset: A required tensor of the quantization offset. Must be of the following
 * types: int8. Shape is the same as "x". \n
 *
 * The following are the supported data types and data formats:
 *\n
| Tensor    | x      | offset | y      |\n
| :-------: | :----: | :----: | :----: |\n
| Data Type | int8   | int8   | int8   |\n
| Format    | HWCN   | HWCN   | HWCN   |\n
 *\n
 * @par Attributes:
 * @li dst_type: An optional int of type int32. Declares the output dtype.
 * Support DT_INT8, DT_INT4. Defaults to DT_INT8.
 * @par Outputs:
 * @li y: A required output tensor of the quantized weight. Must be of the following
 * types: int8, int4. Shape is the same as input "x".
 *\n
 * @par Third-party framework compatibility
 * It is a custom operator. It has no corresponding operator in Caffe, Onnx, Tensorflow or Pythorch.
 */
#ifndef OPS_PROTO_DEF_ASCENDWEIGHTQUANT
#define OPS_PROTO_DEF_ASCENDWEIGHTQUANT
REG_OP(AscendWeightQuant)
    .INPUT(x, TensorType({DT_INT8}))
    .INPUT(offset, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(AscendWeightQuant)
#endif // OPS_PROTO_DEF_ASCENDWEIGHTQUANT

} // namespace ge
#endif // ASCEND_WEIGHT_QUANT_PROTO_H
