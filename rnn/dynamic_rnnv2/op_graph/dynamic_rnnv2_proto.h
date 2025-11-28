/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Mindspore project.
 * Copyright 2019 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*!
 * \file is_finite_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_FINITE_H_
#define OPS_OP_PROTO_INC_IS_FINITE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief: DynamicRNNV2 calculation.
* @par Inputs:
* ten inputs:
* @li x:A required Tensor. Must be one of the following types: float16, float32.
* @li weight_input:A required Tensor. Must be one of the following types: float16, float32.
* @li weight_hidden:A required Tensor. Must be one of the following types: float16, float32.
* @li b:A required Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li seq_length:A optional Tensor. Must be one of the following types: float16, int32.
* @li init_h:A optional Tensor. Must be one of the following types: float16, float32.
* @li init_c:A optional Tensor. Must be one of the following types: float16, float32.
* @li wci:A optional Tensor. Must be one of the following types: float16, float32.
* @li wcf:A optional Tensor. Must be one of the following types: float16, float32.
* @li wco:A optional Tensor. Must be one of the following types: float16, float32.
* @li mask:A optional Tensor. Must be one of the following types: uint8. The format must be ND . \n

* @par Attributes:
* @li cell_type:An string identifying the cell type in the op.
* Default to "LSTM". Only LSTM is currently supported.
* @li direction:An string identifying the direction in the op.
* Default to "UNIDIRECTIONAL". Support "UNIDIRECTIONAL" and "REDIRECTIONAL".
* @li cell_depth:An integer identifying the cell depth in the op.
* Default to 1. Only 1 is currently supported.
* @li use_peephole:An bool identifying if use peephole in the op.
* Default to false. Only false is currently supported.
* @li keep_prob:An float identifying the keep prob in the op.
* Default to 1. Only 1.0 is currently supported.
* @li cell_clip:An float identifying the cell clip in the op.
* Default to -1. Only -1.0 is currently supported.
* @li num_proj:An integer identifying the num projection in the op.
* Default to 0. Only 0 is currently supported.
* @li time_major:An bool identifying the time major in the op. Default to true.
* @li activation:An string identifying the type of activation function in
* the op. Default to "tanh". Only "tanh" is currently supported.
* @li recurrent_activation:An string identifying the type of activation
* function in the op. Default to "sigmoid". Only support "sigmoid" in
* Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component
* and Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
* Support "sigmoid" and "hard_sigmoid"
* in other series produces. In general, set "hard_sigmoid" for TF Keras LSTM.
* @li forget_bias:An float identifying the forget bias in the op. Default to 0.
* @li gate_order:An string identifying the type of gate order in the op.
* Support "ijfo" and "ifco". Default to "ijfo".
* Set "ijfo" for TF operator LSTM, Set "ifco" for TF Keras LSTM.
* @li stateful: An bool identifying the type of stateful in the op.
* Default to fasle. Only false is currently supported.
* @li merge_mode: An string identifying the type of merge_modein the op.
* Default to "concat". Only "concat" is currently supported
* @li is_training:An bool identifying is training in the op. Default to true . \n

* @par Outputs:
* eight outputs:
* @li y:A Tensor. Must be one of the following types: float16, float32.
* @li output_h:A Tensor. Must be one of the following types: float16, float32.
* Return the last output_h.
* @li output_c:A Tensor. Must be one of the following types: float16, float32.
* Return the last output_c.
* @li i:A Tensor. Must be one of the following types: float16, float32.
* @li j:A Tensor. Must be one of the following types: float16, float32.
* @li f:A Tensor. Must be one of the following types: float16, float32.
* @li o:A Tensor. Must be one of the following types: float16, float32.
* @li tanhc:A Tensor. Must be one of the following types: float16, float32.
* @par Third-party framework compatibility:
* Compatible with the TF operator LSTM or TF keras operator LSTM or the Pytorch operator LSTM.
*/

REG_OP(DynamicRNNV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32, DT_FLOAT16}))
    .OPTIONAL_INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(init_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wci, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wcf, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(wco, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(output_c, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(i, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(j, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(f, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(o, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(tanhc, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(cell_type, String, "LSTM")
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(use_peephole, Bool, false)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(activation, String, "tanh")
    .ATTR(recurrent_activation, String, "sigmoid")
    .ATTR(forget_bias, Float, 0.0)
    .ATTR(gate_order, String, "ijfo")
    .ATTR(stateful, Bool, false)
    .ATTR(merge_mode, String, "concat")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicRNNV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_FINITE_H_

