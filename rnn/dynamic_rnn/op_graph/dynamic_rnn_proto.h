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
 * \file dynamic_rnn_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_FINITE_H_
#define OPS_OP_PROTO_INC_IS_FINITE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief: DynamicRNN calculation.
* @par Inputs:
* ten inputs:
* @li x:A required 3D Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li w:A required 2D Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li b:A required 1D Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li seq_length:A optional 3D Tensor. Must be one of the following types: float16, int32, float32.
* The format must be ND.
* @li init_h:A optional Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li init_c:A optional Tensor. Must be one of the following types: float16, float32.
* The format must be ND.
* @li wci:A optional reserved Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li wcf:A optional reserved Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li wco:A optional reserved Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li mask:A optional reserved Tensor. Must be one of the following types: uint8. The format must be ND. \n

* @par Attributes:
* @li cell_type:An string identifying the cell type in the op.
* Default to "LSTM". Only LSTM is currently supported.
* @li direction:An string identifying the direction in the op.
* Default to "UNIDIRECTIONAL". Only UNIDIRECTIONAL is currently supported.
* @li cell_depth:An integer identifying the cell depth in the op.
* Default to 1. Only 1 is currently supported.
* @li use_peephole:An bool identifying if use peephole in the op.
* Default to false. Only false is currently supported.
* @li keep_prob:An float identifying the keep prob in the op. Default to 1.0.
* @li cell_clip:An float identifying the cell clip in the op. Default to -1.0.
* @li num_proj:An integer identifying the num projection in the op.
* Default to 0. Only 0 is currently supported.
* @li time_major:An bool identifying the time major in the op. Default to true.
* @li activation:An string identifying the type of activation function in the op.
* Default to "tanh". Only "tanh" is currently supported.
* @li forget_bias:An float identifying the forget bias in the op. Default to 0.
* @li gate_order:An string identifying the type of gate order in the op.
* Support "ijfo" and "ifjo". Default to "ijfo".
* @li is_training:An bool identifying is training in the op. Default to true . \n

* @par Outputs:
* eight outputs:
* @li y:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li output_h:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li output_c:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li i:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li j:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li f:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li o:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @li tanhc:A 3D Tensor. Must be one of the following types: float16, float32. The format must be ND.
* @par Third-party framework compatibility:
* Compatible with the TF operator LSTM.
*/
REG_OP(DynamicRNN)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
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
    .ATTR(forget_bias, Float, 0.0)
    .ATTR(gate_order, String, "ijfo")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(DynamicRNN)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_FINITE_H_

