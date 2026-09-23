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
 * \file gru_block_cell_grad_proto.h
 * \brief Operator prototype declaration for GRUBlockCellGrad.
 */

#ifndef OPS_OP_PROTO_INC_GRU_BLOCK_CELL_GRAD_H_
#define OPS_OP_PROTO_INC_GRU_BLOCK_CELL_GRAD_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Computes the gradients for a single GRU block-cell step.
 *
 * @par Inputs:
 * @li x: A required 2D Tensor with shape [batch_size, input_size]. The data type must be float32.
 * @li h_prev: A required 2D Tensor with shape [batch_size, cell_size], representing the previous hidden state.
 *     The data type must be float32.
 * @li w_ru: A required 2D Tensor with shape [input_size + cell_size, 2 * cell_size], representing the reset and
 *     update gate weights. The data type must be float32.
 * @li w_c: A required 2D Tensor with shape [input_size + cell_size, cell_size], representing the candidate-state
 *     weights. The data type must be float32.
 * @li b_ru: A required 1D Tensor with shape [2 * cell_size], representing the reset and update gate biases.
 *     The data type must be float32.
 * @li b_c: A required 1D Tensor with shape [cell_size], representing the candidate-state bias. The data type must
 *     be float32.
 * @li r: A required 2D Tensor with shape [batch_size, cell_size], representing the reset gate output. The data type
 *     must be float32.
 * @li u: A required 2D Tensor with shape [batch_size, cell_size], representing the update gate output. The data type
 *     must be float32.
 * @li c: A required 2D Tensor with shape [batch_size, cell_size], representing the candidate-state output. The data
 *     type must be float32.
 * @li d_h: A required 2D Tensor with shape [batch_size, cell_size], representing the gradient of the current hidden
 *     state. The data type must be float32.
 *
 * @par Outputs:
 * @li d_x: A 2D Tensor with shape [batch_size, input_size], representing the gradient of x. The data type is float32.
 * @li d_h_prev: A 2D Tensor with shape [batch_size, cell_size], representing the gradient of h_prev. The data type
 *     is float32.
 * @li d_c_bar: A 2D Tensor with shape [batch_size, cell_size], representing the gradient of the candidate-state
 *     pre-activation. The data type is float32.
 * @li d_r_bar_u_bar: A 2D Tensor with shape [batch_size, 2 * cell_size], containing the concatenated reset and update
 *     gate pre-activation gradients. The data type is float32.
 *
 * @par Third-party framework compatibility:
 * Compatible with the TensorFlow operator GRUBlockCellGrad.
 */

#ifndef OPS_PROTO_DEF_GRUBLOCKCELLGRAD
#define OPS_PROTO_DEF_GRUBLOCKCELLGRAD
REG_OP(GRUBlockCellGrad)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(h_prev, TensorType({DT_FLOAT}))
    .INPUT(w_ru, TensorType({DT_FLOAT}))
    .INPUT(w_c, TensorType({DT_FLOAT}))
    .INPUT(b_ru, TensorType({DT_FLOAT}))
    .INPUT(b_c, TensorType({DT_FLOAT}))
    .INPUT(r, TensorType({DT_FLOAT}))
    .INPUT(u, TensorType({DT_FLOAT}))
    .INPUT(c, TensorType({DT_FLOAT}))
    .INPUT(d_h, TensorType({DT_FLOAT}))
    .OUTPUT(d_x, TensorType({DT_FLOAT}))
    .OUTPUT(d_h_prev, TensorType({DT_FLOAT}))
    .OUTPUT(d_c_bar, TensorType({DT_FLOAT}))
    .OUTPUT(d_r_bar_u_bar, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(GRUBlockCellGrad)
#endif
} // namespace ge

#endif
