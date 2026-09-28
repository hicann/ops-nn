/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_grad_proto.h
 * \brief DynamicAUGRUGrad算子proto定义
 */

#ifndef OPS_OP_PROTO_INC_DYNAMIC_AUGRU_GRAD_H_
#define OPS_OP_PROTO_INC_DYNAMIC_AUGRU_GRAD_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief: 带注意力机制的GRU（AUGRU）反向算子，并融合了seq_length掩码生成：
* 传入seq_length后，kernel内部按 seq_mask[t, b, :] = (t < seq_length[b]) ? 1 : 0
* 在线生成掩码并作用于梯度回传，无需外部再接独立的掩码生成算子。

* @par Inputs:
* @li x: 3D Tensor [T, B, I]，前向输入序列。
* @li weight_input: 2D Tensor [I, 3H]，输入侧权重。
* @li weight_hidden: 2D Tensor [H, 3H]，隐状态侧权重。
* @li weight_att: 3D Tensor [T, B, H]，注意力得分（已按H广播）。
* @li y: 3D Tensor [T, B, H]，前向输出（占位输入，不参与数值计算）。
* @li init_h: 2D Tensor [B, H]，初始隐状态。
* @li h: 3D Tensor [T, B, H]，前向各步隐状态输出（h[t]为第t步输出）。
* @li dy: 3D Tensor [T, B, H]，各时间步输出梯度。
* @li dh: 2D Tensor [B, H]，末时刻隐状态梯度。
* @li update: 3D Tensor [T, B, H]，前向更新门激活值z_t。
* @li update_att: 3D Tensor [T, B, H]，注意力作用后的更新门u_t = z_t * (1 - att)。
* @li reset: 3D Tensor [T, B, H]，前向重置门激活值r_t。
* @li new: 3D Tensor [T, B, H]，前向新门激活值n_t。
* @li hidden_new: 3D Tensor [T, B, H]，新门隐状态侧预激活值。
* @li seq_length: 1D Tensor [B]（可选），INT32，每个batch的实际序列长度；
* 缺省时不做动态掩码（等价于全1掩码）。
* @li mask: 任意形状（可选），UINT8，dropout预留占位（不参与数值计算）。

* @par Outputs:
* @li dw_input: 2D Tensor [I, 3H]。
* @li dw_hidden: 2D Tensor [H, 3H]。
* @li db_input: 1D Tensor [3H]。
* @li db_hidden: 1D Tensor [3H]。
* @li dx: 3D Tensor [T, B, I]。
* @li dh_prev: 2D Tensor [B, H]。
* @li dw_att: 2D Tensor [T, B]。

* @par Attributes（当前实现仅支持默认值语义）:
* @li direction: String，默认"UNIDIRECTIONAL"。
* @li cell_depth: Int，默认1。
* @li keep_prob: Float，默认-1.0。
* @li cell_clip: Float，默认-1.0。
* @li num_proj: Int，默认0。
* @li time_major: Bool，默认true。
* @li gate_order: String，默认"zrh"，权重/梯度中门的排列顺序，可选"rzh"。
* @li reset_after: Bool，默认true。

* @par Constraints:
* @li 仅支持UNIDIRECTIONAL、单层、gate_order为zrh/rzh。
* @li 上述输入中除x为FLOAT16/FLOAT32外，其余浮点输入与x的dtype一致。
*/
#ifndef OPS_PROTO_DEF_DYNAMICAUGRUGRAD
#define OPS_PROTO_DEF_DYNAMICAUGRUGRAD
REG_OP(DynamicAUGRUGrad)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight_att, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(init_h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(h, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(dh, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(update, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(update_att, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(reset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(hidden_new, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(seq_length, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .OUTPUT(dw_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dw_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(db_input, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(db_hidden, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dh_prev, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dw_att, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(direction, String, "UNIDIRECTIONAL")
    .ATTR(cell_depth, Int, 1)
    .ATTR(keep_prob, Float, -1.0)
    .ATTR(cell_clip, Float, -1.0)
    .ATTR(num_proj, Int, 0)
    .ATTR(time_major, Bool, true)
    .ATTR(gate_order, String, "zrh")
    .ATTR(reset_after, Bool, true)
    .OP_END_FACTORY_REG(DynamicAUGRUGrad)
#endif // OPS_PROTO_DEF_DYNAMICAUGRUGRAD
} // namespace ge
#endif // OPS_OP_PROTO_INC_DYNAMIC_AUGRU_GRAD_H_
