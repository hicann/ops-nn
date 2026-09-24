/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_block_cell_tf_plugin.cpp
 * \brief TensorFlow GRUBlockCell op plugin (tf.raw_ops.GRUBlockCell -> GEIR GruBlockCell)
 */

#include "register/register.h"

namespace domi {
// TensorFlow GRUBlockCell maps 1:1 to the CANN GruBlockCell operator.
// Inputs (x, h_prev, w_ru, w_c, b_ru, b_c) and outputs (r, u, c, h) share the same
// order, shapes and merged-weight layout (w_ru[I+H,2H] / w_c[I+H,H], gate order r|u);
// the TF op carries no attributes and fp32 is the only supported dtype on both sides,
// so auto operator mapping suffices — no weight re-layout, no subgraph expansion.
REGISTER_CUSTOM_OP("GruBlockCell")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("GRUBlockCell")
    .ParseParamsByOperatorFn(AutoMappingByOpFn)
    .ImplyType(ImplyType::TVM);
} // namespace domi
