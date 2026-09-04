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
 * \file single_layer_lstm_grad_tiling_arch35.h
 * \brief regbase (Ascend950) small-shape tiling entry for SingleLayerLstmGrad.
 */

#ifndef SINGLE_LAYER_LSTM_GRAD_TILING_ARCH35_H
#define SINGLE_LAYER_LSTM_GRAD_TILING_ARCH35_H

#include "register/op_impl_registry.h"

namespace optiling {

// Tries the regbase small-shape path (tiling key 20000). On success sets handled=true and
// finishes the whole tiling (key/blockDim/tilingData/workspace). If the shape/dtype does not
// qualify, sets handled=false and leaves the context untouched so the legacy tiling can run.
ge::graphStatus TilingSingleLayerLstmGrad4RegbaseSmall(gert::TilingContext* context, bool& handled);

} // namespace optiling

#endif // SINGLE_LAYER_LSTM_GRAD_TILING_ARCH35_H
