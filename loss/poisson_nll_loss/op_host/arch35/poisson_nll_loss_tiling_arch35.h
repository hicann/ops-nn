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
 * \file poisson_nll_loss_tiling_arch35.h
 * \brief hand-written tiling (reduction=none stage-1). No atvoss reduce/elewise deps.
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_POISSON_NLL_LOSS_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_POISSON_NLL_LOSS_TILING_H_

#include "register/op_def_registry.h"
#include "poisson_nll_loss_tiling.h"

namespace optiling {

class PoissonNllLossTiling {
public:
    explicit PoissonNllLossTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling(const PoissonNllLossCompileInfo* compileInfo);

private:
    gert::TilingContext* tilingContext;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_POISSON_NLL_LOSS_TILING_H_
