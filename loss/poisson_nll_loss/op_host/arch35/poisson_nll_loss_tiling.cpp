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
 * \file poisson_nll_loss_tiling.cpp
 * \brief
 */

#include <vector>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "poisson_nll_loss_tiling_arch35.h"
#include "poisson_nll_loss_tiling.h"

namespace optiling {
ge::graphStatus TilingForPoissonNllLoss(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "start tiling");
    // compileInfo may be null on the deployed-binary path (TilingParse produces none);
    // RunTiling does not depend on it, so do not hard-fail on null.
    auto compileInfo = static_cast<const PoissonNllLossCompileInfo*>(context->GetCompileInfo());

    PoissonNllLossTiling tiling(context);
    return tiling.RunTiling(compileInfo);
}

ge::graphStatus TilingPrepareForPoissonNllLoss(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForPoissonNllLoss", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the PoissonNllLoss op.
IMPL_OP_OPTILING(PoissonNllLoss)
    .Tiling(TilingForPoissonNllLoss)
    .TilingParse<PoissonNllLossCompileInfo>(TilingPrepareForPoissonNllLoss);
} // namespace optiling
