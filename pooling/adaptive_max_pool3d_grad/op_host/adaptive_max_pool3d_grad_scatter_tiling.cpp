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
 * \file adaptive_max_pool3d_grad_scatter_tiling.cpp
 * \brief
 */
#include "adaptive_max_pool3d_grad_tiling.h"
#include <iostream>
#include "../../pool_3d_common/op_host/arch32/max_pool3d_grad_tiling_common.h"

namespace optiling {
ge::graphStatus AdaptiveMaxPool3DGradScatterTiling::GetShapeAttrsInfo()
{
    AdaptiveMaxPool3DGradTilingBase::GetShapeAttrsInfo();
    maxPoolGradParams.ncRound = 0UL;
    maxPoolGradParams.ncRoundTail = 0UL;
    maxPoolGradParams.totalRound = 0UL;
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveMaxPool3DGradScatterTiling::IsCapable()
{
    return true;
}

bool AdaptiveMaxPool3DGradScatterTiling::SetScatterTilingParams()
{
    return CalculateScatterTilingParams(
        maxPoolGradParams,
        maxPoolGradParams.doDim,
        maxPoolGradParams.hoDim,
        maxPoolGradParams.woDim,
        maxPoolGradParams.xDtypeSize,
        maxPoolGradParams.indexDtypeSize,
        MAX_BLOCK_COUNT);
}

void AdaptiveMaxPool3DGradScatterTiling::SetOtherTilingParams()
{
    SetCntTailTilingParams();
    
    CalculateRoundParams(
        maxPoolGradParams,
        maxPoolGradParams.isOverLap,
        maxPoolGradParams.diDim,
        maxPoolGradParams.hiDim,
        maxPoolGradParams.wiDim);
}

void AdaptiveMaxPool3DGradScatterTiling::SetScatterTilingData()
{
    SetScatterTilingDataCommon(tilingData, maxPoolGradParams);
}

void AdaptiveMaxPool3DGradScatterTiling::PrintScatterTilingData()
{
    PrintScatterTilingDataCommon(context_->GetNodeName(), tilingData);
}

ge::graphStatus AdaptiveMaxPool3DGradScatterTiling::DoOpTiling()
{
    bool res = SetScatterTilingParams();
    OP_CHECK_IF(!res, OP_LOGE(context_->GetNodeName(), "Scatter cal tiling params failed."), return ge::GRAPH_FAILED);
    maxPoolGradParams.tilingType = TILING_TYPE_SCATTER;
    SetOtherTilingParams();
    SetBaseTilingData();
    SetScatterTilingData();
    PrintTilingData();
    PrintScatterTilingData();
    
    return ge::GRAPH_SUCCESS;
}

// 注册
REGISTER_TILING_TEMPLATE("AdaptiveMaxPool3DGrad", AdaptiveMaxPool3DGradScatterTiling, 2);
} // namespace optiling