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
 * \file adaptive_max_pool3d_grad_scatter.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_H
#define ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_H
#include "adaptive_max_pool3d_grad_scatter_base.h"
#include "adaptive_max_pool3d_grad_common.h"
#include "../pool_3d_common/arch32/max_pool3d_grad_scatter_unified.h"

namespace AdaptiveMaxPool3DGrad {
using namespace AscendC;
using namespace AdaptiveMaxPool3DGradComm;

template <typename TX, typename TGrad, typename TArgmax, typename TY>
class AdaptiveMaxPool3DGradScatter : 
    public MaxPool3DGradCommon::MaxPool3DGradScatterUnified<
        TX, TGrad, TArgmax, TY,
        AdaptiveMaxPool3DGradTilingData,
        TilingParams,
        BlockParams,
        MaxPool3DGradScatterInternal::MaxPool3DGradScatterBaseTemplate>
{
public:
    __aicore__ inline AdaptiveMaxPool3DGradScatter(TPipe* pipe)
        : MaxPool3DGradCommon::MaxPool3DGradScatterUnified<
            TX, TGrad, TArgmax, TY,
            AdaptiveMaxPool3DGradTilingData,
            TilingParams,
            BlockParams,
            MaxPool3DGradScatterInternal::MaxPool3DGradScatterBaseTemplate>(pipe)
    {}
};
} // namespace AdaptiveMaxPool3DGrad
#endif // ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_H