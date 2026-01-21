/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file max_pool3d_grad_with_argmax_scatter.h
 * \brief
 */
#ifndef MAX_POOL_GRAD3D_WITH_ARGMAX_SCATTER_H
#define MAX_POOL_GRAD3D_WITH_ARGMAX_SCATTER_H
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "max_pool3d_grad_with_argmax_common.h"
#include "max_pool3d_grad_with_argmax_scatter_base.h"
#include "../pool_3d_common/arch32/max_pool3d_grad_scatter_unified.h"

namespace MaxPool3DGradWithArgmax {
using namespace AscendC;
using namespace MaxPool3DGradWithArgmaxComm;

template <typename TX, typename TGrad, typename TArgmax, typename TY>
class MaxPoolGradWithArgScatter : 
    public MaxPool3DGradCommon::MaxPool3DGradScatterUnified<
        TX, TGrad, TArgmax, TY,
        MaxPool3DGradWithArgmaxTilingData,
        TilingParams,
        BlockParams,
        MaxPool3DGradScatterInternal::MaxPool3DGradScatterBaseTemplate>
{
public:
    __aicore__ inline MaxPoolGradWithArgScatter(TPipe* pipe)
        : MaxPool3DGradCommon::MaxPool3DGradScatterUnified<
            TX, TGrad, TArgmax, TY,
            MaxPool3DGradWithArgmaxTilingData,
            TilingParams,
            BlockParams,
            MaxPool3DGradScatterInternal::MaxPool3DGradScatterBaseTemplate>(pipe)
    {}
};
} // namespace MaxPool3DGradWithArgmax
#endif // MAX_POOL_GRAD3D_WITH_ARGMAX_SCATTER_H