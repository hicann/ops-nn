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
 * \file turbo_quant.cpp
 * \brief
 */
#include "turbo_quant.h"

extern "C" __global__ __aicore__ void turbo_quant(GM_ADDR latent, GM_ADDR centroids, GM_ADDR y, GM_ADDR scale,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    TurboQuant::KernelTurboQuant op;
    op.Init(latent, centroids, y, scale, tilingData.numTokens, tilingData.tokensPerCore, tilingData.headDim,
            tilingData.packedBytes, tilingData.tokensPerBatch);
    op.Process();
}
