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
 * \file threshold_grad_v2_d.cpp
 * \brief
 */

#include "threshold_grad_v2_d.h"


template <uint32_t schMode>
__global__ __aicore__ void threshold_grad_v2_d(GM_ADDR input_gradient, GM_ADDR input_feature, GM_ADDR output_backprops, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ThresholdGradV2DTilingData);
    GET_TILING_DATA_WITH_STRUCT(ThresholdGradV2DTilingData, tilingData, tiling);
    NsThresholdGradV2D::KernelThresholdGradV2D<DTYPE_INPUT_GRADIENT> op; // 算子kernel实例获取
    op.Init(input_gradient, input_feature, output_backprops, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.finalBigTileNum, tilingData.finalSmallTileNum, tilingData.tileDataNum,
       tilingData.smallTailDataNum, tilingData.bigTailDataNum, tilingData.tailBlockNum, tilingData.threshold);
    op.Process();
}
