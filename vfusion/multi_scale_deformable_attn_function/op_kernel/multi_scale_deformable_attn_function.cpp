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
 * \file multi_scale_deformable_attn_function.cpp
 * \brief
 */

#if __CCE_AICORE__ == 200
#include "ms_deform_attn_310p.h"
#else
#include "ms_deform_attn_generic.h"
#include "ms_deform_attn_high_perf.h"
#endif

extern "C" __global__ __aicore__ void multi_scale_deformable_attn_function(GM_ADDR value, GM_ADDR valueSpatialShapes,
    GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
#if __CCE_AICORE__ == 200
    if (TILING_KEY_IS(1)) {
        MultiScaleDeformableAttn::KernelMultiScaleDeformableAttn310P<float> op;
        op.Init(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output, &tilingData);
        op.MSDAProcess();
    }
#else
    TPipe pipe;
    if (TILING_KEY_IS(1002)) {
        KernelMultiScaleDeformableAttnOpt<2, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1004)) {
        KernelMultiScaleDeformableAttnOpt<4, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1008)) {
        KernelMultiScaleDeformableAttnOpt<8, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2002)) {
        KernelMultiScaleDeformableAttnOpt<2, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2004)) {
        KernelMultiScaleDeformableAttnOpt<4, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2008)) {
        KernelMultiScaleDeformableAttnOpt<8, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelMultiScaleDeformableAttn op;
        op.Init(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output, &tilingData, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.ClearOutput();
        op.Process();
        op.ReleaseEventID();
    }
#endif
}