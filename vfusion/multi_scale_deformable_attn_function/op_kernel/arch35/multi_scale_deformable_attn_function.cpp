/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multi_scale_deformable_attn_function.cpp
 * \brief ascend950 kernel entry for MultiScaleDeformableAttnFunction
 *
 * Routing (channels = embedDims, dtype-agnostic):
 *   - channels >= 64 -> Generic/SIMD path (schMode = MSDA_MODE_GENERIC)
 *   - channels < 64  -> SIMT path (schMode = MSDA_MODE_SIMT)
 */

#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../ms_deform_attn_generic.h"
#include "ms_deform_attn_simt.h"
#include "multi_scale_deformable_attn_function_tiling_key.h"
#include "multi_scale_deformable_attn_function_tiling_data.h"

template <uint32_t schMode>
__global__ __aicore__ void multi_scale_deformable_attn_function(GM_ADDR value, GM_ADDR valueSpatialShapes,
                                                                GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations,
                                                                GM_ADDR attentionWeights, GM_ADDR output,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MsdaRegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(MsdaRegBaseTilingData, tilingData, tiling);

    if constexpr (schMode == MSDA_MODE_SIMT) {
        MsdaSimt::MsdaSimtKernel<DTYPE_VALUE> op(&tilingData);
        op.Init(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output);
        op.Process();
    } else if constexpr (schMode == MSDA_MODE_GENERIC) {
        if constexpr (std::is_same_v<DTYPE_VALUE, float>) {
            TPipe pipe;
            KernelMultiScaleDeformableAttn<MsdaRegBaseTilingData> op;
            op.Init(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output,
                    &tilingData, &pipe);
            op.InitBuffer();
            op.GetLocalTensor();
            op.ClearOutput();
            op.Process();
            op.ReleaseEventID();
        }
    }
}
