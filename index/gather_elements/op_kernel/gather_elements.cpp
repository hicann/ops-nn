/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements.cpp
 * \brief
 */
#include "gather_elements.h"
#include "gather_elements_v2_scalar.h"
#include "gather_elements_v2_transpose.h"
#include "gather_elements_v2_last_dim.h"

template <typename T_X, typename T_IDX>
__aicore__ inline void RouteToV2Kernel(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace, int64_t v2Mode,
                                       const GatherElementsV2TilingData& v2Data, AscendC::TPipe& pipe)
{
    if (v2Mode == 0) {
        AscendC::GatherElementsV2ScalarKernel<T_X, T_IDX> op(x, index, y, workspace, v2Data, pipe);
        op.Process();
    } else if (v2Mode == 1) {
        if constexpr (std::is_same<T_X, bfloat16_t>::value) {
            AscendC::GatherElementsV2TransposeKernel<half, T_IDX> op(x, index, y, workspace, v2Data, pipe);
            op.Process();
        } else {
            AscendC::GatherElementsV2TransposeKernel<T_X, T_IDX> op(x, index, y, workspace, v2Data, pipe);
            op.Process();
        }
    } else if (v2Mode == 2) {
        if constexpr (std::is_same_v<T_X, bfloat16_t> || std::is_same_v<T_X, half> || std::is_same_v<T_X, int16_t>) {
            AscendC::GatherElementsV2LastDim<half, T_IDX> op(x, index, y, v2Data, &pipe);
            op.Process();
        } else if constexpr (std::is_same_v<T_X, float> || std::is_same_v<T_X, int32_t>) {
            AscendC::GatherElementsV2LastDim<float, T_IDX> op(x, index, y, v2Data, &pipe);
            op.Process();
        }
    }
}

extern "C" __global__ __aicore__ void gather_elements(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                                      GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(GatherElementsTilingData);
    GET_TILING_DATA_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
    AscendC::TPipe tpipe;
    if (tilingData.useV2 == 1) {
        RouteToV2Kernel<DTYPE_X, DTYPE_INDEX>(x, index, y, workspace, tilingData.v2Mode, tilingData.v2Data, tpipe);
        return;
    }
    AscendC::GatherElementsKernel<DTYPE_X, DTYPE_INDEX> op(x, index, y, workspace, tilingData, tpipe);
    op.Process();
}
