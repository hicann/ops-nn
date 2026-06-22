/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
#include "scatter_elements_v2_310p.h"
#else
#include "scatter_elements_v2.h"
#include "scatter_elements_v2_low_memory/exec_transpose_and_scatter_elements.h"

template <typename T, typename U, uint32_t MODE, bool IsScalar>
__aicore__ inline void ExecScatterOp(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                      ScatterElementsV2TilingData* tiling_data,
                                      AscendC::TPipe* pipe, GM_ADDR workspace) {
    if constexpr (MODE == 1) {
        if constexpr (!(std::is_same<T, int32_t>::value || std::is_same<T, float>::value ||
                        std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value)) {
            return;
        }
    }
    GM_ADDR userspace = GetUserWorkspace(workspace);
    ScatterElementsV2NS::ExecTransposeAndScatterElements<T, U, MODE, IsScalar> op;
    op.Init(var, indices, updates, tiling_data, pipe, userspace);
    op.Process();
}
#endif

extern "C" __global__ __aicore__ void scatter_elements_v2(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    const ScatterElementsV2TilingData* __restrict tilingDevice = &tiling_data;
    AscendC::TPipe pipe;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
#if (defined(DTYPE_VAR))
    if (TILING_KEY_IS(0)) {
        ScatterElementsV2310PNS::scatter_elements_v2_310p_split_m<DTYPE_VAR, DTYPE_INDICES>(var, indices, updates, &tiling_data, &pipe);
    } else if (TILING_KEY_IS(1)) {
        ScatterElementsV2310PNS::scatter_elements_v2_310p_multy_m<DTYPE_VAR, DTYPE_INDICES>(var, indices, updates, &tiling_data, &pipe);
    }
#endif
#else
    if (TILING_KEY_IS(0)) {
        ExecScatterOp<DTYPE_VAR, DTYPE_INDICES, 0, false>(var, indices, updates, &tiling_data, &pipe, workspace);
    } else if (TILING_KEY_IS(10)) {
        ExecScatterOp<DTYPE_VAR, DTYPE_INDICES, 0, true>(var, indices, updates, &tiling_data, &pipe, workspace);
    } else if (TILING_KEY_IS(100)) {
        ExecScatterOp<DTYPE_VAR, DTYPE_INDICES, 1, false>(var, indices, updates, &tiling_data, &pipe, workspace);
    } else if (TILING_KEY_IS(110)) {
        ExecScatterOp<DTYPE_VAR, DTYPE_INDICES, 1, true>(var, indices, updates, &tiling_data, &pipe, workspace);
    } else if (TILING_KEY_IS(1)) {
        // 原始分支，仅支持尾轴场景，需要在算子外部做transpose
        KernelScatterElementsV2<DTYPE_VAR, DTYPE_INDICES, 1> op;
        op.Init(tilingDevice, &pipe, var, indices, updates);
        if (tilingDevice->modeFlag == 1) {
            op.ProcessSmall();
        } else {
            op.ProcessScatter();
        }
    } else if (TILING_KEY_IS(2)) {
        // 原始分支，仅支持尾轴场景，需要在算子外部做transpose
        KernelScatterElementsV2<DTYPE_VAR, DTYPE_INDICES, 2> op;
        op.Init(tilingDevice, &pipe, var, indices, updates);
        if (tilingDevice->modeFlag == 1) {
            op.ProcessSmall();
        } else {
            op.ProcessScatter();
        }
    }
#endif
}
