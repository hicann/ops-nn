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

__aicore__ inline bool IsCacheOpTiling(const ScatterElementsV2TilingData* tiling_data)
{
    return tiling_data->coreNums != 0 || tiling_data->xDim0 != 0 || tiling_data->xDim1 != 0 ||
           tiling_data->indicesDim0 != 0 || tiling_data->indicesDim1 != 0 ||
           tiling_data->updatesDim0 != 0 || tiling_data->updatesDim1 != 0;
}

template <typename T, typename U, bool IsScalar>
__aicore__ inline void ExecCacheScatterOp(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                          ScatterElementsV2TilingData* tiling_data,
                                          AscendC::TPipe* pipe, GM_ADDR workspace) {
    if (tiling_data->mode >= ScatterElementsV2NS::SCATTER_MODE_REDUCTION_BEGIN &&
        tiling_data->mode <= ScatterElementsV2NS::SCATTER_MODE_REDUCTION_END &&
        !(std::is_same<T, int32_t>::value || std::is_same<T, float>::value ||
          std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value)) {
        return;
    }
    ScatterElementsV2NS::ExecTransposeAndScatterElements<T, U, IsScalar> op;
    op.Init(var, indices, updates, tiling_data, pipe, GetUserWorkspace(workspace));
    op.Process();
}

template <typename T, typename U>
__aicore__ inline void ExecLegacyScatterOp(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                           ScatterElementsV2TilingData* tiling_data,
                                           AscendC::TPipe* pipe) {
    KernelScatterElementsV2<T, U> op;
    op.Init(tiling_data, pipe, var, indices, updates);
    if (tiling_data->modeFlag == SMALL_MODE) {
        op.ProcessSmall();
    } else {
        op.ProcessScatter();
    }
}

template <typename T, typename U, bool IsScalar>
__aicore__ inline void ExecScatterOpImpl(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                         ScatterElementsV2TilingData* tiling_data,
                                         AscendC::TPipe* pipe, GM_ADDR workspace) {
    if (IsCacheOpTiling(tiling_data)) {
        ExecCacheScatterOp<T, U, IsScalar>(var, indices, updates, tiling_data, pipe, workspace);
    } else {
        ExecLegacyScatterOp<T, U>(var, indices, updates, tiling_data, pipe);
    }
}

template <typename T, typename U, bool IsScalar>
__aicore__ inline void ExecScatterOp(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                     ScatterElementsV2TilingData* tiling_data,
                                     AscendC::TPipe* pipe, GM_ADDR workspace) {
    ExecScatterOpImpl<T, U, IsScalar>(var, indices, updates, tiling_data, pipe, workspace);
}
#endif

#define CALL_OP_IMPL(T, U)                                                             \
    do {                                                                               \
        ExecScatterOp<T, U, false>(var, indices, updates, tilingDevice, &pipe, workspace); \
    } while (0)

extern "C" __global__ __aicore__ void scatter_elements_v2(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    ScatterElementsV2TilingData* __restrict tilingDevice = &tiling_data;
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
    if (TILING_KEY_IS(110)) {
        CALL_OP_IMPL(float, int);
    } else if (TILING_KEY_IS(120)) {
        CALL_OP_IMPL(float, long);
    } else if (TILING_KEY_IS(210)) {
        CALL_OP_IMPL(half, int);
    } else if (TILING_KEY_IS(220)) {
        CALL_OP_IMPL(half, long);
    } else if (TILING_KEY_IS(310)) {
        CALL_OP_IMPL(int, int);
    } else if (TILING_KEY_IS(320)) {
        CALL_OP_IMPL(int, long);
    } else if (TILING_KEY_IS(410)) {
        CALL_OP_IMPL(uint8_t, int);
    } else if (TILING_KEY_IS(420)) {
        CALL_OP_IMPL(uint8_t, long);
    } else if (TILING_KEY_IS(510)) {
        CALL_OP_IMPL(int8_t, int);
    } else if (TILING_KEY_IS(520)) {
        CALL_OP_IMPL(int8_t, long);
    } else if (TILING_KEY_IS(610)) {
        CALL_OP_IMPL(bfloat16_t, int);
    } else if (TILING_KEY_IS(620)) {
        CALL_OP_IMPL(bfloat16_t, long);
    }
#endif
}
