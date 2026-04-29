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
 * \file scatter_add_with_sorted_apt.cpp
 * \brief scatter_add_with_sorted_apt.cpp
 */
#include "arch35/scatter_add_with_sorted_struct.h"
#include "arch35/scatter_add_with_sorted_simd.h"
#include "arch35/scatter_add_with_sorted_simd_determ.h"
#include "arch35/scatter_add_with_sorted_determ_workspace.h"
#include "arch35/scatter_add_with_sorted_simt.h"
#include "arch35/scatter_add_with_sorted_simt_determ.h"

using namespace AscendC;
using namespace ScatterAddWithSorted;

template <uint64_t TEMPLATE_MODE, uint64_t IS_SCALAR, uint64_t IS_DETERM, uint64_t ADDR_TYPE>
__global__ __aicore__ void scatter_add_with_sorted(
    GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(ScatterAddWithSortedSimdTilingData);

    if constexpr (TEMPLATE_MODE == TPL_MODE_EMPTY) {
        return;
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_SIMD) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_SIMD", ScatterAddWithSortedSimdTilingData);
        GET_TILING_DATA_WITH_STRUCT(ScatterAddWithSortedSimdTilingData, tilingData, tiling);

        if constexpr (IS_DETERM == TPL_DETERM_TRUE) {
            if (tilingData.withPos) {
                ScatterAddWithSortedSimdDterm<DTYPE_VAR, DTYPE_SORTED_INDEX, true> op;
                op.Init(var, updates, indices, pos, y, workspace, pipe, &tilingData);
                op.Process();
            } else {
                ScatterAddWithSortedSimdDterm<DTYPE_VAR, DTYPE_SORTED_INDEX, false> op;
                op.Init(var, updates, indices, pos, y, workspace, pipe, &tilingData);
                op.Process();
            }
            SyncAll();
            pipe.Reset();
            ScatterAddWithSortedSimdDtermWorkspace<DTYPE_VAR, DTYPE_SORTED_INDEX> workspaceOp;
            workspaceOp.Init(var, y, workspace, pipe, &tilingData);
            workspaceOp.Process();
        } else if constexpr (IS_SCALAR == TPL_SCALAR_TRUE) {
            ScatterAddWithSortedSIMD<DTYPE_VAR, DTYPE_SORTED_INDEX, true, false> op;
            op.Init(var, updates, indices, pos, y, workspace, pipe, &tilingData);
            op.Process();
        } else {
            if (tilingData.withPos) {
                ScatterAddWithSortedSIMD<DTYPE_VAR, DTYPE_SORTED_INDEX, false, true> op;
                op.Init(var, updates, indices, pos, y, workspace, pipe, &tilingData);
                op.Process();
            } else {
                ScatterAddWithSortedSIMD<DTYPE_VAR, DTYPE_SORTED_INDEX, false, false> op;
                op.Init(var, updates, indices, pos, y, workspace, pipe, &tilingData);
                op.Process();
            }
        }
    } else if constexpr (TEMPLATE_MODE == TPL_MODE_SIMT) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_MODE_SIMT", ScatterAddWithSortedSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(ScatterAddWithSortedSimtTilingData, tilingData, tiling);

        if constexpr (IS_DETERM == TPL_DETERM_TRUE) {
            if constexpr (ADDR_TYPE == TPL_ADDR_B32) {
                if (tilingData.withPos) {
                    ScatterAddWithSortedDetermSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint32_t, true> op(tilingData);
                    op.Init(var, updates, indices, pos, workspace);
                    op.Process();
                } else {
                    ScatterAddWithSortedDetermSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint32_t, false> op(tilingData);
                    op.Init(var, updates, indices, pos, workspace);
                    op.Process();
                }
            } else {
                if (tilingData.withPos) {
                    ScatterAddWithSortedDetermSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint64_t, true> op(tilingData);
                    op.Init(var, updates, indices, pos, workspace);
                    op.Process();
                } else {
                    ScatterAddWithSortedDetermSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint64_t, false> op(tilingData);
                    op.Init(var, updates, indices, pos, workspace);
                    op.Process();
                }
            }
        } else if constexpr (IS_SCALAR == TPL_SCALAR_TRUE) {
            if constexpr (ADDR_TYPE == TPL_ADDR_B32) {
                ScatterAddWithSortedSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint32_t, true, true> op(tilingData);
                op.Init(var, updates, indices, pos);
                op.Process();
            } else {
                ScatterAddWithSortedSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint64_t, true, true> op(tilingData);
                op.Init(var, updates, indices, pos);
                op.Process();
            }
        } else {
            if constexpr (ADDR_TYPE == TPL_ADDR_B32) {
                if (tilingData.withPos) {
                    ScatterAddWithSortedSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint32_t, false, true> op(tilingData);
                    op.Init(var, updates, indices, pos);
                    op.Process();
                } else {
                    ScatterAddWithSortedSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint32_t, false, false> op(tilingData);
                    op.Init(var, updates, indices, pos);
                    op.Process();
                }
            } else {
                if (tilingData.withPos) {
                    ScatterAddWithSortedSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint64_t, false, true> op(tilingData);
                    op.Init(var, updates, indices, pos);
                    op.Process();
                } else {
                    ScatterAddWithSortedSIMT<DTYPE_VAR, DTYPE_SORTED_INDEX, uint64_t, false, false> op(tilingData);
                    op.Init(var, updates, indices, pos);
                    op.Process();
                }
            }
        }
    }
}
