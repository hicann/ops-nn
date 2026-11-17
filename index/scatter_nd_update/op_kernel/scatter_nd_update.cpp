/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update.cpp
 * \brief scatter_nd_update arch22 kernel entry
 */
#include "./arch22/scatter_nd_update.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#include "./arch22/scatter_nd_update_balanced.h"
#endif
#include "./arch22/scatter_nd_update_linear_index.h"
#include "./arch22/scatter_nd_update_no_sort.h"
#include "./arch22/scatter_nd_update_large_index.h"
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
#include "./arch22/scatter_nd_update_hp.h"
#endif

using namespace ScatterNdUpdate;

template <typename VarT, template <typename, bool> class ScatterKernelT>
__aicore__ inline void RunScatterAfterSync(GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace,
                                           const ScatterNdUpdateArch22TilingData& tilingData, AscendC::TPipe& tpipe,
                                           bool isView)
{
    AscendC::SyncAll();
    tpipe.Destroy();
    AscendC::TPipe pipe;
    if (isView) {
        ScatterKernelT<VarT, true> op2(updates, varRef, workspace, tilingData, pipe);
        op2.Process();
    } else {
        ScatterKernelT<VarT, false> op2(updates, varRef, workspace, tilingData, pipe);
        op2.Process();
    }
}

template <typename VarT, bool IsSort, typename IndexT, template <typename, bool> class ScatterKernelT>
__aicore__ inline void RunLinearIndexAndScatter(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace,
                                                const ScatterNdUpdateArch22TilingData& tilingData,
                                                AscendC::TPipe& tpipe, bool isView)
{
    ScatterNdUpdate::LinearIndexKernel<IsSort, IndexT> op1(indices, workspace, tilingData, tpipe);
    op1.Process();
    RunScatterAfterSync<VarT, ScatterKernelT>(updates, varRef, workspace, tilingData, tpipe, isView);
}

template <typename VarT>
__aicore__ inline void RunLargeIndex(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                                     const ScatterNdUpdateArch22TilingData& tilingData, AscendC::TPipe& tpipe,
                                     bool isView)
{
    if (isView) {
        ScatterNdUpdate::LargeIndexKernel<VarT, true> op(indices, updates, varRef, tilingData, tpipe);
        op.Process();
    } else {
        ScatterNdUpdate::LargeIndexKernel<VarT, false> op(indices, updates, varRef, tilingData, tpipe);
        op.Process();
    }
}

#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
template <typename VarT, typename IndicesT>
__aicore__ inline void RunHp(GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                             const ScatterNdUpdateArch22TilingData& tilingData, AscendC::TPipe& tpipe, bool isView)
{
    if (isView) {
        ScatterNdUpdate::ScatterNdUpdateHpKernel<VarT, IndicesT, true> op(indices, updates, varRef, tilingData, tpipe);
        op.Process();
    } else {
        ScatterNdUpdate::ScatterNdUpdateHpKernel<VarT, IndicesT, false> op(indices, updates, varRef, tilingData, tpipe);
        op.Process();
    }
}
#endif

extern "C" __global__ __aicore__ void scatter_nd_update(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    // 仅A2/A3注册新路径，其他平台保留原有任务类型推导和算子分支。
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#else
    // 原常规路径包含SyncAll，大索引路径不需要核间屏障。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(30, KERNEL_TYPE_AIV_ONLY);
#endif
    KERNEL_TASK_TYPE(40, KERNEL_TYPE_MIX_AIV_1_0);
#endif
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
#if defined(DTYPE_VAR) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    if (TILING_KEY_IS(40)) {
        ScatterNdUpdateArch22TilingData localTiling;
        AscendC::TPipe localPipe;
        LoadLocalTiling<DTYPE_VAR>(tiling, localTiling, localPipe);
        if (localTiling.localIndexMode != 0) {
            if (localTiling.viewTiling.isViewStride0 != 0) {
                ScatterNdUpdateBalancedKernel<DTYPE_VAR, true>::ProcessLocal(indices, updates, varRef, localTiling,
                                                                             localPipe);
            } else {
                ScatterNdUpdateBalancedKernel<DTYPE_VAR, false>::ProcessLocal(indices, updates, varRef, localTiling,
                                                                              localPipe);
            }
            return;
        }
        localPipe.Destroy();
        AscendC::TPipe tpipe;
        bool isView = localTiling.viewTiling.isViewStride0 != 0;
        if (UseLocalIndices(localTiling)) {
            if (isView) {
                ScatterNdUpdateBalancedKernel<DTYPE_VAR, true> op(updates, varRef, workspace, localTiling, tpipe,
                                                                  indices);
                op.Process();
            } else {
                ScatterNdUpdateBalancedKernel<DTYPE_VAR, false> op(updates, varRef, workspace, localTiling, tpipe,
                                                                   indices);
                op.Process();
            }
            return;
        }
        if (localTiling.linearIndexTiling.isInt64Indices != 0) {
            LinearIndexKernel<true, int64_t, true> op(indices, workspace, localTiling, tpipe);
            op.Process();
        } else {
            LinearIndexKernel<true, int, true> op(indices, workspace, localTiling, tpipe);
            op.Process();
        }
        RunScatterAfterSync<DTYPE_VAR, ScatterNdUpdateBalancedKernel>(updates, varRef, workspace, localTiling, tpipe,
                                                                      isView);
        return;
    }
#endif
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;
#if (defined(DTYPE_VAR))
    // tilingKey: indexType * 10 + sortFlag
    // indexType: 1=int32, 2=int64(cast), 3=int64(large); sortFlag: 0=unsorted, 1=sorted
    bool isView = tilingData.viewTiling.isViewStride0 != 0;
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    // HP path: split by indices, fuse LinearIndex + Scatter on each core, no SyncAll.
    // Trade-off: writes to duplicate indices are non-deterministic across cores.
    // tilingKey 30 (int64 large index) is not representable as int32 linearIndex,
    // so it still falls back to the deterministic LargeIndex kernel.
    if (TILING_KEY_IS(11) || TILING_KEY_IS(10)) {
        RunHp<DTYPE_VAR, int>(indices, updates, varRef, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(21) || TILING_KEY_IS(20)) {
        RunHp<DTYPE_VAR, int64_t>(indices, updates, varRef, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(30)) {
        RunLargeIndex<DTYPE_VAR>(indices, updates, varRef, tilingData, tpipe, isView);
    }
#else
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    if (TILING_KEY_IS(11)) {
        RunLinearIndexAndScatter<DTYPE_VAR, true, int, ScatterNdUpdate::ScatterNdUpdateKernel>(
            indices, updates, varRef, workspace, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(10)) {
        RunLinearIndexAndScatter<DTYPE_VAR, false, int, ScatterNdUpdate::ScatterNdUpdateKernelNoSort>(
            indices, updates, varRef, workspace, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(21)) {
        RunLinearIndexAndScatter<DTYPE_VAR, true, int64_t, ScatterNdUpdate::ScatterNdUpdateKernel>(
            indices, updates, varRef, workspace, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(20)) {
        RunLinearIndexAndScatter<DTYPE_VAR, false, int64_t, ScatterNdUpdate::ScatterNdUpdateKernelNoSort>(
            indices, updates, varRef, workspace, tilingData, tpipe, isView);
    } else if (TILING_KEY_IS(30)) {
        RunLargeIndex<DTYPE_VAR>(indices, updates, varRef, tilingData, tpipe, isView);
    }
#else
    RunLinearIndexAndScatter<DTYPE_VAR, false, int, ScatterNdUpdate::ScatterNdUpdateKernelNoSort>(
        indices, updates, varRef, workspace, tilingData, tpipe, isView);
#endif
#endif
#endif
}
