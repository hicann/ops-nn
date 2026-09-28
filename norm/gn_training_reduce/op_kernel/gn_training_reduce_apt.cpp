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
 * \file gn_training_reduce_apt.cpp
 * \brief GNTrainingReduce kernel entry for ascend950 (arch35).
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/gn_training_reduce_struct.h"
#include "arch35/gn_training_reduce_tiling_struct.h"
#include "arch35/gn_training_reduce_base.h"
#include "arch35/gn_training_reduce_group.h"
#include "arch35/gn_training_reduce_empty.h"
#include "arch35/gn_training_reduce_onepass.h"

using namespace NsGNTrainingReduce;

template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void gn_training_reduce(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        GET_TILING_DATA_WITH_STRUCT(GNTrainingReduceEmptyTilingData, tilingData, tiling);
        GNTrainingReduceEmptyKernel<DTYPE_X> op;
        op.Init(x, sum, squareSum, &tilingData, &pipe);
        for (int32_t processIdx = 0; processIdx < N_REDUCES; processIdx++) {
            op.Process(processIdx);
        }
    } else if constexpr (isGroup) {
        GET_TILING_DATA_WITH_STRUCT(GNTrainingReduceTilingData, tilingData, tiling);
        GNTrainingReduceGroupKernel<DTYPE_X> op;
        op.InitGroup(x, sum, squareSum, workspace, &tilingData, &pipe);
        for (int32_t processIdx = 0; processIdx < N_REDUCES; processIdx++) {
            op.ProcessGroup(processIdx);
            AscendC::SyncAll();
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(GNTrainingReduceTilingData, tilingData, tiling);
        // The single-load dual-moment path only saves a GM re-load when the R axis
        // spans more than one UB tile.  When the whole R segment fits one tile
        // (rLoopCntTotal == 1) the original two-pass base kernel is cheaper and is
        // kept, so route exactly those cases back.
        if (tilingData.rLoopCntTotal <= 1) {
            GNTrainingReduceBaseKernel<DTYPE_X> op;
            op.Init(x, sum, squareSum, &tilingData, &pipe);
            for (int32_t processIdx = 0; processIdx < N_REDUCES; processIdx++) {
                op.Process(processIdx);
            }
        } else {
            GNTrainingReduceOnePassKernel<DTYPE_X> op;
            op.Init(x, sum, squareSum, &tilingData, &pipe);
            op.ProcessAll();
        }
    }
}
