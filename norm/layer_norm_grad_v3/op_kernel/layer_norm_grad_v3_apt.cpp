/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_v3_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/layer_norm_grad_v3_recompute_gamma_beta_impl.h"
#include "arch35/layer_norm_grad_v3_recompute_backward_impl.h"
#include "arch35/layer_norm_grad_v3_grouped_reduce_big_m_impl.h"
#include "arch35/layer_norm_grad_v3_grouped_reduce_big_n_impl.h"

using namespace LayerNormGradV3;

#define RECOMPUTE_KEY 500

#define GROUPED_REDUCE_BIG_M 600

#define GROUPED_REDUCE_BIG_N 700

template <typename DY_TYPE, typename GAMMA_TYPE, typename PD_GAMMA_TYPE>
__aicore__ inline void InvokeLayerNormGradV3RecomputeImpl(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(LayerNormGradV3TilingDataRecompute, tiling_data_in, tiling);
    const LayerNormGradV3TilingDataRecompute* __restrict tilingData = &tiling_data_in;
    TPipe pipeIn;
    if (tilingData->pdgammaIsRequire || tilingData->pdbetaIsRequire) {
        PRELOAD(4);
        LayerNormGradV3RecomputeGammaBeta<DY_TYPE, PD_GAMMA_TYPE> opGammaBetaBackward;
        opGammaBetaBackward.Init(dy, x, rstd, mean, pd_gamma, pd_beta, workspace, tilingData, &pipeIn);
        opGammaBetaBackward.Process();
        if (!tilingData->pdxIsRequire) {
            return;
        }
        SyncAll();
        pipeIn.Reset();
    }
    PRELOAD(4);
    LayerNormGradV3RecomputeBackward<DY_TYPE, GAMMA_TYPE> opBackward;
    opBackward.Init(dy, x, rstd, mean, gamma, pd_x, workspace, tilingData, &pipeIn);
    opBackward.Process();   
}

template <typename DY_TYPE, typename GAMMA_TYPE, typename PD_GAMMA_TYPE>
__aicore__ inline void InvokeLayerNormGradV3GroupedReduceBigMImpl(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(LayerNormGradV3TilingDataGroupedReduceBigM, tiling_data_in, tiling);
    const LayerNormGradV3TilingDataGroupedReduceBigM* __restrict tilingData = &tiling_data_in;

    if (tilingData->pdxIsRequire) {
        PRELOAD(4);
        TPipe pipeIn;
        LayerNormGradV3GroupedReduceBigMBackward<DY_TYPE, GAMMA_TYPE> opBackward;
        opBackward.Init(dy, x, rstd, mean, gamma, pd_x, workspace, tilingData, &pipeIn);
        opBackward.Process();
        pipeIn.Destroy();
    }

    if (tilingData->pdbetaIsRequire || tilingData->pdgammaIsRequire) {
        PRELOAD(4);
        TPipe pipeGammaBetaBackward;
        LayerNormGradV3GroupedReduceBigMGammaBeta<DY_TYPE, PD_GAMMA_TYPE> opGammaBetaBackward;
        opGammaBetaBackward.Init(dy, x, rstd, mean, pd_gamma, pd_beta, workspace, tilingData, &pipeGammaBetaBackward);
        opGammaBetaBackward.Process();
    }
}

template <typename DY_TYPE, typename GAMMA_TYPE, typename PD_GAMMA_TYPE>
__aicore__ inline void InvokeLayerNormGradV3GroupedReduceBigNImpl(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean,
                                                                  GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma,
                                                                  GM_ADDR pd_beta, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(LayerNormGradV3TilingDataGroupedReduceBigN, tiling_data_in, tiling);
    const LayerNormGradV3TilingDataGroupedReduceBigN* __restrict tilingData = &tiling_data_in;
    TPipe pipeIn;
    if (tilingData->pdbetaIsRequire || tilingData->pdgammaIsRequire) {
        LayerNormGradV3GroupedReduceBigNGammaBeta<DY_TYPE, PD_GAMMA_TYPE> opGammaBetaBackward;
        opGammaBetaBackward.Init(dy, x, rstd, mean, pd_gamma, pd_beta, workspace, tilingData, &pipeIn);
        opGammaBetaBackward.Process();
        if (!tilingData->pdxIsRequire) {
            return;
        }
        SyncAll();
        pipeIn.Reset();
    }
    PRELOAD(4);  // 4*2K
    LayerNormGradV3GroupedReduceBigNBackward<DY_TYPE, GAMMA_TYPE> opBackward;
    opBackward.Init(dy, x, rstd, mean, gamma, pd_x, workspace, tilingData, &pipeIn);
    opBackward.Process();
}

extern "C" __global__ __aicore__ void layer_norm_grad_v3(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(RECOMPUTE_KEY)) {
        InvokeLayerNormGradV3RecomputeImpl<DTYPE_DY, DTYPE_GAMMA, DTYPE_PD_GAMMA>(
            dy, x, rstd, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling);
        return;
    }

    if (TILING_KEY_IS(GROUPED_REDUCE_BIG_M)) {
        InvokeLayerNormGradV3GroupedReduceBigMImpl<DTYPE_DY, DTYPE_GAMMA, DTYPE_PD_GAMMA>(
            dy, x, rstd, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling);
        return;
    }

    if (TILING_KEY_IS(GROUPED_REDUCE_BIG_N)) {
        InvokeLayerNormGradV3GroupedReduceBigNImpl<DTYPE_DY, DTYPE_GAMMA, DTYPE_PD_GAMMA>(
            dy, x, rstd, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling);
        return;
    }

    return;
}
