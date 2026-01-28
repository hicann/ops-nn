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
 * \file layer_norm_grad_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/layer_norm_grad_recompute_gamma_beta_impl.h"
#include "arch35/layer_norm_grad_recompute_backward_impl.h"
#include "arch35/layer_norm_grad_grouped_reduce_big_m_impl.h"
#include "arch35/layer_norm_grad_grouped_reduce_big_n_impl.h"

using namespace LayerNormGrad;

#define RECOMPUTE_KEY 500

#define GROUPED_REDUCE_BIG_M 600

#define GROUPED_REDUCE_BIG_N 700

template <typename DY_TYPE, typename GAMMA_TYPE, typename PD_GAMMA_TYPE>
__aicore__ inline void InvokeLayerNormGradRecomputeImpl(
    GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(LayerNormGradTilingDataRecompute, tiling_data_in, tiling);
    const LayerNormGradTilingDataRecompute* __restrict tilingData = &tiling_data_in;
    TPipe pipeIn;
    if (tilingData->pdgammaIsRequire || tilingData->pdbetaIsRequire) {
        PRELOAD(4);
        LayerNormGradRecomputeGammaBeta<DY_TYPE, PD_GAMMA_TYPE> opGammaBetaBackward;
        opGammaBetaBackward.Init(dy, x, var, mean, pd_gamma, pd_beta, workspace, tilingData, &pipeIn);
        opGammaBetaBackward.Process();
        if (!tilingData->pdxIsRequire) {
            return;
        }
        SyncAll();
        pipeIn.Reset();
    }
    PRELOAD(4);
    LayerNormGradRecomputeBackward<DY_TYPE, GAMMA_TYPE> opBackward;
    opBackward.Init(dy, x, var, mean, gamma, pd_x, workspace, tilingData, &pipeIn);
    opBackward.Process();   
}

template <typename DY_TYPE, typename GAMMA_TYPE, typename PD_GAMMA_TYPE>
__aicore__ inline void InvokeLayerNormGradGroupedReduceBigMImpl(
    GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(LayerNormGradTilingDataGroupedReduceBigM, tiling_data_in, tiling);
    const LayerNormGradTilingDataGroupedReduceBigM* __restrict tilingData = &tiling_data_in;

    if (tilingData->pdxIsRequire) {
        PRELOAD(4);
        TPipe pipeIn;
        LayerNormGradGroupedReduceBigMBackward<DY_TYPE, GAMMA_TYPE> opBackward;
        opBackward.Init(dy, x, var, mean, gamma, pd_x, workspace, tilingData, &pipeIn);
        opBackward.Process();
        pipeIn.Destroy();
    }

    if (tilingData->pdbetaIsRequire || tilingData->pdgammaIsRequire) {
        PRELOAD(4);
        TPipe pipeGammaBetaBackward;
        LayerNormGradGroupedReduceBigMGammaBeta<DY_TYPE, PD_GAMMA_TYPE> opGammaBetaBackward;
        opGammaBetaBackward.Init(dy, x, var, mean, pd_gamma, pd_beta, workspace, tilingData, &pipeGammaBetaBackward);
        opGammaBetaBackward.Process();
    }
}

template <typename DY_TYPE, typename GAMMA_TYPE, typename PD_GAMMA_TYPE>
__aicore__ inline void InvokeLayerNormGradGroupedReduceBigNImpl(GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean,
                                                                  GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma,
                                                                  GM_ADDR pd_beta, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(LayerNormGradTilingDataGroupedReduceBigN, tiling_data_in, tiling);
    const LayerNormGradTilingDataGroupedReduceBigN* __restrict tilingData = &tiling_data_in;
    TPipe pipeIn;
    if (tilingData->pdbetaIsRequire || tilingData->pdgammaIsRequire) {
        LayerNormGradGroupedReduceBigNGammaBeta<DY_TYPE, PD_GAMMA_TYPE> opGammaBetaBackward;
        opGammaBetaBackward.Init(dy, x, var, mean, pd_gamma, pd_beta, workspace, tilingData, &pipeIn);
        opGammaBetaBackward.Process();
        if (!tilingData->pdxIsRequire) {
            return;
        }
        SyncAll();
        pipeIn.Reset();
    }
    PRELOAD(4);  // 4*2K
    LayerNormGradGroupedReduceBigNBackward<DY_TYPE, GAMMA_TYPE> opBackward;
    opBackward.Init(dy, x, var, mean, gamma, pd_x, workspace, tilingData, &pipeIn);
    opBackward.Process();
}

extern "C" __global__ __aicore__ void layer_norm_grad(
    GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pd_x, GM_ADDR pd_gamma, GM_ADDR pd_beta,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(RECOMPUTE_KEY)) {
        InvokeLayerNormGradRecomputeImpl<DTYPE_DY, DTYPE_GAMMA, DTYPE_PD_GAMMA>(
            dy, x, var, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling);
        return;
    }

    if (TILING_KEY_IS(GROUPED_REDUCE_BIG_M)) {
        InvokeLayerNormGradGroupedReduceBigMImpl<DTYPE_DY, DTYPE_GAMMA, DTYPE_PD_GAMMA>(
            dy, x, var, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling);
        return;
    }

    if (TILING_KEY_IS(GROUPED_REDUCE_BIG_N)) {
        InvokeLayerNormGradGroupedReduceBigNImpl<DTYPE_DY, DTYPE_GAMMA, DTYPE_PD_GAMMA>(
            dy, x, var, mean, gamma, pd_x, pd_gamma, pd_beta, usrWorkspace, tiling);
        return;
    }

    return;
}
