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
 * \file rms_norm_grad_apt.cpp
 * \brief RmsNormGrad Kernel File
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/rms_norm_grad_regbase_dx_full_load.h"
#include "arch35/rms_norm_grad_regbase_dx_split_d.h"
#include "arch35/rms_norm_grad_regbase_dgamma.h"
#include "arch35/rms_norm_grad_empty_dgamma.h"
#include "arch35/rms_norm_grad_regbase_dgamma_big_m.h"

extern "C" __global__ __aicore__ void rms_norm_grad(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(8000)) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormGradEmptyTilingData, tilingDataIn, tiling);
        AscendC::TPipe pipe;
        const RmsNormGradEmptyTilingData* __restrict tilingData = &tilingDataIn;
        RmsNormGrad::EmptyDgamma<2> opDG(&pipe, tilingData);
        opDG.Init(dgamma);
        opDG.Process();
    } else if (TILING_KEY_IS(9000)) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormGradBigMTilingData, tilingDataIn, tiling);
        AscendC::TPipe pipe;
        const RmsNormGradRegbaseDxTilingData* __restrict tilingDataDx = &tilingDataIn.dxTilingData;
        RmsNormGrad::RegbaseDxFullLoad<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DGAMMA> opDX(&pipe, tilingDataDx);
        opDX.Init(dy, x, rstd, gamma, dx, dgamma);
        opDX.Process();
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe.Reset();
        const RmsNormGradBigMTilingData* __restrict tilingDataDgamma = &tilingDataIn;
        RmsNormGrad::RmsNormGradDgammaBigM<DTYPE_DY> opDgamma;
        opDgamma.Init(dy, x, rstd, dgamma, workspace, tilingDataDgamma, &pipe);
        opDgamma.Process();
    } else if (TILING_KEY_IS(9010)) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormGradBigMTilingData, tilingDataIn, tiling);
        AscendC::TPipe pipe;
        const RmsNormGradRegbaseDxTilingData* __restrict tilingDataDx = &tilingDataIn.dxTilingData;
        RmsNormGrad::RegbaseDxSplitD<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DGAMMA> opDX(&pipe, tilingDataDx);
        opDX.Init(dy, x, rstd, gamma, dx, dgamma);
        opDX.Process();
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe.Reset();
        const RmsNormGradBigMTilingData* __restrict tilingDataDgamma = &tilingDataIn;
        RmsNormGrad::RmsNormGradDgammaBigM<DTYPE_DY> opDgamma;
        opDgamma.Init(dy, x, rstd, dgamma, workspace, tilingDataDgamma, &pipe);
        opDgamma.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(RmsNormGradRegbaseTilingData, tilingDataIn, tiling);
        AscendC::TPipe pipe;
        const RmsNormGradRegbaseTilingData* __restrict tilingDataDgamma = &tilingDataIn;
        const RmsNormGradRegbaseDxTilingData* __restrict tilingDataDx = &tilingDataIn.dxTilingData;
        if (TILING_KEY_IS(7000)) {
            RmsNormGrad::RegbaseDxFullLoad<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DGAMMA> opDX(&pipe, tilingDataDx);
            opDX.Init(dy, x, rstd, gamma, dx, dgamma);
            opDX.Process();
            AscendC::PipeBarrier<PIPE_ALL>();
            pipe.Reset();
            RmsNormGrad::RegbaseDgamma<DTYPE_DY, DTYPE_X, DTYPE_RSTD, 7000, 2> opDG(&pipe, tilingDataDgamma);
            opDG.Init(dy, x, rstd, gamma, dx, dgamma);
            opDG.Process();
        } else if (TILING_KEY_IS(7001)) {
            RmsNormGrad::RegbaseDxFullLoad<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DGAMMA> opDX(&pipe, tilingDataDx);
            opDX.Init(dy, x, rstd, gamma, dx, dgamma);
            opDX.Process();
            AscendC::PipeBarrier<PIPE_ALL>();
            pipe.Reset();
            RmsNormGrad::RegbaseDgamma<DTYPE_DY, DTYPE_X, DTYPE_RSTD, 7001, 2> opDG(&pipe, tilingDataDgamma);
            opDG.Init(dy, x, rstd, gamma, dx, dgamma);
            opDG.ProcessWithLargeRows();
        } else if (TILING_KEY_IS(7010)) {
            RmsNormGrad::RegbaseDxSplitD<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DGAMMA> opDX(&pipe, tilingDataDx);
            opDX.Init(dy, x, rstd, gamma, dx, dgamma);
            opDX.Process();
            AscendC::PipeBarrier<PIPE_ALL>();
            pipe.Reset();
            RmsNormGrad::RegbaseDgamma<DTYPE_DY, DTYPE_X, DTYPE_RSTD, 7000, 2> opDG(&pipe, tilingDataDgamma);
            opDG.Init(dy, x, rstd, gamma, dx, dgamma);
            opDG.Process();
        } else if (TILING_KEY_IS(7011)) {
            RmsNormGrad::RegbaseDxSplitD<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DGAMMA> opDX(&pipe, tilingDataDx);
            opDX.Init(dy, x, rstd, gamma, dx, dgamma);
            opDX.Process();
            AscendC::PipeBarrier<PIPE_ALL>();
            pipe.Reset();
            RmsNormGrad::RegbaseDgamma<DTYPE_DY, DTYPE_X, DTYPE_RSTD, 7001, 2> opDG(&pipe, tilingDataDgamma);
            opDG.Init(dy, x, rstd, gamma, dx, dgamma);
            opDG.ProcessWithLargeRows();
        }
    }
}