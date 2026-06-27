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
 * \file rms_norm_grad_quant.cpp
 * \brief RmsNormGradQuant Kernel File
 */

#include "kernel_operator.h"
#include "arch35/rms_norm_grad_quant_dx_full_load.h"
#include "arch35/rms_norm_grad_quant_dx_split_d.h"
#include "arch35/rms_norm_grad_quant_dgamma.h"
#include "arch35/rms_norm_grad_quant_dgamma_big_m.h"
#include "arch35/rms_norm_grad_quant_tiling_key.h"
#include "arch35/rms_norm_grad_quant_tiling_data.h"
#include "arch35/rms_norm_grad_quant_empty_dgamma.h"
using namespace AscendC;
using namespace RmsNormGradQuant;

template <int8_t COMPUTE_DX_MODE, int8_t COMPUTE_DGAMMA_MODE, int8_t COMPUTE_OFFSET_X_MODE, int8_t COMPUTE_DIV_MODE>
__global__ __aicore__ void rms_norm_grad_quant(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x, GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(RmsNormGradQuantRegbaseTilingData);
    if (workspace == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if constexpr (COMPUTE_DGAMMA_MODE == COMPUTE_MODE_DGAMMA_EMPTY) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormGradQuantRegbaseEmptyTilingData, tilingData, tiling);
        EmptyDgamma opDG(&pipe, &tilingData);
        opDG.Init(dgamma);
        opDG.Process();
    } else if constexpr (COMPUTE_DGAMMA_MODE == COMPUTE_MODE_DGAMMA_BIG_M) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormGradQuantRegbaseBigMTilingData, tilingData, tiling);
        const RmsNormGradQuantRegbaseDxTilingData* __restrict tilingDataDx = &tilingData.dxTilingData;
        if constexpr (COMPUTE_DX_MODE == COMPUTE_MODE_DX_FULL_LOAD) {
            RmsNormGradQuant::RegbaseDxFullLoad<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DX, DTYPE_DGAMMA, DTYPE_SCALES_X, int32_t,
                COMPUTE_OFFSET_X_MODE == COMPUTE_MODE_WITH_OFFSET_X,
                COMPUTE_DIV_MODE == COMPUTE_MODE_DIV_MODE> opDX(&pipe, tilingDataDx);
            opDX.Init(dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma);
            opDX.Process();
        } else if constexpr (COMPUTE_DX_MODE == COMPUTE_MODE_DX_SPLIT_D) {
            RmsNormGradQuant::RegbaseDxSplitD<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DX, DTYPE_DGAMMA, DTYPE_SCALES_X, int32_t,
                COMPUTE_OFFSET_X_MODE == COMPUTE_MODE_WITH_OFFSET_X,
                COMPUTE_DIV_MODE == COMPUTE_MODE_DIV_MODE> opDX(&pipe, tilingDataDx);
            opDX.Init(dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma);
            opDX.Process();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe.Reset();
        RmsNormGradQuant::RmsNormGradQuantDgammaBigM<DTYPE_DY> opDgamma;
        opDgamma.Init(dy, x, rstd, dgamma, workspace, &tilingData, &pipe);
        opDgamma.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(RmsNormGradQuantRegbaseTilingData, tilingData, tiling);
        const RmsNormGradQuantRegbaseDxTilingData* __restrict tilingDataDx = &tilingData.dxTilingData;
        if constexpr (COMPUTE_DX_MODE == COMPUTE_MODE_DX_FULL_LOAD) {
            RmsNormGradQuant::RegbaseDxFullLoad<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DX, DTYPE_DGAMMA, DTYPE_SCALES_X, int32_t,
                COMPUTE_OFFSET_X_MODE == COMPUTE_MODE_WITH_OFFSET_X,
                COMPUTE_DIV_MODE == COMPUTE_MODE_DIV_MODE> opDX(&pipe, tilingDataDx);
            opDX.Init(dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma);
            opDX.Process();
        } else if constexpr (COMPUTE_DX_MODE == COMPUTE_MODE_DX_SPLIT_D) {
            RmsNormGradQuant::RegbaseDxSplitD<DTYPE_DY, DTYPE_X, DTYPE_GAMMA, DTYPE_DX, DTYPE_DGAMMA, DTYPE_SCALES_X, int32_t,
                COMPUTE_OFFSET_X_MODE == COMPUTE_MODE_WITH_OFFSET_X,
                COMPUTE_DIV_MODE == COMPUTE_MODE_DIV_MODE> opDX(&pipe, tilingDataDx);
            opDX.Init(dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma);
            opDX.Process();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe.Reset();
        const RmsNormGradQuantRegbaseTilingData* __restrict tilingDataDgamma = &tilingData;
        if constexpr (COMPUTE_DGAMMA_MODE == COMPUTE_MODE_DGAMMA_FULL_LOAD) {
            RmsNormGradQuant::RegbaseDgamma<DTYPE_DY, DTYPE_X, DTYPE_RSTD, true, 2> opDgamma(&pipe, tilingDataDgamma);
            opDgamma.Init(dy, x, rstd, gamma, dx, dgamma);
            opDgamma.Process();
        } else if constexpr (COMPUTE_DGAMMA_MODE == COMPUTE_MODE_DGAMMA_WITH_LARGE_ROWS) {
            RmsNormGradQuant::RegbaseDgamma<DTYPE_DY, DTYPE_X, DTYPE_RSTD, false, 2> opDgamma(&pipe, tilingDataDgamma);
            opDgamma.Init(dy, x, rstd, gamma, dx, dgamma);
            opDgamma.ProcessWithLargeRows();
        }
    }
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
