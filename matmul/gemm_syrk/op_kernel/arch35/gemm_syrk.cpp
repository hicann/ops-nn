/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_syrk.cpp
 * \brief GemmSyrk kernel entry (DAV_3510 / Ascend 950 & 350, Blaze MIX 1 AIC : 2 AIV).
 *
 * TPL_TRANS selects the storage layout of a: false keeps the row-major
 * (..., m, k) ND view, true binds the transposed (..., k, m) storage through
 * a DNExt (m, k) GM view so the kernel computes C = alpha * (a^T @ a) + beta * C.
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "gemm_syrk_basic.h"
#include "gemm_syrk_tiling_key.h"

using namespace AscendC;

#ifndef DTYPE_A
#define DTYPE_A half
#endif
#ifndef DTYPE_C
#define DTYPE_C half
#endif

template <int TPL_KERNEL_TYPE, bool TPL_TRANS>
__global__ __aicore__ void gemm_syrk(GM_ADDR aGM, GM_ADDR cInGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(GemmSyrkTilingData);
    AscendC::InitSocState();
    if constexpr (TPL_KERNEL_TYPE == SYRK_KERNEL_BASIC) {
        GET_TILING_DATA_WITH_STRUCT(GemmSyrkTilingData, tilingData, tilingGM);
        GemmSyrkAdvanced::GemmSyrkBlazeKernel<DTYPE_A, TPL_TRANS>(aGM, cInGM, cGM, workspaceGM, tilingData);
    }
}
