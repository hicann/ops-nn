/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_input_k_eq_zero_clear_output.h
 * \brief
 */
#ifndef MAT_MUL_INPUT_K_EQ_ZERO_CLEAR_OUTPUT_H
#define MAT_MUL_INPUT_K_EQ_ZERO_CLEAR_OUTPUT_H

#ifndef DTYPE_Y
#define DTYPE_Y half
#endif
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "std/algorithm.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../mat_mul_v3_common.h"

namespace MatmulV3Advanced{

using namespace matmul;

__aicore__ inline void MatMulInputKEqZeroClearOutput(GM_ADDR biasGM, GM_ADDR cGM, const MatMulV3KEqZeroBasicTilingData& tilingData)
{
    if ASCEND_IS_AIC {
        return;
    }

    uint64_t AivNum = tilingData.aivNum;
    uint64_t totalDataAmount = tilingData.totalDataAmount;
    uint64_t everyAivDataCount = Ceil(totalDataAmount, AivNum);
    uint64_t usedAivNum = MMV3DivFloor(totalDataAmount, everyAivDataCount);
    uint64_t tailDataCount = totalDataAmount - everyAivDataCount * usedAivNum;

    AscendC::GlobalTensor<DTYPE_Y> outputGM;
    outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(cGM), totalDataAmount);

    uint64_t coreIdx = AscendC::GetBlockIdx();
    if (coreIdx > usedAivNum) {
        return;
    }

    uint64_t copyDataAmount = everyAivDataCount;
    if (tailDataCount != 0 && coreIdx == usedAivNum) {
        copyDataAmount = tailDataCount;
    }

    AscendC::InitOutput<DTYPE_Y>(outputGM[coreIdx * everyAivDataCount], static_cast<uint64_t>(copyDataAmount), (DTYPE_Y)0);
}
}
#endif