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
 * \file fused_mat_mul_input_k_eq_zero_copy_x3.h
 * \brief
 */
#pragma once

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "std/algorithm.h"
#else
#include "kernel_operator.h"
#endif
#include "../../mat_mul_v3/mat_mul_v3_common.h"
#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"

namespace MatmulV3Advanced {

using namespace AscendC;

constexpr uint64_t K_EQ_ZERO_COPY_UB_SIZE = 32 * 1024;

template <class X_DTYPE>
__aicore__ inline void MatMulInputKEqZeroCopyX3ToOutput(GM_ADDR x3GM, GM_ADDR yGM,
                                                        const MatMulV3KEqZeroBasicTilingData& tilingData)
{
    if ASCEND_IS_AIC {
        return;
    }

    uint64_t aivNum = tilingData.aivNum;
    uint64_t totalDataAmount = tilingData.totalDataAmount;
    if (aivNum == 0 || totalDataAmount == 0) {
        return;
    }
    uint64_t everyAivDataCount = Ceil(totalDataAmount, aivNum);
    uint64_t usedAivNum = MMV3DivCeil(totalDataAmount, everyAivDataCount);
    uint64_t tailDataCount = totalDataAmount - everyAivDataCount * (usedAivNum - 1);

    AscendC::GlobalTensor<X_DTYPE> x3Global;
    AscendC::GlobalTensor<X_DTYPE> outputGlobal;
    x3Global.SetGlobalBuffer(reinterpret_cast<__gm__ X_DTYPE*>(x3GM), totalDataAmount);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ X_DTYPE*>(yGM), totalDataAmount);

    uint64_t coreIdx = AscendC::GetBlockIdx();
    if (coreIdx >= usedAivNum) {
        return;
    }

    uint64_t copyDataAmount = everyAivDataCount;
    if (tailDataCount != 0 && coreIdx == usedAivNum - 1) {
        copyDataAmount = tailDataCount;
    }

    uint64_t coreOffset = coreIdx * everyAivDataCount;
    uint64_t chunkDataCount = K_EQ_ZERO_COPY_UB_SIZE / sizeof(X_DTYPE);
    chunkDataCount = chunkDataCount == 0 ? 1 : chunkDataCount;

    TPipe pipe;
    TBuf<TPosition::VECCALC> localBuffer;
    pipe.InitBuffer(localBuffer, chunkDataCount * sizeof(X_DTYPE));
    LocalTensor<X_DTYPE> tmpBuf = localBuffer.Get<X_DTYPE>();

    for (uint64_t copied = 0; copied < copyDataAmount; copied += chunkDataCount) {
        uint64_t curCopyCount = AscendC::Std::min(chunkDataCount, copyDataAmount - copied);
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(curCopyCount * sizeof(X_DTYPE)), 0, 0, 0};
        DataCopyPadExtParams<X_DTYPE> padParams{false, 0, 0, static_cast<X_DTYPE>(0)};
        DataCopyPad(tmpBuf, x3Global[coreOffset + copied], copyInParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curCopyCount * sizeof(X_DTYPE)), 0, 0, 0};
        DataCopyPad(outputGlobal[coreOffset + copied], tmpBuf, copyOutParams);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    }
}
} // namespace MatmulV3Advanced
