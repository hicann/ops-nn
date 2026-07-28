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
 * \file fused_mat_mul_input_k_eq_zero_copy_x3.h
 * \brief Copies x3 to the output for fused matmul K=0 Add.
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
__aicore__ inline void MatMulInputKEqZeroCopyX3ToOutputImpl(GM_ADDR x3GM, GM_ADDR yGM, uint64_t totalDataAmount,
                                                            uint64_t aivNum, uint64_t singleBatchDataAmount,
                                                            bool batchBroadcast)
{
    if ASCEND_IS_AIC {
        return;
    }

    if (aivNum == 0 || totalDataAmount == 0 || singleBatchDataAmount == 0) {
        return;
    }
    const uint64_t dataCountPerAiv = Ceil(totalDataAmount, aivNum);
    const uint64_t coreOffset = AscendC::GetBlockIdx() * dataCountPerAiv;
    if (coreOffset >= totalDataAmount) {
        return;
    }
    const uint64_t copyDataAmount = AscendC::Std::min(dataCountPerAiv, totalDataAmount - coreOffset);

    AscendC::GlobalTensor<X_DTYPE> x3Global;
    AscendC::GlobalTensor<X_DTYPE> outputGlobal;
    const uint64_t x3DataAmount = batchBroadcast ? singleBatchDataAmount : totalDataAmount;
    x3Global.SetGlobalBuffer(reinterpret_cast<__gm__ X_DTYPE*>(x3GM), x3DataAmount);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ X_DTYPE*>(yGM), totalDataAmount);
    const uint64_t chunkDataCount = K_EQ_ZERO_COPY_UB_SIZE / sizeof(X_DTYPE);

    TPipe pipe;
    TBuf<TPosition::VECCALC> localBuffer;
    pipe.InitBuffer(localBuffer, chunkDataCount * sizeof(X_DTYPE));
    LocalTensor<X_DTYPE> tmpBuf = localBuffer.Get<X_DTYPE>();

    uint64_t copied = 0;
    while (copied < copyDataAmount) {
        const uint64_t outputOffset = coreOffset + copied;
        const uint64_t x3Offset = batchBroadcast ? outputOffset % singleBatchDataAmount : outputOffset;
        uint64_t curCopyCount = AscendC::Std::min(chunkDataCount, copyDataAmount - copied);
        if (batchBroadcast) {
            // Keep a contiguous GM copy inside one x3 batch before wrapping to offset zero.
            curCopyCount = AscendC::Std::min(curCopyCount, singleBatchDataAmount - x3Offset);
        }
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curCopyCount * sizeof(X_DTYPE)), 0, 0, 0};
        DataCopyPadExtParams<X_DTYPE> padParams{false, 0, 0, static_cast<X_DTYPE>(0)};
        DataCopyPad(tmpBuf, x3Global[x3Offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(0));
        DataCopyPad(outputGlobal[outputOffset], tmpBuf, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        copied += curCopyCount;
    }
}

template <class X_DTYPE>
__aicore__ inline void MatMulInputKEqZeroCopyX3ToOutput(GM_ADDR x3GM, GM_ADDR yGM,
                                                        const MatMulV3KEqZeroBasicTilingData& tilingData)
{
    MatMulInputKEqZeroCopyX3ToOutputImpl<X_DTYPE>(x3GM, yGM, tilingData.totalDataAmount, tilingData.aivNum,
                                                  tilingData.totalDataAmount, false);
}

template <class X_DTYPE>
__aicore__ inline void MatMulInputKEqZeroCopyX3ToOutput(GM_ADDR x3GM, GM_ADDR yGM,
                                                        const BatchMatMulV3BasicTilingData& tilingData)
{
    const uint64_t singleBatchDataAmount = static_cast<uint64_t>(tilingData.matMulTilingData.m) *
                                           tilingData.matMulTilingData.n;
    const uint64_t totalDataAmount = singleBatchDataAmount * tilingData.batchDimAll;
    const bool batchBroadcast = tilingData.batchX3 == 1 && tilingData.batchDimAll > 1;
    MatMulInputKEqZeroCopyX3ToOutputImpl<X_DTYPE>(x3GM, yGM, totalDataAmount, tilingData.matMulTilingData.usedCoreNum,
                                                  singleBatchDataAmount, batchBroadcast);
}
} // namespace MatmulV3Advanced
