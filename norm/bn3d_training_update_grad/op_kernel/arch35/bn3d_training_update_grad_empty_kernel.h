/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// bn3d_training_update_grad_package/op_kernel/arch35/bn3d_training_update_grad_empty_kernel.h
// =============================================================================
//
// BN3DTrainingUpdateGradEmptyKernel<DType> — empty tensor variant (tilingKey 2).
// Both outputs are fp32; empty reduction sum = 0 -> Duplicate 0.0f.
// Shared constants / helpers are reused from bn3d_training_update_grad_base_kernel.h.
// =============================================================================
#ifndef BN3D_TRAINING_UPDATE_GRAD_EMPTY_KERNEL_H_
#define BN3D_TRAINING_UPDATE_GRAD_EMPTY_KERNEL_H_

#include "bn3d_training_update_grad_base_kernel.h" // shared constants / helpers

// ===========================================================================
// BN3DTrainingUpdateGradEmptyKernel<DType> — empty tensor (tilingKey 2).
// Both outputs are fp32; empty reduction sum = 0 -> Duplicate 0.0f.
// ===========================================================================
template <typename DType>
class BN3DTrainingUpdateGradEmptyKernel {
public:
    using DT = DType;
    __aicore__ inline BN3DTrainingUpdateGradEmptyKernel() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR x, GM_ADDR batchMean, GM_ADDR batchVariance, GM_ADDR diffScale,
                                GM_ADDR diffOffset, const BN3DTrainingUpdateGradEmptyTilingData* td, TPipe* pipe)
    {
        (void)grads;
        (void)x;
        (void)batchMean;
        (void)batchVariance;
        td_ = td;
        diffScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diffScale));
        diffOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diffOffset));
        pipe_ = pipe;
        pipe_->InitBuffer(outBuf_, td_->postBufSize);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
            return;
        }

        int64_t aStart = 0, aEnd = 0;
        if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
            aStart = blockIdx * td_->aBigCoreLoopCnt * td_->aUbFactor;
            aEnd = aStart + td_->aBigCoreLoopCnt * td_->aUbFactor;
        } else {
            aStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt * td_->aUbFactor +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt * td_->aUbFactor;
            aEnd = aStart + td_->aSmallCoreLoopCnt * td_->aUbFactor;
        }
        if (aEnd > td_->aTotal) {
            aEnd = td_->aTotal;
        }
        if (aStart >= aEnd) {
            return;
        }

        mutexId_ = AscendC::AllocMutexID();

        AscendC::Mutex::Lock<PIPE_V>(mutexId_);
        DuplicateOutput();
        AscendC::Mutex::Unlock<PIPE_V>(mutexId_);

        AscendC::Mutex::Lock<PIPE_MTE3>(mutexId_);
        for (int64_t aOff = aStart; aOff < aEnd; aOff += td_->aUbFactor) {
            const int64_t aLen = (aOff + td_->aUbFactor > aEnd) ? (aEnd - aOff) : td_->aUbFactor;
            CopyOut(aOff, aLen);
        }
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId_);

        AscendC::ReleaseMutexID(mutexId_);
    }

private:
    __aicore__ inline void DuplicateOutput()
    {
        __ubuf__ float* outPtr = reinterpret_cast<__ubuf__ float*>(outBuf_.Get<float>().GetPhyAddr());
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor);
        const uint32_t repDType = kVlBytes / sizeof(float);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(totalElems, repDType));
        asc_vf_call<Bn3dDuplicateVfImpl<float>>(outPtr, 0.0f, totalElems, repeatTime);
    }
    __aicore__ inline void CopyOut(int64_t outOff, int64_t aLen)
    {
        auto outLocal = outBuf_.Get<float>();
        DataCopyExtParams outParams;
        outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
        outParams.blockCount = 1;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(diffOffsetGm_[outOff], outLocal, outParams);
        DataCopyPad(diffScaleGm_[outOff], outLocal, outParams);
    }

    const BN3DTrainingUpdateGradEmptyTilingData* td_ = nullptr;
    GlobalTensor<float> diffScaleGm_, diffOffsetGm_;
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> outBuf_;
    uint8_t mutexId_ = 0;
};

#endif // BN3D_TRAINING_UPDATE_GRAD_EMPTY_KERNEL_H_
