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
 * \file fake_quant_with_min_max_args_gradient_regbase.h
 * \brief fake_quant_with_min_max_args_gradient kernel (regbase / arch35)
 *
 * Self-check (per design §3.2 + §5.11):
 *  5)  gradient uses grad*mask01 multiplicative gating (NOT select(grad,0,mask))
 *  6)  clamp comparisons use >=/<= + Select (no Min/Max -- NaN-safe)
 *  11) sign-bit OR injection after Mul: vSignG = AND(vG_u32, 0x80000000);
 *      vOut_u32 = OR(vOut_u32, vSignG)   -- preserves ±0 sign that Mul drops.
 *  Forward-only items 1-4 are N/A here (no quantization path).
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_REGBASE_H_
#define FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_REGBASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "fake_quant_with_min_max_args_gradient_tilingdata.h"

namespace FakeQuantWithMinMaxArgsGradient {
using namespace AscendC;

constexpr uint32_t FQ_GRAD_BUFFER_NUM = 2;
constexpr uint32_t FQ_GRAD_FP32_SIGN_MASK = 0x80000000u;

template <uint64_t SchMode>
class FakeQuantWithMinMaxArgsGradientRegbase {
public:
    __aicore__ inline FakeQuantWithMinMaxArgsGradientRegbase(
        const FakeQuantWithMinMaxArgsGradientTilingData* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t dataCount);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t dataCount);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, FQ_GRAD_BUFFER_NUM> inQueueG_;
    TQue<QuePosition::VECIN, FQ_GRAD_BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, FQ_GRAD_BUFFER_NUM> outQueueY_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;

    const FakeQuantWithMinMaxArgsGradientTilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
};

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxArgsGradientRegbase<SchMode>::Init(GM_ADDR gradients, GM_ADDR x, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradients));
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    int64_t blockFactor = tilingData_->blockFactor;
    int64_t blockTail = tilingData_->blockTailFactor;
    int64_t numCore = tilingData_->numCore;

    if (blockIdx_ >= numCore) {
        blockOffset_ = 0;
        blockLen_ = 0;
    } else if (blockIdx_ == numCore - 1) {
        blockOffset_ = blockIdx_ * blockFactor;
        blockLen_ = blockTail;
    } else {
        blockOffset_ = blockIdx_ * blockFactor;
        blockLen_ = blockFactor;
    }

    int64_t baseLen = tilingData_->baseLen;
    pipe_.InitBuffer(inQueueG_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));
    pipe_.InitBuffer(inQueueX_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));
    pipe_.InitBuffer(outQueueY_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxArgsGradientRegbase<SchMode>::Process()
{
    if (blockLen_ <= 0) {
        return;
    }
    int64_t baseLen = tilingData_->baseLen;
    int64_t loopNum = blockLen_ / baseLen;
    int64_t loopTail = blockLen_ % baseLen;
    int64_t base = blockOffset_;

    for (int64_t i = 0; i < loopNum; ++i) {
        int64_t off = base + i * baseLen;
        CopyIn(off, baseLen);
        Compute(baseLen);
        CopyOut(off, baseLen);
    }
    if (loopTail > 0) {
        int64_t off = base + loopNum * baseLen;
        CopyIn(off, loopTail);
        Compute(loopTail);
        CopyOut(off, loopTail);
    }
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxArgsGradientRegbase<SchMode>::CopyIn(int64_t gmOffset, int64_t dataCount)
{
    LocalTensor<float> gLocal = inQueueG_.AllocTensor<float>();
    LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(dataCount * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    params.rsv = 0;
    DataCopyPadExtParams<float> padParams = {false, 0, 0, 0};
    DataCopyPad(gLocal, gGm_[gmOffset], params, padParams);
    DataCopyPad(xLocal, xGm_[gmOffset], params, padParams);
    inQueueG_.EnQue(gLocal);
    inQueueX_.EnQue(xLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxArgsGradientRegbase<SchMode>::CopyOut(int64_t gmOffset, int64_t dataCount)
{
    LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(dataCount * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    params.rsv = 0;
    DataCopyPad(yGm_[gmOffset], yLocal, params);
    outQueueY_.FreeTensor(yLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxArgsGradientRegbase<SchMode>::Compute(int64_t dataCount)
{
    LocalTensor<float> gLocal = inQueueG_.DeQue<float>();
    LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();

    __local_mem__ float* gAddr = (__local_mem__ float*)gLocal.GetPhyAddr();
    __local_mem__ float* xAddr = (__local_mem__ float*)xLocal.GetPhyAddr();
    __local_mem__ float* yAddr = (__local_mem__ float*)yLocal.GetPhyAddr();

    const uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint32_t vfLoopNum = static_cast<uint32_t>((dataCount + VL - 1) / VL);

    const float nudgedMin = tilingData_->nudgedMin;
    const float nudgedMax = tilingData_->nudgedMax;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vG;
        AscendC::Reg::RegTensor<float> vX;
        AscendC::Reg::RegTensor<float> vZero;
        AscendC::Reg::RegTensor<float> vOne;
        AscendC::Reg::RegTensor<float> vMask;
        AscendC::Reg::RegTensor<float> vOut;
        AscendC::Reg::RegTensor<uint32_t> vSignMask;
        AscendC::Reg::RegTensor<uint32_t> vSignG;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg geMask;
        AscendC::Reg::MaskReg leMask;

        AscendC::Reg::Duplicate(vZero, 0.0f);
        AscendC::Reg::Duplicate(vOne, 1.0f);
        AscendC::Reg::Duplicate(vSignMask, FQ_GRAD_FP32_SIGN_MASK);

        uint32_t count = static_cast<uint32_t>(dataCount);
        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(count);

            // Step 0: load grad / x.
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vG, gAddr + i * VL);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vX, xAddr + i * VL);

            // Step 1: build float 0/1 mask = (x>=nudgedMin) && (x<=nudgedMax).
            //         NaN-safe: NaN compares false on both >= and <=, => mask=0.
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(geMask, vX, nudgedMin, mask);
            AscendC::Reg::Select<float>(vMask, vOne, vZero, geMask);
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(leMask, vX, nudgedMax, mask);
            AscendC::Reg::Select<float>(vMask, vMask, vZero, leMask);

            // Step 2: multiplicative gating y = grad * mask  (NaN grad propagates via NaN*0=NaN).
            AscendC::Reg::Mul(vOut, vG, vMask, mask);

            // Step 2.5: ±0 sign-bit re-injection (Ascend Mul may drop sign of -0).
            // vSignG = vG_u32 & 0x80000000;  vOut_u32 = vOut_u32 | vSignG.
            AscendC::Reg::And(vSignG, (AscendC::Reg::RegTensor<uint32_t>&)vG, vSignMask, mask);
            AscendC::Reg::Or((AscendC::Reg::RegTensor<uint32_t>&)vOut, (AscendC::Reg::RegTensor<uint32_t>&)vOut, vSignG,
                             mask);

            // Step 3: store.
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + i * VL, vOut, mask);
        }
    }

    inQueueG_.FreeTensor(gLocal);
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(yLocal);
}

} // namespace FakeQuantWithMinMaxArgsGradient

#endif // FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_REGBASE_H_
