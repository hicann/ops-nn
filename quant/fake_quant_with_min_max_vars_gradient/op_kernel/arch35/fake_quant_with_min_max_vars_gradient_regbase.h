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
 * \file fake_quant_with_min_max_vars_gradient_regbase.h
 * \brief fake_quant_with_min_max_vars_gradient kernel (regbase / arch35)
 *
 * Design:
 *  - min/max tensor inputs read via GetValue in Init
 *  - nudgedMin/nudgedMax computed on Device side (scalar, outside VF)
 *  - bothZero dispatched outside VF to ComputePassthrough / ComputeNormal
 *  - Vector compare+select for below_min/above_max (CMPMODE::GE/LE inversion)
 *  - Reg::Add accumulator + Reg::ReduceSum for vector reduce within VF
 *  - Multi-core reduce: SetAtomicAdd<float> + DataCopyPad direct to output GM
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_REGBASE_H_
#define FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_REGBASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "fake_quant_with_min_max_vars_gradient_tilingdata.h"

namespace FakeQuantWithMinMaxVarsGradient {
using namespace AscendC;

constexpr uint32_t FQ_GRAD_BUFFER_NUM = 2;
constexpr uint32_t FQ_GRAD_FP32_SIGN_MASK = 0x80000000u;

template <uint64_t SchMode>
class FakeQuantWithMinMaxVarsGradientRegbase {
public:
    __aicore__ inline FakeQuantWithMinMaxVarsGradientRegbase(
        const FakeQuantWithMinMaxVarsGradientTilingData* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR x, GM_ADDR minGm, GM_ADDR maxGm, GM_ADDR backpropsWrtX,
                                GM_ADDR backpropWrtMin, GM_ADDR backpropWrtMax);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t dataCount);
    __aicore__ inline void ComputePassthrough(int64_t dataCount);
    __aicore__ inline void ComputeNormal(int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t dataCount);
    __aicore__ inline void AtomicWriteReduce();

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, FQ_GRAD_BUFFER_NUM> inQueueG_;
    TQue<QuePosition::VECIN, FQ_GRAD_BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, FQ_GRAD_BUFFER_NUM> outQueueY_;
    TQue<QuePosition::VECOUT, FQ_GRAD_BUFFER_NUM> minQueue_;
    TQue<QuePosition::VECOUT, FQ_GRAD_BUFFER_NUM> maxQueue_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> minOutGm_;
    GlobalTensor<float> maxOutGm_;
    GlobalTensor<float> wsGm_;

    const FakeQuantWithMinMaxVarsGradientTilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;

    bool bothZero_ = false;
    float nudgedMin_ = 0.0f;
    float nudgedMax_ = 0.0f;
};

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::Init(GM_ADDR gradients, GM_ADDR x,
                                                                             GM_ADDR minGm, GM_ADDR maxGm,
                                                                             GM_ADDR backpropsWrtX,
                                                                             GM_ADDR backpropWrtMin,
                                                                             GM_ADDR backpropWrtMax)
{
    blockIdx_ = GetBlockIdx();

    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradients));
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(backpropsWrtX));
    minOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(backpropWrtMin));
    maxOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(backpropWrtMax));

    if (GetBlockIdx() == 0) {
        InitGlobalMemory<float>(maxOutGm_, 1, static_cast<float>(0));
        InitGlobalMemory<float>(minOutGm_, 1, static_cast<float>(0));
    }

    GlobalTensor<float> minTensor, maxTensor;
    minTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(minGm));
    maxTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(maxGm));
    float fMin = minTensor.GetValue(0);
    float fMax = maxTensor.GetValue(0);

    int64_t numBits = tilingData_->numBits;
    bool narrowRange = tilingData_->narrowRange;

    float quantMin = narrowRange ? 1.0f : 0.0f;
    float quantMax = static_cast<float>((1ULL << static_cast<uint32_t>(numBits)) - 1ULL);
    float scale = (fMax - fMin) / (quantMax - quantMin);
    float zpFromMin = quantMin - fMin / scale;

    float nudgedZp;
    if (zpFromMin <= quantMin) {
        nudgedZp = quantMin;
    } else if (zpFromMin >= quantMax) {
        nudgedZp = quantMax;
    } else {
        float val = zpFromMin + 0.5f;
        int intPart = (int)val;
        if (val < 0.0f && val != (float)intPart) {
            intPart -= 1;
        }
        nudgedZp = (float)intPart;
    }

    nudgedMin_ = (quantMin - nudgedZp) * scale;
    nudgedMax_ = (quantMax - nudgedZp) * scale;

    float absMin = (fMin >= 0.0f) ? fMin : -fMin;
    float absMax = (fMax >= 0.0f) ? fMax : -fMax;
    bothZero_ = (absMin + absMax) < 1e-10f;

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

    // UB budget: 5 queues (g, x, y, min, max) × 2 buffers = 10 slots
    int64_t baseLen = tilingData_->baseLen;
    pipe_.InitBuffer(inQueueG_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));
    pipe_.InitBuffer(inQueueX_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));
    pipe_.InitBuffer(outQueueY_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));
    pipe_.InitBuffer(minQueue_, FQ_GRAD_BUFFER_NUM, 32);
    pipe_.InitBuffer(maxQueue_, FQ_GRAD_BUFFER_NUM, 32);

    SyncAll();
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::Process()
{
    int64_t totalLen = tilingData_->totalLen;

    if (totalLen == 0) {
        return;
    }

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
        if (bothZero_) {
            ComputePassthrough(baseLen);
        } else {
            ComputeNormal(baseLen);
            AtomicWriteReduce();
        }
        CopyOut(off, baseLen);
    }
    if (loopTail > 0) {
        int64_t off = base + loopNum * baseLen;
        CopyIn(off, loopTail);
        if (bothZero_) {
            ComputePassthrough(loopTail);
        } else {
            ComputeNormal(loopTail);
            AtomicWriteReduce();
        }
        CopyOut(off, loopTail);
    }
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::CopyIn(int64_t gmOffset, int64_t dataCount)
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
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::CopyOut(int64_t gmOffset, int64_t dataCount)
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
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::ComputePassthrough(int64_t dataCount)
{
    LocalTensor<float> gLocal = inQueueG_.DeQue<float>();
    LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();

    __ubuf__ float* gAddr = (__ubuf__ float*)gLocal.GetPhyAddr();
    __ubuf__ float* yAddr = (__ubuf__ float*)yLocal.GetPhyAddr();

    const uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint32_t vfLoopNum = static_cast<uint32_t>((dataCount + VL - 1) / VL);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vG;
        AscendC::Reg::MaskReg mask;

        uint32_t count = static_cast<uint32_t>(dataCount);
        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vG, gAddr + i * VL);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + i * VL, vG, mask);
        }
    }

    outQueueY_.EnQue(yLocal);
    inQueueG_.FreeTensor(gLocal);
    inQueueX_.FreeTensor(xLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::ComputeNormal(int64_t dataCount)
{
    LocalTensor<float> gLocal = inQueueG_.DeQue<float>();
    LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
    LocalTensor<float> minLocal = minQueue_.AllocTensor<float>();
    LocalTensor<float> maxLocal = maxQueue_.AllocTensor<float>();

    __ubuf__ float* gAddr = (__ubuf__ float*)gLocal.GetPhyAddr();
    __ubuf__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ float* yAddr = (__ubuf__ float*)yLocal.GetPhyAddr();
    __ubuf__ float* minAddr = (__ubuf__ float*)minLocal.GetPhyAddr();
    __ubuf__ float* maxAddr = (__ubuf__ float*)maxLocal.GetPhyAddr();

    const uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint32_t vfLoopNum = static_cast<uint32_t>((dataCount + VL - 1) / VL);
    const float nudgedMin = nudgedMin_;
    const float nudgedMax = nudgedMax_;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vG;
        AscendC::Reg::RegTensor<float> vX;
        AscendC::Reg::RegTensor<float> vZero;
        AscendC::Reg::RegTensor<float> vOne;
        AscendC::Reg::RegTensor<float> vMask;
        AscendC::Reg::RegTensor<float> vOut;
        AscendC::Reg::RegTensor<float> vBelow;
        AscendC::Reg::RegTensor<float> vAbove;
        AscendC::Reg::RegTensor<uint32_t> vSignMask;
        AscendC::Reg::RegTensor<uint32_t> vSignG;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg geMask;
        AscendC::Reg::MaskReg leMask;
        AscendC::Reg::RegTensor<float> vAccMin;
        AscendC::Reg::RegTensor<float> vAccMax;

        AscendC::Reg::Duplicate(vZero, 0.0f);
        AscendC::Reg::Duplicate(vOne, 1.0f);
        AscendC::Reg::Duplicate(vSignMask, FQ_GRAD_FP32_SIGN_MASK);
        AscendC::Reg::Duplicate(vAccMin, 0.0f);
        AscendC::Reg::Duplicate(vAccMax, 0.0f);

        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        uint32_t count = static_cast<uint32_t>(dataCount);
        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(count);

            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vG, gAddr + i * VL);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vX, xAddr + i * VL);

            // mask = (x >= nudgedMin) && (x <= nudgedMax)
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(geMask, vX, nudgedMin, mask);
            AscendC::Reg::Select<float>(vMask, vOne, vZero, geMask);
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(leMask, vX, nudgedMax, mask);
            AscendC::Reg::Select<float>(vMask, vMask, vZero, leMask);

            // backprops_wrt_x = gradients * mask + sign-bit re-injection
            AscendC::Reg::Mul(vOut, vG, vMask, mask);
            AscendC::Reg::And(vSignG, (AscendC::Reg::RegTensor<uint32_t>&)vG, vSignMask, mask);
            AscendC::Reg::Or((AscendC::Reg::RegTensor<uint32_t>&)vOut, (AscendC::Reg::RegTensor<uint32_t>&)vOut, vSignG,
                             mask);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + i * VL, vOut, mask);

            // below_min: x < nudgedMin → Select(vG, vZero, ltMask)
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(geMask, vX, nudgedMin, mask);
            AscendC::Reg::Select<float>(vBelow, vG, vZero, geMask);

            // above_max: x > nudgedMax → Select(vG, vZero, gtMask)
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GT>(leMask, vX, nudgedMax, mask);
            AscendC::Reg::Select<float>(vAbove, vG, vZero, leMask);

            // Vector accumulate across all vf iterations (use ALL mask)
            AscendC::Reg::Add(vAccMin, vAccMin, vBelow, preg1);
            AscendC::Reg::Add(vAccMax, vAccMax, vAbove, preg1);
        }

        // ReduceSum: VL lanes → 1 scalar in lane 0
        AscendC::Reg::RegTensor<float> vSumMin;
        AscendC::Reg::RegTensor<float> vSumMax;
        AscendC::Reg::ReduceSum<float>(vSumMin, vAccMin, preg1);
        AscendC::Reg::ReduceSum<float>(vSumMax, vAccMax, preg1);

        // Write reduced scalars to minLocal / maxLocal UB
        AscendC::Reg::MaskReg vl1Mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(minAddr, vSumMin, vl1Mask);
        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(maxAddr, vSumMax, vl1Mask);
    }

    outQueueY_.EnQue(yLocal);
    minQueue_.EnQue(minLocal);
    maxQueue_.EnQue(maxLocal);
    inQueueG_.FreeTensor(gLocal);
    inQueueX_.FreeTensor(xLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::AtomicWriteReduce()
{
    LocalTensor<float> minLocal = minQueue_.DeQue<float>();
    LocalTensor<float> maxLocal = maxQueue_.DeQue<float>();

    DataCopyExtParams pOne = {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};

    SetAtomicAdd<float>();
    DataCopyPad(minOutGm_[0], minLocal, pOne);
    DataCopyPad(maxOutGm_[0], maxLocal, pOne);
    SetAtomicNone();

    minQueue_.FreeTensor(minLocal);
    maxQueue_.FreeTensor(maxLocal);
}

} // namespace FakeQuantWithMinMaxVarsGradient

#endif // FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_REGBASE_H_
