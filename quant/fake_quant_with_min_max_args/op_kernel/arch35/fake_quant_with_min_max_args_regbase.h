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
 * \file fake_quant_with_min_max_args_regbase.h
 * \brief fake_quant_with_min_max_args kernel (regbase / arch35)
 *
 * Self-check (per design §5):
 *  1) round = floor(scaled + roundBias), roundBias = 0.5 - quantZero
 *  2) scheme A: shifted=clamped+(-nudgedMin); scaled=shifted*scaleInv (no merge)
 *  3) dequant y = q_offset * scale (no large-add)
 *  4) NaN: x!=x mask + final Select(vX,vY,nanMask) passthrough
 *  6) clamp via >/< + Select (NaN-safe), not Min/Max
 *  (gradient-only items 5/11 N/A; host-side 1/7-10 see tiling)
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_ARGS_REGBASE_H_
#define FAKE_QUANT_WITH_MIN_MAX_ARGS_REGBASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "fake_quant_with_min_max_args_tilingdata.h"

namespace FakeQuantWithMinMaxArgs {
using namespace AscendC;

constexpr uint32_t FAKEQUANT_BUFFER_NUM = 2;

// CastTrait must be defined at namespace/file scope so it can bind to the
// Cast<T, U, const CastTrait& trait> template parameter (a const reference).
constexpr static AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_S32_FLOOR = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
constexpr static AscendC::Reg::CastTrait CAST_TRAIT_S32_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <uint64_t SchMode>
class FakeQuantWithMinMaxArgsRegbase {
public:
    __aicore__ inline FakeQuantWithMinMaxArgsRegbase(const FakeQuantWithMinMaxArgsTilingData* tilingData)
        : tilingData_(tilingData) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t dataCount);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t dataCount);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, FAKEQUANT_BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, FAKEQUANT_BUFFER_NUM> outQueueY_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;

    const FakeQuantWithMinMaxArgsTilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
};

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxArgsRegbase<SchMode>::Init(GM_ADDR x, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
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
    pipe_.InitBuffer(inQueueX_, FAKEQUANT_BUFFER_NUM, baseLen * sizeof(float));
    pipe_.InitBuffer(outQueueY_, FAKEQUANT_BUFFER_NUM, baseLen * sizeof(float));
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxArgsRegbase<SchMode>::Process()
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
__aicore__ inline void FakeQuantWithMinMaxArgsRegbase<SchMode>::CopyIn(int64_t gmOffset, int64_t dataCount)
{
    LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(dataCount * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    params.rsv = 0;
    DataCopyPadExtParams<float> padParams = {false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[gmOffset], params, padParams);
    inQueueX_.EnQue(xLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxArgsRegbase<SchMode>::CopyOut(int64_t gmOffset, int64_t dataCount)
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
__aicore__ inline void FakeQuantWithMinMaxArgsRegbase<SchMode>::Compute(int64_t dataCount)
{
    LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();

    __local_mem__ float* xLocalAddr = (__local_mem__ float*)xLocal.GetPhyAddr();
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();

    const uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t vfLoopNum = (dataCount + VL - 1) / VL;

    // Reuse host pre-computed nudged scalars (tiling).
    const float nudgedMin = tilingData_->nudgedMin;
    const float nudgedMax = tilingData_->nudgedMax;
    const float scaleInv = tilingData_->scaleInv;
    const float scale = tilingData_->scale;
    const float negNudgedMin = -nudgedMin;
    const float roundBias = 0.5f - tilingData_->quantZero;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vX;
        AscendC::Reg::RegTensor<float> vClamped;
        AscendC::Reg::RegTensor<float> vNMin;
        AscendC::Reg::RegTensor<float> vNMax;
        AscendC::Reg::RegTensor<float> vScaled;
        AscendC::Reg::RegTensor<float> vQOffsetF;
        AscendC::Reg::RegTensor<int32_t> vQOffsetI;
        AscendC::Reg::RegTensor<float> vY;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg gtMask;
        AscendC::Reg::MaskReg ltMask;
        AscendC::Reg::MaskReg nanMask;

        // Broadcast nudged constants into reg tensors (for Select after compare).
        AscendC::Reg::Duplicate(vNMin, nudgedMin);
        AscendC::Reg::Duplicate(vNMax, nudgedMax);

        uint32_t count = static_cast<uint32_t>(dataCount);
        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            mask = AscendC::Reg::UpdateMask<float>(count);

            // Step 0: load x.
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vX, xLocalAddr + i * VL);

            // Step 1: NaN mask = (x != x). IEEE754: only NaN compares NE with itself.
            AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanMask, vX, vX, mask);

            // Step 2a: upper clamp -- (x > nudgedMax) ? nudgedMax : x  (NaN-safe via Select).
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GT>(gtMask, vX, nudgedMax, mask);
            AscendC::Reg::Select<float>(vClamped, vNMax, vX, gtMask);

            // Step 2b: lower clamp -- (clamped < nudgedMin) ? nudgedMin : clamped.
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(ltMask, vClamped, nudgedMin, mask);
            AscendC::Reg::Select<float>(vClamped, vNMin, vClamped, ltMask);

            // Step 3: scheme A -- shifted = clamped + (-nudgedMin); scaled = shifted * scaleInv.
            AscendC::Reg::Adds(vScaled, vClamped, negNudgedMin, mask);
            AscendC::Reg::Muls(vScaled, vScaled, scaleInv, mask);

            // Step 4: round = floor(scaled + roundBias) via fp32->int32 FLOOR cast.
            AscendC::Reg::Adds(vScaled, vScaled, roundBias, mask);
            AscendC::Reg::Cast<int32_t, float, CAST_TRAIT_FP32_TO_S32_FLOOR>(vQOffsetI, vScaled, mask);
            AscendC::Reg::Cast<float, int32_t, CAST_TRAIT_S32_TO_FP32>(vQOffsetF, vQOffsetI, mask);

            // Step 5: dequant y = q_offset * scale.
            AscendC::Reg::Muls(vY, vQOffsetF, scale, mask);

            // Step 6: NaN passthrough -- nanMask=1 -> vX, else vY.
            AscendC::Reg::Select<float>(vY, vX, vY, nanMask);

            // Step 7: store y.
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yLocalAddr + i * VL, vY, mask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(yLocal);
}

} // namespace FakeQuantWithMinMaxArgs

#endif // FAKE_QUANT_WITH_MIN_MAX_ARGS_REGBASE_H_
