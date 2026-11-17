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
 * \file vf_online_lse.h
 * \brief ChunkLoss arch35 online LSE（max / sum）
 *
 * Reg VF 完成 Sub+Exp；reduce 使用 LocalTensor 接口。
 */
#ifndef CHUNKLOSS_ARCH35_VF_ONLINE_LSE_H_
#define CHUNKLOSS_ARCH35_VF_ONLINE_LSE_H_

#include "../chunk_loss_common_arch35.h"

namespace chunkloss {

__simd_vf__ inline void VfSubBrcbExp(__ubuf__ float* logits, __ubuf__ float* brcbMax, uint16_t rows)
{
    using namespace AscendC::Reg;
    RegTensor<float> z0, z1, z2, z3, m;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t r = 0; r < rows; ++r) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(m, brcbMax + r * 8);
        DataCopy<float, LoadDist::DIST_NORM>(z0, logits + r * kTileN);
        DataCopy<float, LoadDist::DIST_NORM>(z1, logits + r * kTileN + 64);
        DataCopy<float, LoadDist::DIST_NORM>(z2, logits + r * kTileN + 128);
        DataCopy<float, LoadDist::DIST_NORM>(z3, logits + r * kTileN + 192);
        Sub(z0, z0, m, all);
        Sub(z1, z1, m, all);
        Sub(z2, z2, m, all);
        Sub(z3, z3, m, all);
        Exp(z0, z0, all);
        Exp(z1, z1, all);
        Exp(z2, z2, all);
        Exp(z3, z3, all);
        DataCopy<float, StoreDist::DIST_NORM>(logits + r * kTileN, z0, all);
        DataCopy<float, StoreDist::DIST_NORM>(logits + r * kTileN + 64, z1, all);
        DataCopy<float, StoreDist::DIST_NORM>(logits + r * kTileN + 128, z2, all);
        DataCopy<float, StoreDist::DIST_NORM>(logits + r * kTileN + 192, z3, all);
    }
}

CHUNKLOSS_DEVICE void OnlineMax(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<float>& maxTensor,
                                AscendC::LocalTensor<float>& preMax, uint32_t rows)
{
    int32_t repeatTimes = static_cast<int32_t>(rows * kTileN / kDataPerRepeat);
    if (repeatTimes > kMaxReduceRepeat) {
        int32_t first = (repeatTimes / 2 + 7) / 8 * 8;
        int32_t second = repeatTimes - first;
        AscendC::WholeReduceMax(maxTensor, logits, static_cast<uint64_t>(kDataPerRepeat), first, 1, 1, 8);
        AscendC::WholeReduceMax(maxTensor[first * 2], logits[first * kDataPerRepeat],
                                static_cast<uint64_t>(kDataPerRepeat), second, 1, 1, 8);
    } else {
        AscendC::WholeReduceMax(maxTensor, logits, static_cast<uint64_t>(kDataPerRepeat), repeatTimes, 1, 1, 8);
    }
    AscendC::PipeBarrier<PIPE_V>();
    uint8_t dupRepeat = static_cast<uint8_t>((rows * 8 + kDataPerRepeat - 1) / kDataPerRepeat);
    uint64_t dupMask[2] = {0b1010101010101010101010101010101010101010101010101010101010101010u, 0};
    AscendC::Duplicate(maxTensor.template ReinterpretCast<int32_t>(), kNegInfBits, dupMask, dupRepeat, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax(maxTensor, maxTensor, dupRepeat, static_cast<uint64_t>(kDataPerRepeat), 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(maxTensor, maxTensor, preMax, rows);
}

CHUNKLOSS_DEVICE void OnlineSum(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<float>& sumTensor,
                                AscendC::LocalTensor<float>& brcbMax, AscendC::LocalTensor<float>& preMax,
                                AscendC::LocalTensor<float>& preSum, AscendC::LocalTensor<float>& maxTensor,
                                uint32_t rows)
{
    uint8_t brcbRepeat = static_cast<uint8_t>((rows + 7) / 8);
    AscendC::Brcb(brcbMax, maxTensor, brcbRepeat, {1, 8});
    AscendC::Sub(preMax, preMax, maxTensor, rows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(preMax, preMax, rows);
    AscendC::Mul(preSum, preSum, preMax, rows);
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ float* logitsPtr = (__ubuf__ float*)logits.GetPhyAddr();
    __ubuf__ float* brcbPtr = (__ubuf__ float*)brcbMax.GetPhyAddr();
    VfSubBrcbExp(logitsPtr, brcbPtr, static_cast<uint16_t>(rows));
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Duplicate(sumTensor, 0.0f, rows * 8);
    AscendC::PipeBarrier<PIPE_V>();
    int32_t halfRepeat = static_cast<int32_t>(rows * 2);
    AscendC::WholeReduceSum(sumTensor, logits, static_cast<uint64_t>(kDataPerRepeat), halfRepeat, 2, 1, 8);
    AscendC::WholeReduceSum(sumTensor[halfRepeat * 2], logits[halfRepeat * kDataPerRepeat],
                            static_cast<uint64_t>(kDataPerRepeat), halfRepeat, 2, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(sumTensor, sumTensor, static_cast<uint64_t>(8), static_cast<int32_t>(rows), 1, 1, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(sumTensor, preSum, sumTensor, rows);
}

CHUNKLOSS_DEVICE void UpdateOddRowAt(AscendC::LocalTensor<float>& logits, uint32_t row, uint32_t actualN, float& oddM,
                                     float& oddS, AscendC::LocalTensor<float>& scratch)
{
    float tileMax = kPadLogit;
    for (uint32_t c = 0; c < actualN; ++c) {
        float z = logits.GetValue(row * kTileN + c);
        if (z > tileMax) {
            tileMax = z;
        }
    }
    float newMax = oddM > tileMax ? oddM : tileMax;
    float tileSum = 0.0f;
    for (uint32_t c = 0; c < actualN; c += 8) {
        uint32_t nElt = actualN - c > 8 ? 8 : actualN - c;
        for (uint32_t i = 0; i < 8; ++i) {
            float z = i < nElt ? logits.GetValue(row * kTileN + c + i) : newMax - 80.0f;
            scratch.SetValue(i, z - newMax);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::Exp(scratch, scratch, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        for (uint32_t i = 0; i < nElt; ++i) {
            tileSum += scratch.GetValue(i);
        }
    }
    float scale = 0.0f;
    if (oddS != 0.0f) {
        scratch.SetValue(0, oddM - newMax);
        for (uint32_t i = 1; i < 8; ++i) {
            scratch.SetValue(i, -80.0f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::Exp(scratch, scratch, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
        scale = scratch.GetValue(0);
    }
    oddS = oddS * scale + tileSum;
    oddM = newMax;
}

} // namespace chunkloss

#endif
