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
 * \file vf_build_dz.h
 * \brief ChunkLoss arch35 BuildDz（softmax − one_hot）
 *
 * Reg VF 完成 Sub/Exp，再按 dzScale 与标签写出 dZ。
 */
#ifndef CHUNKLOSS_ARCH35_VF_BUILD_DZ_H_
#define CHUNKLOSS_ARCH35_VF_BUILD_DZ_H_

#include "../chunk_loss_common_arch35.h"

namespace chunkloss {

__simd_vf__ inline void VfSubMaxExp(__ubuf__ float* logits, __ubuf__ float* brcbMax, uint16_t rows)
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

template <typename Element>
CHUNKLOSS_DEVICE void BuildDz(AscendC::LocalTensor<float>& logits, AscendC::LocalTensor<Element>& dzUb,
                              AscendC::LocalTensor<float>& preMax, AscendC::LocalTensor<float>& preSum,
                              AscendC::LocalTensor<float>& brcbMax, AscendC::LocalTensor<float>& /*keepMax*/,
                              bool const* valid, int64_t const* lab, uint32_t rows, uint32_t actualN, uint32_t n0,
                              float dzScale, uint32_t padLeft = 0, bool applyOneHot = true)
{
    // dZ = dzScale * softmax；applyOneHot 时再减 one_hot。
    // Vector 路径通常 dzScale=1；Cube 大 V 路径可用更大 dzScale，并将标签项留到 float dhAcc。
    // Buffer 覆盖 [n0, n0+actualN)；one-hot 归属区间为 [n0+padLeft, n0+actualN)。
    AscendC::Brcb(brcbMax, preMax, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    __ubuf__ float* logitsPtr = (__ubuf__ float*)logits.GetPhyAddr();
    __ubuf__ float* brcbPtr = (__ubuf__ float*)brcbMax.GetPhyAddr();
    VfSubMaxExp(logitsPtr, brcbPtr, static_cast<uint16_t>(rows));
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    for (uint32_t r = 0; r < rows; ++r) {
        float s = preSum.GetValue(r);
        float scale = (valid[r] && s > 0.0f) ? (dzScale / s) : 0.0f;
        AscendC::Muls(logits[r * kTileN], logits[r * kTileN], scale, kTileN);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
    const int64_t own0 = static_cast<int64_t>(n0 + padLeft);
    const int64_t own1 = static_cast<int64_t>(n0 + actualN);
    for (uint32_t r = 0; r < rows; ++r) {
        if (!valid[r]) {
            AscendC::Duplicate(logits[r * kTileN], 0.0f, kTileN);
            continue;
        }
        if (actualN < kTileN) {
            uint32_t pad = kTileN - actualN;
            uint32_t aligned = (pad * static_cast<uint32_t>(sizeof(float)) / 32u) * 32u /
                               static_cast<uint32_t>(sizeof(float));
            if (aligned > 0u) {
                AscendC::Duplicate(logits[r * kTileN + actualN], 0.0f, aligned);
            }
            for (uint32_t c = actualN + aligned; c < kTileN; ++c) {
                logits.SetValue(r * kTileN + c, 0.0f);
            }
        }
        for (uint32_t c = 0; c < padLeft; ++c) {
            logits.SetValue(r * kTileN + c, 0.0f);
        }
        if (applyOneHot) {
            int64_t y = lab[r];
            if (y >= own0 && y < own1) {
                uint32_t col = static_cast<uint32_t>(y - static_cast<int64_t>(n0));
                logits.SetValue(r * kTileN + col, logits.GetValue(r * kTileN + col) - dzScale);
            }
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::Cast(dzUb, logits, AscendC::RoundMode::CAST_ROUND, rows * kTileN);
}

} // namespace chunkloss

#endif
