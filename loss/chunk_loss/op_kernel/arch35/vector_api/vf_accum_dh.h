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
 * \file vf_accum_dh.h
 * \brief ChunkLoss arch35 Vector float 累加 dH
 *
 * 在 UB 上以 float 完成 dH += dZ @ W，并提供 UseVecDh 路径选择。
 */
#ifndef CHUNKLOSS_ARCH35_VF_ACCUM_DH_H_
#define CHUNKLOSS_ARCH35_VF_ACCUM_DH_H_

#include "../chunk_loss_common_arch35.h"
#include "copy_z_dz.h"

namespace chunkloss {

/** Zero a [rows, actualH] stripe of gmDhAcc (row stride = h, UB row stride = kTileH). */
CHUNKLOSS_DEVICE void ZeroDhTile(AscendC::GlobalTensor<float> gmDhAcc, AscendC::LocalTensor<float>& ub, uint64_t dhBase,
                                 uint32_t rowBegin, uint32_t rows, uint32_t h0, uint32_t actualH, uint32_t h)
{
    AscendC::Duplicate(ub, 0.0f, rows * kTileH);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    StoreDhTile(gmDhAcc, ub, dhBase, rowBegin, rows, h0, actualH, h);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

/**
 * dhUb[rows, kTileH] += dzFloat[rows, kTileN] @ W[n0Cube:n0Cube+nCube, h0:h0+actualH]
 * W is Element [V,H] row-major. padLeft cols of dz are zero (stub); safe to include.
 */
template <typename Element>
CHUNKLOSS_DEVICE void VecAccumDhTile(AscendC::LocalTensor<float>& dhUb, AscendC::LocalTensor<float>& dzFloat,
                                     AscendC::LocalTensor<Element>& wElem, AscendC::LocalTensor<float>& wFloat,
                                     AscendC::GlobalTensor<Element> gmW, uint32_t rows, uint32_t n0Cube, uint32_t nCube,
                                     uint32_t h0, uint32_t actualH, uint32_t fullH)
{
    for (uint32_t k = 0; k < nCube; ++k) {
        AscendC::GlobalTensor<Element> src = gmW[static_cast<uint64_t>(n0Cube + k) * fullH + h0];
        AscendC::DataCopyPad(wElem, src, {1, static_cast<uint32_t>(actualH * sizeof(Element)), 0, 0, 0},
                             {false, 0, 0, static_cast<Element>(0)});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast(wFloat, wElem, AscendC::RoundMode::CAST_NONE, actualH);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
        for (uint32_t r = 0; r < rows; ++r) {
            float scale = dzFloat.GetValue(r * kTileN + k);
            if (scale == 0.0f) {
                continue;
            }
            AscendC::Axpy(dhUb[r * kTileH], wFloat, scale, actualH);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
 * M*V 低于阈值且 nSplit==1 时走 Vector float dH；超大 shape 走 Cube Pass2。
 */
CHUNKLOSS_DEVICE bool UseVecDh(uint32_t m, uint32_t v, uint32_t nSplit)
{
    if (nSplit > 1u) {
        return false;
    }
    constexpr uint64_t kCubeOnlyMV = 2000000000ull;
    uint64_t mv = static_cast<uint64_t>(m) * static_cast<uint64_t>(v);
    return mv < kCubeOnlyMV;
}

/**
 * Cube path: dh was softmax@W (scaled); finish dH = dh*dhScale - alpha*W[y].
 * dhUb holds [rows, kTileH] float tile at h0; wElem/wFloat scratch for one W row.
 */
template <typename Element>
CHUNKLOSS_DEVICE void SubLabelWy(AscendC::LocalTensor<float>& dhUb, AscendC::LocalTensor<Element>& wElem,
                                 AscendC::LocalTensor<float>& wFloat, AscendC::GlobalTensor<Element> gmW,
                                 bool const* valid, int64_t const* lab, uint32_t rows, uint32_t h0, uint32_t actualH,
                                 uint32_t fullH, float invAlpha)
{
    for (uint32_t r = 0; r < rows; ++r) {
        if (!valid[r]) {
            continue;
        }
        int64_t y = lab[r];
        if (y < 0) {
            continue;
        }
        AscendC::GlobalTensor<Element> src = gmW[static_cast<uint64_t>(y) * fullH + h0];
        AscendC::DataCopyPad(wElem, src, {1, static_cast<uint32_t>(actualH * sizeof(Element)), 0, 0, 0},
                             {false, 0, 0, static_cast<Element>(0)});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast(wFloat, wElem, AscendC::RoundMode::CAST_NONE, actualH);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Axpy(dhUb[r * kTileH], wFloat, -invAlpha, actualH);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

} // namespace chunkloss

#endif
