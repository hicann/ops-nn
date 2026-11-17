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
 * \file copy_z_dz.h
 * \brief ChunkLoss arch35 Z / dZ / dH LocalTensor 拷贝助手
 *
 * 封装 GM↔UB 的 DataCopy / DataCopyPad 分块读写。
 */
#ifndef CHUNKLOSS_ARCH35_COPY_Z_DZ_H_
#define CHUNKLOSS_ARCH35_COPY_Z_DZ_H_

#include "../chunk_loss_common_arch35.h"

namespace chunkloss {

template <typename Element>
CHUNKLOSS_DEVICE void StoreDzTile(AscendC::GlobalTensor<Element> gmZ, AscendC::LocalTensor<Element>& ub, uint64_t zBase,
                                  uint32_t rowBegin, uint32_t rows, uint32_t n0, uint32_t actualN, uint32_t v,
                                  uint32_t srcCol = 0, uint32_t srcStride = kTileN)
{
    AscendC::GlobalTensor<Element> dst = gmZ[zBase + static_cast<uint64_t>(rowBegin) * v + n0];
    if (srcCol == 0u && srcStride == kTileN && actualN == kTileN && v == kTileN) {
        AscendC::DataCopy(dst, ub, rows * kTileN);
        return;
    }
    bool srcAligned = srcCol == 0u && srcStride == kTileN && ((kTileN - actualN) * sizeof(Element)) % 32u == 0;
    if (srcAligned) {
        uint32_t srcGapBlocks = ((kTileN - actualN) * sizeof(Element)) / 32u;
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(actualN * sizeof(Element)),
                                          srcGapBlocks, static_cast<uint32_t>((v - actualN) * sizeof(Element)), 0};
        AscendC::DataCopyPad(dst, ub, params);
        return;
    }
    const uint32_t bytes = actualN * static_cast<uint32_t>(sizeof(Element));
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::DataCopyPad(dst[static_cast<uint64_t>(r) * v], ub[r * srcStride + srcCol], {1, bytes, 0, 0, 0});
    }
}

/** Load fp32 Z checkpoint into logits UB (pad tail cols with kPadLogit). */
CHUNKLOSS_DEVICE void LoadZAsLogits(AscendC::LocalTensor<float>& logits, AscendC::GlobalTensor<float> gmZ,
                                    uint64_t zBase, uint32_t rowBegin, uint32_t rows, uint32_t n0, uint32_t actualN,
                                    uint32_t v)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::GlobalTensor<float> src = gmZ[zBase + static_cast<uint64_t>(rowBegin) * v + n0];
    if (actualN == kTileN) {
        if (v == kTileN) {
            AscendC::DataCopy(logits, src, rows * kTileN);
        } else {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows),
                                              static_cast<uint32_t>(kTileN * sizeof(float)),
                                              static_cast<uint32_t>((v - kTileN) * sizeof(float)), 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(logits, src, params, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        return;
    }
    AscendC::Duplicate(logits, kPadLogit, rows * kTileN);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    bool dstAligned = ((kTileN - actualN) * sizeof(float)) % 32u == 0;
    if (dstAligned) {
        uint32_t dstGapBlocks = ((kTileN - actualN) * sizeof(float)) / 32u;
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(actualN * sizeof(float)),
                                          static_cast<uint32_t>((v - actualN) * sizeof(float)), dstGapBlocks, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(logits, src, params, pad);
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::DataCopyPad(logits[r * kTileN], src[static_cast<uint64_t>(r) * v],
                                 {1, static_cast<uint32_t>(actualN * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0.0f});
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
}

/** Copy Pass1 fp32 GM logits into UB (pad tail cols with kPadLogit). */
CHUNKLOSS_DEVICE void CopyLogits(AscendC::LocalTensor<float>& logits, AscendC::GlobalTensor<float> gmRows,
                                 uint32_t rows, uint32_t actualN)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    if (actualN == kTileN) {
        AscendC::DataCopy(logits, gmRows, rows * kTileN);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        return;
    }
    AscendC::Duplicate(logits, kPadLogit, rows * kTileN);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::DataCopyPad(logits[r * kTileN], gmRows[r * kTileN],
                             {1, static_cast<uint32_t>(actualN * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0.0f});
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
}

CHUNKLOSS_DEVICE void LoadDhTile(AscendC::LocalTensor<float>& ub, AscendC::GlobalTensor<float> gmDhAcc, uint64_t dhBase,
                                 uint32_t rowBegin, uint32_t rows, uint32_t h0, uint32_t actualH, uint32_t h)
{
    AscendC::GlobalTensor<float> src = gmDhAcc[dhBase + static_cast<uint64_t>(rowBegin) * h + h0];
    if (actualH == kTileH && h == kTileH) {
        AscendC::DataCopy(ub, src, rows * kTileH);
        return;
    }
    uint32_t dstGapBlocks = ((kTileH - actualH) * sizeof(float)) / 32u;
    AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(actualH * sizeof(float)),
                                      static_cast<uint32_t>((h - actualH) * sizeof(float)), dstGapBlocks, 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(ub, src, params, pad);
}

CHUNKLOSS_DEVICE void StoreDhTile(AscendC::GlobalTensor<float> gmDhAcc, AscendC::LocalTensor<float>& ub,
                                  uint64_t dhBase, uint32_t rowBegin, uint32_t rows, uint32_t h0, uint32_t actualH,
                                  uint32_t h)
{
    AscendC::GlobalTensor<float> dst = gmDhAcc[dhBase + static_cast<uint64_t>(rowBegin) * h + h0];
    if (actualH == kTileH && h == kTileH) {
        AscendC::DataCopy(dst, ub, rows * kTileH);
        return;
    }
    uint32_t srcGapBlocks = ((kTileH - actualH) * sizeof(float)) / 32u;
    AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(actualH * sizeof(float)),
                                      srcGapBlocks, static_cast<uint32_t>((h - actualH) * sizeof(float)), 0};
    AscendC::DataCopyPad(dst, ub, params);
}

template <typename Element>
CHUNKLOSS_DEVICE void StoreDhElem(AscendC::GlobalTensor<Element> gmDH, AscendC::LocalTensor<Element>& ub,
                                  uint64_t gmOffset, uint32_t rows, uint32_t actualH, uint32_t h)
{
    AscendC::GlobalTensor<Element> dst = gmDH[gmOffset];
    if (actualH == kTileH && h == kTileH) {
        AscendC::DataCopy(dst, ub, rows * kTileH);
        return;
    }
    bool srcAligned = ((kTileH - actualH) * sizeof(Element)) % 32u == 0;
    if (srcAligned) {
        uint32_t srcGapBlocks = ((kTileH - actualH) * sizeof(Element)) / 32u;
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(actualH * sizeof(Element)),
                                          srcGapBlocks, static_cast<uint32_t>((h - actualH) * sizeof(Element)), 0};
        AscendC::DataCopyPad(dst, ub, params);
        return;
    }
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::DataCopyPad(dst[static_cast<uint64_t>(r) * h], ub[r * kTileH],
                             {1, static_cast<uint32_t>(actualH * sizeof(Element)), 0, 0, 0});
    }
}

} // namespace chunkloss

#endif
