/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * ChunkLoss arch22 pure AscendC Cube matmul (910B / Atlas A2).
 * Control flow & LoadData params aligned to Catlass PreloadZ / FullLoadA TileCopy,
 * without any #include of catlass.
 *
 * HandZ: A RowMajor→zN, B ColumnMajor→nZ, L0K=64 / L1K=256, unitFlag SoftSync.
 * HandH: A RowMajor FullLoadA, B RowMajor→zN→nZ (transpose).
 */
#ifndef CHUNK_LOSS_HAND_MMAD_ARCH22_H_
#define CHUNK_LOSS_HAND_MMAD_ARCH22_H_

#include <type_traits>

#include "chunk_loss_common_arch22.h"

namespace chunkloss {

namespace detail {
CHUNK_LOSS_HOST_DEVICE uint32_t RoundUp16(uint32_t x) { return (x + 15u) & ~15u; }
CHUNK_LOSS_HOST_DEVICE uint32_t CeilDivU(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
CHUNK_LOSS_HOST_DEVICE uint32_t MinU(uint32_t a, uint32_t b) { return a < b ? a : b; }
} // namespace detail

struct CubeBuf {
    static constexpr uint32_t STAGES = 2;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;

    // Mirror Catlass::Arch::Resource: InitBuffer via GetTPipePtr, then Destroy local TPipe
    // (alive TPipe injects sync that conflicts with SoftSync unitFlag).
    AscendC::LocalTensor<uint8_t> l1Bytes;
    AscendC::LocalTensor<uint8_t> l0aBytes;
    AscendC::LocalTensor<uint8_t> l0bBytes;
    AscendC::LocalTensor<uint8_t> l0cBytes;

    CHUNK_LOSS_DEVICE void Init()
    {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::A1> l1Buf;
        AscendC::TBuf<AscendC::TPosition::A2> l0aBuf;
        AscendC::TBuf<AscendC::TPosition::B2> l0bBuf;
        AscendC::TBuf<AscendC::TPosition::CO1> l0cBuf;
        GetTPipePtr()->InitBuffer(l1Buf, L1_SIZE);
        GetTPipePtr()->InitBuffer(l0aBuf, L0A_SIZE);
        GetTPipePtr()->InitBuffer(l0bBuf, L0B_SIZE);
        GetTPipePtr()->InitBuffer(l0cBuf, L0C_SIZE);
        l1Bytes = l1Buf.Get<uint8_t>();
        l0aBytes = l0aBuf.Get<uint8_t>();
        l0bBytes = l0bBuf.Get<uint8_t>();
        l0cBytes = l0cBuf.Get<uint8_t>();
        pipe.Destroy();
        AscendC::SetLoadDataPaddingValue<uint64_t>(0);
    }

    template <typename T>
    CHUNK_LOSS_DEVICE AscendC::LocalTensor<T> L1(uint32_t elems, uint32_t byteOff)
    {
        return l1Bytes[byteOff].template ReinterpretCast<T>();
    }
    template <typename T>
    CHUNK_LOSS_DEVICE AscendC::LocalTensor<T> L0A(uint32_t elems, uint32_t byteOff)
    {
        (void)elems;
        return l0aBytes[byteOff].template ReinterpretCast<T>();
    }
    template <typename T>
    CHUNK_LOSS_DEVICE AscendC::LocalTensor<T> L0B(uint32_t elems, uint32_t byteOff)
    {
        (void)elems;
        return l0bBytes[byteOff].template ReinterpretCast<T>();
    }
    template <typename T>
    CHUNK_LOSS_DEVICE AscendC::LocalTensor<T> L0C(uint32_t elems, uint32_t byteOff)
    {
        (void)elems;
        return l0cBytes[byteOff].template ReinterpretCast<T>();
    }
};

template <class Element, uint32_t TileM, uint32_t TileN = 256, uint32_t L1K = 256, uint32_t L0K = 64,
          bool ENABLE_UNIT_FLAG = true>
class HandMmadZ {
public:
    static constexpr uint32_t STAGES = CubeBuf::STAGES;
    static constexpr uint32_t C0 = 16;
    static constexpr uint32_t FRACTAL = 256;
    static constexpr uint32_t L1A_TILE = TileM * L1K;
    static constexpr uint32_t L1B_TILE = L1K * TileN;
    static constexpr uint32_t L1A_BYTES = L1A_TILE * sizeof(Element);
    static constexpr uint32_t L1B_BYTES = L1B_TILE * sizeof(Element);

    CHUNK_LOSS_DEVICE explicit HandMmadZ(CubeBuf& buf) : buf_(buf)
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            l1A_[i] = buf_.L1<Element>(L1A_TILE, L1A_BYTES * i);
            l1B_[i] = buf_.L1<Element>(L1B_TILE, L1A_BYTES * STAGES + L1B_BYTES * i);
            l0A_[i] = buf_.L0A<Element>((CubeBuf::L0A_SIZE / STAGES) / sizeof(Element),
                                        (CubeBuf::L0A_SIZE / STAGES) * i);
            l0B_[i] = buf_.L0B<Element>((CubeBuf::L0B_SIZE / STAGES) / sizeof(Element),
                                        (CubeBuf::L0B_SIZE / STAGES) * i);
            l1AEvt_[i] = static_cast<int32_t>(i);
            l1BEvt_[i] = static_cast<int32_t>(i + STAGES);
            l0AEvt_[i] = static_cast<int32_t>(i);
            l0BEvt_[i] = static_cast<int32_t>(i + STAGES);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvt_[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvt_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvt_[i]);
        }
        l0C_ = buf_.L0C<float>(CubeBuf::L0C_SIZE / sizeof(float), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    CHUNK_LOSS_DEVICE ~HandMmadZ()
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvt_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvt_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvt_[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    // Catlass Preload + Callback: sync next-K/N L1 while Mmad; Wait AIV before Fixpipe;
    // issue next-N 2nd K before Wait so MTE2 overlaps CrossCore Wait (~378 vs ~382 without).
    CHUNK_LOSS_DEVICE void Compute(const AscendC::GlobalTensor<Element>& gmA, const AscendC::GlobalTensor<Element>& gmB,
                                   const AscendC::GlobalTensor<Element>& gmC, uint32_t actualM, uint32_t actualN,
                                   uint32_t actualK, uint32_t aStride, uint32_t bStride,
                                   const AscendC::GlobalTensor<Element>& gmNextA,
                                   const AscendC::GlobalTensor<Element>& gmNextB, uint32_t nextM, uint32_t nextN,
                                   uint32_t nextK, bool isFirstBlock, bool hasNextBlock, bool waitAivBeforeFix = false,
                                   uint32_t aivStage = 0, bool setAicDone = false)
    {
        using detail::CeilDivU;
        using detail::RoundUp16;

        const uint32_t mRound = RoundUp16(actualM);
        const uint32_t nRound = RoundUp16(actualN);
        const uint32_t kTileCount = CeilDivU(actualK, L1K);
        const uint32_t startTileIdx = AscendC::GetBlockIdx();
        const uint32_t firstTileIdx = (kTileCount == 0u) ? 0u : (startTileIdx % kTileCount);
        const uint32_t lastTileIdx = (kTileCount == 0u) ? 0u : ((startTileIdx + kTileCount - 1u) % kTileCount);

        uint32_t kActual = (firstTileIdx + 1u < kTileCount) ? L1K : (actualK - firstTileIdx * L1K);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                LoadL1AB(l1ListId_, shuffleKIdx * L1K, kActual, actualM, actualN, aStride, bStride, gmA, gmB);
            }
            uint32_t l1Next = (l1ListId_ + 1u < STAGES) ? (l1ListId_ + 1u) : 0u;
            uint32_t kActualNext = 0;
            if (kLoopIdx == 0u && nextSecondPreloaded_ && kTileCount > 1u) {
                kActualNext = nextSecondKActual_;
                l1Next = nextSecondL1Id_;
                nextSecondPreloaded_ = false;
            } else if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1u) % kTileCount;
                kActualNext = (shuffleKIdxNext + 1u < kTileCount) ? L1K : (actualK - shuffleKIdxNext * L1K);
                LoadL1AB(l1Next, shuffleKIdxNext * L1K, kActualNext, actualM, actualN, aStride, bStride, gmA, gmB);
            } else if (hasNextBlock) {
                uint32_t kTileCountNext = CeilDivU(nextK, L1K);
                if (kTileCountNext == 0u) {
                    kTileCountNext = 1u;
                }
                uint32_t firstNext = startTileIdx % kTileCountNext;
                kActualNext = (firstNext + 1u < kTileCountNext) ? L1K : (nextK - firstNext * L1K);
                LoadL1AB(l1Next, firstNext * L1K, kActualNext, nextM, nextN, aStride, bStride, gmNextA, gmNextB);
            }

            const uint32_t kPartLoop = CeilDivU(kActual, L0K);
            uint32_t l0AId = 0u;
            uint32_t l0BId = 0u;

            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                uint32_t kPart = (kPartIdx + 1u < kPartLoop) ? L0K : (kActual - kPartIdx * L0K);
                uint32_t kPartAlign = RoundUp16(kPart);

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvt_[l0AId]);
                if (kPartIdx == 0u) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvt_[l1ListId_]);
                }
                {
                    AscendC::LoadData2dParams ld;
                    ld.startIndex = 0;
                    ld.repeatTimes = static_cast<uint16_t>(kPartAlign / C0);
                    ld.srcStride = static_cast<uint16_t>(TileM / C0);
                    ld.sid = 0;
                    ld.dstGap = 0;
                    ld.ifTranspose = false;
                    ld.addrMode = 0;
                    uint64_t aOff = static_cast<uint64_t>(kPartIdx * L0K / C0) * TileM * C0;
                    for (uint32_t i = 0; i < mRound / C0; ++i) {
                        AscendC::LoadData(l0A_[l0AId][i * kPartAlign * C0], l1A_[l1ListId_][aOff + i * FRACTAL], ld);
                    }
                }
                if (kPartIdx + 1u == kPartLoop) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvt_[l1ListId_]);
                }

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvt_[l0BId]);
                if (kPartIdx == 0u) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvt_[l1ListId_]);
                }
                {
                    AscendC::LoadData2dParams ld;
                    ld.startIndex = 0;
                    ld.sid = 0;
                    ld.dstGap = 0;
                    ld.ifTranspose = false;
                    ld.addrMode = 0;
                    uint64_t bOff = static_cast<uint64_t>(kPartIdx * L0K / C0) * TileN * C0;
                    if (nRound == TileN) {
                        ld.repeatTimes = static_cast<uint16_t>((kPartAlign / C0) * (nRound / C0));
                        ld.srcStride = 1;
                        AscendC::LoadData(l0B_[l0BId], l1B_[l1ListId_][bOff], ld);
                    } else {
                        ld.repeatTimes = static_cast<uint16_t>(nRound / C0);
                        ld.srcStride = 1;
                        for (uint32_t i = 0; i < kPartAlign / C0; ++i) {
                            AscendC::LoadData(l0B_[l0BId][i * nRound * C0], l1B_[l1ListId_][bOff + i * TileN * C0], ld);
                        }
                    }
                }
                if (kPartIdx + 1u == kPartLoop) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[l1ListId_]);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                bool initC = (kLoopIdx == 0u && kPartIdx == 0u);
                uint8_t unitFlag = 0b00;
                if constexpr (ENABLE_UNIT_FLAG) {
                    bool last = (kLoopIdx + 1u == kTileCount) && (kPartIdx + 1u == kPartLoop);
                    unitFlag = last ? 0b11 : 0b10;
                }
                AscendC::MmadParams mad;
                mad.m = mRound;
                mad.n = nRound;
                mad.k = kPart;
                mad.cmatrixInitVal = initC;
                mad.unitFlag = unitFlag;
                AscendC::Mmad(l0C_, l0A_[l0AId], l0B_[l0BId], mad);
                if ((mRound / C0) * (nRound / C0) < 10u) {
                    AscendC::PipeBarrier<PIPE_M>();
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvt_[l0BId]);
                l0BId = (l0BId + 1u < STAGES) ? (l0BId + 1u) : 0u;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvt_[l0AId]);
                l0AId = (l0AId + 1u < STAGES) ? (l0AId + 1u) : 0u;
            }
            l1ListId_ = l1Next;
            kActual = kActualNext;
        }

        // K2 of next-N before Wait/Fixpipe (Callback MTE2 ∩ Wait).
        if (hasNextBlock && (waitAivBeforeFix || setAicDone)) {
            uint32_t kTileCountNext = CeilDivU(nextK, L1K);
            if (kTileCountNext > 1u) {
                uint32_t startNext = AscendC::GetBlockIdx() % kTileCountNext;
                uint32_t shuffle1 = (startNext + 1u) % kTileCountNext;
                uint32_t kAct1 = (shuffle1 + 1u < kTileCountNext) ? L1K : (nextK - shuffle1 * L1K);
                uint32_t freeId = (l1ListId_ + 1u < STAGES) ? (l1ListId_ + 1u) : 0u;
                LoadL1AB(freeId, shuffle1 * L1K, kAct1, nextM, nextN, aStride, bStride, gmNextA, gmNextB);
                nextSecondPreloaded_ = true;
                nextSecondL1Id_ = freeId;
                nextSecondKActual_ = kAct1;
            }
        }

        if (waitAivBeforeFix) {
            AscendC::CrossCoreWaitFlag(FlagAivFree(aivStage));
        }
        AscendC::FixpipeParamsV220 fp;
        fp.nSize = actualN;
        fp.mSize = actualM;
        fp.srcStride = mRound;
        fp.dstStride = TileN;
        fp.ndNum = 1;
        fp.srcNdStride = 0;
        fp.dstNdStride = 0;
        if constexpr (ENABLE_UNIT_FLAG) {
            fp.unitFlag = 0b11;
        } else {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            fp.unitFlag = 0;
        }
        if constexpr (std::is_same_v<Element, half>) {
            fp.quantPre = QuantMode_t::F322F16;
        } else {
            fp.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<Element, float, AscendC::CFG_ROW_MAJOR>(gmC, l0C_, fp);
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        if (setAicDone) {
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(kFlagAicDone);
        }
    }

private:
    CHUNK_LOSS_DEVICE void LoadL1AB(uint32_t stage, uint32_t k0, uint32_t kAct, uint32_t mAct, uint32_t nAct,
                                    uint32_t aStride, uint32_t bStride, const AscendC::GlobalTensor<Element>& aGm,
                                    const AscendC::GlobalTensor<Element>& bGm)
    {
        AscendC::Nd2NzParams nd;
        nd.ndNum = 1;
        nd.srcNdMatrixStride = 0;
        nd.dstNzMatrixStride = 0;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvt_[stage]);
        nd.nValue = mAct;
        nd.dValue = kAct;
        nd.srcDValue = aStride;
        nd.dstNzC0Stride = TileM;
        nd.dstNzNStride = 1;
        AscendC::DataCopy(l1A_[stage], aGm[k0], nd);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvt_[stage]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[stage]);
        nd.nValue = nAct;
        nd.dValue = kAct;
        nd.srcDValue = bStride;
        nd.dstNzC0Stride = TileN;
        nd.dstNzNStride = 1;
        AscendC::DataCopy(l1B_[stage], bGm[k0], nd);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvt_[stage]);
    }

    CubeBuf& buf_;
    AscendC::LocalTensor<Element> l1A_[STAGES];
    AscendC::LocalTensor<Element> l1B_[STAGES];
    AscendC::LocalTensor<Element> l0A_[STAGES];
    AscendC::LocalTensor<Element> l0B_[STAGES];
    AscendC::LocalTensor<float> l0C_;
    int32_t l1AEvt_[STAGES]{};
    int32_t l1BEvt_[STAGES]{};
    int32_t l0AEvt_[STAGES]{};
    int32_t l0BEvt_[STAGES]{};
    uint32_t l1ListId_{0};
    bool nextSecondPreloaded_{false};
    uint32_t nextSecondL1Id_{0};
    uint32_t nextSecondKActual_{0};
};

template <class Element, uint32_t TileM, uint32_t TileN = 256, uint32_t L1K = 256, uint32_t L0K = 64,
          bool ENABLE_UNIT_FLAG = true>
class HandMmadH {
public:
    static constexpr uint32_t STAGES = CubeBuf::STAGES;
    static constexpr uint32_t C0 = 16;
    static constexpr uint32_t FRACTAL = 256;
    static constexpr uint32_t L1A_ELEMS = (CubeBuf::L1_SIZE / 2) / sizeof(Element);
    static constexpr uint32_t L1B_TILE = L1K * TileN; // zN (K, N_out)
    static constexpr uint32_t L1B_BYTES = L1B_TILE * sizeof(Element);

    CHUNK_LOSS_DEVICE explicit HandMmadH(CubeBuf& buf) : buf_(buf)
    {
        // FullLoadA: L1A sticky across H tiles — no ctor/dtor MTE1_MTE2 on l1A.
        l1A_ = buf_.L1<Element>(L1A_ELEMS, 0);
        l1AEvt_ = 0;
        for (uint32_t i = 0; i < STAGES; ++i) {
            l1B_[i] = buf_.L1<Element>(L1B_TILE, CubeBuf::L1_SIZE / 2 + L1B_BYTES * i);
            l0A_[i] = buf_.L0A<Element>((CubeBuf::L0A_SIZE / STAGES) / sizeof(Element),
                                        (CubeBuf::L0A_SIZE / STAGES) * i);
            l0B_[i] = buf_.L0B<Element>((CubeBuf::L0B_SIZE / STAGES) / sizeof(Element),
                                        (CubeBuf::L0B_SIZE / STAGES) * i);
            l1BEvt_[i] = static_cast<int32_t>(i + STAGES);
            l0AEvt_[i] = static_cast<int32_t>(i);
            l0BEvt_[i] = static_cast<int32_t>(i + STAGES);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvt_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvt_[i]);
        }
        l0C_ = buf_.L0C<float>(CubeBuf::L0C_SIZE / sizeof(float), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    CHUNK_LOSS_DEVICE ~HandMmadH()
    {
        for (uint32_t i = 0; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvt_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvt_[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    CHUNK_LOSS_DEVICE bool HasPrefetchedB() const { return bPrefetched_; }

    // Prefetch first B L1 tile before WaitAivFreeP2 so MTE2 overlaps AIV BuildDz.
    CHUNK_LOSS_DEVICE void PrefetchB(const AscendC::GlobalTensor<Element>& gmB, uint32_t actualNOut, uint32_t actualK,
                                     uint32_t bStride)
    {
        using detail::MinU;
        uint32_t kActual = MinU(actualK, L1K);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[l1ListId_]);
        AscendC::Nd2NzParams nd;
        nd.ndNum = 1;
        nd.nValue = kActual;
        nd.dValue = actualNOut;
        nd.srcDValue = bStride;
        nd.srcNdMatrixStride = 0;
        nd.dstNzC0Stride = L1K;
        nd.dstNzNStride = 1;
        nd.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1B_[l1ListId_], gmB, nd);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvt_[l1ListId_]);
        bPrefetched_ = true;
        prefetchedKActual_ = kActual;
    }

    CHUNK_LOSS_DEVICE void Compute(const AscendC::GlobalTensor<Element>& gmA, const AscendC::GlobalTensor<Element>& gmB,
                                   const AscendC::GlobalTensor<float>& gmC, uint32_t actualM, uint32_t actualNOut,
                                   uint32_t actualK, uint32_t aStride, uint32_t bStride, uint32_t cStride,
                                   bool needLoadL1, const AscendC::GlobalTensor<Element>& gmNextB, uint32_t nextNOut,
                                   uint32_t nextK, bool hasNextB)
    {
        using detail::CeilDivU;
        using detail::MinU;
        using detail::RoundUp16;

        const uint32_t mRound = RoundUp16(actualM);
        const uint32_t nRound = RoundUp16(actualNOut);
        const uint32_t kTileCount = CeilDivU(actualK, L1K);
        uint32_t kActual = MinU(actualK, L1K);

        if (needLoadL1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvt_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvt_);
            AscendC::Nd2NzParams nd;
            nd.ndNum = 1;
            nd.nValue = actualM;
            nd.dValue = actualK;
            nd.srcDValue = aStride;
            nd.srcNdMatrixStride = 0;
            nd.dstNzC0Stride = TileM;
            nd.dstNzNStride = 1;
            nd.dstNzMatrixStride = 0;
            AscendC::DataCopy(l1A_, gmA, nd);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvt_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvt_);
        }

        // B RowMajor (K, N_out) → zN (skip if PrefetchB already issued).
        if (bPrefetched_) {
            kActual = prefetchedKActual_;
            bPrefetched_ = false;
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[l1ListId_]);
            {
                AscendC::Nd2NzParams nd;
                nd.ndNum = 1;
                nd.nValue = kActual;
                nd.dValue = actualNOut;
                nd.srcDValue = bStride;
                nd.srcNdMatrixStride = 0;
                nd.dstNzC0Stride = L1K; // zN(L1K, TileN) stride(3)/C0
                nd.dstNzNStride = 1;
                nd.dstNzMatrixStride = 0;
                AscendC::DataCopy(l1B_[l1ListId_], gmB, nd);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvt_[l1ListId_]);
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            uint32_t l1Next = (l1ListId_ + 1u < STAGES) ? (l1ListId_ + 1u) : 0u;
            uint32_t kActualNext = 0;
            if (kLoopIdx + 1u < kTileCount) {
                uint32_t k0Next = (kLoopIdx + 1u) * L1K;
                kActualNext = MinU(actualK - k0Next, L1K);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[l1Next]);
                AscendC::Nd2NzParams nd;
                nd.ndNum = 1;
                nd.nValue = kActualNext;
                nd.dValue = actualNOut;
                nd.srcDValue = bStride;
                nd.srcNdMatrixStride = 0;
                nd.dstNzC0Stride = L1K;
                nd.dstNzNStride = 1;
                nd.dstNzMatrixStride = 0;
                AscendC::DataCopy(l1B_[l1Next], gmB[static_cast<uint64_t>(k0Next) * bStride], nd);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvt_[l1Next]);
            }

            const uint32_t kPartLoop = CeilDivU(kActual, L0K);
            uint32_t l0AId = 0u;
            uint32_t l0BId = 0u;

            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                uint32_t kPart = (kPartIdx < kPartLoop - 1u) ? L0K : (kActual - kPartIdx * L0K);
                uint32_t kPartAlign = RoundUp16(kPart);

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvt_[l0AId]);
                {
                    AscendC::LoadData2dParams ld;
                    ld.startIndex = 0;
                    ld.repeatTimes = static_cast<uint16_t>(kPartAlign / C0);
                    ld.srcStride = static_cast<uint16_t>(TileM / C0);
                    ld.sid = 0;
                    ld.dstGap = 0;
                    ld.ifTranspose = false;
                    ld.addrMode = 0;
                    uint32_t kBase = kLoopIdx * L1K + kPartIdx * L0K;
                    uint64_t aOff = static_cast<uint64_t>(kBase / C0) * TileM * C0;
                    for (uint32_t i = 0; i < mRound / C0; ++i) {
                        AscendC::LoadData(l0A_[l0AId][i * kPartAlign * C0], l1A_[aOff + i * FRACTAL], ld);
                    }
                }

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvt_[l0BId]);
                if (kPartIdx == 0u) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvt_[l1ListId_]);
                }
                {
                    AscendC::LoadData2dParams ld;
                    ld.startIndex = 0;
                    ld.repeatTimes = static_cast<uint16_t>(nRound / C0);
                    ld.srcStride = static_cast<uint16_t>(L1K / C0);
                    ld.sid = 0;
                    ld.dstGap = 0;
                    ld.ifTranspose = true;
                    ld.addrMode = 0;
                    uint64_t bOff = static_cast<uint64_t>(kPartIdx * L0K / C0) * FRACTAL;
                    for (uint32_t i = 0; i < kPartAlign / C0; ++i) {
                        AscendC::LoadData(l0B_[l0BId][i * nRound * C0], l1B_[l1ListId_][bOff + i * FRACTAL], ld);
                    }
                }
                if (kPartIdx == kPartLoop - 1u) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvt_[l1ListId_]);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                bool initC = (kLoopIdx == 0u && kPartIdx == 0u);
                uint8_t unitFlag = 0b00;
                if constexpr (ENABLE_UNIT_FLAG) {
                    bool last = (kLoopIdx == kTileCount - 1u) && (kPartIdx == kPartLoop - 1u);
                    unitFlag = last ? 0b11 : 0b10;
                }
                AscendC::MmadParams mad;
                mad.m = mRound;
                mad.n = nRound;
                mad.k = kPart;
                mad.cmatrixInitVal = initC;
                mad.unitFlag = unitFlag;
                AscendC::Mmad(l0C_, l0A_[l0AId], l0B_[l0BId], mad);
                if ((mRound / C0) * (nRound / C0) < 10u) {
                    AscendC::PipeBarrier<PIPE_M>();
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvt_[l0BId]);
                l0BId = (l0BId + 1u < STAGES) ? (l0BId + 1u) : 0u;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvt_[l0AId]);
                l0AId = (l0AId + 1u < STAGES) ? (l0AId + 1u) : 0u;
            }
            l1ListId_ = l1Next;
            kActual = kActualNext;
        }

        // Next H/N B into free L1 before Fixpipe (MTE2 ∩ Fixpipe), like Pass1 K2 Callback.
        if (hasNextB) {
            PrefetchB(gmNextB, nextNOut, nextK, bStride);
        }

        AscendC::FixpipeParamsV220 fp;
        fp.nSize = actualNOut;
        fp.mSize = actualM;
        fp.srcStride = mRound;
        fp.dstStride = cStride;
        fp.ndNum = 1;
        fp.srcNdStride = 0;
        fp.dstNdStride = 0;
        fp.quantPre = QuantMode_t::NoQuant;
        if constexpr (ENABLE_UNIT_FLAG) {
            fp.unitFlag = 0b11;
        } else {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            fp.unitFlag = 0;
        }
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(gmC, l0C_, fp);
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
    }

private:
    CubeBuf& buf_;
    AscendC::LocalTensor<Element> l1A_;
    AscendC::LocalTensor<Element> l1B_[STAGES];
    AscendC::LocalTensor<Element> l0A_[STAGES];
    AscendC::LocalTensor<Element> l0B_[STAGES];
    AscendC::LocalTensor<float> l0C_;
    int32_t l1AEvt_{0};
    int32_t l1BEvt_[STAGES]{};
    int32_t l0AEvt_[STAGES]{};
    int32_t l0BEvt_[STAGES]{};
    uint32_t l1ListId_{0};
    bool bPrefetched_{false};
    uint32_t prefetchedKActual_{0};
};

} // namespace chunkloss

#endif
