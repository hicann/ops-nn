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
 * \file cube_op.h
 * \brief ChunkLoss arch35 手写 Cube matmul 流水
 *
 * Pass1 ShuffleK / L1 分片加载；Pass2 FullLoadA 与 dual L0C Fixpipe 重叠。
 */
#ifndef CHUNKLOSS_ARCH35_CUBE_OP_H_
#define CHUNKLOSS_ARCH35_CUBE_OP_H_

#include "chunk_loss_common_arch35.h"
#include "chunk_loss_cube_helper.h"

namespace chunkloss {

template <typename Element, uint32_t TileM, bool kL0cToUb = false>
class CubeOp {
public:
    static constexpr uint32_t STAGES = 2;
    static constexpr uint32_t kTileHPass2 = kTileH;
    static constexpr uint32_t kL0KPass2Use = kL0KPass2;
    static constexpr uint32_t L1A_ELEMS = TileM * kL1K;
    // L1B 按 Pass1（N×L1K）与 Pass2（L1K×H）取较大值。
    static constexpr uint32_t L1B_ELEMS = (kTileN * kL1K) > (kL1K * kTileH) ? (kTileN* kL1K) : (kL1K* kTileH);
    static constexpr uint32_t L0A_ELEMS = 32 * 1024 / sizeof(Element);
    static constexpr uint32_t L0B_ELEMS = 32 * 1024 / sizeof(Element);
    static constexpr uint32_t L0C_ELEMS = 256 * 1024 / sizeof(float);

    CHUNKLOSS_DEVICE CubeOp() {}

    CHUNKLOSS_DEVICE void Init(AscendC::TPipe* tPipe)
    {
        tPipe_ = tPipe;
        constexpr uint32_t l1Bytes = (L1A_ELEMS + L1B_ELEMS) * STAGES * static_cast<uint32_t>(sizeof(Element));
        static_assert(l1Bytes <= static_cast<uint32_t>(kL1Size), "CubeOp L1 OOB");

        tPipe_->InitBuffer(l1Buf_, l1Bytes);
        tPipe_->InitBuffer(l0aBuf_, kL0AbSize);
        tPipe_->InitBuffer(l0bBuf_, kL0AbSize);
        tPipe_->InitBuffer(l0cBuf_, kL0CSize);

        uint32_t off = 0;
        for (uint32_t i = 0; i < STAGES; ++i) {
            l1A_[i] = l1Buf_.GetWithOffset<Element>(L1A_ELEMS, off);
            off += L1A_ELEMS * sizeof(Element);
        }
        for (uint32_t i = 0; i < STAGES; ++i) {
            l1B_[i] = l1Buf_.GetWithOffset<Element>(L1B_ELEMS, off);
            off += L1B_ELEMS * sizeof(Element);
        }

        l0A_[0] = l0aBuf_.GetWithOffset<Element>(L0A_ELEMS, 0);
        l0A_[1] = l0aBuf_.GetWithOffset<Element>(L0A_ELEMS, L0A_ELEMS * sizeof(Element));
        l0B_[0] = l0bBuf_.GetWithOffset<Element>(L0B_ELEMS, 0);
        l0B_[1] = l0bBuf_.GetWithOffset<Element>(L0B_ELEMS, L0B_ELEMS * sizeof(Element));
        // Dual L0C for Pass2: each bank holds TileM×kTileH (128×256) floats.
        static_assert(TileM * kTileHPass2 <= L0C_ELEMS / STAGES, "Pass2 L0C bank OOB");
        constexpr uint32_t kL0cBank = L0C_ELEMS / STAGES;
        l0C_[0] = l0cBuf_.GetWithOffset<float>(kL0cBank, 0);
        l0C_[1] = l0cBuf_.GetWithOffset<float>(kL0cBank, kL0cBank * sizeof(float));

        l1Id_ = 0;
        l0AId_ = 0;
        l0BId_ = 0;
        dhCId_ = 0;
        eventPingPong_ = 0;
        writeToUb_ = false;
        dhAResident_ = false;
        dhBPreloaded_ = false;
        SetEventFlag();
    }

    CHUNKLOSS_DEVICE void SetEventFlag()
    {
        // L1A:0/1  L1B:2/3  L0A:4/5  L0B:6/7  FIX L0C banks:0/1
        CHUNKLOSS_SET_FLAG(MTE1, MTE2, EVENT_ID0);
        CHUNKLOSS_SET_FLAG(MTE1, MTE2, EVENT_ID1);
        CHUNKLOSS_SET_FLAG(MTE1, MTE2, EVENT_ID2);
        CHUNKLOSS_SET_FLAG(MTE1, MTE2, EVENT_ID3);
        CHUNKLOSS_SET_FLAG(M, MTE1, EVENT_ID4);
        CHUNKLOSS_SET_FLAG(M, MTE1, EVENT_ID5);
        CHUNKLOSS_SET_FLAG(M, MTE1, EVENT_ID6);
        CHUNKLOSS_SET_FLAG(M, MTE1, EVENT_ID7);
        CHUNKLOSS_SET_FLAG(FIX, M, EVENT_ID0);
        CHUNKLOSS_SET_FLAG(FIX, M, EVENT_ID1);
    }

    CHUNKLOSS_DEVICE void WaitEventFlag()
    {
        CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        CHUNKLOSS_WAIT_FLAG(M, MTE1, EVENT_ID4);
        CHUNKLOSS_WAIT_FLAG(M, MTE1, EVENT_ID5);
        CHUNKLOSS_WAIT_FLAG(M, MTE1, EVENT_ID6);
        CHUNKLOSS_WAIT_FLAG(M, MTE1, EVENT_ID7);
        CHUNKLOSS_WAIT_FLAG(FIX, M, EVENT_ID0);
        CHUNKLOSS_WAIT_FLAG(FIX, M, EVENT_ID1);
    }

    CHUNKLOSS_DEVICE void Fini() { WaitEventFlag(); }

    /** SoftSync: BlockMmadZ dtor drains flags, BlockMmadH ctor re-posts before Pass2. */
    CHUNKLOSS_DEVICE void ResetForPass2()
    {
        WaitEventFlag();
        l1Id_ = 0;
        l0AId_ = 0;
        l0BId_ = 0;
        dhCId_ = 0;
        eventPingPong_ = 0;
        writeToUb_ = false;
        dhAResident_ = false;
        dhBPreloaded_ = false;
        SetEventFlag();
    }

    CHUNKLOSS_DEVICE void SetUbDst(const AscendC::LocalTensor<float>& ub, uint32_t dstStride)
    {
        writeToUb_ = true;
        ubC_ = ub;
        ubDstStride_ = dstStride;
    }

    // gmOut: fp32 GM ping-pong for Pass1 LSE (L0C→UB ignores gmOut and writes UB).
    CHUNKLOSS_DEVICE void MatmulZW(AscendC::GlobalTensor<Element> gmX, AscendC::GlobalTensor<Element> gmW,
                                   AscendC::GlobalTensor<float> gmOut, uint32_t actualM, uint32_t actualN, uint32_t h,
                                   uint32_t n0, uint32_t n0Next, uint32_t actualNNext, bool isFirst, bool hasNext)
    {
        MatmulZWOnce(gmX, gmW, gmOut, actualM, actualN, h, n0, n0Next, actualNNext, isFirst, hasNext);
    }

    CHUNKLOSS_DEVICE void MatmulZWOnce(AscendC::GlobalTensor<Element> gmX, AscendC::GlobalTensor<Element> gmW,
                                       AscendC::GlobalTensor<float> gmOut, uint32_t actualM, uint32_t actualN,
                                       uint32_t h, uint32_t n0, uint32_t n0Next, uint32_t actualNNext, bool isFirst,
                                       bool hasNext)
    {
        const int32_t mAlign = static_cast<int32_t>(CeilAlignU(actualM, kCubeBlock));
        const int32_t nAlign = static_cast<int32_t>(CeilAlignU(actualN, kCubeBlock));
        const uint32_t mmadM = MmadMFloor(actualM);
        const uint32_t kTileCount = CeilDivU(h, kL1K);
        if (kTileCount == 0u) {
            return;
        }

        // ShuffleK/split/indep-L0 only on L0C→UB path.
        const bool largeH = kL0cToUb && (h >= 1024u) && (kTileCount >= 4u);
        const uint32_t startK = largeH ? (AscendC::GetBlockIdx() % kTileCount) : 0u;
        uint32_t k0Cur = startK * kL1K;
        uint32_t kAct = (k0Cur + kL1K <= h) ? kL1K : (h - k0Cur);
        if (isFirst) {
            if (largeH) {
                LoadZwL1Split(l1Id_, gmX, gmW, actualM, actualN, h, n0, k0Cur, kAct, mAlign, nAlign);
            } else {
                LoadZwL1(l1Id_, gmX, gmW, actualM, actualN, h, n0, k0Cur, kAct, mAlign, nAlign);
            }
        }

        for (uint32_t kLoop = 0; kLoop < kTileCount; ++kLoop) {
            const uint32_t l1Next = (l1Id_ + 1u < STAGES) ? (l1Id_ + 1u) : 0u;
            uint32_t kActNext = 0u;
            const bool lastK = (kLoop + 1u >= kTileCount);

            if (!lastK) {
                const uint32_t kIdxNext = largeH ? ((startK + kLoop + 1u) % kTileCount) : (kLoop + 1u);
                const uint32_t k0Next = kIdxNext * kL1K;
                kActNext = (k0Next + kL1K <= h) ? kL1K : (h - k0Next);
                if (largeH) {
                    LoadZwL1Split(l1Next, gmX, gmW, actualM, actualN, h, n0, k0Next, kActNext, mAlign, nAlign);
                } else {
                    LoadZwL1(l1Next, gmX, gmW, actualM, actualN, h, n0, k0Next, kActNext, mAlign, nAlign);
                }
            } else if (hasNext) {
                const uint32_t k0Next = startK * kL1K;
                kActNext = (k0Next + kL1K <= h) ? kL1K : (h - k0Next);
                const int32_t nAlignNext = static_cast<int32_t>(CeilAlignU(actualNNext, kCubeBlock));
                if (largeH) {
                    LoadZwL1Split(l1Next, gmX, gmW, actualM, actualNNext, h, n0Next, k0Next, kActNext, mAlign,
                                  nAlignNext);
                } else {
                    LoadZwL1(l1Next, gmX, gmW, actualM, actualNNext, h, n0Next, k0Next, kActNext, mAlign, nAlignNext);
                }
            }

            AscendC::TEventID evA = (l1Id_ == 0u) ? EVENT_ID0 : EVENT_ID1;
            AscendC::TEventID evB = (l1Id_ == 0u) ? EVENT_ID2 : EVENT_ID3;

            const uint32_t kPartLoop = CeilDivU(kAct, kL0K);
            for (uint32_t kp = 0; kp < kPartLoop; ++kp) {
                const uint32_t kPart = (kp + 1u < kPartLoop) ? kL0K : (kAct - kp * kL0K);
                const uint32_t kPartAlign = CeilAlignU(kPart, kC0Fp16);
                const uint32_t l1AOff = kp * kL0K * static_cast<uint32_t>(mAlign);
                const uint32_t l1BOff = kp * kL0K * static_cast<uint32_t>(nAlign);

                if (largeH) {
                    AscendC::TEventID evL0A = (l0AId_ == 0u) ? EVENT_ID4 : EVENT_ID5;
                    AscendC::TEventID evL0B = (l0BId_ == 0u) ? EVENT_ID6 : EVENT_ID7;

                    CHUNKLOSS_WAIT_FLAG(M, MTE1, evL0A);
                    if (kp == 0u) {
                        CHUNKLOSS_WAIT_FLAG(MTE2, MTE1, evA);
                    }
                    LoadL1ToL0Nz(l0A_[l0AId_], l1A_[l1Id_][l1AOff], mAlign, static_cast<int32_t>(kPartAlign));
                    if (kp + 1u >= kPartLoop) {
                        CHUNKLOSS_SET_FLAG(MTE1, MTE2, evA);
                    }

                    CHUNKLOSS_WAIT_FLAG(M, MTE1, evL0B);
                    if (kp == 0u) {
                        CHUNKLOSS_WAIT_FLAG(MTE2, MTE1, evB);
                    }
                    LoadL1ToL0Nz(l0B_[l0BId_], l1B_[l1Id_][l1BOff], nAlign, static_cast<int32_t>(kPartAlign));
                    if (kp + 1u >= kPartLoop) {
                        CHUNKLOSS_SET_FLAG(MTE1, MTE2, evB);
                    }

                    CHUNKLOSS_SET_FLAG(MTE1, M, EVENT_ID0);
                    CHUNKLOSS_WAIT_FLAG(MTE1, M, EVENT_ID0);

                    // Always Wait FIX_M on first K of a tile (GM + L0C). SoftSync skips this
                    // under ENABLE_UNIT_FLAG because Pass2 uses a fresh BlockMmad; we share CubeOp.
                    if (kLoop == 0u && kp == 0u) {
                        CHUNKLOSS_WAIT_FLAG(FIX, M, EVENT_ID0);
                    }
                    const bool initC = (kLoop == 0u && kp == 0u);
                    uint8_t unitFlag = (lastK && kp + 1u >= kPartLoop) ? static_cast<uint8_t>(0b11) :
                                                                         static_cast<uint8_t>(0b10);
                    MmadUnit(l0C_[0], l0A_[l0AId_], l0B_[l0BId_], mmadM, actualN, kPart, initC, unitFlag);
                    CHUNKLOSS_SET_FLAG(M, MTE1, evL0B);
                    l0BId_ = (l0BId_ + 1u < STAGES) ? (l0BId_ + 1u) : 0u;
                    CHUNKLOSS_SET_FLAG(M, MTE1, evL0A);
                    l0AId_ = (l0AId_ + 1u < STAGES) ? (l0AId_ + 1u) : 0u;
                } else {
                    const uint32_t l0Id = static_cast<uint32_t>(eventPingPong_);
                    AscendC::TEventID ev = eventPingPong_ ? EVENT_ID5 : EVENT_ID4;

                    CHUNKLOSS_WAIT_FLAG(M, MTE1, ev);
                    if (kp == 0u) {
                        CHUNKLOSS_WAIT_FLAG(MTE2, MTE1, evA);
                    }
                    LoadL1ToL0Nz(l0A_[l0Id], l1A_[l1Id_][l1AOff], mAlign, static_cast<int32_t>(kPartAlign));
                    LoadL1ToL0Nz(l0B_[l0Id], l1B_[l1Id_][l1BOff], nAlign, static_cast<int32_t>(kPartAlign));
                    if (kp + 1u >= kPartLoop) {
                        CHUNKLOSS_SET_FLAG(MTE1, MTE2, evA);
                    }
                    CHUNKLOSS_SET_FLAG(MTE1, M, EVENT_ID0);
                    CHUNKLOSS_WAIT_FLAG(MTE1, M, EVENT_ID0);

                    if (kLoop == 0u && kp == 0u) {
                        CHUNKLOSS_WAIT_FLAG(FIX, M, EVENT_ID0);
                    }
                    const bool initC = (kLoop == 0u && kp == 0u);
                    uint8_t unitFlag = (lastK && kp + 1u >= kPartLoop) ? static_cast<uint8_t>(0b11) :
                                                                         static_cast<uint8_t>(0b10);
                    MmadUnit(l0C_[0], l0A_[l0Id], l0B_[l0Id], mmadM, actualN, kPart, initC, unitFlag);
                    CHUNKLOSS_SET_FLAG(M, MTE1, ev);
                    eventPingPong_ = 1 - eventPingPong_;
                }
            }

            l1Id_ = l1Next;
            kAct = kActNext;
        }

        const uint8_t unitFlag = 0b11;
        if constexpr (kL0cToUb) {
            if (writeToUb_) {
                // Must pass actualM (not mAlign): dualDstCtl splits mSize/2 per AIV.
                // Rem e.g. M=28 → half=14; mAlign=32 would split 16+16 and misalign AIV1.
                FixpipeUbSplitM(ubC_, l0C_[0], actualM, actualN, ubDstStride_, unitFlag);
                writeToUb_ = false;
            }
            // Post FIX_M so next Pass1 tile / Pass2 can Wait (shared CubeOp, not SoftSync).
            CHUNKLOSS_SET_FLAG(FIX, M, EVENT_ID0);
        } else {
            // Keep fp32 for AIV online LSE (avoid Element round-trip vs fp32 golden).
            FixpipeGmRowMajor<float, float>(gmOut, l0C_[0], actualM, actualN, kTileN, unitFlag);
            CHUNKLOSS_SET_FLAG(FIX, M, EVENT_ID0);
        }
    }

    CHUNKLOSS_DEVICE void MatmulDH(AscendC::GlobalTensor<Element> gmDz, AscendC::GlobalTensor<Element> gmW,
                                   AscendC::GlobalTensor<float> gmOut, uint32_t actualM, uint32_t actualN,
                                   uint32_t actualH, uint32_t h0, uint32_t n0, uint32_t fullH, bool loadA = true,
                                   bool hasNextH = false, uint32_t actualHNext = 0, uint32_t h0Next = 0)
    {
        MatmulDHOnce(gmDz, gmW, gmOut, actualM, actualN, actualH, h0, n0, fullH, true, true, loadA, hasNextH,
                     actualHNext, h0Next);
    }

    CHUNKLOSS_DEVICE void MatmulDHOnce(AscendC::GlobalTensor<Element> gmDz, AscendC::GlobalTensor<Element> gmW,
                                       AscendC::GlobalTensor<float> gmOut, uint32_t actualM, uint32_t actualN,
                                       uint32_t actualH, uint32_t h0, uint32_t n0, uint32_t fullH, bool initCTile,
                                       bool doFixpipe, bool loadA = true, bool hasNextH = false,
                                       uint32_t actualHNext = 0, uint32_t h0Next = 0)
    {
        const int32_t mAlign = static_cast<int32_t>(CeilAlignU(actualM, kCubeBlock));
        const int32_t hAlign = static_cast<int32_t>(CeilAlignU(actualH, kCubeBlock));
        const uint32_t mmadM = MmadMFloor(actualM);
        const uint32_t kTileCount = CeilDivU(actualN, kL1K);
        // FullLoadA only when A fits one L1K tile (typical N-tile=256).
        const bool fullLoadA = (kTileCount == 1u);

        if (!fullLoadA) {
            // Multi-K: reload A+B each K tile.
            l1Id_ = 0u;
            for (uint32_t kLoop = 0; kLoop < kTileCount; ++kLoop) {
                const uint32_t k0 = kLoop * kL1K;
                const uint32_t kAct = (k0 + kL1K <= actualN) ? kL1K : (actualN - k0);
                const uint32_t kAlign = CeilAlignU(kAct, kC0Fp16);
                const int32_t kRowAlign = static_cast<int32_t>(CeilAlignU(kAct, kCubeBlock));
                const bool lastK = (kLoop + 1u >= kTileCount);

                CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
                LoadGm2L1(l1A_[0], gmDz[k0], static_cast<int32_t>(actualM), static_cast<int32_t>(kAct), mAlign,
                          static_cast<int32_t>(kAlign), static_cast<int32_t>(kTileN));
                const uint64_t wOff = static_cast<uint64_t>(n0 + k0) * fullH + h0;
                LoadGm2L1(l1B_[0], gmW[wOff], static_cast<int32_t>(kAct), static_cast<int32_t>(actualH), kRowAlign,
                          hAlign, static_cast<int32_t>(fullH));
                CHUNKLOSS_SET_FLAG(MTE2, MTE1, EVENT_ID2);
                CHUNKLOSS_WAIT_FLAG(MTE2, MTE1, EVENT_ID2);

                RunDhKParts(actualM, actualH, kAct, kRowAlign, hAlign, mAlign, mmadM, initCTile && kLoop == 0u,
                            doFixpipe && lastK);
                CHUNKLOSS_SET_FLAG(MTE1, MTE2, EVENT_ID0);
            }
        } else {
            const uint32_t kAct = actualN;
            const uint32_t kAlign = CeilAlignU(kAct, kC0Fp16);
            const int32_t kRowAlign = static_cast<int32_t>(CeilAlignU(kAct, kCubeBlock));

            if (loadA || !dhAResident_) {
                CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
                LoadGm2L1(l1A_[0], gmDz[0], static_cast<int32_t>(actualM), static_cast<int32_t>(kAct), mAlign,
                          static_cast<int32_t>(kAlign), static_cast<int32_t>(kTileN));
                CHUNKLOSS_SET_FLAG(MTE2, MTE1, EVENT_ID0);
                CHUNKLOSS_WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
                dhAResident_ = true;
            }

            AscendC::TEventID evB = (l1Id_ == 0u) ? EVENT_ID2 : EVENT_ID3;
            if (!dhBPreloaded_) {
                CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, evB);
                const uint64_t wOff = static_cast<uint64_t>(n0) * fullH + h0;
                LoadGm2L1(l1B_[l1Id_], gmW[wOff], static_cast<int32_t>(kAct), static_cast<int32_t>(actualH), kRowAlign,
                          hAlign, static_cast<int32_t>(fullH));
                CHUNKLOSS_SET_FLAG(MTE2, MTE1, evB);
            }
            dhBPreloaded_ = false;

            const uint32_t l1Next = (l1Id_ + 1u < STAGES) ? (l1Id_ + 1u) : 0u;
            if (hasNextH && actualHNext > 0u) {
                const int32_t hAlignNext = static_cast<int32_t>(CeilAlignU(actualHNext, kCubeBlock));
                AscendC::TEventID evBNext = (l1Next == 0u) ? EVENT_ID2 : EVENT_ID3;
                CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, evBNext);
                const uint64_t wOffNext = static_cast<uint64_t>(n0) * fullH + h0Next;
                LoadGm2L1(l1B_[l1Next], gmW[wOffNext], static_cast<int32_t>(kAct), static_cast<int32_t>(actualHNext),
                          kRowAlign, hAlignNext, static_cast<int32_t>(fullH));
                CHUNKLOSS_SET_FLAG(MTE2, MTE1, evBNext);
                dhBPreloaded_ = true;
            }

            CHUNKLOSS_WAIT_FLAG(MTE2, MTE1, evB);
            RunDhKParts(actualM, actualH, kAct, kRowAlign, hAlign, mAlign, mmadM, initCTile, doFixpipe);
            CHUNKLOSS_SET_FLAG(MTE1, MTE2, evB);
            l1Id_ = l1Next;
        }

        if (doFixpipe) {
            FixpipeGmRowMajor<float, float>(gmOut, l0C_[dhCId_], actualM, actualH, fullH, 0b11);
            // Per-bank FIX flag: next reuse of this L0C bank waits; other bank can Mmad now.
            AscendC::TEventID evFix = (dhCId_ == 0u) ? EVENT_ID0 : EVENT_ID1;
            CHUNKLOSS_SET_FLAG(FIX, M, evFix);
            dhCId_ = (dhCId_ + 1u < STAGES) ? (dhCId_ + 1u) : 0u;
        }
    }

    CHUNKLOSS_DEVICE void ReleaseDhA()
    {
        if (dhAResident_) {
            CHUNKLOSS_SET_FLAG(MTE1, MTE2, EVENT_ID0);
            dhAResident_ = false;
        }
        dhBPreloaded_ = false;
    }

private:
    CHUNKLOSS_DEVICE void RunDhKParts(uint32_t actualM, uint32_t actualH, uint32_t kAct, int32_t kRowAlign,
                                      int32_t hAlign, int32_t mAlign, uint32_t mmadM, bool initCTile, bool lastTile)
    {
        const uint32_t kPartLoop = CeilDivU(kAct, kL0KPass2Use);
        AscendC::TEventID evFix = (dhCId_ == 0u) ? EVENT_ID0 : EVENT_ID1;
        for (uint32_t kp = 0; kp < kPartLoop; ++kp) {
            const uint32_t kPart = (kp + 1u < kPartLoop) ? kL0KPass2Use : (kAct - kp * kL0KPass2Use);
            const uint32_t kPartAlign = CeilAlignU(kPart, kC0Fp16);
            const uint32_t l1AOff = kp * kL0KPass2Use * static_cast<uint32_t>(mAlign);
            const int32_t mFracStart = static_cast<int32_t>((kp * kL0KPass2Use) / kCubeBlock);
            const int32_t mFracCount = static_cast<int32_t>(kPartAlign / kCubeBlock);
            AscendC::TEventID evL0A = (l0AId_ == 0u) ? EVENT_ID4 : EVENT_ID5;
            AscendC::TEventID evL0B = (l0BId_ == 0u) ? EVENT_ID6 : EVENT_ID7;

            CHUNKLOSS_WAIT_FLAG(M, MTE1, evL0A);
            LoadL1ToL0Nz(l0A_[l0AId_], l1A_[0][l1AOff], mAlign, static_cast<int32_t>(kPartAlign));

            CHUNKLOSS_WAIT_FLAG(M, MTE1, evL0B);
            LoadL1ToL0Zn(l0B_[l0BId_], l1B_[l1Id_], kRowAlign, hAlign, mFracStart, mFracCount);
            CHUNKLOSS_SET_FLAG(MTE1, M, EVENT_ID0);
            CHUNKLOSS_WAIT_FLAG(MTE1, M, EVENT_ID0);

            // Wait only this bank's prior Fixpipe (dual L0C: other bank can still be draining).
            if (initCTile && kp == 0u) {
                CHUNKLOSS_WAIT_FLAG(FIX, M, evFix);
            }
            const bool initC = initCTile && (kp == 0u);
            const bool lastPart = lastTile && (kp + 1u >= kPartLoop);
            uint8_t unitFlag = lastPart ? static_cast<uint8_t>(0b11) : static_cast<uint8_t>(0b10);
            MmadUnit(l0C_[dhCId_], l0A_[l0AId_], l0B_[l0BId_], mmadM, actualH, kPart, initC, unitFlag);
            CHUNKLOSS_SET_FLAG(M, MTE1, evL0B);
            l0BId_ = (l0BId_ + 1u < STAGES) ? (l0BId_ + 1u) : 0u;
            CHUNKLOSS_SET_FLAG(M, MTE1, evL0A);
            l0AId_ = (l0AId_ + 1u < STAGES) ? (l0AId_ + 1u) : 0u;
        }
        (void)actualM;
    }

    CHUNKLOSS_DEVICE void LoadZwL1(uint32_t l1Id, AscendC::GlobalTensor<Element> gmX,
                                   AscendC::GlobalTensor<Element> gmW, uint32_t actualM, uint32_t actualN, uint32_t h,
                                   uint32_t nBase, uint32_t k0, uint32_t kAct, int32_t mAlign, int32_t nAlign)
    {
        const uint32_t kAl = CeilAlignU(kAct, kC0Fp16);
        AscendC::TEventID evL1 = (l1Id == 0u) ? EVENT_ID0 : EVENT_ID1;
        CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, evL1);
        LoadGm2L1(l1A_[l1Id], gmX[k0], static_cast<int32_t>(actualM), static_cast<int32_t>(kAct), mAlign,
                  static_cast<int32_t>(kAl), static_cast<int32_t>(h));
        LoadGm2L1(l1B_[l1Id], gmW[static_cast<uint64_t>(nBase) * h + k0], static_cast<int32_t>(actualN),
                  static_cast<int32_t>(kAct), nAlign, static_cast<int32_t>(kAl), static_cast<int32_t>(h));
        CHUNKLOSS_SET_FLAG(MTE2, MTE1, evL1);
    }

    /** Issue A then B MTE2 so both can be in flight. */
    CHUNKLOSS_DEVICE void LoadZwL1Split(uint32_t l1Id, AscendC::GlobalTensor<Element> gmX,
                                        AscendC::GlobalTensor<Element> gmW, uint32_t actualM, uint32_t actualN,
                                        uint32_t h, uint32_t nBase, uint32_t k0, uint32_t kAct, int32_t mAlign,
                                        int32_t nAlign)
    {
        const uint32_t kAl = CeilAlignU(kAct, kC0Fp16);
        AscendC::TEventID evA = (l1Id == 0u) ? EVENT_ID0 : EVENT_ID1;
        AscendC::TEventID evB = (l1Id == 0u) ? EVENT_ID2 : EVENT_ID3;
        CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, evA);
        LoadGm2L1(l1A_[l1Id], gmX[k0], static_cast<int32_t>(actualM), static_cast<int32_t>(kAct), mAlign,
                  static_cast<int32_t>(kAl), static_cast<int32_t>(h));
        CHUNKLOSS_SET_FLAG(MTE2, MTE1, evA);
        CHUNKLOSS_WAIT_FLAG(MTE1, MTE2, evB);
        LoadGm2L1(l1B_[l1Id], gmW[static_cast<uint64_t>(nBase) * h + k0], static_cast<int32_t>(actualN),
                  static_cast<int32_t>(kAct), nAlign, static_cast<int32_t>(kAl), static_cast<int32_t>(h));
        CHUNKLOSS_SET_FLAG(MTE2, MTE1, evB);
    }

    AscendC::TPipe* tPipe_{nullptr};
    AscendC::TBuf<AscendC::TPosition::A1> l1Buf_;
    AscendC::TBuf<AscendC::TPosition::A2> l0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> l0bBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> l0cBuf_;
    AscendC::LocalTensor<Element> l1A_[STAGES];
    AscendC::LocalTensor<Element> l1B_[STAGES];
    AscendC::LocalTensor<Element> l0A_[STAGES];
    AscendC::LocalTensor<Element> l0B_[STAGES];
    AscendC::LocalTensor<float> l0C_[STAGES];
    uint32_t l1Id_{0};
    uint32_t l0AId_{0};
    uint32_t l0BId_{0};
    uint32_t dhCId_{0};
    int32_t eventPingPong_{0};
    bool writeToUb_{false};
    bool dhAResident_{false};
    bool dhBPreloaded_{false};
    AscendC::LocalTensor<float> ubC_;
    uint32_t ubDstStride_{0};
};

} // namespace chunkloss

#endif
