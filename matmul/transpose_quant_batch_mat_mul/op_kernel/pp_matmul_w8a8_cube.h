/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PP_MATMUL_W8A8_CUBE_H
#define PP_MATMUL_W8A8_CUBE_H

#include "kernel_operator.h"
#include "../transpose_batch_mat_mul/utils/iterator.h"
#include "../transpose_batch_mat_mul/utils/mem.h"
#include "../transpose_batch_mat_mul/utils/utils.h"

#ifndef PP_W8A8_CONSTANTS_H
#define PP_W8A8_CONSTANTS_H
namespace PpW8A8 {
constexpr unsigned L1_K = 512;
constexpr unsigned L1_K_ND = 256;
constexpr unsigned L1_SLOTS_MIN = 2;
constexpr unsigned L1_SLOTS_MAX = 8;
constexpr unsigned L0_K = 128;
constexpr unsigned L0C_SLOT_BYTES = 65536;
} // namespace PpW8A8
#endif

namespace PpMatMulNS {

// INT8 producer. CV ownership stays in the W8A8 epilogue wrapper.
template <bool B_NZ>
class PpMatmulW8A8Cube {
public:
    __aicore__ inline void Init(uint32_t batch, uint32_t k, uint32_t n, uint32_t tileM, uint32_t tileN,
                                uint32_t l0cBuffers, bool weightNz)
    {
        l0cBuffers_ = l0cBuffers;
        batch_ = batch;
        k_ = k;
        n_ = n;
        weightNz_ = weightNz;
        l1K_ = weightNz ? PpW8A8::L1_K : PpW8A8::L1_K_ND;
        uint32_t nPad = (tileN + 31U) / 32U * 32U;
        const uint32_t mAlign = (tileM + 15U) / 16U * 16U;
        const uint32_t nAlign16 = (tileN + 15U) / 16U * 16U;
        if (weightNz) {
            l0cSlotElems_ = PpW8A8::L0C_SLOT_BYTES / sizeof(int32_t);
        } else {
            l0cSlotElems_ = mAlign * nAlign16;
            const uint32_t nAlign = (n_ + 31U) / 32U * 32U;
            nPad = nPad * (l0cBuffers_ == 0 ? 1U : l0cBuffers_);
            if (nPad > nAlign) {
                nPad = nAlign;
            }
            if (l0cBuffers_ >= 4U) {
                l1K_ = 128U;
            } else if (l0cBuffers_ == 1U) {
                // 96x256: match NZ panel K so B is one 512-wide K pass per N tile.
                l1K_ = PpW8A8::L1_K;
            }
        }
        bL1Offset_ = tileM * l1K_;
        l1SlotBytes_ = (tileM + nPad) * l1K_;
        // Same rule as PpW8A8::L1Slots. Written out because cce leaves that call unresolved.
        uint32_t slots = l1SlotBytes_ == 0 ? 0 : HardwareInfo<ArchType::ASCEND_V220>::l1Size / l1SlotBytes_;
        while (slots < PpW8A8::L1_SLOTS_MIN && l1K_ > PpW8A8::L0_K) {
            l1K_ /= 2U;
            bL1Offset_ = tileM * l1K_;
            l1SlotBytes_ = (tileM + nPad) * l1K_;
            slots = l1SlotBytes_ == 0 ? 0 : HardwareInfo<ArchType::ASCEND_V220>::l1Size / l1SlotBytes_;
        }
        if (slots > PpW8A8::L1_SLOTS_MAX) {
            slots = PpW8A8::L1_SLOTS_MAX;
        }
        // Tiling already rejects infeasible L1; never store 0 (kt % l1Slots_ would fault).
        l1Slots_ = slots < 1U ? 1U : slots;
        OnChipBuffer<ArchType::ASCEND_V220> buf;
        l1_ = buf.template GetBuffer<BufferType::ASCEND_CB, int8_t>(0);
        l0a_ = buf.template GetBuffer<BufferType::ASCEND_L0A, int8_t>(0);
        l0b_ = buf.template GetBuffer<BufferType::ASCEND_L0B, int8_t>(0);
        l0c_ = buf.template GetBuffer<BufferType::ASCEND_L0C, int32_t>(0);
        SetPadding<uint64_t>(0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        for (uint32_t slot = 0; slot < l1Slots_; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
        }
        // TPipe reserves and seeds M_MTE1 IDs 0/1/2 on A2.
        // Own 4/5 so both direct and mix launch have exactly one seed.
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
    }

    __aicore__ inline void Compute(const AscendC::GlobalTensor<int8_t>& a, const AscendC::GlobalTensor<int8_t>& b,
                                   uint32_t curM, uint32_t curN, uint32_t resultSlot)
    {
        mRound_ = (curM + 15U) / 16U * 16U;
        nRound_ = (curN + 15U) / 16U * 16U;
        c_ = l0c_[resultSlot * l0cSlotElems_];
        // MMAD CHECK/SET protects each L0C unit against the previous Fixpipe
        // reader; do not wait for the entire previous result to be released.
        if (!prefetched_)
            CopyL1(a, b, 0, 0, curM, curN);
        prefetched_ = false;
        const uint32_t kTiles = (k_ - 1) / l1K_ + 1;
        // Fill the ring once. Slot 0 may already contain the cross-task
        // prefetch; every other ready event is produced exactly once here.
        for (uint32_t kt = 1; kt < kTiles && kt < l1Slots_; ++kt) {
            CopyL1(a, b, kt, kt * l1K_, curM, curN);
        }
        uint32_t l0Sequence = 0;
        uint8_t padding[4] = {0, 0, 0, 0};
        AscendC::Load3DSetFMatrixCal(1, mRound_, padding);
        for (uint32_t kt = 0; kt < kTiles; ++kt) {
            const uint32_t slot = kt % l1Slots_;
            const uint32_t remainingK = k_ - kt * l1K_;
            const uint32_t panelK = remainingK < l1K_ ? remainingK : l1K_;
            const auto a1 = l1_[slot * l1SlotBytes_];
            const auto b1 = l1_[slot * l1SlotBytes_ + bL1Offset_];
            // Refill the slot consumed on the previous iteration. Waiting
            // for its last MTE1 read never overwrites a live L1 tile.
            const uint32_t ahead = kt + l1Slots_ - 1;
            if (kt > 0 && ahead < kTiles) {
                CopyL1(a, b, ahead % l1Slots_, ahead * l1K_, curM, curN);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
            for (uint32_t ki = 0; ki < panelK; ki += PpW8A8::L0_K) {
                const uint32_t remainingPart = panelK - ki;
                const uint32_t partK = remainingPart < PpW8A8::L0_K ? remainingPart : PpW8A8::L0_K;
                const uint32_t l0Slot = l0Sequence++ % 2;
                auto a2 = l0a_[l0Slot * 32768U];
                auto b2 = l0b_[l0Slot * 32768U];
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Slot + 4);
                // Same non-transposed INT8 A Load3DV2 shape as the SDK path:
                // feature matrix width=M, channels=actual L1 K, K offset in bits 32+.
                AscendC::LoadData3DParamsV2Pro params;
                params.channelSize = panelK;
                params.extConfig = (static_cast<unsigned long long>(ki) << 32) |
                                   (static_cast<unsigned long long>(mRound_) << 16) | partK;
                AscendC::LoadData<int8_t>(a2, a1, params);
                // [K, N] NZ in L1 is (N/32, K/16, 16, 32). Transpose into L0B.
                const uint32_t l1KRound = (panelK + 15U) / 16U * 16U;
                l1_to_l0_b<ArchType::ASCEND_V220, int8_t, false, DataFormat::ZN, DataFormat::NZ>(
                    b2, b1[ki * 32U], nRound_, partK, l1KRound / 16U, 1, 1, nRound_ / 16U);
                if (ki + partK == panelK) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
                // Match CANN's unit-flag protocol: CHECK=2 for intermediate
                // accumulations, SET=3 only for the final K contribution.
                const bool lastK = kt + 1 == kTiles && ki + partK == panelK;
                AscendC::Mmad(c_, a2, b2,
                              AscendC::MmadParams(mRound_, nRound_, partK, lastK ? 3 : 2, false, kt == 0 && ki == 0));
                AfterMmad();
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Slot + 4);
            }
        }
        // Fixpipe waits for MMAD per unit; a whole-block M_FIX wait here
        // would serialize the pair and hide the fine-grained overlap.
    }

    // One GM->L1 Nd2Nz of [K, nA+nB]. Each N tile uses the same MTE1/M
    // handshake as Compute, so L0 ping-pong is never shared across MMADs.
    __aicore__ inline void ComputePair(const AscendC::GlobalTensor<int8_t>& a, const AscendC::GlobalTensor<int8_t>& b,
                                       uint32_t curM, uint32_t nA, uint32_t nB, uint32_t slotA, uint32_t slotB)
    {
        mRound_ = (curM + 15U) / 16U * 16U;
        const uint32_t nRoundA = (nA + 15U) / 16U * 16U;
        const uint32_t nRoundB = nB ? (nB + 15U) / 16U * 16U : 0;
        const uint32_t nPadA = (nA + 31U) / 32U * 32U;
        const uint32_t copyN = nA + nB;
        auto c0 = l0c_[slotA * l0cSlotElems_];
        auto c1 = l0c_[slotB * l0cSlotElems_];
        prefetched_ = false;
        CopyL1(a, b, 0, 0, curM, copyN);
        const uint32_t kTiles = (k_ - 1) / l1K_ + 1;
        for (uint32_t kt = 1; kt < kTiles && kt < l1Slots_; ++kt) {
            CopyL1(a, b, kt, kt * l1K_, curM, copyN);
        }
        uint32_t l0Sequence = 0;
        uint8_t padding[4] = {0, 0, 0, 0};
        AscendC::Load3DSetFMatrixCal(1, mRound_, padding);
        for (uint32_t kt = 0; kt < kTiles; ++kt) {
            const uint32_t slot = kt % l1Slots_;
            const uint32_t remainingK = k_ - kt * l1K_;
            const uint32_t panelK = remainingK < l1K_ ? remainingK : l1K_;
            const auto a1 = l1_[slot * l1SlotBytes_];
            const auto b1 = l1_[slot * l1SlotBytes_ + bL1Offset_];
            const uint32_t ahead = kt + l1Slots_ - 1;
            if (kt > 0 && ahead < kTiles) {
                CopyL1(a, b, ahead % l1Slots_, ahead * l1K_, curM, copyN);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
            const uint32_t l1KRound = (panelK + 15U) / 16U * 16U;
            for (uint32_t pass = 0; pass < (nB ? 2U : 1U); ++pass) {
                const uint32_t nRound = pass == 0 ? nRoundA : nRoundB;
                auto c = pass == 0 ? c0 : c1;
                for (uint32_t ki = 0; ki < panelK; ki += PpW8A8::L0_K) {
                    const uint32_t remainingPart = panelK - ki;
                    const uint32_t partK = remainingPart < PpW8A8::L0_K ? remainingPart : PpW8A8::L0_K;
                    const uint32_t l0Slot = l0Sequence++ % 2;
                    auto a2 = l0a_[l0Slot * 32768U];
                    auto b2 = l0b_[l0Slot * 32768U];
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Slot + 4);
                    AscendC::LoadData3DParamsV2Pro params;
                    params.channelSize = panelK;
                    params.extConfig = (static_cast<unsigned long long>(ki) << 32) |
                                       (static_cast<unsigned long long>(mRound_) << 16) | partK;
                    AscendC::LoadData<int8_t>(a2, a1, params);
                    const uint32_t bOff = pass == 0 ? ki * 32U : nPadA * l1KRound + ki * 32U;
                    l1_to_l0_b<ArchType::ASCEND_V220, int8_t, false, DataFormat::ZN, DataFormat::NZ>(
                        b2, b1[bOff], nRound, partK, l1KRound / 16U, 1, 1, nRound / 16U);
                    if (pass == (nB ? 1U : 0U) && ki + partK == panelK) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
                    const bool lastK = kt + 1 == kTiles && ki + partK == panelK;
                    AscendC::Mmad(
                        c, a2, b2,
                        AscendC::MmadParams(mRound_, nRound, partK, lastK ? 3 : 2, false, kt == 0 && ki == 0));
                    AfterMmad(true);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Slot + 4);
                }
            }
        }
    }

    // One GM->L1 of [K, full group N]. Each N tile gets its own L0C slot.
    __aicore__ inline void ComputeAllN(const AscendC::GlobalTensor<int8_t>& a, const AscendC::GlobalTensor<int8_t>& b,
                                       uint32_t curM, uint32_t tileN, uint32_t nTiles, uint32_t availN)
    {
        if (nTiles == 0U) {
            return;
        }
        if (nTiles > 8U) {
            nTiles = 8U;
        }
        constexpr uint32_t L0C_BYTES = 128U * 1024U;
        while (nTiles > 1U && static_cast<uint64_t>(nTiles) * l0cSlotElems_ * sizeof(int32_t) > L0C_BYTES) {
            nTiles--;
        }
        mRound_ = (curM + 15U) / 16U * 16U;
        uint32_t nCur[8];
        uint32_t nRound[8];
        uint32_t nOff[8];
        uint32_t copyN = 0;
        for (uint32_t i = 0; i < nTiles && i < 8U; ++i) {
            const uint32_t remain = availN - copyN;
            nCur[i] = remain < tileN ? remain : tileN;
            nRound[i] = (nCur[i] + 15U) / 16U * 16U;
            nOff[i] = (copyN + 31U) / 32U * 32U;
            copyN += nCur[i];
        }
        prefetched_ = false;
        CopyL1(a, b, 0, 0, curM, copyN);
        const uint32_t kTiles = (k_ - 1) / l1K_ + 1;
        for (uint32_t kt = 1; kt < kTiles && kt < l1Slots_; ++kt) {
            CopyL1(a, b, kt, kt * l1K_, curM, copyN);
        }
        uint8_t padding[4] = {0, 0, 0, 0};
        AscendC::Load3DSetFMatrixCal(1, mRound_, padding);
        for (uint32_t kt = 0; kt < kTiles; ++kt) {
            const uint32_t slot = kt % l1Slots_;
            const uint32_t remainingK = k_ - kt * l1K_;
            const uint32_t panelK = remainingK < l1K_ ? remainingK : l1K_;
            const auto a1 = l1_[slot * l1SlotBytes_];
            const auto b1 = l1_[slot * l1SlotBytes_ + bL1Offset_];
            const uint32_t ahead = kt + l1Slots_ - 1;
            if (kt > 0 && ahead < kTiles) {
                CopyL1(a, b, ahead % l1Slots_, ahead * l1K_, curM, copyN);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
            const uint32_t l1KRound = (panelK + 15U) / 16U * 16U;
            uint32_t l0Sequence = 0;
            for (uint32_t i = 0; i < nTiles && i < 8U; ++i) {
                auto c = l0c_[i * l0cSlotElems_];
                for (uint32_t ki = 0; ki < panelK; ki += PpW8A8::L0_K) {
                    const uint32_t remainingPart = panelK - ki;
                    const uint32_t partK = remainingPart < PpW8A8::L0_K ? remainingPart : PpW8A8::L0_K;
                    const uint32_t l0Slot = l0Sequence++ % 2;
                    auto a2 = l0a_[l0Slot * 32768U];
                    auto b2 = l0b_[l0Slot * 32768U];
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Slot + 4);
                    AscendC::LoadData3DParamsV2Pro params;
                    params.channelSize = panelK;
                    params.extConfig = (static_cast<unsigned long long>(ki) << 32) |
                                       (static_cast<unsigned long long>(mRound_) << 16) | partK;
                    AscendC::LoadData<int8_t>(a2, a1, params);
                    l1_to_l0_b<ArchType::ASCEND_V220, int8_t, false, DataFormat::ZN, DataFormat::NZ>(
                        b2, b1[nOff[i] * l1KRound + ki * 32U], nRound[i], partK, l1KRound / 16U, 1, 1, nRound[i] / 16U);
                    if (i + 1 == nTiles && ki + partK == panelK) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
                    const bool lastK = kt + 1 == kTiles && ki + partK == panelK;
                    AscendC::Mmad(
                        c, a2, b2,
                        AscendC::MmadParams(mRound_, nRound[i], partK, lastK ? 3 : 2, false, kt == 0 && ki == 0));
                    AfterMmad(true);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Slot + 4);
                }
            }
        }
    }

    // Issue the next task's first GM->L1 tile while the last MMAD of the
    // current task and its Fixpipe/Vector consumer are still in flight.
    __aicore__ inline void Prefetch(const AscendC::GlobalTensor<int8_t>& a, const AscendC::GlobalTensor<int8_t>& b,
                                    uint32_t curM, uint32_t curN)
    {
        CopyL1(a, b, 0, 0, curM, curN);
        prefetched_ = true;
    }

    __aicore__ inline void Store(const AscendC::GlobalTensor<int32_t>& dst, uint32_t curM, uint32_t curN,
                                 uint32_t resultSlot)
    {
        const uint32_t mRound = (curM + 15U) / 16U * 16U;
        l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, int32_t, int32_t>(dst, l0c_[resultSlot * l0cSlotElems_], curM,
                                                                           curN, mRound, curN, 3);
    }

    __aicore__ inline void End()
    {
        for (uint32_t slot = 0; slot < l1Slots_; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        // Same lifetime fence as CANN's unit-flag CubeOutBuffer::Destroy.
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

private:
    // NZ and multi-N ND (ComputePair/AllN) keep the drain. Single-N ND Compute
    // uses M_MTE1 ping-pong so L0 issues can overlap like BF16.
    __aicore__ inline void AfterMmad(bool drain = false)
    {
        if (weightNz_ || drain) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    __aicore__ inline void CopyL1(const AscendC::GlobalTensor<int8_t>& a, const AscendC::GlobalTensor<int8_t>& b,
                                  uint32_t slot, uint32_t kStart, uint32_t curM, uint32_t curN)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
        const uint32_t remainingK = k_ - kStart;
        const uint32_t copyK = remainingK < l1K_ ? remainingK : l1K_;
        auto a1 = l1_[slot * l1SlotBytes_];
        auto b1 = l1_[slot * l1SlotBytes_ + bL1Offset_];
        gm_to_l1<ArchType::ASCEND_V220, int8_t, DataFormat::ND, DataFormat::NZ>(
            a1, a[kStart], curM, (curM + 15U) / 16U * 16U, curM, copyK, copyK, static_cast<uint64_t>(batch_) * k_);
        const uint32_t kRound = (copyK + 15U) / 16U * 16U;
        const uint32_t nRound32 = (curN + 31U) / 32U * 32U;
        if (weightNz_) {
            const uint32_t kAlign = (k_ + 15U) / 16U * 16U;
            const uint32_t nAlign = (n_ + 31U) / 32U * 32U;
            gm_to_l1<ArchType::ASCEND_V220, int8_t, DataFormat::NZ, DataFormat::NZ>(
                b1, b[static_cast<unsigned long long>(kStart) * 32U], copyK, kRound, kAlign, curN, nRound32, nAlign);
        } else {
            gm_to_l1<ArchType::ASCEND_V220, int8_t, DataFormat::ND, DataFormat::NZ>(
                b1, b[static_cast<unsigned long long>(kStart) * n_], copyK, kRound, kRound, curN, nRound32, n_);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
    }

    AscendC::LocalTensor<int8_t> l1_, l0a_, l0b_;
    AscendC::LocalTensor<int32_t> l0c_, c_;
    bool prefetched_ = false;
    bool weightNz_ = true;
    uint32_t l0cBuffers_, batch_, k_, n_, l1K_, l0cSlotElems_, bL1Offset_, l1SlotBytes_, l1Slots_, mRound_, nRound_;
};
} // namespace PpMatMulNS
#endif
