/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MASKED_SCATTER_V2_H
#define MASKED_SCATTER_V2_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "masked_scatter_v2_tiling_data.h"

// ---------------------------------------------------------------------------
// MaskedScatter fused kernel (mask-driven conditional move), Ascend950PR.
//
//   out[i] = mask[i] ? source[globalPrefix(i) - 1] : self[i]
//
// Path: scatter-patterns.md §4.5.10 verified template + field-notes 3rd-round
// refinements (zero-head prefix, counting soft-sync, single-core fast path).
//
// DAV_3510 constraints baked in (all triaged on-device):
//   - raw fixed-address LocalTensors (official add_custom pattern; TPipe/TBuf
//     buffer management interacts badly with cross-chunk MTE2 on this arch)
//   - official add_custom event chain: V_MTE2 / MTE2_V / MTE3_V / V_MTE3
//   - Compare bit-stream dst is a dedicated buffer (b_ holds the zero head)
//   - Gather src in low UB; Cast T->float uses CAST_NONE
//   - SetFlag/WaitFlag require explicit eventID; budget 8/core respected
//   - no-arg SyncAll() silently broken under direct launch -> counting soft-sync
// ---------------------------------------------------------------------------

template <typename T>
class KernelMaskedScatterV2 {
public:
    // CHUNK=4096: UB budget ~205KB (fp32 worst) < 248KB (DAV_3510 UB)
    static constexpr int32_t CHUNK = 4096;
    static constexpr int32_t SEG = 64;

    // Manual event IDs (each type gets an independent slot; budget 8/core).
    static constexpr int32_t EV_MTE2V = 0; // MTE2 -> V
    static constexpr int32_t EV_VS = 1;    // V -> scalar
    static constexpr int32_t EV_SMTE3 = 2; // scalar -> MTE3
    static constexpr int32_t EV_VMTE3 = 3; // V -> MTE3
    static constexpr int32_t EV_MTE2S = 4; // MTE2 -> scalar
    static constexpr int32_t EV_SV = 5;    // scalar -> V (ramp init)
    static constexpr int32_t EV_M2M3 = 6;  // MTE2 -> MTE3
    static constexpr int32_t EV_M3M2 = 7;  // MTE3 -> MTE2

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR mask, GM_ADDR source, GM_ADDR out, GM_ADDR workspace,
                                const MaskedScatterV2TilingData* tiling)
    {
        tiling_ = *tiling;

        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), tiling_.total);
        // [FIX-BCAST] native broadcast: mask GM holds only the physical
        // (un-expanded) mask, so its extent is maskTotal, not total.
        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask),
                                tiling_.maskIsBcast != 0 ? tiling_.maskTotal : tiling_.total);
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source), tiling_.sourceLen);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), tiling_.total);
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), tiling_.coreNum * 32 + 512);

        coreId_ = AscendC::GetBlockIdx();
        myFirst_ = coreId_ * tiling_.chunksBase + (coreId_ < tiling_.chunksRem ? coreId_ : tiling_.chunksRem);
        myChunks_ = tiling_.chunksBase + (coreId_ < tiling_.chunksRem ? 1 : 0);

        // Raw fixed-address UB buffers (all 32B aligned). srcBuf_ low in UB
        // (Gather src constraint), cmpBit_ dedicated (protects b_ zero head).
        uint32_t addr = 0;
        srcBuf_ = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, addr, CHUNK * sizeof(T) + 32);
        addr += CHUNK * sizeof(T) + 32;
        maskByte_ = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECCALC, addr, CHUNK);
        addr += CHUNK;
        data_ = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, addr, CHUNK * sizeof(T));
        addr += CHUNK * sizeof(T);
        vals_ = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, addr, CHUNK * sizeof(T));
        addr += CHUNK * sizeof(T);
        a_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, addr, CHUNK * sizeof(int32_t));
        addr += CHUNK * sizeof(int32_t);
        b_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, addr, (CHUNK + SEG) * sizeof(int32_t));
        addr += (CHUNK + SEG) * sizeof(int32_t);
        c_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, addr, CHUNK * sizeof(int32_t));
        addr += CHUNK * sizeof(int32_t);
        off_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, addr, CHUNK * sizeof(int32_t));
        addr += CHUNK * sizeof(int32_t);
        ramp4_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, addr, CHUNK * sizeof(int32_t));
        addr += CHUNK * sizeof(int32_t);
        zero_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, addr, CHUNK * sizeof(int32_t));
        addr += CHUNK * sizeof(int32_t);
        cmpBit_ = AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECCALC, addr, CHUNK);
        addr += CHUNK;
        cnt_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, addr, REDUCE_TMP_BYTES);
        addr += REDUCE_TMP_BYTES;
        slot_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, addr, 32);

        // ramp4_[i] = (i + SEG) * 4 : first 64 scalar, then vector doubling.
        for (int32_t i = 0; i < SEG; i++) {
            ramp4_.SetValue(i, (i + SEG) * 4);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t m = SEG;
        while (m < CHUNK) {
            Adds(ramp4_[m], ramp4_[0], static_cast<int32_t>(m * 4), m);
            m <<= 1;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        Duplicate(zero_, 0, CHUNK);
        Duplicate(b_, 0, SEG); // persistent zero head of b_
    }

    __aicore__ inline void Process()
    {
        int32_t srcIdx = 0;
        if (tiling_.useSync != 0) {
            int32_t myCount = PhaseA();
            srcIdx = SoftSync(myCount);
        }
        int32_t offset = myFirst_ * CHUNK;
        for (int32_t c = 0; c < myChunks_; c++) {
            ProcessChunk(offset, srcIdx);
            offset += CHUNK;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr int32_t REDUCE_TMP_BYTES = 8192;

    __aicore__ inline void CopyInMaskOrData(AscendC::LocalTensor<uint8_t> ubU8, AscendC::GlobalTensor<uint8_t> gmU8,
                                            int32_t offset, int32_t nRaw)
    {
        if (nRaw == CHUNK) {
            DataCopy(ubU8, gmU8[offset], CHUNK);
        } else {
            DataCopyPad(ubU8, gmU8[offset], {1, static_cast<uint32_t>(nRaw), 0, 0, 0},
                        {true, 0, 0, static_cast<uint8_t>(0)});
        }
    }

    __aicore__ inline void CopyInData(AscendC::LocalTensor<T> ub, AscendC::GlobalTensor<T> gm, int32_t offset,
                                      int32_t nRaw)
    {
        if (nRaw == CHUNK) {
            DataCopy(ub, gm[offset], CHUNK);
        } else {
            DataCopyPad(ub, gm[offset], {1, static_cast<uint32_t>(nRaw * static_cast<int32_t>(sizeof(T))), 0, 0, 0},
                        {true, 0, 0, static_cast<T>(0)});
        }
    }

    // [FIX-BCAST] Segment-wise mask copy-in for native broadcast masks: the
    // expanded-space chunk [offset, offset+nRaw) is decomposed into contiguous
    // runs along the innermost dim (host guarantees stride 1 there), each run
    // issued as one DataCopyPad into maskByte_[done]. Same ordered-issue MTE2
    // pattern as the paired mask/self copy (single MTE2_V wait after batch).
    // Mode2 (maskIsBcast==2): the innermost expanded dim is broadcast, so every
    // run maps to ONE physical mask byte. The byte is fetched with an ALIGNED
    // 32B DataCopyPad into a staging slot at the srcBuf_ head (raw scalar GM
    // reads and unaligned DataCopyPad both hang/fault on this arch), then read
    // back via MTE2_S + GetValue and each run is expanded with a scalar
    // Duplicate (<= 16 runs per chunk; host enforces innermost expanded >= 256).
    __aicore__ inline void CopyInMaskBcastRowRep(int32_t offset, int32_t nRaw)
    {
        const int32_t rank = tiling_.maskRank;
        int32_t coords[8];
        int32_t rem = offset;
        for (int32_t d = rank - 1; d >= 0; --d) {
            coords[d] = rem % tiling_.maskSize[d];
            rem /= tiling_.maskSize[d];
        }
        int32_t offM = 0;
        for (int32_t d = 0; d + 1 < rank; ++d) {
            offM += coords[d] * tiling_.maskStride[d];
        }
        int32_t done = 0;
        int32_t nseg = 0;
        int32_t segDone[16];
        int32_t segRun[16];
        int32_t segLane[16];
        auto tmpU8 = srcBuf_.template ReinterpretCast<uint8_t>();
        while (done < nRaw && nseg < 16) {
            int32_t run = tiling_.maskSize[rank - 1] - coords[rank - 1];
            if (run > nRaw - done) {
                run = nRaw - done;
            }
            const int32_t blk = offM & ~31;
            const int32_t lane = offM - blk;
            DataCopyPad(tmpU8[nseg * 32], maskGm_[blk], {1, 32U, 0, 0, 0}, {false, 0, 0, static_cast<uint8_t>(0)});
            segDone[nseg] = done;
            segRun[nseg] = run;
            segLane[nseg] = lane;
            ++nseg;
            done += run;
            int32_t d = rank - 1;
            coords[d] += run;
            while (d > 0 && coords[d] >= tiling_.maskSize[d]) {
                coords[d] = 0;
                coords[d - 1] += 1;
                --d;
            }
            offM = 0;
            for (int32_t k = 0; k + 1 < rank; ++k) {
                offM += coords[k] * tiling_.maskStride[k];
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EV_MTE2S);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EV_MTE2S);
        for (int32_t k = 0; k < nseg; ++k) {
            uint8_t v = tmpU8.GetValue(k * 32 + segLane[k]);
            Duplicate(maskByte_[segDone[k]], v, segRun[k]);
        }
    }

    __aicore__ inline void CopyInMaskBcast(int32_t offset, int32_t nRaw)
    {
        if (tiling_.maskIsBcast == 2) {
            CopyInMaskBcastRowRep(offset, nRaw);
            return;
        }
        const int32_t rank = tiling_.maskRank;
        int32_t coords[8];
        int32_t rem = offset;
        for (int32_t d = rank - 1; d >= 0; --d) {
            coords[d] = rem % tiling_.maskSize[d];
            rem /= tiling_.maskSize[d];
        }
        int32_t offM = 0;
        for (int32_t d = 0; d < rank; ++d) {
            offM += coords[d] * tiling_.maskStride[d];
        }
        int32_t done = 0;
        while (done < nRaw) {
            int32_t run = tiling_.maskSize[rank - 1] - coords[rank - 1];
            if (run > nRaw - done) {
                run = nRaw - done;
            }
            DataCopyPad(maskByte_[done], maskGm_[offM], {1, static_cast<uint32_t>(run), 0, 0, 0},
                        {true, 0, 0, static_cast<uint8_t>(0)});
            done += run;
            int32_t d = rank - 1;
            coords[d] += run;
            while (d > 0 && coords[d] >= tiling_.maskSize[d]) {
                coords[d] = 0;
                coords[d - 1] += 1;
                --d;
            }
            offM = 0;
            for (int32_t k = 0; k < rank; ++k) {
                offM += coords[k] * tiling_.maskStride[k];
            }
        }
    }

    __aicore__ inline void CopyInMask(int32_t offset, int32_t nRaw)
    {
        if (tiling_.maskIsBcast != 0) {
            CopyInMaskBcast(offset, nRaw);
        } else {
            CopyInMaskOrData(maskByte_, maskGm_, offset, nRaw);
        }
    }

    __aicore__ inline void CopyOutChunk(int32_t offset, AscendC::LocalTensor<T> ub, int32_t nRaw)
    {
        if (nRaw == CHUNK) {
            DataCopy(outGm_[offset], ub, CHUNK);
        } else {
            DataCopyPad(outGm_[offset], ub, {1, static_cast<uint16_t>(nRaw * static_cast<int32_t>(sizeof(T))), 0, 0});
        }
    }

    // Phase A: vectorized true-count over this core's chunk segment.
    __aicore__ inline int32_t PhaseA()
    {
        auto cF = c_.ReinterpretCast<float>();
        int32_t sum = 0;
        int32_t offset = myFirst_ * CHUNK;
        for (int32_t c = 0; c < myChunks_; c++) {
            int32_t nRaw = (offset + CHUNK <= tiling_.total) ? CHUNK : (tiling_.total - offset);
            int32_t nAligned = (nRaw + SEG - 1) / SEG * SEG;
            CopyInMask(offset, nRaw);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EV_MTE2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EV_MTE2V);
            // [FIX] same pad-zone zeroing as ProcessChunk (garbage in
            // [nRaw, nAligned) would corrupt the per-core true count).
            if (nAligned > nRaw) {
                for (int32_t i = nRaw; i < nAligned; i++) {
                    maskByte_.SetValue(i, 0);
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EV_SV);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EV_SV);
            }
            // [FIX] v2-verified two-step float conversion: uint8 -> int32 (bit
            // pattern 0/1), then int32 -> float (numeric cast, yields 0.0f/1.0f).
            // Reinterpreting the int32 bitmask as float gives denormals flushed
            // to zero by ReduceSum on DAV_3510 (v2 field-note trap #6). Direct
            // Cast<float,uint8_t> faults on this arch (unsupported vconv pair),
            // hence the intermediate int32 step. c_ is free during Phase A.
            Cast(a_.ReinterpretCast<uint32_t>(), maskByte_, AscendC::RoundMode::CAST_NONE, nAligned);
            AscendC::Cast(c_.ReinterpretCast<float>(), a_, AscendC::RoundMode::CAST_NONE, nAligned);
            AscendC::ReduceSum<float, true>(cnt_, cF, cF, nAligned);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EV_VS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EV_VS);
            sum += static_cast<int32_t>(cnt_.GetValue(0));
            offset += CHUNK;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return sum;
    }

    // Phase B: cross-core sync + source-base computation.
    // [FIX-HANG] pre-restructure v2 (verified all-18-case PASS on 950PR) used the
    // official barrier-mode SyncAll<true> + counts in a dedicated 32B-aligned
    // region read with DataCopy (NOT DataCopyPad). The restructure's hand-rolled
    // DataCopyPad polling loop hit the documented visibility trap ("unaligned
    // DataCopyPad read does not observe cross-core MTE3 writes"). Layout mirrors
    // v2: ws[0, C*8) = SyncAll flags (host at::zeros), ws[C*8 + core*16] = counts.
    __aicore__ inline int32_t SoftSync(int32_t myCount)
    {
        // [FIX-R0] encode slot as count+1: workspace is host-zeroed, so a slot
        // value > 0 unambiguously means "this core's count has been written and
        // is visible". A legal count of 0 (all-False segment) reads back as 1,
        // which removes the old full-timeout spin when every segment is empty.
        slot_.SetValue(0, myCount + 1);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EV_SMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EV_SMTE3);
        AscendC::DataCopyPad(wsGm_[tiling_.coreNum * 8 + coreId_ * 16], slot_, {1, 4U, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EV_M3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EV_M3M2);

        AscendC::SyncAll<true>(wsGm_, c_, tiling_.coreNum);

        int32_t srcBase = 0;
        if (coreId_ > 0) {
            // [FIX-R0] readiness = "slot written", not "count non-zero": every
            // core writes count+1 (>= 1) into a zeroed workspace, so slot==0
            // now truly means "not yet visible to the MTE2 read path" (the
            // documented post-SyncAll visibility lag). The old semantics hung
            // 4096 rounds when preceding cores legitimately had count==0.
            bool ready = false;
            for (int32_t t = 0; t < 4096 && !ready; ++t) {
                AscendC::DataCopy(c_, wsGm_[tiling_.coreNum * 8], static_cast<uint32_t>(coreId_ * 16));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EV_MTE2S);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EV_MTE2S);
                ready = true;
                for (int32_t i = 0; i < coreId_; i++) {
                    if (c_.GetValue(i * 16) == 0) {
                        ready = false;
                        break;
                    }
                }
            }
            for (int32_t i = 0; i < coreId_; i++) {
                int32_t v = c_.GetValue(i * 16);
                // Defensive: v==0 here would mean visibility never happened
                // (pathological hardware case, not a legal count — legal zeros
                // are encoded as 1). Clamp to 0 instead of decoding to -1 so a
                // timeout can never yield a negative source index / OOB read.
                srcBase += (v > 0) ? (v - 1) : 0;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        return srcBase;
    }

    // Phase C core: one chunk of mask-driven conditional move.
    __aicore__ inline void ProcessChunk(int32_t offset, int32_t& srcIdx)
    {
        int32_t nRaw = (offset + CHUNK <= tiling_.total) ? CHUNK : (tiling_.total - offset);
        int32_t nAligned = (nRaw + SEG - 1) / SEG * SEG;

        // 1. Copy in mask + self (MTE2)
        CopyInMask(offset, nRaw);
        CopyInData(data_, selfGm_, offset, nRaw);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EV_MTE2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EV_MTE2V);
        AscendC::PipeBarrier<PIPE_ALL>(); // flush after copy-in
        // [FIX] DataCopyPad only transfers nRaw bytes + pads to the 32B block
        // boundary; mask bytes in [nRaw, nAligned) are UB garbage. Zero them
        // (after MTE2 completion) so the prefix stays P[nAligned-1]==P[nRaw-1].
        if (nAligned > nRaw) {
            for (int32_t i = nRaw; i < nAligned; i++) {
                maskByte_.SetValue(i, 0);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EV_SV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EV_SV);
        }

        // 2. mask 0/1 -> a_ (int32)
        Cast(a_.ReinterpretCast<uint32_t>(), maskByte_, AscendC::RoundMode::CAST_NONE, nAligned);

        // 3. Gather-shift inclusive prefix (zero-head refined, pure vector rounds)
        auto bData = b_[SEG];
        Copy(bData, a_, nAligned);
        int32_t step = 1;
        while (step < nAligned) {
            Adds(off_, ramp4_, static_cast<int32_t>(-step * 4), nAligned);
            if (step > SEG) {
                Max(off_, off_, zero_, nAligned);
            }
            Gather(c_, b_, off_.ReinterpretCast<uint32_t>(), 0, nAligned);
            Add(a_, bData, c_, nAligned);
            Copy(bData, a_, nAligned);
            step <<= 1;
        }

        // 4. count = P[nAligned-1] (pad zone mask=0 keeps P[nAligned-1]==P[nRaw-1])
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EV_VS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EV_VS);
        int32_t count = a_.GetValue(nAligned - 1);

        if (count == 0) {
            // Fast path: nothing to write; data_ holds self values, copy straight out.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EV_M2M3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EV_M2M3);
            CopyOutChunk(offset, data_, nRaw);
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        // 5. Source window -> srcBuf_. Aligned 32B-block read; Gather srcBaseAddr
        // absorbs the shift (sub-32B DataCopyPad blockLen faults on 3510).
        int32_t alignElems = 32 / static_cast<int32_t>(sizeof(T)); // 8 (fp32/i32) or 16 (fp16/bf16)
        int32_t alignedIdx = (srcIdx / alignElems) * alignElems;
        int32_t shiftElems = srcIdx - alignedIdx;
        int32_t elemsToCopy = shiftElems + count;
        int32_t blockLen = ((elemsToCopy + 7) / 8) * 8 * static_cast<int32_t>(sizeof(T));
        // [FIX-WIN] clamp must keep the srcBuf_ 32B tail usable: with a shifted
        // window (srcIdx unaligned) and a near-full chunk, blockLen legitimately
        // reaches CHUNK*sizeof(T)+32; clamping at CHUNK*sizeof(T) truncated the
        // window tail so Gather read uninitialized UB (constant garbage) at the
        // last 2 elements of full chunks (broadcast-mask cases were hit hardest).
        if (blockLen > CHUNK * static_cast<int32_t>(sizeof(T)) + 32) {
            blockLen = CHUNK * static_cast<int32_t>(sizeof(T)) + 32;
        }
        DataCopyPad(srcBuf_, sourceGm_[alignedIdx], {1, static_cast<uint32_t>(blockLen), 0, 0, 0},
                    {false, 0, 0, static_cast<T>(0)});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EV_MTE2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EV_MTE2V);

        // 6. idx[i] = (P[i]-1)*mask[i]*sizeof(T), fully vectorized (c_ reused).
        Cast(c_.ReinterpretCast<uint32_t>(), maskByte_, AscendC::RoundMode::CAST_NONE, nAligned);
        Mul(off_, a_, c_, nAligned);
        Sub(off_, off_, c_, nAligned);
        Muls(off_, off_, static_cast<int32_t>(sizeof(T)), nAligned);

        // 7. Gather source values by byte offset.
        Gather(vals_, srcBuf_, off_.ReinterpretCast<uint32_t>(),
               static_cast<uint32_t>(shiftElems * static_cast<int32_t>(sizeof(T))), nAligned);

        // 8. Compare bit-stream (dedicated dst: never write into b_) + Select.
        Duplicate(off_, 0, nAligned);
        Compare(cmpBit_, c_, off_, AscendC::CMPMODE::NE, nAligned);
        Select(data_, cmpBit_, vals_, data_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, nAligned);

        // 9. Copy out.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EV_VMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EV_VMTE3);
        CopyOutChunk(offset, data_, nRaw);
        srcIdx += count;
        AscendC::PipeBarrier<PIPE_ALL>(); // per-chunk flush (BufferID semantics)
    }

    AscendC::GlobalTensor<T> selfGm_;
    AscendC::GlobalTensor<uint8_t> maskGm_;
    AscendC::GlobalTensor<T> sourceGm_;
    AscendC::GlobalTensor<T> outGm_;
    AscendC::GlobalTensor<int32_t> wsGm_;

    // Raw fixed-address UB tensors (official add_custom pattern).
    AscendC::LocalTensor<T> srcBuf_, data_, vals_;
    AscendC::LocalTensor<uint8_t> maskByte_, cmpBit_;
    AscendC::LocalTensor<int32_t> a_, b_, c_, off_, ramp4_, zero_, slot_;
    AscendC::LocalTensor<float> cnt_;

    MaskedScatterV2TilingData tiling_{};
    int32_t coreId_ = 0;
    int32_t myFirst_ = 0;
    int32_t myChunks_ = 0;
};

#endif // MASKED_SCATTER_V2_H
