/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_grad_v2_kernel.h
 * \brief arch35 device kernel for MseLossGradV2 (broadcast paradigm):
 *          y = (predict - label) * cof * dout
 *
 *        - TBuf (no TQue) 4-slot UB pool (PHYS_NODES); perBufBytes from TilingData.
 *        - CopyIn: NDDMA DataCopy<T, ND, cfg> with on-the-fly broadcast (broadcast
 *          axis loopSrcStride = 0); RANK > 5 adds a software outer loop for the dims
 *          between the split axis and the 5-dim NDDMA window.
 *        - Compute: shared fp32 VF chain MseLossGradV2VF (Reg::Sub -> Reg::Muls(cof)
 *          -> Reg::Mul) via asc_vf_call; fp16/bf16 wrap it with Cast outside the chain.
 *        - CopyOut: dense DataCopyPad (blockLen = count * sizeof(T), tail aITail safe).
 *        - Sync: manual SetFlag/WaitFlag on TBuf pipeline crossings (MTE2_V / V_MTE2 /
 *          V_MTE3 / MTE3_MTE2); no cross-core sync (pure elementwise).
 *
 *        TilingData arrays are front-padded with delta = RANK - rank slots of
 *        shape=1/stride=0, and td->split.axis stays in the normalized effective-rank
 *        frame; this kernel shifts it by delta (splitAxisR_ = td->split.axis + delta).
 */

#ifndef MSE_LOSS_GRAD_V2_KERNEL_H_
#define MSE_LOSS_GRAD_V2_KERNEL_H_

#include <type_traits>                    // std::is_same (NEED_CAST)
#include "kernel_operator.h"              // Ascend C core framework
#include "mse_loss_grad_v2_tiling_data.h" // MseLossGradV2TilingData<RANK>, PHYS_NODES, slots
#include "mse_loss_grad_v2_tiling_key.h"  // ASCENDC_TPL_ARGS_DECL carrier (template compile path)

// ---------------------------------------------------------------------------
// Generic scheduling helpers (design/Kernel.md §3; adam_apply_one_assign
// reference implementation). Input and output share one CalcOffset copy.
// ---------------------------------------------------------------------------

// GetCoreRange — per-core flat tile range [start, end): the first coresTail
// cores take tilesMain+1 tiles, the rest take tilesMain (big-little balance,
// load difference <= 1 tile); blockIdx >= numCores yields an empty range.
__aicore__ inline void GetCoreRange(int64_t coreId, int64_t tilesMain, int64_t coresTail, int64_t& start, int64_t& end)
{
    if (coreId < coresTail) {
        start = coreId * (tilesMain + 1);
        end = start + tilesMain + 1;
    } else {
        start = coresTail * (tilesMain + 1) + (coreId - coresTail) * tilesMain;
        end = start + tilesMain;
    }
}

// GetUBSplitRange — inner chunk size of outer tile aOOff: the last outer tile
// carries the aITail remainder, all others carry the full aI.
__aicore__ inline int64_t GetUBSplitRange(int64_t aOOff, int64_t aO, int64_t aI, int64_t aITail)
{
    return (aOOff == aO - 1) ? aITail : aI;
}

// FlatToEffectiveCoord — decode a flat tile index into R-frame coordinates:
// aOOff = flat % aO addresses the split axis (coord[splitAxis] = aOOff * aI),
// outer = flat / aO decodes row-major into dims [0, splitAxis); dims after
// the split axis stay 0 (covered by the NDDMA inner transfer).
__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* maxBroShape, int64_t rank, int64_t splitAxis,
                                            int64_t aI, int64_t aO, int64_t* effCoord)
{
    for (int64_t d = 0; d < rank; d++) {
        effCoord[d] = 0;
    }
    if (aO <= 0) {
        return false;
    }
    int64_t aOOff = flat % aO;
    int64_t outer = flat / aO;
    for (int64_t d = splitAxis - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[splitAxis] = aOOff * aI;
    return true;
}

// CalcOffset — flat GM element offset from coordinates and strides
// (broadcast axes have stride 0, so their coordinate contribution vanishes).
__aicore__ inline int64_t CalcOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) {
        offset += effCoord[d] * strides[d];
    }
    return offset; // element count, index of gmIn_[]/gmOut_[]
}

// CalcTransferCount — elements to write back for one tile: split-axis segment
// (a tensor broadcast along the split axis contributes 1) times the product of
// its inner dims.
__aicore__ inline int64_t CalcTransferCount(const int64_t* normalShape, int64_t rank, int64_t splitAxis, int64_t aISeg)
{
    int64_t splitElems = (normalShape[splitAxis] == 1) ? 1 : aISeg;
    int64_t innerElems = 1;
    for (int64_t d = splitAxis + 1; d < rank; d++) {
        innerElems *= normalShape[d];
    }
    return splitElems * innerElems;
}

// ---------------------------------------------------------------------------
// MseLossGradV2VF — cross-branch shared VF chain (design/Kernel.md §9).
// fp32-domain single chain Sub -> Muls(cof) -> Mul; intermediates diff/scaled
// stay in VF registers and never touch UB. C function (not a class member,
// __simd_vf__ constraint); called via asc_vf_call from both RANK templates.
//
//   dst[i] = (srcP[i] - srcL[i]) * cof * srcD[i]
//
// cof is the host-folded gradient coefficient (reduction="mean" -> 2/N,
// "none"/"sum" -> 2.0) passed by value as a register scalar.
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void MseLossGradV2VF(__ubuf__ T* dstAddr,  // fp32 result buffer (VF chain's only UB write port)
                                        __ubuf__ T* srcPAddr, // predict fp32 buffer
                                        __ubuf__ T* srcLAddr, // label fp32 buffer
                                        __ubuf__ T* srcDAddr, // dout fp32 buffer
                                        float cof,            // host-folded gradient coefficient (register scalar)
                                        uint32_t totalElems,  // element count of this tile
                                        uint16_t repeatTime)  // repeats = CeilDivision(totalElems, VL)
{
    constexpr int32_t kVlElems = 256 / static_cast<int32_t>(sizeof(T)); // float: 64 lanes / 256B
    AscendC::Reg::RegTensor<T> regP, regL, regD, midReg, dstReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems; // UpdateMask decrements it in place (by VL)
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * kVlElems; // int32 arithmetic, no uint16 overflow
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(regP, srcPAddr + off);
        AscendC::Reg::LoadAlign(regL, srcLAddr + off);
        AscendC::Reg::LoadAlign(regD, srcDAddr + off);
        AscendC::Reg::Sub(midReg, regP, regL, mask);   // mid = predict - label
        AscendC::Reg::Muls(midReg, midReg, cof, mask); // mid = mid * cof
        AscendC::Reg::Mul(dstReg, midReg, regD, mask); // dst = mid * dout
        AscendC::Reg::StoreAlign(dstAddr + off, dstReg, mask);
    }
    // Each iteration touches a different 256B window (off = i * VL): no
    // LocalMemBar needed inside this chain.
}

// ---------------------------------------------------------------------------
// class MseLossGradV2Kernel<T, RANK> — Broadcast Standard kernel template.
//   T    — DTYPE_DOUT injected by the build (float / half / bfloat16_t)
//   RANK — 4 (tilingKey 0) or 8 (tilingKey 1), from ASCENDC_TPL_SEL
// ---------------------------------------------------------------------------
template <typename T, int64_t RANK>
class MseLossGradV2Kernel {
    static constexpr int64_t MAX_RANK = 8;       // coordinate buffer size
    static constexpr int64_t MAX_NDDMA_DIMS = 5; // NDDMA hardware descriptor limit
    static constexpr int64_t ND = (RANK <= MAX_NDDMA_DIMS) ? RANK : MAX_NDDMA_DIMS;
    // fp16/bf16 compute in fp32: Cast wraps the VF chain (never fused into it)
    static constexpr bool NEED_CAST = !std::is_same<T, float>::value;
    // fp32 vector length in elements (VECTOR_REG_WIDTH 256B / 4B = 64)
    static constexpr uint32_t VL_F32 = AscendC::GetVecLen() / sizeof(float);

    // Input slot order = OpDef declaration order
    static constexpr int64_t IN_PREDICT = 0;
    static constexpr int64_t IN_LABEL = 1;
    static constexpr int64_t IN_DOUT = 2;
    static constexpr int64_t OUT_Y = 0;

    AscendC::TPipe pipe_;                                          // UB allocation + events
    const MseLossGradV2TilingData<RANK>* td_;                      // host tiling data (read-only)
    AscendC::GlobalTensor<T> gmIn_[MAX_INPUT_SLOTS];               // predict / label / dout
    AscendC::GlobalTensor<T> gmOut_[MAX_OUTPUT_SLOTS];             // y
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[PHYS_NODES];   // 4-slot TBuf pool (no TQue)
    AscendC::MultiCopyParams<T, ND> nddmaParams_[MAX_INPUT_SLOTS]; // per-input NDDMA descriptor
    int64_t nddmaDims_;                                            // dims actually described by NDDMA
    int64_t delta_;                                                // RANK - rank (front padding width)
    int64_t splitAxisR_;                                           // split axis in the R frame (+delta)
    int64_t innerCount_;                                           // prod of dims after splitAxisR_

public:
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const MseLossGradV2TilingData<RANK>* td)
    {
        td_ = td;
        // Empty-tensor / no-work short-circuit: the host zeroes TilingData for an
        // empty output, so totalTiles == 0 and perBufBytes == 0 — no UB buffer may
        // be allocated. Process() hits the matching guard and returns without
        // issuing any NDDMA/VEC instruction.
        if (td_->multicore.totalTiles <= 0) {
            return;
        }
        // R-frame mapping: TilingData arrays carry delta front padding and
        // split.axis in the normalized frame (HostTiling.md §7) — shift here.
        delta_ = RANK - td_->rank;
        splitAxisR_ = td_->split.axis + delta_;

        for (int i = 0; i < MAX_INPUT_SLOTS; i++) {
            gmIn_[i].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputs[i]));
        }
        for (int o = 0; o < MAX_OUTPUT_SLOTS; o++) {
            gmOut_[o].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputs[o]));
        }
        // 4 TBuf slots evenly split UB (perBufBytes from TilingData, host
        // measured; no hardcoded size — device UB is 248 KiB, not 256 KiB).
        for (int b = 0; b < PHYS_NODES; b++) {
            pipe_.InitBuffer(buf_[b], td_->perBufBytes);
        }

        innerCount_ = 1;
        for (int64_t d = splitAxisR_ + 1; d < RANK; d++) {
            innerCount_ *= td_->maxBroShape[d];
        }

        // Precompute NDDMA descriptors: loop index nd counts from the
        // innermost dim (d = RANK-1) toward the split axis; loopSize of the
        // split axis is patched per tile in CopyInBrc (aISeg); broadcast axes
        // keep their GM stride 0 (expanded in flight by MTE2).
        const int64_t* dstShape = td_->maxBroShape;
        const int64_t k = splitAxisR_;
        nddmaDims_ = (RANK - k <= ND) ? (RANK - k) : ND;
        for (int inp = 0; inp < MAX_INPUT_SLOTS; inp++) {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = (d == k) ? 0 : dstShape[d];
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->inputStrides[inp][d];
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
                inner *= (d == k) ? td_->split.aI : dstShape[d];
                nd++;
            }
            // Fill unused NDDMA dims with size 1 (no-op loops)
            for (; nd < ND; nd++) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = 1;
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = 0;
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
            }
        }
    }

    __aicore__ inline void Process()
    {
        // Empty-tensor / no-work short-circuit: totalTiles == 0 makes GetCoreRange
        // return the empty range and split.aO == 0 would be a modulo-by-zero below.
        if (td_->multicore.totalTiles <= 0 || td_->split.aO <= 0) {
            return;
        }
        // Event IDs fetched once, reused per type (same-type reuse is legal
        // once the previous WaitFlag has drained; 6/7 are reserved — never
        // hardcode IDs).
        event_t evMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        event_t evVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        event_t evVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        event_t evMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start = 0;
        int64_t end = 0;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tilesMain, td_->multicore.coresTail, start, end);

        int64_t coord[MAX_RANK] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t aISeg = GetUBSplitRange(flat % td_->split.aO, td_->split.aO, td_->split.aI, td_->split.aITail);
            int64_t count = aISeg * innerCount_;
            FlatToEffectiveCoord(flat, td_->maxBroShape, RANK, splitAxisR_, td_->split.aI, td_->split.aO, coord);
            // Cross-tile WAR: previous tile's CopyOut (MTE3) must drain
            // before this tile's CopyIn (MTE2) overwrites the 4-slot pool.
            // First tile has no predecessor — skip the wait.
            if (flat != start) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
            }
            if constexpr (NEED_CAST) {
                TileCast(coord, count, aISeg, evMte2V, evVMte2, evVMte3, evMte3Mte2, flat != end - 1);
            } else {
                TileFP32(coord, count, aISeg, evMte2V, evVMte3, evMte3Mte2, flat != end - 1);
            }
        }
    }

private:
    // -----------------------------------------------------------------------
    // TileFP32 — fp32 path (design DESIGN-BRANCH §5.3 T1–T5):
    //   B0 = dout, B1 = predict, B2 = label (fp32 sources), B3 = VF dst.
    // Three MTE2 writes to distinct slots need no interleaved sync; one
    // MTE2_V pair after the third CopyIn covers all three buffers.
    // -----------------------------------------------------------------------
    __aicore__ inline void TileFP32(const int64_t* coord, int64_t count, int64_t aISeg, event_t evMte2V,
                                    event_t evVMte3, event_t evMte3Mte2, bool notLast)
    {
        constexpr int64_t B0 = 0, B1 = 1, B2 = 2, B3 = 3;
        // T1–T3: NDDMA broadcast copy-in (MTE2 writes B0/B1/B2)
        CopyInBrc(coord, IN_DOUT, B0, aISeg);
        CopyInBrc(coord, IN_PREDICT, B1, aISeg);
        CopyInBrc(coord, IN_LABEL, B2, aISeg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V); // one pair covers 3 slots
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
        // T4: VF chain — cof extracted to a local (VF forbids member access)
        float cof = td_->cof;
        uint16_t repeatTime = static_cast<uint16_t>(
            AscendC::CeilDivision(static_cast<int32_t>(count), static_cast<int32_t>(VL_F32)));
        asc_vf_call<MseLossGradV2VF<float>>(
            (__ubuf__ float*)buf_[B3].Get<float>().GetPhyAddr(), (__ubuf__ float*)buf_[B1].Get<float>().GetPhyAddr(),
            (__ubuf__ float*)buf_[B2].Get<float>().GetPhyAddr(), (__ubuf__ float*)buf_[B0].Get<float>().GetPhyAddr(),
            cof, static_cast<uint32_t>(count), repeatTime);
        // T5: dense DataCopyPad write-back
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
        CopyOutOne(coord, OUT_Y, B3, aISeg);
        if (notLast) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
        }
    }

    // -----------------------------------------------------------------------
    // TileCast — fp16/bf16 path (design DESIGN-BRANCH §5.3 U1–U9):
    //   B0 = 16b staging (serially reused), B1 = predict fp32, B2 = label
    //   fp32, B3 = dout fp32; U7 reuses B0 as the fp32 VF dst; U8 reuses B1
    //   as the 16b output slot. Casts strictly wrap the VF chain:
    //   IN CAST_NONE (widening, only option), OUT CAST_RINT (nearest,
    //   ties-to-even; the only mode common to half and bfloat16).
    // -----------------------------------------------------------------------
    __aicore__ inline void TileCast(const int64_t* coord, int64_t count, int64_t aISeg, event_t evMte2V,
                                    event_t evVMte2, event_t evVMte3, event_t evMte3Mte2, bool notLast)
    {
        constexpr int64_t B0 = 0, B1 = 1, B2 = 2, B3 = 3;
        // U1: CopyIn predict(16b) -> B0 (staging)
        CopyInBrc(coord, IN_PREDICT, B0, aISeg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
        // U2: Cast B0 -> B1 (fp32)
        AscendC::Cast(buf_[B1].Get<float>(), buf_[B0].Get<T>(), AscendC::RoundMode::CAST_NONE,
                      static_cast<uint32_t>(count));
        // WAR: Cast read of B0 must drain before the next CopyIn overwrites it
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
        // U3: CopyIn label(16b) -> B0
        CopyInBrc(coord, IN_LABEL, B0, aISeg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
        // U4: Cast B0 -> B2 (fp32)
        AscendC::Cast(buf_[B2].Get<float>(), buf_[B0].Get<T>(), AscendC::RoundMode::CAST_NONE,
                      static_cast<uint32_t>(count));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVMte2);
        // U5: CopyIn dout(16b) -> B0
        CopyInBrc(coord, IN_DOUT, B0, aISeg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
        // U6: Cast B0 -> B3 (fp32)
        AscendC::Cast(buf_[B3].Get<float>(), buf_[B0].Get<T>(), AscendC::RoundMode::CAST_NONE,
                      static_cast<uint32_t>(count));
        // U7: VF chain (V pipe in-order after U6 — no extra pair needed)
        float cof = td_->cof;
        uint16_t repeatTime = static_cast<uint16_t>(
            AscendC::CeilDivision(static_cast<int32_t>(count), static_cast<int32_t>(VL_F32)));
        asc_vf_call<MseLossGradV2VF<float>>(
            (__ubuf__ float*)buf_[B0].Get<float>().GetPhyAddr(), (__ubuf__ float*)buf_[B1].Get<float>().GetPhyAddr(),
            (__ubuf__ float*)buf_[B2].Get<float>().GetPhyAddr(), (__ubuf__ float*)buf_[B3].Get<float>().GetPhyAddr(),
            cof, static_cast<uint32_t>(count), repeatTime);
        // U8: Cast B0(fp32) -> B1(16b), nearest ties-to-even round-trip
        AscendC::Cast(buf_[B1].Get<T>(), buf_[B0].Get<float>(), AscendC::RoundMode::CAST_RINT,
                      static_cast<uint32_t>(count));
        // U9: dense DataCopyPad write-back
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVMte3);
        CopyOutOne(coord, OUT_Y, B1, aISeg);
        if (notLast) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3Mte2);
        }
    }

    // -----------------------------------------------------------------------
    // CopyInBrc — GM -> UB NDDMA copy with on-the-fly broadcast (MTE2).
    // Broadcast axes carry GM stride 0: the DMA re-reads the same source
    // position while advancing the destination. UB destination is contiguous
    // (loopDstStride = running product of inner loop sizes).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInBrc(const int64_t* coord, int64_t inputIdx, int64_t slot, int64_t aISeg)
    {
        const int64_t k = splitAxisR_;
        int64_t off = CalcOffset(coord, td_->inputStrides[inputIdx], RANK);
        const int64_t* dstShape = td_->maxBroShape;

        auto params = nddmaParams_[inputIdx];
        int64_t kNd = RANK - 1 - k; // NDDMA loop index of the split axis
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == kNd) {
                params.loopInfo.loopSize[nd] = aISeg; // per-tile split-axis segment
            }
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};

        if constexpr (RANK <= MAX_NDDMA_DIMS) {
            // Single descriptor covers every dim from the split axis inward
            AscendC::DataCopy<T, ND, cfg>(buf_[slot].Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            // Dims between the split axis and the NDDMA window run in software
            // (per-tile aISeg for the split axis — the tail tile iterates less)
            AscendC::LocalTensor<T> buf = buf_[slot].Get<T>();
            int64_t outerIters = 1;
            for (int64_t d = k; d < RANK - nddmaDims_; d++) {
                outerIters *= (d == k) ? aISeg : dstShape[d];
            }
            int64_t elemBase = off;
            for (int64_t oi = 0; oi < outerIters; oi++) {
                int64_t elemAdj = 0;
                int64_t tmp = oi;
                for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? aISeg : dstShape[d];
                    elemAdj += (tmp % sz) * td_->inputStrides[inputIdx][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<T, ND, cfg>(buf[static_cast<uint32_t>(oi * inner)],
                                              gmIn_[inputIdx][elemBase + elemAdj], params);
            }
        }
    }

    // -----------------------------------------------------------------------
    // CopyOutOne — UB -> GM single-path dense DataCopyPad (MTE3): blockLen
    // counts valid bytes only, no 32B alignment requirement, the aITail tail
    // block rides the last tile. Output y is dense (maxBro result shape).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOutOne(const int64_t* coord, int64_t outputIdx, int64_t slot, int64_t aISeg)
    {
        int64_t off = CalcOffset(coord, td_->outputStrides[outputIdx], RANK);
        int64_t cnt = CalcTransferCount(td_->outputShapes[outputIdx], RANK, splitAxisR_, aISeg);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = cnt * sizeof(T); // valid bytes (count * 4 or * 2)
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].Get<T>(), extParams);
    }
};

#endif // MSE_LOSS_GRAD_V2_KERNEL_H_
