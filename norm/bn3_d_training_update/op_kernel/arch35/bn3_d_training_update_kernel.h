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
 * \file bn3_d_training_update_kernel.h
 * \brief
 */
#ifndef BN3_D_TRAINING_UPDATE_KERNEL_H
#define BN3_D_TRAINING_UPDATE_KERNEL_H

#pragma once

#include "kernel_operator.h"
#include "bn3_d_training_update_tiling_struct.h" // BN3DTrainingUpdateTilingData<RANK>
#include "bn3_d_training_update_struct.h"        // ASCENDC_TPL_* (RANK template)

// ============================================================
// MulAddDstVF — VF wrapper for Reg::MulAddDst (dst = src0*src1 + dst).
//   Pre-condition: dst buffer already holds the broadcast `add` values.
//   Post-condition: dst holds y = x*mult + add (in-place overwrite).
// Marked __simd_vf__ so it can be invoked via asc_vf_call<> (hardware fusion).
// Signature mirrors adam MulAddVF (proven on ascend950).
//
// Used by the channel-last (NHWC/NDHWC) streaming path: per chunk the y buffer
// is seeded with the add_b pattern (buffer-level Adds y += 0.0f — exact copy),
// then this VF computes y = x*mult_b + y in one fused call. Lives here in the
// arch35 kernel header alongside its only user (DESIGN §5.5) — a separate
// forwarding header would just duplicate the include path.
//
//   Computes y = dst = src0 * src1 + dst in TWO STEPS inside the VF:
//     tmp = round_fp32(src0 * src1)      (Reg::Mul)
//     dst = round_fp32(tmp + dst)        (Reg::Add)
//   The two-step (non-fused) sequence intentionally matches the numpy golden
//   `mul_b * xf + add_b` (Mul then Add, two roundings): a fused FMA is
//   mathematically more accurate but diverges ~1 ULP at catastrophic-
//   cancellation points (y ≈ 0), which stat_rel_err reports as a large
//   relative error. Still wrapped in __simd_vf__ + asc_vf_call (S_B4 fusion
//   per Task-24); the intermediate tmp lives in a register only.
// ============================================================
template <typename T>
__simd_vf__ inline void MulAddDstVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count,
                                    uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg, tmpReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        uint32_t remain = count - static_cast<uint32_t>(i) * oneRepeatSize;
        mask = AscendC::Reg::UpdateMask<T>(remain);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::LoadAlign(dstReg, dstAddr, aReg);
        AscendC::Reg::Mul<T>(tmpReg, srcReg0, srcReg1, mask);
        AscendC::Reg::Add<T>(dstReg, tmpReg, dstReg, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// ============================================================
// VfMulAddDst — the fused call site used by the kernel:
//   y_dst = x_src0 * mult_src1 + y_dst (per-element, in-place on dst).
//   Computes VL / repeatTimes from `count` the same way the kernel does.
// ============================================================
template <typename T>
__aicore__ inline void VfMulAddDst(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count)
{
    const uint32_t vl = AscendC::GetVecLen() / sizeof(T);
    asc_vf_call<MulAddDstVF<T>>(dstAddr, src0Addr, src1Addr, count, vl,
                                static_cast<uint16_t>(AscendC::CeilDivision(count, vl)));
}

// Segment-B pipeline event IDs (parity-split double buffering).
struct Bn3dSegBEvents {
    int32_t m2v[2];  // MTE2 -> V (x chunk landed)
    int32_t v2m[2];  // V -> MTE2 (x buffer consumed, refill ok)
    int32_t v3[2];   // V -> MTE3 (y buffer ready)
    int32_t m3m2[2]; // MTE3 -> MTE2 (y written out, x buffer reusable)
};

template <typename T, int64_t RANK>
class BN3DTrainingUpdateKernel {
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;

    AscendC::TPipe pipe_;
    const BN3DTrainingUpdateTilingData<RANK>* td_;
    AscendC::GlobalTensor<T> gmIn_[kMaxInputSlots];        // 7 GM inputs (user dtype view)
    AscendC::GlobalTensor<float> gmInF_[kMaxInputSlots];   // 7 GM inputs (fp32 view for stats)
    AscendC::GlobalTensor<T> gmOut_[kMaxOutputSlots];      // 5 GM outputs (y is user dtype)
    AscendC::GlobalTensor<float> gmOutF_[kMaxOutputSlots]; // stats outputs are fp32
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[kPhysNodes];
    AscendC::MultiCopyParams<T, ND> nddmaParamsT_[kMaxInputSlots];
    AscendC::MultiCopyParams<float, ND> nddmaParamsF_[kMaxInputSlots];

public:
    __aicore__ inline void Init(GM_ADDR inputs[kMaxInputSlots], GM_ADDR outputs[kMaxOutputSlots], GM_ADDR workspace,
                                const BN3DTrainingUpdateTilingData<RANK>* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInBrcF(const int64_t* coord, int inputIdx, int slot, int64_t a_i_seg);
    __aicore__ inline void CopyOutOneF(const int64_t* coord, int outputIdx, int slot, int64_t a_i_seg);
    // Per-spatial-point strided gather of C user-dtype elements (T) from
    // gmIn_[inputIdx] starting at gmOff, with srcStride=srcStrideElems between
    // consecutive channels, into a compact UB slot[0..C-1]. Used by the
    // segment-B sliver path (channel-last, stride 1) and the big-C defensive
    // per-point fallback (channel-major, stride = plane size). NDDMA strides
    // are element-level, so no 32B alignment is required on the GM side.
    __aicore__ inline void CopyInXPointGatherT(int inputIdx, int slot, int64_t gmOff, int64_t srcStrideElems);
    // Per-spatial-point strided scatter of C fp32 / C user-dtype elements from
    // UB slot[0..C-1] to gmOutF_[outputIdx] / gmOut_[0] starting at gmOff, with
    // dstStride=dstStrideElems between consecutive channels (NCHW). Only used
    // by the big-C defensive per-point fallback.
    __aicore__ inline void CopyOutYPointScatterF(int outputIdx, int slot, int64_t gmOff, int64_t dstStrideElems);
    __aicore__ inline void CopyOutYPointScatterT(int64_t gmOff, int slot, int64_t dstStrideElems);
    // Segment B: channel-major (NCHW/NCDHW) streaming — contiguous (n,c)-plane
    // chunks with per-plane Duplicate + seed + VfMulAddDst (the proven
    // bit-exact chain; mult/add scalars via GetValue).
    __aicore__ inline void SegmentBChannelMajor(int64_t start, int64_t end, int64_t C, int64_t S, int64_t Cpad,
                                                const Bn3dSegBEvents& ev);
    // Small-plane (S < slot cap) multi-plane serialized path: chunk = G
    // consecutive planes; per-plane Muls/Adds at 32B-aligned offsets (dense
    // layout when S%8==0, padded stride otherwise).
    __aicore__ inline void SegmentBChannelMajorSmall(int64_t start, int64_t end, int64_t C, int64_t S, int64_t Cpad,
                                                     const Bn3dSegBEvents& ev);
    // Segment B: channel-last (NHWC/NDHWC) streaming — contiguous Wc-point
    // chunks with periodic mult_b/add_b broadcast buffers (VfMulAddDst on a
    // dst seeded with add_b).
    __aicore__ inline void SegmentBChannelLast(int64_t start, int64_t end, int64_t C, int64_t Cpad, int32_t evM2V,
                                               int32_t evV3, int32_t evM3M2, const Bn3dSegBEvents& ev);
    // Defensive per-point path for C > 6552 (never hit by the test suites;
    // correctness only). step=1 (channel-major, off enumerates point bases) or
    // step=C (channel-last, off enumerates point starts); chanStride = plane
    // size (channel-major) or 1 (channel-last).
    __aicore__ inline void SegmentBFallbackPerPoint(int64_t start, int64_t end, int64_t step, int64_t chanStride,
                                                    int64_t C, int64_t Cpad, const Bn3dSegBEvents& ev);
    // ── big-C (C > 6552) helpers ─────────────────────────────────────────
    // NDDMA compact slice copies (element-granular GM addressing, so unaligned
    // offsets are safe — same pattern as CopyInXPointGatherT).
    __aicore__ inline void CopyInXSliceT(int inputIdx, int slot, int64_t gmOff, int64_t count);
    __aicore__ inline void CopyOutYSliceT(int64_t gmOff, int slot, int64_t ubOff, int64_t count);
    __aicore__ inline void CopyInStatsSlice(int inputIdx, int slot, int64_t c0, int64_t ct);
    __aicore__ inline void CopyOutStatsSlice(int outputIdx, int slot, int64_t c0, int64_t ct);
    // Chunked Segment A for C > 6552: recomputes batch_mean / save_variance on
    // C slices (cCap = per_buf/4 each) and writes the 4 (C,) statistics outputs
    // (block 0 only). No mult/add staging — the C-tiled Segment B recomputes
    // its per-tile mult/add from the (C,) inputs directly (workspace unused).
    __aicore__ inline void SegmentAOutputsBigC(int64_t C, int32_t evM2V, int32_t evV2M, int32_t evV3, int32_t evM3M2);
    // C-tiled Segment B (one tile [c0, c0+ct), ct <= per_buf/8 = 6552 so the
    // tile's mult/add fit slot0 together with the streaming buffers).
    //   channel-major: planes whose channel ch = plane%C falls in the tile.
    //   channel-last: per-point channel slice [pt*C+c0, pt*C+c0+ct).
    __aicore__ inline void ComputeMultAddTile(int64_t c0, int64_t ct, int64_t cpadT, int32_t evM2V, int32_t evV2M);
    __aicore__ inline void SegmentBChannelMajorBigC(int64_t start, int64_t end, int64_t C, int64_t S, int64_t c0,
                                                    int64_t ct, int64_t cpadT, const Bn3dSegBEvents& ev);
    __aicore__ inline void SegmentBChannelLastBigC(int64_t start, int64_t end, int64_t C, int64_t c0, int64_t ct,
                                                   int64_t cpadT, const Bn3dSegBEvents& ev);
    // Empty-batch statistics contract (tiling num_rec = 0): plain (C,)-shaped
    // copies only — split/a_i are pathological for empty shapes (N=0 → a_i=0,
    // zero inner dims → inner_count = 0) and must not be consulted.
    __aicore__ inline void SegmentAEmpty(int32_t evM2V, int32_t evV3, int32_t evM3M2);
};

// ============================================================
// Scheduling helpers
// ============================================================
__aicore__ inline void Bn3dGetCoreRange(int64_t core_id, int64_t num_cores, int64_t total_tiles, int64_t tiles_main,
                                        int64_t cores_tail, int64_t& start, int64_t& end)
{
    if (num_cores <= 0) {
        start = 0;
        end = 0;
        return;
    }
    if (core_id < cores_tail) {
        start = core_id * (tiles_main + 1);
        end = start + tiles_main + 1;
    } else {
        start = cores_tail * (tiles_main + 1) + (core_id - cores_tail) * tiles_main;
        end = start + tiles_main;
    }
}

__aicore__ inline int64_t Bn3dCalcOffset(const int64_t* eff_coord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += eff_coord[d] * strides[d];
    return offset;
}

__aicore__ inline int64_t Bn3dCalcTransferCount(const int64_t* normal_shape, int64_t rank, int64_t split_axis,
                                                int64_t a_i_seg)
{
    int64_t split_elems = (normal_shape[split_axis] == 1) ? 1 : a_i_seg;
    int64_t inner_elems = 1;
    for (int64_t d = split_axis + 1; d < rank; d++)
        inner_elems *= normal_shape[d];
    return split_elems * inner_elems;
}

// ============================================================
// Init
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::Init(GM_ADDR inputs[kMaxInputSlots],
                                                               GM_ADDR outputs[kMaxOutputSlots], GM_ADDR workspace,
                                                               const BN3DTrainingUpdateTilingData<RANK>* td)
{
    td_ = td;
    (void)workspace; // workspace unused by this kernel (big-C recomputes per-tile mult/add)
    for (int i = 0; i < kMaxInputSlots; i++) {
        gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        gmInF_[i].SetGlobalBuffer((__gm__ float*)inputs[i]);
    }
    for (int i = 0; i < kMaxOutputSlots; i++) {
        gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        gmOutF_[i].SetGlobalBuffer((__gm__ float*)outputs[i]);
    }

    for (int i = 0; i < kPhysNodes; i++)
        pipe_.InitBuffer(buf_[i], td_->per_buf_bytes);

    const int64_t* dstShape = td_->max_bro_shape;
    int64_t k = td_->split.axis;
    for (int inp = 0; inp < kMaxInputSlots; inp++) {
        int64_t inner = 1;
        int64_t nd = 0;
        for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
            int64_t sz = (d == k) ? 0 : dstShape[d];
            nddmaParamsT_[inp].loopInfo.loopSize[nd] = sz;
            nddmaParamsT_[inp].loopInfo.loopSrcStride[nd] = td_->input_strides[inp][d];
            nddmaParamsT_[inp].loopInfo.loopDstStride[nd] = inner;
            nddmaParamsT_[inp].loopInfo.loopLpSize[nd] = 0;
            nddmaParamsT_[inp].loopInfo.loopRpSize[nd] = 0;
            nddmaParamsF_[inp].loopInfo.loopSize[nd] = sz;
            nddmaParamsF_[inp].loopInfo.loopSrcStride[nd] = td_->input_strides[inp][d];
            nddmaParamsF_[inp].loopInfo.loopDstStride[nd] = inner;
            nddmaParamsF_[inp].loopInfo.loopLpSize[nd] = 0;
            nddmaParamsF_[inp].loopInfo.loopRpSize[nd] = 0;
            inner *= (d == k) ? td_->split.a_i : dstShape[d];
            nd++;
        }
        for (; nd < ND; nd++) {
            nddmaParamsT_[inp].loopInfo.loopSize[nd] = 1;
            nddmaParamsT_[inp].loopInfo.loopSrcStride[nd] = 0;
            nddmaParamsT_[inp].loopInfo.loopDstStride[nd] = inner;
            nddmaParamsT_[inp].loopInfo.loopLpSize[nd] = 0;
            nddmaParamsT_[inp].loopInfo.loopRpSize[nd] = 0;
            nddmaParamsF_[inp].loopInfo.loopSize[nd] = 1;
            nddmaParamsF_[inp].loopInfo.loopSrcStride[nd] = 0;
            nddmaParamsF_[inp].loopInfo.loopDstStride[nd] = inner;
            nddmaParamsF_[inp].loopInfo.loopLpSize[nd] = 0;
            nddmaParamsF_[inp].loopInfo.loopRpSize[nd] = 0;
        }
    }
}

// ============================================================
// CopyInBrcF — NDDMA GM→UB (broadcast via stride=0)
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyInBrcF(const int64_t* coord, int inputIdx, int slot,
                                                                     int64_t a_i_seg)
{
    int64_t k = td_->split.axis;
    int64_t off = Bn3dCalcOffset(coord, td_->input_strides[inputIdx], RANK);
    auto params = nddmaParamsF_[inputIdx];
    int64_t k_nd = RANK - 1 - k;
    int64_t inner = 1;
    for (int64_t nd = 0; nd < ND; nd++) {
        if (nd == k_nd)
            params.loopInfo.loopSize[nd] = a_i_seg;
        params.loopInfo.loopDstStride[nd] = inner;
        inner *= params.loopInfo.loopSize[nd];
    }
    // Cap the broadcast volume to the slot capacity (per_buf_bytes fp32).
    const int64_t maxVol = td_->per_buf_bytes / 4;
    while (inner > maxVol) {
        int64_t vol = 1;
        bool shrunk = false;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (!shrunk && nd >= 1 && params.loopInfo.loopSize[nd] > 1) {
                params.loopInfo.loopSize[nd] = 1;
                shrunk = true;
            }
            vol *= params.loopInfo.loopSize[nd];
        }
        inner = vol;
        if (!shrunk)
            break;
    }
    inner = 1;
    for (int64_t nd = 0; nd < ND; nd++) {
        params.loopInfo.loopDstStride[nd] = inner;
        inner *= params.loopInfo.loopSize[nd];
    }
    static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad,
                                                 false};
    if constexpr (RANK <= 5) {
        AscendC::DataCopy<float, ND, cfg>(buf_[slot].Get<float>(), gmInF_[inputIdx][off], params);
    }
}

// ============================================================
// CopyOutOneF — DataCopyPad UB→GM
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyOutOneF(const int64_t* coord, int outputIdx, int slot,
                                                                      int64_t a_i_seg)
{
    int64_t off = Bn3dCalcOffset(coord, td_->output_strides[outputIdx], RANK);
    int64_t cnt = Bn3dCalcTransferCount(td_->output_shapes[outputIdx], RANK, td_->split.axis, a_i_seg);
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = cnt * sizeof(float);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad(gmOutF_[outputIdx][off], buf_[slot].Get<float>(), extParams);
}

// ============================================================
// CopyInXPointGatherT — NDDMA strided gather
//   Reads C channel values of ONE spatial point from GM into compact UB
//   slot[0..C-1]. NDDMA strides are element-level, so no 32B alignment is
//   required on the GM side. When srcStrideElems==1 (NHWC, channels
//   contiguous) this degenerates to a compact C-element read.
//   ND=2 (one dummy dim1) — ND=1 trips a bisheng clang ICE (exit 139).
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyInXPointGatherT(int inputIdx, int slot, int64_t gmOff,
                                                                              int64_t srcStrideElems)
{
    const int64_t C = td_->C;
    AscendC::MultiCopyParams<T, 2> params;
    params.loopInfo.loopSize[0] = C;
    params.loopInfo.loopSrcStride[0] = srcStrideElems;
    params.loopInfo.loopDstStride[0] = 1;
    params.loopInfo.loopLpSize[0] = 0;
    params.loopInfo.loopRpSize[0] = 0;
    params.loopInfo.loopSize[1] = 1;
    params.loopInfo.loopSrcStride[1] = 0;
    params.loopInfo.loopDstStride[1] = C;
    params.loopInfo.loopLpSize[1] = 0;
    params.loopInfo.loopRpSize[1] = 0;
    static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad,
                                                 false};
    AscendC::DataCopy<T, 2, cfg>(buf_[slot].Get<T>(), gmIn_[inputIdx][gmOff], params);
}

// ============================================================
// CopyOutYPointScatterF / CopyOutYPointScatterT — pattern-selecting DataCopyPad
//   Writes C channel values of ONE spatial point from compact UB slot[0..C-1]
//   to GM (which may be NCHW-dense, so consecutive channels land dstStrideElems
//   elements apart). Only used by the big-C defensive per-point fallback.
//     dstStrideElems==1 → compact: 1 block of C elements (NHWC / HW==1).
//     otherwise         → per-element blocks via PaddingMode::Compact.
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyOutYPointScatterF(int outputIdx, int slot, int64_t gmOff,
                                                                                int64_t dstStrideElems)
{
    const int64_t C = td_->C;
    AscendC::DataCopyExtParams ext;
    if (dstStrideElems == 1) {
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(C * sizeof(float));
        ext.srcStride = 0;
        ext.dstStride = 0;
        AscendC::DataCopyPad(gmOutF_[outputIdx][gmOff], buf_[slot].Get<float>(), ext);
    } else {
        ext.blockCount = static_cast<uint16_t>(C);
        ext.blockLen = static_cast<uint32_t>(sizeof(float));
        ext.srcStride = 0;
        ext.dstStride = static_cast<uint32_t>((dstStrideElems - 1) * sizeof(float));
        AscendC::DataCopyPad<float, AscendC::PaddingMode::Compact>(gmOutF_[outputIdx][gmOff], buf_[slot].Get<float>(),
                                                                   ext);
    }
}

template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyOutYPointScatterT(int64_t gmOff, int slot,
                                                                                int64_t dstStrideElems)
{
    const int64_t C = td_->C;
    AscendC::DataCopyExtParams ext;
    if (dstStrideElems == 1) {
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(C * sizeof(T));
        ext.srcStride = 0;
        ext.dstStride = 0;
        AscendC::DataCopyPad(gmOut_[0][gmOff], buf_[slot].Get<T>(), ext); // OUT_Y=0
    } else {
        ext.blockCount = static_cast<uint16_t>(C);
        ext.blockLen = static_cast<uint32_t>(sizeof(T));
        ext.srcStride = 0;
        ext.dstStride = static_cast<uint32_t>((dstStrideElems - 1) * sizeof(T));
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(gmOut_[0][gmOff], buf_[slot].Get<T>(), ext); // OUT_Y=0
    }
}

// ============================================================
// NDDMA compact slice helpers (big-C path). NDDMA strides are element-level,
// so unaligned GM offsets/counts are safe (no 32B-alignment requirement on
// the GM side — same pattern as CopyInXPointGatherT). ND=2 keeps bisheng's
// ND=1 ICE away. These read/write exactly `count` elements (no pad over-run).
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyInXSliceT(int inputIdx, int slot, int64_t gmOff,
                                                                        int64_t count)
{
    AscendC::MultiCopyParams<T, 2> params;
    params.loopInfo.loopSize[0] = count;
    params.loopInfo.loopSrcStride[0] = 1;
    params.loopInfo.loopDstStride[0] = 1;
    params.loopInfo.loopLpSize[0] = 0;
    params.loopInfo.loopRpSize[0] = 0;
    params.loopInfo.loopSize[1] = 1;
    params.loopInfo.loopSrcStride[1] = 0;
    params.loopInfo.loopDstStride[1] = count;
    params.loopInfo.loopLpSize[1] = 0;
    params.loopInfo.loopRpSize[1] = 0;
    static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad,
                                                 false};
    AscendC::DataCopy<T, 2, cfg>(buf_[slot].Get<T>(), gmIn_[inputIdx][gmOff], params);
}

template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyOutYSliceT(int64_t gmOff, int slot, int64_t ubOff,
                                                                         int64_t count)
{
    // MTE3 DataCopyPad UB→GM. NDDMA has no UB→GM MultiCopyParams form, and the
    // plain blockCount=1 DataCopyPad needs 32B-aligned GM offsets — the big-C
    // path writes at arbitrary (S or C) strides, so use per-element Compact
    // blocks (blockLen = sizeof(T), srcStride/dstStride = 0) which handle any
    // alignment and write exactly `count` elements (no pad over-run).
    AscendC::DataCopyExtParams ext;
    ext.blockCount = static_cast<uint16_t>(count);
    ext.blockLen = static_cast<uint32_t>(sizeof(T));
    ext.srcStride = 0;
    ext.dstStride = 0;
    AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(gmOut_[0][gmOff],
                                                           buf_[slot].Get<T>()[static_cast<uint32_t>(ubOff)], ext);
}

template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyInStatsSlice(int inputIdx, int slot, int64_t c0,
                                                                           int64_t ct)
{
    // CopyInBrcF-style: copy the member NDDMA params (proven -O2 pattern) and
    // override every dim so dim0 = ct contiguous channels, upper dims = 1.
    auto params = nddmaParamsF_[inputIdx];
    for (int64_t nd = 0; nd < ND; ++nd) {
        params.loopInfo.loopSize[nd] = (nd == 0) ? ct : 1;
        params.loopInfo.loopSrcStride[nd] = (nd == 0) ? 1 : 0;
        params.loopInfo.loopDstStride[nd] = (nd == 0) ? 1 : ct;
        params.loopInfo.loopLpSize[nd] = 0;
        params.loopInfo.loopRpSize[nd] = 0;
    }
    static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad,
                                                 false};
    AscendC::DataCopy<float, ND, cfg>(buf_[slot].Get<float>(), gmInF_[inputIdx][c0], params);
}

template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::CopyOutStatsSlice(int outputIdx, int slot, int64_t c0,
                                                                            int64_t ct)
{
    // Stats outputs are (C,) fp32 tensors; every c0 is 32B-aligned (multiple of
    // 8), so the plain blockCount=1 DataCopyPad (same as CopyOutOneF) applies.
    AscendC::DataCopyExtParams ext;
    ext.blockCount = 1;
    ext.blockLen = static_cast<uint32_t>(ct * sizeof(float));
    ext.srcStride = 0;
    ext.dstStride = 0;
    AscendC::DataCopyPad(gmOutF_[outputIdx][c0], buf_[slot].Get<float>(), ext);
}

// ============================================================
// SegmentBChannelMajor — NCHW/NCDHW (channel_axis != RANK-1) streaming.
//   The tensor is contiguous in (n,c,s) order; every (n,c) plane is a
//   contiguous S-element run whose channel is (plane % C). Chunks are single-
//   plane segments of up to L elements; each chunk is computed with the proven
//   bit-exact chain: Cast(x)→xf, Duplicate(mb, mult), Duplicate(yf, add),
//   VfMulAddDst (Reg::Mul + Reg::Add — two RN roundings), Cast→yT, MTE3 out.
//   Serialized per chunk (the double-buffered + VF variant hangs on ascend950
//   — the VF-unit writes to the shared yf/mb buffers are not reliably ordered
//   by the available events; serialization keeps ≫10x vs the old per-point
//   path). mult/add fetched via S-pipe GetValue once per plane.
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::SegmentBChannelMajor(int64_t start, int64_t end, int64_t C,
                                                                               int64_t S, int64_t Cpad,
                                                                               const Bn3dSegBEvents& ev)
{
    constexpr bool kIsFp32 = std::is_same_v<T, float>;
    constexpr int IN_X = 0, OUT_Y = 0;
    int64_t L = td_->per_buf_bytes / 4; // 13104 fp32 = one full slot
    if constexpr (!kIsFp32) {
        // mb + yf live in slot0 after mult/add: 8Cpad + 8L <= slot.
        const int64_t cap = (td_->per_buf_bytes / 4 - 2 * Cpad) / 2;
        if (L > cap)
            L = cap;
    }
    // L must be a 32B-byte multiple (16 elems fp16/bf16, 8 fp32): the chunk
    // GM offsets at start + k*L must stay 32B-aligned for the MTE2 reads —
    // 16B-aligned offsets (e.g. L=6488 -> 12976B) silently corrupt ~1/128 of
    // the elements on ascend950.
    L = (L / (32 / static_cast<int64_t>(sizeof(T)))) * (32 / static_cast<int64_t>(sizeof(T)));

    // Buffers: slot0 = mult/add + mb (mult broadcast) + yf (add broadcast /
    // VF dst); slot1 = xb (T, L); fp16/bf16: slot3 = xf (fp32, L), slot4 = yT
    // (T, L). fp32: mb slot3, yf slot4.
    AscendC::LocalTensor<T> xb = buf_[1].Get<T>();
    AscendC::LocalTensor<T> yT = buf_[4].Get<T>();
    AscendC::LocalTensor<float> xf = buf_[3].Get<float>();
    AscendC::LocalTensor<float> mb;
    AscendC::LocalTensor<float> yf;
    if constexpr (kIsFp32) {
        mb = buf_[3].Get<float>();
        yf = buf_[4].Get<float>();
    } else {
        mb = buf_[0].Get<float>()[static_cast<uint32_t>(2 * Cpad)];
        yf = buf_[0].Get<float>()[static_cast<uint32_t>(2 * Cpad + L)];
    }
    // S-pipe scalar staging slots (aligned; tensor-src Duplicate reads src[0]).
    AscendC::LocalTensor<float> multScr = buf_[0].Get<float>()[static_cast<uint32_t>(2 * Cpad + 2 * L)];
    AscendC::LocalTensor<float> addScr = buf_[0].Get<float>()[static_cast<uint32_t>(2 * Cpad + 2 * L + 8)];
    const AscendC::LocalTensor<float> multB = buf_[0].Get<float>();
    const AscendC::LocalTensor<float> addB = buf_[0].Get<float>()[static_cast<uint32_t>(Cpad)];

    // Chunk schedule: chunks never cross plane boundaries; the first chunk
    // starts at the plane base (p0*S) — the prefix before `start` is computed
    // and written in full (identical values across cores, benign duplicate
    // writes) and keeps every chunk start at a 32B-aligned plane offset.
    int64_t plane = start / S;
    int64_t off = plane * S;
    int64_t prevPlane = -1;
    float curMult = 0.0f, curAdd = 0.0f;
    while (off < end) {
        int64_t planeEnd = (plane + 1) * S;
        if (planeEnd > end)
            planeEnd = end;
        int64_t len = planeEnd - off;
        if (len > L)
            len = L;

        AscendC::DataCopyExtParams eIn;
        eIn.blockCount = 1;
        eIn.blockLen = static_cast<uint32_t>(len * sizeof(T));
        eIn.srcStride = 0;
        eIn.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> pIn{false, 0, 0, 0};
        AscendC::DataCopyPad(xb, gmIn_[IN_X][off], eIn, pIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (plane != prevPlane) {
            curMult = multB.GetValue(static_cast<uint32_t>(plane % C));
            curAdd = addB.GetValue(static_cast<uint32_t>(plane % C));
            prevPlane = plane;
        }
        if constexpr (kIsFp32) {
            multScr.SetValue(0u, curMult);
            addScr.SetValue(0u, curAdd);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Duplicate(mb, multScr, static_cast<int32_t>(len));
            AscendC::Duplicate(yf, addScr, static_cast<int32_t>(len));
            AscendC::PipeBarrier<PIPE_V>();
            VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xb.GetPhyAddr(),
                               (__ubuf__ float*)mb.GetPhyAddr(), static_cast<uint32_t>(len));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
            AscendC::DataCopyExtParams eOut;
            eOut.blockCount = 1;
            eOut.blockLen = static_cast<uint32_t>(len * sizeof(T));
            eOut.srcStride = 0;
            eOut.dstStride = 0;
            AscendC::DataCopyPad(gmOut_[OUT_Y][off], yf, eOut);
        } else {
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(len));
            // Scalar-src Duplicate loses precision (bf16-ish broadcast); the
            // tensor-src Duplicate copies exact bits. Stage the per-plane
            // scalars through an S-pipe scratch at aligned offsets.
            multScr.SetValue(0u, curMult);
            addScr.SetValue(0u, curAdd);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Duplicate(mb, multScr, static_cast<int32_t>(len));
            AscendC::Duplicate(yf, addScr, static_cast<int32_t>(len));
            AscendC::PipeBarrier<PIPE_V>();
            VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xf.GetPhyAddr(),
                               (__ubuf__ float*)mb.GetPhyAddr(), static_cast<uint32_t>(len));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yT, yf, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(len));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
            AscendC::DataCopyExtParams eOut;
            eOut.blockCount = 1;
            eOut.blockLen = static_cast<uint32_t>(len * sizeof(T));
            eOut.srcStride = 0;
            eOut.dstStride = 0;
            AscendC::DataCopyPad(gmOut_[OUT_Y][off], yT, eOut);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);

        off += len;
        if (off == planeEnd)
            ++plane;
    }
}

// ============================================================
// SegmentBChannelMajorSmall — small-plane (S < per-buf cap) channel-major path.
//   Chunk = G consecutive (n,c) planes (one channel each). Serialized per
//   chunk (M2 in → compute → M3 out → drain), same reasoning as the
//   channel-last loop: fixed per-chunk sync on tiny single-plane chunks cost
//   ~750us on the (64,512,7,28) family (S=196), while G-plane chunks amortize
//   the sync over G planes.
//   S % (32/sizeof(T)) == 0: dense layout — one big DMA into xb, per-plane
//   scalar Muls/Adds at 32B-aligned offsets, per-plane MTE3.
//   otherwise: padded layout (S_al = roundup(S, 32/sizeof(T))) — per-plane DMA
//   into xb + p*S_al, same per-plane compute. (The DataCopyPad multi-block
//   srcStride form reads wrong data for non-32B-multiple strides, so the
//   padded read must be per-plane calls.)
//   (scalar-operand Muls/Adds: the vector-vector Mul/Add and scalar-Duplicate
//   paths on ascend950 both show bf16-ish broadcast precision issues at
//   y≈0 cancellation points; the scalar path is the closest to the golden.)
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::SegmentBChannelMajorSmall(int64_t start, int64_t end,
                                                                                    int64_t C, int64_t S, int64_t Cpad,
                                                                                    const Bn3dSegBEvents& ev)
{
    constexpr bool kIsFp32 = std::is_same_v<T, float>;
    constexpr int IN_X = 0, OUT_Y = 0;
    const int64_t slotElems = td_->per_buf_bytes / 4;
    // S_al = 32B multiple in elements (16 for fp16/bf16, 8 for fp32): the
    // padded xb/xf/yT plane stride must keep every plane base 32B-aligned for
    // the MTE2 UB dst and the V-pipe ops (errcode 80 otherwise).
    const int64_t s32 = 32 / static_cast<int64_t>(sizeof(T));
    const int64_t S_al = (S % s32 == 0) ? S : ((S + s32 - 1) / s32) * s32;
    int64_t G = slotElems / S_al; // planes per chunk

    AscendC::LocalTensor<T> xb = buf_[1].Get<T>();
    AscendC::LocalTensor<T> yT = buf_[4].Get<T>();
    AscendC::LocalTensor<float> xf = buf_[3].Get<float>();
    const AscendC::LocalTensor<float> multB = buf_[0].Get<float>();
    const AscendC::LocalTensor<float> addB = buf_[0].Get<float>()[static_cast<uint32_t>(Cpad)];

    int64_t prevPlane = -1;
    float curMult = 0.0f, curAdd = 0.0f;
    // pos starts at the FIRST PLANE'S BASE (p0*S), not at `start`: chunks and
    // the dense read/write assume plane-aligned starts. The prefix before
    // `start` (and the first/last partial planes) are computed and written in
    // full — identical values across cores, benign duplicate writes (same
    // reasoning as the channel-last slivers).
    int64_t p0 = start / S;
    int64_t pos = p0 * S;
    while (pos < end) {
        const int64_t g = ((end - pos + S - 1) / S < G) ? ((end - pos + S - 1) / S) : G;
        // ---- MTE2 in (dense: one call; padded: per plane) ----
        if (S_al == S) {
            const int64_t rlen = ((g * S) < (end - pos)) ? (g * S) : (end - pos);
            AscendC::DataCopyExtParams eIn;
            eIn.blockCount = 1;
            eIn.blockLen = static_cast<uint32_t>(rlen * sizeof(T));
            eIn.srcStride = 0;
            eIn.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> pIn{false, 0, 0, 0};
            AscendC::DataCopyPad(xb, gmIn_[IN_X][pos], eIn, pIn);
        } else {
            for (int64_t i = 0; i < g; ++i) {
                const int64_t poff = pos + i * S; // plane i base (clamped at end)
                const int64_t plen = ((poff + S) > end) ? (end - poff) : S;
                AscendC::DataCopyExtParams eIn;
                eIn.blockCount = 1;
                eIn.blockLen = static_cast<uint32_t>(plen * sizeof(T));
                eIn.srcStride = 0;
                eIn.dstStride = 0;
                AscendC::DataCopyPadExtParams<T> pIn{false, 0, 0, 0};
                AscendC::DataCopyPad(xb[static_cast<uint32_t>(i * S_al)], gmIn_[IN_X][poff], eIn, pIn);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
        AscendC::PipeBarrier<PIPE_ALL>();

        // ---- compute + MTE3 out per plane (yf is reused per plane, so the
        //      output must leave it before the next plane's seed overwrites) ----
        for (int64_t i = 0; i < g; ++i) {
            const int64_t plane = p0 + i;
            const int64_t poff = pos + i * S;
            const int64_t plen = ((poff + S) > end) ? (end - poff) : S;
            if (plane != prevPlane) {
                curMult = multB.GetValue(static_cast<uint32_t>(plane % C));
                curAdd = addB.GetValue(static_cast<uint32_t>(plane % C));
                prevPlane = plane;
            }
            if constexpr (kIsFp32) {
                AscendC::Muls(xb[static_cast<uint32_t>(i * S_al)], xb[static_cast<uint32_t>(i * S_al)], curMult,
                              static_cast<uint32_t>(plen));
                AscendC::Adds(xb[static_cast<uint32_t>(i * S_al)], xb[static_cast<uint32_t>(i * S_al)], curAdd,
                              static_cast<uint32_t>(plen));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::DataCopyExtParams eOut;
                eOut.blockCount = 1;
                eOut.blockLen = static_cast<uint32_t>(plen * sizeof(T));
                eOut.srcStride = 0;
                eOut.dstStride = 0;
                AscendC::DataCopyPad(gmOut_[OUT_Y][poff], xb[static_cast<uint32_t>(i * S_al)], eOut);
            } else {
                AscendC::Cast(xf[static_cast<uint32_t>(i * S_al)], xb[static_cast<uint32_t>(i * S_al)],
                              AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(plen));
                AscendC::Muls(xf[static_cast<uint32_t>(i * S_al)], xf[static_cast<uint32_t>(i * S_al)], curMult,
                              static_cast<uint32_t>(plen));
                AscendC::Adds(xf[static_cast<uint32_t>(i * S_al)], xf[static_cast<uint32_t>(i * S_al)], curAdd,
                              static_cast<uint32_t>(plen));
                AscendC::Cast(yT[static_cast<uint32_t>(i * S_al)], xf[static_cast<uint32_t>(i * S_al)],
                              AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(plen));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::DataCopyExtParams eOut;
                eOut.blockCount = 1;
                eOut.blockLen = static_cast<uint32_t>(plen * sizeof(T));
                eOut.srcStride = 0;
                eOut.dstStride = 0;
                AscendC::DataCopyPad(gmOut_[OUT_Y][poff], yT[static_cast<uint32_t>(i * S_al)], eOut);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);

        pos += g * S;
        p0 += g;
    }
}

// ============================================================
// SegmentBChannelLast — NHWC/NDHWC (channel_axis == RANK-1) streaming.
//   The tensor is contiguous in spatial-major order; a chunk of Wc spatial
//   points is Wc*C contiguous elements with channel pattern period C. The
//   periodic mult_b/add_b buffers are built once per core from the UB
//   mult/add arrays (C%8==0: MTE2 UB→UB 1D DataCopy loop; else S-pipe
//   GetValue/SetValue loop — the DataCopyPad GM→UB srcStride-0 broadcast
//   hangs MTE2 and the 2D-N-NDMA broadcast ICEs bisheng, so both are out).
//   Per chunk (serialized — the double-buffered variant showed non-
//   deterministic value corruption on some shapes): cast x to fp32, seed yf
//   with add_b (Adds +0 — exact), fused VfMulAddDst (Reg::Mul then Reg::Add —
//   the proven two-rounding chain), cast back to T, MTE3 out. Point-boundary
//   slivers ([start, pStart) and [pEnd, end)) use the compact per-point gather
//   + plain mult[C]/add[C]; both sliver cores compute the full point and write
//   it in full (identical values across cores — benign duplicate writes).
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::SegmentBChannelLast(int64_t start, int64_t end, int64_t C,
                                                                              int64_t Cpad, int32_t evM2V, int32_t evV3,
                                                                              int32_t evM3M2, const Bn3dSegBEvents& ev)
{
    constexpr bool kIsFp32 = std::is_same_v<T, float>;
    constexpr int IN_X = 0, OUT_Y = 0;
    const int64_t slotElems = td_->per_buf_bytes / 4;
    // Lb = multiple of lcm(C, 16): chunk byte offsets must be 32B-aligned —
    // the MTE2 DataCopyPad GM→UB faults (errcode 80, MTE address not aligned)
    // on 16B-aligned GM sources (e.g. C=28, Lb=6552 -> 13104B offsets).
    const int64_t cAl = (C % 16 == 0) ? C : (C * 16); // lcm(C, 16) (C,16 coprime cases)
    int64_t Lb = ((slotElems / 2) / cAl) * cAl;
    if (Lb < C)
        Lb = C; // at least one full point per chunk (defensive)
    // Work-adaptive chunk: the periodic mult_b/add_b build costs WcMax = Lb/C
    // replicas (S-pipe GetValue/SetValue for C%8!=0). For cores with little
    // flat work, a large Lb makes the build dominate (case00162/186/188:
    // ~125us of a 145us kernel). Shrink Lb toward ~8 chunks per core so the
    // build cost tracks the actual work; large-work cores keep a large Lb.
    // Lb stays a multiple of cAl (so chunk offsets keep the 32B/point alignment).
    {
        const int64_t flatWork = end - start;
        const int64_t LbAdapt = ((flatWork / 8 + cAl - 1) / cAl) * cAl;
        if (LbAdapt < Lb)
            Lb = LbAdapt;
        if (Lb < C)
            Lb = C;
    }
    // fp16/bf16: yf lives in slot0 after mult/add — cap so 8C + 4Lb fits.
    if constexpr (!kIsFp32) {
        const int64_t cap = slotElems - 2 * Cpad;
        if (Lb > cap)
            Lb = (cap / cAl) * cAl;
    }
    const int64_t WcMax = Lb / C;

    // Buffers: slot0 = mult + add + yf (Lb fp32 at 2*Cpad); slot1 = mult_b
    // [Lb fp32]; slot2 = add_b; slot3 = xb[2] (T, Lb each); slot4 = xf + yT[2].
    AscendC::LocalTensor<T> xb = buf_[3].Get<T>();
    AscendC::LocalTensor<T> yT[2];
    AscendC::LocalTensor<float> xf = buf_[4].Get<float>();
    if constexpr (!kIsFp32) {
        // xf occupies bytes [0, 4*Lb) = T-elems [0, 2*Lb); yT must start AFTER
        // it (byte 4*Lb = T-elem 2*Lb), else chunk k+1's cast into xf clobbers
        // yT[k] while MTE3(k) still reads it. The yT base must ALSO be 32B-
        // aligned (the fp32→T Cast dst and the MTE3 src both need it): with a
        // tiny Lb (work-adaptive clamp to Lb=C, C=1/3) byte 4*Lb = 4/12 is
        // misaligned → VECTOR_CORE_EXCEPTION (VEC_ERROR). Round the T-elem
        // offsets up to 16 (32B). yT[1] is unused today but kept aligned.
        yT[0] = buf_[4].Get<T>()[static_cast<uint32_t>(((2 * Lb + 15) / 16) * 16)];
        yT[1] = buf_[4].Get<T>()[static_cast<uint32_t>(((3 * Lb + 15) / 16) * 16)];
    }
    AscendC::LocalTensor<float> yf = buf_[0].Get<float>()[static_cast<uint32_t>(2 * Cpad)];
    const AscendC::LocalTensor<float> multB = buf_[0].Get<float>();
    const AscendC::LocalTensor<float> addB = buf_[0].Get<float>()[static_cast<uint32_t>(Cpad)];

    // ---- periodic mult_b/add_b build: mult_b[w*C + c] = mult[c] ----
    // (C % 8 == 0): MTE2 UB→UB 1D DataCopy loop (count = C fp32, 32B-aligned).
    // (C % 8 != 0): S-pipe GetValue/SetValue scalar loop (once per core, so the
    // S-pipe cost is amortized; any C — C=28/14 included). The DataCopyPad
    // GM→UB srcStride-0 broadcast and the 2D-N-NDMA broadcast both proved
    // unusable here (MTE2 hang / bisheng ICE respectively).
    {
        const AscendC::LocalTensor<float> multBB = buf_[1].Get<float>();
        const AscendC::LocalTensor<float> addBB = buf_[2].Get<float>();
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t w = 0; w < WcMax; ++w) {
            if ((C % 8) == 0) {
                AscendC::DataCopy(multBB[static_cast<uint32_t>(w * C)], multB, static_cast<uint32_t>(C));
                AscendC::DataCopy(addBB[static_cast<uint32_t>(w * C)], addB, static_cast<uint32_t>(C));
            } else {
                for (int64_t c = 0; c < C; ++c) {
                    const float mv = multB.GetValue(static_cast<uint32_t>(c));
                    multBB.SetValue(static_cast<uint64_t>(w * C + c), mv);
                    const float av = addB.GetValue(static_cast<uint32_t>(c));
                    addBB.SetValue(static_cast<uint64_t>(w * C + c), av);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Chunk boundaries are lcm(C, 16)-aligned: 16 elements (32B) for the MTE2
    // GM reads AND a whole number of points (chunk k's mult_b pattern must
    // start at channel 0 — a 16-only alignment starts mid-point for C=14 and
    // rotates the whole pattern).
    const int64_t cP = ((C % 16 == 0) ? C : (C * 16)); // lcm(C, 16)
    const int64_t pStart = (((start + C - 1) / C) * C + cP - 1) / cP * cP;
    const int64_t pEnd = (((end / C) * C) / cP) * cP;

    // ---- head sliver [start, pStart) — full points start/C .. pStart/C ----
    // (the extra aligned padding points are also written by chunk 0 — identical
    // values, benign duplicate writes).
    for (int64_t pt = start / C; pt * C < pStart; ++pt) {
        CopyInXPointGatherT(IN_X, 3, pt * C, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (kIsFp32) {
            AscendC::Adds(yf, addB, 0.0f, static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xb.GetPhyAddr(),
                               (__ubuf__ float*)multB.GetPhyAddr(), static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::DataCopyExtParams eOut;
            eOut.blockCount = 1;
            eOut.blockLen = static_cast<uint32_t>(C * sizeof(T));
            eOut.srcStride = 0;
            eOut.dstStride = 0;
            AscendC::DataCopyPad(gmOut_[OUT_Y][pt * C], yf, eOut);
        } else {
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(C));
            AscendC::Adds(yf, addB, 0.0f, static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xf.GetPhyAddr(),
                               (__ubuf__ float*)multB.GetPhyAddr(), static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yT[0], yf, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(C));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::DataCopyExtParams eOut;
            eOut.blockCount = 1;
            eOut.blockLen = static_cast<uint32_t>(C * sizeof(T));
            eOut.srcStride = 0;
            eOut.dstStride = 0;
            AscendC::DataCopyPad(gmOut_[OUT_Y][pt * C], yT[0], eOut);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
    }

    // ---- main loop: serialized Wc-point chunks ----
    // (fully serialized per chunk: the double-buffered variant of this loop
    // showed non-deterministic value corruption on some shapes — the MTE2/V/
    // MTE3 overlap of the shared xf/yf buffers is not reliably ordered by the
    // available events on ascend950; serialization is still ≫10x faster than
    // the old per-point path.)
    {
        const AscendC::LocalTensor<float> multBB = buf_[1].Get<float>();
        const AscendC::LocalTensor<float> addBB = buf_[2].Get<float>();
        const int64_t work = pEnd - pStart;
        const int64_t nn = (work + Lb - 1) / Lb;
        for (int64_t k = 0; k < nn; ++k) {
            const int64_t curOff = pStart + k * Lb;
            const int64_t curLen = (work - k * Lb < Lb) ? (work - k * Lb) : Lb;
            AscendC::DataCopyExtParams eIn;
            eIn.blockCount = 1;
            eIn.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
            eIn.srcStride = 0;
            eIn.dstStride = 0;
            AscendC::DataCopyPadExtParams<T> pIn{false, 0, 0, 0};
            AscendC::DataCopyPad(xb, gmIn_[IN_X][curOff], eIn, pIn);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (kIsFp32) {
                AscendC::Adds(yf, addBB, 0.0f, static_cast<uint32_t>(curLen));
                AscendC::PipeBarrier<PIPE_V>();
                VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xb.GetPhyAddr(),
                                   (__ubuf__ float*)multBB.GetPhyAddr(), static_cast<uint32_t>(curLen));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::DataCopyExtParams eOut;
                eOut.blockCount = 1;
                eOut.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
                eOut.srcStride = 0;
                eOut.dstStride = 0;
                AscendC::DataCopyPad(gmOut_[OUT_Y][curOff], yf, eOut);
            } else {
                AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(curLen));
                AscendC::Adds(yf, addBB, 0.0f, static_cast<uint32_t>(curLen));
                AscendC::PipeBarrier<PIPE_V>();
                VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xf.GetPhyAddr(),
                                   (__ubuf__ float*)multBB.GetPhyAddr(), static_cast<uint32_t>(curLen));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(yT[0], yf, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(curLen));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::DataCopyExtParams eOut;
                eOut.blockCount = 1;
                eOut.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
                eOut.srcStride = 0;
                eOut.dstStride = 0;
                AscendC::DataCopyPad(gmOut_[OUT_Y][curOff], yT[0], eOut);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
        }
    }

    // Final drain: last MTE3 done before the tail sliver reuses xb/xf/yf
    // (the serialized loop already drained its last chunk — no-op drain kept
    // for the tail sliver's MTE2 reuse ordering).
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);

    // ---- tail sliver [pEnd, end) — full points pEnd/C .. end/C ----
    // (chunk-last stop at pEnd leaves the aligned padding + true tail points
    // to the sliver; both writers produce identical values — benign).
    for (int64_t pt = pEnd / C; pt * C < end; ++pt) {
        CopyInXPointGatherT(IN_X, 3, pt * C, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (kIsFp32) {
            AscendC::Adds(yf, addB, 0.0f, static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xb.GetPhyAddr(),
                               (__ubuf__ float*)multB.GetPhyAddr(), static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::DataCopyExtParams eOut;
            eOut.blockCount = 1;
            eOut.blockLen = static_cast<uint32_t>(C * sizeof(T));
            eOut.srcStride = 0;
            eOut.dstStride = 0;
            AscendC::DataCopyPad(gmOut_[OUT_Y][pt * C], yf, eOut);
        } else {
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(C));
            AscendC::Adds(yf, addB, 0.0f, static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xf.GetPhyAddr(),
                               (__ubuf__ float*)multB.GetPhyAddr(), static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(yT[0], yf, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(C));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::DataCopyExtParams eOut;
            eOut.blockCount = 1;
            eOut.blockLen = static_cast<uint32_t>(C * sizeof(T));
            eOut.srcStride = 0;
            eOut.dstStride = 0;
            AscendC::DataCopyPad(gmOut_[OUT_Y][pt * C], yT[0], eOut);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
    }
}
// ============================================================
// ============================================================
// SegmentBFallbackPerPoint — defensive per-point path for C > 6552
//   (no test suite hits it; correctness only). Gathers one spatial point's C
//   channel values, computes y = x*mult + add with the plain mult[C]/add[C]
//   seed + fused VF, scatters back. step = 1 for channel-major (off enumerates
//   point bases), step = C for channel-last (off enumerates point starts).
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::SegmentBFallbackPerPoint(int64_t start, int64_t end,
                                                                                   int64_t step, int64_t chanStride,
                                                                                   int64_t C, int64_t Cpad,
                                                                                   const Bn3dSegBEvents& ev)
{
    constexpr bool kIsFp32 = std::is_same_v<T, float>;
    constexpr int IN_X = 0, OUT_Y = 0;
    AscendC::LocalTensor<T> xb = buf_[1].Get<T>();
    AscendC::LocalTensor<float> xf = buf_[3].Get<float>();
    AscendC::LocalTensor<float> yf = buf_[4].Get<float>();
    const AscendC::LocalTensor<float> multB = buf_[0].Get<float>();
    const AscendC::LocalTensor<float> addB = buf_[0].Get<float>()[static_cast<uint32_t>(Cpad)];
    for (int64_t off = start; off < end; off += step) {
        CopyInXPointGatherT(IN_X, 1, off, chanStride);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(yf, addB, 0.0f, static_cast<uint32_t>(C));
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (kIsFp32) {
            VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xb.GetPhyAddr(),
                               (__ubuf__ float*)multB.GetPhyAddr(), static_cast<uint32_t>(C));
        } else {
            AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xf.GetPhyAddr(),
                               (__ubuf__ float*)multB.GetPhyAddr(), static_cast<uint32_t>(C));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(xb, yf, AscendC::RoundMode::CAST_ROUND, static_cast<uint32_t>(C));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
        if constexpr (kIsFp32) {
            CopyOutYPointScatterF(OUT_Y, 4, off, chanStride);
        } else {
            CopyOutYPointScatterT(off, 1, chanStride);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
    }
}
// ============================================================
// SegmentAOutputsBigC — chunked Segment A for C > 6552.
//   The 4 (C,) statistics outputs (mean_out / variance_out / batch_mean /
//   batch_variance) are elementwise independent in c, so they are computed on
//   C slices of cCap = per_buf/4 channels (each intermediate (ct,) fits one UB
//   slot). Only block 0 writes the outputs (same contract as the C<=6552
//   chain). The per-tile mult/add for Segment B are NOT staged here — the
//   C-tiled Segment B recomputes them from the (C,) inputs per tile (correctness-
//   only; workspace is left unused).
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::SegmentAOutputsBigC(int64_t C, int32_t evM2V, int32_t evV2M,
                                                                              int32_t evV3, int32_t evM3M2)
{
    const int64_t cCap = td_->per_buf_bytes / 4; // 13104 fp32 per slot
    constexpr int IN_SUM = 1, IN_SQ = 2, IN_MEAN = 5, IN_VAR = 6;
    constexpr int OUT_MEAN = 1, OUT_VAR = 2, OUT_BM = 3, OUT_BV = 4;
    constexpr int UB0 = 0, UB1 = 1, UB2 = 2, UB3 = 3, UB4 = 4;

    for (int64_t c0 = 0; c0 < C; c0 += cCap) {
        const int64_t ct = (C - c0 < cCap) ? (C - c0) : cCap;

        // ── batch_mean = sum * num_rec → UB1 ──
        CopyInStatsSlice(IN_SUM, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::Muls(buf_[UB1].Get<float>(), buf_[UB0].Get<float>(), td_->num_rec, ct);

        // ── save_variance = sq*num_rec - bm² → UB4 ──
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evV2M);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evV2M);
        CopyInStatsSlice(IN_SQ, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::Muls(buf_[UB2].Get<float>(), buf_[UB0].Get<float>(), td_->num_rec, ct);          // sq*num_rec UB2
        AscendC::Mul(buf_[UB3].Get<float>(), buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), ct); // bm² UB3
        AscendC::Sub(buf_[UB4].Get<float>(), buf_[UB2].Get<float>(), buf_[UB3].Get<float>(), ct); // save_var UB4

        // ── mean_out = mean*(1-f) + bm*f → UB2 ──
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evV2M);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evV2M);
        CopyInStatsSlice(IN_MEAN, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::Muls(buf_[UB2].Get<float>(), buf_[UB0].Get<float>(), td_->one_minus_factor, ct); // mean*(1-f) UB2
        AscendC::Muls(buf_[UB3].Get<float>(), buf_[UB1].Get<float>(), td_->factor, ct);           // bm*factor UB3
        AscendC::Add(buf_[UB2].Get<float>(), buf_[UB2].Get<float>(), buf_[UB3].Get<float>(), ct); // mean_out UB2

        // ── variance_out = var*(1-f) + uv*f → UB3 ──
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evV2M);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evV2M);
        CopyInStatsSlice(IN_VAR, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::Muls(buf_[UB0].Get<float>(), buf_[UB0].Get<float>(), td_->one_minus_factor, ct); // var*(1-f) UB0
        AscendC::Muls(buf_[UB3].Get<float>(), buf_[UB4].Get<float>(), td_->bessel_scaler, ct);    // unbiased_var UB3
        AscendC::Muls(buf_[UB3].Get<float>(), buf_[UB3].Get<float>(), td_->factor, ct);           // uv*factor UB3
        AscendC::Add(buf_[UB3].Get<float>(), buf_[UB0].Get<float>(), buf_[UB3].Get<float>(), ct); // var_out UB3

        // ── stats outputs (block 0 only) ──
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
            CopyOutStatsSlice(OUT_MEAN, UB2, c0, ct);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
            CopyOutStatsSlice(OUT_VAR, UB3, c0, ct);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
            CopyOutStatsSlice(OUT_BM, UB1, c0, ct);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
            CopyOutStatsSlice(OUT_BV, UB4, c0, ct);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
        }
    }
}

// ============================================================
// ComputeMultAddTile — recompute mult[c0:c0+ct] + add[c0:c0+ct] for one big-C
//   tile from the (C,) inputs (no workspace needed). The per-channel statistics
//   chain (sum→bm, sq→save_var, sqrt/div→inv_std, mult=scale*inv_std,
//   add=offset-mult*bm) is recomputed per tile; every core computes identical
//   values so no cross-core sync is required. Result staged in slot0:
//   mult at [0:ct], add at [cpadT:cpadT+ct] (ct <= per_buf/8 so they fit).
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::ComputeMultAddTile(int64_t c0, int64_t ct, int64_t cpadT,
                                                                             int32_t evM2V, int32_t evV2M)
{
    constexpr int IN_SUM = 1, IN_SQ = 2, IN_SCALE = 3, IN_OFFSET = 4;
    constexpr int UB0 = 0, UB1 = 1, UB2 = 2, UB3 = 3, UB4 = 4;

    // ── batch_mean = sum * num_rec → UB1 ──
    CopyInStatsSlice(IN_SUM, UB0, c0, ct);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
    AscendC::Muls(buf_[UB1].Get<float>(), buf_[UB0].Get<float>(), td_->num_rec, ct);

    // ── save_variance = sq*num_rec - bm² → UB4 ──
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evV2M);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evV2M);
    CopyInStatsSlice(IN_SQ, UB0, c0, ct);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
    AscendC::Muls(buf_[UB2].Get<float>(), buf_[UB0].Get<float>(), td_->num_rec, ct);          // sq*num_rec UB2
    AscendC::Mul(buf_[UB3].Get<float>(), buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), ct); // bm² UB3
    AscendC::Sub(buf_[UB4].Get<float>(), buf_[UB2].Get<float>(), buf_[UB3].Get<float>(), ct); // save_var UB4

    // ── inv_std = 1/sqrt(save_var + eps) → UB3 ──
    AscendC::Adds(buf_[UB2].Get<float>(), buf_[UB4].Get<float>(), td_->epsilon, ct);
    static constexpr AscendC::SqrtConfig sqrt0Ulp = {AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
    AscendC::Sqrt<float, sqrt0Ulp>(buf_[UB2].Get<float>(), buf_[UB2].Get<float>(), ct);
    AscendC::Duplicate(buf_[UB3].Get<float>(), 1.0f, ct);
    static constexpr AscendC::DivConfig div0Ulp = {AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    AscendC::Div<float, div0Ulp>(buf_[UB3].Get<float>(), buf_[UB3].Get<float>(), buf_[UB2].Get<float>(),
                                 ct); // inv_std UB3

    // ── multiplier = scale * inv_std → UB3 ──
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evV2M);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evV2M);
    CopyInStatsSlice(IN_SCALE, UB0, c0, ct);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
    AscendC::Mul(buf_[UB3].Get<float>(), buf_[UB0].Get<float>(), buf_[UB3].Get<float>(), ct); // mult UB3

    // ── addend = offset - mult*bm → UB0 ──
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evV2M);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evV2M);
    CopyInStatsSlice(IN_OFFSET, UB0, c0, ct);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
    AscendC::Mul(buf_[UB2].Get<float>(), buf_[UB3].Get<float>(), buf_[UB1].Get<float>(), ct); // mult*bm UB2
    AscendC::Sub(buf_[UB0].Get<float>(), buf_[UB0].Get<float>(), buf_[UB2].Get<float>(), ct); // addend UB0

    // ── stage into slot0: add → [cpadT], mult → [0] (add first, then mult
    //      overwrites [0:ct]; cpadT >= ct so the copies never overlap) ──
    AscendC::Adds(buf_[UB0].Get<float>()[static_cast<uint32_t>(cpadT)], buf_[UB0].Get<float>(), 0.0f, ct);
    AscendC::Adds(buf_[UB0].Get<float>(), buf_[UB3].Get<float>(), 0.0f, ct);
    // V→S fence: the channel-major processing reads the staged mult/add back
    // with S-pipe GetValue — V_S is the canonical Vector→Scalar ordering.
    {
        int32_t evVS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evVS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVS);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ============================================================
// SegmentBChannelMajorBigC — one C-tile [c0, c0+ct) of the big-C channel-major
//   (NCHW/NCDHW) path. For each plane in [start/S, ceil(end/S)) whose channel
//   ch = plane%C falls in the tile, streams the plane (chunked) with the proven
//   bit-exact VF chain (Cast→Duplicate mult/add→VfMulAddDst→Cast back), reading
//   the per-plane scalar mult/add from the tile's slot0 slice. Planes outside
//   the tile are skipped — the tile loop covers every channel exactly once.
//   The C<=6552 SegmentBChannelMajor path is untouched.
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::SegmentBChannelMajorBigC(int64_t start, int64_t end,
                                                                                   int64_t C, int64_t S, int64_t c0,
                                                                                   int64_t ct, int64_t cpadT,
                                                                                   const Bn3dSegBEvents& ev)
{
    constexpr bool kIsFp32 = std::is_same_v<T, float>;
    constexpr int IN_X = 0, OUT_Y = 0;
    const int64_t slotElems = td_->per_buf_bytes / 4; // 13104 fp32 per slot
    // L = safe per-chunk element count.
    int64_t L = slotElems - 16;
    if constexpr (!kIsFp32) {
        // xb (T) + yT (T) share slot1: 2*sizeof(T)*L <= per_buf_bytes.
        const int64_t cap = slotElems / 2;
        if (L > cap)
            L = cap;
    }
    L = (L / (32 / static_cast<int64_t>(sizeof(T)))) * (32 / static_cast<int64_t>(sizeof(T)));

    AscendC::LocalTensor<T> xb = buf_[1].Get<T>();
    AscendC::LocalTensor<T> yT;
    AscendC::LocalTensor<float> xf;
    AscendC::LocalTensor<float> mb;
    AscendC::LocalTensor<float> yf;
    AscendC::LocalTensor<float> multScr;
    AscendC::LocalTensor<float> addScr;
    if constexpr (kIsFp32) {
        mb = buf_[3].Get<float>();
        yf = buf_[4].Get<float>();
    } else {
        xf = buf_[2].Get<float>();
        mb = buf_[3].Get<float>();
        yf = buf_[4].Get<float>();
        yT = buf_[1].Get<T>()[static_cast<uint32_t>(L)];
    }
    // S-pipe mult/add staging scratch in slot2 (free for fp32; unused by the
    // non-fp32 Muls/Adds branch which applies curMult/curAdd directly).
    multScr = buf_[2].Get<float>()[0];
    addScr = buf_[2].Get<float>()[8];
    // Recompute this tile's mult/add from the (C,) inputs into slot0.
    ComputeMultAddTile(c0, ct, cpadT, ev.m2v[0], ev.v2m[0]);
    const AscendC::LocalTensor<float> multT = buf_[0].Get<float>();
    const AscendC::LocalTensor<float> addT = buf_[0].Get<float>()[static_cast<uint32_t>(cpadT)];
    int64_t plane = start / S;
    int64_t off = plane * S;
    float curMult = 0.0f, curAdd = 0.0f;
    while (off < end) {
        const int64_t ch = plane % C;
        int64_t planeEnd = (plane + 1) * S;
        if (planeEnd > end)
            planeEnd = end;
        if (ch >= c0 && ch < c0 + ct) {
            curMult = multT.GetValue(static_cast<uint32_t>(ch - c0));
            curAdd = addT.GetValue(static_cast<uint32_t>(ch - c0));
            int64_t poff = off;
            while (poff < planeEnd) {
                const int64_t len = ((planeEnd - poff) < L) ? (planeEnd - poff) : L;
                CopyInXSliceT(IN_X, 1, poff, len);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
                AscendC::PipeBarrier<PIPE_ALL>();
                if constexpr (kIsFp32) {
                    multScr.SetValue(0u, curMult);
                    addScr.SetValue(0u, curAdd);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Duplicate(mb, multScr, static_cast<int32_t>(len));
                    AscendC::Duplicate(yf, addScr, static_cast<int32_t>(len));
                    AscendC::PipeBarrier<PIPE_V>();
                    VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xb.GetPhyAddr(),
                                       (__ubuf__ float*)mb.GetPhyAddr(), static_cast<uint32_t>(len));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                    CopyOutYSliceT(poff, 4, 0, len);
                } else {
                    AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(len));
                    AscendC::Muls(xf, xf, curMult, static_cast<uint32_t>(len));
                    AscendC::Adds(xf, xf, curAdd, static_cast<uint32_t>(len));
                    AscendC::Cast(yT, xf, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(len));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                    CopyOutYSliceT(poff, 1, L, len);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
                poff += len;
            }
        }
        off = planeEnd;
        ++plane;
    }
}

// ============================================================
// SegmentBChannelLastBigC — one C-tile [c0, c0+ct) of the big-C channel-last
//   (NHWC/NDHWC) path. Each spatial point's C channels are contiguous; per
//   point the tile slice [pt*C+c0, pt*C+c0+ct) is gathered (in Lc=1024 chunks),
//   computed with the periodic mult/add slices (seed yf with add, then
//   VfMulAddDst / Mul+Add), and scattered back. Whole points from start/C to
//   ceil(end/C) are processed — overlapping cores write identical values
//   (benign duplicates; ch0 stays a multiple of Lc so the VF src stays 32B
//   aligned).
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::SegmentBChannelLastBigC(int64_t start, int64_t end, int64_t C,
                                                                                  int64_t c0, int64_t ct, int64_t cpadT,
                                                                                  const Bn3dSegBEvents& ev)
{
    constexpr bool kIsFp32 = std::is_same_v<T, float>;
    constexpr int IN_X = 0, OUT_Y = 0;
    // Per-point channel slice is processed in chunks of Lc (the VF + fp16-cast
    // combo faults on the 950 at 8-element block boundaries for large counts,
    // and the fp32 VF / fp16 Mul+Add are bit-exact per element anyway).
    const int64_t Lc = 1024;
    AscendC::LocalTensor<T> xb = buf_[1].Get<T>();
    AscendC::LocalTensor<T> yT;
    AscendC::LocalTensor<float> xf;
    AscendC::LocalTensor<float> yf = buf_[4].Get<float>();
    if constexpr (!kIsFp32) {
        yT = buf_[1].Get<T>()[static_cast<uint32_t>(Lc)];
        xf = buf_[2].Get<float>();
    }
    // Recompute this tile's mult/add from the (C,) inputs into slot0.
    ComputeMultAddTile(c0, ct, cpadT, ev.m2v[0], ev.v2m[0]);
    const AscendC::LocalTensor<float> multT = buf_[0].Get<float>();
    const AscendC::LocalTensor<float> addT = buf_[0].Get<float>()[static_cast<uint32_t>(cpadT)];

    const int64_t pt0 = start / C;
    const int64_t ptEnd = (end + C - 1) / C;
    for (int64_t pt = pt0; pt < ptEnd; ++pt) {
        const int64_t pbase = pt * C + c0;
        for (int64_t ch0 = 0; ch0 < ct; ch0 += Lc) {
            const int64_t clen = (ct - ch0 < Lc) ? (ct - ch0) : Lc;
            const int64_t base = pbase + ch0;
            CopyInXSliceT(IN_X, 1, base, clen);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev.m2v[0]);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (kIsFp32) {
                AscendC::Adds(yf, addT[static_cast<uint32_t>(ch0)], 0.0f, static_cast<uint32_t>(clen));
                AscendC::PipeBarrier<PIPE_V>();
                VfMulAddDst<float>((__ubuf__ float*)yf.GetPhyAddr(), (__ubuf__ float*)xb.GetPhyAddr(),
                                   (__ubuf__ float*)multT[static_cast<uint32_t>(ch0)].GetPhyAddr(),
                                   static_cast<uint32_t>(clen));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                CopyOutYSliceT(base, 4, 0, clen);
            } else {
                AscendC::Cast(xf, xb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(clen));
                AscendC::Mul(xf, xf, multT[static_cast<uint32_t>(ch0)], static_cast<uint32_t>(clen));
                AscendC::Add(xf, xf, addT[static_cast<uint32_t>(ch0)], static_cast<uint32_t>(clen));
                AscendC::Cast(yT, xf, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(clen));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev.v3[0]);
                CopyOutYSliceT(base, 1, Lc, clen);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev.m3m2[0]);
        }
    }
}
// ============================================================
// SegmentAEmpty — empty-batch statistics contract (num_rec = 0):
//   batch_mean = sum·num_rec, batch_variance = sq·num_rec − bm²,
//   mean_out = mean·(1−f) (+f·bm = +0), variance_out = var·(1−f)
//   (+f·bv·scaler = +0). Competitor-aligned: torch CUDA native gives zero
//   save_mean for an empty reduce domain; the guarded compose extrapolates
//   to the same values. Plain (C,)-shaped copies only — the split/a_i
//   parameters are pathological for empty shapes (N=0 → a_i=0, zero inner
//   dims → inner_count=0) and must not be consulted. blockDim is clamped
//   to ≥1, so exactly one core runs here (single writer, no duplicates).
//   Chunked over C (cCap = per_buf/4) so C > 13104 cannot overflow a slot.
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::SegmentAEmpty(int32_t evM2V, int32_t evV3, int32_t evM3M2)
{
    const int64_t C = td_->C;
    const int64_t cCap = td_->per_buf_bytes / 4;
    constexpr int IN_SUM = 1, IN_SQ = 2, IN_MEAN = 5, IN_VAR = 6;
    constexpr int OUT_MEAN = 1, OUT_VAR = 2, OUT_BM = 3, OUT_BV = 4;
    constexpr int UB0 = 0, UB1 = 1, UB2 = 2;
    for (int64_t c0 = 0; c0 < C; c0 += cCap) {
        const int64_t ct = (C - c0 < cCap) ? (C - c0) : cCap;
        // batch_mean = sum · num_rec  (num_rec = 0 from the tiling) → UB0.
        CopyInStatsSlice(IN_SUM, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::Muls(buf_[UB0].Get<float>(), buf_[UB0].Get<float>(), td_->num_rec, ct);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
        CopyOutStatsSlice(OUT_BM, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);

        // batch_variance = sq·num_rec − bm²  → UB1 (bm kept in UB0).
        CopyInStatsSlice(IN_SQ, UB1, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::Muls(buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), td_->num_rec, ct);
        AscendC::Mul(buf_[UB2].Get<float>(), buf_[UB0].Get<float>(), buf_[UB0].Get<float>(), ct); // bm²
        AscendC::Sub(buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), buf_[UB2].Get<float>(), ct);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
        CopyOutStatsSlice(OUT_BV, UB1, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);

        // mean_out = mean·(1−f)  (bm = 0 makes the +f·bm term an exact +0) → UB0.
        CopyInStatsSlice(IN_MEAN, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::Muls(buf_[UB0].Get<float>(), buf_[UB0].Get<float>(), td_->one_minus_factor, ct);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
        CopyOutStatsSlice(OUT_MEAN, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);

        // variance_out = var·(1−f)  (uv = bv·scaler contributes +0) → UB0.
        CopyInStatsSlice(IN_VAR, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evM2V);
        AscendC::Muls(buf_[UB0].Get<float>(), buf_[UB0].Get<float>(), td_->one_minus_factor, ct);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evV3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evV3);
        CopyOutStatsSlice(OUT_VAR, UB0, c0, ct);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evM3M2);
    }
}
// ============================================================
// Process — Segment A (statistics chain, once per core) +
//           Segment B (streaming y = x*mult + add)
// ============================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingUpdateKernel<T, RANK>::Process()
{
    int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    int32_t evVtoMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
    int32_t evVtoMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    // V→S fence: Segment B reads the staged mult/add back with S-pipe GetValue;
    // the V_S event is the canonical Vector→Scalar ordering on 950 (PIPE_ALL
    // alone does not reliably fence the S-pipe reads).
    int32_t evVtoS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    // NOTE: FetchEventID returns the SAME id for repeated calls with the same
    // HardEvent (it does not mark the id occupied), so the parity-split double-
    // buffered event ids MUST come from AllocEventID, which marks each id
    // occupied and returns a distinct one. Interleaved Set/Wait pairs on one
    // shared id deadlock the 950 sync machinery.
    Bn3dSegBEvents ev;
    for (int i = 0; i < 2; ++i) {
        ev.m2v[i] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
        ev.v2m[i] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>());
        ev.v3[i] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>());
        ev.m3m2[i] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }
    // All y writes go through MTE3 DataCopyPad (compact chunk writes / per-point
    // fallback scatters), which stores to GM directly — no S-pipe SetValue and
    // no core-L1 write-back involvement, so cross-core 64B-line conflicts cannot
    // arise from stale L1 lines. Neighbouring cores may both write shared
    // boundary points (slivers) with IDENTICAL values — benign duplicates.

    // ---- flat element schedule (storage order; both layouts are fully
    //      contiguous: channel-major in (n,c,s) order, channel-last in
    //      spatial-major order) ----
    const int64_t C = td_->C;
    int64_t totalElems = 1;
    for (int64_t d = 0; d < RANK; ++d)
        totalElems *= td_->max_bro_shape[d];
    int64_t start = 0, end = 0;
    {
        const int64_t cores = static_cast<int64_t>(AscendC::GetBlockNum());
        const int64_t perCore = totalElems / cores;
        const int64_t tail = totalElems % cores;
        Bn3dGetCoreRange(static_cast<int64_t>(AscendC::GetBlockIdx()), cores, totalElems, perCore, tail, start, end);
    }
    if (end <= start) {
        // Empty batch (a non-C axis of x is 0 — no pixels at all): honour the
        // num_rec = 0 contract from the tiling — statistics outputs are still
        // written (zero statistics + (1−f)-scaled running stats); only the y
        // streaming is skipped (y has no elements). C = 0 empties every
        // output, so there is nothing to write at all.
        if (C > 0) {
            SegmentAEmpty(evMTE2toV, evVtoMTE3, evMTE3toMTE2);
        }
        return;
    }

    const int64_t Cpad = ((C + 7) / 8) * 8;
    const int64_t N = (RANK > 0 && td_->max_bro_shape[0] > 0) ? td_->max_bro_shape[0] : 1;
    const int64_t S = (C > 0) ? (totalElems / (C * N)) : totalElems; // (n,c)-plane size

    // ════════════════════ Big-C path (C > 6552) ════════════════════
    // slot0 holds mult[Cpad]+add[Cpad] — 2*Cpad*4 > per_buf_bytes once
    // C > 6552, so both Segment A and B must be C-tiled. Segment A computes
    // the 4 (C,) stats outputs on C slices; Segment B recomputes each tile's
    // mult/add from the (C,) inputs (tile width segCTile = per_buf/8 = 6552,
    // so the tile's mult+add fit slot0) and streams the planes / points whose
    // channel falls in the tile. The C<=6552 path below is untouched.
    const int64_t bigCTile = td_->per_buf_bytes / 8; // 6552
    const bool chLast = (td_->channel_axis == RANK - 1);
    // SegmentBChannelLast 的 cAl = lcm(C,16) 块起点对齐在 C*16 > slotElems/2
    // （即 C>=410 且 C%16!=0）时失效：Lb 被 0 强置/除零、第二块起 2 字节错位
    // （MTE2 DataCopyPad 需 32B 对齐）。该几何脆弱的 channel-last C 一律走
    // big-C（SegmentBChannelLastBigC 逐点×1024 通道，无对齐约束）。此条件也
    // 覆盖 C=6552（6552%16==8）——原 yf 在 slot0 边界的 OOB 特例一并规避。
    const bool chLastBroken = chLast && (C >= 410 && (C % 16) != 0);
    const bool useBigC = (C > bigCTile) || chLastBroken;
    if (useBigC) {
        SegmentAOutputsBigC(C, evMTE2toV, evVtoMTE2, evVtoMTE3, evMTE3toMTE2);
        // Drain Segment A before Segment B: non-block-0 cores end Segment A on
        // V-pipe ops (no MTE3) — a following MTE2 (ComputeMultAddTile's stat
        // read) must not overwrite UB0..UB4 while those V-pipe ops still read them.
        AscendC::PipeBarrier<PIPE_ALL>();
        const int64_t segCTile = td_->per_buf_bytes / 8; // 6552
        if (td_->channel_axis == RANK - 1) {
            for (int64_t c0 = 0; c0 < C; c0 += segCTile) {
                const int64_t ct = (C - c0 < segCTile) ? (C - c0) : segCTile;
                const int64_t cpadT = ((ct + 7) / 8) * 8;
                SegmentBChannelLastBigC(start, end, C, c0, ct, cpadT, ev);
            }
        } else {
            for (int64_t c0 = 0; c0 < C; c0 += segCTile) {
                const int64_t ct = (C - c0 < segCTile) ? (C - c0) : segCTile;
                const int64_t cpadT = ((ct + 7) / 8) * 8;
                SegmentBChannelMajorBigC(start, end, C, S, c0, ct, cpadT, ev);
            }
        }
        return;
    }

    // UB slot map.
    //   Segment A: UB0..UB4 used as scratch for the stats chain.
    //   Segment B:
    //     UB0: mult[C] fp32 at +0, add[C] fp32 at +Cpad (persistent).
    //     Channel-major (NCHW/NCDHW): x double buffer UB1/UB2 (T); fp16/bf16:
    //       xf UB3 (fp32) + yT double buffer UB4 (T).
    //     Channel-last (NHWC/NDHWC): mult_b/add_b periodic broadcast buffers
    //       UB1/UB2 (fp32, Lb elems); x double buffer UB3 (T); fp16/bf16:
    //       xf + yT double buffer in UB4.
    constexpr int UB0 = 0, UB1 = 1, UB2 = 2, UB3 = 3, UB4 = 4;
    constexpr int IN_X = 0, IN_SUM = 1, IN_SQ = 2, IN_SCALE = 3, IN_OFFSET = 4, IN_MEAN = 5, IN_VAR = 6;
    constexpr int OUT_Y = 0, OUT_MEAN = 1, OUT_VAR = 2, OUT_BM = 3, OUT_BV = 4;

    int64_t inner_count = 1;
    for (int64_t d = td_->split.axis + 1; d < RANK; d++)
        inner_count *= td_->max_bro_shape[d];
    const int64_t statCapElems = td_->per_buf_bytes / (4 * inner_count);
    const int64_t statCap = (statCapElems < 1) ? 1 : statCapElems;
    const int64_t a_i_seg_stat = (td_->split.a_i > statCap) ? statCap : td_->split.a_i;

    int64_t coord[8] = {};
    // ═══════ Segment A: (C,) statistics chain (fp32), once per core ═══════
    // ── S_A1: batch_mean = sum * num_rec  →  UB1 ──
    CopyInBrcF(coord, IN_SUM, UB0, a_i_seg_stat);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Muls(buf_[UB1].Get<float>(), buf_[UB0].Get<float>(), td_->num_rec, C); // batch_mean UB1

    // ── S_A2: sq*num_rec - bm^2 = save_variance  →  UB4 ──
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    CopyInBrcF(coord, IN_SQ, UB0, a_i_seg_stat);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Muls(buf_[UB2].Get<float>(), buf_[UB0].Get<float>(), td_->num_rec, C);          // sq*num_rec UB2
    AscendC::Mul(buf_[UB3].Get<float>(), buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), C); // bm^2 UB3
    AscendC::Sub(buf_[UB4].Get<float>(), buf_[UB2].Get<float>(), buf_[UB3].Get<float>(), C); // save_variance UB4

    // ── S_A3: inv_std = 1/sqrt(save_var + epsilon)  →  UB3 ──
    AscendC::Adds(buf_[UB2].Get<float>(), buf_[UB4].Get<float>(), td_->epsilon, C);
    static constexpr AscendC::SqrtConfig sqrt0Ulp = {AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
    AscendC::Sqrt<float, sqrt0Ulp>(buf_[UB2].Get<float>(), buf_[UB2].Get<float>(), C); // r = RN(sqrt(var+eps)) UB2
    AscendC::Duplicate(buf_[UB3].Get<float>(), 1.0f, C);                               // ones → UB3
    static constexpr AscendC::DivConfig div0Ulp = {AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    AscendC::Div<float, div0Ulp>(buf_[UB3].Get<float>(), buf_[UB3].Get<float>(), buf_[UB2].Get<float>(),
                                 C); // inv_std UB3

    // ── S_A4: multiplier = scale * inv_std  →  UB3 ──
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    CopyInBrcF(coord, IN_SCALE, UB0, a_i_seg_stat);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Mul(buf_[UB3].Get<float>(), buf_[UB0].Get<float>(), buf_[UB3].Get<float>(), C); // multiplier UB3

    // ── S_A5: addend = offset - multiplier*batch_mean  →  UB0 ──
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    CopyInBrcF(coord, IN_OFFSET, UB0, a_i_seg_stat);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Mul(buf_[UB2].Get<float>(), buf_[UB3].Get<float>(), buf_[UB1].Get<float>(), C); // mult*bm temp UB2
    AscendC::Sub(buf_[UB0].Get<float>(), buf_[UB0].Get<float>(), buf_[UB2].Get<float>(), C); // addend UB0

    // ── S_A6: unbiased_var = save_var * bessel_scaler  →  UB2 ──
    AscendC::Muls(buf_[UB2].Get<float>(), buf_[UB4].Get<float>(), td_->bessel_scaler, C); // unbiased_var UB2

    // ── S_A7: mean_out = mean*(1-f) + bm*f  →  UB4 ──
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    CopyInBrcF(coord, IN_MEAN, UB4, a_i_seg_stat);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Muls(buf_[UB4].Get<float>(), buf_[UB4].Get<float>(), td_->one_minus_factor, C); // mean*(1-f) UB4
    AscendC::Muls(buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), td_->factor, C);           // bm*factor UB1
    AscendC::Add(buf_[UB4].Get<float>(), buf_[UB4].Get<float>(), buf_[UB1].Get<float>(), C); // mean_out UB4

    // ── S_A8: variance_out = var*(1-f) + uv*f  →  UB1 ──
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
    CopyInBrcF(coord, IN_VAR, UB1, a_i_seg_stat);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Muls(buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), td_->one_minus_factor, C); // var*(1-f) UB1
    AscendC::Muls(buf_[UB2].Get<float>(), buf_[UB2].Get<float>(), td_->factor, C);           // uv*factor UB2
    AscendC::Add(buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), buf_[UB2].Get<float>(), C); // variance_out UB1

    // CopyOut mean_out / variance_out — only block 0 actually writes: with
    // the inplace prototype the outputs share the input GM slots, and a
    // multi-core write would clobber the other cores' Segment-A reads of the
    // same addresses. The V_MTE3 event stays on ALL cores so the later
    // MTE3_MTE2 waits (re-derive / stage) keep their normal semantics.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
    if (AscendC::GetBlockIdx() == 0) {
        CopyOutOneF(coord, OUT_MEAN, UB4, C);
        CopyOutOneF(coord, OUT_VAR, UB1, C);
    }

    // ── batch_mean CopyOut: re-derive from sum  →  UB1 ──
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    CopyInBrcF(coord, IN_SUM, UB1, a_i_seg_stat);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Muls(buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), td_->num_rec, C); // batch_mean UB1
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
    CopyOutOneF(coord, OUT_BM, UB1, C);

    // ── batch_variance CopyOut: biased save_variance  →  UB1 ──
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    CopyInBrcF(coord, IN_SUM, UB4, a_i_seg_stat); // re-fetch sum → UB4
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Muls(buf_[UB4].Get<float>(), buf_[UB4].Get<float>(), td_->num_rec, C); // batch_mean UB4
    CopyInBrcF(coord, IN_SQ, UB1, a_i_seg_stat);                                    // re-fetch sq → UB1
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Muls(buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), td_->num_rec, C);          // sq*num_rec UB1
    AscendC::Mul(buf_[UB2].Get<float>(), buf_[UB4].Get<float>(), buf_[UB4].Get<float>(), C); // bm^2 UB2  (UB2 free now)
    AscendC::Sub(buf_[UB1].Get<float>(), buf_[UB1].Get<float>(), buf_[UB2].Get<float>(), C); // save_variance UB1
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
    CopyOutOneF(coord, OUT_BV, UB1, C);

    // ── Stage mult(UB3) → slot0+0, add(UB0) → slot0+Cpad for segment B ──
    // add FIRST: the addend lives at slot0+0 (UB0, from S_A5) and the mult
    // copy overwrites slot0+0 afterwards (V-pipe in-order).
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    AscendC::Adds(buf_[UB0].Get<float>()[static_cast<uint32_t>(Cpad)], buf_[UB0].Get<float>(), 0.0f,
                  C);                                                       // add → slot0+Cpad
    AscendC::Adds(buf_[UB0].Get<float>(), buf_[UB3].Get<float>(), 0.0f, C); // mult → slot0+0
    // V→S fence: Segment B reads mult/add back via S-pipe GetValue. The V_S
    // event is the canonical Vector→Scalar ordering on 950 (PIPE_ALL alone does
    // not reliably fence S-pipe reads — Issue 2: C=6552 boundary occasionally
    // read stale mult/add → wrong y).
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evVtoS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVtoS);
    AscendC::PipeBarrier<PIPE_ALL>();

    // ═══════ Segment B: y = x · mult + add  (streaming) ═══════
    if (C <= 6552) {
        if (td_->channel_axis == RANK - 1) {
            // NHWC / NDHWC: channel-last streaming.
            SegmentBChannelLast(start, end, C, Cpad, evMTE2toV, evVtoMTE3, evMTE3toMTE2, ev);
        } else {
            // NCHW / NCDHW: channel-major streaming. Small planes (S < slot
            // cap) use the multi-plane serialized path — single-plane chunks
            // cost ~750us on the (64,512,7,28) family (S=196).
            const int64_t slotCap = td_->per_buf_bytes / 4;
            if (S < slotCap) {
                SegmentBChannelMajorSmall(start, end, C, S, Cpad, ev);
            } else {
                SegmentBChannelMajor(start, end, C, S, Cpad, ev);
            }
        }
    } else {
        // Big-C defensive per-point path (never hit by the test suites).
        if (td_->channel_axis == RANK - 1) {
            SegmentBFallbackPerPoint((start / C) * C, end, C, 1, C, Cpad, ev);
        } else {
            SegmentBFallbackPerPoint(start, end, 1, S, C, Cpad, ev);
        }
    }
}

#endif // BN3_D_TRAINING_UPDATE_KERNEL_H
