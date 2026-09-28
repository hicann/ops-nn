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
 * Ascend C device-side kernel for LSTMBlockCellGrad on arch35 (Ascend 950).
 * Unified template class NsLSTMBlockCellGrad::LSTMBlockCellGradKernel<T,
 * USE_PEEPHOLE>, instantiated for all 4 tilingKey combinations {0, 1, 4, 5}
 * via the TPL rows in lstm_block_cell_grad_struct.h.
 *
 * Process = empty-tensor short-circuit
 *          -> peephole-output exact zero-fill (USE_PEEPHOLE=false only,
 *             per-core cell slice, via Phase0ZeroFill)
 *          -> main compute: for cTile -> for bTile double tile loop,
 *             per-row VF1..VF5 chain, tile-level VF6 peephole accumulation
 *             (peep=true only: cTile tiles THIS CORE'S column slice
 *             [cStart_, cStart_+sliceLen_) and walks the FULL batch [0, B),
 *             acc -> Cast -> direct write to the core's column slice at each
 *             cTile round end — no workspace, no cross-core merge: each
 *             core's accumulator chain walks rows 0..B-1 ascending = TF
 *             serial single-chain order)
 *
 * Multiply/evaluation order: all gate-gradient chains are evaluated in the
 * TF CPU functor's exact left-associative order (TF v2.21.0 lstm_ops.cc
 * LSTMBlockCellBpropWithEigen):
 *   dcs          = ((dig*h_grad)*o + cs_grad) + do_pre*wco
 *   cs_prev_grad = ((dcs*f) + wci*di_pre) + wcf*df_pre
 *   wci_grad     = sum_b(di_pre*cs_prev)   (row-ascending fp32)
 *   wcf_grad     = sum_b(df_pre*cs_prev)   (wco_grad multiplies cs,
 *   wco_grad     = sum_b(do_pre*cs)         NOT co)
 * FP16 instances additionally round each intermediate op result back to
 * fp16 in-register (RoundFp16InPlace) — reproduces the TF half kernel's
 * true per-operator fp16 semantics, bitwise; the FP32 path is untouched
 * (if constexpr, TF float kernel is natively fp32).
 *
 * Synchronization: single-buffer pipeline, 4 event IDs (MTE2_V / V_MTE3 /
 * V_MTE2 / MTE3_V) fetched via FetchEventID, strict Set->Wait 1:1 pairing
 * over the (j, r) flattened global round sequence; no cross-core barrier
 * anywhere (each core owns its column slice end-to-end).
 */

#ifndef LSTM_BLOCK_CELL_GRAD_KERNEL_H_
#define LSTM_BLOCK_CELL_GRAD_KERNEL_H_

#include "kernel_operator.h"
#include "lstm_block_cell_grad_tiling_data.h"

namespace NsLSTMBlockCellGrad {

using namespace AscendC;

// =============================================================================
// Shared constants (platform values via interfaces, not hard-coded —
// VECTOR_REG_WIDTH is 256B on arch35)
// =============================================================================
constexpr uint32_t VL_BYTES = AscendC::GetVecLen();    // 256B (arch35)
constexpr uint32_t REP_F32 = VL_BYTES / sizeof(float); // = 64: fp32 lanes per repeat
constexpr uint16_t REP_F32_U = static_cast<uint16_t>(REP_F32);
constexpr uint32_t BLOCK_BYTES = 32; // UB datablock size (32B)

// Widen b16->fp32 cast trait (FP16 path chain head; FP32 path is N/A —
// LoadAlign passthrough, no Cast anywhere).
constexpr AscendC::Reg::CastTrait kCastTraitB16ToF32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_NONE};

// Narrow fp32->b16 cast trait (FP16 path chain tail:
// CAST_RINT = round-to-nearest-even, NO_SAT = IEEE 754 overflow to +-Inf).
constexpr AscendC::Reg::CastTrait kCastTraitF32ToB16{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_RINT};

// B==0 empty-tensor zero-fill fixed segment width in T elements
// (2048 >= 256B minimum buffer for both fp32 (8KB) and fp16 (4KB)).
constexpr uint32_t EMPTY_ZERO_SEG_ELEMS = 2048;

// =============================================================================
// Per-op fp16 rounding: TF's half kernel computes every binary op as
// fp32-op + round-to-nearest-even back to fp16 (Eigen::half arithmetic =
// half(float(a) OP float(b))).  The FP16 path therefore rounds each
// intermediate op result back to fp16 in-register (Cast f32->b16
// NO_SAT+CAST_RINT, then exact re-widen b16->f32) before it feeds the next
// op.  The FP32 path keeps full fp32 intermediates (no rounding — the TF
// float kernel is natively fp32; if constexpr guarded).  Round-trip side
// effects on mask-outside lanes (ZEROING writes 0) are absorbed by the next
// masked op.
// __simd_callee__ register-level helper (same attribute category as the Cast
// API itself) — NOT a VF entry, called inside the VF compute chains only.
// =============================================================================
template <typename T>
__simd_callee__ inline void RoundFp16InPlace(AscendC::Reg::RegTensor<float>& val, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<T> b16Reg;
    AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, val, mask);
    AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(val, b16Reg, mask);
}

// =============================================================================
// VF implementations (__simd_vf__ free functions,
// K1: dst and all src buffers independent; loops from 0, uint16_t counters;
// no arrays, no object members, no runtime if/else — if constexpr only)
// =============================================================================

// ============ VF1: do_pre = h_grad * co * o * (1 - o)  (eq 1) ============
// Chain-head PreElewise: b16->fp32 widening fused at the load point (FP16).
// Multiply order: TF-verified evaluation order (TF v2.21.0 lstm_ops.cc
// LSTMBlockCellBpropWithEigen do_ = o*(1-o)*h_grad*co, left-associative per
// element).
template <typename T>
__simd_vf__ inline void DoPreVfImpl(__ubuf__ T* hGradUb, __ubuf__ T* coUb, __ubuf__ T* oUb, __ubuf__ float* doPreUb,
                                    uint32_t segElems, uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<T, float>;
    AscendC::Reg::RegTensor<float> hgReg;
    AscendC::Reg::RegTensor<float> coReg;
    AscendC::Reg::RegTensor<float> oReg;
    AscendC::Reg::RegTensor<float> t0Reg;
    AscendC::Reg::RegTensor<float> t1Reg;
    AscendC::Reg::RegTensor<float> oneReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::Duplicate(oneReg, 1.0f); // 1 (minuend of 1 - a)
    uint32_t remaining = segElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining); // auto-decrements VL

        if constexpr (IsFp32) { // FP32: passthrough
            AscendC::Reg::LoadAlign(hgReg, reinterpret_cast<__ubuf__ float*>(hGradUb) + off);
            AscendC::Reg::LoadAlign(coReg, reinterpret_cast<__ubuf__ float*>(coUb) + off);
            AscendC::Reg::LoadAlign(oReg, reinterpret_cast<__ubuf__ float*>(oUb) + off);
        } else { // FP16: widen b16->fp32
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, hGradUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(hgReg, b16Reg, mask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, coUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(coReg, b16Reg, mask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, oUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(oReg, b16Reg, mask);
        }

        // do_pre = ((o * (1 - o)) * h_grad) * co — TF evaluation order.
        // FP16: per-op round to fp16 (TF half kernel semantics).
        AscendC::Reg::Sub(t1Reg, oneReg, oReg, mask); // t1 = 1 - o
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t1Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, oReg, t1Reg, mask); // t0 = o * (1 - o)
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t0Reg, hgReg, mask); // t0 = o*(1-o) * h_grad
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t0Reg, coReg, mask); // do_pre = t0 * co
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::StoreAlign(doPreUb + off, t0Reg, mask);
    }
}

// ============ VF2: dig folded in-chain: dcs = cs_grad + h_grad*o*dig
// ============ [+ do_pre*wco, use_peephole branch term]  (eq 1 dig + eq 2)
// dig = 1 - co*co (tanh(cs) derivative, activation form) stays in registers.
// Multiply order: TF evaluation order, left-associative per element — the
// association ((dig*h_grad)*o)+cs_grad, NOT ((h_grad*o)*dig)+cs_grad.
template <typename T, bool USE_PEEPHOLE>
__simd_vf__ inline void DcsVfImpl(__ubuf__ T* csGradUb, __ubuf__ T* hGradUb, __ubuf__ T* oUb, __ubuf__ T* coUb,
                                  __ubuf__ float* doPreUb, __ubuf__ T* wcoUb, __ubuf__ float* dcsUb, uint32_t segElems,
                                  uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<T, float>;
    AscendC::Reg::RegTensor<float> cgReg;
    AscendC::Reg::RegTensor<float> hgReg;
    AscendC::Reg::RegTensor<float> oReg;
    AscendC::Reg::RegTensor<float> coReg;
    AscendC::Reg::RegTensor<float> wcoReg;
    AscendC::Reg::RegTensor<float> doPreReg;
    AscendC::Reg::RegTensor<float> digReg;
    AscendC::Reg::RegTensor<float> t0Reg;
    AscendC::Reg::RegTensor<float> oneReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    uint32_t remaining = segElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(cgReg, reinterpret_cast<__ubuf__ float*>(csGradUb) + off);
            AscendC::Reg::LoadAlign(hgReg, reinterpret_cast<__ubuf__ float*>(hGradUb) + off);
            AscendC::Reg::LoadAlign(oReg, reinterpret_cast<__ubuf__ float*>(oUb) + off);
            AscendC::Reg::LoadAlign(coReg, reinterpret_cast<__ubuf__ float*>(coUb) + off);
            if constexpr (USE_PEEPHOLE) {
                AscendC::Reg::LoadAlign(wcoReg, reinterpret_cast<__ubuf__ float*>(wcoUb) + off);
            }
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, csGradUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(cgReg, b16Reg, mask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, hGradUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(hgReg, b16Reg, mask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, oUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(oReg, b16Reg, mask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, coUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(coReg, b16Reg, mask);
            if constexpr (USE_PEEPHOLE) {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, wcoUb + off);
                AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(wcoReg, b16Reg, mask);
            }
        }
        AscendC::Reg::LoadAlign(doPreReg, doPreUb + off); // workDoPre (fp32, VF1 output)

        // dig = 1 - co * co (folded into the dcs chain, not stored to UB).
        // FP16: per-op round to fp16 (TF half kernel semantics).
        AscendC::Reg::Mul(digReg, coReg, coReg, mask); // dig = co^2
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(digReg, mask);
        }
        AscendC::Reg::Sub(digReg, oneReg, digReg, mask); // dig = 1 - co^2
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(digReg, mask);
        }

        // dcs = ((dig * h_grad) * o) + cs_grad (+ do_pre*wco) — TF
        // evaluation order.
        AscendC::Reg::Mul(t0Reg, digReg, hgReg, mask); // t0 = dig * h_grad
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t0Reg, oReg, mask); // t0 = dig * h_grad * o
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::Add(t0Reg, t0Reg, cgReg, mask); // t0 = term + cs_grad
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        if constexpr (USE_PEEPHOLE) {
            AscendC::Reg::Mul(digReg, doPreReg, wcoReg, mask); // dig slot reused: do_pre * wco
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(digReg, mask);
            }
            AscendC::Reg::Add(t0Reg, t0Reg, digReg, mask); // dcs += do_pre * wco
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(t0Reg, mask);
            }
        }
        AscendC::Reg::StoreAlign(dcsUb + off, t0Reg, mask);
    }
}

// ============ VF3: three gate pre-activation gradients (eq 3) ============
// di_pre = ((i*(1-i))*dcs)*ci; dci_pre = ((1-ci*ci)*dcs)*i;
// df_pre = ((f*(1-f))*dcs)*cs_prev — all three in TF evaluation order
// (left-associative per element).
template <typename T>
__simd_vf__ inline void GateGradVfImpl(__ubuf__ float* dcsUb, __ubuf__ T* ciUb, __ubuf__ T* iUb, __ubuf__ T* csPrevUb,
                                       __ubuf__ T* fUb, __ubuf__ float* diPreUb, __ubuf__ float* dciPreUb,
                                       __ubuf__ float* dfPreUb, uint32_t segElems, uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<T, float>;
    AscendC::Reg::RegTensor<float> dcsReg;
    AscendC::Reg::RegTensor<float> ciReg;
    AscendC::Reg::RegTensor<float> iReg;
    AscendC::Reg::RegTensor<float> cspReg;
    AscendC::Reg::RegTensor<float> fReg;
    AscendC::Reg::RegTensor<float> t0Reg;
    AscendC::Reg::RegTensor<float> t1Reg;
    AscendC::Reg::RegTensor<float> oneReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    uint32_t remaining = segElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(dcsReg, dcsUb + off); // workDcs (fp32, VF2 output)
        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(ciReg, reinterpret_cast<__ubuf__ float*>(ciUb) + off);
            AscendC::Reg::LoadAlign(iReg, reinterpret_cast<__ubuf__ float*>(iUb) + off);
            AscendC::Reg::LoadAlign(cspReg, reinterpret_cast<__ubuf__ float*>(csPrevUb) + off);
            AscendC::Reg::LoadAlign(fReg, reinterpret_cast<__ubuf__ float*>(fUb) + off);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, ciUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(ciReg, b16Reg, mask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, iUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(iReg, b16Reg, mask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, csPrevUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(cspReg, b16Reg, mask);
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, fUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(fReg, b16Reg, mask);
        }

        // di_pre = ((i * (1 - i)) * dcs) * ci — TF order (eq 3).
        // FP16: per-op round to fp16 (TF half kernel semantics).
        AscendC::Reg::Sub(t1Reg, oneReg, iReg, mask); // t1 = 1 - i
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t1Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, iReg, t1Reg, mask); // t0 = i * (1 - i)
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t0Reg, dcsReg, mask); // t0 = i*(1-i) * dcs
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t0Reg, ciReg, mask); // di_pre = t0 * ci
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::StoreAlign(diPreUb + off, t0Reg, mask);

        // dci_pre = ((1 - ci * ci) * dcs) * i — TF order (eq 3)
        AscendC::Reg::Mul(t1Reg, ciReg, ciReg, mask); // t1 = ci^2
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t1Reg, mask);
        }
        AscendC::Reg::Sub(t1Reg, oneReg, t1Reg, mask); // t1 = 1 - ci^2
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t1Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t1Reg, dcsReg, mask); // t0 = (1 - ci^2) * dcs
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t0Reg, iReg, mask); // dci_pre = t0 * i
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::StoreAlign(dciPreUb + off, t0Reg, mask);

        // df_pre = ((f * (1 - f)) * dcs) * cs_prev — TF order (eq 3)
        AscendC::Reg::Sub(t1Reg, oneReg, fReg, mask); // t1 = 1 - f
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t1Reg, mask);
        }
        AscendC::Reg::Mul(t1Reg, fReg, t1Reg, mask); // t1 = f * (1 - f)
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t1Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t1Reg, dcsReg, mask); // t0 = f*(1-f) * dcs
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::Mul(t0Reg, t0Reg, cspReg, mask); // df_pre = t0 * cs_prev
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        AscendC::Reg::StoreAlign(dfPreUb + off, t0Reg, mask);
    }
}

// ============ VF4: cs_prev_grad = dcs*f [+ wci*di_pre + wcf*df_pre] (eq 5) =
// Compute + chain-tail narrowing Cast fused in one asc_vf_call (FP16).
template <typename T, bool USE_PEEPHOLE>
__simd_vf__ inline void CsPrevGradVfImpl(__ubuf__ float* dcsUb, __ubuf__ float* diPreUb, __ubuf__ float* dfPreUb,
                                         __ubuf__ T* fUb, __ubuf__ T* wciUb, __ubuf__ T* wcfUb, __ubuf__ T* outUb,
                                         uint32_t segElems, uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<T, float>;
    AscendC::Reg::RegTensor<float> dcsReg;
    AscendC::Reg::RegTensor<float> diReg;
    AscendC::Reg::RegTensor<float> dfReg;
    AscendC::Reg::RegTensor<float> fReg;
    AscendC::Reg::RegTensor<float> wciReg;
    AscendC::Reg::RegTensor<float> wcfReg;
    AscendC::Reg::RegTensor<float> t0Reg;
    AscendC::Reg::RegTensor<float> t1Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = segElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(dcsReg, dcsUb + off); // workDcs (fp32)
        if constexpr (USE_PEEPHOLE) {
            AscendC::Reg::LoadAlign(diReg, diPreUb + off); // workDiPre (fp32)
            AscendC::Reg::LoadAlign(dfReg, dfPreUb + off); // workDfPre (fp32)
        }
        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(fReg, reinterpret_cast<__ubuf__ float*>(fUb) + off);
            if constexpr (USE_PEEPHOLE) {
                AscendC::Reg::LoadAlign(wciReg, reinterpret_cast<__ubuf__ float*>(wciUb) + off);
                AscendC::Reg::LoadAlign(wcfReg, reinterpret_cast<__ubuf__ float*>(wcfUb) + off);
            }
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, fUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(fReg, b16Reg, mask);
            if constexpr (USE_PEEPHOLE) {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, wciUb + off);
                AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(wciReg, b16Reg, mask);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, wcfUb + off);
                AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(wcfReg, b16Reg, mask);
            }
        }

        // cs_prev_grad = dcs * f (+ wci*di_pre + wcf*df_pre).
        // FP16: per-op round to fp16 (TF half kernel semantics)
        // — the rounded dcs*f is already fp16-exact, so the chain-tail
        // narrowing Cast below is value-preserving (no double rounding).
        AscendC::Reg::Mul(t0Reg, dcsReg, fReg, mask); // t0 = dcs * f
        if constexpr (!IsFp32) {
            RoundFp16InPlace<T>(t0Reg, mask);
        }
        if constexpr (USE_PEEPHOLE) {
            AscendC::Reg::Mul(t1Reg, wciReg, diReg, mask); // t1 = wci * di_pre
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(t1Reg, mask);
            }
            AscendC::Reg::Add(t0Reg, t0Reg, t1Reg, mask);
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(t0Reg, mask);
            }
            AscendC::Reg::Mul(t1Reg, wcfReg, dfReg, mask); // t1 = wcf * df_pre
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(t1Reg, mask);
            }
            AscendC::Reg::Add(t0Reg, t0Reg, t1Reg, mask);
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(t0Reg, mask);
            }
        }
        // Chain-tail narrowing Cast: fp32 -> T (round-to-nearest-even + NO_SAT)
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outUb) + off, t0Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, t0Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outUb + off, b16Reg, mask);
        }
    }
}

// ============ VF5: dicfo four column blocks narrowing Cast (eq 4) ============
// [di | dc | df | do] -> T (icfo column order). FP32: passthrough, no Cast.
template <typename T>
__simd_vf__ inline void DicfoPostVfImpl(__ubuf__ float* diPreUb, __ubuf__ float* dciPreUb, __ubuf__ float* dfPreUb,
                                        __ubuf__ float* doPreUb, __ubuf__ T* outDiUb, __ubuf__ T* outDciUb,
                                        __ubuf__ T* outDfUb, __ubuf__ T* outDoUb, uint32_t segElems,
                                        uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<T, float>;
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = segElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(f32Reg, diPreUb + off); // di column (workDiPre)
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outDiUb) + off, f32Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outDiUb + off, b16Reg, mask);
        }

        AscendC::Reg::LoadAlign(f32Reg, dciPreUb + off); // dc column (workDciPre)
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outDciUb) + off, f32Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outDciUb + off, b16Reg, mask);
        }

        AscendC::Reg::LoadAlign(f32Reg, dfPreUb + off); // df column (workDfPre)
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outDfUb) + off, f32Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outDfUb + off, b16Reg, mask);
        }

        AscendC::Reg::LoadAlign(f32Reg, doPreUb + off); // do column (workDoPre)
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outDoUb) + off, f32Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outDoUb + off, b16Reg, mask);
        }
    }
}

// ============ ZeroFillVf: +0.0 fill in T dtype ============
// Peephole-output exact zeroing + B==0 empty zero-fill + fp32 accumulator init
// (acc init belongs to the peep=true branches).
// Note: the masked store's ZEROING mode writes 0 to the mask-outside lanes of
// the last repeat — the destination buffer must cover the full repeat.
template <typename T>
__simd_vf__ inline void ZeroFillVfImpl(__ubuf__ T* outUb, uint32_t segElems, uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<T, float>;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::Duplicate(zeroReg, 0.0f); // +0.0 (exact, not pad-clear semantics)
    uint32_t remaining = segElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outUb) + off, zeroReg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, zeroReg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outUb + off, b16Reg, mask);
        }
    }
}

// ============ VF6: peephole tile-level accumulation (eq 6 pre-body; ============
// ============ USE_PEEPHOLE=true only) — accWci += cs_prev*di_pre; ============
// ============ accWcf += cs_prev*df_pre; accWco += cs*do_pre (b = 0..rowCount-1
// ============ ascending, row-wise FP32 sequential add; wco_grad multiplies cs,
// ============ NOT co). Segment-outer x row-inner:
// ============ the acc registers stay resident over the whole tile, no
// ============ same-address read/write within a segment -> no LocalMemBar;
// ============ hardware-serial across asc_vf_call. The padded columns
// ============ [cTileCur, cTileAlign) accumulate isolated garbage that never
// ============ reaches GM (valid-length partial write).
// ============ K1: dst (accWci/accWcf/accWco) independent of all srcs.
// ============ TF alignment: wci_grad = sum_b(di*cs_prev) etc. — the products
// ============ are multiplication-commutative (cs_prev*di == di*cs_prev bitwise)
// ============ and each core's partial starts from +0.0 (0.0 + x == x exactly),
// ============ so the flattened addition sequence matches TF's sequential sum.
// ============ FP16 per-op term rounding + PER-ADD ACCUMULATOR ROUNDING: the
// ============ TF half kernel computes every peephole PRODUCT and EVERY
// ============ ADDITION of the column sum as fp32-op + RNE round back to
// ============ fp16 (Eigen::half semantics — same rule the main chain
// ============ follows). The three products round in-register to fp16
// ============ (RoundFp16InPlace) before entering the accumulator, and each
// ============ Add result is rounded back to fp16 as well — the accumulator
// ============ (fp32 register storage) therefore only ever holds fp16-exact
// ============ values, so the PeepWriteOut chain-tail Cast is value-preserving
// ============ and the written column sum is bitwise the fp16 serial chain.
template <typename T>
__simd_vf__ inline void PeepAccumVfImpl(__ubuf__ T* csPrevUb, __ubuf__ T* csUb, __ubuf__ float* diPreUb,
                                        __ubuf__ float* dfPreUb, __ubuf__ float* doPreUb, __ubuf__ float* accWciUb,
                                        __ubuf__ float* accWcfUb, __ubuf__ float* accWcoUb, uint32_t segElems,
                                        uint16_t repeatTime, uint16_t rowStepU16, uint16_t rowCountU16)
{
    constexpr bool IsFp32 = std::is_same_v<T, float>;
    AscendC::Reg::RegTensor<float> accWciReg;
    AscendC::Reg::RegTensor<float> accWcfReg;
    AscendC::Reg::RegTensor<float> accWcoReg;
    AscendC::Reg::RegTensor<float> cspReg;
    AscendC::Reg::RegTensor<float> csReg;
    AscendC::Reg::RegTensor<float> tReg;
    AscendC::Reg::RegTensor<float> prodReg;
    AscendC::Reg::RegTensor<T> b16Reg;
    AscendC::Reg::MaskReg mask;

    for (uint16_t i = 0; i < repeatTime; ++i) { // segment outer (REP_F32 lanes)
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        uint32_t remaining = segElems;
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(accWciReg, accWciUb + off); // accumulator current value (round start: 0)
        AscendC::Reg::LoadAlign(accWcfReg, accWcfUb + off);
        AscendC::Reg::LoadAlign(accWcoReg, accWcoUb + off);

        for (uint16_t b = 0; b < rowCountU16; ++b) { // row inner (ascending = fixed order)
            int32_t rowOff = static_cast<int32_t>(b) * static_cast<int32_t>(rowStepU16) + off;
            if constexpr (IsFp32) {
                AscendC::Reg::LoadAlign(cspReg, reinterpret_cast<__ubuf__ float*>(csPrevUb) + rowOff);
                AscendC::Reg::LoadAlign(csReg, reinterpret_cast<__ubuf__ float*>(csUb) + rowOff);
            } else {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, csPrevUb + rowOff);
                AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(cspReg, b16Reg, mask);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, csUb + rowOff);
                AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(csReg, b16Reg, mask);
            }
            AscendC::Reg::LoadAlign(tReg, diPreUb + rowOff);
            AscendC::Reg::Mul(prodReg, cspReg, tReg, mask);
            // per-op fp16 term rounding + per-add accumulator rounding (TF
            // half-kernel semantics — every intermediate of the chain is
            // fp16-exact).
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(prodReg, mask);
            }
            AscendC::Reg::Add(accWciReg, accWciReg, prodReg, mask); // accWci += cs_prev*di_pre
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(accWciReg, mask);
            }
            AscendC::Reg::LoadAlign(tReg, dfPreUb + rowOff);
            AscendC::Reg::Mul(prodReg, cspReg, tReg, mask);
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(prodReg, mask);
            }
            AscendC::Reg::Add(accWcfReg, accWcfReg, prodReg, mask); // accWcf += cs_prev*df_pre
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(accWcfReg, mask);
            }
            AscendC::Reg::LoadAlign(tReg, doPreUb + rowOff);
            AscendC::Reg::Mul(prodReg, csReg, tReg, mask);
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(prodReg, mask);
            }
            AscendC::Reg::Add(accWcoReg, accWcoReg, prodReg, mask); // accWco += cs*do_pre
            if constexpr (!IsFp32) {
                RoundFp16InPlace<T>(accWcoReg, mask);
            }
        }

        AscendC::Reg::StoreAlign(accWciUb + off, accWciReg, mask); // write back, resident across bTile rounds
        AscendC::Reg::StoreAlign(accWcfUb + off, accWcfReg, mask);
        AscendC::Reg::StoreAlign(accWcoUb + off, accWcoReg, mask);
    }
}

// ============ PeepOutCastVf: chain-tail narrowing Cast of the three fp32 ====
// ============ accumulators -> T staging (eq 6 tail; USE_PEEPHOLE=true only).
// ============ Every intermediate of the accumulation chain is fp16-exact
// ============ (per-add rounding) / natively fp32, so this Cast is
// ============ value-preserving — the staging slice is the bitwise serial
// ============ column sum. K1: dst (3 staging slices in the out front)
// ============ independent of src (3 acc buffers); fp32 path is
// ============ a register passthrough (no Cast anywhere).
template <typename T>
__simd_vf__ inline void PeepOutCastVfImpl(__ubuf__ float* accWciUb, __ubuf__ float* accWcfUb, __ubuf__ float* accWcoUb,
                                          __ubuf__ T* outWciUb, __ubuf__ T* outWcfUb, __ubuf__ T* outWcoUb,
                                          uint32_t segElems, uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<T, float>;
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = segElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(f32Reg, accWciUb + off);
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outWciUb) + off, f32Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outWciUb + off, b16Reg, mask);
        }

        AscendC::Reg::LoadAlign(f32Reg, accWcfUb + off);
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outWcfUb) + off, f32Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outWcfUb + off, b16Reg, mask);
        }

        AscendC::Reg::LoadAlign(f32Reg, accWcoUb + off);
        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float*>(outWcoUb) + off, f32Reg, mask);
        } else {
            AscendC::Reg::RegTensor<T> b16Reg;
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outWcoUb + off, b16Reg, mask);
        }
    }
}

// =============================================================================
// VF call-site wrappers (__aicore__ inline; compute repeatTime then
// asc_vf_call)
// =============================================================================

// Per-row VF: segElems = cTileAlign (padded row width, always a multiple of
// REP_F32 -> masks are always full; row base = tile base + b * cTileAlign).
template <typename T>
__aicore__ inline void DoPreVf(__ubuf__ T* hGradRow, __ubuf__ T* coRow, __ubuf__ T* oRow, __ubuf__ float* doPreRow,
                               uint32_t segElems)
{
    const uint16_t repeatTime = static_cast<uint16_t>((segElems + REP_F32_U - 1) / REP_F32_U);
    asc_vf_call<DoPreVfImpl<T>>(hGradRow, coRow, oRow, doPreRow, segElems, repeatTime);
}

template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void DcsVf(__ubuf__ T* csGradRow, __ubuf__ T* hGradRow, __ubuf__ T* oRow, __ubuf__ T* coRow,
                             __ubuf__ float* doPreRow, __ubuf__ T* wcoVec, __ubuf__ float* dcsRow, uint32_t segElems)
{
    const uint16_t repeatTime = static_cast<uint16_t>((segElems + REP_F32_U - 1) / REP_F32_U);
    asc_vf_call<DcsVfImpl<T, USE_PEEPHOLE>>(csGradRow, hGradRow, oRow, coRow, doPreRow, wcoVec, dcsRow, segElems,
                                            repeatTime);
}

template <typename T>
__aicore__ inline void GateGradVf(__ubuf__ float* dcsRow, __ubuf__ T* ciRow, __ubuf__ T* iRow, __ubuf__ T* csPrevRow,
                                  __ubuf__ T* fRow, __ubuf__ float* diPreRow, __ubuf__ float* dciPreRow,
                                  __ubuf__ float* dfPreRow, uint32_t segElems)
{
    const uint16_t repeatTime = static_cast<uint16_t>((segElems + REP_F32_U - 1) / REP_F32_U);
    asc_vf_call<GateGradVfImpl<T>>(dcsRow, ciRow, iRow, csPrevRow, fRow, diPreRow, dciPreRow, dfPreRow, segElems,
                                   repeatTime);
}

template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void CsPrevGradVf(__ubuf__ float* dcsRow, __ubuf__ float* diPreRow, __ubuf__ float* dfPreRow,
                                    __ubuf__ T* fRow, __ubuf__ T* wciVec, __ubuf__ T* wcfVec, __ubuf__ T* outRow,
                                    uint32_t segElems)
{
    const uint16_t repeatTime = static_cast<uint16_t>((segElems + REP_F32_U - 1) / REP_F32_U);
    asc_vf_call<CsPrevGradVfImpl<T, USE_PEEPHOLE>>(dcsRow, diPreRow, dfPreRow, fRow, wciVec, wcfVec, outRow, segElems,
                                                   repeatTime);
}

template <typename T>
__aicore__ inline void DicfoPostVf(__ubuf__ float* diPreRow, __ubuf__ float* dciPreRow, __ubuf__ float* dfPreRow,
                                   __ubuf__ float* doPreRow, __ubuf__ T* outDiRow, __ubuf__ T* outDciRow,
                                   __ubuf__ T* outDfRow, __ubuf__ T* outDoRow, uint32_t segElems)
{
    const uint16_t repeatTime = static_cast<uint16_t>((segElems + REP_F32_U - 1) / REP_F32_U);
    asc_vf_call<DicfoPostVfImpl<T>>(diPreRow, dciPreRow, dfPreRow, doPreRow, outDiRow, outDciRow, outDfRow, outDoRow,
                                    segElems, repeatTime);
}

// Zero-fill wrapper (outUb width >= CeilAlign(segElems, REP_F32) T elements).
template <typename T>
__aicore__ inline void ZeroFillVf(__ubuf__ T* outUb, uint32_t segElems)
{
    const uint16_t repeatTime = static_cast<uint16_t>((segElems + REP_F32_U - 1) / REP_F32_U);
    asc_vf_call<ZeroFillVfImpl<T>>(outUb, segElems, repeatTime);
}

// Tile-level VF wrapper (USE_PEEPHOLE=true only): segElems = cTileAlign
// (padded row width, always a REP_F32 multiple -> full masks), rowStep =
// cTileAlign, rowCount = this round's valid row count.
template <typename T>
__aicore__ inline void PeepAccumVf(__ubuf__ T* csPrevTile, __ubuf__ T* csTile, __ubuf__ float* diPreTile,
                                   __ubuf__ float* dfPreTile, __ubuf__ float* doPreTile, __ubuf__ float* accWci,
                                   __ubuf__ float* accWcf, __ubuf__ float* accWco, uint32_t segElems,
                                   uint16_t rowStepU16, uint16_t rowCountU16)
{
    const uint16_t repeatTime = static_cast<uint16_t>((segElems + REP_F32_U - 1) / REP_F32_U);
    asc_vf_call<PeepAccumVfImpl<T>>(csPrevTile, csTile, diPreTile, dfPreTile, doPreTile, accWci, accWcf, accWco,
                                    segElems, repeatTime, rowStepU16, rowCountU16);
}

// Peep out chain-tail cast wrapper: 3 fp32 accumulators -> 3
// T staging slices in the out front; segElems = cTileCur (valid columns,
// masked tail — staging slot width cTileAlign always covers the last repeat).
template <typename T>
__aicore__ inline void PeepOutCastVf(__ubuf__ float* accWci, __ubuf__ float* accWcf, __ubuf__ float* accWco,
                                     __ubuf__ T* outWci, __ubuf__ T* outWcf, __ubuf__ T* outWco, uint32_t segElems)
{
    const uint16_t repeatTime = static_cast<uint16_t>((segElems + REP_F32_U - 1) / REP_F32_U);
    asc_vf_call<PeepOutCastVfImpl<T>>(accWci, accWcf, accWco, outWci, outWcf, outWco, segElems, repeatTime);
}

// =============================================================================
// LSTMBlockCellGradKernel<T, USE_PEEPHOLE> — unified kernel class template
// =============================================================================

// Tile inputs consumed per instance: the
// USE_PEEPHOLE=false instances copy 8 (no `cs`); the true instances insert
// `cs` at in-slot 2 to make 9 (wco_grad multiplies cs).
constexpr int32_t NUM_DICFO_BLOCKS = 4; // icfo = [i|c|f|o] column blocks
constexpr int32_t NUM_PEEP_OUTPUTS = 3; // wci_grad / wcf_grad / wco_grad
constexpr int32_t MAX_IN_SLOTS = 9;     // static array bound (peep slot count)

template <typename T, bool USE_PEEPHOLE>
class LSTMBlockCellGradKernel {
public:
    __aicore__ inline LSTMBlockCellGradKernel() = default;

    // GM_ADDR 形参沿用 OpDef 张量名（snake_case, 与 REG_OP/INPUT 逐字对应,
    // tf_plugin 按名/按位映射依赖该拼写）; 类内自声明变量遵循小驼峰.
    __aicore__ inline void Init(GM_ADDR cs_prev, GM_ADDR wci, GM_ADDR wcf, GM_ADDR wco, GM_ADDR i, GM_ADDR cs,
                                GM_ADDR f, GM_ADDR o, GM_ADDR ci, GM_ADDR co, GM_ADDR cs_grad, GM_ADDR h_grad,
                                GM_ADDR cs_prev_grad, GM_ADDR dicfo, GM_ADDR wci_grad, GM_ADDR wcf_grad,
                                GM_ADDR wco_grad, GM_ADDR workspace, const LSTMBlockCellGradTilingData* td,
                                AscendC::TPipe* pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void EmptyZeroFill();  // B==0: 3x(H,) zero fill
    __aicore__ inline void Phase0ZeroFill(); // peep=false: peep outputs exact zero
    __aicore__ inline void CopyInRound(int64_t bStartCur, int64_t bTileCur, int64_t cOff, int64_t cTileCur);
    __aicore__ inline void CopyInVecRound(int64_t cOff, int64_t cTileCur); // peep: 3 vector slices
    __aicore__ inline void ComputeRound(int64_t bTileCur);
    __aicore__ inline void CopyOutRound(int64_t bStartCur, int64_t bTileCur, int64_t cOff, int64_t cTileCur);
    // peep=true: acc Cast -> column-slice direct write
    __aicore__ inline void PeepWriteOut(int64_t cOff, int64_t cTileCur, bool issueTailSet);

    // ---- tile-domain slot map (three-region TBuf
    //      aggregation with slot-stride bTile*cTileAlign; `cs` occupies in-slot
    //      2 only for the peephole instances, shifting f/o/ci/co/cs_grad/h_grad
    //      by one) ----
    static constexpr int32_t SLOT_CS_PREV = 0;
    static constexpr int32_t SLOT_I = 1;
    static constexpr int32_t SLOT_CS = 2; // USE_PEEPHOLE only
    static constexpr int32_t SLOT_F = USE_PEEPHOLE ? 3 : 2;
    static constexpr int32_t SLOT_O = USE_PEEPHOLE ? 4 : 3;
    static constexpr int32_t SLOT_CI = USE_PEEPHOLE ? 5 : 4;
    static constexpr int32_t SLOT_CO = USE_PEEPHOLE ? 6 : 5;
    static constexpr int32_t SLOT_CS_GRAD = USE_PEEPHOLE ? 7 : 6;
    static constexpr int32_t SLOT_H_GRAD = USE_PEEPHOLE ? 8 : 7;
    static constexpr int32_t NUM_IN_SLOTS = USE_PEEPHOLE ? 9 : 8;
    static constexpr int32_t NUM_WORK_SLOTS = 5; // DoPre/Dcs/DiPre/DciPre/DfPre
    static constexpr int32_t NUM_OUT_SLOTS = 5;  // CsPrevGrad + dicfo di/dc/df/do
    static constexpr int32_t OUT_SLOT_CS_PREV_GRAD = 0;
    static constexpr int32_t OUT_SLOT_DICFO_BASE = 1; // +g, g = 0..3 (icfo order)

    // ---- work-domain slot map (fp32 intermediates, slot stride = slotElems_) ----
    static constexpr int32_t WORK_SLOT_DO_PRE = 0;
    static constexpr int32_t WORK_SLOT_DCS = 1;
    static constexpr int32_t WORK_SLOT_DI_PRE = 2;
    static constexpr int32_t WORK_SLOT_DCI_PRE = 3;
    static constexpr int32_t WORK_SLOT_DF_PRE = 4;

    // ---- tiling data / pipe ----
    const LSTMBlockCellGradTilingData* td_ = nullptr;
    AscendC::TPipe* pipe_ = nullptr;

    // ---- empty-tensor / idle flags (judged only from the
    //      global shape fields batchSize/cellSize, never from the degenerate
    //      split fields) ----
    bool emptyH_ = false; // H==0: every output is 0-dim, early return
    bool emptyB_ = false; // B==0 (H>=1): 3x(H,) zero-fill path
    bool idle_ = false;   // blockId >= usedCoreNum (defensive)

    // ---- GM bindings ----
    AscendC::GlobalTensor<T> gmIn_[MAX_IN_SLOTS]; // slot-indexed: cs_prev/i/[cs]/f/o/ci/co/cs_grad/h_grad
    AscendC::GlobalTensor<T> gmCsPrevGrad_;
    AscendC::GlobalTensor<T> gmDicfo_;
    AscendC::GlobalTensor<T> gmPeepOut_[NUM_PEEP_OUTPUTS]; // wci_grad / wcf_grad / wco_grad
    AscendC::GlobalTensor<T> gmWci_;                       // peephole weight vectors (cell,) — peep only
    AscendC::GlobalTensor<T> gmWcf_;
    AscendC::GlobalTensor<T> gmWco_;

    // ---- UB buffers (single buffer, no double
    //      buffering; tile domain aggregated into 3 TBuf regions so
    //      PeepWriteOut can reuse the out front as the 3 staging slices
    //      without extra UB) ----
    AscendC::TBuf<AscendC::TPosition::VECCALC> inRegion_;   // NUM_IN_SLOTS x T tile inputs
    AscendC::TBuf<AscendC::TPosition::VECCALC> workRegion_; // 5 x fp32 intermediates
                                                            // (cross-asc_vf_call boundary)
    AscendC::TBuf<AscendC::TPosition::VECCALC> outRegion_;  // CsPrevGrad + 4 dicfo blocks (T)
    AscendC::TBuf<AscendC::TPosition::VECCALC> vecWci_;     // 3 peephole vectors (cTileAlign x T) — peep
    AscendC::TBuf<AscendC::TPosition::VECCALC> vecWcf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vecWco_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accWci_; // 3 fp32 accumulators (cTileAlign x fp32) — peep
    AscendC::TBuf<AscendC::TPosition::VECCALC> accWcf_; //   (resident across bTile rounds)
    AscendC::TBuf<AscendC::TPosition::VECCALC> accWco_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroSeg_; // B==0 zero-fill only
                                                         // (EMPTY_ZERO_SEG_ELEMS x T)

    // ---- event IDs (fetched via FetchEventID, never hard-coded) ----
    int32_t evMte2V_ = 0; // MTE2_V: CopyIn done -> VF may read in/vec domain
    int32_t evVmte3_ = 0; // V_MTE3:  VF done writing out/acc domain -> CopyOut / partial write
    int32_t evVmte2_ = 0; // V_MTE2:  VF done reading in/vec domain -> next CopyIn
    int32_t evMte3v_ = 0; // MTE3_V:  CopyOut/partial write done reading out/acc domain -> next VF

    // ---- this core's partition ----
    int64_t slotElems_ = 0;   // bTile * cTileAlign (tile slot width in elements)
    int64_t bStart_ = 0;      // first batch row of this core (0 for peep=true: full batch)
    int64_t rows_ = 0;        // batch rows of this core (B for peep=true: full batch)
    int64_t batchRounds_ = 0; // CeilDiv(rows_, bTile), computed here (not in TilingData)
    int64_t cStart_ = 0;      // cell slice start (peep=false: peephole zero-fill slice;
                              //   peep=true: this core's output column slice)
    int64_t sliceLen_ = 0;    // cell slice length (peep=false: may be 0 -> zero-fill skipped;
                              //   peep=true: >= 1 on every participating core)
};

// -----------------------------------------------------------------------------
// Init — GM binding + UB allocation + event IDs + partition precompute.
// Per-buffer TBufs land as three aggregated regions with slot-stride
// indexing — aggregation gives PeepWriteOut the contiguous in/out reuse
// fronts.
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::Init(
    GM_ADDR cs_prev, GM_ADDR wci, GM_ADDR wcf, GM_ADDR wco, GM_ADDR i, GM_ADDR cs, GM_ADDR f, GM_ADDR o, GM_ADDR ci,
    GM_ADDR co, GM_ADDR cs_grad, GM_ADDR h_grad, GM_ADDR cs_prev_grad, GM_ADDR dicfo, GM_ADDR wci_grad,
    GM_ADDR wcf_grad, GM_ADDR wco_grad, GM_ADDR workspace, const LSTMBlockCellGradTilingData* td, AscendC::TPipe* pipe)
{
    td_ = td;
    pipe_ = pipe;

    // Empty-tensor flags (only the global shape fields).
    emptyH_ = (td->cellSize == 0);                // H==0: all outputs 0-dim
    emptyB_ = (!emptyH_) && (td->batchSize == 0); // B==0 (H>=1): zero-fill path

    // GM bindings (the 8 common (batch,cell) inputs + 5 outputs; slot-indexed
    // with `cs` at in-slot 2 for the peephole instances).
    if (!emptyH_) {
        gmIn_[SLOT_CS_PREV].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cs_prev));
        gmIn_[SLOT_I].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(i));
        if constexpr (USE_PEEPHOLE) {
            gmIn_[SLOT_CS].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cs)); // wco_grad multiplies cs
            gmWci_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wci));        // 3 peephole weight vectors
            gmWcf_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wcf));
            gmWco_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wco));
        }
        gmIn_[SLOT_F].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(f));
        gmIn_[SLOT_O].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(o));
        gmIn_[SLOT_CI].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ci));
        gmIn_[SLOT_CO].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(co));
        gmIn_[SLOT_CS_GRAD].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cs_grad));
        gmIn_[SLOT_H_GRAD].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h_grad));
        gmCsPrevGrad_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cs_prev_grad));
        gmDicfo_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dicfo));
        // use_peephole=false 时首步写精确 0 / peep=true 列切分直接写本核列片
        gmPeepOut_[0].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wci_grad));
        gmPeepOut_[1].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wcf_grad));
        gmPeepOut_[2].SetGlobalBuffer(reinterpret_cast<__gm__ T*>(wco_grad));
    }
    // workspace is not consumed by any tilingKey (the peep column-split
    // writes its output slices directly — no user segment; only the
    // framework-managed system segment exists).
    (void)workspace;
    if constexpr (!USE_PEEPHOLE) {
        // wci/wcf/wco/cs are consumed only by the USE_PEEPHOLE=true instances.
        (void)cs;
        (void)wci;
        (void)wcf;
        (void)wco;
    }

    if (emptyH_) {
        return; // Process first step early-returns; no buffer, no partition
    }

    // Event IDs (fetched via the pipe's FetchEventID, never hard-coded; fetched
    // before the emptyB_ early return because EmptyZeroFill uses evVmte3_).
    evMte2V_ = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
    evVmte3_ = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE3));
    evVmte2_ = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE2));
    evMte3v_ = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_V));

    if (emptyB_) {
        // B==0 zero-fill dedicated small TBuf (mutually
        // exclusive with the main-compute tile domain, not part of the tile budget).
        pipe_->InitBuffer(zeroSeg_, static_cast<uint32_t>(EMPTY_ZERO_SEG_ELEMS * sizeof(T)));
        return; // Process: EmptyZeroFill then return — no zero-fill, no tile loop
    }

    // Non-empty: the three-region tile domain (in: NUM_IN_SLOTS x T,
    // work: 5 x fp32, out: 5 x T) + the peephole vector domain (3 vec + 3 acc,
    // USE_PEEPHOLE only; budget = 19 tile + 6 vec slots).
    slotElems_ = td->bTile * td->cTileAlign;
    const int64_t tileBytesT = slotElems_ * static_cast<int64_t>(sizeof(T));
    const int64_t tileBytesF32 = slotElems_ * static_cast<int64_t>(sizeof(float));
    pipe_->InitBuffer(inRegion_, static_cast<uint32_t>(NUM_IN_SLOTS) * static_cast<uint32_t>(tileBytesT));
    pipe_->InitBuffer(workRegion_, static_cast<uint32_t>(NUM_WORK_SLOTS) * static_cast<uint32_t>(tileBytesF32));
    pipe_->InitBuffer(outRegion_, static_cast<uint32_t>(NUM_OUT_SLOTS) * static_cast<uint32_t>(tileBytesT));
    if constexpr (USE_PEEPHOLE) {
        const uint32_t vecBytes = static_cast<uint32_t>(td->cTileAlign * sizeof(T));
        const uint32_t accBytes = static_cast<uint32_t>(td->cTileAlign * sizeof(float));
        pipe_->InitBuffer(vecWci_, vecBytes);
        pipe_->InitBuffer(vecWcf_, vecBytes);
        pipe_->InitBuffer(vecWco_, vecBytes);
        pipe_->InitBuffer(accWci_, accBytes);
        pipe_->InitBuffer(accWcf_, accBytes);
        pipe_->InitBuffer(accWco_, accBytes);
    }

    // blockId -> this core's partition (big/small core
    // mapping over the SPLIT AXIS: batch rows for peep=false, cell columns
    // for peep=true).
    const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
    if (blockId >= static_cast<int64_t>(td->usedCoreNum)) {
        idle_ = true; // out-of-range core: defensive early exit (idle core)
        return;
    }
    if constexpr (USE_PEEPHOLE) {
        // Column-split instances: this core owns the output column
        // slice [cStart_, cStart_+sliceLen_) (sliceLen_ >= 1 whenever C >= 1,
        // since usedCoreNum = min(coreNum, C) by the host) and walks the FULL
        // batch [0, B) so the accumulator chain is a row-ascending serial
        // single chain.
        rows_ = td->batchSize;
        bStart_ = 0;
    } else {
        // Batch-split instances: this core's batch range [bStart_, bStart_+rows_).
        if (blockId < static_cast<int64_t>(td->bigCoreCnt)) {
            rows_ = td->bigCoreCols;
            bStart_ = blockId * td->bigCoreCols;
        } else {
            rows_ = td->smallCoreCols;
            bStart_ = static_cast<int64_t>(td->bigCoreCnt) * td->bigCoreCols +
                      (blockId - static_cast<int64_t>(td->bigCoreCnt)) * td->smallCoreCols;
        }
    }
    batchRounds_ = (rows_ + td->bTile - 1) / td->bTile; // batch rounds, computed here

    // Cell slice derivation (big/small balanced over the split axis; slice
    // boundaries have no alignment constraint; peep=false: sliceLen_==0 ->
    // zero-fill skipped entirely, blockLen must never be 0; peep=true: this
    // core's output column slice, read back from the host's column split).
    if constexpr (USE_PEEPHOLE) {
        if (blockId < static_cast<int64_t>(td->bigCoreCnt)) {
            sliceLen_ = td->bigCoreCols;
            cStart_ = blockId * td->bigCoreCols;
        } else {
            sliceLen_ = td->smallCoreCols;
            cStart_ = static_cast<int64_t>(td->bigCoreCnt) * td->bigCoreCols +
                      (blockId - static_cast<int64_t>(td->bigCoreCnt)) * td->smallCoreCols;
        }
    } else {
        const int64_t C = td->cellSize;
        const int64_t K = static_cast<int64_t>(td->usedCoreNum);
        const int64_t sliceSmall = C / K;
        const int64_t sliceBigCnt = C % K;
        const int64_t sliceBig = sliceSmall + ((sliceBigCnt > 0) ? 1 : 0);
        if (blockId < sliceBigCnt) {
            sliceLen_ = sliceBig;
            cStart_ = blockId * sliceBig;
        } else {
            sliceLen_ = sliceSmall;
            cStart_ = sliceBigCnt * sliceBig + (blockId - sliceBigCnt) * sliceSmall;
        }
    }
}

// -----------------------------------------------------------------------------
// Process — empty-tensor short-circuit -> [peephole-output zero-fill
// (peep=false)] -> main double tile loop -> [PeepWriteOut column-slice direct write (peep=true)]
// (sync positions marked (V0)-(V3), (1)-(10), (P1') — strict Set->Wait 1:1
// over the (j, r) flattened global round sequence; no cross-core barrier:
// each peep core owns its column slice end-to-end)
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::Process()
{
    // Empty-tensor short-circuit (all keys, before
    // any copy or compute; single core guaranteed by host usedCoreNum=1).
    if (emptyH_) {
        return; // H==0: every output is 0-dim — no compute, no GM write
    }
    if (emptyB_) {
        // B==0: (0,H)/(0,4H) outputs are 0-element (nothing to write) +
        // 3x(H,) zero fill (empty sum = zero vector; same implementation
        // path as the peep=false "exact all-zero" contract). Short-circuit
        // precedes the main compute — no compute rounds, no slice write.
        EmptyZeroFill();
        return;
    }
    if (idle_) {
        return;
    }

    // Peephole-output zero-fill (USE_PEEPHOLE=false only): zero this core's cell slice of the
    // three peephole gradient outputs with exact +0.0. The peep=true
    // instances compute these outputs instead (VF6 + PeepWriteOut).
    if constexpr (!USE_PEEPHOLE) {
        Phase0ZeroFill();
    }

    // Main compute — for cTile (outer) -> for bTile (inner).
    // peep=false: cTile rounds tile the GLOBAL cell axis [0, C) (host
    // cTileNum/cTileLast); peep=true: cTile rounds tile THIS CORE'S column
    // slice [cStart_, cStart_+sliceLen_) with the tail localized.
    int64_t cRoundNum = 0;
    if constexpr (USE_PEEPHOLE) {
        cRoundNum = (sliceLen_ + td_->cTile - 1) / td_->cTile; // CeilDiv(sliceLen_, cTile)
    } else {
        cRoundNum = td_->cTileNum;
    }
    for (int64_t j = 0; j < cRoundNum; ++j) {
        int64_t cOff;
        int64_t cTileCur;
        if constexpr (USE_PEEPHOLE) {
            cOff = cStart_ + j * td_->cTile;
            const int64_t colsLeft = sliceLen_ - j * td_->cTile;
            cTileCur = (td_->cTile < colsLeft) ? td_->cTile : colsLeft;
        } else {
            cOff = j * td_->cTile;
            cTileCur = (j == td_->cTileNum - 1) ? td_->cTileLast : td_->cTile;
        }
        if constexpr (USE_PEEPHOLE) {
            // (V0) in+vec domain WAR (j>0): the previous cTile round's last
            //      VF finished reading before this round's CopyInVec may
            //      overwrite (one Set covers both domains — VF reads both).
            if (j > 0) {
                WaitFlag<AscendC::HardEvent::V_MTE2>(evVmte2_);
            }
            // (V1) 3 peephole vector slices (MTE2); covered by this
            //      round's first (3) Set (MTE2 single stream, earlier issue).
            CopyInVecRound(cOff, cTileCur);
            // (V2) out+acc(+staging) domain WAR (j>0): pairs the previous
            //      cTile round's (P1') tail Set — MTE3 stream order means the
            //      flag being set implies BOTH the peep slice write (staging
            //      read) and ALL earlier CopyOuts (out domain) finished
            //      reading (single-Set dual-domain argument; r==0 of
            //      this round needs no (3)).
            if (j > 0) {
                WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
            }
            // (V3) accumulator zero-init (V writes acc; +0.0 exact — the
            //      reduction's initial value, not a pad clear).
            ZeroFillVf<float>(reinterpret_cast<__ubuf__ float*>(accWci_.template Get<float>().GetPhyAddr()),
                              static_cast<uint32_t>(td_->cTileAlign));
            ZeroFillVf<float>(reinterpret_cast<__ubuf__ float*>(accWcf_.template Get<float>().GetPhyAddr()),
                              static_cast<uint32_t>(td_->cTileAlign));
            ZeroFillVf<float>(reinterpret_cast<__ubuf__ float*>(accWco_.template Get<float>().GetPhyAddr()),
                              static_cast<uint32_t>(td_->cTileAlign));
        }
        for (int64_t r = 0; r < batchRounds_; ++r) {
            const int64_t bStartCur = bStart_ + r * td_->bTile;
            const int64_t rowsLeft = rows_ - r * td_->bTile;
            const int64_t bTileCur = (td_->bTile < rowsLeft) ? td_->bTile : rowsLeft; // valid tail rows
            // Global round sequence: "first" = (j==0, r==0), "last" = (j==cRoundNum-1,
            // r==batchRounds-1); the cTile round boundary is NOT a pairing boundary.
            const bool notLastRound = (r < batchRounds_ - 1) || (j < cRoundNum - 1);

            // (1) in-domain WAR: previous global round's VF finished reading the
            //     in-buffers before this round's CopyIn overwrites them. The peep
            //     instances consumed the j>0 case at (V0) — only r>0 waits here.
            if constexpr (USE_PEEPHOLE) {
                if (r > 0) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(evVmte2_);
                }
            } else {
                if (r > 0 || j > 0) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(evVmte2_);
                }
            }
            // (2) CopyIn the tile inputs (MTE2; 9 routes for peep incl. cs),
            //     then signal MTE2 -> V (covers the earlier CopyInVec too).
            CopyInRound(bStartCur, bTileCur, cOff, cTileCur);
            SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            // (3) out-domain WAR: rounds after the global first pair the
            //     previous round's (9); round (j==0, r==0) pairs this core's
            //     zero-fill tail Set (only when the zero-fill ran, i.e. sliceLen_ > 0;
            //     otherwise the outDicfoDo slot is first-written by this
            //     round's VF — no Wait). The peep instances have no zero-fill;
            //     their (j>0, r==0) case is covered by (V2) instead.
            if constexpr (!USE_PEEPHOLE) {
                if (j > 0 || r > 0) {
                    WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
                } else if (sliceLen_ > 0) {
                    WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
                }
            } else {
                if (r > 0) {
                    WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
                }
            }
            // (4) in-domain RAW: this round's CopyIn complete before the VF reads.
            WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V_);
            // (5) per-row VF1 -> VF2 -> VF3 -> VF4 -> VF5 (V pipeline; hardware
            //     serial between VFs — zero sync primitives between them) +
            //     tile-level VF6 peephole accumulation (peep only).
            ComputeRound(bTileCur);
            // (6) in(+vec)-domain WAR for the next global round's CopyIn;
            //     issued before CopyOut so CopyIn(r+1) overlaps CopyOut(r).
            //     Uniform over all tilingKeys.
            if (notLastRound) {
                SetFlag<AscendC::HardEvent::V_MTE2>(evVmte2_);
            }
            // (7) out-domain RAW: VF finished writing out-buffers -> CopyOut.
            SetFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);
            WaitFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);
            // (8) CopyOut cs_prev_grad tile + dicfo's four column blocks (MTE3).
            CopyOutRound(bStartCur, bTileCur, cOff, cTileCur);
            // (9) out-domain WAR: signals "this round's CopyOut finished
            //     reading the out front". peep=false: the global last round
            //     has no consumer — no Set. peep=true: EVERY round Sets —
            //     consumers are (3) of the next round (r+1) or, for the last
            //     batch round of the cTile round, PeepWriteOut's leading
            //     Wait (staging reuse of out slots 0-2 requires the last
            //     CopyOut's MTE3 read to be complete).
            if constexpr (USE_PEEPHOLE) {
                SetFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
            } else {
                if (notLastRound) {
                    SetFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
                }
            }
        }
        // (P1') cTile round end (this core's full-batch walk done): the three
        //       accumulators -> Cast -> direct write to this core's
        //       gmPeepOut_ column slice [cOff, +cTileCur) (no workspace, no
        //       cross-core merge). Tail Set serves the next
        //       cTile round's (V2) out+staging WAR.
        if constexpr (USE_PEEPHOLE) {
            PeepWriteOut(cOff, cTileCur, j < cRoundNum - 1);
        }
    }
}

// -----------------------------------------------------------------------------
// CopyInRound — GM -> UB, NUM_IN_SLOTS isomorphic 2D tiles via DataCopyPad (MTE2)
// (parameter derivation: blockCount = valid rows, blockLen = valid columns in
// bytes, GM srcStride = row gap in bytes, UB dstStride = row gap in datablocks
// so that the UB row pitch = cTileAlign elements; isPad=false — padded region
// is garbage, isolated by valid-length writeback)
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::CopyInRound(int64_t bStartCur, int64_t bTileCur,
                                                                             int64_t cOff, int64_t cTileCur)
{
    const int64_t dtBytes = static_cast<int64_t>(sizeof(T));
    const int64_t blkBytes = cTileCur * dtBytes; // valid row bytes
    const int64_t blkAligned = (blkBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
    const int64_t ubRowGapBlk = (static_cast<int64_t>(td_->cTileAlign) * dtBytes - blkAligned) /
                                BLOCK_BYTES; // UB gap (datablock units)

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(bTileCur);     // valid rows (batch tail)
    copyParams.blockLen = static_cast<uint32_t>(blkBytes);       // valid columns (cell tail)
    copyParams.srcStride = (td_->cellSize - cTileCur) * dtBytes; // GM row gap (bytes, row pitch C)
    copyParams.dstStride = ubRowGapBlk;                          // UB gap (datablocks)
    copyParams.rsv = 0;                                          // always explicitly 0

    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0}; // isPad=false

    const int64_t gmElemOff = bStartCur * td_->cellSize + cOff;
    auto inBase = inRegion_.template Get<T>();
    for (int32_t k = 0; k < NUM_IN_SLOTS; ++k) {
        AscendC::DataCopyPad(inBase[static_cast<int64_t>(k) * slotElems_], gmIn_[k][gmElemOff], copyParams, padParams);
    }
}

// -----------------------------------------------------------------------------
// CopyInVecRound — GM -> UB, the 3 peephole weight vector column slices
// (MTE2, USE_PEEPHOLE=true only, once per cTile round before the first bTile
// round's CopyIn; row-direction data reuse — every batch row of the tile
// reuses the same vector row)
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::CopyInVecRound(int64_t cOff, int64_t cTileCur)
{
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;                                         // (cell,) vector: single block
    copyParams.blockLen = static_cast<uint32_t>(cTileCur * sizeof(T)); // valid column width (bytes)
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    AscendC::DataCopyPad(vecWci_.template Get<T>(), gmWci_[cOff], copyParams, padParams);
    AscendC::DataCopyPad(vecWcf_.template Get<T>(), gmWcf_[cOff], copyParams, padParams);
    AscendC::DataCopyPad(vecWco_.template Get<T>(), gmWco_[cOff], copyParams, padParams);
}

// -----------------------------------------------------------------------------
// ComputeRound — per-row VF1..VF5 chain + tile-level VF6 peephole accumulation
// (row base = tile base + b * cTileAlign;
// segElems = cTileAlign -> full masks; the peephole vec/acc pointers are wired
// only for the USE_PEEPHOLE=true instances — the false instances never
// dereference them, the loads are if-constexpr-eliminated in the VF impls)
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::ComputeRound(int64_t bTileCur)
{
    __ubuf__ T* inBase = reinterpret_cast<__ubuf__ T*>(inRegion_.template Get<T>().GetPhyAddr());
    __ubuf__ float* workBase = reinterpret_cast<__ubuf__ float*>(workRegion_.template Get<float>().GetPhyAddr());
    __ubuf__ T* outBase = reinterpret_cast<__ubuf__ T*>(outRegion_.template Get<T>().GetPhyAddr());
    const int32_t slot = static_cast<int32_t>(slotElems_);
    __ubuf__ T* inCsPrev = inBase + SLOT_CS_PREV * slot;
    __ubuf__ T* inI = inBase + SLOT_I * slot;
    __ubuf__ T* inF = inBase + SLOT_F * slot;
    __ubuf__ T* inO = inBase + SLOT_O * slot;
    __ubuf__ T* inCi = inBase + SLOT_CI * slot;
    __ubuf__ T* inCo = inBase + SLOT_CO * slot;
    __ubuf__ T* inCsGrad = inBase + SLOT_CS_GRAD * slot;
    __ubuf__ T* inHGrad = inBase + SLOT_H_GRAD * slot;
    __ubuf__ float* workDoPre = workBase + WORK_SLOT_DO_PRE * slot;
    __ubuf__ float* workDcs = workBase + WORK_SLOT_DCS * slot;
    __ubuf__ float* workDiPre = workBase + WORK_SLOT_DI_PRE * slot;
    __ubuf__ float* workDciPre = workBase + WORK_SLOT_DCI_PRE * slot;
    __ubuf__ float* workDfPre = workBase + WORK_SLOT_DF_PRE * slot;
    __ubuf__ T* outCsPrevGrad = outBase + OUT_SLOT_CS_PREV_GRAD * slot;
    __ubuf__ T* outDicfoDi = outBase + (OUT_SLOT_DICFO_BASE + 0) * slot;
    __ubuf__ T* outDicfoDci = outBase + (OUT_SLOT_DICFO_BASE + 1) * slot;
    __ubuf__ T* outDicfoDf = outBase + (OUT_SLOT_DICFO_BASE + 2) * slot;
    __ubuf__ T* outDicfoDo = outBase + (OUT_SLOT_DICFO_BASE + 3) * slot;

    // Peephole wiring: the row-reuse vectors
    // vecWci/vecWcf/vecWco, the 9th tile input cs and the 3 fp32 accumulators.
    __ubuf__ T* inCs = nullptr;
    __ubuf__ T* wciVec = nullptr;
    __ubuf__ T* wcfVec = nullptr;
    __ubuf__ T* wcoVec = nullptr;
    __ubuf__ float* accWci = nullptr;
    __ubuf__ float* accWcf = nullptr;
    __ubuf__ float* accWco = nullptr;
    if constexpr (USE_PEEPHOLE) {
        inCs = inBase + SLOT_CS * slot;
        wciVec = reinterpret_cast<__ubuf__ T*>(vecWci_.template Get<T>().GetPhyAddr());
        wcfVec = reinterpret_cast<__ubuf__ T*>(vecWcf_.template Get<T>().GetPhyAddr());
        wcoVec = reinterpret_cast<__ubuf__ T*>(vecWco_.template Get<T>().GetPhyAddr());
        accWci = reinterpret_cast<__ubuf__ float*>(accWci_.template Get<float>().GetPhyAddr());
        accWcf = reinterpret_cast<__ubuf__ float*>(accWcf_.template Get<float>().GetPhyAddr());
        accWco = reinterpret_cast<__ubuf__ float*>(accWco_.template Get<float>().GetPhyAddr());
    }

    const uint32_t segElems = static_cast<uint32_t>(td_->cTileAlign); // padded width, full masks
    const int32_t rowStep = static_cast<int32_t>(td_->cTileAlign);    // UB row pitch (elements)

    for (int64_t b = 0; b < bTileCur; ++b) { // rows ascending (fixed order, determinism)
        const int32_t rowOff = static_cast<int32_t>(b) * rowStep;

        DoPreVf<T>(inHGrad + rowOff, inCo + rowOff, inO + rowOff, workDoPre + rowOff, segElems);
        DcsVf<T, USE_PEEPHOLE>(inCsGrad + rowOff, inHGrad + rowOff, inO + rowOff, inCo + rowOff, workDoPre + rowOff,
                               wcoVec, workDcs + rowOff, segElems);
        GateGradVf<T>(workDcs + rowOff, inCi + rowOff, inI + rowOff, inCsPrev + rowOff, inF + rowOff,
                      workDiPre + rowOff, workDciPre + rowOff, workDfPre + rowOff, segElems);
        CsPrevGradVf<T, USE_PEEPHOLE>(workDcs + rowOff, workDiPre + rowOff, workDfPre + rowOff, inF + rowOff, wciVec,
                                      wcfVec, outCsPrevGrad + rowOff, segElems);
        DicfoPostVf<T>(workDiPre + rowOff, workDciPre + rowOff, workDfPre + rowOff, workDoPre + rowOff,
                       outDicfoDi + rowOff, outDicfoDci + rowOff, outDicfoDf + rowOff, outDicfoDo + rowOff, segElems);
    }
    // VF6 (tile-level, after the row loop; USE_PEEPHOLE=true only):
    // accWci += cs_prev*di_pre; accWcf += cs_prev*df_pre; accWco += cs*do_pre
    // — rows ascending within the tile (fixed fp32 sequential order).
    if constexpr (USE_PEEPHOLE) {
        PeepAccumVf<T>(inCsPrev, inCs, workDiPre, workDfPre, workDoPre, accWci, accWcf, accWco, segElems,
                       static_cast<uint16_t>(td_->cTileAlign), static_cast<uint16_t>(bTileCur));
    }
}

// -----------------------------------------------------------------------------
// CopyOutRound — UB -> GM: cs_prev_grad tile + dicfo's four column blocks
// (single dense valid-length path, MTE3 has no padParams; padded region
// [cTileCur, cTileAlign) never reaches GM)
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::CopyOutRound(int64_t bStartCur, int64_t bTileCur,
                                                                              int64_t cOff, int64_t cTileCur)
{
    const int64_t dtBytes = static_cast<int64_t>(sizeof(T));
    const int64_t blkBytes = cTileCur * dtBytes;
    const int64_t blkAligned = (blkBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
    const int64_t ubRowGapBlk = (static_cast<int64_t>(td_->cTileAlign) * dtBytes - blkAligned) / BLOCK_BYTES;
    auto outBase = outRegion_.template Get<T>();

    // (c) cs_prev_grad tile: UB out slot 0 -> GM (row pitch C)
    AscendC::DataCopyExtParams ext;
    ext.blockCount = static_cast<uint16_t>(bTileCur);     // valid rows
    ext.blockLen = static_cast<uint32_t>(blkBytes);       // valid columns (bytes)
    ext.srcStride = ubRowGapBlk;                          // UB gap (datablocks)
    ext.dstStride = (td_->cellSize - cTileCur) * dtBytes; // GM row gap (bytes)
    ext.rsv = 0;
    AscendC::DataCopyPad(gmCsPrevGrad_[bStartCur * td_->cellSize + cOff], outBase[OUT_SLOT_CS_PREV_GRAD * slotElems_],
                         ext);

    // (d) dicfo column blocks g = 0..3 (di|dc|df|do): same block geometry,
    //     GM row pitch 4C, column-block base offset g*C within each row.
    AscendC::DataCopyExtParams dExt = ext;
    dExt.dstStride = (td_->dicfoWidth - cTileCur) * dtBytes; // GM row gap (row pitch 4C)
    for (int32_t g = 0; g < NUM_DICFO_BLOCKS; ++g) {
        AscendC::DataCopyPad(gmDicfo_[bStartCur * td_->dicfoWidth + static_cast<int64_t>(g) * td_->cellSize + cOff],
                             outBase[(OUT_SLOT_DICFO_BASE + g) * slotElems_], dExt);
    }
}

// -----------------------------------------------------------------------------
// PeepWriteOut — (P1') cTile round end: the three fp32 accumulators -> Cast ->
// T staging (out slots 0-2) -> direct write to this core's gmPeepOut_ column
// slice [cOff, +cTileCur) (USE_PEEPHOLE=true only). Each core
// walks the FULL batch for its own columns, so the accumulator chain IS the
// row-ascending serial single chain — fp32 bitwise; fp16 bitwise via
// the per-add rounding (every intermediate fp16-exact, the
// chain-tail Cast value-preserving).
// Sync: leading Wait<MTE3_V> pairs this round's last (9) Set — the staging
// slots are reused out-front storage and the last CopyOut's MTE3 read must be
// complete before the Cast (V) overwrites them; Set/Wait<V_MTE3> covers the
// Cast->MTE3 RAW; tail Set<MTE3_V> (non-last round) serves the next cTile
// round's (V2) out+staging WAR (MTE3 stream order carries all earlier reads).
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::PeepWriteOut(int64_t cOff, int64_t cTileCur,
                                                                              bool issueTailSet)
{
    __ubuf__ float* accWci = reinterpret_cast<__ubuf__ float*>(accWci_.template Get<float>().GetPhyAddr());
    __ubuf__ float* accWcf = reinterpret_cast<__ubuf__ float*>(accWcf_.template Get<float>().GetPhyAddr());
    __ubuf__ float* accWco = reinterpret_cast<__ubuf__ float*>(accWco_.template Get<float>().GetPhyAddr());
    __ubuf__ T* outBase = reinterpret_cast<__ubuf__ T*>(outRegion_.template Get<T>().GetPhyAddr());
    const int32_t slot = static_cast<int32_t>(slotElems_);
    __ubuf__ T* stageWci = outBase + OUT_SLOT_CS_PREV_GRAD * slot;     // out slot 0
    __ubuf__ T* stageWcf = outBase + (OUT_SLOT_DICFO_BASE + 0) * slot; // out slot 1
    __ubuf__ T* stageWco = outBase + (OUT_SLOT_DICFO_BASE + 1) * slot; // out slot 2

    // Staging WAR: last CopyOut's MTE3 read of out slots done (pairs this
    // round's last (9) Set) before the Cast overwrites slots 0-2.
    WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
    PeepOutCastVf<T>(accWci, accWcf, accWco, stageWci, stageWcf, stageWco, static_cast<uint32_t>(cTileCur));
    // Cast (V) -> staging RAW before MTE3 reads it.
    SetFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);
    WaitFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);

    AscendC::DataCopyExtParams ext;
    ext.blockCount = 1;
    ext.blockLen = static_cast<uint32_t>(cTileCur * sizeof(T)); // valid columns (bytes)
    ext.srcStride = 0;
    ext.dstStride = 0;
    ext.rsv = 0;
    auto outT = outRegion_.template Get<T>();
    AscendC::DataCopyPad(gmPeepOut_[0][cOff], outT[static_cast<int64_t>(OUT_SLOT_CS_PREV_GRAD) * slotElems_], ext);
    AscendC::DataCopyPad(gmPeepOut_[1][cOff], outT[static_cast<int64_t>(OUT_SLOT_DICFO_BASE) * slotElems_], ext);
    AscendC::DataCopyPad(gmPeepOut_[2][cOff], outT[static_cast<int64_t>(OUT_SLOT_DICFO_BASE + 1) * slotElems_], ext);
    // out front (and, via MTE3 stream order, the acc domain) WAR for the next
    // cTile round's (V2)/(V3); no consumer on the last round -> no Set.
    if (issueTailSet) {
        SetFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
    }
}

// -----------------------------------------------------------------------------
// Phase0ZeroFill — peep=false peephole-gradient outputs exact zero-fill:
// zero this core's cell
// slice [cStart_, cStart_+sliceLen_) of wci/wcf/wco_grad with +0.0, reusing
// the outDicfoDo slot (out slot 4) as the zeroOut buffer (mutually exclusive
// in time with the main compute's use of that slot). Empty slice -> skipped entirely
// (blockLen must never be 0). The tail Set<MTE3_V> is consumed by round (0,0)'s (3).
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::Phase0ZeroFill()
{
    if (sliceLen_ <= 0) {
        return;
    }
    const int64_t chunkW = td_->bTile * td_->cTileAlign; // slot width in elements
    auto outBase = outRegion_.template Get<T>();
    __ubuf__ T* zeroOut = reinterpret_cast<__ubuf__ T*>(outBase.GetPhyAddr()) +
                          (OUT_SLOT_DICFO_BASE + 3) * static_cast<int32_t>(slotElems_);
    const int64_t dtBytes = static_cast<int64_t>(sizeof(T));

    for (int64_t off = 0; off < sliceLen_; off += chunkW) {
        const int64_t left = sliceLen_ - off;
        const int64_t chunkLen = (chunkW < left) ? chunkW : left; // tail chunk valid length

        // Previous chunk's CopyOut finished reading the zeroOut slot -> V may
        // overwrite it (first chunk: slot's first write, no Wait).
        if (off > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
        }
        ZeroFillVf<T>(zeroOut, static_cast<uint32_t>(chunkLen)); // V write (+0.0, exact)
        SetFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);
        WaitFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);
        // One zero chunk -> three output tensors (MTE3; dense single block).
        AscendC::DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(chunkLen * dtBytes);
        ext.srcStride = 0;
        ext.dstStride = 0;
        ext.rsv = 0;
        for (int32_t t = 0; t < NUM_PEEP_OUTPUTS; ++t) {
            AscendC::DataCopyPad(gmPeepOut_[t][cStart_ + off], outBase[(OUT_SLOT_DICFO_BASE + 3) * slotElems_], ext);
        }
        // MTE3 finished reading the zeroOut slot (consumed by the next chunk's
        // Wait or by the main compute's round (0,0)'s (3) — this core's zero-fill being
        // non-empty is exactly the condition for that Wait to execute).
        SetFlag<AscendC::HardEvent::MTE3_V>(evMte3v_);
    }
}

// -----------------------------------------------------------------------------
// EmptyZeroFill — B==0 3x(H,) zero fill: single core (host
// guarantees usedCoreNum=1), fixed compile-time segment width, valid-length
// MTE3 writes; only the global cellSize field is consumed (never the
// degenerate split fields). Segment loop needs no MTE3_V pairing: every
// segment writes identical +0.0 over the same buffer, so V-overwrite vs
// MTE3-read interleavings still read zeros (idempotent fill).
// -----------------------------------------------------------------------------
template <typename T, bool USE_PEEPHOLE>
__aicore__ inline void LSTMBlockCellGradKernel<T, USE_PEEPHOLE>::EmptyZeroFill()
{
    __ubuf__ T* zeroSeg = reinterpret_cast<__ubuf__ T*>(zeroSeg_.template Get<T>().GetPhyAddr());
    const int64_t dtBytes = static_cast<int64_t>(sizeof(T));

    for (int32_t t = 0; t < NUM_PEEP_OUTPUTS; ++t) { // wci_grad / wcf_grad / wco_grad
        for (int64_t cOff = 0; cOff < td_->cellSize; cOff += EMPTY_ZERO_SEG_ELEMS) {
            const int64_t left = td_->cellSize - cOff;
            const int64_t segLen = (static_cast<int64_t>(EMPTY_ZERO_SEG_ELEMS) < left) ?
                                       static_cast<int64_t>(EMPTY_ZERO_SEG_ELEMS) :
                                       left; // tail segment valid width
            ZeroFillVf<T>(zeroSeg, static_cast<uint32_t>(segLen));
            SetFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);
            WaitFlag<AscendC::HardEvent::V_MTE3>(evVmte3_);
            AscendC::DataCopyExtParams ext;
            ext.blockCount = 1;
            ext.blockLen = static_cast<uint32_t>(segLen * dtBytes); // valid length, never OOB
            ext.srcStride = 0;
            ext.dstStride = 0;
            ext.rsv = 0;
            AscendC::DataCopyPad(gmPeepOut_[t][cOff], zeroSeg_.template Get<T>(), ext);
        }
    }
}

} // namespace NsLSTMBlockCellGrad

#endif // LSTM_BLOCK_CELL_GRAD_KERNEL_H_
