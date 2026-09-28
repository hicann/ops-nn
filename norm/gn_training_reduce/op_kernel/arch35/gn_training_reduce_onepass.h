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
 * \file gn_training_reduce_onepass.h
 * \brief GNTrainingReduce single-load dual-moment base (tilingKey=0) kernel.
 *
 * The baseline base kernel is invoked twice from apt.cpp (processIdx 0 -> Σx,
 * 1 -> Σx²), so every x element is loaded from GM twice. This class performs both
 * reductions in one A-chunk pass: each (A,R) tile is CopyIn'd once into preInBuf
 * and consumed by both Σx and Σx² before it is overwritten.
 *
 * Only the base (else) branch of apt.cpp uses this class; group (tilingKey=1) and
 * empty (tilingKey=2) keep their original classes unchanged.
 */

#ifndef GN_TRAINING_REDUCE_ONEPASS_H
#define GN_TRAINING_REDUCE_ONEPASS_H

#include "gn_training_reduce_base.h"

namespace NsGNTrainingReduce {

template <typename DType>
class GNTrainingReduceOnePassKernel : public GNTrainingReduceBaseKernel<DType> {
public:
    using DT = DType;
    using Base = GNTrainingReduceBaseKernel<DType>;

    __aicore__ inline GNTrainingReduceOnePassKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, const GNTrainingReduceTilingData* td,
                                AscendC::TPipe* pipe)
    {
        Base::Init(x, sum, squareSum, td, pipe);
    }

    // Scheme dual-moment-single-load-materialize: each (A,R) tile is loaded from
    // preInBuf exactly once by a single __simd_vf__ that casts fp16->fp32 once and
    // writes BOTH the Σx operand (Muls SUM_ACC_SCALE) and the Σx² operand (Mul)
    // into the existing fp32 operand buffers in the round-1 [A,R] layout, so the
    // two separate PreElewise passes collapse into one.  preInBuf_ is released
    // before either ReduceTile runs, so the next tile's CopyIn overlaps both
    // reductions.  Reduction calls / shapes / accumulate order stay identical to
    // the round-1 dual-moment path.
    __aicore__ inline void ProcessAll()
    {
        // ── small-R Σx accuracy guard ──
        // The original base kernel applies a compensated (double-double two-sum)
        // scalar reduction to Σx whenever the whole R segment fits in a single tile
        // and R <= COMPENSATED_SUM_MAX_R (Base::useCompensatedSum_), which is
        // required to meet the 2^-13 relative tolerance on catastrophic-cancellation
        // inputs (e.g. R=5, full fp16 range).  The single-pass dual-moment vector
        // tree reduce cannot reproduce that accuracy, so route exactly those cases
        // through the original two-pass base Process path (numerically identical to
        // the original baseline) and keep the one-pass fast path for every
        // non-trigger case.  This guard covers only the base (tilingKey=0) branch.
        if (Base::useCompensatedSum_) {
            Base::Process(0);
            Base::Process(1);
            return;
        }

        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(Base::td_->usedCoreNum)) {
            return;
        }

        int64_t aLoopStart = 0;
        int64_t aLoopEnd = 0;
        Base::UnravelBlockLoop(aLoopStart, aLoopEnd);

        const int64_t aSplitAxisSize = Base::td_->axisShape[Base::td_->aSplitIdx];
        const int64_t aSplitStride = Base::td_->axisStride[Base::td_->aSplitIdx];
        const int64_t aSplitOutStr = Base::outStride_[Base::td_->aSplitIdx];

        bool firstChunk = true;
        for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
            int64_t aIdx[MAX_PATTERN_RANK] = {0};
            int64_t aSplitChunkIdx = 0;
            Base::UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);

            int64_t chunkGmOff = 0;
            int64_t chunkOutOff = 0;
            for (int32_t k = Base::td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
                chunkGmOff += aIdx[k] * Base::td_->axisStride[k];
                chunkOutOff += aIdx[k] * Base::outStride_[k];
            }
            const int64_t aChunkStart = aSplitChunkIdx * Base::td_->aUbFactor;
            const int64_t aEnd = aChunkStart + Base::td_->aUbFactor;
            const int64_t aLen = (aEnd > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : Base::td_->aUbFactor;
            chunkGmOff += aChunkStart * aSplitStride;
            chunkOutOff += aChunkStart * aSplitOutStr;

            const uint32_t laneN = static_cast<uint32_t>(Base::td_->aUbFactor * Base::td_->innerAProdAlign);
            const uint32_t alignN = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
            const uint16_t repN = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));

            // previous chunk's CopyOut must have finished reading outBuf before this
            // chunk reuses it as per-tile scratch / output buffer.
            if (!firstChunk) {
                WaitFlag<HardEvent::MTE3_V>(Base::evMte3toV_);
            }

            auto cache = Base::cacheBuf_.template Get<float>();
            auto accSum = cache[0];
            auto accSq = cache[static_cast<int32_t>(alignN)];
            // zero the per-A-chunk accumulators
            AscendC::Duplicate(accSum, 0.0f, static_cast<int32_t>(alignN));
            AscendC::Duplicate(accSq, 0.0f, static_cast<int32_t>(alignN));

            RunDualWrite(chunkGmOff, aLen, accSum, accSq, laneN, alignN, firstChunk);

            // Σx output: accumulators hold Σ(SUM_ACC_SCALE * x); unscale on write-out.
            WriteMoment(accSum, chunkOutOff, aLen, laneN, repN, /*isSum=*/true, /*needWait=*/false);
            WriteMoment(accSq, chunkOutOff, aLen, laneN, repN, /*isSum=*/false, /*needWait=*/true);
            firstChunk = false;
        }
    }

private:
    // Out-of-line definition below (the register VF it calls must be declared at
    // namespace scope before the definition).
    __aicore__ inline void RunDualWrite(int64_t outerGmOff, int64_t aLen, const AscendC::LocalTensor<float>& accSum,
                                        const AscendC::LocalTensor<float>& accSq, uint32_t laneN, uint32_t alignN,
                                        bool firstChunk);

    __aicore__ inline void ReduceTile(const AscendC::LocalTensor<float>& src, const AscendC::LocalTensor<float>& dst,
                                      const AscendC::LocalTensor<uint8_t>& tmp, uint32_t laneN, uint32_t rProd,
                                      bool isTailR)
    {
        if (isTailR) {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {laneN, rProd};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(dst, src, tmp, srcShape,
                                                                                            /*srcInnerPad=*/true);
        } else {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {rProd, laneN};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(dst, src, tmp, srcShape,
                                                                                            /*srcInnerPad=*/true);
        }
    }

    __aicore__ inline void WriteMoment(const AscendC::LocalTensor<float>& acc, int64_t chunkOutOff, int64_t aLen,
                                       uint32_t laneN, uint16_t repN, bool isSum, bool needWait)
    {
        if (needWait) {
            WaitFlag<HardEvent::MTE3_V>(Base::evMte3toV_);
        }
        __ubuf__ float* accPtr = reinterpret_cast<__ubuf__ float*>(acc.GetPhyAddr());
        __ubuf__ float* outPtr = reinterpret_cast<__ubuf__ float*>(Base::outBuf_.template Get<float>().GetPhyAddr());
        const float scale = isSum ? SUM_ACC_UNSCALE : 1.0f;
        asc_vf_call<PostElewiseVfImpl>(accPtr, outPtr, laneN, repN, scale);

        SetFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);
        WaitFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);
        Base::CopyOut(chunkOutOff, aLen, isSum ? 0 : 1);
        SetFlag<HardEvent::MTE3_V>(Base::evMte3toV_);
    }
};

// ── Single-load dual-write register VF ──
// Per repeat: load one fp32 vector from the resident raw tile (fp16: one
// DIST_UNPACK_B16 load + one Cast), then write Σx = x * SUM_ACC_SCALE to dstSum
// and Σx² = x * x to dstSq.  Values are byte-identical to the two round-1
// PreElewise passes; only the load/cast is shared.
template <typename DType>
__simd_vf__ inline void DualMomentWriteVfImpl(__ubuf__ DType* src, __ubuf__ float* dstSum, __ubuf__ float* dstSq,
                                              uint32_t totalElems, uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::RegTensor<float> outReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        } else {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, CAST_TRAIT_TO_FP32>(f32Reg, b16Reg, mask);
        }
        AscendC::Reg::Muls(outReg, f32Reg, SUM_ACC_SCALE, mask);
        AscendC::Reg::StoreAlign(dstSum + off, outReg, mask);
        AscendC::Reg::Mul(outReg, f32Reg, f32Reg, mask);
        AscendC::Reg::StoreAlign(dstSq + off, outReg, mask);
    }
}

// ── Out-of-line RunDualWrite (round-1 reduction contract unchanged) ──
template <typename DType>
__aicore__ inline void GNTrainingReduceOnePassKernel<DType>::RunDualWrite(int64_t outerGmOff, int64_t aLen,
                                                                          const AscendC::LocalTensor<float>& accSum,
                                                                          const AscendC::LocalTensor<float>& accSq,
                                                                          uint32_t laneN, uint32_t alignN,
                                                                          bool firstChunk)
{
    const int64_t rLoop = Base::td_->rLoopCntTotal;
    const uint32_t rProd = static_cast<uint32_t>(Base::td_->rUbFactorAlign * Base::td_->innerRProdAlign);
    auto part = Base::outBuf_.template Get<float>(); // per-tile reduce dst while R loop runs
    auto tmp = Base::cacheBuf_.template Get<uint8_t>()[static_cast<int32_t>(2 * alignN * sizeof(float))];
    auto preResT = Base::preReduceResult_.template Get<float>();
    auto preResTailT = Base::preReduceResultTail_.template Get<float>();

    // cross-chunk WAR on preInBuf: previous chunk's last tile V-read must finish.
    if (!firstChunk) {
        WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);
    }

    for (int64_t rIdx = 0; rIdx < rLoop; ++rIdx) {
        if (rIdx != 0) {
            WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);
        }

        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkIdx = 0;
        int64_t rLen = 0;
        const int64_t rOff = Base::UnravelRLoop(rIdx, rOuterIdx, rChunkIdx, rLen);

        __ubuf__ DT* preIn = reinterpret_cast<__ubuf__ DT*>(Base::preInBuf_.template Get<DT>().GetPhyAddr());
        __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preResT.GetPhyAddr());
        __ubuf__ float* preResTail = reinterpret_cast<__ubuf__ float*>(preResTailT.GetPhyAddr());

        Base::DoCopyInTile(outerGmOff + rOff, aLen, rLen);

        SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
        WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);

        // ── one load + one cast, both moments materialized in a single VF pass ──
        const uint32_t totalElems = static_cast<uint32_t>(Base::td_->aUbFactor * Base::td_->innerAProdAlign *
                                                          Base::td_->rUbFactorAlign * Base::td_->innerRProdAlign);
        const uint16_t repeatTime = static_cast<uint16_t>(
            Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
        asc_vf_call<DualMomentWriteVfImpl<DT>>(preIn, preRes, preResTail, totalElems, repeatTime);

        Base::ClearChunkExtensionVf(preRes, rLen);
        if (Base::isTailR_) {
            Base::ClearInnerBurstTailPadVf(preRes, rLen);
        }
        Base::ClearChunkExtensionVf(preResTail, rLen);
        if (Base::isTailR_) {
            Base::ClearInnerBurstTailPadVf(preResTail, rLen);
        }

        // Both operands are materialized: preInBuf_ is no longer read, so release
        // it before EITHER reduction runs (next tile's CopyIn overlaps both).
        SetFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);

        // ── round-1 reduction contract, byte-identical calls ──
        ReduceTile(preResT, part, tmp, laneN, rProd, Base::isTailR_);
        AscendC::Add(accSum, accSum, part, static_cast<int32_t>(alignN));
        ReduceTile(preResTailT, part, tmp, laneN, rProd, Base::isTailR_);
        AscendC::Add(accSq, accSq, part, static_cast<int32_t>(alignN));
    }
}

} // namespace NsGNTrainingReduce

#endif // GN_TRAINING_REDUCE_ONEPASS_H
