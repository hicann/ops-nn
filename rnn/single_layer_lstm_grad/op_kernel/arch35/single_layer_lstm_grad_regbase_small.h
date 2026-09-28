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
 * \file single_layer_lstm_grad_regbase_small.h
 * \brief arch35 LSTM grad kernel: AIV-only, zero cross-core sync.
 *
 * Math (identical to the legacy membase pipeline, fp32 internal):
 *   dh_t   = dy_t + dh_next
 *   dc_t   = (1 - tanhct^2) * (o * dh_t) + dc_next
 *   do     = (o * dh_t) * tanhct * (1 - o)
 *   dj     = (dc_t * i) * (1 - j^2)
 *   di     = (1 - i) * j * (dc_t * i)
 *   df     = (1 - f) * f * dc_t * c_prev
 *   dc_prev= dc_t * f
 *   dh_next(next step) = dgate_t @ w[:, I:I+H]
 *   dx     = dgate @ w[:, 0:I]
 *   dw     = sum_t dgate_t^T @ [x_t | h_(t-1)] ; db = sum dgate
 *
 * Four axes are blocked; full hidden-state rows must still fit UB.
 *   time    tBlock timesteps staged at once; only dh / dc cross the cut
 *   batch   bBlock rows staged at once; nothing crosses the cut
 *   gate    gBlock rows of one slot at a time, for the three 4H-scaled buffers
 *   column  chunkCols columns at a time, for dx / dw and for the streamed w
 * The host picks all four against this file's own layout formula, so the two sides cannot
 * disagree about where anything lives.
 *
 * All floating IO use T. Narrow saved states are recomputed in private FP32 workspace;
 * FP32 calls consume their supplied states. Every accumulator remains FP32.
 * Output conversion happens after each output's reduction, not after each partial sum.
 *
 * Every UB row uses a 32B-aligned pitch (hAlignT / hAlignF elements) so all vector
 * loads/stores are aligned vlds/vsts; masks cover the H tail. dgate keeps one
 * hAlignF-pitched row per (m, gate-slot); slot order matches the w row layout.
 */

#ifndef SINGLE_LAYER_LSTM_GRAD_REGBASE_SMALL_H
#define SINGLE_LAYER_LSTM_GRAD_REGBASE_SMALL_H

#include "kernel_operator.h"
#include "single_layer_lstm_grad_regbase_tiling_data.h"
#include "../../single_layer_lstm/arch35/single_layer_lstm_gate_math.h"

namespace LstmGradRegbase {

namespace Micro = AscendC::MicroAPI;

/* Elements of fp32 one vector register holds -- every loop that walks the hidden axis steps by
 * this, and calls UpdateMask once per step (UpdateMask decrements its counter by exactly one
 * register's worth, so the element offset and the mask have to advance together). A hidden_size
 * larger than this is therefore several steps, not a refusal. */
constexpr uint32_t VL_F32 = AscendC::GetVecLen() / sizeof(float);
constexpr uint32_t LSTM_GATE_NUM = 4;
constexpr uint32_t UB_BLOCK_BYTES = 32; // DataCopyPad UB strides count 32-byte blocks
constexpr int32_t GATE_ORDER_IJFO = 0;
constexpr int32_t OFFSET_I = 0;
constexpr int32_t OFFSET_J = 1;
constexpr int32_t OFFSET_F = 2;
constexpr int32_t OFFSET_O = 3;
constexpr int64_t FP32_SIZE = 4; // the accumulators dw, db and dx are summed in are always fp32

constexpr Micro::CastTrait LSTM_CAST_UP_TRAIT = {
    Micro::RegLayout::ZERO,
    Micro::SatMode::UNKNOWN,
    Micro::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr Micro::CastTrait LSTM_CAST_DOWN_TRAIT = {
    Micro::RegLayout::ZERO,
    Micro::SatMode::NO_SAT,
    Micro::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// dtype-generic UB load into fp32 lanes (32B-aligned offset; masked-off lanes zeroed)
template <typename T>
__aicore__ inline void LoadF32(__local_mem__ T* src, Micro::RegTensor<float>& dst, Micro::MaskReg& mask,
                               uint32_t offset)
{
    if constexpr (std::is_same<T, float>::value) {
        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        Micro::RegTensor<T> tmp;
        Micro::DataCopy<T, Micro::LoadDist::DIST_UNPACK_B16>(tmp, src + offset);
        Micro::Cast<float, T, LSTM_CAST_UP_TRAIT>(dst, tmp, mask);
    }
}

// dtype-generic UB store from fp32 lanes (32B-aligned offset)
template <typename T>
__aicore__ inline void StoreF32(__local_mem__ T* dst, Micro::RegTensor<float>& src, Micro::MaskReg& mask,
                                uint32_t offset)
{
    if constexpr (std::is_same<T, float>::value) {
        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dst + offset, src, mask);
    } else {
        Micro::RegTensor<T> tmp;
        Micro::Cast<T, float, LSTM_CAST_DOWN_TRAIT>(tmp, src, mask);
        Micro::DataCopy<T, Micro::StoreDist::DIST_PACK_B32>(dst + offset, tmp, mask);
    }
}

template <AscendC::HardEvent EV>
__aicore__ inline void PipeSync()
{
    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(EV));
    AscendC::SetFlag<EV>(e);
    AscendC::WaitFlag<EV>(e);
}

template <typename T>
class LstmGradRegbaseSmall {
public:
    __aicore__ inline LstmGradRegbaseSmall() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR initH, GM_ADDR initC, GM_ADDR h, GM_ADDR c,
                                GM_ADDR dy, GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j, GM_ADDR f, GM_ADDR o,
                                GM_ADDR tanhct, GM_ADDR dw, GM_ADDR db, GM_ADDR dx, GM_ADDR dhPrev, GM_ADDR dcPrev,
                                GM_ADDR workspace, const LstmGradRegbaseSmallTilingData* tiling, AscendC::TPipe* pipe)
    {
        timeStep_ = static_cast<int32_t>(tiling->timeStep);
        batch_ = static_cast<int32_t>(tiling->batch);
        inputSize_ = static_cast<int32_t>(tiling->inputSize);
        hidden_ = static_cast<int32_t>(tiling->hiddenSize);
        isBias_ = tiling->isBias != 0;
        biasComponents_ = static_cast<int32_t>(tiling->biasComponents);
        backward_ = tiling->direction != 0;
        gateOrder_ = static_cast<int32_t>(tiling->gateOrder);
        usedCores_ = static_cast<int32_t>(tiling->usedCores);
        chunkCols_ = static_cast<int32_t>(tiling->chunkCols);
        mBlock_ = static_cast<int32_t>(tiling->mBlock);
        numIChunks_ = static_cast<int32_t>(tiling->numIChunks);
        gates_ = LSTM_GATE_NUM * hidden_;
        mAll_ = timeStep_ * batch_;
        /* 0 means "stage all of it", which is what a caller that predates the blocking passes and
         * which reproduces the original layout exactly. */
        tBlock_ = (tiling->tBlock > 0) ? static_cast<int32_t>(tiling->tBlock) : timeStep_;
        if (tBlock_ > timeStep_) {
            tBlock_ = timeStep_;
        }
        bBlock_ = (tiling->bBlock > 0) ? static_cast<int32_t>(tiling->bBlock) : batch_;
        if (bBlock_ > batch_) {
            bBlock_ = batch_;
        }
        gBlock_ = (tiling->gBlock > 0 && tiling->gBlock < hidden_) ? static_cast<int32_t>(tiling->gBlock) : hidden_;
        cols_ = inputSize_ + hidden_;
        blockIdx_ = static_cast<int32_t>(AscendC::GetBlockIdx());
        // physical slot order of j/f follows the w row layout selected by gate_order
        slotJ_ = (gateOrder_ == GATE_ORDER_IJFO) ? OFFSET_J : OFFSET_F;
        slotF_ = (gateOrder_ == GATE_ORDER_IJFO) ? OFFSET_F : OFFSET_J;

        layout_.Fill(tBlock_, bBlock_, hidden_, chunkCols_, mBlock_, sizeof(T), gBlock_);
        haT_ = static_cast<uint32_t>(layout_.hAlignT);
        haF_ = static_cast<uint32_t>(layout_.hAlignF);
        pipe->InitBuffer(ubBuf_, static_cast<uint32_t>(layout_.totalBytes));
        AscendC::LocalTensor<uint8_t> base = ubBuf_.Get<uint8_t>();
        ubBase_ = (__local_mem__ uint8_t*)base.GetPhyAddr();
        baseTensor_ = base;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(w));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
        dyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dy));
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dh));
        dcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dc));
        initHGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH));
        initCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC));
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(c));
        iGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(i));
        jGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(j));
        fGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(f));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(o));
        tanhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tanhct));
        dwGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dw));
        dbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(db));
        dxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dx));
        dhPrevGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dhPrev));
        dcPrevGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dcPrev));
        /* THE TIME AND BATCH SUMS FOR dw AND db ACCUMULATE AT fp32, NEVER AT THE OPERATOR'S WIDTH.
         *
         * Each block on either axis contributes a partial sum, and adding those at a narrow width
         * rounds every one of them: measured at T=64 B=2 I=128 H=128 bfloat16, mare reached 11.4 on
         * dw and 14.9 on db against a limit of 2.0, while the same shape at float32 passed. At
         * float the accumulator IS the output and this aliases it, so nothing is copied and nothing
         * is narrowed; at half and bfloat16 it is the caller's workspace, and each core
         * narrows the columns it owns into dw once, after its last block. */
        if constexpr (std::is_same<T, float>::value) {
            dwAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dw));
            dbAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(db));
        } else {
            dwAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
            dbAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + static_cast<int64_t>(gates_) * cols_);
            replayPlaneSize_ = static_cast<int64_t>(timeStep_) * bBlock_ * hidden_;
            replayGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) +
                                      static_cast<int64_t>(gates_) * (cols_ + 1) +
                                      blockIdx_ * REPLAY_PLANES * replayPlaneSize_);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCores_ || mAll_ <= 0 || hidden_ <= 0) {
            return;
        }
        /* Two nested block walks: the batch outside, the sequence inside.
         *
         * Nothing crosses a batch cut -- h_t[b] depends only on h_{t-1}[b] -- so a batch block seeds
         * its own dh / dc, runs the whole sequence, and writes its own rows of dx, dh_prev and
         * dc_prev. Across a time cut only dh and dc carry: dx[t] depends on dgate[t] alone, and
         * dw / db are reductions over t. `lt` always runs T-1 down to 0 whichever direction the
         * layer has, so blocking on it keeps one rule for both; the block's rows in GM are an actT
         * range, which BlockActBase() is the one place to know.
         *
         * dw accumulates into fp32 GM after the first block. The bias owner keeps its compensated
         * sum in UB across both walks and writes db once. No caller-side initialization is needed.
         * dx is written once per row and needs no accumulation. */
        for (int32_t b0 = 0; b0 < batch_; b0 += bBlock_) {
            b0_ = b0;
            bCur_ = ((batch_ - b0) < bBlock_) ? (batch_ - b0) : bBlock_;
            if constexpr (!std::is_same<T, float>::value) {
                ReplayForward();
            }
            StageBatchState();
            Prologue();
            parity_ = 0;
            for (int32_t lt0 = ((timeStep_ - 1) / tBlock_) * tBlock_; lt0 >= 0; lt0 -= tBlock_) {
                blkLt0_ = lt0;
                blkSteps_ = ((timeStep_ - lt0) < tBlock_) ? (timeStep_ - lt0) : tBlock_;
                blkRows_ = blkSteps_ * bCur_;
                blkA0_ = BlockActBase();
                StageBlock();
                ProcessChain();
                ProcessColumns();
                if (blockIdx_ == usedCores_ - 1) {
                    ProcessTail();
                }
            }
            if (blockIdx_ == usedCores_ - 1) {
                WritePrevState();
            }
        }
        /* THE ONE NARROWING, AFTER EVERY BLOCK ON BOTH AXES. Everything above accumulated at fp32;
         * this is where dw and db reach the operator's own width. Compiled away at float, where the
         * accumulators ARE the outputs. Each core narrows only the columns it owns, so there is
         * nothing to synchronise. */
        if constexpr (!std::is_same<T, float>::value) {
            FinishColumns();
            if (blockIdx_ == usedCores_ - 1) {
                FinishTailColumns();
                FinishDb();
            }
        }
    }

private:
    template <typename U>
    __aicore__ inline __local_mem__ U* UbPtr(int64_t byteOff)
    {
        return (__local_mem__ U*)(ubBase_ + byteOff);
    }

    template <typename U>
    __aicore__ inline AscendC::LocalTensor<U> UbTensor(int64_t byteOff)
    {
        return baseTensor_[byteOff].template ReinterpretCast<U>();
    }

    /* First actT of the current time block. Forward walks actT down with lt, so the block's lowest
     * actT IS lt0; REDIRECTIONAL walks actT up, so it is measured from the far end. */
    __aicore__ inline int32_t BlockActBase() const { return backward_ ? (timeStep_ - blkLt0_ - blkSteps_) : blkLt0_; }

    /* HOW MANY BLOCK-LOCAL ROWS STARTING AT m0 ARE CONTIGUOUS IN GM.
     *
     * GM rows are (t, b) with b fastest. With the whole batch staged, a block's rows are one
     * contiguous run. With a batch block they are one column range per timestep, so a run stops at
     * the end of the timestep it started in -- the column phase stages and stores per run, which is
     * what keeps its DataCopyPad strides constant. Every row of the block is in UB at once either
     * way; this only decides how many bursts that takes. */
    __aicore__ inline int32_t GmRowRun(int32_t m0) const
    {
        const int32_t avail = blkRows_ - m0;
        if (bCur_ == batch_) {
            return avail;
        }
        const int32_t toEnd = bCur_ - (m0 % bCur_);
        return (toEnd < avail) ? toEnd : avail;
    }

    // GM row index of the block-local row m0
    __aicore__ inline int64_t BlockGmRow(int32_t m0) const
    {
        if (bCur_ == batch_) {
            return static_cast<int64_t>(blkA0_) * batch_ + m0;
        }
        return static_cast<int64_t>(blkA0_ + m0 / bCur_) * batch_ + b0_ + (m0 % bCur_);
    }

    /* `gmRow0` is a ROW index into a [rows, rowElems] GM matrix; the blocking is the only thing
     * that passes anything but 0. UB rows are auto-rounded to a 32B pitch. */
    template <typename U>
    __aicore__ inline void CopyInRows(int64_t ubOff, const AscendC::GlobalTensor<U>& gm, int64_t rows, int64_t rowElems,
                                      int64_t gmRow0 = 0)
    {
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(rowElems * sizeof(U)), 0,
                                          0, 0};
        AscendC::DataCopyPadExtParams<U> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(UbTensor<U>(ubOff), gm[gmRow0 * rowElems], params, pad);
    }

    // Reuse outStage for one narrow row. Its allocation covers both a hidden row and a column chunk.
    // Each row is widened before scratch reuse; no FP32 saved-state tensor crosses the op boundary.
    __aicore__ inline void CopyInRowsF32(int64_t ubOff, const AscendC::GlobalTensor<T>& gm, int64_t rows,
                                         int64_t rowElems, int64_t gmElem0, int64_t gmPitch)
    {
        if constexpr (std::is_same<T, float>::value) {
            AscendC::DataCopyExtParams p{static_cast<uint16_t>(rows), static_cast<uint32_t>(rowElems * sizeof(T)),
                                         static_cast<uint32_t>((gmPitch - rowElems) * sizeof(T)), 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(UbTensor<float>(ubOff), gm[gmElem0], p, pad);
        } else {
            AscendC::PipeBarrier<PIPE_ALL>();
            const int64_t dstPitch = AlignUpI64(rowElems * FP32_SIZE, UB_BLOCK_BYTES);
            for (int64_t row = 0; row < rows; ++row) {
                PipeSync<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(rowElems * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                AscendC::DataCopyPad(UbTensor<T>(layout_.outStageOff), gm[gmElem0 + row * gmPitch], p, pad);
                PipeSync<AscendC::HardEvent::MTE2_V>();
                __local_mem__ T* src = UbPtr<T>(layout_.outStageOff);
                __local_mem__ float* dst = UbPtr<float>(ubOff + row * dstPitch);
                const uint32_t count = static_cast<uint32_t>(rowElems);
                const uint16_t steps = static_cast<uint16_t>((count + VL_F32 - 1) / VL_F32);
                __VEC_SCOPE__
                {
                    uint32_t remaining = count;
                    for (uint16_t k = 0; k < steps; ++k) {
                        Micro::MaskReg mask = Micro::UpdateMask<float>(remaining);
                        Micro::RegTensor<float> value;
                        LoadF32<T>(src, value, mask, k * VL_F32);
                        StoreF32<float>(dst, value, mask, k * VL_F32);
                    }
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    /* One time block's worth of ONE plane: blkSteps_ timesteps by bCur_ batch rows. With the whole
     * batch staged those rows are contiguous and it is one burst; otherwise it is one burst per
     * timestep, which is what DataCopyPad can express with a constant stride. */
    __aicore__ inline void CopyInBlockPlane(int64_t ubOff, const AscendC::GlobalTensor<T>& gm, int64_t rowElems,
                                            int32_t actT0)
    {
        if (bCur_ == batch_) {
            CopyInRowsF32(ubOff, gm, blkRows_, rowElems, static_cast<int64_t>(actT0) * batch_ * rowElems, rowElems);
            return;
        }
        const int64_t ubRowBytes = AlignUpI64(rowElems * FP32_SIZE, UB_BLOCK_BYTES);
        for (int32_t lt = 0; lt < blkSteps_; ++lt) {
            CopyInRowsF32(ubOff + static_cast<int64_t>(lt) * bCur_ * ubRowBytes, gm, bCur_, rowElems,
                          (static_cast<int64_t>(actT0 + lt) * batch_ + b0_) * rowElems, rowElems);
        }
    }

    // Staged once per BATCH block: the incoming dh / dc for this block's rows.
    __aicore__ inline void StageBatchState()
    {
        PipeSync<AscendC::HardEvent::V_MTE2>();
        CopyInRows(layout_.dh0Off, dhGm_, bCur_, hidden_, b0_);
        CopyInRows(layout_.dc0Off, dcGm_, bCur_, hidden_, b0_);
        PipeSync<AscendC::HardEvent::MTE2_V>();
    }

    /* w[:, col0:col0+w] for gate rows [gRow0, gRow0+gCur) -> wChunkOff, out of a cols_-pitched GM
     * matrix, UB pitch AlignUp(w * sizeof(T), 32). Three callers stage through this: the input
     * columns, the recurrent columns the dh_next product streams in one step at a time, and the
     * hidden columns of dw. */
    __aicore__ inline void StageWChunk(int32_t col0, int32_t w, int32_t gRow0, int32_t gCur)
    {
        AscendC::DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(gCur);
        p.blockLen = static_cast<uint32_t>(w * sizeof(T));
        p.srcStride = static_cast<uint32_t>((cols_ - w) * sizeof(T));
        p.dstStride = 0; // auto 32B rounding -> wAlign pitch
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(UbTensor<T>(layout_.wChunkOff), wGm_[static_cast<int64_t>(gRow0) * cols_ + col0], p, pad);
    }

    // Compute one affine gate row tile using FP32 products/reductions. Only the
    // input and weight storage are narrow; recurrent h never leaves FP32 here.
    __aicore__ inline void ReplayDot(int32_t actT, int32_t b, int32_t slot, int64_t gateOff)
    {
        for (int32_t source = 0; source < 2; ++source) {
            const int32_t width = source == 0 ? inputSize_ : hidden_;
            for (int32_t c0 = 0; c0 < width; c0 += chunkCols_) {
                const int32_t cw = width - c0 < chunkCols_ ? width - c0 : chunkCols_;
                int64_t inputOff = layout_.initHOff + (static_cast<int64_t>(b) * haF_ + c0) * sizeof(float);
                if (source == 0) {
                    CopyInRowsF32(layout_.xChunkOff, xGm_, 1, cw,
                                  (static_cast<int64_t>(actT) * batch_ + b0_ + b) * inputSize_ + c0, cw);
                    inputOff = layout_.xChunkOff;
                }
                for (int32_t g0 = 0; g0 < hidden_; g0 += gBlock_) {
                    const int32_t gc = hidden_ - g0 < gBlock_ ? hidden_ - g0 : gBlock_;
                    PipeSync<AscendC::HardEvent::V_MTE2>();
                    StageWChunk((source == 0 ? 0 : inputSize_) + c0, cw, slot * hidden_ + g0, gc);
                    PipeSync<AscendC::HardEvent::MTE2_V>();
                    auto* weights = UbPtr<T>(layout_.wChunkOff);
                    auto* input = UbPtr<float>(inputOff);
                    auto* gate = UbPtr<float>(gateOff);
                    const uint32_t weightPitch = AlignUpI64(cw * sizeof(T), UB_BLOCK_BYTES) / sizeof(T);
                    const uint16_t rows = static_cast<uint16_t>(gc);
                    const uint32_t columns = cw;
                    const uint32_t gateStart = g0;
                    __VEC_SCOPE__
                    {
                        Micro::MaskReg one = Micro::CreateMask<float, Micro::MaskPattern::VL1>();
                        uint32_t count = columns;
                        Micro::MaskReg mask = Micro::UpdateMask<float>(count);
                        Micro::RegTensor<float> x, w, product, partial, sum;
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(x, input);
                        for (uint16_t g = 0; g < rows; ++g) {
                            LoadF32<T>(weights, w, mask, g * weightPitch);
                            Micro::Mul(product, x, w, mask);
                            Micro::ReduceSum(partial, product, mask);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sum, gate + gateStart + g);
                            Micro::Add(sum, sum, partial, one);
                            Micro::DataCopy<float, Micro::StoreDist::DIST_FIRST_ELEMENT_B32>(gate + gateStart + g, sum,
                                                                                             one);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void ReplayForward()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        CopyInRowsF32(layout_.initHOff, initHGm_, bCur_, hidden_, static_cast<int64_t>(b0_) * hidden_, hidden_);
        CopyInRowsF32(layout_.initCOff, initCGm_, bCur_, hidden_, static_cast<int64_t>(b0_) * hidden_, hidden_);
        const int64_t offsets[REPLAY_PLANES] = {layout_.igOff,   layout_.jgOff,    layout_.fgOff,   layout_.ogOff,
                                                layout_.tanhOff, layout_.initCOff, layout_.initHOff};
        const int32_t slots[LSTM_GATE_NUM] = {OFFSET_I, slotJ_, slotF_, OFFSET_O};
        // Gate helpers clear comparison tails starting at n. Use the padded UB
        // row width so that address is 32B aligned; only hidden_ lanes reach GM.
        const uint32_t gateElements = haF_;
        for (int32_t step = 0; step < timeStep_; ++step) {
            const int32_t actT = backward_ ? timeStep_ - 1 - step : step;
            for (int32_t b = 0; b < bCur_; ++b) {
                for (int32_t gate = 0; gate < static_cast<int32_t>(LSTM_GATE_NUM); ++gate) {
                    auto dst = UbTensor<float>(offsets[gate]);
                    AscendC::Duplicate(dst, 0.0f, gateElements);
                    // The two original biases are widened separately. Fused [4H]
                    // bias callers remain supported without changing their ABI.
                    for (int32_t part = 0; part < biasComponents_; ++part) {
                        CopyInRowsF32(layout_.dbStageOff, biasGm_, 1, hidden_,
                                      static_cast<int64_t>(part) * gates_ + slots[gate] * hidden_, hidden_);
                        AscendC::Add(dst, dst, UbTensor<float>(layout_.dbStageOff), hidden_);
                    }
                    ReplayDot(actT, b, slots[gate], offsets[gate]);
                }
                const int64_t stateOff = static_cast<int64_t>(b) * haF_ * sizeof(float);
                SingleLayerLstmVec::SingleLayerLstmGates(
                    UbTensor<float>(layout_.igOff), UbTensor<float>(layout_.fgOff), UbTensor<float>(layout_.jgOff),
                    UbTensor<float>(layout_.ogOff), UbTensor<float>(layout_.initCOff + stateOff),
                    UbTensor<float>(layout_.initHOff + stateOff), UbTensor<float>(layout_.tanhOff),
                    UbTensor<float>(layout_.replayScratchOff),
                    UbTensor<float>(layout_.replayScratchOff + layout_.replayPitch),
                    UbTensor<float>(layout_.replayScratchOff + 2 * layout_.replayPitch),
                    UbTensor<uint8_t>(layout_.replayMaskOff), gateElements);
                PipeSync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(hidden_ * sizeof(float)), 0, 0, 0};
                for (int32_t plane = 0; plane < REPLAY_PLANES; ++plane) {
                    const int64_t src = offsets[plane] + (plane >= 5 ? stateOff : 0);
                    const int64_t dst = plane * replayPlaneSize_ + (static_cast<int64_t>(actT) * bBlock_ + b) * hidden_;
                    AscendC::DataCopyPad(replayGm_[dst], UbTensor<float>(src), p);
                }
                // A following batch row reuses gate UB; a following time step
                // reads the updated recurrent state. Order both against DMA.
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void StageReplayBlock()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        CopyInBlockPlane(layout_.dyOff, dyGm_, hidden_, blkA0_);
        const int64_t offsets[REPLAY_PLANES] = {layout_.igOff,   layout_.jgOff, layout_.fgOff, layout_.ogOff,
                                                layout_.tanhOff, layout_.cOff,  layout_.hOff};
        for (int32_t plane = 0; plane < REPLAY_PLANES; ++plane) {
            for (int32_t t = 0; t < blkSteps_; ++t) {
                CopyInRows(offsets[plane] + static_cast<int64_t>(t) * bCur_ * haF_ * sizeof(float), replayGm_, bCur_,
                           hidden_, plane * replayPlaneSize_ / hidden_ + static_cast<int64_t>(blkA0_ + t) * bBlock_);
            }
        }
        if (blkLt0_ == 0) {
            CopyInRowsF32(layout_.initHOff, initHGm_, bCur_, hidden_, static_cast<int64_t>(b0_) * hidden_, hidden_);
            CopyInRowsF32(layout_.initCOff, initCGm_, bCur_, hidden_, static_cast<int64_t>(b0_) * hidden_, hidden_);
        } else {
            const int32_t outsideT = backward_ ? blkA0_ + blkSteps_ : blkA0_ - 1;
            const int64_t row = static_cast<int64_t>(outsideT) * bBlock_;
            CopyInRows(layout_.initHOff, replayGm_, bCur_, hidden_, 6 * replayPlaneSize_ / hidden_ + row);
            CopyInRows(layout_.initCOff, replayGm_, bCur_, hidden_, 5 * replayPlaneSize_ / hidden_ + row);
        }
        PipeSync<AscendC::HardEvent::MTE2_V>();
    }

    /* Staged per time block: the eight planes for this block's rows, plus the ONE state that sits
     * just outside it.
     *
     * `initHOff` / `initCOff` hold "the h and c one step before this block", which at the first
     * block IS init_h / init_c and at every later one is a row of the h / c histories. Reusing
     * those two regions rather than adding a pair keeps every reader unchanged -- ProcessChain's
     * `useInit` arm and ProcessTail's `useInitH` arm already mean "reach outside". */
    __aicore__ inline void StageBlock()
    {
        if constexpr (!std::is_same<T, float>::value) {
            StageReplayBlock();
            return;
        }
        /* Wait for the previous block to stop reading these regions before overwriting them. Every
         * plane below is re-staged per block into the same UB address, and the previous block's last
         * reader is ProcessTail's dw[:, I:I+H] loop, which nothing else orders against this DMA.
         * Measured on T=4 B=1 I=8 H=8 with two blocks: dw[:, I:I+H] came out built from init_h and
         * h[0] for steps 2 and 3 -- exactly the rows the following block stages -- while dx,
         * dw[:, :I], db, dh_prev and dc_prev were correct to 5e-7. */
        PipeSync<AscendC::HardEvent::V_MTE2>();
        CopyInBlockPlane(layout_.dyOff, dyGm_, hidden_, blkA0_);
        CopyInBlockPlane(layout_.igOff, iGm_, hidden_, blkA0_);
        CopyInBlockPlane(layout_.jgOff, jGm_, hidden_, blkA0_);
        CopyInBlockPlane(layout_.fgOff, fGm_, hidden_, blkA0_);
        CopyInBlockPlane(layout_.ogOff, oGm_, hidden_, blkA0_);
        CopyInBlockPlane(layout_.tanhOff, tanhGm_, hidden_, blkA0_);
        CopyInBlockPlane(layout_.cOff, cGm_, hidden_, blkA0_);
        CopyInBlockPlane(layout_.hOff, hGm_, hidden_, blkA0_);
        /* The step just outside the block, on the side the recurrence came from. At lt0 == 0 that
         * is the caller's initial state; otherwise it is a row of the histories. */
        if (blkLt0_ == 0) {
            CopyInRowsF32(layout_.initHOff, initHGm_, bCur_, hidden_, static_cast<int64_t>(b0_) * hidden_, hidden_);
            CopyInRowsF32(layout_.initCOff, initCGm_, bCur_, hidden_, static_cast<int64_t>(b0_) * hidden_, hidden_);
        } else {
            const int32_t outsideT = backward_ ? (blkA0_ + blkSteps_) : (blkA0_ - 1);
            const int64_t outRow = static_cast<int64_t>(outsideT) * batch_ + b0_;
            CopyInRowsF32(layout_.initHOff, hGm_, bCur_, hidden_, outRow * hidden_, hidden_);
            CopyInRowsF32(layout_.initCOff, cGm_, bCur_, hidden_, outRow * hidden_, hidden_);
        }
        PipeSync<AscendC::HardEvent::MTE2_V>();
    }

    // cast dh0/dc0 into fp32 ping buffers (parity 0), row-wise
    __aicore__ inline void Prologue()
    {
        __local_mem__ T* dh0 = UbPtr<T>(layout_.dh0Off);
        __local_mem__ T* dc0 = UbPtr<T>(layout_.dc0Off);
        __local_mem__ float* dhCur = UbPtr<float>(layout_.dhCurOff);
        __local_mem__ float* dcCur = UbPtr<float>(layout_.dcCurOff);
        const uint16_t B = static_cast<uint16_t>(bCur_);
        const uint32_t H = static_cast<uint32_t>(hidden_);
        const uint32_t haT = haT_;
        const uint32_t haF = haF_;
        const uint16_t hSteps = static_cast<uint16_t>((hidden_ + VL_F32 - 1) / VL_F32);
        __VEC_SCOPE__
        {
            Micro::RegTensor<float> r;
            for (uint16_t b = 0; b < B; ++b) {
                uint32_t maskCntH = H;
                for (uint16_t k = 0; k < hSteps; ++k) {
                    Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                    uint32_t ho = static_cast<uint32_t>(k) * VL_F32;
                    uint32_t srcOff = static_cast<uint32_t>(b) * haT + ho;
                    uint32_t dstOff = static_cast<uint32_t>(b) * haF + ho;
                    LoadF32<T>(dh0, r, mH, srcOff);
                    Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dhCur + dstOff, r, mH);
                    LoadF32<T>(dc0, r, mH, srcOff);
                    Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dcCur + dstOff, r, mH);
                }
            }
        }
    }

    __aicore__ inline void ProcessChain()
    {
        __local_mem__ float* dyU = UbPtr<float>(layout_.dyOff);
        __local_mem__ float* iU = UbPtr<float>(layout_.igOff);
        __local_mem__ float* jU = UbPtr<float>(layout_.jgOff);
        __local_mem__ float* fU = UbPtr<float>(layout_.fgOff);
        __local_mem__ float* oU = UbPtr<float>(layout_.ogOff);
        __local_mem__ float* tanhU = UbPtr<float>(layout_.tanhOff);
        __local_mem__ float* cU = UbPtr<float>(layout_.cOff);
        __local_mem__ float* initCU = UbPtr<float>(layout_.initCOff);
        __local_mem__ T* wChunkU = UbPtr<T>(layout_.wChunkOff);
        __local_mem__ float* dgateU = UbPtr<float>(layout_.dgateOff);
        __local_mem__ float* dhBase = UbPtr<float>(layout_.dhCurOff);
        __local_mem__ float* dcBase = UbPtr<float>(layout_.dcCurOff);

        const uint32_t H = static_cast<uint32_t>(hidden_);
        const uint32_t haT = haT_;
        const uint32_t haF = haF_;
        const uint32_t bhF = static_cast<uint32_t>(bCur_) * haF;
        const uint16_t B = static_cast<uint16_t>(bCur_);
        const uint32_t sI = static_cast<uint32_t>(OFFSET_I) * haF;
        const uint32_t sJ = static_cast<uint32_t>(slotJ_) * haF;
        const uint32_t sF = static_cast<uint32_t>(slotF_) * haF;
        const uint32_t sO = static_cast<uint32_t>(OFFSET_O) * haF;
        const uint16_t hSteps = static_cast<uint16_t>((hidden_ + VL_F32 - 1) / VL_F32);

        /* THE PING-PONG PARITY CARRIES ACROSS TIME BLOCKS, so it lives in the object rather than
         * here -- the first step of a block must read the dh / dc the last step of the previous
         * block wrote. Prologue() seeds it once per batch block. */
        int32_t parity = parity_;
        const int32_t ltEnd = blkLt0_;
        for (int32_t lt = blkLt0_ + blkSteps_ - 1; lt >= ltEnd; --lt) {
            const int32_t actT = backward_ ? (timeStep_ - 1 - lt) : lt;
            /* "Reach outside the block" -- at lt == 0 that is the caller's initial state and
             * otherwise the history row StageBlock put in the same place. */
            const bool useInit = (lt == ltEnd);
            const int32_t cPrevT = (backward_ ? (actT + 1) : (actT - 1)) - blkA0_;
            __local_mem__ float* cPrevBase = useInit ? initCU : (cU + static_cast<int64_t>(cPrevT) * bCur_ * haF);
            __local_mem__ float* dhSrc = dhBase + static_cast<uint32_t>(parity) * bhF;
            __local_mem__ float* dcSrc = dcBase + static_cast<uint32_t>(parity) * bhF;
            __local_mem__ float* dhDst = dhBase + static_cast<uint32_t>(1 - parity) * bhF;
            __local_mem__ float* dcDst = dcBase + static_cast<uint32_t>(1 - parity) * bhF;
            // UB rows are block-local: StageBlock copied actT range [blkA0_, blkA0_+blkSteps_).
            const uint32_t lr = static_cast<uint32_t>(actT - blkA0_);
            const uint32_t rowBaseS = lr * static_cast<uint32_t>(bCur_) * haF; // dy and the saved planes
            const uint32_t gateBase = lr * static_cast<uint32_t>(bCur_) * LSTM_GATE_NUM * haF;

            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                Micro::RegTensor<float> one;
                Micro::Duplicate(one, 1.0f);
                for (uint16_t b = 0; b < B; ++b) {
                    /* THE HIDDEN AXIS IS WALKED ONE REGISTER AT A TIME. Every vector below is a row
                     * of hidden_size, which is an operator input and not bounded by the register
                     * width; UpdateMask hands out one register's worth per call, so the element
                     * offset `ho` advances in step with it. */
                    uint32_t maskCntH = H;
                    for (uint16_t k = 0; k < hSteps; ++k) {
                        Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                        uint32_t ho = static_cast<uint32_t>(k) * VL_F32;
                        uint32_t rs = rowBaseS + static_cast<uint32_t>(b) * haF + ho;
                        uint32_t bo = static_cast<uint32_t>(b) * haF + ho;
                        uint32_t go = gateBase + static_cast<uint32_t>(b) * LSTM_GATE_NUM * haF + ho;
                        Micro::RegTensor<float> dyR, iR, jR, fR, oR, tanhR, cPrevR, dhN, dcN;
                        Micro::RegTensor<float> dht, tmpC, dcT, tmpJ, t0, dI, dJ, dF, dO, dcOut;
                        LoadF32<float>(dyU, dyR, mH, rs);
                        LoadF32<float>(iU, iR, mH, rs);
                        LoadF32<float>(jU, jR, mH, rs);
                        LoadF32<float>(fU, fR, mH, rs);
                        LoadF32<float>(oU, oR, mH, rs);
                        LoadF32<float>(tanhU, tanhR, mH, rs);
                        LoadF32<float>(cPrevBase, cPrevR, mH, static_cast<uint32_t>(b) * haF + ho);
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(dhN, dhSrc + bo);
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(dcN, dcSrc + bo);
                        /* 1 - x*x, plainly. Two more accurate forms were measured and neither pays
                         * here. The concern is real in principle: where tanh saturates, `1 - x*x`
                         * cancels because x*x rounds relative to 1 while the difference is tiny.
                         * Measured at fp32 against a float64 reference over 4000 draws, binned by
                         * the pre-activation |u| (median relative error of the difference):
                         *
                         *   |u|          [0,1)   [1,2)   [2,3)   [3,4)   [4,6)
                         *   1-x*x       1.9e-8  7.4e-8  4.9e-7  3.6e-6  2.2e-5
                         *   (1-x)(1+x)  2.7e-8  2.4e-8  2.3e-8  2.4e-8  2.3e-8
                         *
                         * On device, over the three shapes whose only failure is dw_hh / dw_ih,
                         * (1-x)(1+x) was better on 3 of 9 outputs and worse on 5, and 1 - x*x by FMA
                         * bit-identical on 5, better on 2, worse on 2 -- both costing one operation
                         * more per element in the recurrence's inner loop. The reason is the data:
                         * every failing case sits in the unsaturated band where the product form is
                         * 1.4x worse. Revisit only with a saturated case that fails here. */
                        // dh_t, dc_t
                        Micro::Add(dht, dyR, dhN, mH);
                        Micro::Mul(tmpC, oR, dht, mH);
                        Micro::Mul(t0, tanhR, tanhR, mH);
                        Micro::Sub(t0, one, t0, mH);
                        Micro::Mul(dcT, t0, tmpC, mH);
                        Micro::Add(dcT, dcT, dcN, mH);
                        // do
                        Micro::Sub(t0, one, oR, mH);
                        Micro::Mul(dO, tmpC, tanhR, mH);
                        Micro::Mul(dO, dO, t0, mH);
                        // dj -- 1 - j*j, see the note at the tanh derivative above
                        Micro::Mul(tmpJ, dcT, iR, mH);
                        Micro::Mul(t0, jR, jR, mH);
                        Micro::Sub(t0, one, t0, mH);
                        Micro::Mul(dJ, tmpJ, t0, mH);
                        // di
                        Micro::Sub(t0, one, iR, mH);
                        Micro::Mul(dI, t0, jR, mH);
                        Micro::Mul(dI, dI, tmpJ, mH);
                        // df
                        Micro::Sub(t0, one, fR, mH);
                        Micro::Mul(dF, t0, fR, mH);
                        Micro::Mul(dF, dF, dcT, mH);
                        Micro::Mul(dF, dF, cPrevR, mH);
                        // dc_prev
                        Micro::Mul(dcOut, dcT, fR, mH);
                        // store dgate slots + recurrent dc
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dgateU + go + sI, dI, mH);
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dgateU + go + sJ, dJ, mH);
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dgateU + go + sF, dF, mH);
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dgateU + go + sO, dO, mH);
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dcDst + bo, dcOut, mH);
                    }
                }
            }

            /* dh_next for the following step: dh[b][:] = sum_g dgate[b][g] * w_h[g][:], walked in column
             * and gate chunks with W_hh streamed in. `acc` spans the output columns so it is chunked for
             * the same reason the input-column phase is, and the gate axis is its reduction axis, so the
             * partial sums live in dhDst across gate chunks. The weight tile it reads is the same
             * [gate chunk, chunkCols] that phase stages. Nothing of W_hh stays resident between chunks
             * or steps; that residency was O(hidden_size^2) and was the ceiling on this path. Outside
             * the gate-math scope above because it issues a DMA, and __VEC_SCOPE__ is vector-only. */
            for (int32_t c0 = 0; c0 < hidden_; c0 += chunkCols_) {
                const int32_t cw = ((hidden_ - c0) < chunkCols_) ? (hidden_ - c0) : chunkCols_;
                const uint32_t cwAlign = static_cast<uint32_t>(AlignUpI64(static_cast<int64_t>(cw) * sizeof(T), 32) /
                                                               sizeof(T));
                const uint32_t cwU = static_cast<uint32_t>(cw);
                const uint32_t c0U = static_cast<uint32_t>(c0);
                ZeroDhDst(dhDst, c0U, cwU);
                for (int32_t slot = 0; slot < static_cast<int32_t>(LSTM_GATE_NUM); ++slot) {
                    for (int32_t g0 = 0; g0 < hidden_; g0 += gBlock_) {
                        const int32_t gCur = ((hidden_ - g0) < gBlock_) ? (hidden_ - g0) : gBlock_;
                        const uint16_t gLoop = static_cast<uint16_t>(gCur);
                        const uint32_t slotBase = static_cast<uint32_t>(slot) * haF + static_cast<uint32_t>(g0);
                        PipeSync<AscendC::HardEvent::V_MTE2>();
                        StageWChunk(inputSize_ + c0, cw, slot * hidden_ + g0, gCur);
                        PipeSync<AscendC::HardEvent::MTE2_V>();
                        __VEC_SCOPE__
                        {
                            Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                            uint32_t maskCntW = cwU;
                            Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
                            /* Neumaier compensated summation, not a plain chain. This is the
                             * longest reduction in the kernel -- 4H terms, 2008 at hidden_size 502
                             * -- and it feeds the recurrence, so every earlier step's dgate inherits
                             * its error and through dgate so do dw and db. Measured at fp32 against
                             * a float64 reference (30 draws, median error reduction at 2008 terms):
                             * four interleaved partial sums 1.64x, eight 2.66x, sixteen 2.98x,
                             * two-level blocking 3.75x, compensated summation 12.78x.
                             *
                             * Two terms per iteration with the accumulator and its successor
                             * swapping roles: the textbook form ends each step with `s = t`, and
                             * unrolling twice lets the even step accumulate s -> t and the odd step
                             * t -> s, so that move disappears. */
                            const uint16_t gPairs = static_cast<uint16_t>(gLoop / 2);
                            const uint16_t gTail = static_cast<uint16_t>(gLoop - gPairs * 2);
                            for (uint16_t b = 0; b < B; ++b) {
                                uint32_t go = gateBase + static_cast<uint32_t>(b) * LSTM_GATE_NUM * haF + slotBase;
                                uint32_t bo = static_cast<uint32_t>(b) * haF + c0U;
                                Micro::RegTensor<float> sum, nxt, comp, wRow, sc, prod, y;
                                Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(sum, dhDst + bo);
                                Micro::Duplicate(comp, 0.0f);
                                for (uint16_t q = 0; q < gPairs; ++q) {
                                    uint16_t k = static_cast<uint16_t>(q * 2);
                                    LoadF32<T>(wChunkU, wRow, mW, static_cast<uint32_t>(k) * cwAlign);
                                    Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sc, dgateU + go + k);
                                    Micro::Mul(prod, wRow, sc, mW);
                                    Micro::Sub(y, prod, comp, mW);
                                    Micro::Add(nxt, sum, y, mW);
                                    Micro::Sub(comp, nxt, sum, mW);
                                    Micro::Sub(comp, comp, y, mW);
                                    LoadF32<T>(wChunkU, wRow, mW, static_cast<uint32_t>(k + 1) * cwAlign);
                                    Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sc, dgateU + go + k + 1);
                                    Micro::Mul(prod, wRow, sc, mW);
                                    Micro::Sub(y, prod, comp, mW);
                                    Micro::Add(sum, nxt, y, mW);
                                    Micro::Sub(comp, sum, nxt, mW);
                                    Micro::Sub(comp, comp, y, mW);
                                }
                                /* The odd tail, and the compensation folded back in. `comp` holds
                                 * what the accumulator dropped; adding it back is the last chance
                                 * to keep it, and costs one operation per row. */
                                for (uint16_t u = 0; u < gTail; ++u) {
                                    uint16_t k = static_cast<uint16_t>(gPairs * 2);
                                    LoadF32<T>(wChunkU, wRow, mW, static_cast<uint32_t>(k) * cwAlign);
                                    Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sc, dgateU + go + k);
                                    Micro::Mul(prod, wRow, sc, mW);
                                    Micro::Sub(y, prod, comp, mW);
                                    Micro::Add(nxt, sum, y, mW);
                                    Micro::Sub(comp, nxt, sum, mW);
                                    Micro::Sub(comp, comp, y, mW);
                                    Micro::Sub(sum, nxt, comp, mW);
                                    Micro::Duplicate(comp, 0.0f);
                                }
                                Micro::Sub(sum, sum, comp, mW);
                                Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dhDst + bo, sum, mW);
                            }
                        }
                    }
                }
            }
            parity = 1 - parity;
        }
        parity_ = parity;
        finalParity_ = parity;
    }

    // dhDst[:, c0:c0+cw] = 0, the seed for the gate reduction that follows
    __aicore__ inline void ZeroDhDst(__local_mem__ float* dhDst, uint32_t c0, uint32_t cw)
    {
        const uint16_t B = static_cast<uint16_t>(bCur_);
        const uint32_t haF = haF_;
        __VEC_SCOPE__
        {
            Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
            uint32_t maskCntW = cw;
            Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
            Micro::RegTensor<float> z;
            Micro::Duplicate(z, 0.0f);
            for (uint16_t b = 0; b < B; ++b) {
                Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dhDst + static_cast<uint32_t>(b) * haF + c0, z, mW);
            }
        }
    }

    // dx[:, cols] and dw[:, cols] for this core's input-column chunks
    __aicore__ inline void ProcessColumns()
    {
        const int32_t per = numIChunks_ / usedCores_;
        const int32_t former = numIChunks_ % usedCores_;
        const int32_t myCount = per + ((blockIdx_ < former) ? 1 : 0);
        const int32_t myBegin = blockIdx_ * per + ((blockIdx_ < former) ? blockIdx_ : former);
        for (int32_t ci = 0; ci < myCount; ++ci) {
            const int32_t col0 = (myBegin + ci) * chunkCols_;
            const int32_t w = ((inputSize_ - col0) < chunkCols_) ? (inputSize_ - col0) : chunkCols_;
            ProcessOneChunk(col0, w);
        }
    }

    /* dx[:, cols] and dw[:, cols] for one input-column chunk, walked in gate chunks. The gate axis is
     * dx's reduction axis and dw's output axis, so cutting it carries nothing across: dx's partial
     * sums live in dxAcc and dw's chunk is complete when its rows are. That is what lets the three
     * 4H-scaled buffers stop being the whole budget at hidden_size 512. x is staged once for every
     * row of the block; the weight chunk is what is re-staged. */
    __aicore__ inline void ProcessOneChunk(int32_t col0, int32_t w)
    {
        const int32_t wAlign = static_cast<int32_t>(AlignUpI64(static_cast<int64_t>(w) * sizeof(T), 32) / sizeof(T));
        // x is fp32, so its chunk has its own 32-byte-rounded pitch
        const int32_t xAlign = static_cast<int32_t>(AlignUpI64(static_cast<int64_t>(w) * FP32_SIZE, 32) / FP32_SIZE);
        __local_mem__ T* wChunkU = UbPtr<T>(layout_.wChunkOff);
        __local_mem__ float* xChunkU = UbPtr<float>(layout_.xChunkOff);
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        __local_mem__ float* dxAccU = UbPtr<float>(layout_.dxAccOff);
        __local_mem__ float* dgateU = UbPtr<float>(layout_.dgateOff);

        StageXChunk(col0, w, xAlign);
        ZeroDxAcc(w);

        const uint32_t haF = haF_;
        const uint32_t wAlignU = static_cast<uint32_t>(wAlign);
        const uint32_t xAlignU = static_cast<uint32_t>(xAlign);
        const uint32_t CW = static_cast<uint32_t>(chunkCols_);
        const uint16_t mLoop = static_cast<uint16_t>(blkRows_);

        for (int32_t slot = 0; slot < static_cast<int32_t>(LSTM_GATE_NUM); ++slot) {
            for (int32_t h0 = 0; h0 < hidden_; h0 += gBlock_) {
                const int32_t gCur = ((hidden_ - h0) < gBlock_) ? (hidden_ - h0) : gBlock_;
                const int32_t gRow0 = slot * hidden_ + h0;
                const uint16_t gLoop = static_cast<uint16_t>(gCur);
                const uint32_t slotBase = static_cast<uint32_t>(slot) * haF + static_cast<uint32_t>(h0);
                PipeSync<AscendC::HardEvent::V_MTE2>();
                StageWChunk(col0, w, gRow0, gCur);
                PipeSync<AscendC::HardEvent::MTE2_V>();

                // dx: dxAcc[m, chunk] += sum_k w[k, chunk] * dgate[m, slot, h0+k]
                __VEC_SCOPE__
                {
                    Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                    uint32_t maskCntW = static_cast<uint32_t>(w);
                    Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
                    /* One chain here, and the measurement says so. dx is a terminal output, so
                     * restructuring this reduction changes dx's numbers and nothing else. Four
                     * interleaved partial sums were compared against this form on the same inputs
                     * (TTK's 500-case backward set, --seed 20260914): they fixed no dx failure and
                     * turned one passing case into a failing one -- T233_B1_I45_H32 float32, dx mare
                     * 2.57 against a limit of 2.0. Measured over 30 draws of a 2008-term reduction,
                     * four partial sums cut the error 1.64x at the median but the quartiles span
                     * 0.60x to 3.98x, so individual elements move both ways. Where a real gain was
                     * needed -- dh_next, whose error the recurrence compounds -- compensated
                     * summation is used instead, 12.78x at the same length. */
                    const uint16_t gPairs = static_cast<uint16_t>(gLoop / 2);
                    const uint16_t gTail = static_cast<uint16_t>(gLoop - gPairs * 2);
                    for (uint16_t mi = 0; mi < mLoop; ++mi) {
                        // Compensated summation, in the form spelled out at dh_next's reduction.
                        Micro::RegTensor<float> sum, nxt, comp, wRow, sc, prod, y;
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(sum,
                                                                           dxAccU + static_cast<uint32_t>(mi) * CW);
                        Micro::Duplicate(comp, 0.0f);
                        uint32_t gOff = static_cast<uint32_t>(mi) * LSTM_GATE_NUM * haF + slotBase;
                        for (uint16_t q = 0; q < gPairs; ++q) {
                            uint16_t k = static_cast<uint16_t>(q * 2);
                            LoadF32<T>(wChunkU, wRow, mW, static_cast<uint32_t>(k) * wAlignU);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sc, dgateU + gOff + k);
                            Micro::Mul(prod, wRow, sc, mW);
                            Micro::Sub(y, prod, comp, mW);
                            Micro::Add(nxt, sum, y, mW);
                            Micro::Sub(comp, nxt, sum, mW);
                            Micro::Sub(comp, comp, y, mW);
                            LoadF32<T>(wChunkU, wRow, mW, static_cast<uint32_t>(k + 1) * wAlignU);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sc, dgateU + gOff + k + 1);
                            Micro::Mul(prod, wRow, sc, mW);
                            Micro::Sub(y, prod, comp, mW);
                            Micro::Add(sum, nxt, y, mW);
                            Micro::Sub(comp, sum, nxt, mW);
                            Micro::Sub(comp, comp, y, mW);
                        }
                        for (uint16_t u = 0; u < gTail; ++u) {
                            uint16_t k = static_cast<uint16_t>(gPairs * 2);
                            LoadF32<T>(wChunkU, wRow, mW, static_cast<uint32_t>(k) * wAlignU);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sc, dgateU + gOff + k);
                            Micro::Mul(prod, wRow, sc, mW);
                            Micro::Sub(y, prod, comp, mW);
                            Micro::Add(nxt, sum, y, mW);
                            Micro::Sub(comp, nxt, sum, mW);
                            Micro::Sub(comp, comp, y, mW);
                            Micro::Sub(sum, nxt, comp, mW);
                            Micro::Duplicate(comp, 0.0f);
                        }
                        Micro::Sub(sum, sum, comp, mW);
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dxAccU + static_cast<uint32_t>(mi) * CW,
                                                                            sum, mW);
                    }
                }

                /* dw: dwAcc[k, chunk] = sum_m dgate[m, slot, h0+k] * x[m, chunk]. Every row of the
                 * block is in UB, so this chunk's rows are finished in one pass and the accumulator
                 * starts at zero rather than being read back. */
                ZeroDwAcc(w);
                __VEC_SCOPE__
                {
                    Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                    uint32_t maskCntW = static_cast<uint32_t>(w);
                    Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
                    for (uint16_t k = 0; k < gLoop; ++k) {
                        // compensated summation, in the form spelled out at dh_next's reduction
                        Micro::RegTensor<float> sum, nxt, comp, xRow, sc, prod, y;
                        Micro::Duplicate(sum, 0.0f);
                        Micro::Duplicate(comp, 0.0f);
                        /* Every for inside a __VEC_SCOPE__ needs its own initialiser, so the
                         * unrolled body and the tail are counted rather than sharing a cursor. */
                        const uint16_t mPairs = static_cast<uint16_t>(mLoop / 2);
                        const uint16_t mTail = static_cast<uint16_t>(mLoop - mPairs * 2);
                        for (uint16_t q = 0; q < mPairs; ++q) {
                            uint16_t mi = static_cast<uint16_t>(q * 2);
                            uint32_t g0 = static_cast<uint32_t>(mi) * LSTM_GATE_NUM * haF + slotBase + k;
                            LoadF32<float>(xChunkU, xRow, mW, static_cast<uint32_t>(mi) * xAlignU);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sc, dgateU + g0);
                            Micro::Mul(prod, xRow, sc, mW);
                            Micro::Sub(y, prod, comp, mW);
                            Micro::Add(nxt, sum, y, mW);
                            Micro::Sub(comp, nxt, sum, mW);
                            Micro::Sub(comp, comp, y, mW);
                            LoadF32<float>(xChunkU, xRow, mW, static_cast<uint32_t>(mi + 1) * xAlignU);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(sc,
                                                                                  dgateU + g0 + LSTM_GATE_NUM * haF);
                            Micro::Mul(prod, xRow, sc, mW);
                            Micro::Sub(y, prod, comp, mW);
                            Micro::Add(sum, nxt, y, mW);
                            Micro::Sub(comp, sum, nxt, mW);
                            Micro::Sub(comp, comp, y, mW);
                        }
                        for (uint16_t u = 0; u < mTail; ++u) {
                            uint16_t mi = static_cast<uint16_t>(mPairs * 2);
                            LoadF32<float>(xChunkU, xRow, mW, static_cast<uint32_t>(mi) * xAlignU);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(
                                sc, dgateU + static_cast<uint32_t>(mi) * LSTM_GATE_NUM * haF + slotBase + k);
                            Micro::Mul(prod, xRow, sc, mW);
                            Micro::Sub(y, prod, comp, mW);
                            Micro::Add(nxt, sum, y, mW);
                            Micro::Sub(comp, nxt, sum, mW);
                            Micro::Sub(comp, comp, y, mW);
                            Micro::Sub(sum, nxt, comp, mW);
                            Micro::Duplicate(comp, 0.0f);
                        }
                        Micro::Sub(sum, sum, comp, mW);
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dwAccU + static_cast<uint32_t>(k) * CW, sum,
                                                                            mW);
                    }
                }
                StoreDwAcc(gRow0, col0, w, gCur);
            }
        }
        StoreDxChunk(col0, w);
    }

    // x[:, col0:col0+w] for every row of the block -> xChunk, fp32, pitch xAlign
    __aicore__ inline void StageXChunk(int32_t col0, int32_t w, int32_t xAlign)
    {
        PipeSync<AscendC::HardEvent::V_MTE2>();
        for (int32_t m0 = 0, run = 0; m0 < blkRows_; m0 += run) {
            run = GmRowRun(m0);
            CopyInRowsF32(layout_.xChunkOff + static_cast<int64_t>(m0) * xAlign * FP32_SIZE, xGm_, run, w,
                          BlockGmRow(m0) * inputSize_ + col0, inputSize_);
        }
        PipeSync<AscendC::HardEvent::MTE2_V>();
    }

    /* Narrow only after the complete gate reduction for this dx chunk. */
    __aicore__ inline void StoreDxChunk(int32_t col0, int32_t w)
    {
        const int64_t burst = AlignUpI64(static_cast<int64_t>(w) * sizeof(T), UB_BLOCK_BYTES);
        int64_t sourceOffset = layout_.dxAccOff;
        int64_t sourcePitch = static_cast<int64_t>(chunkCols_) * FP32_SIZE;
        if constexpr (!std::is_same<T, float>::value) {
            PipeSync<AscendC::HardEvent::MTE3_V>();
            __local_mem__ float* src = UbPtr<float>(layout_.dxAccOff);
            __local_mem__ T* dst = UbPtr<T>(layout_.outStageOff);
            const uint32_t srcPitch = static_cast<uint32_t>(chunkCols_);
            const uint32_t dstPitch = static_cast<uint32_t>(burst / sizeof(T));
            const uint16_t rows = static_cast<uint16_t>(blkRows_);
            const uint32_t width = static_cast<uint32_t>(w);
            const uint16_t steps = static_cast<uint16_t>((width + VL_F32 - 1) / VL_F32);
            __VEC_SCOPE__
            {
                for (uint16_t row = 0; row < rows; ++row) {
                    uint32_t remaining = width;
                    for (uint16_t k = 0; k < steps; ++k) {
                        Micro::MaskReg mask = Micro::UpdateMask<float>(remaining);
                        Micro::RegTensor<float> value;
                        LoadF32<float>(src, value, mask, row * srcPitch + k * VL_F32);
                        StoreF32<T>(dst, value, mask, row * dstPitch + k * VL_F32);
                    }
                }
            }
            sourceOffset = layout_.outStageOff;
            sourcePitch = burst;
        }
        PipeSync<AscendC::HardEvent::V_MTE3>();
        for (int32_t m0 = 0, run = 0; m0 < blkRows_; m0 += run) {
            run = GmRowRun(m0);
            AscendC::DataCopyExtParams p;
            p.blockCount = static_cast<uint16_t>(run);
            p.blockLen = static_cast<uint32_t>(w * sizeof(T));
            p.srcStride = static_cast<uint32_t>((sourcePitch - burst) / UB_BLOCK_BYTES);
            p.dstStride = static_cast<uint32_t>((inputSize_ - w) * sizeof(T));
            AscendC::DataCopyPad(dxGm_[BlockGmRow(m0) * inputSize_ + col0],
                                 UbTensor<T>(sourceOffset + static_cast<int64_t>(m0) * sourcePitch), p);
        }
    }

    // dxAcc = 0 over w columns of every block row
    __aicore__ inline void ZeroDxAcc(int32_t w)
    {
        __local_mem__ float* dxAccU = UbPtr<float>(layout_.dxAccOff);
        const uint32_t CW = static_cast<uint32_t>(chunkCols_);
        PipeSync<AscendC::HardEvent::MTE3_V>();
        __VEC_SCOPE__
        {
            Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
            uint32_t maskCntW = static_cast<uint32_t>(w);
            Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
            Micro::RegTensor<float> z;
            Micro::Duplicate(z, 0.0f);
            for (uint16_t mi = 0; mi < static_cast<uint16_t>(blkRows_); ++mi) {
                Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dxAccU + static_cast<uint32_t>(mi) * CW, z, mW);
            }
        }
    }

    __aicore__ inline void ZeroDwAcc(int32_t w)
    {
        /* THE PREVIOUS CHUNK'S ADD IS STILL READING THIS BUFFER. StoreDwAcc sends the accumulator
         * itself out to GM, not a narrowed copy of it, so zeroing for the next chunk is a
         * write-after-read against that copy and needs the event. Measured without it, dw came back
         * essentially zero -- mare 2048 at half, which is exactly 1 / small_value. */
        PipeSync<AscendC::HardEvent::MTE3_V>();
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        const uint16_t GLoop = static_cast<uint16_t>(gBlock_);
        const uint32_t CW = static_cast<uint32_t>(chunkCols_);
        __VEC_SCOPE__
        {
            uint32_t maskCntW = static_cast<uint32_t>(w);
            Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
            Micro::RegTensor<float> z;
            Micro::Duplicate(z, 0.0f);
            for (uint16_t g = 0; g < GLoop; ++g) {
                Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dwAccU + static_cast<uint32_t>(g) * CW, z, mW);
            }
        }
    }

    /* dwAcc -> dw[gate rows, col0:col0+w], as an fp32 add into the accumulator. Atomic add because
     * the sum is a reduction split across time and batch blocks; column chunks belong to one core
     * each and db is written by the last core alone, so the accumulation is across blocks, never
     * across cores. The first block overwrites, because AllocTensor and the public ACLNN workspace
     * do not guarantee zero initialization. The source pitch is chunkCols_ floats. */
    __aicore__ inline void StoreDwAcc(int32_t gRow0, int32_t col0, int32_t w, int32_t gCur)
    {
        PipeSync<AscendC::HardEvent::V_MTE3>();
        const int64_t burst = AlignUpI64(static_cast<int64_t>(w) * FP32_SIZE, 32);
        AscendC::DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(gCur);
        p.blockLen = static_cast<uint32_t>(w * FP32_SIZE);
        p.srcStride = static_cast<uint32_t>((static_cast<int64_t>(chunkCols_) * FP32_SIZE - burst) / 32);
        p.dstStride = static_cast<uint32_t>((cols_ - w) * FP32_SIZE);
        if (b0_ != 0 || blkLt0_ != ((timeStep_ - 1) / tBlock_) * tBlock_) {
            AscendC::SetAtomicAdd<float>();
        }
        AscendC::DataCopyPad(dwAccGm_[static_cast<int64_t>(gRow0) * cols_ + col0], UbTensor<float>(layout_.dwAccOff),
                             p);
        AscendC::SetAtomicNone();
    }

    // last core: dw hidden columns (from h_prev) and db, for this time block
    __aicore__ inline void ProcessTail()
    {
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        __local_mem__ float* dgateU = UbPtr<float>(layout_.dgateOff);
        __local_mem__ float* hU = UbPtr<float>(layout_.hOff);
        __local_mem__ float* initHU = UbPtr<float>(layout_.initHOff);
        const uint16_t B = static_cast<uint16_t>(bCur_);
        const uint32_t H = static_cast<uint32_t>(hidden_);
        const uint32_t haF = haF_;
        const uint32_t CW = static_cast<uint32_t>(chunkCols_);

        /* dw[:, I:I+H] += h_prev(t)^T @ dgate(t), in column and gate chunks like the input columns,
         * not one hidden_-wide pass. dwAcc's rows are chunkCols_ wide and there are only gBlock_ of
         * them, so a single pass at hidden_ lanes is correct only while chunkCols_ >= hidden_, and
         * the budget breaks that exactly where hidden_ is largest: measured at I=8 H=57 fp32 the
         * search drops chunkCols_ to 32 and dw[:, I:I+H] came out right in its first 32 columns and
         * wrong in the rest, the accumulation having run off the end of each row. At larger 4H the
         * same overrun leaves the region and the launch ends in 507035 vector core exception
         * (T=32 B=1 I=16 H=64). */
        for (int32_t hc0 = 0; hc0 < hidden_; hc0 += chunkCols_) {
            const int32_t hw = ((hidden_ - hc0) < chunkCols_) ? (hidden_ - hc0) : chunkCols_;
            const uint32_t hwU = static_cast<uint32_t>(hw);
            const uint32_t hcOff = static_cast<uint32_t>(hc0);
            for (int32_t slot = 0; slot < static_cast<int32_t>(LSTM_GATE_NUM); ++slot) {
                for (int32_t g0 = 0; g0 < hidden_; g0 += gBlock_) {
                    const int32_t gCur = ((hidden_ - g0) < gBlock_) ? (hidden_ - g0) : gBlock_;
                    const uint16_t gLoop = static_cast<uint16_t>(gCur);
                    const uint32_t slotBase = static_cast<uint32_t>(slot) * haF + static_cast<uint32_t>(g0);
                    ZeroDwAcc(hw);
                    /* The block's actT range, not the whole sequence, and the whole range in one
                     * vector pass. dgate and h hold only this block's rows, so walking all of T
                     * would read past them. Exactly one step has its h_prev outside -- init_h at the
                     * first block, the history row StageBlock put in the same place at every later
                     * one -- and every other step's h_prev sits in hU at a regular stride. Splitting
                     * that way keeps the time sum in registers for the compensated accumulator. A
                     * plain running total was measured at T=32 B=1 I=16 H=64 float32 to put
                     * dw[:, I:I+H]'s worst element at mare 2.3 against a limit of 2.0 while its mean
                     * ratio was 0.008 -- one cancellation-dominated element, not the aggregate. */
                    const int32_t ltIn0 = backward_ ? 0 : 1;
                    const int32_t ltInN = backward_ ? (blkSteps_ - 1) : blkSteps_;
                    const int32_t ltEdge = backward_ ? (blkSteps_ - 1) : 0;
                    const int32_t hShift = backward_ ? 1 : -1;
                    const uint32_t rowStrideS = static_cast<uint32_t>(bCur_) * haF;
                    const uint32_t gateStride = static_cast<uint32_t>(bCur_) * LSTM_GATE_NUM * haF;
                    __VEC_SCOPE__
                    {
                        Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                        uint32_t maskCntW = hwU;
                        Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
                        Micro::RegTensor<float> zero;
                        Micro::Duplicate(zero, 0.0f);
                        const uint16_t ltCount = static_cast<uint16_t>((ltInN > ltIn0) ? (ltInN - ltIn0) : 0);
                        for (uint16_t k = 0; k < gLoop; ++k) {
                            /* Compensated summation, no unrolling: the compensation is what the
                             * four interleaved partial sums used to approximate, an order of
                             * magnitude better at no measurable device cost. The accumulator is
                             * copied with an add of zero rather than the two-step role swap dh_next
                             * and dw use -- those reduce over one axis and can pair their terms,
                             * while this walks (timestep, batch row) pairs whose count is
                             * B * ltCount plus one edge step, so a pairing would straddle the inner
                             * loop. Adding +0.0 is exact for every finite value. */
                            Micro::RegTensor<float> sum, nxt, comp, hRow, sc, prod, y;
                            Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(sum,
                                                                               dwAccU + static_cast<uint32_t>(k) * CW);
                            Micro::Duplicate(comp, 0.0f);
                            for (uint16_t t = 0; t < ltCount; ++t) {
                                int32_t lt = ltIn0 + static_cast<int32_t>(t);
                                for (uint16_t b = 0; b < B; ++b) {
                                    LoadF32<float>(hU, hRow, mW,
                                                   static_cast<uint32_t>(lt + hShift) * rowStrideS +
                                                       static_cast<uint32_t>(b) * haF + hcOff);
                                    Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(
                                        sc, dgateU + static_cast<uint32_t>(lt) * gateStride +
                                                static_cast<uint32_t>(b) * LSTM_GATE_NUM * haF + slotBase + k);
                                    Micro::Mul(prod, hRow, sc, mW);
                                    Micro::Sub(y, prod, comp, mW);
                                    Micro::Add(nxt, sum, y, mW);
                                    Micro::Sub(comp, nxt, sum, mW);
                                    Micro::Sub(comp, comp, y, mW);
                                    Micro::Add(sum, nxt, zero, mW);
                                }
                            }
                            // the one step whose h_prev lies outside the block
                            for (uint16_t b = 0; b < B; ++b) {
                                LoadF32<float>(initHU, hRow, mW, static_cast<uint32_t>(b) * haF + hcOff);
                                Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(
                                    sc, dgateU + static_cast<uint32_t>(ltEdge) * gateStride +
                                            static_cast<uint32_t>(b) * LSTM_GATE_NUM * haF + slotBase + k);
                                Micro::Mul(prod, hRow, sc, mW);
                                Micro::Sub(y, prod, comp, mW);
                                Micro::Add(nxt, sum, y, mW);
                                Micro::Sub(comp, nxt, sum, mW);
                                Micro::Sub(comp, comp, y, mW);
                                Micro::Add(sum, nxt, zero, mW);
                            }
                            Micro::Sub(sum, sum, comp, mW);
                            Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dwAccU + static_cast<uint32_t>(k) * CW,
                                                                                sum, mW);
                        }
                    }
                    StoreDwAcc(slot * hidden_ + g0, inputSize_ + hc0, hw, gCur);
                }
            }
        }

        // db[slot*H : slot*H+H] += sum_m dgate[m][slot][:]
        if (isBias_) {
            __local_mem__ float* dbStage = UbPtr<float>(layout_.dbStageOff);
            __local_mem__ float* dbComp = UbPtr<float>(layout_.dbCompOff);
            const bool firstBlock = b0_ == 0 && blkLt0_ == ((timeStep_ - 1) / tBlock_) * tBlock_;
            const bool lastBlock = b0_ + bCur_ == batch_ && blkLt0_ == 0;
            const uint16_t mLoop = static_cast<uint16_t>(blkRows_);
            const uint16_t hSteps = static_cast<uint16_t>((hidden_ + VL_F32 - 1) / VL_F32);
            /* The previous block's db copy reads this same staging buffer, and these stores would
             * otherwise overtake it -- the write-after-read counterpart of the MTE3_V that
             * StoreDwAcc already does for dwAcc. */
            PipeSync<AscendC::HardEvent::MTE3_V>();
            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                for (uint16_t slot = 0; slot < static_cast<uint16_t>(LSTM_GATE_NUM); ++slot) {
                    uint32_t maskCntH = H;
                    for (uint16_t k = 0; k < hSteps; ++k) {
                        Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                        uint32_t ho = static_cast<uint32_t>(k) * VL_F32;
                        uint32_t slotOff = static_cast<uint32_t>(slot) * haF + ho;
                        // Carry the residual across time and batch blocks. Rounding each partial
                        // sum before an atomic GM add discards the block-local compensation.
                        Micro::RegTensor<float> sum, nxt, comp, r, y, zero;
                        Micro::Duplicate(zero, 0.0f);
                        if (firstBlock) {
                            Micro::Duplicate(sum, 0.0f);
                            Micro::Duplicate(comp, 0.0f);
                        } else {
                            Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(sum, dbStage + slotOff);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(comp, dbComp + slotOff);
                        }
                        const uint16_t mPairs = static_cast<uint16_t>(mLoop / 2);
                        const uint16_t mTail = static_cast<uint16_t>(mLoop - mPairs * 2);
                        for (uint16_t q = 0; q < mPairs; ++q) {
                            uint16_t m = static_cast<uint16_t>(q * 2);
                            uint32_t o0 = static_cast<uint32_t>(m) * LSTM_GATE_NUM * haF + slotOff;
                            Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(r, dgateU + o0);
                            Micro::Sub(y, r, comp, mH);
                            Micro::Add(nxt, sum, y, mH);
                            Micro::Sub(comp, nxt, sum, mH);
                            Micro::Sub(comp, comp, y, mH);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(r, dgateU + o0 + LSTM_GATE_NUM * haF);
                            Micro::Sub(y, r, comp, mH);
                            Micro::Add(sum, nxt, y, mH);
                            Micro::Sub(comp, sum, nxt, mH);
                            Micro::Sub(comp, comp, y, mH);
                        }
                        for (uint16_t u = 0; u < mTail; ++u) {
                            uint16_t m = static_cast<uint16_t>(mPairs * 2);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(
                                r, dgateU + static_cast<uint32_t>(m) * LSTM_GATE_NUM * haF + slotOff);
                            Micro::Sub(y, r, comp, mH);
                            Micro::Add(nxt, sum, y, mH);
                            Micro::Sub(comp, nxt, sum, mH);
                            Micro::Sub(comp, comp, y, mH);
                            Micro::Add(sum, nxt, zero, mH);
                        }
                        if (lastBlock) {
                            Micro::Sub(sum, sum, comp, mH);
                        }
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dbStage + slotOff, sum, mH);
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dbComp + slotOff, comp, mH);
                    }
                }
            }
            if (lastBlock) {
                PipeSync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyExtParams p{static_cast<uint16_t>(LSTM_GATE_NUM),
                                             static_cast<uint32_t>(hidden_ * FP32_SIZE), 0, 0, 0};
                AscendC::DataCopyPad(dbAccGm_[0], UbTensor<float>(layout_.dbStageOff), p);
            }
        }
    }

    /* WRITTEN ONCE PER BATCH BLOCK, AFTER ITS LAST TIME BLOCK. dh_prev / dc_prev are the
     * recurrence's state at the far end of the sequence, so they are ready only when every time
     * block has been walked -- writing them per time block would leave the value of whichever block
     * happened to run last. Everything else ProcessTail produces is a reduction and accumulates. */
    __aicore__ inline void WritePrevState()
    {
        const uint16_t B = static_cast<uint16_t>(bCur_);
        const uint32_t H = static_cast<uint32_t>(hidden_);
        const uint32_t haF = haF_;
        const uint32_t haT = haT_;
        const uint16_t hSteps = static_cast<uint16_t>((hidden_ + VL_F32 - 1) / VL_F32);
        __local_mem__ float* dhFinal = UbPtr<float>(layout_.dhCurOff) +
                                       static_cast<uint32_t>(finalParity_) * static_cast<uint32_t>(bCur_) * haF;
        __local_mem__ float* dcFinal = UbPtr<float>(layout_.dcCurOff) +
                                       static_cast<uint32_t>(finalParity_) * static_cast<uint32_t>(bCur_) * haF;
        const int64_t dhStageOff = layout_.smallStageOff;
        const int64_t dcStageOff = dhStageOff + static_cast<int64_t>(bCur_) * haT_ * sizeof(T);
        __local_mem__ T* dhStage = UbPtr<T>(dhStageOff);
        __local_mem__ T* dcStage = UbPtr<T>(dcStageOff);
        PipeSync<AscendC::HardEvent::MTE3_V>(); // the last block's stores still read this region
        __VEC_SCOPE__
        {
            Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
            Micro::RegTensor<float> r;
            for (uint16_t b = 0; b < B; ++b) {
                uint32_t maskCntH = H;
                for (uint16_t k = 0; k < hSteps; ++k) {
                    Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                    uint32_t ho = static_cast<uint32_t>(k) * VL_F32;
                    uint32_t srcOff = static_cast<uint32_t>(b) * haF + ho;
                    uint32_t dstOff = static_cast<uint32_t>(b) * haT + ho;
                    Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(r, dhFinal + srcOff);
                    StoreF32<T>(dhStage, r, mH, dstOff);
                    Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(r, dcFinal + srcOff);
                    StoreF32<T>(dcStage, r, mH, dstOff);
                }
            }
        }
        PipeSync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams p{static_cast<uint16_t>(bCur_), static_cast<uint32_t>(hidden_ * sizeof(T)), 0, 0, 0};
        const int64_t rowOff = static_cast<int64_t>(b0_) * hidden_;
        AscendC::DataCopyPad(dhPrevGm_[rowOff], UbTensor<T>(dhStageOff), p);
        AscendC::DataCopyPad(dcPrevGm_[rowOff], UbTensor<T>(dcStageOff), p);
    }

    // the narrowing counterpart of ProcessColumns: this core's own input-column chunks
    __aicore__ inline void FinishColumns()
    {
        const int32_t per = numIChunks_ / usedCores_;
        const int32_t former = numIChunks_ % usedCores_;
        const int32_t myCount = per + ((blockIdx_ < former) ? 1 : 0);
        const int32_t myBegin = blockIdx_ * per + ((blockIdx_ < former) ? blockIdx_ : former);
        for (int32_t ci = 0; ci < myCount; ++ci) {
            const int32_t col0 = (myBegin + ci) * chunkCols_;
            const int32_t w = ((inputSize_ - col0) < chunkCols_) ? (inputSize_ - col0) : chunkCols_;
            FinishDwChunk(col0, w);
        }
    }

    // the narrowing counterpart of ProcessTail's dw[:, I:I+H] loop
    __aicore__ inline void FinishTailColumns()
    {
        for (int32_t hc0 = 0; hc0 < hidden_; hc0 += chunkCols_) {
            const int32_t hw = ((hidden_ - hc0) < chunkCols_) ? (hidden_ - hc0) : chunkCols_;
            FinishDwChunk(inputSize_ + hc0, hw);
        }
    }

    /* The one narrowing, after every block: read this core's own column chunk back out of the fp32
     * accumulator and write it into dw at the operator's width. Not atomic and not raced -- a
     * column chunk belongs to exactly one core, and it is written once. Compiled away at float,
     * where dwAccGm_ IS dw. */
    __aicore__ inline void FinishDwChunk(int32_t col0, int32_t w)
    {
        if constexpr (std::is_same<T, float>::value) {
            (void)col0;
            (void)w;
            return;
        } else {
            const int32_t wAlign = static_cast<int32_t>(AlignUpI64(static_cast<int64_t>(w) * sizeof(T), 32) /
                                                        sizeof(T));
            const int64_t burst = AlignUpI64(static_cast<int64_t>(w) * FP32_SIZE, 32);
            for (int32_t slot = 0; slot < static_cast<int32_t>(LSTM_GATE_NUM); ++slot) {
                for (int32_t g0 = 0; g0 < hidden_; g0 += gBlock_) {
                    const int32_t gCur = ((hidden_ - g0) < gBlock_) ? (hidden_ - g0) : gBlock_;
                    const int64_t gRow0 = static_cast<int64_t>(slot) * hidden_ + g0;
                    PipeSync<AscendC::HardEvent::V_MTE2>();
                    PipeSync<AscendC::HardEvent::MTE3_MTE2>(); // the last block's add still reads dwAcc
                    {
                        AscendC::DataCopyExtParams p;
                        p.blockCount = static_cast<uint16_t>(gCur);
                        p.blockLen = static_cast<uint32_t>(w * FP32_SIZE);
                        p.srcStride = static_cast<uint32_t>((cols_ - w) * FP32_SIZE);
                        p.dstStride = static_cast<uint32_t>((static_cast<int64_t>(chunkCols_) * FP32_SIZE - burst) /
                                                            32);
                        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                        AscendC::DataCopyPad(UbTensor<float>(layout_.dwAccOff), dwAccGm_[gRow0 * cols_ + col0], p, pad);
                    }
                    PipeSync<AscendC::HardEvent::MTE2_V>();
                    NarrowDwAcc(w, wAlign, gCur);
                    PipeSync<AscendC::HardEvent::V_MTE3>();
                    AscendC::DataCopyExtParams p;
                    p.blockCount = static_cast<uint16_t>(gCur);
                    p.blockLen = static_cast<uint32_t>(w * sizeof(T));
                    p.srcStride = 0; // outStage's pitch IS the burst's 32B rounding
                    p.dstStride = static_cast<uint32_t>((cols_ - w) * sizeof(T));
                    AscendC::DataCopyPad(dwGm_[gRow0 * cols_ + col0], UbTensor<T>(layout_.outStageOff), p);
                }
            }
        }
    }

    // dwAcc (fp32, pitch chunkCols_) -> outStage (dtype T, pitch wAlign), w columns of gCur rows
    __aicore__ inline void NarrowDwAcc(int32_t w, int32_t wAlign, int32_t gCur)
    {
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        __local_mem__ T* outU = UbPtr<T>(layout_.outStageOff);
        const uint16_t GLoop = static_cast<uint16_t>(gCur);
        const uint32_t CW = static_cast<uint32_t>(chunkCols_);
        const uint32_t wAlignU = static_cast<uint32_t>(wAlign);
        __VEC_SCOPE__
        {
            Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
            uint32_t maskCntW = static_cast<uint32_t>(w);
            Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
            for (uint16_t g = 0; g < GLoop; ++g) {
                Micro::RegTensor<float> r;
                Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(r, dwAccU + static_cast<uint32_t>(g) * CW);
                StoreF32<T>(outU, r, mW, static_cast<uint32_t>(g) * wAlignU);
            }
        }
    }

    /* db's fp32 accumulator -> db, at the operator's width. Walked one vector register at a time
     * like every other loop over the hidden axis; dbStage is reused as the staging buffer on the
     * way back in, which is free because nothing else reads it after the last block. */
    __aicore__ inline void FinishDb()
    {
        if constexpr (std::is_same<T, float>::value) {
            return;
        } else {
            if (!isBias_) {
                return;
            }
            __local_mem__ float* dbStage = UbPtr<float>(layout_.dbStageOff);
            __local_mem__ T* outU = UbPtr<T>(layout_.outStageOff);
            const uint32_t H = static_cast<uint32_t>(hidden_);
            const uint32_t haT = haT_;
            const uint32_t haF = haF_;
            const uint16_t hSteps = static_cast<uint16_t>((hidden_ + VL_F32 - 1) / VL_F32);
            PipeSync<AscendC::HardEvent::V_MTE2>();
            PipeSync<AscendC::HardEvent::MTE3_MTE2>();
            {
                AscendC::DataCopyExtParams p{static_cast<uint16_t>(LSTM_GATE_NUM),
                                             static_cast<uint32_t>(hidden_ * FP32_SIZE), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                AscendC::DataCopyPad(UbTensor<float>(layout_.dbStageOff), dbAccGm_[0], p, pad);
            }
            PipeSync<AscendC::HardEvent::MTE2_V>();
            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                for (uint16_t slot = 0; slot < static_cast<uint16_t>(LSTM_GATE_NUM); ++slot) {
                    uint32_t maskCntH = H;
                    for (uint16_t k = 0; k < hSteps; ++k) {
                        Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                        uint32_t ho = static_cast<uint32_t>(k) * VL_F32;
                        Micro::RegTensor<float> r;
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(
                            r, dbStage + static_cast<uint32_t>(slot) * haF + ho);
                        StoreF32<T>(outU, r, mH, static_cast<uint32_t>(slot) * haT + ho);
                    }
                }
            }
            PipeSync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyExtParams p{static_cast<uint16_t>(LSTM_GATE_NUM),
                                         static_cast<uint32_t>(hidden_ * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(dbGm_[0], UbTensor<T>(layout_.outStageOff), p);
        }
    }

private:
    static constexpr int64_t REPLAY_PLANES = 7;
    int32_t biasComponents_{0};
    int64_t replayPlaneSize_{0};
    int32_t timeStep_{0};
    int32_t batch_{0};
    int32_t inputSize_{0};
    int32_t hidden_{0};
    int32_t gates_{0};
    int32_t mAll_{0};
    int32_t cols_{0};
    /* Time blocking. `tBlock_` is how many steps are staged at once; the four `blk*` fields are the
     * block being walked right now, and `parity_` is the ping-pong index carried across it. */
    int32_t tBlock_{0};
    int32_t blkLt0_{0};
    int32_t blkSteps_{0};
    int32_t blkRows_{0};
    int32_t blkA0_{0};
    int32_t parity_{0};
    /* Batch blocking: `bBlock_` is the block size, `b0_` / `bCur_` the block being walked. */
    int32_t bBlock_{0};
    int32_t b0_{0};
    int32_t bCur_{0};
    /* Gate blocking: rows of the 4H axis staged at once, always within one slot. */
    int32_t gBlock_{0};
    bool isBias_{false};
    bool backward_{false};
    int32_t gateOrder_{0};
    int32_t slotJ_{OFFSET_J};
    int32_t slotF_{OFFSET_F};
    int32_t usedCores_{1};
    int32_t chunkCols_{64};
    int32_t mBlock_{64};
    int32_t numIChunks_{1};
    int32_t blockIdx_{0};
    int32_t finalParity_{0};
    uint32_t haT_{0};
    uint32_t haF_{0};
    LstmGradRegbaseSmallUbLayout layout_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<uint8_t> baseTensor_;
    __local_mem__ uint8_t* ubBase_{nullptr};

    AscendC::GlobalTensor<T> wGm_, dhGm_, dcGm_;
    AscendC::GlobalTensor<T> biasGm_;
    AscendC::GlobalTensor<float> replayGm_;
    AscendC::GlobalTensor<T> xGm_, dyGm_;
    // Public states retain T; narrow recurrence replay does not read rounded histories.
    AscendC::GlobalTensor<T> initHGm_, initCGm_, hGm_, cGm_;
    AscendC::GlobalTensor<T> iGm_, jGm_, fGm_, oGm_, tanhGm_;
    AscendC::GlobalTensor<T> dwGm_, dbGm_, dhPrevGm_, dcPrevGm_;
    AscendC::GlobalTensor<T> dxGm_;
    AscendC::GlobalTensor<float> dwAccGm_, dbAccGm_; // see the note where they are bound
};

} // namespace LstmGradRegbase

#endif // SINGLE_LAYER_LSTM_GRAD_REGBASE_SMALL_H
