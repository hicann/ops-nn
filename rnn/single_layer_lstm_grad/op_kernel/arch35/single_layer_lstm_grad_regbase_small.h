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
 * \brief arch35 small-shape LSTM grad kernel: AIV-only, zero cross-core sync, zero workspace.
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
 * Every UB row uses a 32B-aligned pitch (hAlignT / hAlignF elements) so all vector
 * loads/stores are aligned vlds/vsts; masks cover the H tail. dgate keeps one
 * hAlignF-pitched row per (m, gate-slot); slot order matches the w row layout.
 */

#ifndef SINGLE_LAYER_LSTM_GRAD_REGBASE_SMALL_H
#define SINGLE_LAYER_LSTM_GRAD_REGBASE_SMALL_H

#include "kernel_operator.h"
#include "single_layer_lstm_grad_regbase_tiling_data.h"

namespace LstmGradRegbase {

namespace Micro = AscendC::MicroAPI;

constexpr uint32_t VL_F32 = 64; // 256B vector register / 4B
constexpr uint32_t LSTM_GATE_NUM = 4;

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

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR initH, GM_ADDR initC, GM_ADDR h, GM_ADDR c, GM_ADDR dy,
                                GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j, GM_ADDR f, GM_ADDR o, GM_ADDR tanhct,
                                GM_ADDR dw, GM_ADDR db, GM_ADDR dx, GM_ADDR dhPrev, GM_ADDR dcPrev,
                                const LstmGradRegbaseSmallTilingData* tiling, AscendC::TPipe* pipe)
    {
        timeStep_ = static_cast<int32_t>(tiling->timeStep);
        batch_ = static_cast<int32_t>(tiling->batch);
        inputSize_ = static_cast<int32_t>(tiling->inputSize);
        hidden_ = static_cast<int32_t>(tiling->hiddenSize);
        isBias_ = tiling->isBias != 0;
        backward_ = tiling->direction != 0;
        gateOrder_ = static_cast<int32_t>(tiling->gateOrder);
        usedCores_ = static_cast<int32_t>(tiling->usedCores);
        chunkCols_ = static_cast<int32_t>(tiling->chunkCols);
        mBlock_ = static_cast<int32_t>(tiling->mBlock);
        numIChunks_ = static_cast<int32_t>(tiling->numIChunks);
        gates_ = LSTM_GATE_NUM * hidden_;
        mAll_ = timeStep_ * batch_;
        cols_ = inputSize_ + hidden_;
        blockIdx_ = static_cast<int32_t>(AscendC::GetBlockIdx());
        // physical slot order of j/f follows the w row layout selected by gate_order
        slotJ_ = (gateOrder_ == 0) ? 1 : 2;
        slotF_ = (gateOrder_ == 0) ? 2 : 1;

        layout_.Fill(timeStep_, batch_, hidden_, chunkCols_, mBlock_, sizeof(T));
        haT_ = static_cast<uint32_t>(layout_.hAlignT);
        haF_ = static_cast<uint32_t>(layout_.hAlignF);
        pipe->InitBuffer(ubBuf_, static_cast<uint32_t>(layout_.totalBytes));
        AscendC::LocalTensor<uint8_t> base = ubBuf_.Get<uint8_t>();
        ubBase_ = (__local_mem__ uint8_t*)base.GetPhyAddr();
        baseTensor_ = base;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(w));
        initHGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH));
        initCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC));
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(c));
        dyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dy));
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dh));
        dcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dc));
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
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCores_ || mAll_ <= 0 || hidden_ <= 0) {
            return;
        }
        StageResident();
        Prologue();
        ProcessChain();
        ProcessColumns();
        if (blockIdx_ == usedCores_ - 1) {
            ProcessTail();
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

    // GM rows (contiguous, rowElems each) -> UB rows auto-rounded to 32B pitch (= hAlignT)
    __aicore__ inline void CopyInRows(int64_t ubOff, const AscendC::GlobalTensor<T>& gm, int64_t rows, int64_t rowElems)
    {
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(rowElems * sizeof(T)), 0,
                                          0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(UbTensor<T>(ubOff), gm[0], params, pad);
    }

    __aicore__ inline void StageResident()
    {
        CopyInRows(layout_.dyOff, dyGm_, mAll_, hidden_);
        CopyInRows(layout_.igOff, iGm_, mAll_, hidden_);
        CopyInRows(layout_.jgOff, jGm_, mAll_, hidden_);
        CopyInRows(layout_.fgOff, fGm_, mAll_, hidden_);
        CopyInRows(layout_.ogOff, oGm_, mAll_, hidden_);
        CopyInRows(layout_.tanhOff, tanhGm_, mAll_, hidden_);
        CopyInRows(layout_.cOff, cGm_, mAll_, hidden_);
        CopyInRows(layout_.hOff, hGm_, mAll_, hidden_);
        CopyInRows(layout_.initHOff, initHGm_, batch_, hidden_);
        CopyInRows(layout_.initCOff, initCGm_, batch_, hidden_);
        CopyInRows(layout_.dh0Off, dhGm_, batch_, hidden_);
        CopyInRows(layout_.dc0Off, dcGm_, batch_, hidden_);
        // w[:, I:I+H]: 4H rows of H cols out of a (I+H)-pitched GM matrix
        AscendC::DataCopyExtParams wParams;
        wParams.blockCount = static_cast<uint16_t>(gates_);
        wParams.blockLen = static_cast<uint32_t>(hidden_ * sizeof(T));
        wParams.srcStride = static_cast<uint32_t>(inputSize_ * sizeof(T)); // GM gap in bytes
        wParams.dstStride = 0;                                             // auto 32B rounding -> hAlignT pitch
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(UbTensor<T>(layout_.whOff), wGm_[inputSize_], wParams, pad);
        PipeSync<AscendC::HardEvent::MTE2_V>();
    }

    // cast dh0/dc0 into fp32 ping buffers (parity 0), row-wise
    __aicore__ inline void Prologue()
    {
        __local_mem__ T* dh0 = UbPtr<T>(layout_.dh0Off);
        __local_mem__ T* dc0 = UbPtr<T>(layout_.dc0Off);
        __local_mem__ float* dhCur = UbPtr<float>(layout_.dhCurOff);
        __local_mem__ float* dcCur = UbPtr<float>(layout_.dcCurOff);
        const uint16_t B = static_cast<uint16_t>(batch_);
        const uint32_t H = static_cast<uint32_t>(hidden_);
        const uint32_t haT = haT_;
        const uint32_t haF = haF_;
        __VEC_SCOPE__
        {
            Micro::RegTensor<float> r;
            uint32_t maskCntH = H;
            Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
            for (uint16_t b = 0; b < B; ++b) {
                uint32_t srcOff = static_cast<uint32_t>(b) * haT;
                uint32_t dstOff = static_cast<uint32_t>(b) * haF;
                LoadF32<T>(dh0, r, mH, srcOff);
                Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dhCur + dstOff, r, mH);
                LoadF32<T>(dc0, r, mH, srcOff);
                Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dcCur + dstOff, r, mH);
            }
        }
    }

    __aicore__ inline void ProcessChain()
    {
        __local_mem__ T* dyU = UbPtr<T>(layout_.dyOff);
        __local_mem__ T* iU = UbPtr<T>(layout_.igOff);
        __local_mem__ T* jU = UbPtr<T>(layout_.jgOff);
        __local_mem__ T* fU = UbPtr<T>(layout_.fgOff);
        __local_mem__ T* oU = UbPtr<T>(layout_.ogOff);
        __local_mem__ T* tanhU = UbPtr<T>(layout_.tanhOff);
        __local_mem__ T* cU = UbPtr<T>(layout_.cOff);
        __local_mem__ T* initCU = UbPtr<T>(layout_.initCOff);
        __local_mem__ T* whU = UbPtr<T>(layout_.whOff);
        __local_mem__ float* dgateU = UbPtr<float>(layout_.dgateOff);
        __local_mem__ float* dhBase = UbPtr<float>(layout_.dhCurOff);
        __local_mem__ float* dcBase = UbPtr<float>(layout_.dcCurOff);

        const uint32_t H = static_cast<uint32_t>(hidden_);
        const uint32_t haT = haT_;
        const uint32_t haF = haF_;
        const uint32_t bhF = static_cast<uint32_t>(batch_) * haF;
        const uint16_t B = static_cast<uint16_t>(batch_);
        const uint16_t HLoop = static_cast<uint16_t>(hidden_);
        const uint32_t sJ = static_cast<uint32_t>(slotJ_) * haF;
        const uint32_t sF = static_cast<uint32_t>(slotF_) * haF;
        const uint32_t sO = 3 * haF;

        int32_t parity = 0;
        for (int32_t lt = timeStep_ - 1; lt >= 0; --lt) {
            const int32_t actT = backward_ ? (timeStep_ - 1 - lt) : lt;
            const bool useInit = (lt == 0);
            const int32_t cPrevT = backward_ ? (actT + 1) : (actT - 1);
            __local_mem__ T* cPrevBase = useInit ? initCU : (cU + static_cast<int64_t>(cPrevT) * batch_ * haT);
            __local_mem__ float* dhSrc = dhBase + static_cast<uint32_t>(parity) * bhF;
            __local_mem__ float* dcSrc = dcBase + static_cast<uint32_t>(parity) * bhF;
            __local_mem__ float* dhDst = dhBase + static_cast<uint32_t>(1 - parity) * bhF;
            __local_mem__ float* dcDst = dcBase + static_cast<uint32_t>(1 - parity) * bhF;
            const uint32_t rowBase = static_cast<uint32_t>(actT) * static_cast<uint32_t>(batch_) * haT;
            const uint32_t gateBase = static_cast<uint32_t>(actT) * static_cast<uint32_t>(batch_) * LSTM_GATE_NUM * haF;

            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                uint32_t maskCntH = H;
                Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                Micro::RegTensor<float> one;
                Micro::Duplicate(one, 1.0f);
                for (uint16_t b = 0; b < B; ++b) {
                    uint32_t ro = rowBase + static_cast<uint32_t>(b) * haT;
                    uint32_t bo = static_cast<uint32_t>(b) * haF;
                    uint32_t go = gateBase + static_cast<uint32_t>(b) * LSTM_GATE_NUM * haF;
                    Micro::RegTensor<float> dyR, iR, jR, fR, oR, tanhR, cPrevR, dhN, dcN;
                    Micro::RegTensor<float> dht, tmpC, dcT, tmpJ, t0, dI, dJ, dF, dO, dcOut;
                    LoadF32<T>(dyU, dyR, mH, ro);
                    LoadF32<T>(iU, iR, mH, ro);
                    LoadF32<T>(jU, jR, mH, ro);
                    LoadF32<T>(fU, fR, mH, ro);
                    LoadF32<T>(oU, oR, mH, ro);
                    LoadF32<T>(tanhU, tanhR, mH, ro);
                    LoadF32<T>(cPrevBase, cPrevR, mH, static_cast<uint32_t>(b) * haT);
                    Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(dhN, dhSrc + bo);
                    Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(dcN, dcSrc + bo);
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
                    // dj
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
                    Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dgateU + go, dI, mH);
                    Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dgateU + go + sJ, dJ, mH);
                    Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dgateU + go + sF, dF, mH);
                    Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dgateU + go + sO, dO, mH);
                    Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dcDst + bo, dcOut, mH);
                }
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                // dh_next for the following step: dh[b][:] = sum_g dgate[b][g] * w_h[g][:]
                for (uint16_t b = 0; b < B; ++b) {
                    uint32_t go = gateBase + static_cast<uint32_t>(b) * LSTM_GATE_NUM * haF;
                    uint32_t bo = static_cast<uint32_t>(b) * haF;
                    Micro::RegTensor<float> acc;
                    Micro::Duplicate(acc, 0.0f);
                    for (uint16_t slot = 0; slot < static_cast<uint16_t>(LSTM_GATE_NUM); ++slot) {
                        uint32_t slotOff = go + static_cast<uint32_t>(slot) * haF;
                        uint32_t wRowOff = static_cast<uint32_t>(slot) * H * haT;
                        for (uint16_t hh = 0; hh < HLoop; ++hh) {
                            Micro::RegTensor<float> wRow;
                            Micro::RegTensor<float> s;
                            LoadF32<T>(whU, wRow, mH, wRowOff + static_cast<uint32_t>(hh) * haT);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(s, dgateU + slotOff + hh);
                            Micro::MulAddDst(acc, wRow, s, mH);
                        }
                    }
                    Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dhDst + bo, acc, mH);
                }
            }
            parity = 1 - parity;
        }
        finalParity_ = parity;
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

    __aicore__ inline void ProcessOneChunk(int32_t col0, int32_t w)
    {
        const int32_t wAlign = static_cast<int32_t>(AlignUpI64(static_cast<int64_t>(w) * sizeof(T), 32) / sizeof(T));
        __local_mem__ T* wChunkU = UbPtr<T>(layout_.wChunkOff);
        __local_mem__ T* xChunkU = UbPtr<T>(layout_.xChunkOff);
        __local_mem__ T* outU = UbPtr<T>(layout_.outStageOff);
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        __local_mem__ float* dgateU = UbPtr<float>(layout_.dgateOff);
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // stage w[:, col0:col0+w]
        PipeSync<AscendC::HardEvent::V_MTE2>();
        {
            AscendC::DataCopyExtParams p;
            p.blockCount = static_cast<uint16_t>(gates_);
            p.blockLen = static_cast<uint32_t>(w * sizeof(T));
            p.srcStride = static_cast<uint32_t>((cols_ - w) * sizeof(T));
            p.dstStride = 0; // auto 32B rounding -> wAlign pitch
            AscendC::DataCopyPad(UbTensor<T>(layout_.wChunkOff), wGm_[col0], p, pad);
        }
        PipeSync<AscendC::HardEvent::MTE2_V>();

        const uint16_t HLoop = static_cast<uint16_t>(hidden_);
        const uint32_t H = static_cast<uint32_t>(hidden_);
        const uint32_t haF = haF_;
        const uint32_t wAlignU = static_cast<uint32_t>(wAlign);

        // ---- dx: for every row m, dx[m, chunk] = sum_g dgate[m,g] * w[g, chunk]
        for (int32_t m0 = 0; m0 < mAll_; m0 += mBlock_) {
            const int32_t mbCur = ((mAll_ - m0) < mBlock_) ? (mAll_ - m0) : mBlock_;
            PipeSync<AscendC::HardEvent::MTE3_V>();
            const uint16_t mLoop = static_cast<uint16_t>(mbCur);
            const uint32_t mBase = static_cast<uint32_t>(m0) * LSTM_GATE_NUM * haF;
            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                uint32_t maskCntW = static_cast<uint32_t>(w);
                Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
                for (uint16_t mi = 0; mi < mLoop; ++mi) {
                    Micro::RegTensor<float> acc;
                    Micro::Duplicate(acc, 0.0f);
                    uint32_t gOff = mBase + static_cast<uint32_t>(mi) * LSTM_GATE_NUM * haF;
                    for (uint16_t slot = 0; slot < static_cast<uint16_t>(LSTM_GATE_NUM); ++slot) {
                        uint32_t slotOff = gOff + static_cast<uint32_t>(slot) * haF;
                        uint32_t wRowOff = static_cast<uint32_t>(slot) * H * wAlignU;
                        for (uint16_t hh = 0; hh < HLoop; ++hh) {
                            Micro::RegTensor<float> wRow;
                            Micro::RegTensor<float> s;
                            LoadF32<T>(wChunkU, wRow, mW, wRowOff + static_cast<uint32_t>(hh) * wAlignU);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(s, dgateU + slotOff + hh);
                            Micro::MulAddDst(acc, wRow, s, mW);
                        }
                    }
                    StoreF32<T>(outU, acc, mW, static_cast<uint32_t>(mi) * wAlignU);
                }
            }
            PipeSync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyExtParams p;
            p.blockCount = static_cast<uint16_t>(mbCur);
            p.blockLen = static_cast<uint32_t>(w * sizeof(T));
            p.srcStride = static_cast<uint32_t>((wAlign - w) * sizeof(T) / 32);
            p.dstStride = static_cast<uint32_t>((inputSize_ - w) * sizeof(T));
            AscendC::DataCopyPad(dxGm_[static_cast<int64_t>(m0) * inputSize_ + col0], UbTensor<T>(layout_.outStageOff),
                                 p);
        }

        // ---- dw: dwAcc[g, chunk] = sum_m dgate[m,g] * x[m, chunk]
        ZeroDwAcc(w);
        for (int32_t m0 = 0; m0 < mAll_; m0 += mBlock_) {
            const int32_t mbCur = ((mAll_ - m0) < mBlock_) ? (mAll_ - m0) : mBlock_;
            PipeSync<AscendC::HardEvent::V_MTE2>();
            {
                AscendC::DataCopyExtParams p;
                p.blockCount = static_cast<uint16_t>(mbCur);
                p.blockLen = static_cast<uint32_t>(w * sizeof(T));
                p.srcStride = static_cast<uint32_t>((inputSize_ - w) * sizeof(T));
                p.dstStride = 0; // auto 32B rounding -> wAlign pitch
                AscendC::DataCopyPad(UbTensor<T>(layout_.xChunkOff), xGm_[static_cast<int64_t>(m0) * inputSize_ + col0],
                                     p, pad);
            }
            PipeSync<AscendC::HardEvent::MTE2_V>();
            const uint16_t mLoop = static_cast<uint16_t>(mbCur);
            const uint32_t mBase = static_cast<uint32_t>(m0) * LSTM_GATE_NUM * haF;
            const uint32_t CW = static_cast<uint32_t>(chunkCols_);
            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                uint32_t maskCntW = static_cast<uint32_t>(w);
                Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntW);
                for (uint16_t slot = 0; slot < static_cast<uint16_t>(LSTM_GATE_NUM); ++slot) {
                    for (uint16_t hh = 0; hh < HLoop; ++hh) {
                        uint32_t g = static_cast<uint32_t>(slot) * H + static_cast<uint32_t>(hh);
                        Micro::RegTensor<float> acc;
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(acc, dwAccU + g * CW);
                        for (uint16_t mi = 0; mi < mLoop; ++mi) {
                            Micro::RegTensor<float> xRow;
                            Micro::RegTensor<float> s;
                            LoadF32<T>(xChunkU, xRow, mW, static_cast<uint32_t>(mi) * wAlignU);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(
                                s, dgateU + mBase + static_cast<uint32_t>(mi) * LSTM_GATE_NUM * haF +
                                       static_cast<uint32_t>(slot) * haF + hh);
                            Micro::MulAddDst(acc, xRow, s, mW);
                        }
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dwAccU + g * CW, acc, mW);
                    }
                }
            }
        }
        StoreDwChunk(col0, w, wAlign);
    }

    __aicore__ inline void ZeroDwAcc(int32_t w)
    {
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        const uint16_t GLoop = static_cast<uint16_t>(gates_);
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

    // cast dwAcc into dtype T staging and DMA into dw[:, col0:col0+w]
    __aicore__ inline void StoreDwChunk(int32_t col0, int32_t w, int32_t wAlign)
    {
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        __local_mem__ T* outU = UbPtr<T>(layout_.outStageOff);
        const uint16_t GLoop = static_cast<uint16_t>(gates_);
        const uint32_t CW = static_cast<uint32_t>(chunkCols_);
        const uint32_t wAlignU = static_cast<uint32_t>(wAlign);
        PipeSync<AscendC::HardEvent::MTE3_V>();
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
        PipeSync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(gates_);
        p.blockLen = static_cast<uint32_t>(w * sizeof(T));
        p.srcStride = static_cast<uint32_t>((wAlign - w) * sizeof(T) / 32);
        p.dstStride = static_cast<uint32_t>((cols_ - w) * sizeof(T));
        AscendC::DataCopyPad(dwGm_[col0], UbTensor<T>(layout_.outStageOff), p);
    }

    // last core: dw hidden columns (from h_prev), db, dh_prev, dc_prev
    __aicore__ inline void ProcessTail()
    {
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        __local_mem__ float* dgateU = UbPtr<float>(layout_.dgateOff);
        __local_mem__ T* hU = UbPtr<T>(layout_.hOff);
        __local_mem__ T* initHU = UbPtr<T>(layout_.initHOff);
        const uint16_t B = static_cast<uint16_t>(batch_);
        const uint16_t HLoop = static_cast<uint16_t>(hidden_);
        const uint32_t H = static_cast<uint32_t>(hidden_);
        const uint32_t haT = haT_;
        const uint32_t haF = haF_;
        const uint32_t CW = static_cast<uint32_t>(chunkCols_);

        // dw[:, I:I+H] += h_prev(t)^T @ dgate(t) over all t
        ZeroDwAcc(hidden_);
        for (int32_t actT = 0; actT < timeStep_; ++actT) {
            const bool useInitH = backward_ ? (actT == timeStep_ - 1) : (actT == 0);
            const int32_t hPrevT = backward_ ? (actT + 1) : (actT - 1);
            __local_mem__ T* hBase = useInitH ? initHU : (hU + static_cast<int64_t>(hPrevT) * batch_ * haT);
            const uint32_t gateBase = static_cast<uint32_t>(actT) * static_cast<uint32_t>(batch_) * LSTM_GATE_NUM * haF;
            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                uint32_t maskCntH = H;
                Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                for (uint16_t slot = 0; slot < static_cast<uint16_t>(LSTM_GATE_NUM); ++slot) {
                    for (uint16_t hh = 0; hh < HLoop; ++hh) {
                        uint32_t g = static_cast<uint32_t>(slot) * H + static_cast<uint32_t>(hh);
                        Micro::RegTensor<float> acc;
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(acc, dwAccU + g * CW);
                        for (uint16_t b = 0; b < B; ++b) {
                            Micro::RegTensor<float> hRow;
                            Micro::RegTensor<float> s;
                            LoadF32<T>(hBase, hRow, mH, static_cast<uint32_t>(b) * haT);
                            Micro::DataCopy<float, Micro::LoadDist::DIST_BRC_B32>(
                                s, dgateU + gateBase + static_cast<uint32_t>(b) * LSTM_GATE_NUM * haF +
                                       static_cast<uint32_t>(slot) * haF + hh);
                            Micro::MulAddDst(acc, hRow, s, mH);
                        }
                        Micro::DataCopy<float, Micro::StoreDist::DIST_NORM>(dwAccU + g * CW, acc, mH);
                    }
                }
            }
        }
        {
            const int32_t wAlign = static_cast<int32_t>(AlignUpI64(static_cast<int64_t>(hidden_) * sizeof(T), 32) /
                                                        sizeof(T));
            StoreDwChunkTail(wAlign);
        }

        // db[slot*H : slot*H+H] = sum_m dgate[m][slot][:]
        if (isBias_) {
            __local_mem__ T* dbStage = UbPtr<T>(layout_.smallStageOff);
            const uint16_t mLoop = static_cast<uint16_t>(mAll_);
            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                uint32_t maskCntH = H;
                Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                for (uint16_t slot = 0; slot < static_cast<uint16_t>(LSTM_GATE_NUM); ++slot) {
                    uint32_t slotOff = static_cast<uint32_t>(slot) * haF;
                    Micro::RegTensor<float> acc;
                    Micro::Duplicate(acc, 0.0f);
                    for (uint16_t m = 0; m < mLoop; ++m) {
                        Micro::RegTensor<float> r;
                        Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(
                            r, dgateU + static_cast<uint32_t>(m) * LSTM_GATE_NUM * haF + slotOff);
                        Micro::Add(acc, acc, r, mH);
                    }
                    StoreF32<T>(dbStage, acc, mH, static_cast<uint32_t>(slot) * haT);
                }
            }
            PipeSync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyExtParams p{static_cast<uint16_t>(LSTM_GATE_NUM),
                                         static_cast<uint32_t>(hidden_ * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(dbGm_[0], UbTensor<T>(layout_.smallStageOff), p);
        }

        // dh_prev / dc_prev
        {
            __local_mem__ float* dhFinal = UbPtr<float>(layout_.dhCurOff) +
                                           static_cast<uint32_t>(finalParity_) * static_cast<uint32_t>(batch_) * haF;
            __local_mem__ float* dcFinal = UbPtr<float>(layout_.dcCurOff) +
                                           static_cast<uint32_t>(finalParity_) * static_cast<uint32_t>(batch_) * haF;
            const int64_t dhStageOff = layout_.smallStageOff + static_cast<int64_t>(LSTM_GATE_NUM) * haT_ * sizeof(T);
            const int64_t dcStageOff = dhStageOff + static_cast<int64_t>(batch_) * haT_ * sizeof(T);
            __local_mem__ T* dhStage = UbPtr<T>(dhStageOff);
            __local_mem__ T* dcStage = UbPtr<T>(dcStageOff);
            __VEC_SCOPE__
            {
                Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
                Micro::RegTensor<float> r;
                uint32_t maskCntH = H;
                Micro::MaskReg mH = Micro::UpdateMask<float>(maskCntH);
                for (uint16_t b = 0; b < B; ++b) {
                    uint32_t srcOff = static_cast<uint32_t>(b) * haF;
                    uint32_t dstOff = static_cast<uint32_t>(b) * haT;
                    Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(r, dhFinal + srcOff);
                    StoreF32<T>(dhStage, r, mH, dstOff);
                    Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(r, dcFinal + srcOff);
                    StoreF32<T>(dcStage, r, mH, dstOff);
                }
            }
            PipeSync<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyExtParams p{static_cast<uint16_t>(batch_), static_cast<uint32_t>(hidden_ * sizeof(T)), 0,
                                         0, 0};
            AscendC::DataCopyPad(dhPrevGm_[0], UbTensor<T>(dhStageOff), p);
            AscendC::DataCopyPad(dcPrevGm_[0], UbTensor<T>(dcStageOff), p);
        }
    }

    // dwAcc -> dw[:, I:I+H] (tail chunk of width H)
    __aicore__ inline void StoreDwChunkTail(int32_t wAlign)
    {
        __local_mem__ float* dwAccU = UbPtr<float>(layout_.dwAccOff);
        __local_mem__ T* outU = UbPtr<T>(layout_.outStageOff);
        const uint16_t GLoop = static_cast<uint16_t>(gates_);
        const uint32_t CW = static_cast<uint32_t>(chunkCols_);
        const uint32_t wAlignU = static_cast<uint32_t>(wAlign);
        const uint32_t H = static_cast<uint32_t>(hidden_);
        PipeSync<AscendC::HardEvent::MTE3_V>();
        __VEC_SCOPE__
        {
            Micro::LocalMemBar<Micro::MemType::VEC_STORE, Micro::MemType::VEC_LOAD>();
            uint32_t maskCntH = H;
            Micro::MaskReg mW = Micro::UpdateMask<float>(maskCntH);
            for (uint16_t g = 0; g < GLoop; ++g) {
                Micro::RegTensor<float> r;
                Micro::DataCopy<float, Micro::LoadDist::DIST_NORM>(r, dwAccU + static_cast<uint32_t>(g) * CW);
                StoreF32<T>(outU, r, mW, static_cast<uint32_t>(g) * wAlignU);
            }
        }
        PipeSync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams p;
        p.blockCount = static_cast<uint16_t>(gates_);
        p.blockLen = static_cast<uint32_t>(hidden_ * sizeof(T));
        p.srcStride = static_cast<uint32_t>((wAlign - hidden_) * sizeof(T) / 32);
        p.dstStride = static_cast<uint32_t>(inputSize_ * sizeof(T));
        AscendC::DataCopyPad(dwGm_[inputSize_], UbTensor<T>(layout_.outStageOff), p);
    }

private:
    int32_t timeStep_{0};
    int32_t batch_{0};
    int32_t inputSize_{0};
    int32_t hidden_{0};
    int32_t gates_{0};
    int32_t mAll_{0};
    int32_t cols_{0};
    bool isBias_{false};
    bool backward_{false};
    int32_t gateOrder_{0};
    int32_t slotJ_{1};
    int32_t slotF_{2};
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

    AscendC::GlobalTensor<T> xGm_, wGm_, initHGm_, initCGm_, hGm_, cGm_, dyGm_, dhGm_, dcGm_;
    AscendC::GlobalTensor<T> iGm_, jGm_, fGm_, oGm_, tanhGm_;
    AscendC::GlobalTensor<T> dwGm_, dbGm_, dxGm_, dhPrevGm_, dcPrevGm_;
};

} // namespace LstmGradRegbase

#endif // SINGLE_LAYER_LSTM_GRAD_REGBASE_SMALL_H
