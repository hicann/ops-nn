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
 * \file turbo_quant.h
 * \brief TurboQuant 4-bit quantization of the MLA KV latent.
 *
 * Input  latent    [numTokens, headDim] fp32, already rotated by the signed Hadamard and NOT normalized.
 * Input  centroids [16]                 fp32, the Lloyd-Max codebook, sorted ascending.
 * Output y         [numTokens, headDim / 2] uint8, packed TurboQuant 4-bit indices.
 * Output scale     [numTokens] float16, the per-token L2 norm.
 *
 * Per token: norm = ||z|| ; u = z / norm ; nibble[d] = #{midpoint boundaries <= u[d]}, i.e. the index of
 * the nearest centroid ; the nibbles are packed two per byte in dim order into y, and the fp16 norm is
 * written to scale.
 *
 * The nearest-centroid search is a chain of 15 dependent compare/select/add rounds whose per-instruction
 * issue cost dominates the actual arithmetic, so tokensPerBatch tokens are folded into every vector
 * instruction. Only the steps that are inherently per-token stay narrow: the L2 reduction, the per-token
 * rescale and the strided nibble store.
 */
#ifndef TURBO_QUANT_H
#define TURBO_QUANT_H

#include "kernel_operator.h"

namespace TurboQuant {
using namespace AscendC;

constexpr uint32_t N_CENT = 16;
constexpr uint32_t ALIGN_BYTES = 64;
// Keep in sync with TQ_MAX_TOKENS_PER_BATCH in the tiling header.
constexpr uint32_t MAX_TOKENS_PER_BATCH = 12;
// Each token's L2 norm reduces into its own 64B-aligned slot so one V->S sync covers the whole batch.
constexpr uint32_t NORM_SLOT_FLOATS = 16;
constexpr float NORM_EPS = 1e-16f;
constexpr float NIBBLE_SIGN_THRESHOLD = 8.0f;
constexpr float NIBBLE_SIGN_OFFSET = -16.0f;

__aicore__ inline uint32_t AlignUpTo(uint32_t value, uint32_t align) { return (value + align - 1) / align * align; }

class KernelTurboQuant {
public:
    __aicore__ inline KernelTurboQuant() {}

    __aicore__ inline void Init(GM_ADDR latent, GM_ADDR centroids, GM_ADDR yOut, GM_ADDR scaleOut, uint32_t numTokens,
                                uint32_t tokensPerCore, uint32_t headDim, uint32_t packedBytes, uint32_t tokensPerBatch)
    {
        numTokens_ = numTokens;
        headDim_ = headDim;
        packedBytes_ = packedBytes;
        batch_ = tokensPerBatch < 1 ? 1 : tokensPerBatch;
        if (batch_ > MAX_TOKENS_PER_BATCH) {
            batch_ = MAX_TOKENS_PER_BATCH;
        }

        uint32_t coreIdx = GetBlockIdx();
        tokStart_ = coreIdx * tokensPerCore;
        tokEnd_ = tokStart_ + tokensPerCore;
        if (tokEnd_ > numTokens_) {
            tokEnd_ = numTokens_;
        }

        latentGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(latent));
        centGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(centroids));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(yOut));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scaleOut));

        uint32_t batchElems = batch_ * headDim_;
        uint32_t fp32Bytes = AlignUpTo(batchElems * sizeof(float), ALIGN_BYTES);
        uint32_t halfBytes = AlignUpTo(batchElems * sizeof(half), ALIGN_BYTES);
        uint32_t maskBytes = AlignUpTo(batchElems / 8, ALIGN_BYTES);
        uint32_t yBytes = AlignUpTo(batch_ * packedBytes_, ALIGN_BYTES);
        uint32_t workBytes = AlignUpTo(headDim_ * sizeof(float), ALIGN_BYTES);
        uint32_t normBytes = AlignUpTo(batch_ * NORM_SLOT_FLOATS * sizeof(float), ALIGN_BYTES);

        pipe_.InitBuffer(inBuf_, fp32Bytes);
        pipe_.InitBuffer(yBuf_, yBytes);
        pipe_.InitBuffer(uBuf_, fp32Bytes);
        pipe_.InitBuffer(nibBuf_, fp32Bytes);
        pipe_.InitBuffer(tmpBuf_, fp32Bytes);
        pipe_.InitBuffer(selBuf_, fp32Bytes);
        pipe_.InitBuffer(oneBuf_, fp32Bytes);
        pipe_.InitBuffer(packHalfBuf_, halfBytes);
        pipe_.InitBuffer(maskBuf_, maskBytes);
        pipe_.InitBuffer(workBuf_, workBytes);
        pipe_.InitBuffer(normBuf_, normBytes);
        pipe_.InitBuffer(centBuf_, AlignUpTo(N_CENT * sizeof(float), ALIGN_BYTES));
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> cent = centBuf_.Get<float>();
        DataCopyExtParams centCopyParams{1, static_cast<uint32_t>(N_CENT * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> centPadParams{false, 0, 0, 0};
        DataCopyPad(cent, centGm_, centCopyParams, centPadParams);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i + 1 < N_CENT; ++i) {
            // 15 midpoint boundaries; counting how many a value exceeds yields the nearest-centroid index
            bnd_[i] = (cent.GetValue(i) + cent.GetValue(i + 1)) * 0.5f;
        }
        LocalTensor<float> one = oneBuf_.Get<float>();
        Duplicate(one, 1.0f, batchElems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = tokStart_; t < tokEnd_; t += batch_) {
            uint32_t count = tokEnd_ - t;
            if (count > batch_) {
                count = batch_;
            }
            CopyIn(t, count);
            Compute(count);
            CopyOut(t, count);
        }
    }

private:
    __aicore__ inline void WaitVectorToScalar()
    {
        event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
    }

    __aicore__ inline void CopyIn(uint32_t t, uint32_t count)
    {
        LocalTensor<float> in = inBuf_.Get<float>();
        DataCopy(in, latentGm_[static_cast<uint64_t>(t) * headDim_], count * headDim_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        const uint32_t elems = count * headDim_;
        LocalTensor<float> in = inBuf_.Get<float>();
        ComputeCore(count, in, elems);
    }

    __aicore__ inline void ComputeCore(uint32_t count, LocalTensor<float> in, uint32_t elems)
    {
        LocalTensor<uint8_t> y = yBuf_.Get<uint8_t>();
        // Reuse the packed-y half workspace after the y cast has completed.
        LocalTensor<half> scale = packHalfBuf_.Get<half>();
        LocalTensor<float> u = uBuf_.Get<float>();
        LocalTensor<float> nib = nibBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> sel = selBuf_.Get<float>();
        LocalTensor<float> one = oneBuf_.Get<float>();
        LocalTensor<float> work = workBuf_.Get<float>();
        LocalTensor<float> norm = normBuf_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        Mul(tmp, in, in, elems);
        PipeBarrier<PIPE_V>();
        // The L2 reduction is per token; each result lands in its own slot so one V->S sync covers them all.
        for (uint32_t i = 0; i < count; ++i) {
            ReduceSum(norm[i * NORM_SLOT_FLOATS], tmp[i * headDim_], work, headDim_);
        }
        PipeBarrier<PIPE_V>();
        WaitVectorToScalar();
        for (uint32_t i = 0; i < count; ++i) {
            normScalar_[i] = sqrt(norm.GetValue(i * NORM_SLOT_FLOATS) + NORM_EPS);
            Muls(u[i * headDim_], in[i * headDim_], 1.0f / normScalar_[i], headDim_); // u = z / norm
        }
        PipeBarrier<PIPE_V>();

        Duplicate(nib, 0.0f, elems);
        PipeBarrier<PIPE_V>();
        for (uint32_t b = 0; b + 1 < N_CENT; ++b) {
            CompareScalar(mask, u, bnd_[b], CMPMODE::GE, elems); // mask = u >= bnd[b]
            PipeBarrier<PIPE_V>();
            Select(sel, mask, one, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, elems);
            PipeBarrier<PIPE_V>();
            Add(nib, nib, sel, elems);
            PipeBarrier<PIPE_V>();
        }

        // int4b_t HW pack: nib(0..15, dim order) -> signed s(-8..7) -> half -> int4b_t (low nibble first).
        // s = (nib < 8) ? nib : nib - 16, i.e. the same 4 bits reinterpreted as two's complement.
        LocalTensor<half> packHalf = packHalfBuf_.Get<half>();
        CompareScalar(mask, nib, NIBBLE_SIGN_THRESHOLD, CMPMODE::LT, elems);
        PipeBarrier<PIPE_V>();
        Adds(sel, nib, NIBBLE_SIGN_OFFSET, elems);
        PipeBarrier<PIPE_V>();
        Select(tmp, mask, nib, sel, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
        PipeBarrier<PIPE_V>();
        Cast(packHalf, tmp, RoundMode::CAST_RINT, elems);
        PipeBarrier<PIPE_V>();
        // y is contiguous per token, so the packed nibbles are stored with no row padding.
        for (uint32_t i = 0; i < count; ++i) {
            LocalTensor<int4b_t> packed = y[i * packedBytes_].ReinterpretCast<int4b_t>();
            Cast(packed, packHalf[i * headDim_], RoundMode::CAST_RINT, headDim_);
        }
        PipeBarrier<PIPE_V>();

        WaitVectorToScalar();
        for (uint32_t i = 0; i < count; ++i) {
            scale.SetValue(i, static_cast<half>(normScalar_[i]));
        }

        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOut(uint32_t t, uint32_t count)
    {
        LocalTensor<uint8_t> y = yBuf_.Get<uint8_t>();
        LocalTensor<half> scale = packHalfBuf_.Get<half>();
        DataCopy(yGm_[static_cast<uint64_t>(t) * packedBytes_], y, count * packedBytes_);
        DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        DataCopyPad(scaleGm_[t], scale, scaleCopyParams);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> inBuf_, yBuf_;
    TBuf<TPosition::VECCALC> uBuf_, nibBuf_, tmpBuf_, selBuf_, oneBuf_, maskBuf_;
    TBuf<TPosition::VECCALC> packHalfBuf_, workBuf_, normBuf_, centBuf_;
    GlobalTensor<float> latentGm_;
    GlobalTensor<float> centGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<half> scaleGm_;
    uint32_t numTokens_ = 0;
    uint32_t headDim_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t batch_ = 1;
    uint32_t tokStart_ = 0;
    uint32_t tokEnd_ = 0;
    float bnd_[N_CENT] = {};
    float normScalar_[MAX_TOKENS_PER_BATCH] = {};
};

} // namespace TurboQuant
#endif // TURBO_QUANT_H
