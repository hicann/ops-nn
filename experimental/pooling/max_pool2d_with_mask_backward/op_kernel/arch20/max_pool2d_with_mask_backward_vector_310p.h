// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

#ifndef MAXPOOL_BACKWARD_VECTOR_310P_H
#define MAXPOOL_BACKWARD_VECTOR_310P_H
#include "copy32_310p.h"
namespace MaxPool2dMask310P {
using namespace AscendC;
// 910B Normal-style multi-row / column / C8 or C16 block. Each output rectangle
// has one writer; its gradient rectangle is loaded and transposed once.
// All overlapping contributions accumulate in FP32, with one final FP16 cast.
template <typename T, int64_t NC = 16, int64_t INPUT_ELEMENTS = 8192, int64_t OUTPUT_ELEMENTS = 12288,
          bool CACHE_COLUMNS = false, bool PLANAR_RELATIVE = false, bool INITIALIZE_ONCE = false, int STATIC_KH = 0,
          int STATIC_KW = 0, int STATIC_SH = 0, int STATIC_SW = 0, bool REUSE_PLANAR = false, int RAMP_ELEMENTS = 0,
          bool ROUND_PADDED_COLS = false>
class VectorKernel {
public:
    template <class TD>
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR index, GM_ADDR out, const TD* t, TPipe* pipe)
    {
        pipe_ = pipe;
        nc_ = t->ncDim;
        hi_ = t->hiDim;
        wi_ = t->wiDim;
        ho_ = t->hoDim;
        wo_ = t->woDim;
        kh_ = t->kh;
        kw_ = t->kw;
        sh_ = t->sh;
        sw_ = t->sw;
        ph_ = t->padHTop;
        pw_ = t->padWTop;
        tileH_ = t->baseHo;
        tileW_ = t->baseWo;
        cuts_ = Max(1, wi_ / tileW_);
        hcuts_ = (hi_ + tileH_ - 1) / tileH_;
        tiles_ = ((nc_ + NC - 1) / NC) * hcuts_ * cuts_;
        grad_.SetGlobalBuffer((__gm__ T*)grad);
        index_.SetGlobalBuffer((__gm__ int32_t*)index);
        out_.SetGlobalBuffer((__gm__ T*)out);
        if constexpr (REUSE_PLANAR) {
            static_assert(NC == 8 && !INITIALIZE_ONCE, "Shared planar path is C8 with per-tile initialization");
            pipe->InitBuffer(shared_, SHARED_BYTES);
            // The accumulator is dead after transpose and stages the output tails.
        } else {
            pipe->InitBuffer(gp_, GP_BYTES);
            pipe->InitBuffer(ip_, NC * IC * 4);
            pipe->InitBuffer(gt_, NC * IC * 4);
            pipe->InitBuffer(it_, NC * IC * 4);
            pipe->InitBuffer(op_, NC * OC * 4);
        }
        pipe->InitBuffer(ot_, NC * OC * 4);
        pipe->InitBuffer(mask_, NC * IC / 8);
        pipe->InitBuffer(low_, RC * 4);
        copy_.Init(pipe);
        icopy_.Init(pipe);
        outputEvent_ = pipe->FetchEventID(HardEvent::MTE3_V);
        if constexpr (INITIALIZE_ONCE) {
            static_assert(sizeof(T) == 4, "One-time initialization is FP32 only");
            Duplicate(GetGP<float>(), 0.0f, NC * IC);
            Duplicate(GetIP<int32_t>(), static_cast<int32_t>(0), NC * IC);
            PipeBarrier<PIPE_V>();
        }
        auto ramp = low_.Get<int32_t>();
        CreateVecIndex(ramp, static_cast<int32_t>(0), static_cast<uint32_t>(RC));
        PipeBarrier<PIPE_V>();
        Muls(ramp, ramp, static_cast<int32_t>(sw_), static_cast<int32_t>(RC));
        PipeBarrier<PIPE_V>();
        InitTransposeIn(gradD_, gradS_, GetGT<float>(), GetGP<float>());
        InitTransposeIn(idxD_, idxS_, GetIT<int32_t>(), GetIP<int32_t>());
        InitTransposeIn(baseD_, baseS_, GetGP<int32_t>(), GetIP<int32_t>());
        InitTransposeOut(outD_, outS_, GetOP<float>(), ot_.Get<float>());
    }
    __aicore__ inline void Process()
    {
        const int64_t kh_ = STATIC_KH ? STATIC_KH : this->kh_;
        const int64_t kw_ = STATIC_KW ? STATIC_KW : this->kw_;
        const int64_t sh_ = STATIC_SH ? STATIC_SH : this->sh_;
        const int64_t sw_ = STATIC_SW ? STATIC_SW : this->sw_;

        for (int64_t tile = GetBlockIdx(); tile < tiles_; tile += GetBlockNum()) {
            int64_t group = tile / (hcuts_ * cuts_), y = (tile / cuts_ % hcuts_) * tileH_;
            int64_t cut = tile % cuts_, x = cut * tileW_, width = cut + 1 == cuts_ ? wi_ - x : tileW_;
            int64_t height = Min(tileH_, hi_ - y), opitch = Align16(width);
            int64_t nc0 = group * NC, channels = Min(NC, nc_ - nc0);
            if constexpr (REUSE_PLANAR)
                WaitOut();
            auto acc = ot_.Get<float>();
            Duplicate(acc, 0.0f, NC * height * opitch);
            PipeBarrier<PIPE_V>();
            int64_t q0 = Max(0, Ceil(x + pw_ - kw_ + 1, sw_));
            int64_t qend = Min(wo_, (x + width - 1 + pw_) / sw_ + 1);
            int64_t r0 = Max(0, Ceil(y + ph_ - kh_ + 1, sh_));
            int64_t rend = Min(ho_, (y + height - 1 + ph_) / sh_ + 1);
            if (q0 < qend && r0 < rend) {
                int64_t pitch = AlignInput(qend - q0), rows = rend - r0, positions = rows * pitch;
                Load(nc0, channels, r0, q0, rows, qend - q0, pitch);
                auto gp = GetGP<float>();
                auto ip = GetIP<int32_t>();
                auto grad = GetGT<float>();
                auto idx = GetIT<int32_t>();
                Transpose<float>(gradD_, gradS_, positions, true);
                if constexpr (PLANAR_RELATIVE)
                    PipeBarrier<PIPE_V>();
                int32_t count = static_cast<int32_t>(positions * NC);
                if constexpr (PLANAR_RELATIVE) {
                    // Build one planar base and subtract it before transposing indices.
                    auto ramp = low_.Get<int32_t>();
                    for (int64_t r = 0; r < rows; ++r)
                        Adds(idx[r * pitch], ramp, static_cast<int32_t>(((r + r0) * sh_ - ph_) * wi_ + q0 * sw_ - pw_),
                             static_cast<int32_t>(pitch));
                    PipeBarrier<PIPE_V>();
                    for (int64_t c = 0; c < NC; ++c)
                        Sub(ip[c * IC], ip[c * IC], idx, static_cast<int32_t>(positions));
                    PipeBarrier<PIPE_V>();
                    Transpose<int32_t>(idxD_, idxS_, positions, true);
                    PipeBarrier<PIPE_V>();
                } else {
                    Transpose<int32_t>(idxD_, idxS_, positions, true);
                    PipeBarrier<PIPE_V>();
                    // Form one planar spatial ramp, broadcast it to the channel block, then transpose.
                    // This avoids issuing one Duplicate per spatial position.
                    auto ramp = low_.Get<int32_t>();
                    for (int64_t r = 0; r < rows; ++r)
                        Adds(ip[r * pitch], ramp, static_cast<int32_t>(((r + r0) * sh_ - ph_) * wi_ + q0 * sw_ - pw_),
                             static_cast<int32_t>(pitch));
                    PipeBarrier<PIPE_V>();
                    for (int64_t c = 1; c < NC; ++c)
                        Adds(ip[c * IC], ip, static_cast<int32_t>(0), static_cast<int32_t>(positions));
                    PipeBarrier<PIPE_V>();
                    Transpose<int32_t>(baseD_, baseS_, positions, true);
                    PipeBarrier<PIPE_V>();
                    Sub(idx, idx, gp.template ReinterpretCast<int32_t>(), count);
                    PipeBarrier<PIPE_V>();
                }
                // Relative offsets are bounded by tiling to <2^24. FP32 equality is exact.
                Cast(gp, idx, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                auto selected = ip.template ReinterpretCast<float>();
                auto mask = mask_.Get<uint8_t>();
                // Descending kh/kw means ascending contributing grad rows/columns at each
                // output pixel, matching the scalar baseline's accumulation order.
                int64_t cachedLo[16], cachedEnd[16];
                if constexpr (CACHE_COLUMNS) {
                    for (int64_t b = kw_ - 1; b >= 0; --b) {
                        cachedLo[b] = Max(q0, Ceil(x + pw_ - b, sw_));
                        cachedEnd[b] = Min(qend, Ceil(x + width + pw_ - b, sw_));
                    }
                }
                for (int64_t a = kh_ - 1; a >= 0; --a) {
                    int64_t first = Max(r0, Ceil(y + ph_ - a, sh_));
                    int64_t last = Min(rend, Ceil(y + height + ph_ - a, sh_));
                    if (first >= last)
                        continue;
                    for (int64_t b = kw_ - 1; b >= 0; --b) {
                        int64_t lo, end;
                        if constexpr (CACHE_COLUMNS) {
                            lo = cachedLo[b];
                            end = cachedEnd[b];
                        } else {
                            lo = Max(q0, Ceil(x + pw_ - b, sw_));
                            end = Min(qend, Ceil(x + width + pw_ - b, sw_));
                        }
                        if (lo >= end)
                            continue;
                        int64_t paddedColumns = end - lo;
                        if constexpr (ROUND_PADDED_COLS) {
                            // qend may stop before the last valid output pixel (e.g. even W,
                            // b=0). Extra columns are legal only after the stored row ends.
                            if (end * sw_ - pw_ + b - x >= width)
                                paddedColumns = (opitch - 1 - (lo * sw_ - pw_ + b - x)) / sw_ + 1;
                        }
                        const int64_t activeOffset = (first - r0) * pitch * NC;
                        const int32_t activeCount = static_cast<int32_t>((last - first) * pitch * NC);
                        CompareScalar(mask, gp[activeOffset], static_cast<float>(a * wi_ + b), CMPMODE::EQ,
                                      activeCount);
                        PipeBarrier<PIPE_V>();
                        Select(selected[activeOffset], mask, grad[activeOffset], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                               activeCount);
                        PipeBarrier<PIPE_V>();
                        if constexpr (CACHE_COLUMNS) {
                            uint64_t dst = ((first * sh_ - ph_ + a - y) * opitch + lo * sw_ - pw_ + b - x) * NC;
                            uint64_t src = ((first - r0) * pitch + lo - q0) * NC;
                            const uint64_t dstStep = sh_ * opitch * NC, srcStep = pitch * NC;
                            for (int64_t r = first; r < last; ++r) {
                                Col2ImgAdd(acc, selected, dst, src, end - lo, paddedColumns);
                                dst += dstStep;
                                src += srcStep;
                            }
                        } else {
                            for (int64_t r = first; r < last; ++r) {
                                uint64_t dst = ((r * sh_ - ph_ + a - y) * opitch + lo * sw_ - pw_ + b - x) * NC;
                                uint64_t src = ((r - r0) * pitch + lo - q0) * NC;
                                Col2ImgAdd(acc, selected, dst, src, end - lo, paddedColumns);
                            }
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
            Store(nc0, channels, y, x, height, width, opitch, acc);
        }
        WaitOut();
    }

private:
    // Default C16: 232640 bytes; C8: 234688 bytes including Copy32 scratch.
    // Sparse C8 (6144/16384): FP32 233408 / FP16 241600 bytes.
    // gp_ is reused by the FP16 output cast and must fit both lifetimes.
    static constexpr int64_t IC = INPUT_ELEMENTS / NC, OC = OUTPUT_ELEMENTS / NC;
    static constexpr int64_t RC = RAMP_ELEMENTS ? RAMP_ELEMENTS : IC;
    static constexpr int64_t GP_BYTES = sizeof(T) == 2 && OUTPUT_ELEMENTS * 2 > INPUT_ELEMENTS * 4 ?
                                            OUTPUT_ELEMENTS * 2 :
                                            INPUT_ELEMENTS * 4;
    static constexpr int64_t SHARED_BYTES = 4 * INPUT_ELEMENTS * 4 > OUTPUT_ELEMENTS * 4 ? 4 * INPUT_ELEMENTS * 4 :
                                                                                           OUTPUT_ELEMENTS * 4;
    static_assert((REUSE_PLANAR ? SHARED_BYTES + OUTPUT_ELEMENTS * 4 :
                                  GP_BYTES + 3 * INPUT_ELEMENTS * 4 + 2 * OUTPUT_ELEMENTS * 4) +
                          INPUT_ELEMENTS / 8 + RC * 4 + 192 <=
                      245760,
                  "Vector UB budget exceeded");
    template <class U>
    __aicore__ inline LocalTensor<U> GetGP()
    {
        if constexpr (REUSE_PLANAR)
            return shared_.Get<U>()[0 * INPUT_ELEMENTS * 4 / sizeof(U)];
        else
            return gp_.Get<U>();
    }
    template <class U>
    __aicore__ inline LocalTensor<U> GetIP()
    {
        if constexpr (REUSE_PLANAR)
            return shared_.Get<U>()[1 * INPUT_ELEMENTS * 4 / sizeof(U)];
        else
            return ip_.Get<U>();
    }
    template <class U>
    __aicore__ inline LocalTensor<U> GetGT()
    {
        if constexpr (REUSE_PLANAR)
            return shared_.Get<U>()[2 * INPUT_ELEMENTS * 4 / sizeof(U)];
        else
            return gt_.Get<U>();
    }
    template <class U>
    __aicore__ inline LocalTensor<U> GetIT()
    {
        if constexpr (REUSE_PLANAR)
            return shared_.Get<U>()[3 * INPUT_ELEMENTS * 4 / sizeof(U)];
        else
            return it_.Get<U>();
    }
    template <class U>
    __aicore__ inline LocalTensor<U> GetOP()
    {
        if constexpr (REUSE_PLANAR)
            return shared_.Get<U>()[0 * INPUT_ELEMENTS * 4 / sizeof(U)];
        else
            return op_.Get<U>();
    }
    __aicore__ inline int64_t Min(int64_t a, int64_t b) { return a < b ? a : b; }
    __aicore__ inline int64_t Max(int64_t a, int64_t b) { return a > b ? a : b; }
    __aicore__ inline int64_t Align16(int64_t a) { return (a + 15) / 16 * 16; }
    __aicore__ inline int64_t AlignInput(int64_t a)
    {
        // FP32 C8 rows need 32B (8 elements); FP16 staging still needs 16 elements.
        constexpr int64_t alignment = NC == 8 && sizeof(T) == 4 ? 8 : 16;
        return (a + alignment - 1) / alignment * alignment;
    }
    __aicore__ inline int64_t Ceil(int64_t a, int64_t b) { return a >= 0 ? (a + b - 1) / b : a / b; }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        auto e = pipe_->FetchEventID(E);
        SetFlag<E>(e);
        WaitFlag<E>(e);
    }
    __aicore__ inline void WaitOut()
    {
        if (outputPending_) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent_);
            outputPending_ = false;
        }
    }
    __aicore__ inline void Col2ImgAdd(const LocalTensor<float>& acc, const LocalTensor<float>& selected, uint64_t dst,
                                      uint64_t src, int64_t columns, int64_t paddedColumns)
    {
        const int64_t sw_ = STATIC_SW ? STATIC_SW : this->sw_;
        if constexpr (ROUND_PADDED_COLS) {
            const int64_t rounded = (columns + 7) / 8 * 8;
            // Extra lanes only touch this row's non-stored padding. All source lanes
            // remain inside the initialized input staging allocation, even at a row tail.
            if (rounded <= paddedColumns && src + rounded * NC <= INPUT_ELEMENTS)
                columns = rounded;
        }
        if (sw_ * NC <= 255 && columns >= 8 && (columns >= 32 || columns % 8 == 0)) {
            // Fill vector lanes with eight spatial positions, one 32B channel block
            // from each position. This preserves every pixel's summation order.
            const uint8_t full = static_cast<uint8_t>(columns / 8), tail = static_cast<uint8_t>(columns % 8);
            BinaryRepeatParams p{static_cast<uint8_t>(sw_ * NC / 8), static_cast<uint8_t>(sw_ * NC / 8),
                                 static_cast<uint8_t>(NC / 8),       static_cast<uint8_t>(sw_ * NC),
                                 static_cast<uint8_t>(sw_ * NC),     static_cast<uint8_t>(NC)};
            for (int64_t part = 0; part < NC / 8; ++part) {
                Add(acc[dst + part * 8], acc[dst + part * 8], selected[src + part * 8], static_cast<uint64_t>(64), full,
                    p);
                if (tail) {
                    const uint64_t d = dst + part * 8 + full * 8 * sw_ * NC, s = src + part * 8 + full * 8 * NC;
                    Add(acc[d], acc[d], selected[s], static_cast<uint64_t>(tail * 8), static_cast<uint8_t>(1), p);
                }
            }
        } else {
            Add(acc[dst], acc[dst], selected[src], static_cast<uint64_t>(NC), static_cast<uint8_t>(columns),
                {1, 1, 1, static_cast<uint8_t>(sw_ * (NC / 8)), static_cast<uint8_t>(sw_ * (NC / 8)),
                 static_cast<uint8_t>(NC / 8)});
        }
    }
    template <class U>
    __aicore__ inline void Input(const LocalTensor<U>& dst, const GlobalTensor<U>& src, int64_t offset, int64_t count,
                                 int64_t total, MaxPool310P::Copy32<U, true>& copier)
    {
        constexpr int64_t block = 32 / sizeof(U);
        // Rounded reads may include adjacent input elements but never leave the
        // logical input allocation. Padded columns are excluded by Col2Img bounds.
        const int64_t rounded = (count + block - 1) / block * block;
        if (offset + rounded <= total)
            DataCopy(dst, src[offset], static_cast<uint32_t>(rounded));
        else
            copier.In(dst, src, offset, count, total, Max(total, block), IC);
    }
    template <class U>
    __aicore__ inline void InputRows(const LocalTensor<U>& dst, const GlobalTensor<U>& src, int64_t offset,
                                     int64_t rows, int64_t width, int64_t pitch, int64_t total,
                                     MaxPool310P::Copy32<U, true>& copier)
    {
        constexpr int64_t block = 32 / sizeof(U);
        const int64_t rounded = (width + block - 1) / block * block;
        if (wo_ % block == 0 && rounded <= wo_ && (wo_ - rounded) / block <= 65535 &&
            offset + (rows - 1) * wo_ + rounded <= total) {
            DataCopyParams p{static_cast<uint16_t>(rows), static_cast<uint16_t>(rounded / block),
                             static_cast<uint16_t>((wo_ - rounded) / block),
                             static_cast<uint16_t>((pitch - rounded) / block)};
            DataCopy(dst, src[offset], p);
        } else {
            for (int64_t row = 0; row < rows; ++row)
                Input(dst[row * pitch], src, offset + row * wo_, width, total, copier);
        }
    }
    __aicore__ inline void Load(int64_t nc, int64_t channels, int64_t row, int64_t col, int64_t rows, int64_t width,
                                int64_t pitch)
    {
        if (outputPending_ && (outputTail_ || REUSE_PLANAR))
            WaitOut();
        auto gp = GetGP<float>();
        auto ip = GetIP<int32_t>();
        auto halfStage = GetGT<T>();
        if constexpr (sizeof(T) == 2)
            Duplicate(halfStage, static_cast<T>(0), NC * IC);
        else if constexpr (!INITIALIZE_ONCE)
            Duplicate(gp, 0.0f, NC * IC);
        if constexpr (!INITIALIZE_ONCE)
            Duplicate(ip, static_cast<int32_t>(0), NC * IC);
        Sync<HardEvent::V_MTE2>();
        int64_t total = nc_ * ho_ * wo_;
        for (int64_t c = 0; c < channels; ++c) {
            const int64_t offset = ((nc + c) * ho_ + row) * wo_ + col;
            InputRows(ip[c * IC], index_, offset, rows, width, pitch, total, icopy_);
            if constexpr (sizeof(T) == 2)
                InputRows(halfStage[c * IC], grad_, offset, rows, width, pitch, total, copy_);
            else
                InputRows(gp[c * IC], grad_, offset, rows, width, pitch, total, copy_);
        }
        Sync<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) == 2) {
            WaitOut();
            Cast(gp, halfStage, RoundMode::CAST_NONE, static_cast<int32_t>(NC * IC));
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Store(int64_t nc, int64_t channels, int64_t y, int64_t x, int64_t height, int64_t width,
                                 int64_t pitch, const LocalTensor<float>& acc)
    {
        WaitOut();
        auto planar = GetOP<float>();
        Transpose<float>(outD_, outS_, height * pitch, false);
        PipeBarrier<PIPE_V>();
        auto source = GetGP<T>();
        if constexpr (sizeof(T) == 2) {
            // The interleaved accumulator is dead after output transpose. Convert into
            // that region, disjoint from the shared planar source. Next tile waits for DMA.
            if constexpr (REUSE_PLANAR)
                source = ot_.Get<T>();
            Cast(source, planar, RoundMode::CAST_NONE, static_cast<int32_t>(NC * OC));
            PipeBarrier<PIPE_V>();
        } else {
            source = planar;
        }
        constexpr int64_t block = 32 / sizeof(T);
        using U = typename MaxPool310P::Copy32<T>::U;
        const int64_t main = width / block * block, tail = width % block;
        auto tails = GetIP<U>();
        if constexpr (REUSE_PLANAR) {
            if constexpr (sizeof(T) == 2)
                tails = shared_.Get<U>();
            else
                tails = ot_.Get<U>();
        }
        if (tail) {
            // Prepare every row tail on V first. All main DMA writes then all tail
            // writes share two block-level handshakes, not one handshake per row.
            auto bits = mask_.Get<uint32_t>();
            Duplicate(bits, static_cast<uint32_t>(((1ULL << block) - 1) << tail), 8);
            PipeBarrier<PIPE_V>();
            auto pattern = bits.template ReinterpretCast<U>();
            for (int64_t c = 0; c < channels; ++c)
                for (int64_t r = 0; r < height; ++r) {
                    uint64_t selected = 0;
                    GatherMask<U, U, GatherMaskMode::VERSION_V1>(
                        tails[(c * height + r) * block],
                        source[c * OC + r * pitch + main - block].template ReinterpretCast<U>(), pattern, true,
                        2 * block, GatherMaskParams{1, 1, 8, 0}, selected);
                }
            PipeBarrier<PIPE_V>();
        }
        Sync<HardEvent::V_MTE3>();
        if (tail == 0 && wi_ % block == 0 && (wi_ - width) / block <= 65535) {
            DataCopyParams params{static_cast<uint16_t>(height), static_cast<uint16_t>(width / block),
                                  static_cast<uint16_t>((pitch - width) / block),
                                  static_cast<uint16_t>((wi_ - width) / block)};
            for (int64_t c = 0; c < channels; ++c) {
                const int64_t offset = ((nc + c) * hi_ + y) * wi_ + x;
                DataCopy(out_[offset], source[c * OC], params);
            }
        } else {
            for (int64_t c = 0; c < channels; ++c)
                for (int64_t r = 0; r < height; ++r) {
                    const int64_t offset = ((nc + c) * hi_ + y + r) * wi_ + x;
                    DataCopy(out_[offset], source[c * OC + r * pitch], static_cast<uint32_t>(main));
                }
        }
        if (tail) {
            PipeBarrier<PIPE_MTE3>();
            for (int64_t c = 0; c < channels; ++c)
                for (int64_t r = 0; r < height; ++r) {
                    const int64_t offset = ((nc + c) * hi_ + y + r) * wi_ + x + width - block;
                    DataCopy(out_[offset], tails[(c * height + r) * block].template ReinterpretCast<T>(),
                             static_cast<uint32_t>(block));
                }
        }
        SetFlag<HardEvent::MTE3_V>(outputEvent_);
        outputPending_ = true;
        outputTail_ = tail != 0;
    }

    template <class U>
    __aicore__ inline void InitTransposeIn(uint64_t d[16], uint64_t s[16], const LocalTensor<U>& dst,
                                           const LocalTensor<U>& src)
    {
        for (int i = 0; i < 16; ++i) {
            if constexpr (NC == 16) {
                s[i] = (uint64_t)src[i * IC].GetPhyAddr();
                d[i] = (uint64_t)dst[(i / 2) * 16 + (i % 2) * 8].GetPhyAddr();
            } else {
                // Two adjacent groups of eight spatial positions share eight channels.
                s[i] = (uint64_t)src[(i % 8) * IC + (i / 8) * 8].GetPhyAddr();
                d[i] = (uint64_t)dst[(i / 2) * 8 + (i % 2) * 64].GetPhyAddr();
            }
        }
    }
    __aicore__ inline void InitTransposeOut(uint64_t d[16], uint64_t s[16], const LocalTensor<float>& dst,
                                            const LocalTensor<float>& src)
    {
        for (int i = 0; i < 16; ++i) {
            if constexpr (NC == 16) {
                s[i] = (uint64_t)src[(i % 8) * 16 + (i / 8) * 8].GetPhyAddr();
                d[i] = (uint64_t)dst[(i % 2 * 8 + i / 2) * OC].GetPhyAddr();
            } else {
                s[i] = (uint64_t)src[(i % 8) * 8 + (i / 8) * 64].GetPhyAddr();
                d[i] = (uint64_t)dst[(i / 2) * OC + (i % 2) * 8].GetPhyAddr();
            }
        }
    }
    template <class U>
    __aicore__ inline void Transpose(uint64_t d[16], uint64_t s[16], int64_t positions, bool input)
    {
        // Input rows may end on an 8-position boundary. Transpose one padded
        // 16-position group for C8; those extra positions are never accumulated.
        TransDataTo5HDParams p;
        p.repeatTimes = (positions + (128 / NC) - 1) / (128 / NC);
        p.srcRepStride = p.repeatTimes == 1 ? 0 : (input ? 16 / NC : 16);
        p.dstRepStride = p.repeatTimes == 1 ? 0 : (input ? 16 : 16 / NC);
        TransDataTo5HD<U>(d, s, p);
    }
    uint64_t gradD_[16], gradS_[16], idxD_[16], idxS_[16];
    uint64_t baseD_[16], baseS_[16], outD_[16], outS_[16];
    TEventID outputEvent_;
    bool outputPending_ = false, outputTail_ = false;
    TPipe* pipe_;
    GlobalTensor<T> grad_, out_;
    GlobalTensor<int32_t> index_;
    TBuf<TPosition::VECCALC> gp_, ip_, gt_, it_, ot_, op_, mask_, low_, shared_, sharedTails_;
    MaxPool310P::Copy32<T, true> copy_;
    MaxPool310P::Copy32<int32_t, true> icopy_;
    int64_t nc_, hi_, wi_, ho_, wo_, kh_, kw_, sh_, sw_, ph_, pw_, cuts_, hcuts_, tileH_, tileW_, tiles_;
};
} // namespace MaxPool2dMask310P
#endif
