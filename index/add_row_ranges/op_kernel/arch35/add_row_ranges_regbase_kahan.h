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
 * \file add_row_ranges_regbase_kahan.h
 * \brief Ascend950 (DAV_3510) RegBase Kahan kernel for add_row_ranges operator
 *
 * 方案 arr-regvec-kahan-v2：(行, lane 组) 任务 + 逐 lane、j 严格升序的 fp32 Kahan 链。
 *
 *   1. 任务并行：tasks = M * ceil(N / tileCols)，tileCols = numLaneGroups * 64；
 *      按 blockIdx 做 grid-stride，usedCoreNum = min(coreNum, tasks)。
 *   2. 每个任务覆盖 numLaneGroups ∈ {1,2} 个独立 64-fp32-lane 组，每组各有自己的
 *      acc/comp RegTensor 与 MaskReg；不同组的 lane 永不相加。
 *   3. 行范围 [start, end) 在标量侧每任务解算一次（-1 标记、负索引 +K、裁剪到
 *      [0, K]、空范围→原样拷贝），作为 VEC 循环上界；VEC 体内无逐元素整除。
 *   4. src 以 DataCopyPad 块搬：blockCount = min(blockRows, end-j)、
 *      blockLen = numLaneGroups*64*sizeof(T)、srcStride = (N-cols)*sizeof(T)，
 *      UB 行距 = numLaneGroups*256B；TQue 深度 2，chunk g+1 的 MTE2 在 chunk g
 *      计算前提交（软件预取），使 Kahan 链与 DMA 重叠。
 *   5. 每 lane 组只有一对 acc/comp，j 严格升序执行
 *      y=v-c; t=a+y; d=t-a; g=d-y; c=(isfinite(t)&&isfinite(a)) ? g : 0; a=t，
 *      与 golden `_reference_float32` 逐 op、逐顺序相同（逐位一致）。
 *      非有限复位用 Reg::Compares 两次 + mask-and + Reg::Select 无分支实现。
 *   6. __VEC_SCOPE__ 内只用 Reg::LoadAlign/Sub/Add/Abs/Compares/And/Select/Move/
 *      Cast/StoreAlign；不使用 asc_vf_call（其调用期间 MTE2 被阻塞）。
 *   7. epilogue：y = x + acc（comp 不参与，与 golden 一致），Cast 一次到输出 dtype
 *      (CAST_RINT) 后 StoreAlign + DataCopy；空范围行按 T 原样搬运 x（不经 Cast）。
 */

#ifndef ADD_ROW_RANGES_REGBASE_KAHAN_H_
#define ADD_ROW_RANGES_REGBASE_KAHAN_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_row_ranges_tiling_data.h"
#include "add_row_ranges_tiling_key.h"

namespace NsAddRowRangesRegBaseKahan {
using namespace AscendC;

constexpr uint32_t KAHAN_LANES = 64U;         // fp32 lanes per lane group (one 256B VL)
constexpr uint32_t KAHAN_UB_ROW_BYTES = 256U; // per lane group row stride in UB
constexpr uint32_t KAHAN_QUEUE_DEPTH = 2U;    // depth-2 double buffering of src chunks
constexpr uint32_t KAHAN_MAX_GROUPS = 2U;
// |x| <= FLT_MAX  <=>  x is finite (excludes +/-Inf and NaN without an operand branch).
constexpr float KAHAN_MAX_FINITE = 3.4028234663852886e+38F;

template <typename T>
class AddRowRangesKahanKernel {
public:
    __aicore__ inline AddRowRangesKahanKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR src, GM_ADDR indices, GM_ADDR y,
                                const AddRowRangesTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));

        n_ = tilingData->N;
        k_ = tilingData->K;
        groups_ = static_cast<uint32_t>(tilingData->numLaneGroups);
        groups_ = (groups_ == 0U) ? 1U : ((groups_ > KAHAN_MAX_GROUPS) ? KAHAN_MAX_GROUPS : groups_);
        tileCols_ = static_cast<uint32_t>(tilingData->tileCols);
        numColTiles_ = static_cast<uint32_t>(tilingData->numColTiles);
        blockRows_ = static_cast<uint32_t>(tilingData->blockRows);
        m_ = static_cast<uint32_t>(tilingData->M);
        rowsPerTask_ = static_cast<uint32_t>(tilingData->rowsPerTask);
        rowsPerTask_ = (rowsPerTask_ == 0U) ? 1U : rowsPerTask_;
        numRowBlocks_ = static_cast<uint32_t>(tilingData->numRowBlocks);
        tasks_ = numRowBlocks_ * numColTiles_;

        // x/y staging is row-block sized so an all-empty block copies R rows per DMA.
        pipe_->InitBuffer(srcQue_, KAHAN_QUEUE_DEPTH, blockRows_ * groups_ * KAHAN_UB_ROW_BYTES);
        pipe_->InitBuffer(xQue_, KAHAN_QUEUE_DEPTH, blockRows_ * groups_ * KAHAN_UB_ROW_BYTES);
        pipe_->InitBuffer(yQue_, KAHAN_QUEUE_DEPTH, blockRows_ * groups_ * KAHAN_UB_ROW_BYTES);
        pipe_->InitBuffer(stateBuf_, 2U * groups_ * KAHAN_UB_ROW_BYTES);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
        if (groups_ >= KAHAN_MAX_GROUPS) {
            for (uint32_t task = coreIdx; task < tasks_; task += coreNum) {
                ProcessTask<KAHAN_MAX_GROUPS>(task);
            }
        } else {
            for (uint32_t task = coreIdx; task < tasks_; task += coreNum) {
                ProcessTask<1U>(task);
            }
        }
    }

private:
    // numpy slice semantics identical to the baseline SIMT path; true => empty range.
    __aicore__ inline bool ResolveRange(uint32_t row, int64_t& start, int64_t& end) const
    {
        const int64_t base = static_cast<int64_t>(row) * 2;
        const int32_t rawStart = idxGm_.GetValue(base);
        const int32_t rawEnd = idxGm_.GetValue(base + 1);
        if (rawStart >= rawEnd || rawStart == -1 || rawEnd == -1) {
            return true;
        }
        int64_t s = static_cast<int64_t>(rawStart);
        int64_t e = static_cast<int64_t>(rawEnd);
        if (s < 0) {
            s += k_;
            s = (s < 0) ? 0 : s;
        }
        if (e < 0) {
            e += k_;
            e = (e < 0) ? 0 : e;
        }
        s = (s > k_) ? k_ : s;
        e = (e > k_) ? k_ : e;
        if (s >= e) {
            return true;
        }
        start = s;
        end = e;
        return false;
    }

    __aicore__ inline uint32_t RowsInChunk(int64_t rowIdx, int64_t end) const
    {
        const int64_t left = end - rowIdx;
        return (left > static_cast<int64_t>(blockRows_)) ? blockRows_ : static_cast<uint32_t>(left);
    }

    // Empty rows: copy x verbatim in T (no cast, no accumulation). Consecutive empty rows
    // of one task are staged together so a row block costs one DMA pair instead of one per
    // row. MTE2 -> VEC (UB-to-UB copy) -> MTE3 keeps the standard queue ordering.
    __aicore__ inline void CopyRowsVerbatim(uint32_t row0, uint32_t nRows, uint32_t col0, uint32_t cols)
    {
        const int64_t gmOffset = static_cast<int64_t>(row0) * n_ + col0;
        const uint32_t usedBytes = cols * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBytes = (usedBytes + 31U) / 32U * 32U;
        const uint32_t rowBytes = groups_ * KAHAN_UB_ROW_BYTES;
        const int64_t gmStride = (n_ - static_cast<int64_t>(cols)) * static_cast<int64_t>(sizeof(T));
        const int64_t ubStride = static_cast<int64_t>((rowBytes - paddedBytes) / 32U);
        LocalTensor<T> inLocal = xQue_.template AllocTensor<T>();
        DataCopyExtParams inParams{static_cast<uint16_t>(nRows), usedBytes, gmStride, ubStride, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(inLocal, xGm_[gmOffset], inParams, padParams);
        xQue_.EnQue(inLocal);
        LocalTensor<T> in = xQue_.template DeQue<T>();
        LocalTensor<T> outLocal = yQue_.template AllocTensor<T>();
        DataCopy(outLocal, in, static_cast<int32_t>(nRows * rowBytes / sizeof(T)));
        xQue_.FreeTensor(in);
        yQue_.EnQue(outLocal);
        LocalTensor<T> out = yQue_.template DeQue<T>();
        DataCopyExtParams outParams{static_cast<uint16_t>(nRows), usedBytes, ubStride, gmStride, 0};
        DataCopyPad(yGm_[gmOffset], out, outParams);
        yQue_.FreeTensor(out);
    }

    __aicore__ inline void CopyInChunk(const LocalTensor<T>& dst, int64_t srcRow0, uint32_t col0, uint32_t cols,
                                       uint32_t nRows)
    {
        const int64_t gmOffset = srcRow0 * n_ + static_cast<int64_t>(col0);
        const uint32_t usedBytes = cols * static_cast<uint32_t>(sizeof(T));
        const uint32_t paddedBytes = (usedBytes + 31U) / 32U * 32U;
        const uint32_t rowBytes = groups_ * KAHAN_UB_ROW_BYTES;
        DataCopyExtParams params{static_cast<uint16_t>(nRows), static_cast<uint32_t>(usedBytes),
                                 static_cast<int64_t>((n_ - static_cast<int64_t>(cols)) * sizeof(T)),
                                 static_cast<int64_t>((rowBytes - paddedBytes) / 32U), 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(dst, srcGm_[gmOffset], params, padParams);
    }

    // Load 64 source elements of one lane group as fp32 registers.
    __aicore__ inline void LoadValue(Reg::RegTensor<float>& dst, __ubuf__ T* addr, Reg::RegTensor<T>& packed,
                                     Reg::MaskReg& mask)
    {
        if constexpr (sizeof(T) == sizeof(float)) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dst, reinterpret_cast<__ubuf__ float*>(addr));
        } else {
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packed, addr);
            Reg::Cast<float, T, kTToF32CastTrait>(dst, packed, mask);
        }
    }

    // One Kahan tap: y=v-c; t=a+y; d=t-a; g=d-y; c=finite(t)&finite(a)?g:0; a=t.
    // absPrev carries |a| between taps so both Compares are issued per tap.
    __aicore__ inline void KahanStep(Reg::RegTensor<float>& acc, Reg::RegTensor<float>& comp,
                                     Reg::RegTensor<float>& absPrev, Reg::RegTensor<float>& v,
                                     Reg::RegTensor<float>& zero, Reg::MaskReg& mask)
    {
        Reg::RegTensor<float> y;
        Reg::RegTensor<float> t;
        Reg::RegTensor<float> dg;
        Reg::MaskReg finiteT;
        Reg::MaskReg finiteA;
        Reg::MaskReg finite;
        Reg::Sub(y, v, comp, mask);
        Reg::Add(t, acc, y, mask);
        Reg::Sub(dg, t, acc, mask);
        Reg::Sub(dg, dg, y, mask);
        Reg::Compares<float, CMPMODE::LE>(finiteA, absPrev, KAHAN_MAX_FINITE, mask);
        Reg::Abs(absPrev, t, mask);
        Reg::Compares<float, CMPMODE::LE>(finiteT, absPrev, KAHAN_MAX_FINITE, mask);
        Reg::And(finite, finiteT, finiteA, mask);
        Reg::Select<float>(comp, dg, zero, finite);
        Reg::Move(acc, t);
    }

    template <uint32_t G>
    __aicore__ inline void ComputeChunk(__ubuf__ T* chunkAddr, uint32_t cols, uint32_t nRows)
    {
        const uint32_t rowElems = G * KAHAN_UB_ROW_BYTES / static_cast<uint32_t>(sizeof(T));
        __ubuf__ float* stateAddr = reinterpret_cast<__ubuf__ float*>(stateBuf_.Get<float>().GetPhyAddr());
        __VEC_SCOPE__
        {
            Reg::MaskReg mask0;
            uint32_t remaining0 = cols;
            mask0 = Reg::UpdateMask<float>(remaining0);
            Reg::MaskReg mask1;
            uint32_t remaining1 = (cols > KAHAN_LANES) ? (cols - KAHAN_LANES) : 0U;
            if constexpr (G == KAHAN_MAX_GROUPS) {
                mask1 = Reg::UpdateMask<float>(remaining1);
            }

            Reg::RegTensor<float> zero;
            Reg::Duplicate(zero, 0.0f, mask0);

            Reg::RegTensor<float> acc0;
            Reg::RegTensor<float> comp0;
            Reg::RegTensor<float> abs0;
            Reg::RegTensor<T> packed0;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(acc0, stateAddr);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(comp0, stateAddr + 1U * KAHAN_LANES);
            Reg::Abs(abs0, acc0, mask0);

            Reg::RegTensor<float> acc1;
            Reg::RegTensor<float> comp1;
            Reg::RegTensor<float> abs1;
            Reg::RegTensor<T> packed1;
            if constexpr (G == KAHAN_MAX_GROUPS) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(acc1, stateAddr + 2U * KAHAN_LANES);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(comp1, stateAddr + 3U * KAHAN_LANES);
                Reg::Abs(abs1, acc1, mask1);
            }

            for (uint16_t i = 0; i < static_cast<uint16_t>(nRows); ++i) {
                const uint32_t base = static_cast<uint32_t>(i) * rowElems;
                Reg::RegTensor<float> v0;
                LoadValue(v0, chunkAddr + base, packed0, mask0);
                KahanStep(acc0, comp0, abs0, v0, zero, mask0);
                if constexpr (G == KAHAN_MAX_GROUPS) {
                    Reg::RegTensor<float> v1;
                    LoadValue(v1, chunkAddr + base + KAHAN_LANES, packed1, mask1);
                    KahanStep(acc1, comp1, abs1, v1, zero, mask1);
                }
            }

            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stateAddr, acc0, mask0);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stateAddr + 1U * KAHAN_LANES, comp0, mask0);
            if constexpr (G == KAHAN_MAX_GROUPS) {
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stateAddr + 2U * KAHAN_LANES, acc1, mask1);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stateAddr + 3U * KAHAN_LANES, comp1, mask1);
            }
        }
    }

    template <uint32_t G>
    __aicore__ inline void Epilogue(uint32_t row, uint32_t col0, uint32_t cols)
    {
        const int64_t gmOffset = static_cast<int64_t>(row) * n_ + col0;
        LocalTensor<T> xLocal = xQue_.template AllocTensor<T>();
        DataCopyExtParams xParams{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> xPadParams{true, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[gmOffset], xParams, xPadParams);
        xQue_.EnQue(xLocal);
        LocalTensor<T> xIn = xQue_.template DeQue<T>();
        __ubuf__ T* xAddr = reinterpret_cast<__ubuf__ T*>(xIn.GetPhyAddr());

        LocalTensor<T> yLocal = yQue_.template AllocTensor<T>();
        __ubuf__ T* yAddr = reinterpret_cast<__ubuf__ T*>(yLocal.GetPhyAddr());
        __ubuf__ float* stateAddr = reinterpret_cast<__ubuf__ float*>(stateBuf_.Get<float>().GetPhyAddr());
        __VEC_SCOPE__
        {
            Reg::MaskReg mask0;
            uint32_t remaining0 = cols;
            mask0 = Reg::UpdateMask<float>(remaining0);
            Reg::MaskReg mask1;
            uint32_t remaining1 = (cols > KAHAN_LANES) ? (cols - KAHAN_LANES) : 0U;
            if constexpr (G == KAHAN_MAX_GROUPS) {
                mask1 = Reg::UpdateMask<float>(remaining1);
            }

            Reg::RegTensor<float> acc0;
            Reg::RegTensor<float> xReg0;
            Reg::RegTensor<T> packed0;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(acc0, stateAddr);
            LoadValue(xReg0, xAddr, packed0, mask0);
            Reg::Add(acc0, xReg0, acc0, mask0);
            if constexpr (sizeof(T) == sizeof(float)) {
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(reinterpret_cast<__ubuf__ float*>(yAddr), acc0,
                                                                  mask0);
            } else {
                Reg::Cast<T, float, kF32ToTCastTrait>(packed0, acc0, mask0);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr, packed0, mask0);
            }

            if constexpr (G == KAHAN_MAX_GROUPS) {
                Reg::RegTensor<float> acc1;
                Reg::RegTensor<float> xReg1;
                Reg::RegTensor<T> packed1;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(acc1, stateAddr + 2U * KAHAN_LANES);
                LoadValue(xReg1, xAddr + KAHAN_LANES, packed1, mask1);
                Reg::Add(acc1, xReg1, acc1, mask1);
                if constexpr (sizeof(T) == sizeof(float)) {
                    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
                        reinterpret_cast<__ubuf__ float*>(yAddr) + KAHAN_LANES, acc1, mask1);
                } else {
                    Reg::Cast<T, float, kF32ToTCastTrait>(packed1, acc1, mask1);
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(yAddr + KAHAN_LANES, packed1, mask1);
                }
            }
        }
        xQue_.FreeTensor(xIn);
        yQue_.EnQue(yLocal);
        LocalTensor<T> yOut = yQue_.template DeQue<T>();
        DataCopyExtParams yParams{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[gmOffset], yOut, yParams);
        yQue_.FreeTensor(yOut);
    }

    template <uint32_t G>
    __aicore__ inline void ComputeRow(uint32_t row, uint32_t col0, uint32_t cols, int64_t start, int64_t end)
    {
        LocalTensor<float> state = stateBuf_.Get<float>();
        Duplicate<float>(state, 0.0f, static_cast<int32_t>(2U * G * KAHAN_LANES));

        int64_t rowIdx = start;
        LocalTensor<T> pending = srcQue_.template AllocTensor<T>();
        CopyInChunk(pending, rowIdx, col0, cols, RowsInChunk(rowIdx, end));
        srcQue_.EnQue(pending);
        while (rowIdx < end) {
            const uint32_t nRows = RowsInChunk(rowIdx, end);
            const int64_t nextIdx = rowIdx + static_cast<int64_t>(nRows);
            if (nextIdx < end) {
                LocalTensor<T> ahead = srcQue_.template AllocTensor<T>();
                CopyInChunk(ahead, nextIdx, col0, cols, RowsInChunk(nextIdx, end));
                srcQue_.EnQue(ahead);
            }
            LocalTensor<T> chunk = srcQue_.template DeQue<T>();
            __ubuf__ T* chunkAddr = reinterpret_cast<__ubuf__ T*>(chunk.GetPhyAddr());
            ComputeChunk<G>(chunkAddr, cols, nRows);
            srcQue_.FreeTensor(chunk);
            rowIdx = nextIdx;
        }

        Epilogue<G>(row, col0, cols);
    }

    template <uint32_t G>
    __aicore__ inline void ProcessTask(uint32_t task)
    {
        const uint32_t rowBlock = task / numColTiles_;
        const uint32_t colTile = task - rowBlock * numColTiles_;
        const uint32_t col0 = colTile * tileCols_;
        const int64_t rest = static_cast<int64_t>(n_) - static_cast<int64_t>(col0);
        const uint32_t cols = (rest > static_cast<int64_t>(tileCols_)) ? tileCols_ : static_cast<uint32_t>(rest);
        if (cols == 0U) {
            return;
        }

        const uint32_t rowBegin = rowBlock * rowsPerTask_;
        uint32_t rowEnd = rowBegin + rowsPerTask_;
        if (rowEnd > m_) {
            rowEnd = m_;
        }

        uint32_t emptyStart = 0U;
        uint32_t emptyRows = 0U;
        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            int64_t start = 0;
            int64_t end = 0;
            if (ResolveRange(row, start, end)) {
                if (emptyRows == 0U) {
                    emptyStart = row;
                }
                ++emptyRows;
                continue;
            }
            if (emptyRows != 0U) {
                CopyRowsVerbatim(emptyStart, emptyRows, col0, cols);
                emptyRows = 0U;
            }
            ComputeRow<G>(row, col0, cols, start, end);
        }
        if (emptyRows != 0U) {
            CopyRowsVerbatim(emptyStart, emptyRows, col0, cols);
        }
    }

    static constexpr Reg::CastTrait kF32ToTCastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait kTToF32CastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

    TPipe* pipe_ = nullptr;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> srcGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> idxGm_;
    TQue<QuePosition::VECIN, KAHAN_QUEUE_DEPTH> srcQue_;
    TQue<QuePosition::VECIN, KAHAN_QUEUE_DEPTH> xQue_;
    TQue<QuePosition::VECOUT, KAHAN_QUEUE_DEPTH> yQue_;
    TBuf<QuePosition::VECCALC> stateBuf_;
    int64_t n_ = 0;
    int64_t k_ = 0;
    uint32_t m_ = 0U;
    uint32_t rowsPerTask_ = 1U;
    uint32_t numRowBlocks_ = 0U;
    uint32_t groups_ = 1U;
    uint32_t tileCols_ = KAHAN_LANES;
    uint32_t numColTiles_ = 1U;
    uint32_t blockRows_ = 16U;
    uint32_t tasks_ = 0U;
};

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR src, GM_ADDR indices, GM_ADDR y,
                               const AddRowRangesTilingData* tilingData)
{
    if (tilingData->totalElements <= 0 || tilingData->N <= 0) {
        return;
    }
    TPipe pipe;
    AddRowRangesKahanKernel<T> kernel;
    kernel.Init(x, src, indices, y, tilingData, &pipe);
    kernel.Process();
}

} // namespace NsAddRowRangesRegBaseKahan

#endif // ADD_ROW_RANGES_REGBASE_KAHAN_H_
