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
 * \file scatter_elements_v2_bucket_scatter.h
 * \brief ScatterElementsV2 的分桶散射实现（末轴 reduction=none 的稀疏大 var 场景）。
 *
 *  语义与 ScatterElementsV2 主路径一致：var[r, indices[r,i]] = updates[r,i]，
 *  未被 indices 命中的位置保持 var 原值；同一位置发生碰撞时按 i 升序覆盖（末次写赢）。
 *
 *  适用场景：末轴长度 varN 极大而每行更新数 indicesN 远小于它（varN >> indicesN）。
 *  此时主路径需按 varN 分多轮搬运整行，而本实现复杂度只与 indicesN 相关：
 *
 *    Pass1 直方图：扫 k 个 idx，count[idx >> shift]++（标量 4 连读）。
 *    Pass2 前缀和：计算每桶在 GM 桶区的 offset/cursor（每桶起点对齐 16，尾部留 16 guard，
 *                  避免 DataCopyPad 的 32B 向上取整写越界到邻桶）。
 *    Pass3 放置：扫 k 个 (idx,val)，按 i 序写入 UB 每桶 FIFO；FIFO 满 fifoDepth 个即突发
 *                DataCopy 到 GM 桶区，把随机散射写合并成顺序突发写。桶内保持 i 序。
 *    Pass4 逐 tile：读入 var 的该 tile 作底图 → 流式读本桶的 cnt 个更新 → 按 i 序 SetValue
 *                   覆盖（末次写赢）→ 单次写回。
 */
#ifndef SCATTER_ELEMENTS_V2_BUCKET_SCATTER_H
#define SCATTER_ELEMENTS_V2_BUCKET_SCATTER_H

#include "kernel_operator.h"

namespace ScatterElementsV2NS {

using namespace AscendC;

constexpr uint32_t BUCKET_UPD_CHUNK = 8192; // 流式块元素数（pass1/3 载更新、pass4 载桶）
constexpr int32_t BUCKET_ALIGN = 16;        // 每桶起点/容量对齐粒度

// T = var/updates 数据类型，U = indices 数据类型
template <typename T, typename U>
class BucketScatterElements {
public:
    __aicore__ inline BucketScatterElements() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                const ScatterElementsV2TilingData* tilingData, TPipe* pipe, GM_ADDR userWorkspace)
    {
        pipe_ = pipe;
        rows_ = static_cast<int64_t>(tilingData->bktRows);       // 独立散射行数
        m_ = static_cast<int64_t>(tilingData->bktVarN);          // 每行 var 元素数
        k_ = static_cast<int64_t>(tilingData->bktIndicesN);      // 每行更新元素数
        tileLen_ = static_cast<int64_t>(tilingData->bktTileLen); // 单个 tile 元素数（2 的幂）
        numTiles_ = static_cast<int64_t>(tilingData->bktNumTiles);
        shift_ = static_cast<int64_t>(tilingData->bktShift); // log2(tileLen)
        fifoDepth_ = static_cast<int64_t>(tilingData->bktFifoDepth);
        bktStride_ = static_cast<int64_t>(tilingData->bktStride);
        int64_t usedCore = static_cast<int64_t>(tilingData->usedCoreNum);

        // 行到核的划分：前 bktFrontCore 个核各多处理 1 行
        int64_t blk = GetBlockIdx();
        int64_t base = static_cast<int64_t>(tilingData->bktRowsPerCore);
        int64_t fc = static_cast<int64_t>(tilingData->bktFrontCore);
        if (blk < fc) {
            rowStart_ = blk * (base + 1);
            rowEnd_ = rowStart_ + (base + 1);
        } else {
            rowStart_ = fc * (base + 1) + (blk - fc) * base;
            rowEnd_ = rowStart_ + base;
        }
        if (rowEnd_ > rows_) {
            rowEnd_ = rows_;
        }

        varGm_.SetGlobalBuffer((__gm__ T*)var);
        idxGm_.SetGlobalBuffer((__gm__ U*)indices);
        updGm_.SetGlobalBuffer((__gm__ T*)updates);

        // GM 桶区：bktIdx[usedCore*stride] 紧接 bktVal[usedCore*stride]，各核独占一段
        __gm__ uint8_t* ws = userWorkspace;
        __gm__ int32_t* bktIdxBase = (__gm__ int32_t*)ws;
        __gm__ T* bktValBase = (__gm__ T*)(ws + static_cast<uint64_t>(usedCore) * bktStride_ * sizeof(int32_t));
        bktIdxGm_.SetGlobalBuffer(bktIdxBase + static_cast<uint64_t>(blk) * bktStride_, bktStride_);
        bktValGm_.SetGlobalBuffer(bktValBase + static_cast<uint64_t>(blk) * bktStride_, bktStride_);

        pipe_->InitBuffer(metaBuf_, 4 * numTiles_ * sizeof(int32_t)); // count/offset/cursor/fifoCnt
        // bigBuf 复用于 pass3 与 pass4，取两者所需的较大值
        uint32_t pass4 = static_cast<uint32_t>(tileLen_) * sizeof(T) + BUCKET_UPD_CHUNK * sizeof(int32_t) +
                         BUCKET_UPD_CHUNK * sizeof(T);
        uint32_t pass3 = BUCKET_UPD_CHUNK * sizeof(U) + BUCKET_UPD_CHUNK * sizeof(T) +
                         static_cast<uint32_t>(numTiles_) * fifoDepth_ * sizeof(int32_t) +
                         static_cast<uint32_t>(numTiles_) * fifoDepth_ * sizeof(T);
        bigBytes_ = pass4 > pass3 ? pass4 : pass3;
        pipe_->InitBuffer(bigBuf_, bigBytes_);
    }

    __aicore__ inline void Process()
    {
        if (rowStart_ >= rowEnd_) {
            return;
        }
        LocalTensor<int32_t> meta = metaBuf_.Get<int32_t>();
        LocalTensor<int32_t> count = meta;
        LocalTensor<int32_t> offset = meta[numTiles_];
        LocalTensor<int32_t> cursor = meta[2 * numTiles_];
        LocalTensor<int32_t> fcnt = meta[3 * numTiles_];

        eM2S_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        eSM2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        eSM3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        eM3S_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        eM3M2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

        for (int64_t r = rowStart_; r < rowEnd_; ++r) {
            Histogram(count, r);
            PrefixSum(count, offset, cursor, fcnt);
            Place(offset, cursor, fcnt, r);
            ScatterTiles(count, offset, r);
        }
    }

private:
    // Pass1：按 idx>>shift 统计每个 tile 落入的更新条数
    __aicore__ inline void Histogram(LocalTensor<int32_t>& count, int64_t r)
    {
        for (int64_t b = 0; b < numTiles_; ++b) {
            count.SetValue(b, 0);
        }
        LocalTensor<U> idxL = bigBuf_.GetWithOffset<U>(BUCKET_UPD_CHUNK, 0);
        int64_t base = r * k_;
        bool first = true;
        for (int64_t off = 0; off < k_; off += BUCKET_UPD_CHUNK) {
            uint32_t cn = (k_ - off < static_cast<int64_t>(BUCKET_UPD_CHUNK)) ? static_cast<uint32_t>(k_ - off) :
                                                                                BUCKET_UPD_CHUNK;
            if (!first) {
                WaitFlag<HardEvent::S_MTE2>(eSM2_);
            }
            DataCopyExtParams cp{1, cn * static_cast<uint32_t>(sizeof(U)), 0, 0, 0};
            DataCopyPadExtParams<U> dp{false, 0, 0, 0};
            DataCopyPad(idxL, idxGm_[base + off], cp, dp);
            SetFlag<HardEvent::MTE2_S>(eM2S_);
            WaitFlag<HardEvent::MTE2_S>(eM2S_);
            uint32_t c4 = cn & ~3u;
            uint32_t j = 0;
            for (; j < c4; j += 4) {
                int64_t b0 = static_cast<int64_t>(idxL.GetValue(j)) >> shift_;
                int64_t b1 = static_cast<int64_t>(idxL.GetValue(j + 1)) >> shift_;
                int64_t b2 = static_cast<int64_t>(idxL.GetValue(j + 2)) >> shift_;
                int64_t b3 = static_cast<int64_t>(idxL.GetValue(j + 3)) >> shift_;
                count.SetValue(b0, count.GetValue(b0) + 1);
                count.SetValue(b1, count.GetValue(b1) + 1);
                count.SetValue(b2, count.GetValue(b2) + 1);
                count.SetValue(b3, count.GetValue(b3) + 1);
            }
            for (; j < cn; ++j) {
                int64_t b = static_cast<int64_t>(idxL.GetValue(j)) >> shift_;
                count.SetValue(b, count.GetValue(b) + 1);
            }
            SetFlag<HardEvent::S_MTE2>(eSM2_);
            first = false;
        }
        WaitFlag<HardEvent::S_MTE2>(eSM2_);
    }

    // Pass2：前缀和定位每桶在 GM 桶区的起点（对齐 16 + 16 guard，防 DataCopyPad 尾部越界写邻桶）
    __aicore__ inline void PrefixSum(LocalTensor<int32_t>& count, LocalTensor<int32_t>& offset,
                                     LocalTensor<int32_t>& cursor, LocalTensor<int32_t>& fcnt)
    {
        int32_t acc = 0;
        for (int64_t b = 0; b < numTiles_; ++b) {
            offset.SetValue(b, acc);
            cursor.SetValue(b, acc);
            fcnt.SetValue(b, 0);
            int32_t cnt = count.GetValue(b);
            int32_t padded = ((cnt + BUCKET_ALIGN - 1) / BUCKET_ALIGN) * BUCKET_ALIGN + BUCKET_ALIGN;
            acc += padded;
        }
    }

    // Pass3：按 i 序把 (tile 内偏移, 值) 放入各桶 UB FIFO，满则突发写入 GM 桶区
    __aicore__ inline void Place(LocalTensor<int32_t>& offset, LocalTensor<int32_t>& cursor, LocalTensor<int32_t>& fcnt,
                                 int64_t r)
    {
        LocalTensor<U> idxL = bigBuf_.GetWithOffset<U>(BUCKET_UPD_CHUNK, 0);
        LocalTensor<T> valL = bigBuf_.GetWithOffset<T>(BUCKET_UPD_CHUNK, BUCKET_UPD_CHUNK * sizeof(U));
        uint32_t fifoBase = BUCKET_UPD_CHUNK * sizeof(U) + BUCKET_UPD_CHUNK * sizeof(T);
        LocalTensor<int32_t> fIdx = bigBuf_.GetWithOffset<int32_t>(numTiles_ * fifoDepth_, fifoBase);
        LocalTensor<T> fVal = bigBuf_.GetWithOffset<T>(
            numTiles_ * fifoDepth_, fifoBase + static_cast<uint32_t>(numTiles_) * fifoDepth_ * sizeof(int32_t));
        int64_t base = r * k_;
        bool first = true;
        for (int64_t off = 0; off < k_; off += BUCKET_UPD_CHUNK) {
            uint32_t cn = (k_ - off < static_cast<int64_t>(BUCKET_UPD_CHUNK)) ? static_cast<uint32_t>(k_ - off) :
                                                                                BUCKET_UPD_CHUNK;
            if (!first) {
                WaitFlag<HardEvent::S_MTE2>(eSM2_);
            }
            DataCopyExtParams cpI{1, cn * static_cast<uint32_t>(sizeof(U)), 0, 0, 0};
            DataCopyExtParams cpV{1, cn * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<U> dpI{false, 0, 0, 0};
            DataCopyPadExtParams<T> dpV{false, 0, 0, 0};
            DataCopyPad(idxL, idxGm_[base + off], cpI, dpI);
            DataCopyPad(valL, updGm_[base + off], cpV, dpV);
            SetFlag<HardEvent::MTE2_S>(eM2S_);
            WaitFlag<HardEvent::MTE2_S>(eM2S_);
            for (uint32_t j = 0; j < cn; ++j) {
                int64_t ix = static_cast<int64_t>(idxL.GetValue(j));
                int64_t b = ix >> shift_;
                int32_t p = fcnt.GetValue(b);
                fIdx.SetValue(b * fifoDepth_ + p, static_cast<int32_t>(ix - (b << shift_)));
                fVal.SetValue(b * fifoDepth_ + p, valL.GetValue(j));
                p += 1;
                if (p == fifoDepth_) {
                    int32_t cur = cursor.GetValue(b);
                    SetFlag<HardEvent::S_MTE3>(eSM3_);
                    WaitFlag<HardEvent::S_MTE3>(eSM3_);
                    DataCopyExtParams fcI{1, static_cast<uint32_t>(fifoDepth_) * static_cast<uint32_t>(sizeof(int32_t)),
                                          0, 0, 0};
                    DataCopyExtParams fcV{1, static_cast<uint32_t>(fifoDepth_) * static_cast<uint32_t>(sizeof(T)), 0, 0,
                                          0};
                    DataCopyPad(bktIdxGm_[cur], fIdx[b * fifoDepth_], fcI);
                    DataCopyPad(bktValGm_[cur], fVal[b * fifoDepth_], fcV);
                    SetFlag<HardEvent::MTE3_S>(eM3S_);
                    WaitFlag<HardEvent::MTE3_S>(eM3S_);
                    cursor.SetValue(b, cur + fifoDepth_);
                    p = 0;
                }
                fcnt.SetValue(b, p);
            }
            SetFlag<HardEvent::S_MTE2>(eSM2_);
            first = false;
        }
        WaitFlag<HardEvent::S_MTE2>(eSM2_);
        // flush 各桶 FIFO 残余
        for (int64_t b = 0; b < numTiles_; ++b) {
            int32_t p = fcnt.GetValue(b);
            if (p > 0) {
                int32_t cur = cursor.GetValue(b);
                SetFlag<HardEvent::S_MTE3>(eSM3_);
                WaitFlag<HardEvent::S_MTE3>(eSM3_);
                DataCopyExtParams fcI{1, static_cast<uint32_t>(p) * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                DataCopyExtParams fcV{1, static_cast<uint32_t>(p) * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyPad(bktIdxGm_[cur], fIdx[b * fifoDepth_], fcI);
                DataCopyPad(bktValGm_[cur], fVal[b * fifoDepth_], fcV);
                SetFlag<HardEvent::MTE3_S>(eM3S_);
                WaitFlag<HardEvent::MTE3_S>(eM3S_);
                cursor.SetValue(b, cur + p);
            }
        }
    }

    // Pass4：逐 tile 读入 var 原值作底图 → 按 i 序覆盖本桶更新（末次写赢）→ 写回
    __aicore__ inline void ScatterTiles(LocalTensor<int32_t>& count, LocalTensor<int32_t>& offset, int64_t r)
    {
        LocalTensor<T> tile = bigBuf_.GetWithOffset<T>(tileLen_, 0);
        LocalTensor<int32_t> bIdx = bigBuf_.GetWithOffset<int32_t>(BUCKET_UPD_CHUNK,
                                                                   static_cast<uint32_t>(tileLen_) * sizeof(T));
        LocalTensor<T> bVal = bigBuf_.GetWithOffset<T>(
            BUCKET_UPD_CHUNK, static_cast<uint32_t>(tileLen_) * sizeof(T) + BUCKET_UPD_CHUNK * sizeof(int32_t));
        int64_t outBase = r * m_;
        bool firstTile = true;
        for (int64_t t = 0; t < numTiles_; ++t) {
            int64_t tileBegin = t * tileLen_;
            int64_t tileN = (m_ - tileBegin < tileLen_) ? (m_ - tileBegin) : tileLen_;
            int32_t cntB = count.GetValue(t);
            int32_t offB = offset.GetValue(t);

            // 该 tile 无任何更新落入时，var 的这一段保持原值不变，无需读回再原样写出：
            // 直接跳过可省下 2 段 GM 流量。稀疏度越高（更新数远小于桶数）收益越大。
            if (cntB == 0) {
                continue;
            }

            // 读入 var 该 tile 作为底图：未被 indices 命中的位置由此保持原值，与主路径语义一致。
            // tile 缓冲跨轮复用：需等上一轮写出(MTE3)完成后才能再次载入(MTE2)，故用 MTE3_MTE2。
            if (!firstTile) {
                WaitFlag<HardEvent::MTE3_MTE2>(eM3M2_);
            }
            DataCopyExtParams cpIn{1, static_cast<uint32_t>(tileN) * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> dpIn{false, 0, 0, 0};
            DataCopyPad(tile, varGm_[outBase + tileBegin], cpIn, dpIn);
            // 载入后由标量 SetValue 覆盖命中位置，故等待 MTE2→S
            SetFlag<HardEvent::MTE2_S>(eM2S_);
            WaitFlag<HardEvent::MTE2_S>(eM2S_);

            bool firstC = true;
            for (int32_t bo = 0; bo < cntB; bo += BUCKET_UPD_CHUNK) {
                uint32_t bn = (cntB - bo < static_cast<int32_t>(BUCKET_UPD_CHUNK)) ? static_cast<uint32_t>(cntB - bo) :
                                                                                     BUCKET_UPD_CHUNK;
                if (!firstC) {
                    WaitFlag<HardEvent::S_MTE2>(eSM2_);
                }
                DataCopyExtParams cpI{1, bn * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
                DataCopyExtParams cpV{1, bn * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> dpI{false, 0, 0, 0};
                DataCopyPadExtParams<T> dpV{false, 0, 0, 0};
                DataCopyPad(bIdx, bktIdxGm_[offB + bo], cpI, dpI);
                DataCopyPad(bVal, bktValGm_[offB + bo], cpV, dpV);
                SetFlag<HardEvent::MTE2_S>(eM2S_);
                WaitFlag<HardEvent::MTE2_S>(eM2S_);
                uint32_t c4 = bn & ~3u;
                uint32_t j = 0;
                for (; j < c4; j += 4) {
                    tile.SetValue(static_cast<uint32_t>(bIdx.GetValue(j)), bVal.GetValue(j));
                    tile.SetValue(static_cast<uint32_t>(bIdx.GetValue(j + 1)), bVal.GetValue(j + 1));
                    tile.SetValue(static_cast<uint32_t>(bIdx.GetValue(j + 2)), bVal.GetValue(j + 2));
                    tile.SetValue(static_cast<uint32_t>(bIdx.GetValue(j + 3)), bVal.GetValue(j + 3));
                }
                for (; j < bn; ++j) {
                    tile.SetValue(static_cast<uint32_t>(bIdx.GetValue(j)), bVal.GetValue(j));
                }
                SetFlag<HardEvent::S_MTE2>(eSM2_);
                firstC = false;
            }
            if (!firstC) {
                WaitFlag<HardEvent::S_MTE2>(eSM2_);
            }

            SetFlag<HardEvent::S_MTE3>(eSM3_);
            WaitFlag<HardEvent::S_MTE3>(eSM3_);
            DataCopyExtParams cpO{1, static_cast<uint32_t>(tileN) * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(varGm_[outBase + tileBegin], tile, cpO);
            SetFlag<HardEvent::MTE3_MTE2>(eM3M2_);
            firstTile = false;
        }
        // 仅当至少处理过一个 tile（即发出过 SetFlag）时才等待，避免全跳过时等待未置位的事件
        if (!firstTile) {
            WaitFlag<HardEvent::MTE3_MTE2>(eM3M2_);
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> metaBuf_;
    TBuf<TPosition::VECCALC> bigBuf_;
    GlobalTensor<T> varGm_;
    GlobalTensor<U> idxGm_;
    GlobalTensor<T> updGm_;
    GlobalTensor<int32_t> bktIdxGm_;
    GlobalTensor<T> bktValGm_;
    int64_t rows_ = 0;
    int64_t m_ = 0;
    int64_t k_ = 0;
    int64_t tileLen_ = 0;
    int64_t numTiles_ = 0;
    int64_t shift_ = 0;
    int64_t fifoDepth_ = 0;
    int64_t bktStride_ = 0;
    int64_t rowStart_ = 0;
    int64_t rowEnd_ = 0;
    uint32_t bigBytes_ = 0;
    int32_t eM2S_ = 0;
    int32_t eSM2_ = 0;
    int32_t eSM3_ = 0;
    int32_t eM3S_ = 0;
    int32_t eM3M2_ = 0;
};

} // namespace ScatterElementsV2NS
#endif // SCATTER_ELEMENTS_V2_BUCKET_SCATTER_H
