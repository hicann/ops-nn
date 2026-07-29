/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEDIAN_KERNEL_H
#define MEDIAN_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "median_tiling_data.h"
#include "median_tiling_key.h"
using namespace AscendC;

// ============================================================
// Median 算子 — 全 Atanh 风格 5 个 path：
//   - MedianMain<T>     : L<=8192 多行 Sort32+MrgSort 主路
//   - MedianBigSel<T>   : fp 多行 8192<L<=24576 整行常驻 + 22 iter 值二分计数
//   - MedianBig<T>      : 大 L fallback (fp L>24576 / int L>8192) 多 pass 回读二分
//   - MedianHeap<T>     : int64 only 标量堆排序
//   - MedianBigSort<T>  : fp 大 L 多核分段 Sort + 全行值二分（nSeg > 0 触发）
//
// 风格约定（参考 S8/Atanh）：
//   1) TPipe 外置，Init 收 TPipe*
//   2) 一个大 TBuf workBuf + GetWithOffset 切片（用户建议）
//   3) Compute >> IO 时不开 double buffer（BUFFER_NUM=1）
//   4) Init/Process/CopyIn/Compute/CopyOut 三阶段结构，能用 TQue 同步则用
//   5) ⚠ VEC dst 起始地址必须 32B 对齐（feedback_vec_unaligned_dst.md 血泪）
// ============================================================

namespace NsMedian {

constexpr uint32_t P_MAIN = 8192;      // 主路 UB sort 单元
constexpr float NEG = -3.0e38f;        // 负无穷哨兵
constexpr float POS = 3.0e38f;         // 正无穷哨兵
constexpr uint32_t BIGSEL_FIT = 24576; // BigSel 单行常驻最大长度
constexpr uint32_t BIG_CHUNK = 8192;   // Big/BigSel 每块大小
constexpr uint32_t BIG_RDW = 1024;     // ReduceXxx work tensor 大小（最少 count/8 floats）
constexpr uint32_t HEAP_MAX_L = 16384; // i64 整行堆排上限（L*8+L*2<180KB）；超过走分块标量值二分计数

// ------------------------------------------------------------
// 公共 Cast helper：u8/i8 走 T→half→float 两步；其他 dtype 直接 Cast<float,T>
// 每个 class 持有自己的 tmpHalf 引用并初始化
// ------------------------------------------------------------
template <typename T>
__aicore__ inline void CastInRow(const LocalTensor<half>& tmpHalf, const LocalTensor<float>& dst,
                                 const LocalTensor<T>& src, uint32_t len)
{
    if constexpr (sizeof(T) == 1) {
        Cast<half, T>(tmpHalf, src, RoundMode::CAST_NONE, len);
        Cast<float, half>(dst, tmpHalf, RoundMode::CAST_NONE, len);
    } else {
        Cast<float, T>(dst, src, RoundMode::CAST_NONE, len);
    }
}

template <typename T>
__aicore__ inline void CastOutRow(const LocalTensor<half>& tmpHalf, const LocalTensor<T>& dst,
                                  const LocalTensor<float>& src, uint32_t cnt)
{
    if constexpr (sizeof(T) == 1) {
        Cast<half, float>(tmpHalf, src, RoundMode::CAST_RINT, cnt);
        Cast<T, half>(dst, tmpHalf, RoundMode::CAST_RINT, cnt);
    } else {
        Cast<T, float>(dst, src, RoundMode::CAST_RINT, cnt);
    }
}

// ============================================================
// MedianMain<T>：主路 L<=8192，Sort32+MrgSort 多行批量化
// ============================================================
template <typename T>
class MedianMain {
public:
    __aicore__ inline MedianMain() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR in, GM_ADDR val, GM_ADDR idxo, uint32_t rowStart,
                                uint32_t rowEnd, uint32_t L, uint32_t mid, uint32_t Pad)
    {
        if (Pad == 0)
            Pad = 1; // 防御：Pad 恒 > 0（tiling 保证），clamp 消除静态检查除零告警，不改行为
        pipe_ = pipeIn;
        rb_ = rowStart;
        re_ = rowEnd;
        L_ = L;
        mid_ = mid;
        Pad_ = Pad;
        gpr_ = Pad / 32;
        K_ = P_MAIN / Pad;
        if (K_ < 1)
            K_ = 1;

        inGm_.SetGlobalBuffer((__gm__ T*)in);
        valGm_.SetGlobalBuffer((__gm__ T*)val);
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)idxo);

        pipe_->InitBuffer(inQueue_, 1, K_ * Pad_ * sizeof(T));
        pipe_->InitBuffer(outValQ_, 1, ((K_ * sizeof(T)) + 31) & ~31u);
        pipe_->InitBuffer(outIdxQ_, 1, ((K_ * sizeof(int32_t)) + 31) & ~31u);

        // workBuf layout（bank conflict 优化：src↔src 差 512n+256, src↔dst 差 512n）
        //   sc:    [0, P*4)                      32KB  score（float）       base=0
        //   ix:    [P*4+256, 2P*4+256)           32KB  index（int32/uint32）base=32K+256 → sc↔ix=512n+256 ✓
        //   pa:    [2P*4+512, 4P*4+512)          64KB  sort pair A         base=64K+512 → sc↔pa=512n ✓ ix↔pa=512n+256 ✓
        //   pb:    [0, 2P*4)                     64KB  alias sc+ix          base=0       → pa↔pb=512n ✓
        //   tmpVF: [4P*4+512, 4P*4+512+1024)      1KB  K_max=256 fp32 中位临时槽
        //   half:  [4P*4+512+1024, ...)          16KB  u8/i8 2-step Cast
        constexpr uint32_t halfBytes = (sizeof(T) == 1) ? (P_MAIN * sizeof(half)) : 0;
        pipe_->InitBuffer(workBuf_, 4 * P_MAIN * sizeof(float) + 512 + 1024 + halfBytes);
        // Gather offset 区（K_max=256 × 4 个 uint32/int32 数组 = 4KB），Process 入口生成一次
        pipe_->InitBuffer(offBuf_, 256 * 4 * sizeof(uint32_t));
    }

    __aicore__ inline void Process()
    {
        // 优化：CreateVecIndex 移到 Process 入口，所有 batch 复用同一份 ix=0..K*Pad-1
        //       (kCnt < K 时只用前 kCnt*Pad 个位置，多余位置无影响)
        LocalTensor<int32_t> ixiInit = workBuf_.GetWithOffset<int32_t>(P_MAIN, P_MAIN * sizeof(float) + 256);
        CreateVecIndex(ixiInit, (int32_t)0, K_ * Pad_);

        // Gather offset 生成（一次，所有趟共用）：每趟取每行第 mid 个 sorted pair (score,idx)
        //   scoreOff[k] = (k*Pad+mid)*2*4 bytes；idxOff[k] = scoreOff[k]+4；kPadVec[k] = k*Pad（idx 修正）
        scoreOff_ = offBuf_.GetWithOffset<int32_t>(256, 0);
        idxOff_ = offBuf_.GetWithOffset<int32_t>(256, 256 * sizeof(int32_t));
        kPadVec_ = offBuf_.GetWithOffset<int32_t>(256, 512 * sizeof(int32_t));
        idxTmp_ = offBuf_.GetWithOffset<int32_t>(256, 768 * sizeof(int32_t));
        CreateVecIndex(scoreOff_, (int32_t)0, K_);                    // [0,1,...,K-1]
        Muls<int32_t>(kPadVec_, scoreOff_, (int32_t)Pad_, K_);        // k*Pad
        Muls<int32_t>(scoreOff_, scoreOff_, (int32_t)(Pad_ * 8), K_); // k*Pad*8
        Adds<int32_t>(scoreOff_, scoreOff_, (int32_t)(mid_ * 8), K_); // +mid*8 → scoreOff
        Adds<int32_t>(idxOff_, scoreOff_, (int32_t)4, K_);            // +4 → idxOff
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        for (uint32_t r = rb_; r < re_; r += K_) {
            uint32_t kCnt = (re_ - r < K_) ? (re_ - r) : K_;
            CopyIn(r, kCnt);
            Compute(kCnt);
            CopyOut(r, kCnt);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t r, uint32_t kCnt)
    {
        LocalTensor<T> raw = inQueue_.AllocTensor<T>();
        uint32_t gapBytes = (Pad_ - L_) * (uint32_t)sizeof(T);
        if (kCnt > 1 && (gapBytes & 31) == 0 && (gapBytes / 32) <= 65535) {
            DataCopyExtParams cp{(uint16_t)kCnt, L_ * (uint32_t)sizeof(T), 0, (uint16_t)(gapBytes / 32), 0};
            DataCopyPadExtParams<T> dp{false, 0, 0, 0};
            DataCopyPad(raw, inGm_[(uint64_t)r * L_], cp, dp);
        } else {
            DataCopyExtParams cp{1, L_ * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> dp{false, 0, 0, 0};
            for (uint32_t k = 0; k < kCnt; ++k) {
                DataCopyPad(raw[k * Pad_], inGm_[(uint64_t)(r + k) * L_], cp, dp);
            }
        }
        inQueue_.EnQue(raw);
    }

    __aicore__ inline void Compute(uint32_t kCnt)
    {
        LocalTensor<T> raw = inQueue_.DeQue<T>();
        LocalTensor<T> outV = outValQ_.AllocTensor<T>();
        LocalTensor<int32_t> outI = outIdxQ_.AllocTensor<int32_t>();

        LocalTensor<float> sc = workBuf_.GetWithOffset<float>(P_MAIN, 0);
        LocalTensor<int32_t> ixi = workBuf_.GetWithOffset<int32_t>(P_MAIN, P_MAIN * sizeof(float) + 256);
        LocalTensor<uint32_t> ix = ixi.template ReinterpretCast<uint32_t>();
        LocalTensor<float> pa = workBuf_.GetWithOffset<float>(2 * P_MAIN, 2 * P_MAIN * sizeof(float) + 512);
        LocalTensor<float> pb = workBuf_.GetWithOffset<float>(2 * P_MAIN, 0);
        if constexpr (sizeof(T) == 1) {
            tmpHalf_ = workBuf_.GetWithOffset<half>(P_MAIN, 4 * P_MAIN * sizeof(float) + 512 + 1024);
        }

        uint32_t totalGroups = kCnt * gpr_;
        uint32_t totalElem = kCnt * Pad_;

        // CreateVecIndex 已在 Process 入口完成一次，此处复用
        // L==Pad 时 raw 无 gap → 单 call batched Cast+Muls 代替 K 次 per-row call
        // L<Pad 时仍需 Duplicate(NEG) 填 pad 槽 + per-row 覆盖 data 段
        if (L_ == Pad_) {
            if constexpr (std::is_same_v<T, float>) {
                Muls<float>(sc, raw.template ReinterpretCast<float>(), -1.0f, totalElem);
            } else {
                CastInRow<T>(tmpHalf_, sc, raw, totalElem);
                Muls<float>(sc, sc, -1.0f, totalElem);
            }
        } else {
            Duplicate<float>(sc, NEG, totalElem);
            for (uint32_t k = 0; k < kCnt; ++k) {
                uint32_t off = k * Pad_;
                if constexpr (std::is_same_v<T, float>) {
                    Muls<float>(sc[off], raw[off].template ReinterpretCast<float>(), -1.0f, L_);
                } else {
                    CastInRow<T>(tmpHalf_, sc[off], raw[off], L_);
                    Muls<float>(sc[off], sc[off], -1.0f, L_);
                }
            }
        }

        // 每 batch 重建 ix=0..kCnt*Pad-1（关键修正）：pb（MrgSort ping-pong 缓冲）别名 ixi 这块内存，
        // 上一趟 batch 的 merge 会把 ix 覆盖污染；多趟迭代时（每核>1趟）第 2 趟起 Sort32 用到坏 ix → index 乱码。
        CreateVecIndex(ixi, (int32_t)0, totalElem);
        PipeBarrier<PIPE_V>();

        // Sort32 全部段
        for (uint32_t g = 0; g < totalGroups; g += 255) {
            uint32_t rp = (totalGroups - g < 255) ? (totalGroups - g) : 255;
            Sort32<float>(pa[g * 64], sc[g * 32], ix[g * 32], rp);
        }

        // MrgSort：4-way 主体 + (Pad=1024) 2-way 收尾
        LocalTensor<float> src = pa, dst = pb;
        uint32_t segs = gpr_, seg = 32;
        while (segs > 1) {
            if (segs >= 4) {
                uint32_t ns = segs / 4;
                MrgSortSrcList<float> sl;
                sl.src1 = src[0];
                sl.src2 = src[seg * 2];
                sl.src3 = src[seg * 2 * 2];
                sl.src4 = src[seg * 2 * 3];
                MrgSort4Info p;
                for (int j = 0; j < 4; ++j)
                    p.elementLengths[j] = seg;
                p.ifExhaustedSuspension = false;
                p.validBit = 15;
                p.repeatTimes = (uint16_t)(kCnt * ns);
                MrgSort<float>(dst, sl, p);
                seg *= 4;
                segs = ns;
            } else { // segs==2：4-way 批量化 (拆每 seg 为 2 半 → 4 个 seg/2 子段 → repeatTimes=K)
                //   sorted_desc[seg] 拆 [0..seg/2)+[seg/2..seg) 仍各自 sorted_desc
                //   故 4-way merge {A_high, A_low, B_high, B_low} = sorted_desc(A∪B)
                //   单 MrgSort call 处理所有 K 行，省下 K-1 次 2-way 单 call 开销
                uint32_t half = seg / 2;
                MrgSortSrcList<float> sl;
                sl.src1 = src[0];
                sl.src2 = src[half * 2];
                sl.src3 = src[seg * 2];
                sl.src4 = src[seg * 2 + half * 2];
                MrgSort4Info p;
                for (int j = 0; j < 4; ++j)
                    p.elementLengths[j] = (uint16_t)half;
                p.ifExhaustedSuspension = false;
                p.validBit = 15;
                p.repeatTimes = (uint16_t)kCnt;
                MrgSort<float>(dst, sl, p);
                seg *= 2;
                segs = 1;
            }
            LocalTensor<float> t = src;
            src = dst;
            dst = t;
        }
        LocalTensor<float> sorted = src;

        // 向量化 Gather 提取（替代 K 次 scalar GetValue/SetValue）：
        //   每行第 mid 个 sorted pair 的 (score, idx)，offset 已在 Process 入口生成
        LocalTensor<float> tmpVF = workBuf_.GetWithOffset<float>(256, 4 * P_MAIN * sizeof(float) + 512);
        Gather<float>(tmpVF, sorted, scoreOff_.template ReinterpretCast<uint32_t>(), 0, kCnt);
        Muls<float>(tmpVF, tmpVF, -1.0f, kCnt); // 排序前 Muls(-1)，此处还原
        LocalTensor<uint32_t> sortedU = sorted.template ReinterpretCast<uint32_t>();
        Gather<uint32_t>(idxTmp_.template ReinterpretCast<uint32_t>(), sortedU,
                         idxOff_.template ReinterpretCast<uint32_t>(), 0, kCnt);
        Sub<int32_t>(outI, idxTmp_, kPadVec_, kCnt); // raw global idx - k*Pad → 行内 idx
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<T, float>) {
            Adds<float>(outV.template ReinterpretCast<float>(), tmpVF, 0.0f, kCnt);
        } else {
            CastOutRow<T>(tmpHalf_, outV, tmpVF, kCnt);
        }

        outValQ_.EnQue(outV);
        outIdxQ_.EnQue(outI);
        inQueue_.FreeTensor(raw);
    }

    __aicore__ inline void CopyOut(uint32_t r, uint32_t kCnt)
    {
        LocalTensor<T> outV = outValQ_.DeQue<T>();
        LocalTensor<int32_t> outI = outIdxQ_.DeQue<int32_t>();
        DataCopyExtParams cpv{1, kCnt * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyExtParams cpi{1, kCnt * 4u, 0, 0, 0};
        DataCopyPad(valGm_[r], outV, cpv);
        DataCopyPad(idxGm_[r], outI, cpi);
        outValQ_.FreeTensor(outV);
        outIdxQ_.FreeTensor(outI);
    }

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outValQ_;
    TQue<QuePosition::VECOUT, 1> outIdxQ_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> offBuf_; // Gather offset 区（Process 入口生成一次）
    LocalTensor<half> tmpHalf_;
    LocalTensor<int32_t> scoreOff_, idxOff_, kPadVec_, idxTmp_;

    GlobalTensor<T> inGm_, valGm_;
    GlobalTensor<int32_t> idxGm_;

    uint32_t rb_, re_, L_, mid_, Pad_, gpr_, K_;
};

// ============================================================
// MedianSmallL<T>：极小 L（L<=16）大 batch — 连续搬 + Gather 散到 [K,Pad=32] 槽 + Sort32
//   解决 MainPath 小 L 走 per-row K 次 DataCopyPad 的搬运瓶颈：
//   1) 一次连续 DataCopyPad 把 [K,L] 整块搬进 UB（高效，blockLen=K*L 整块）
//   2) Muls(-1) 真实数据 + 末尾 NEG 哨兵；一次 Gather 按 offset 把每行 L 个散到 Pad=32 槽前 L，
//      pad 槽（j>=L）gather 指向 NEG 哨兵（排序到末尾，不影响 mid）
//   3) Sort32 降序 + Gather 提取每行第 mid 个 pair → median value + 行内 index
// ============================================================
template <typename T>
class MedianSmallL {
public:
    __aicore__ inline MedianSmallL() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR in, GM_ADDR val, GM_ADDR idxo, uint32_t rowStart,
                                uint32_t rowEnd, uint32_t L, uint32_t mid)
    {
        pipe_ = pipeIn;
        rb_ = rowStart;
        re_ = rowEnd;
        L_ = L;
        mid_ = mid;
        Pad_ = 32;
        K_ = (L <= 16) ? 224 : 160; // K 按 L 控总 UB < 192KB（L>16 时 chunkF/inQ 更大）
        inGm_.SetGlobalBuffer((__gm__ T*)in);
        valGm_.SetGlobalBuffer((__gm__ T*)val);
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)idxo);

        uint32_t KP = K_ * Pad_;
        pipe_->InitBuffer(inQ_, 1, K_ * L_ * sizeof(T));
        pipe_->InitBuffer(outVQ_, 1, (K_ * (uint32_t)sizeof(T) + 31) & ~31u);
        pipe_->InitBuffer(outIQ_, 1, (K_ * 4u + 31) & ~31u);
        // wb: sc(KP) + ix(KP) + pa(KP*2) + chunkF(K*L+32) + gatherOff(KP) + scoreOff/idxOff/kPadVec/tmpVF(K each)
        pipe_->InitBuffer(wb_, KP * 4 + KP * 4 + KP * 2 * 4 + (K_ * L_ + 32) * 4 + KP * 4 + 4 * K_ * 4 + 256);
    }

    __aicore__ inline void Process()
    {
        uint32_t KP = K_ * Pad_;
        uint32_t RL = K_ * L_;
        sc_ = wb_.GetWithOffset<float>(KP, 0);
        ixi_ = wb_.GetWithOffset<int32_t>(KP, KP * 4);
        pa_ = wb_.GetWithOffset<float>(KP * 2, 2 * KP * 4);
        uint32_t bAfterPa = 2 * KP * 4 + KP * 2 * 4;
        chunkF_ = wb_.GetWithOffset<float>(RL + 32, bAfterPa);
        gatherOff_ = wb_.GetWithOffset<int32_t>(KP, bAfterPa + (RL + 32) * 4);
        uint32_t bOff = bAfterPa + (RL + 32) * 4 + KP * 4;
        scoreOff_ = wb_.GetWithOffset<int32_t>(K_, bOff);
        idxOff_ = wb_.GetWithOffset<int32_t>(K_, bOff + K_ * 4);
        kPadVec_ = wb_.GetWithOffset<int32_t>(K_, bOff + 2 * K_ * 4);
        tmpVF_ = wb_.GetWithOffset<float>(K_, bOff + 3 * K_ * 4);

        // ix = [0,1,...,KP-1]（Sort32 的 index 输入，sorted 后即原 sc 槽位 = k*Pad+列）
        CreateVecIndex(ixi_, (int32_t)0, KP);

        // gather 散列 offset：off[k*Pad+j] = (j<L)? (k*L+j)*4 : RL*4（pad 槽指向 chunkF[RL] = NEG 哨兵）
        Duplicate<int32_t>(gatherOff_, (int32_t)RL, KP);
        for (uint32_t k = 0; k < K_; ++k) {
            CreateVecIndex(gatherOff_[k * Pad_], (int32_t)(k * L_), L_);
        }
        Muls<int32_t>(gatherOff_, gatherOff_, (int32_t)4, KP);

        // mid pair 提取 offset（同 MedianMain）：scoreOff[k]=(k*Pad+mid)*8，idxOff=+4，kPadVec=k*Pad
        CreateVecIndex(scoreOff_, (int32_t)0, K_);
        Muls<int32_t>(kPadVec_, scoreOff_, (int32_t)Pad_, K_);
        Muls<int32_t>(scoreOff_, scoreOff_, (int32_t)(Pad_ * 8), K_);
        Adds<int32_t>(scoreOff_, scoreOff_, (int32_t)(mid_ * 8), K_);
        Adds<int32_t>(idxOff_, scoreOff_, (int32_t)4, K_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        for (uint32_t rs = rb_; rs < re_; rs += K_) {
            uint32_t kCnt = (re_ - rs < K_) ? (re_ - rs) : K_;
            CopyIn(rs, kCnt);
            Compute(kCnt);
            CopyOut(rs, kCnt);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rs, uint32_t kCnt)
    {
        LocalTensor<T> raw = inQ_.AllocTensor<T>();
        DataCopyExtParams cp{1, kCnt * L_ * (uint32_t)sizeof(T), 0, 0, 0}; // 一次连续整块
        DataCopyPadExtParams<T> dp{false, 0, 0, 0};
        DataCopyPad(raw, inGm_[(uint64_t)rs * L_], cp, dp);
        inQ_.EnQue(raw);
    }

    __aicore__ inline void Compute(uint32_t kCnt)
    {
        LocalTensor<T> raw = inQ_.DeQue<T>();
        uint32_t RL = K_ * L_;

        // 先整体填 NEG（起始对齐；pad 槽 + 哨兵都是 NEG，排序到末尾不影响 mid），
        // 再 Muls(-1)/Cast 覆盖前 kCnt*L（真实数据取负 → Sort32 降序得实际升序）
        Duplicate<float>(chunkF_, NEG, RL + 32);
        if constexpr (std::is_same_v<T, float>) {
            Muls<float>(chunkF_, raw.template ReinterpretCast<float>(), -1.0f, kCnt * L_);
        } else {
            Cast<float, T>(chunkF_, raw, RoundMode::CAST_NONE, kCnt * L_);
            Muls<float>(chunkF_, chunkF_, -1.0f, kCnt * L_);
        }
        inQ_.FreeTensor(raw);
        PipeBarrier<PIPE_V>();

        // 一次 Gather：把连续 [kCnt,L] 散到 sc 的 [kCnt,Pad] 槽（前 L 真实，pad=NEG）
        Gather<float>(sc_, chunkF_, gatherOff_.template ReinterpretCast<uint32_t>(), 0, kCnt * Pad_);
        PipeBarrier<PIPE_V>();

        // Sort32 降序（每行 Pad=32 一组，gpr=1 无需 MrgSort）
        LocalTensor<uint32_t> ix = ixi_.template ReinterpretCast<uint32_t>();
        for (uint32_t g = 0; g < kCnt; g += 255) {
            uint32_t rp = (kCnt - g < 255) ? (kCnt - g) : 255;
            Sort32<float>(pa_[g * Pad_ * 2], sc_[g * Pad_], ix[g * Pad_], rp);
        }
        PipeBarrier<PIPE_V>();

        // Gather 提取每行第 mid 个 pair：score（negate 还原）+ 行内 index
        LocalTensor<T> outV = outVQ_.AllocTensor<T>();
        LocalTensor<int32_t> outI = outIQ_.AllocTensor<int32_t>();
        Gather<float>(tmpVF_, pa_, scoreOff_.template ReinterpretCast<uint32_t>(), 0, kCnt);
        Muls<float>(tmpVF_, tmpVF_, -1.0f, kCnt);
        LocalTensor<uint32_t> paU = pa_.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> idxTmp = chunkF_.template ReinterpretCast<int32_t>(); // Sort 后 chunkF 空，借作 idx 临时
        Gather<uint32_t>(idxTmp.template ReinterpretCast<uint32_t>(), paU, idxOff_.template ReinterpretCast<uint32_t>(),
                         0, kCnt);
        Sub<int32_t>(outI, idxTmp, kPadVec_, kCnt); // 非原地：global 槽位 - k*Pad → 行内列 idx
        if constexpr (std::is_same_v<T, float>) {
            Adds<float>(outV.template ReinterpretCast<float>(), tmpVF_, 0.0f, kCnt);
        } else {
            Cast<T, float>(outV, tmpVF_, RoundMode::CAST_RINT, kCnt);
        }
        PipeBarrier<PIPE_V>();
        outVQ_.EnQue(outV);
        outIQ_.EnQue(outI);
    }

    __aicore__ inline void CopyOut(uint32_t rs, uint32_t kCnt)
    {
        LocalTensor<T> outV = outVQ_.DeQue<T>();
        LocalTensor<int32_t> outI = outIQ_.DeQue<int32_t>();
        DataCopyExtParams cv{1, kCnt * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyExtParams ci{1, kCnt * 4u, 0, 0, 0};
        DataCopyPad(valGm_[rs], outV, cv);
        DataCopyPad(idxGm_[rs], outI, ci);
        outVQ_.FreeTensor(outV);
        outIQ_.FreeTensor(outI);
    }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQ_;
    TQue<QuePosition::VECOUT, 1> outVQ_;
    TQue<QuePosition::VECOUT, 1> outIQ_;
    TBuf<TPosition::VECCALC> wb_;

    GlobalTensor<T> inGm_, valGm_;
    GlobalTensor<int32_t> idxGm_;

    LocalTensor<float> sc_, pa_, chunkF_, tmpVF_;
    LocalTensor<int32_t> ixi_, gatherOff_, scoreOff_, idxOff_, kPadVec_;

    uint32_t rb_, re_, L_, mid_, Pad_, K_;
};

// ============================================================
// MedianBigSel<T>：fp 大 L 多行二分计数主路（8192<L<=24576）
// 算法：每行整行常驻 UB → range scan → 22 iter 二分 → value find → index find
// ============================================================
template <typename T>
class MedianBigSel {
public:
    __aicore__ inline MedianBigSel() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR in, GM_ADDR val, GM_ADDR idxo, uint32_t rowStart,
                                uint32_t rowEnd, uint32_t L, uint32_t mid)
    {
        pipe_ = pipeIn;
        rb_ = rowStart;
        re_ = rowEnd;
        L_ = L;
        mid_ = mid;
        Lp_ = ((L + 63) / 64) * 64; // L 64-align 上取，尾 +POS 不计数

        inGm_.SetGlobalBuffer((__gm__ T*)in);
        valGm_.SetGlobalBuffer((__gm__ T*)val);
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)idxo);

        // inBuf 用 TBuf（chunk load 走 TBuf + 手填 SetFlag，避免 TQue Alloc/EnQue/DeQue/Free 在
        //   chunk-loop 中的累积开销 — IO-bound 路径上 TQue 开销显著）
        pipe_->InitBuffer(inBuf_, BIG_CHUNK * sizeof(T));
        pipe_->InitBuffer(outValBuf_, 32); // 1 T
        pipe_->InitBuffer(outIdxBuf_, 32); // 1 int32

        constexpr uint32_t halfBytes = (sizeof(T) == 1) ? (BIG_CHUNK * sizeof(half)) : 0;
        uint32_t bytes = BIGSEL_FIT * sizeof(float) + BIG_CHUNK * sizeof(float) + BIG_RDW * sizeof(float) + 256 +
                         (BIG_CHUNK / 8 + 32) + halfBytes;
        pipe_->InitBuffer(workBuf_, bytes);
    }

    __aicore__ inline void Process()
    {
        // 持久 fp32 数据 + 工作 tensor（一次 GetWithOffset）
        data_ = workBuf_.GetWithOffset<float>(BIGSEL_FIT, 0);
        ind_ = workBuf_.GetWithOffset<float>(BIG_CHUNK, BIGSEL_FIT * sizeof(float));
        rdw_ = workBuf_.GetWithOffset<float>(BIG_RDW, (BIGSEL_FIT + BIG_CHUNK) * sizeof(float));
        red_ = workBuf_.GetWithOffset<float>(64, (BIGSEL_FIT + BIG_CHUNK + BIG_RDW) * sizeof(float));
        mk_ = workBuf_.GetWithOffset<uint8_t>(BIG_CHUNK / 8 + 32,
                                              (BIGSEL_FIT + BIG_CHUNK + BIG_RDW + 64) * sizeof(float));
        if constexpr (sizeof(T) == 1) {
            uint32_t halfOff = BIGSEL_FIT * sizeof(float) + BIG_CHUNK * sizeof(float) + BIG_RDW * sizeof(float) + 256 +
                               (BIG_CHUNK / 8 + 32);
            tmpHalf_ = workBuf_.GetWithOffset<half>(BIG_CHUNK, halfOff);
        }

        for (uint32_t r = rb_; r < re_; ++r) {
            LoadRowToData(r); // CopyIn 整行（chunked）+ Cast 到 data_
            float v;
            int32_t idx;
            ComputeMedian(v, &idx);       // range scan + 22 iter + value find + index find
            EnqueueAndCopyOut(r, v, idx); // 写 GM
        }
    }

private:
    // ---- 整行 chunk-load → CastInRow → data_[off..off+clen)（TBuf + 手填 sync）----
    // ⚠ VEC dst 必须 32B 对齐：先整批 Duplicate(data_, POS, Lp_) 填 +INF，
    //    然后 chunk-load Cast/Adds 覆盖 data_[0..L)。data_ 起始 0 对齐，covered 段 dst=data_[k*CHUNK] 也对齐。
    __aicore__ inline void LoadRowToData(uint32_t r)
    {
        // 1) 整批 fill POS（aligned dst + aligned len Lp_）
        Duplicate<float>(data_, POS, Lp_);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        LocalTensor<T> raw = inBuf_.Get<T>();
        // 2) chunk-load 覆盖 data_[0..L) 真实数据；尾段 data_[L..Lp) 保留 POS
        for (uint32_t off = 0; off < L_; off += BIG_CHUNK) {
            uint32_t clen = (L_ - off < BIG_CHUNK) ? (L_ - off) : BIG_CHUNK;
            DataCopyExtParams cp{1, clen * (uint32_t)sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> dp{false, 0, 0, 0};
            DataCopyPad(raw, inGm_[(uint64_t)r * L_ + off], cp, dp);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr (std::is_same_v<T, float>) {
                Adds<float>(data_[off], raw.template ReinterpretCast<float>(), 0.0f, clen);
            } else {
                CastInRow<T>(tmpHalf_, data_[off], raw, clen);
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
    }

    // 向量化计数 count(data <= m)
    __aicore__ inline float CountLE(float m)
    {
        const uint32_t C = BIG_CHUNK;
        uint32_t nc = 0;
        for (uint32_t off = 0; off < L_; off += C) {
            uint32_t clen = (L_ - off < C) ? (L_ - off) : C;
            uint32_t cl = ((clen + 63) / 64) * 64;
            CompareScalar(mk_, data_[off], m, CMPMODE::LE, cl);
            Duplicate<float>(ind_, 1.0f, cl);
            Select(ind_, mk_, ind_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, cl);
            ReduceSum<float>(red_[nc * 8], ind_, rdw_, cl);
            ++nc;
        }
        // V→S 同步：1 次 PipeBarrier 即可（ReduceSum 输出在 V，GetValue 在 S）
        PipeBarrier<PIPE_V>();
        float le = 0;
        for (uint32_t i = 0; i < nc; ++i)
            le += red_[i * 8].GetValue(0);
        return le;
    }

    // 主算法：range scan → 22 iter 值二分 → value find（min{data>lo}）→ idx find（min{j:data[j]==v}）
    __aicore__ inline void ComputeMedian(float& v, int32_t* idxOut)
    {
        const uint32_t C = BIG_CHUNK;

        // 1) range scan：分块 ReduceMin/Max 写不同槽，1 次同步
        float lo = POS, hi = NEG;
        uint32_t nr = 0;
        for (uint32_t off = 0; off < L_; off += C) {
            uint32_t clen = (L_ - off < C) ? (L_ - off) : C;
            ReduceMin<float>(red_[nr * 16], data_[off], rdw_, clen);
            ReduceMax<float>(red_[nr * 16 + 8], data_[off], rdw_, clen);
            ++nr;
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < nr; ++i) {
            float mn = red_[i * 16].GetValue(0);
            float mx = red_[i * 16 + 8].GetValue(0);
            if (mn < lo)
                lo = mn;
            if (mx > hi)
                hi = mx;
        }

        // 2) 22 iter 二分计数
        float mid1 = (float)((int)mid_ + 1);
        for (int it = 0; it < 22; ++it) {
            float m = (lo + hi) * 0.5f;
            float le = CountLE(m);
            if (le >= mid1)
                hi = m;
            else
                lo = m;
        }

        // 3) value find：v = min{data > lo} = 第 (mid+1) 小的值（落在真实数据上）
        {
            float best = POS;
            uint32_t nc = 0;
            for (uint32_t off = 0; off < L_; off += C) {
                uint32_t clen = (L_ - off < C) ? (L_ - off) : C;
                uint32_t cl = ((clen + 63) / 64) * 64;
                CompareScalar(mk_, data_[off], lo, CMPMODE::GT, cl);
                Select(ind_, mk_, data_[off], POS, SELMODE::VSEL_TENSOR_SCALAR_MODE, cl);
                ReduceMin<float>(red_[nc * 8], ind_, rdw_, cl);
                ++nc;
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < nc; ++i) {
                float b2 = red_[i * 8].GetValue(0);
                if (b2 < best)
                    best = b2;
            }
            v = best;
        }

        // 4) index find：idx = min{j : data[j] == v}（向量化，匹配 torch.median）
        {
            float best = POS;
            uint32_t nc = 0;
            for (uint32_t off = 0; off < L_; off += C) {
                uint32_t clen = (L_ - off < C) ? (L_ - off) : C;
                uint32_t cl = ((clen + 63) / 64) * 64;
                CompareScalar(mk_, data_[off], v, CMPMODE::EQ, cl);
                CreateVecIndex(ind_.template ReinterpretCast<int32_t>(), (int32_t)off, cl);
                Cast<float, int32_t>(ind_, ind_.template ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, cl);
                Select(ind_, mk_, ind_, POS, SELMODE::VSEL_TENSOR_SCALAR_MODE, cl);
                ReduceMin<float>(red_[nc * 8], ind_, rdw_, cl);
                ++nc;
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < nc; ++i) {
                float b2 = red_[i * 8].GetValue(0);
                if (b2 < best)
                    best = b2;
            }
            *idxOut = (int32_t)best;
        }
    }

    // per-row CopyOut：TBuf + 手填 sync 写 GM
    __aicore__ inline void EnqueueAndCopyOut(uint32_t r, float v, int32_t idx)
    {
        LocalTensor<T> outV = outValBuf_.Get<T>();
        LocalTensor<int32_t> outI = outIdxBuf_.Get<int32_t>();

        if constexpr (std::is_same_v<T, float>) {
            outV.template ReinterpretCast<float>().SetValue(0, v);
        } else {
            LocalTensor<float> tmpFp = workBuf_.GetWithOffset<float>(
                64, (BIGSEL_FIT + BIG_CHUNK + BIG_RDW) * sizeof(float));
            tmpFp.SetValue(0, v);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            CastOutRow<T>(tmpHalf_, outV, tmpFp, 8);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        outI.SetValue(0, idx);

        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cpv{1, (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyExtParams cpi{1, 4u, 0, 0, 0};
        DataCopyPad(valGm_[r], outV, cpv);
        DataCopyPad(idxGm_[r], outI, cpi);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> inBuf_;     // T-input chunk（TBuf + 手填 sync）
    TBuf<TPosition::VECCALC> outValBuf_; // 1 T 输出槽
    TBuf<TPosition::VECCALC> outIdxBuf_; // 1 int32 输出槽
    TBuf<TPosition::VECCALC> workBuf_;

    GlobalTensor<T> inGm_, valGm_;
    GlobalTensor<int32_t> idxGm_;

    LocalTensor<float> data_, ind_, rdw_, red_;
    LocalTensor<uint8_t> mk_;
    LocalTensor<half> tmpHalf_;

    uint32_t rb_, re_, L_, mid_, Lp_;
};

// ============================================================
// MedianBig<T>：大 L fallback（fp L>24576 或 int L>8192）
// 算法：每行多 pass HBM 回读：range scan + 40 iter 二分 + idx find
// 性能不是首要（fallback 路径），代码简洁优先
// ============================================================
template <typename T>
class MedianBig {
public:
    __aicore__ inline MedianBig() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR in, GM_ADDR val, GM_ADDR idxo, uint32_t rowStart,
                                uint32_t rowEnd, uint32_t L, uint32_t mid)
    {
        pipe_ = pipeIn;
        rb_ = rowStart;
        re_ = rowEnd;
        L_ = L;
        mid_ = mid;

        inGm_.SetGlobalBuffer((__gm__ T*)in);
        valGm_.SetGlobalBuffer((__gm__ T*)val);
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)idxo);

        // Big 路径 40 iter × N chunks 重读 HBM，chunk 内层用 TBuf + 手填 sync
        // 否则 TQue 每 chunk 4 个 Alloc/EnQue/DeQue/Free 开销在 200+ chunk 循环中累积可观
        pipe_->InitBuffer(inBuf_, BIG_CHUNK * sizeof(T));
        pipe_->InitBuffer(outValBuf_, 32);
        pipe_->InitBuffer(outIdxBuf_, 32);

        constexpr uint32_t halfBytes = (sizeof(T) == 1) ? (BIG_CHUNK * sizeof(half)) : 0;
        pipe_->InitBuffer(workBuf_,
                          BIG_CHUNK * sizeof(float) + BIG_RDW * sizeof(float) + 256 + (BIG_CHUNK / 8 + 32) + halfBytes);
    }

    __aicore__ inline void Process()
    {
        sc_ = workBuf_.GetWithOffset<float>(BIG_CHUNK, 0);
        rdw_ = workBuf_.GetWithOffset<float>(BIG_RDW, BIG_CHUNK * sizeof(float));
        red_ = workBuf_.GetWithOffset<float>(64, (BIG_CHUNK + BIG_RDW) * sizeof(float));
        mk_ = workBuf_.GetWithOffset<uint8_t>(BIG_CHUNK / 8 + 32, (BIG_CHUNK + BIG_RDW) * sizeof(float) + 256);
        if constexpr (sizeof(T) == 1) {
            uint32_t halfOff = (BIG_CHUNK + BIG_RDW) * sizeof(float) + 256 + (BIG_CHUNK / 8 + 32);
            tmpHalf_ = workBuf_.GetWithOffset<half>(BIG_CHUNK, halfOff);
        }

        for (uint32_t r = rb_; r < re_; ++r) {
            float v;
            int32_t idx;
            ComputeRowMedian(r, v, idx);
            CopyOutOne(r, v, idx);
        }
    }

private:
    // 载入一个 chunk 到 sc_ 作 fp32（TBuf + 手填 sync，避免 TQue 累积开销）
    __aicore__ inline void LoadChunk(uint32_t r, uint32_t off, uint32_t clen)
    {
        LocalTensor<T> raw = inBuf_.Get<T>();
        DataCopyExtParams cp{1, clen * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> dp{false, 0, 0, 0};
        DataCopyPad(raw, inGm_[(uint64_t)r * L_ + off], cp, dp);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr (std::is_same_v<T, float>) {
            Adds<float>(sc_, raw.template ReinterpretCast<float>(), 0.0f, clen);
        } else {
            CastInRow<T>(tmpHalf_, sc_, raw, clen);
        }
    }

    __aicore__ inline void ComputeRowMedian(uint32_t r, float& v, int32_t& idx)
    {
        // 1) range scan
        float lo = POS, hi = NEG;
        for (uint32_t off = 0; off < L_; off += BIG_CHUNK) {
            uint32_t clen = (L_ - off < BIG_CHUNK) ? (L_ - off) : BIG_CHUNK;
            LoadChunk(r, off, clen);
            ReduceMin<float>(red_, sc_, rdw_, clen);
            ReduceMax<float>(red_[8], sc_, rdw_, clen);
            PipeBarrier<PIPE_V>();
            float mn = red_.GetValue(0);
            float mx = red_[8].GetValue(0);
            if (mn < lo)
                lo = mn;
            if (mx > hi)
                hi = mx;
        }

        // 2) 40 iter 二分计数（每 iter 重读全行 HBM）
        float mid1 = (float)((int)mid_ + 1);
        for (int it = 0; it < 40; ++it) {
            float m = (lo + hi) * 0.5f;
            float le = 0;
            for (uint32_t off = 0; off < L_; off += BIG_CHUNK) {
                uint32_t clen = (L_ - off < BIG_CHUNK) ? (L_ - off) : BIG_CHUNK;
                LoadChunk(r, off, clen);
                uint32_t cl = ((clen + 63) / 64) * 64;
                CompareScalar(mk_, sc_, m, CMPMODE::LE, cl);
                Duplicate<float>(sc_, 1.0f, clen);
                Select(sc_, mk_, sc_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, clen);
                ReduceSum<float>(red_, sc_, rdw_, clen);
                PipeBarrier<PIPE_V>();
                le += red_.GetValue(0);
            }
            if (le >= mid1)
                hi = m;
            else
                lo = m;
        }

        // 3) value & idx find（再扫一遍 HBM）
        v = hi;
        idx = 0;
        bool found = false;
        for (uint32_t off = 0; off < L_ && !found; off += BIG_CHUNK) {
            uint32_t clen = (L_ - off < BIG_CHUNK) ? (L_ - off) : BIG_CHUNK;
            LoadChunk(r, off, clen);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < clen; ++j) {
                if (sc_.GetValue(j) == v) {
                    idx = off + j;
                    found = true;
                    break;
                }
            }
        }
    }

    __aicore__ inline void CopyOutOne(uint32_t r, float v, int32_t idx)
    {
        LocalTensor<T> outV = outValBuf_.Get<T>();
        LocalTensor<int32_t> outI = outIdxBuf_.Get<int32_t>();
        if constexpr (std::is_same_v<T, float>) {
            outV.template ReinterpretCast<float>().SetValue(0, v);
        } else {
            LocalTensor<float> tmpFp = workBuf_.GetWithOffset<float>(64, (BIG_CHUNK + BIG_RDW) * sizeof(float));
            tmpFp.SetValue(0, v);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            CastOutRow<T>(tmpHalf_, outV, tmpFp, 8);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        outI.SetValue(0, idx);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cpv{1, (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyExtParams cpi{1, 4u, 0, 0, 0};
        DataCopyPad(valGm_[r], outV, cpv);
        DataCopyPad(idxGm_[r], outI, cpi);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> inBuf_;
    TBuf<TPosition::VECCALC> outValBuf_;
    TBuf<TPosition::VECCALC> outIdxBuf_;
    TBuf<TPosition::VECCALC> workBuf_;

    GlobalTensor<T> inGm_, valGm_;
    GlobalTensor<int32_t> idxGm_;

    LocalTensor<float> sc_, rdw_, red_;
    LocalTensor<uint8_t> mk_;
    LocalTensor<half> tmpHalf_;

    uint32_t rb_, re_, L_, mid_;
};

// ============================================================
// MedianHeap<T>：int64 only 标量堆排序（dt0==4）
// ============================================================
template <typename T>
class MedianHeap {
public:
    __aicore__ inline MedianHeap() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR in, GM_ADDR val, GM_ADDR idxo, uint32_t rowStart,
                                uint32_t rowEnd, uint32_t L, uint32_t mid)
    {
        pipe_ = pipeIn;
        rb_ = rowStart;
        re_ = rowEnd;
        L_ = L;
        mid_ = mid;
        inGm_.SetGlobalBuffer((__gm__ T*)in);
        valGm_.SetGlobalBuffer((__gm__ T*)val);
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)idxo);

        // L 小（<=HEAP_MAX_L）：整行堆排（inQueue 整行 i64 + int16 索引，L≤32767 适用）
        // L 大：分块标量值二分计数（bigBuf BIG_CHUNK，UB 固定），避免整行 i64 进 UB 越界 + int16 溢出
        useBig_ = (L_ > HEAP_MAX_L);
        pipe_->InitBuffer(outValQ_, 1, 32);
        pipe_->InitBuffer(outIdxQ_, 1, 32);
        if (!useBig_) {
            pipe_->InitBuffer(inQueue_, 1, L_ * sizeof(T));
            pipe_->InitBuffer(workBuf_, L_ * sizeof(int16_t) + 32);
        } else {
            pipe_->InitBuffer(bigBuf_, BIG_CHUNK * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (sizeof(T) != 8)
            return; // 仅 int64

        if (useBig_) {
            for (uint32_t r = rb_; r < re_; ++r) {
                T v;
                int32_t idx;
                ComputeBig(r, v, idx);
                CopyOutOne(r, v, idx);
            }
        } else {
            for (uint32_t r = rb_; r < re_; ++r) {
                CopyIn(r);
                T v;
                int32_t idx;
                Compute(v, idx);
                CopyOutOne(r, v, idx);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t r)
    {
        LocalTensor<T> raw = inQueue_.AllocTensor<T>();
        DataCopyExtParams cp{1, L_ * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> dp{false, 0, 0, 0};
        DataCopyPad(raw, inGm_[(uint64_t)r * L_], cp, dp);
        inQueue_.EnQue(raw);
    }

    // 小 L i64：整行常驻 UB + 标量值二分计数（4-way unroll 顺序扫，替代堆排 O(NlogN) 随机访问）
    __aicore__ inline void Compute(T& v, int32_t& idx)
    {
        LocalTensor<T> a = inQueue_.DeQue<T>();
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        uint32_t n = L_, c4 = n & ~3u;

        // 1) range scan（unroll）
        int64_t lo = (int64_t)0x7FFFFFFFFFFFFFFFLL;
        int64_t hi = (int64_t)0x8000000000000000LL;
        {
            uint32_t j = 0;
            for (; j < c4; j += 4) {
                int64_t x0 = (int64_t)a.GetValue(j), x1 = (int64_t)a.GetValue(j + 1);
                int64_t x2 = (int64_t)a.GetValue(j + 2), x3 = (int64_t)a.GetValue(j + 3);
                int64_t mn01 = x0 < x1 ? x0 : x1, mn23 = x2 < x3 ? x2 : x3;
                int64_t mx01 = x0 > x1 ? x0 : x1, mx23 = x2 > x3 ? x2 : x3;
                int64_t mn = mn01 < mn23 ? mn01 : mn23, mx = mx01 > mx23 ? mx01 : mx23;
                if (mn < lo)
                    lo = mn;
                if (mx > hi)
                    hi = mx;
            }
            for (; j < n; ++j) {
                int64_t x = (int64_t)a.GetValue(j);
                if (x < lo)
                    lo = x;
                if (x > hi)
                    hi = x;
            }
        }

        // 2) 整数二分计数（unroll，整行常驻不重读）
        int64_t mid1 = (int64_t)mid_ + 1;
        while (lo < hi) {
            int64_t m = lo + (int64_t)(((uint64_t)hi - (uint64_t)lo) >> 1);
            int64_t cnt = 0;
            uint32_t j = 0;
            for (; j < c4; j += 4) {
                int64_t x0 = (int64_t)a.GetValue(j), x1 = (int64_t)a.GetValue(j + 1);
                int64_t x2 = (int64_t)a.GetValue(j + 2), x3 = (int64_t)a.GetValue(j + 3);
                cnt += (int64_t)(x0 <= m) + (int64_t)(x1 <= m) + (int64_t)(x2 <= m) + (int64_t)(x3 <= m);
            }
            for (; j < n; ++j) {
                if ((int64_t)a.GetValue(j) <= m)
                    cnt++;
            }
            if (cnt >= mid1)
                hi = m;
            else
                lo = m + 1;
        }
        v = (T)lo;

        // 3) 对齐 torch：升序第 mid 个元素 = 第 (mid - count(<M)) 个 M 的下标（stable sort 语义，
        //    与 MedianMain/Sort32 路一致）。先数严格小于 M 的个数 cntLess，occ=mid-cntLess，
        //    再顺序找第 occ 个 == M 的位置（不能取首个 == M）。
        int64_t cntLess = 0;
        {
            uint32_t j = 0;
            for (; j < c4; j += 4) {
                int64_t x0 = (int64_t)a.GetValue(j), x1 = (int64_t)a.GetValue(j + 1);
                int64_t x2 = (int64_t)a.GetValue(j + 2), x3 = (int64_t)a.GetValue(j + 3);
                cntLess += (int64_t)(x0 < lo) + (int64_t)(x1 < lo) + (int64_t)(x2 < lo) + (int64_t)(x3 < lo);
            }
            for (; j < n; ++j) {
                if ((int64_t)a.GetValue(j) < lo)
                    cntLess++;
            }
        }
        int64_t occ = (int64_t)mid_ - cntLess;
        if (occ < 0)
            occ = 0;
        idx = 0;
        {
            int64_t seen = 0;
            for (uint32_t j = 0; j < n; ++j) {
                if ((int64_t)a.GetValue(j) == lo) {
                    if (seen == occ) {
                        idx = (int32_t)j;
                        break;
                    }
                    ++seen;
                }
            }
        }
        inQueue_.FreeTensor(a);
    }

    // 大 L i64：分块 load 一段到 bigBuf（标量读），返回该段 LocalTensor
    __aicore__ inline LocalTensor<T> LoadChunkBig(uint32_t r, uint32_t off, uint32_t clen)
    {
        LocalTensor<T> raw = bigBuf_.Get<T>();
        DataCopyExtParams cp{1, clen * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> dp{false, 0, 0, 0};
        DataCopyPad(raw, inGm_[(uint64_t)r * L_ + off], cp, dp);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        return raw;
    }

    // 大 L i64：分块标量值二分计数（整数二分，UB 固定 = BIG_CHUNK*8）
    //   标量循环 4-way unroll：一次连续取 4 个 GetValue，标量流水更密、吞吐更高
    __aicore__ inline void ComputeBig(uint32_t r, T& v, int32_t& idx)
    {
        // 1) range scan：标量 min/max（INT64 极值初始化，去 first 分支）
        int64_t lo = (int64_t)0x7FFFFFFFFFFFFFFFLL;
        int64_t hi = (int64_t)0x8000000000000000LL;
        for (uint32_t off = 0; off < L_; off += BIG_CHUNK) {
            uint32_t clen = (L_ - off < BIG_CHUNK) ? (L_ - off) : BIG_CHUNK;
            LocalTensor<T> a = LoadChunkBig(r, off, clen);
            uint32_t c4 = clen & ~3u, j = 0;
            for (; j < c4; j += 4) {
                int64_t x0 = (int64_t)a.GetValue(j);
                int64_t x1 = (int64_t)a.GetValue(j + 1);
                int64_t x2 = (int64_t)a.GetValue(j + 2);
                int64_t x3 = (int64_t)a.GetValue(j + 3);
                int64_t mn01 = x0 < x1 ? x0 : x1, mn23 = x2 < x3 ? x2 : x3;
                int64_t mx01 = x0 > x1 ? x0 : x1, mx23 = x2 > x3 ? x2 : x3;
                int64_t mn = mn01 < mn23 ? mn01 : mn23, mx = mx01 > mx23 ? mx01 : mx23;
                if (mn < lo)
                    lo = mn;
                if (mx > hi)
                    hi = mx;
            }
            for (; j < clen; ++j) {
                int64_t x = (int64_t)a.GetValue(j);
                if (x < lo)
                    lo = x;
                if (x > hi)
                    hi = x;
            }
        }

        // 2) 整数二分计数：找最小 m 使 count(<=m) >= mid+1
        int64_t mid1 = (int64_t)mid_ + 1;
        while (lo < hi) {
            int64_t m = lo + (int64_t)(((uint64_t)hi - (uint64_t)lo) >> 1); // 无符号中点避免溢出
            int64_t cnt = 0;
            for (uint32_t off = 0; off < L_; off += BIG_CHUNK) {
                uint32_t clen = (L_ - off < BIG_CHUNK) ? (L_ - off) : BIG_CHUNK;
                LocalTensor<T> a = LoadChunkBig(r, off, clen);
                uint32_t c4 = clen & ~3u, j = 0;
                for (; j < c4; j += 4) {
                    int64_t x0 = (int64_t)a.GetValue(j);
                    int64_t x1 = (int64_t)a.GetValue(j + 1);
                    int64_t x2 = (int64_t)a.GetValue(j + 2);
                    int64_t x3 = (int64_t)a.GetValue(j + 3);
                    cnt += (int64_t)(x0 <= m) + (int64_t)(x1 <= m) + (int64_t)(x2 <= m) + (int64_t)(x3 <= m);
                }
                for (; j < clen; ++j) {
                    if ((int64_t)a.GetValue(j) <= m)
                        cnt++;
                }
            }
            if (cnt >= mid1)
                hi = m;
            else
                lo = m + 1;
        }
        v = (T)lo;

        // 3) 对齐 torch：升序第 mid 个 = 第 (mid - count(<M)) 个 M（stable）。
        //    先分块数严格小于 M 的个数 cntLess，occ=mid-cntLess，再分块顺序找第 occ 个 == M。
        int64_t cntLess = 0;
        for (uint32_t off = 0; off < L_; off += BIG_CHUNK) {
            uint32_t clen = (L_ - off < BIG_CHUNK) ? (L_ - off) : BIG_CHUNK;
            LocalTensor<T> a = LoadChunkBig(r, off, clen);
            uint32_t c4 = clen & ~3u, j = 0;
            for (; j < c4; j += 4) {
                int64_t x0 = (int64_t)a.GetValue(j), x1 = (int64_t)a.GetValue(j + 1);
                int64_t x2 = (int64_t)a.GetValue(j + 2), x3 = (int64_t)a.GetValue(j + 3);
                cntLess += (int64_t)(x0 < lo) + (int64_t)(x1 < lo) + (int64_t)(x2 < lo) + (int64_t)(x3 < lo);
            }
            for (; j < clen; ++j) {
                if ((int64_t)a.GetValue(j) < lo)
                    cntLess++;
            }
        }
        int64_t occ = (int64_t)mid_ - cntLess;
        if (occ < 0)
            occ = 0;
        idx = 0;
        int64_t seen = 0;
        bool found = false;
        for (uint32_t off = 0; off < L_ && !found; off += BIG_CHUNK) {
            uint32_t clen = (L_ - off < BIG_CHUNK) ? (L_ - off) : BIG_CHUNK;
            LocalTensor<T> a = LoadChunkBig(r, off, clen);
            for (uint32_t j = 0; j < clen; ++j) {
                if ((int64_t)a.GetValue(j) == lo) {
                    if (seen == occ) {
                        idx = (int32_t)(off + j);
                        found = true;
                        break;
                    }
                    ++seen;
                }
            }
        }
    }

    __aicore__ inline void Sift(LocalTensor<T>& h, int n, int i)
    {
        // 有左孩子才可能下沉；每轮 i 严格增大（i=lg>i），2*i+1 单调增 → 必然 ≥n 退出（≤log2(n) 次），保证安全退出
        while (2 * i + 1 < n) {
            int lg = i, l = 2 * i + 1, rr = 2 * i + 2;
            if (l < n && h.GetValue(l) > h.GetValue(lg))
                lg = l;
            if (rr < n && h.GetValue(rr) > h.GetValue(lg))
                lg = rr;
            if (lg == i)
                break;
            Swap(h, i, lg);
            SwapIdx(i, lg);
            i = lg;
        }
    }

    __aicore__ inline void Swap(LocalTensor<T>& h, int a, int b)
    {
        T t = h.GetValue(a);
        h.SetValue(a, h.GetValue(b));
        h.SetValue(b, t);
    }
    __aicore__ inline void SwapIdx(int a, int b)
    {
        int16_t t = ixBuf_.GetValue(a);
        ixBuf_.SetValue(a, ixBuf_.GetValue(b));
        ixBuf_.SetValue(b, t);
    }

    __aicore__ inline void CopyOutOne(uint32_t r, T v, int32_t idx)
    {
        LocalTensor<T> outV = outValQ_.AllocTensor<T>();
        LocalTensor<int32_t> outI = outIdxQ_.AllocTensor<int32_t>();
        outV.SetValue(0, v);
        outI.SetValue(0, idx);
        outValQ_.EnQue(outV);
        outIdxQ_.EnQue(outI);
        outV = outValQ_.DeQue<T>();
        outI = outIdxQ_.DeQue<int32_t>();
        DataCopyExtParams cpv{1, (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyExtParams cpi{1, 4u, 0, 0, 0};
        DataCopyPad(valGm_[r], outV, cpv);
        DataCopyPad(idxGm_[r], outI, cpi);
        outValQ_.FreeTensor(outV);
        outIdxQ_.FreeTensor(outI);
    }

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outValQ_;
    TQue<QuePosition::VECOUT, 1> outIdxQ_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> bigBuf_; // 大 L i64 分块 buffer
    LocalTensor<int16_t> ixBuf_;
    GlobalTensor<T> inGm_, valGm_;
    GlobalTensor<int32_t> idxGm_;
    uint32_t rb_, re_, L_, mid_;
    bool useBig_;
};

// ============================================================
// MedianBigSort<T>：fp 大 L 多核分段排序（nSeg > 0 触发）
// stage1：每核排自己 segment，写 sorted value 到 GM workspace
// stage2：lead 核（seg==0）读全行 segments，跨段 value 二分选 mid
// ============================================================
template <typename T>
class MedianBigSort {
public:
    __aicore__ inline MedianBigSort() {}

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR in, GM_ADDR val, GM_ADDR idxo, GM_ADDR wsp, uint32_t batch,
                                uint32_t L, uint32_t mid, uint32_t nSeg, uint32_t bid, uint32_t bn)
    {
        if (nSeg == 0)
            nSeg = 1; // 防御：nSeg 恒 > 0（tiling 保证），clamp 消除静态检查除零告警，不改行为
        pipe_ = pipeIn;
        batch_ = batch;
        L_ = L;
        mid_ = mid;
        nSeg_ = nSeg;
        bid_ = bid;
        bn_ = bn;

        row_ = bid / nSeg;
        seg_ = bid % nSeg;
        // L 小（blocks≤nSeg）：每核 1 个对齐 8192 块（满块向量计算无碎片，barrier 主导时减核数）；
        // L 大（blocks>nSeg=40）：满核非对齐 ceil(L/40)（计算主导时并行度优先）。与 tiling 一致。
        uint32_t blocks = (L + 8191) / 8192;
        if (blocks <= nSeg)
            segLen_ = 8192;
        else
            segLen_ = (L + nSeg - 1) / nSeg;
        segStart_ = seg_ * segLen_;
        myLen_ = 0;
        if (row_ < batch && segStart_ < L) {
            myLen_ = segLen_;
            if (segStart_ + myLen_ > L)
                myLen_ = L - segStart_;
        }
        // 多核值二分计数：无需排序/Pad，每核扫自己段 [segStart_, segStart_+myLen_)
        inGm_.SetGlobalBuffer((__gm__ T*)in);
        valGm_.SetGlobalBuffer((__gm__ T*)val);
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)idxo);

        SetSysWorkspace(wsp);
        GM_ADDR uws = GetUserWorkspace(wsp);
        // workspace(float)：gMM[0,2*nSeg) 每核 min(2s)/max(2s+1)；gCnt[2*nSeg,+40*nSeg) 每 iter 每核计数；
        //                   gIdx(int32) 紧随；gmSync 再后
        gVal_.SetGlobalBuffer((__gm__ float*)uws);
        uint64_t idxOff = (uint64_t)(2 * nSeg + 40 * nSeg);
        gIdx_.SetGlobalBuffer((__gm__ int32_t*)uws + idxOff, nSeg);
        uint64_t syncOff = idxOff + nSeg + 64;
        gmSync_.SetGlobalBuffer((__gm__ int32_t*)uws + syncOff, 512);

        constexpr uint32_t halfBytes = (sizeof(T) == 1) ? (BIG_CHUNK * sizeof(half)) : 0;
        pipe_->InitBuffer(inBuf_, BIG_CHUNK * sizeof(T));
        pipe_->InitBuffer(outValQ_, 1, 32);
        pipe_->InitBuffer(outIdxQ_, 1, 32);
        pipe_->InitBuffer(workBuf_,
                          BIG_CHUNK * sizeof(float)                     // sc
                              + BIG_RDW * sizeof(float)                 // rdw
                              + 256                                     // red(64) 区
                              + (BIG_CHUNK / 8 + 32)                    // mk
                              + 32                                      // ubSync(8 int)
                              + halfBytes + BIG_CHUNK * sizeof(float)); // cntScratch（K 路计数 Select 暂存，保留 sc_）
    }

    __aicore__ inline void Process()
    {
        sc_ = workBuf_.GetWithOffset<float>(BIG_CHUNK, 0);
        rdw_ = workBuf_.GetWithOffset<float>(BIG_RDW, BIG_CHUNK * sizeof(float));
        red_ = workBuf_.GetWithOffset<float>(64, (BIG_CHUNK + BIG_RDW) * sizeof(float));
        mk_ = workBuf_.GetWithOffset<uint8_t>(BIG_CHUNK / 8 + 32, (BIG_CHUNK + BIG_RDW) * sizeof(float) + 256);
        ubSync_ = workBuf_.GetWithOffset<int32_t>(8,
                                                  (BIG_CHUNK + BIG_RDW) * sizeof(float) + 256 + (BIG_CHUNK / 8 + 32));
        if constexpr (sizeof(T) == 1) {
            tmpHalf_ = workBuf_.GetWithOffset<half>(
                BIG_CHUNK, (BIG_CHUNK + BIG_RDW) * sizeof(float) + 256 + (BIG_CHUNK / 8 + 32) + 32);
        }
        constexpr uint32_t csOff = (BIG_CHUNK + BIG_RDW) * sizeof(float) + 256 + (BIG_CHUNK / 8 + 32) + 32 +
                                   ((sizeof(T) == 1) ? (BIG_CHUNK * sizeof(half)) : 0);
        cntScratch_ = workBuf_.GetWithOffset<float>(BIG_CHUNK, csOff);

        ResetSyncSlot();

        // ===== Phase 1: range scan（每核扫自己段 → 局部 min/max → 跨核全局 lo/hi） =====
        float lmin = POS, lmax = NEG;
        for (uint32_t off = 0; off < myLen_; off += BIG_CHUNK) {
            uint32_t clen = (myLen_ - off < BIG_CHUNK) ? (myLen_ - off) : BIG_CHUNK;
            LoadChunk(off, clen);
            ReduceMin<float>(red_, sc_, rdw_, clen);
            ReduceMax<float>(red_[8], sc_, rdw_, clen);
            PipeBarrier<PIPE_V>();
            float mn = red_.GetValue(0), mx = red_[8].GetValue(0);
            if (mn < lmin)
                lmin = mn;
            if (mx > lmax)
                lmax = mx;
        }
        WriteF(seg_ * 2, lmin);
        WriteF(seg_ * 2 + 1, lmax);
        SyncAll<true>(gmSync_, ubSync_, bn_);
        float lo, hi;
        ReadMinMax(lo, hi);

        // ===== Phase 2: K 路值划分（每核计 NP=K-1 个累计计数 → 跨核求和 → 区间缩到 1/K） =====
        //   每轮 1 次 barrier 缩小值域 K 倍（vs 二分缩 2 倍）→ barrier 数 ~20→~6（软件 SyncAll
        //   是 005/008 主瓶颈）。绝对早停：(hi-lo)≤kEpsAbs（value 仅需 ±1e-3，留 2x 余量；BigSort
        //   只服务全局 Median，输出 value，无 index → 可放宽精度）。全核 lo/hi 一致 → 同步 break。
        constexpr int KW = 8;            // 路数
        constexpr int NP = KW - 1;       // 内部探针数
        constexpr float kEpsAbs = 4e-4f; // |v-M| ≤ kEpsAbs < 1e-3 阈值
        float mid1 = (float)((int)mid_ + 1);
        const uint32_t cntRegA = 2 * nSeg_;              // ping
        const uint32_t cntRegB = 2 * nSeg_ + nSeg_ * NP; // pong（≤ 40*nSeg 预留区内）
        for (int rd = 0; rd < 10; ++rd) {
            if ((hi - lo) <= kEpsAbs)
                break; // 收敛早停
            float oldlo = lo;
            float step = (hi - oldlo) * (1.0f / (float)KW);
            float cnts[NP];
            for (int j = 0; j < NP; ++j)
                cnts[j] = 0.0f;
            for (uint32_t off = 0; off < myLen_; off += BIG_CHUNK) {
                uint32_t clen = (myLen_ - off < BIG_CHUNK) ? (myLen_ - off) : BIG_CHUNK;
                LoadChunk(off, clen);
                uint32_t cl = ((clen + 63) / 64) * 64;
                for (int j = 0; j < NP; ++j) { // sc_ 整段常驻 → NP 个探针共用
                    float pj = oldlo + step * (float)(j + 1);
                    CompareScalar(mk_, sc_, pj, CMPMODE::LE, cl);
                    Duplicate<float>(cntScratch_, 1.0f, clen);
                    Select(cntScratch_, mk_, cntScratch_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, clen);
                    ReduceSum<float>(red_, cntScratch_, rdw_, clen);
                    PipeBarrier<PIPE_V>();
                    cnts[j] += red_.GetValue(0);
                }
            }
            uint32_t base = (rd & 1) ? cntRegB : cntRegA;
            for (int j = 0; j < NP; ++j)
                red_.SetValue(16 + j, cnts[j]);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyExtParams cpw{1, (uint32_t)(NP * 4), 0, 0, 0};
            DataCopyPad(gVal_[base + seg_ * NP], red_[16], cpw);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SyncAll<true>(gmSync_, ubSync_, bn_);
            // 读 nSeg*NP 计数 → 每探针跨核求和，找首个 ≥mid1 的探针
            DataCopyExtParams cpr{1, (uint32_t)(nSeg_ * NP * 4), 0, 0, 0};
            DataCopyPadExtParams<float> dpr{false, 0, 0, 0};
            DataCopyPad(sc_, gVal_[base], cpr, dpr);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            int jc = -1;
            for (int j = 0; j < NP; ++j) {
                float c = 0;
                for (uint32_t s = 0; s < nSeg_; ++s)
                    c += sc_.GetValue(s * NP + j);
                if (c >= mid1) {
                    jc = j;
                    break;
                } // 累计计数单调 → 首越界即解
            }
            if (jc < 0) {
                lo = oldlo + step * (float)NP; // 全未越界 → median 在 (probe_NP, hi]
            } else {
                lo = oldlo + step * (float)jc;       // probe_{jc-1}（jc=0 时即 oldlo）
                hi = oldlo + step * (float)(jc + 1); // probe_jc
            }
        }

        // ===== Phase 3: value find + 全局最小 idx =====
        float v = hi;
        int32_t lidx = 0x7fffffff;
        bool found = false;
        for (uint32_t off = 0; off < myLen_ && !found; off += BIG_CHUNK) {
            uint32_t clen = (myLen_ - off < BIG_CHUNK) ? (myLen_ - off) : BIG_CHUNK;
            LoadChunk(off, clen);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < clen; ++j) {
                if (sc_.GetValue(j) == v) {
                    lidx = (int32_t)(segStart_ + off + j);
                    found = true;
                    break;
                }
            }
        }
        WriteIdx(seg_, lidx);
        SyncAll<true>(gmSync_, ubSync_, bn_);
        if (seg_ == 0 && row_ < batch_) {
            int32_t gmin = ReadMinIdx(nSeg_);
            CopyOutOne(row_, v, gmin);
        }
    }

private:
    __aicore__ inline void ResetSyncSlot()
    {
        Duplicate<int32_t>(ubSync_, 0, 8);
        DataCopyExtParams cz{1, 32u, 0, 0, 0};
        DataCopyPad(gmSync_[bid_ * 8], ubSync_, cz);
        // 等 MTE3 完 → MTE2 才能读（跨核同步前 wash 一下）
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    // 载入一段 chunk 到 sc_（fp32）；off 相对本核段起点
    __aicore__ inline void LoadChunk(uint32_t off, uint32_t clen)
    {
        LocalTensor<T> raw = inBuf_.Get<T>();
        DataCopyExtParams cp{1, clen * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> dp{false, 0, 0, 0};
        DataCopyPad(raw, inGm_[(uint64_t)row_ * L_ + segStart_ + off], cp, dp);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr (std::is_same_v<T, float>) {
            Adds<float>(sc_, raw.template ReinterpretCast<float>(), 0.0f, clen);
        } else {
            CastInRow<T>(tmpHalf_, sc_, raw, clen);
        }
    }

    // 写 1 个 float 到 gVal_[off]（red_[16] 暂存，避开 reduce 用的 [0]/[8]）
    __aicore__ inline void WriteF(uint32_t off, float v)
    {
        red_[16].SetValue(0, v);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, 4u, 0, 0, 0};
        DataCopyPad(gVal_[off], red_[16], cp);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    // 读 gVal_[0, 2*nSeg) → 全局 lo/hi
    __aicore__ inline void ReadMinMax(float& lo, float& hi)
    {
        DataCopyExtParams cp{1, 2 * nSeg_ * 4u, 0, 0, 0};
        DataCopyPadExtParams<float> dp{false, 0, 0, 0};
        DataCopyPad(sc_, gVal_[0], cp, dp);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        lo = POS;
        hi = NEG;
        for (uint32_t s = 0; s < nSeg_; ++s) {
            float mn = sc_.GetValue(2 * s), mx = sc_.GetValue(2 * s + 1);
            if (mn < lo)
                lo = mn;
            if (mx > hi)
                hi = mx;
        }
    }

    // 读 gVal_[off, off+n) 求和
    __aicore__ inline float ReadSum(uint32_t off, uint32_t n)
    {
        DataCopyExtParams cp{1, n * 4u, 0, 0, 0};
        DataCopyPadExtParams<float> dp{false, 0, 0, 0};
        DataCopyPad(sc_, gVal_[off], cp, dp);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        float s = 0;
        for (uint32_t i = 0; i < n; ++i)
            s += sc_.GetValue(i);
        return s;
    }

    // 写 1 个 int32 到 gIdx_[s]（red_ 复用区第 16 int32 暂存）
    __aicore__ inline void WriteIdx(uint32_t s, int32_t v)
    {
        LocalTensor<int32_t> ib = red_.template ReinterpretCast<int32_t>();
        ib[16].SetValue(0, v);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, 4u, 0, 0, 0};
        DataCopyPad(gIdx_[s], ib[16], cp);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    // 读 gIdx_[0, n) 取全局最小（= median value 的最小下标）
    __aicore__ inline int32_t ReadMinIdx(uint32_t n)
    {
        LocalTensor<int32_t> ib = sc_.template ReinterpretCast<int32_t>();
        DataCopyExtParams cp{1, n * 4u, 0, 0, 0};
        DataCopyPadExtParams<int32_t> dp{false, 0, 0, 0};
        DataCopyPad(ib, gIdx_[0], cp, dp);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int32_t mn = 0x7fffffff;
        for (uint32_t i = 0; i < n; ++i) {
            int32_t x = ib.GetValue(i);
            if (x < mn)
                mn = x;
        }
        return mn;
    }

    // lead 核（seg==0）写最终 (value, idx)
    __aicore__ inline void CopyOutOne(uint32_t r, float v, int32_t idx)
    {
        LocalTensor<T> outV = outValQ_.AllocTensor<T>();
        LocalTensor<int32_t> outI = outIdxQ_.AllocTensor<int32_t>();
        if constexpr (std::is_same_v<T, float>) {
            outV.template ReinterpretCast<float>().SetValue(0, v);
        } else {
            red_[16].SetValue(0, v);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            CastOutRow<T>(tmpHalf_, outV, red_[16], 1);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
        outI.SetValue(0, idx);
        outValQ_.EnQue(outV);
        outIdxQ_.EnQue(outI);
        outV = outValQ_.DeQue<T>();
        outI = outIdxQ_.DeQue<int32_t>();
        DataCopyExtParams co{1, (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyExtParams ci{1, 4u, 0, 0, 0};
        DataCopyPad(valGm_[r], outV, co);
        DataCopyPad(idxGm_[r], outI, ci);
        outValQ_.FreeTensor(outV);
        outIdxQ_.FreeTensor(outI);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> inBuf_; // chunk load buffer
    TQue<QuePosition::VECOUT, 1> outValQ_;
    TQue<QuePosition::VECOUT, 1> outIdxQ_;
    TBuf<TPosition::VECCALC> workBuf_;

    GlobalTensor<T> inGm_, valGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<float> gVal_;
    GlobalTensor<int32_t> gIdx_;
    GlobalTensor<int32_t> gmSync_;

    LocalTensor<float> sc_, rdw_, red_, cntScratch_;
    LocalTensor<uint8_t> mk_;
    LocalTensor<int32_t> ubSync_;
    LocalTensor<half> tmpHalf_;

    uint32_t batch_, L_, mid_, nSeg_, bid_, bn_;
    uint32_t row_, seg_, segLen_, segStart_, myLen_, Pad_, groups_;
};

} // namespace NsMedian

#endif // MEDIAN_KERNEL_H
