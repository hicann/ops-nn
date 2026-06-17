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
 * \file euclidean_norm.h
 * \brief EuclideanNorm 算子 kernel 类实现（v2：支持多轴 UB bundle）。
 *
 * 数学：y = sqrt( sum( x^2 ) along axes )
 *
 * 几何骨架：normal 模板。
 *   - aSplit / rSplit 可向外 climb（tiling 阶段算好），UB 内可含多根 A/R 轴
 *   - CopyIn 用 DataCopyPad + LoopMode（loop1/loop2）覆盖最内 4 根 UB 轴；
 *     超过 4 根时外层 host for 拆掉余下的
 *   - 所有 UB 元素数公式统一用 aUbFactorAlign × innerAProdAlign × rUbFactor × innerRProdAlign
 *
 * 4 dtype 路径（编译期 if constexpr 分发）：
 *   fp16 / bf16：LoadAlign DIST_UNPACK_B16 + Cast→fp32 / fp32→Cast + StoreAlign DIST_PACK_B32
 *   fp32：     LoadAlign / StoreAlign 直通
 *   int32：    LoadAlign + Cast→fp32 / fp32→Cast(trunc) + StoreAlign
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_H_
#define OPS_NORM_EUCLIDEAN_NORM_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/reduce/reduce.h"
#include "euclidean_norm_tiling_data.h"
#include "euclidean_norm_tiling_key.h"

namespace NsEuclideanNorm {

using namespace AscendC;

// ─── 常量 ───
constexpr uint32_t kVlBytes = 256;                     // VL = 256B
constexpr uint32_t kRepF32 = kVlBytes / sizeof(float); // = 64
constexpr uint16_t kRepF32U = static_cast<uint16_t>(kRepF32);
constexpr uint32_t kBlockBytes = 32;                        // 32B
constexpr uint32_t kBlockF32 = kBlockBytes / sizeof(float); // = 8

// UB 内一根轴的描述（innermost-first 排列）
struct UBAxisDesc {
    int32_t gmIdx;      // 该轴在 tilingdata.axisShape / axisStride 中的下标
    int64_t ubSize;     // 该轴在 UB 中的有效长度（split 轴为 aLen/rLen，其它为整根 axisShape）
    int64_t paddedSize; // 该轴在 UB 中的占位长度（含 LastA/LastR 行宽 CeilAlign / split 轴 rUbFactor·aUbFactor[Align]）
    int64_t gmStride;   // 该轴在 GM 上的 stride（元素，来自 axisStride）
};

template <typename DType, bool isTailR>
class EuclideanNormKernel {
public:
    using D_T = DType;
    using ComputeT = float;

    static constexpr bool kIsFp32 = std::is_same_v<D_T, float>;
    static constexpr bool kIsInt32 = std::is_same_v<D_T, int32_t>;
    static constexpr bool kIsB16 = (sizeof(D_T) == 2); // fp16 / bf16
    static constexpr bool kNeedCast = !kIsFp32;

    // CastTrait：X → fp32（扩位，硬件不读 sat / round，按 Cast_regbase_detail 推荐模板用 UNKNOWN + CAST_NONE）
    static constexpr AscendC::Reg::CastTrait kCastTraitToFp32{
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_NONE};
    // CastTrait：fp32 → X（硬件不允许 CAST_NONE）
    //   int32：CAST_TRUNC（与 TensorFlow reduce_euclidean_norm(int32) trunc-to-zero 一致）
    //   fp16/bf16：CAST_RINT
    static constexpr AscendC::Reg::CastTrait kCastTraitFromFp32{
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        kIsInt32 ? AscendC::RoundMode::CAST_TRUNC : AscendC::RoundMode::CAST_RINT};

    __aicore__ inline EuclideanNormKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const EuclideanNormTilingData* td);
    __aicore__ inline void Process();

    // ─── group 模板（public：供 kernel 入口 apt.cpp 调用）───
    __aicore__ inline void InitGroup(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const EuclideanNormTilingData* td);
    __aicore__ inline void ProcessGroup();

private:
    // ─── 主流程 ───
    __aicore__ inline void DoOneAChunk(int64_t outerGmOff, int64_t aLen);
    __aicore__ inline void PostElewiseAndCopyOut(int64_t outerOutOff, int64_t aLen);

    // ─── group 模板（内部 helper）───
    __aicore__ inline void DoOneAChunkGroup(int64_t outerGmOff, int64_t aLen, int64_t rStart, int64_t rEnd);
    __aicore__ inline void Phase1OutputToWorkspace(int64_t wsColOff, int64_t aLen, int64_t rChunkIdx, int64_t rCount);
    __aicore__ inline void Phase2Kernel();

    // ─── CopyIn ───
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]);
    __aicore__ inline void DoCopyInTile(
        int64_t baseGmOff, int64_t aLen, int64_t rLen, AscendC::LocalTensor<D_T>& preInLocal);

    // ─── VF ───
    __aicore__ inline void CastSquareVf(AscendC::LocalTensor<D_T>& preIn, uint32_t slot);
    __aicore__ inline void ClearChunkExtensionVf(uint32_t slot, int64_t rLen);
    __aicore__ inline void MergeTmpBufVf();
    __aicore__ inline void ReduceRPattern();
    __aicore__ inline void ClearCacheTreeVf();
    __aicore__ inline void DoCachingVf(uint16_t cacheID);

    // ─── 辅助 ───
    __aicore__ inline int32_t LastAAxis() const;
    __aicore__ inline int32_t LastRAxis() const;
    __aicore__ inline int64_t RLenOfChunk(int64_t rChunkIdx) const;
    __aicore__ inline int64_t UnravelOuterR(int64_t rIdx, int64_t& rChunkIdxOut, int64_t& rLenOut) const;
    __aicore__ inline uint16_t GetCacheID(int64_t idx) const;
    __aicore__ inline uint64_t FindNearestPower2(uint64_t v) const;
    __aicore__ inline uint64_t CalLog2(uint64_t v) const;
    __aicore__ inline uint32_t LastBurstAlignElems() const;

    // ─── tilingdata 镜像 ───
    int32_t axisNum_ = 0;
    int64_t axisShape_[MAX_PATTERN_RANK] = {0};
    int64_t axisStride_[MAX_PATTERN_RANK] = {0};
    int32_t aSplit_ = 0;
    int32_t rSplit_ = 0;
    // 多核（fused aLoop）
    int64_t aLoopCntTotal_ = 0;
    int64_t aSplitChunkCnt_ = 0;
    int64_t aBigCoreLoopCnt_ = 0;
    int64_t aSmallCoreLoopCnt_ = 0;
    int32_t aBigCoreCnt_ = 0;
    int32_t usedCoreNum_ = 0;
    int64_t aUbFactor_ = 0;
    int64_t aUbFactorAlign_ = 0;
    int64_t rUbFactor_ = 0;      // valid R count（GM dense 步距 / RLenOfChunk）
    int64_t rUbFactorAlign_ = 0; // padded R count（UB stride / ReduceSum srcShape[1] / totalElems）
    int64_t innerAProd_ = 0;
    int64_t innerAProdAlign_ = 0;
    int64_t innerRProd_ = 0;
    int64_t innerRProdAlign_ = 0;
    int64_t rLoopCntTotal_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheCount_ = 0;
    int64_t preReduceUbSize_ = 0;  // = tilingData.preReduceUbSize，Phase 2 kernel 侧反推 aUbFactor 用
    int64_t postReduceUbSize_ = 0; // = tilingData.postReduceUbSize
    int64_t tmpSlotElems_ = 0;
    int64_t cacheBufElems_ = 0; // = cacheBufUbSize / sizeof(float)，用于 ClearCacheTreeVf

    // 输出端 stride（A 轴 dense 排列）
    int64_t outStride_[MAX_PATTERN_RANK] = {0};

    // ─── GM + UB ───
    GlobalTensor<D_T> xGm_;
    GlobalTensor<D_T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> preInQue_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> cacheBuf_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    // ─── group 模板 ───
    GlobalTensor<float> wsGm_; // workspace GM 基址（fp32，Phase 1 写、Phase 2 读）
    int64_t rGroupCnt_ = 0;    // Phase 1 分组数 = Phase 2 workspace R 维
    int64_t aTotal_ = 0;       // ∏ 所有 A 轴 axisShape（workspace 列数）
};

// ════════════════════════════════════════════════════════════════════════════
// 工具函数
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline int32_t EuclideanNormKernel<DType, isTailR>::LastAAxis() const
{
    for (int32_t i = axisNum_ - 1; i >= 0; --i) {
        if (i % 2 == 0) {
            return i;
        }
    }
    return 0;
}

template <typename DType, bool isTailR>
__aicore__ inline int32_t EuclideanNormKernel<DType, isTailR>::LastRAxis() const
{
    for (int32_t i = axisNum_ - 1; i >= 0; --i) {
        if (i % 2 == 1) {
            return i;
        }
    }
    return 1;
}

template <typename DType, bool isTailR>
__aicore__ inline uint32_t EuclideanNormKernel<DType, isTailR>::LastBurstAlignElems() const
{
    // 最内层 UB 轴（burst tail）的 padded UB 行宽（元素）
    //   tail R: burst tail = LastR；若 rSplit == LastR，size = rUbFactor，否则 = axisShape[LastR]
    //   tail A: burst tail = LastA；若 aSplit == LastA，size = aUbFactor，否则 = axisShape[LastA]
    int64_t sz;
    if constexpr (isTailR) {
        const int32_t lastR = LastRAxis();
        sz = (rSplit_ == lastR) ? rUbFactor_ : axisShape_[lastR];
    } else {
        const int32_t lastA = LastAAxis();
        sz = (aSplit_ == lastA) ? aUbFactor_ : axisShape_[lastA];
    }
    const int64_t bsElem = static_cast<int64_t>(kBlockBytes) / static_cast<int64_t>(sizeof(D_T));
    return static_cast<uint32_t>((sz + bsElem - 1) / bsElem * bsElem);
}

template <typename DType, bool isTailR>
__aicore__ inline uint64_t EuclideanNormKernel<DType, isTailR>::FindNearestPower2(uint64_t v) const
{
    if (v == 0) {
        return 0;
    }
    if (v <= 2) {
        return 1;
    }
    if (v <= 4) {
        return 2;
    }
    const uint64_t num = v - 1;
    const uint64_t pow = 63 - AscendC::ScalarCountLeadingZero(num);
    return static_cast<uint64_t>(1) << pow;
}

template <typename DType, bool isTailR>
__aicore__ inline uint64_t EuclideanNormKernel<DType, isTailR>::CalLog2(uint64_t v) const
{
    uint64_t res = 0;
    while (v > 1) {
        v >>= 1;
        ++res;
    }
    return res;
}

template <typename DType, bool isTailR>
__aicore__ inline uint16_t EuclideanNormKernel<DType, isTailR>::GetCacheID(int64_t idx) const
{
    const uint64_t v = static_cast<uint64_t>(idx);
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(v ^ (v + 1)) - 1);
}

template <typename DType, bool isTailR>
__aicore__ inline int64_t EuclideanNormKernel<DType, isTailR>::RLenOfChunk(int64_t rChunkIdx) const
{
    const int64_t rAxisSize = axisShape_[rSplit_];
    const int64_t start = rChunkIdx * rUbFactor_;
    return (start + rUbFactor_ > rAxisSize) ? (rAxisSize - start) : rUbFactor_;
}

// rIdx → 多维 (外层 R axis indices, rSplit 上的 chunk idx)；返回外层 R 贡献的 GM 偏移
template <typename DType, bool isTailR>
__aicore__ inline int64_t EuclideanNormKernel<DType, isTailR>::UnravelOuterR(
    int64_t rIdx, int64_t& rChunkIdxOut, int64_t& rLenOut) const
{
    const int64_t rChunksOnSplit = (axisShape_[rSplit_] + rUbFactor_ - 1) / rUbFactor_;
    rChunkIdxOut = rIdx % rChunksOnSplit;
    int64_t cur = rIdx / rChunksOnSplit;
    int64_t gmOff = 0;
    // 外层 R 轴（idx < rSplit_ 且 axisType=R），从内到外遍历做 unravel
    for (int32_t i = rSplit_ - 1; i >= 0; --i) {
        if (i % 2 == 1) {
            const int64_t sz = axisShape_[i];
            const int64_t ix = cur % sz;
            cur /= sz;
            gmOff += ix * axisStride_[i];
        }
    }
    rLenOut = RLenOfChunk(rChunkIdxOut);
    return gmOff + rChunkIdxOut * rUbFactor_ * axisStride_[rSplit_];
}

// ════════════════════════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::Init(
    GM_ADDR x, GM_ADDR y, const EuclideanNormTilingData* td)
{
    axisNum_ = td->axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        axisShape_[i] = td->axisShape[i];
        axisStride_[i] = td->axisStride[i];
    }
    aSplit_ = td->aSplitAxisIdx;
    rSplit_ = td->rSplitAxisIdx;
    aLoopCntTotal_ = td->aLoopCntTotal;
    aSplitChunkCnt_ = td->aSplitChunkCnt;
    aBigCoreLoopCnt_ = td->aBigCoreLoopCnt;
    aSmallCoreLoopCnt_ = td->aSmallCoreLoopCnt;
    aBigCoreCnt_ = td->aBigCoreCnt;
    usedCoreNum_ = td->usedCoreNum;
    aUbFactor_ = td->aUbFactor;
    aUbFactorAlign_ = td->aUbFactorAlign;
    rUbFactor_ = td->rUbFactor;
    rUbFactorAlign_ = td->rUbFactorAlign;
    innerAProd_ = td->innerAProd;
    innerAProdAlign_ = td->innerAProdAlign;
    innerRProd_ = td->innerRProd;
    innerRProdAlign_ = td->innerRProdAlign;
    rLoopCntTotal_ = td->rLoopCntTotal;
    preReduceUbSize_ = td->preReduceUbSize;
    postReduceUbSize_ = td->postReduceUbSize;
    tmpSlotElems_ = td->tmpBufUbSize / static_cast<int64_t>(sizeof(float));
    cacheBufElems_ = td->cacheBufUbSize / static_cast<int64_t>(sizeof(float));

    bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rLoopCntTotal_)));
    bisectionTail_ = rLoopCntTotal_ - bisectionPos_;
    cacheCount_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_))) + 1;

    // 输出端 stride：output 上仅 A 轴 dense 排列、顺序与 input 一致
    {
        int64_t acc = 1;
        for (int32_t i = axisNum_ - 1; i >= 0; --i) {
            if (i % 2 == 0) {
                outStride_[i] = acc;
                acc *= axisShape_[i];
            }
        }
    }

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(y));

    pipe_.InitBuffer(preInQue_, /*bufNum=*/2, td->preReduceUbSize);
    pipe_.InitBuffer(tmpBuf_, td->tmpBufUbSize * 2);
    pipe_.InitBuffer(cacheBuf_, td->cacheBufUbSize);
    pipe_.InitBuffer(outQue_, /*bufNum=*/1, td->postReduceUbSize);
}

// ════════════════════════════════════════════════════════════════════════════
// Process: fused aLoop 多核解码 + aLoopIdx unravel
//
// blockIdx → [aLoopStart, aLoopEnd)：
//   前 aBigCoreCnt 个核处理 aBigCoreLoopCnt 个 aLoop；
//   其余核处理 aSmallCoreLoopCnt 个 aLoop；
//   blockIdx ≥ usedCoreNum 早退（idle）。
//
// aLoopIdx unravel（行 major，aSplit chunk 在最内）：
//   aSplitChunkIdx = aLoopIdx % aSplitChunkCnt
//   rem            = aLoopIdx / aSplitChunkCnt
//   for k in [aSplit-2, 0] step -2:  aIdx[k] = rem % axisShape[k]; rem /= axisShape[k]
//
// (aIdx[], aSplitChunkIdx) → GM/Out offset → DoOneAChunk + PostElewiseAndCopyOut
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
        return;
    }

    // ─── blockIdx → [aLoopStart, aLoopEnd) ───
    int64_t aLoopStart;
    int64_t aLoopEnd;
    if (blockIdx < static_cast<int64_t>(aBigCoreCnt_)) {
        aLoopStart = blockIdx * aBigCoreLoopCnt_;
        aLoopEnd = aLoopStart + aBigCoreLoopCnt_;
    } else {
        aLoopStart = static_cast<int64_t>(aBigCoreCnt_) * aBigCoreLoopCnt_ +
                     (blockIdx - static_cast<int64_t>(aBigCoreCnt_)) * aSmallCoreLoopCnt_;
        aLoopEnd = aLoopStart + aSmallCoreLoopCnt_;
    }

    const int64_t aSplitAxisSize = axisShape_[aSplit_];
    const int64_t aSplitStride = axisStride_[aSplit_];
    const int64_t aSplitOutStr = outStride_[aSplit_];

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        // ─── unravel aLoopIdx → (aIdx[outer A], aSplitChunkIdx) ───
        int64_t rem = aLoopIdx;
        const int64_t aSplitChunkIdx = rem % aSplitChunkCnt_;
        rem /= aSplitChunkCnt_;

        int64_t chunkGmOff = 0;
        int64_t chunkOutOff = 0;
        // 从 aSplit 的左邻 A 向外侧依次拆（步长 -2 越过 R 轴）
        for (int32_t k = aSplit_ - 2; k >= 0; k -= 2) {
            const int64_t sz = axisShape_[k];
            const int64_t ix = rem % sz;
            rem /= sz;
            chunkGmOff += ix * axisStride_[k];
            chunkOutOff += ix * outStride_[k];
        }

        const int64_t aChunkStart = aSplitChunkIdx * aUbFactor_;
        const int64_t aEnd = aChunkStart + aUbFactor_;
        const int64_t aLen = (aEnd > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : aUbFactor_;
        if (aLen <= 0) {
            continue; // aLoopCntTotal 严格上界，正常不触发；防御性兜底
        }

        chunkGmOff += aChunkStart * aSplitStride;
        chunkOutOff += aChunkStart * aSplitOutStr;

        DoOneAChunk(chunkGmOff, aLen);
        PostElewiseAndCopyOut(chunkOutOff, aLen);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 单 A chunk 完整流程：清缓存树 → Phase A 配对 → Phase B 单块
// ────────────────────────────────────────────────────────────────────────────
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::DoOneAChunk(int64_t outerGmOff, int64_t aLen)
{
    ClearCacheTreeVf();

    // Phase A：主-尾配对
    for (int64_t i = 0; i < bisectionTail_; ++i) {
        // 主块 slot=0
        {
            int64_t rChunk = 0, rLen = 0;
            const int64_t rOff = UnravelOuterR(i, rChunk, rLen);
            auto preIn = preInQue_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, preIn);
            preInQue_.EnQue(preIn);
            auto deq = preInQue_.DeQue<D_T>();
            CastSquareVf(deq, /*slot=*/0U);
            preInQue_.FreeTensor(deq);
            if (rLen < rUbFactor_) {
                ClearChunkExtensionVf(0U, rLen);
            }
        }
        // 尾块 slot=1
        {
            int64_t rChunk = 0, rLen = 0;
            const int64_t rOff = UnravelOuterR(i + bisectionPos_, rChunk, rLen);
            auto preIn = preInQue_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, preIn);
            preInQue_.EnQue(preIn);
            auto deq = preInQue_.DeQue<D_T>();
            CastSquareVf(deq, /*slot=*/1U);
            preInQue_.FreeTensor(deq);
            if (rLen < rUbFactor_) {
                ClearChunkExtensionVf(1U, rLen);
            }
        }
        MergeTmpBufVf();
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
    // Phase B：单块
    for (int64_t i = bisectionTail_; i < bisectionPos_; ++i) {
        int64_t rChunk = 0, rLen = 0;
        const int64_t rOff = UnravelOuterR(i, rChunk, rLen);
        auto preIn = preInQue_.AllocTensor<D_T>();
        DoCopyInTile(outerGmOff + rOff, aLen, rLen, preIn);
        preInQue_.EnQue(preIn);
        auto deq = preInQue_.DeQue<D_T>();
        CastSquareVf(deq, /*slot=*/0U);
        preInQue_.FreeTensor(deq);
        if (rLen < rUbFactor_) {
            ClearChunkExtensionVf(0U, rLen);
        }
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BuildUBAxes：根据 aSplit/rSplit/isTailR 枚举 UB 内轴（innermost→outermost）
//
// 返回值 = UB 内轴数 K（2 ≤ K ≤ axisNum）
// 每个 UBAxisDesc 含 actual ubSize 与 paddedSize：
//   - split 轴（aSplit/rSplit）：ubSize = aLen/rLen，paddedSize = aUbFactor[Align]/rUbFactor
//   - burst-tail 整根（tail-R 下 LastR != rSplit，tail-A 下 LastA != aSplit）：
//        ubSize = axisShape[i]，paddedSize = CeilAlign(axisShape[i], bsElem)
//   - 其他普通轴：ubSize = paddedSize = axisShape[i]
//
// out[0] 是 burst 尾轴（最内层 stride 1），out[K-1] 是 UB 内最外侧轴。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline int32_t EuclideanNormKernel<DType, isTailR>::BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[])
{
    int32_t k = 0;
    const int32_t lastA = LastAAxis();
    const int32_t lastR = LastRAxis();
    const int64_t bsElem = static_cast<int64_t>(kBlockBytes) / static_cast<int64_t>(sizeof(D_T));

    if constexpr (isTailR) {
        // 内 bundle = R bundle（innermost first：从大 idx 到 rSplit）
        for (int32_t i = axisNum_ - 1; i >= rSplit_; --i) {
            if (i % 2 != 1) {
                continue;
            }
            int64_t actual;
            int64_t padded;
            if (i == rSplit_) {
                actual = rLen;
                padded = rUbFactorAlign_; // tail-R + rSplit==lastR: CeilAlign；
                                          // tail-R + rSplit!=lastR: = rUbFactor（rSplit 是外层 R）
            } else if (i == lastR) {
                actual = axisShape_[i];
                padded = (actual + bsElem - 1) / bsElem * bsElem; // 内层 LastR 行宽 block-align
            } else {
                actual = axisShape_[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].ubSize = actual;
            out[k].paddedSize = padded;
            out[k].gmStride = axisStride_[i];
            ++k;
        }
        // 外 bundle = A bundle（tail-R 下 LastA 无 burst-tail 对齐）
        for (int32_t i = axisNum_ - 1; i >= aSplit_; --i) {
            if (i % 2 != 0) {
                continue;
            }
            int64_t actual;
            int64_t padded;
            if (i == aSplit_) {
                actual = aLen;
                padded = aUbFactor_; // tail-R: aUbFactorAlign == aUbFactor
            } else {
                actual = axisShape_[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].ubSize = actual;
            out[k].paddedSize = padded;
            out[k].gmStride = axisStride_[i];
            ++k;
        }
    } else {
        // 内 bundle = A bundle（innermost first：从大 idx 到 aSplit）
        for (int32_t i = axisNum_ - 1; i >= aSplit_; --i) {
            if (i % 2 != 0) {
                continue;
            }
            int64_t actual;
            int64_t padded;
            if (i == aSplit_) {
                actual = aLen;
                padded = (i == lastA) ? aUbFactorAlign_ : aUbFactor_; // tail-A 下若 aSplit==LastA, 用 align 版
            } else if (i == lastA) {
                actual = axisShape_[i];
                padded = (actual + bsElem - 1) / bsElem * bsElem; // 内层 LastA 行宽 block-align
            } else {
                actual = axisShape_[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].ubSize = actual;
            out[k].paddedSize = padded;
            out[k].gmStride = axisStride_[i];
            ++k;
        }
        // 外 bundle = R bundle（tail-A 下 LastR 无 burst-tail 对齐）
        for (int32_t i = axisNum_ - 1; i >= rSplit_; --i) {
            if (i % 2 != 1) {
                continue;
            }
            int64_t actual;
            int64_t padded;
            if (i == rSplit_) {
                actual = rLen;
                padded = rUbFactorAlign_; // tail-A 下 = rUbFactor（rSplit 不是 burst tail）
            } else {
                actual = axisShape_[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].ubSize = actual;
            out[k].paddedSize = padded;
            out[k].gmStride = axisStride_[i];
            ++k;
        }
    }
    return k;
}

// ════════════════════════════════════════════════════════════════════════════
// DoCopyInTile：用 DataCopyPad + LoopMode（可选 host for）把一块 GM tile 装入 preIn
//
// UB 内轴最多 4 根用 DataCopyPad+LoopMode 一次性搞定；超过的部分 host for 拆。
//
// 行宽以 paddedSize × sizeof(D_T) 为节奏（与下游 VF 一致）：
//   - dstStride = (paddedSize[0] × s − CeilAlign(blockLen, 32)) / 32（partial chunk 时 > 0）
//   - loop1/2DstStride 用 ubStride[k] = ∏ paddedSize[0..k-1] × s 累积
//   - host for 同样按 ubStride[k] 步进
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::DoCopyInTile(
    int64_t baseGmOff, int64_t aLen, int64_t rLen, AscendC::LocalTensor<D_T>& preInLocal)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t K = BuildUBAxes(aLen, rLen, ubAxes);

    // ─── 准备 DataCopyPad + LoopMode 参数（仅基于 ubAxes[0..min(K,4))）───
    DataCopyExtParams extParams;
    LoopModeParams loopParams;
    loopParams.loop1Size = 0;
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    loopParams.loop2Size = 0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    // blockLen 不 32B 对齐时必须 rightPadding > 0，否则 dummy 段填首元素值（≠ paddingValue=0）。
    // dstStride 跳过段的 stale 由 ClearChunkExtensionVf 清。
    const int64_t dtBytes = static_cast<int64_t>(sizeof(D_T));
    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].ubSize * dtBytes);

    uint32_t misalign = extParams.blockLen & (kBlockBytes - 1u);
    uint8_t rPad = 0;
    if (misalign != 0) {
        uint32_t gapBytes = kBlockBytes - misalign;
        rPad = static_cast<uint8_t>(gapBytes / static_cast<uint32_t>(dtBytes));
    }
    DataCopyPadExtParams<D_T> padParams{true, 0, rPad, static_cast<D_T>(0)};

    const int64_t copyPadBytes =
        (static_cast<int64_t>(extParams.blockLen) + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
    const int64_t target0Bytes = ubAxes[0].paddedSize * dtBytes;
    extParams.dstStride = static_cast<uint32_t>((target0Bytes - copyPadBytes) / static_cast<int64_t>(kBlockBytes));

    if (K >= 2) {
        extParams.blockCount = static_cast<uint16_t>(ubAxes[1].ubSize);
        extParams.srcStride =
            static_cast<uint32_t>(ubAxes[1].gmStride * dtBytes - static_cast<int64_t>(extParams.blockLen));
    } else {
        extParams.blockCount = 1;
        extParams.srcStride = 0;
    }

    // 每层 ubAxes[i] 在 UB 上的字节步长 = ∏ paddedSize[0..i-1] × sizeof(D_T)
    //   stride[0] = sizeof(D_T)（元素步长）；stride[k] (k≥1) = stride[k-1] × paddedSize[k-1]
    int64_t ubStride[MAX_PATTERN_RANK];
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < K; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedSize;
    }

    if (K >= 3) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[2].ubSize);
        loopParams.loop1SrcStride = static_cast<uint64_t>(ubAxes[2].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop1DstStride = static_cast<uint64_t>(ubStride[2]);
        loopParams.loop2Size = 1; // ★ 哪怕外层不用，也必须 ≥ 1，loop2Size==0 会让整批搬运 0 次
    }
    if (K >= 4) {
        loopParams.loop2Size = static_cast<uint32_t>(ubAxes[3].ubSize);
        loopParams.loop2SrcStride = static_cast<uint64_t>(ubAxes[3].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop2DstStride = static_cast<uint64_t>(ubStride[3]);
    }

    const bool useLoopMode = (K >= 3);
    if (useLoopMode) {
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    }

    // host for 拆 ubAxes[4..K)；如无超过 4 根，outerProd=1，循环一次。
    int64_t outerProd = 1;
    for (int32_t k = 4; k < K; ++k)
        outerProd *= ubAxes[k].ubSize;

    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        int64_t addGmOffElem = 0;
        int64_t addUbOffBytes = 0;
        int64_t cur = outerFlat;
        for (int32_t k = 4; k < K; ++k) {
            const int64_t sz = ubAxes[k].ubSize;
            const int64_t ix = cur % sz;
            cur /= sz;
            addGmOffElem += ix * ubAxes[k].gmStride;
            addUbOffBytes += ix * ubStride[k];
        }
        const int64_t ubOffElems = addUbOffBytes / dtBytes;
        DataCopyPad(preInLocal[ubOffElems], xGm_[baseGmOff + addGmOffElem], extParams, padParams);
    }

    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// CastSquareVf：Load → (Cast→fp32) → Square → Store(tmpBuf[slot])
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::CastSquareVf(
    AscendC::LocalTensor<D_T>& preIn, uint32_t slot)
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto tmpSlot = tmpAll[static_cast<int32_t>(slot) * static_cast<int32_t>(tmpSlotElems_)];

    __ubuf__ D_T* srcPtr = reinterpret_cast<__ubuf__ D_T*>(preIn.GetPhyAddr());
    __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(tmpSlot.GetPhyAddr());

    // 整个 UB tile 元素数（按 padded 占用 —— rUbFactorAlign 覆盖 rSplit==lastR 时的 burst tail pad）
    const uint32_t totalElems =
        static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ * innerRProdAlign_);
    const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> f32Reg;
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = totalElems;

        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);

            if constexpr (kIsFp32) {
                AscendC::Reg::LoadAlign(f32Reg, srcPtr + off);
            } else if constexpr (kIsB16) {
                AscendC::Reg::RegTensor<D_T> b16Reg;
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, srcPtr + off);
                AscendC::Reg::Cast<float, D_T, kCastTraitToFp32>(f32Reg, b16Reg, mask);
            } else { // int32
                AscendC::Reg::RegTensor<int32_t> iReg;
                AscendC::Reg::LoadAlign(iReg, srcPtr + off);
                AscendC::Reg::Cast<float, int32_t, kCastTraitToFp32>(f32Reg, iReg, mask);
            }

            // ★ pre-elewise: x²
            AscendC::Reg::Mul(f32Reg, f32Reg, f32Reg, mask);

            AscendC::Reg::StoreAlign(dstPtr + off, f32Reg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ClearChunkExtensionVf：rSplit 方向 [rLen, rUbFactor) 段填 0.0f
//   tail R / tail A 都需要；rLen < rUbFactor 时才生效。
//
// 几何分两种：
//   tail-R（UB = [A_bundle 外, R_bundle 内]，rSplit 在 R_bundle 外端）：
//     每个 A bundle 入口（共 aUbFactorAlign × innerAProdAlign 个）内，
//     清 R_bundle 中 rSplit 方向 [rLen × innerRProdAlign, rUbFactor × innerRProdAlign) 段。
//   tail-A（UB = [R_bundle 外, A_bundle 内]，rSplit 在 UB 最外层）：
//     UB 是 [rSplit × inner_R × A_bundle]，每个 rSplit 切片 = cellElems lanes，
//     要清的是 [rLen × cellElems, rUbFactor × cellElems) 单段连续。
//     cellElems = aUbFactorAlign × innerAProdAlign × innerRProdAlign，
//     在 tail-A 下 aUbFactorAlign × innerAProdAlign 必是 bsElem 倍数（CeilAlign 已在 tiling 注入），
//     所以 cellElems 是 fp32 的 8-lane 倍数，StoreAlign 起点 32B 对齐。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::ClearChunkExtensionVf(uint32_t slot, int64_t rLen)
{
    if (rLen >= rUbFactor_) {
        return;
    }

    auto tmpAll = tmpBuf_.Get<float>();
    auto tmpSlot = tmpAll[static_cast<int32_t>(slot) * static_cast<int32_t>(tmpSlotElems_)];
    __ubuf__ float* base = reinterpret_cast<__ubuf__ float*>(tmpSlot.GetPhyAddr());

    if constexpr (isTailR) {
        // EuclideanNorm 满足 fast path：pre_elewise = square, pad_value = 0, identity = 0
        //   → square(0) = 0 = sum.identity → DataCopyPad 写入的 padValue 经 CastSquare 后已等价 identity
        //   → 只需清 CeilAlign(rLen×innerRPA, bsElem_fp32) 之后的 tail（dstStride 跳过的未初始化段）
        //   head 区段 [rLen×innerRPA, CeilAlign(...)) 由 DataCopyPad isPad=true 兜底
        //   起点天然 32B 对齐，全程 StoreAlign
        const uint32_t aBundleEntries = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
        const uint32_t innerRPA = static_cast<uint32_t>(innerRProdAlign_);
        const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
        const uint32_t extStart = (rLenInner + kBlockF32 - 1) / kBlockF32 * kBlockF32; // CeilAlign 到 fp32 block
        const uint32_t aStride = static_cast<uint32_t>(rUbFactorAlign_) * innerRPA;    // padded UB 步距
        if (extStart >= aStride) {
            return; // head 已覆盖所有 pad lane，无 tail 需清
        }
        const uint32_t extLanes = aStride - extStart;
        const uint32_t repPerA = (extLanes + kRepF32 - 1) / kRepF32;
        const uint16_t aU16 = static_cast<uint16_t>(aBundleEntries);

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> idReg;
            AscendC::Reg::Duplicate(idReg, 0.0f);

            for (uint16_t a = 0; a < aU16; ++a) {
                int32_t aOff = static_cast<int32_t>(a) * static_cast<int32_t>(aStride);
                uint32_t remaining = extLanes;
                for (uint16_t r = 0; r < static_cast<uint16_t>(repPerA); ++r) {
                    int32_t off =
                        aOff + static_cast<int32_t>(extStart) + static_cast<int32_t>(r) * static_cast<int32_t>(kRepF32);
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::StoreAlign(base + off, idReg, mask);
                }
            }
        }
    } else {
        // tail-A：rSplit 在 UB 最外层，单段连续清零
        const uint32_t cellElems = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * innerRProdAlign_);
        const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
        const uint32_t totalClear = (static_cast<uint32_t>(rUbFactor_) - static_cast<uint32_t>(rLen)) * cellElems;
        const uint32_t repCount = (totalClear + kRepF32 - 1) / kRepF32;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> idReg;
            AscendC::Reg::Duplicate(idReg, 0.0f);
            AscendC::Reg::MaskReg mask;
            uint32_t remaining = totalClear;
            for (uint16_t i = 0; i < static_cast<uint16_t>(repCount); ++i) {
                int32_t off = static_cast<int32_t>(startElem) + static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
                mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::StoreAlign(base + off, idReg, mask);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MergeTmpBufVf：tmpBuf[0] += tmpBuf[1]
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::MergeTmpBufVf()
{
    auto tmpAll = tmpBuf_.Get<float>();
    __ubuf__ float* p0 = reinterpret_cast<__ubuf__ float*>(tmpAll.GetPhyAddr());
    __ubuf__ float* p1 = p0 + tmpSlotElems_;

    const uint32_t totalElems =
        static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ * innerRProdAlign_);
    const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = totalElems;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::LoadAlign(aReg, p0 + off);
            AscendC::Reg::LoadAlign(bReg, p1 + off);
            AscendC::Reg::Add(aReg, aReg, bReg, mask);
            AscendC::Reg::StoreAlign(p0 + off, aReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ReduceRPattern：AscendC::ReduceSum<fp32, AR/RA>
//   src = tmpBuf[0]，dst = tmpBuf[1]（src/dst 不可重叠）
//
// srcShape 顺序：{外层维, 内层维}。
//   tail R：UB=[A_bundle, R_bundle]，外=A_bundle 元素数，内=R_bundle 元素数（含 LastR align）→ Pattern=AR
//   tail A：UB=[R_bundle, A_bundle]，外=R_bundle，内=A_bundle → Pattern=RA
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::ReduceRPattern()
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto src = tmpAll;
    auto dst = tmpAll[static_cast<int32_t>(tmpSlotElems_)];

    const uint32_t aBundle = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t rBundle =
        static_cast<uint32_t>(rUbFactorAlign_ * innerRProdAlign_); // padded（ReduceSum srcInnerPad=true 要求）

    if constexpr (isTailR) {
        uint32_t srcShape[2] = {aBundle, rBundle};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
            dst, src, srcShape, /*srcInnerPad=*/true);
    } else {
        uint32_t srcShape[2] = {rBundle, aBundle};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
            dst, src, srcShape, /*srcInnerPad=*/true);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ClearCacheTreeVf：整个 cacheBuf 清 0（容量从 tilingdata cacheBufUbSize 下发）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::ClearCacheTreeVf()
{
    __ubuf__ float* base = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    const uint32_t totalElems = static_cast<uint32_t>(cacheBufElems_);
    const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> zReg;
        AscendC::Reg::Duplicate(zReg, 0.0f);
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = totalElems;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(base + off, zReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DoCachingVf：ReduceSum 结果（tmpBuf[1] 前 laneN 个 fp32）→ 缓存树 level cacheID
//   laneN = aUbFactorAlign × innerAProdAlign（v2：含多 A 轴 bundle 因子）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::DoCachingVf(uint16_t cacheID)
{
    auto tmpAll = tmpBuf_.Get<float>();
    __ubuf__ float* srcPtr =
        reinterpret_cast<__ubuf__ float*>(tmpAll[static_cast<int32_t>(tmpSlotElems_)].GetPhyAddr());
    __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());

    const uint32_t laneN = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (laneN + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    const uint16_t repeatTime = static_cast<uint16_t>((laneN + kRepF32U - 1) / kRepF32U);
    const uint16_t cacheLvlU16 = cacheID;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = laneN;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);

            AscendC::Reg::LoadAlign(aReg, srcPtr + off);

            for (uint16_t j = 0; j < cacheLvlU16; ++j) {
                int32_t jOff = static_cast<int32_t>(j) * static_cast<int32_t>(levelStride) + off;
                AscendC::Reg::LoadAlign(bReg, cachePtr + jOff);
                AscendC::Reg::Add(aReg, aReg, bReg, mask);
            }
            AscendC::Reg::StoreAlign(cachePtr + levelOff + off, aReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PostElewiseAndCopyOut：读树根 → vSqrt → Cast→D_T → outBuf → DataCopyPad
//
// CopyOut 字段映射（按 isTailR 拆两路径）：
//   tail-R: blockLen = aLen × innerAProd × sizeof,  blockCount = 1
//           UB outBuf 是 dense (a0,...,lastA)，单 burst 一次搬完
//   tail-A + aSplit==lastA: blockLen = aLen × sizeof, blockCount = 1
//           单 A 轴，整段连续
//   tail-A + aSplit!=lastA: blockLen = lastA × sizeof,
//                            blockCount = aLen × innerAProd / lastA
//           UB per-a0 含 CeilAlign(lastA,bsElem) padding，HW 按 32B 跨过
//   srcStride = dstStride = 0
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::PostElewiseAndCopyOut(int64_t outerOutOff, int64_t aLen)
{
    const uint32_t laneN = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (laneN + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    const int32_t rootOff = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);

    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOff;

    auto outLocal = outQue_.AllocTensor<D_T>();
    __ubuf__ D_T* outPtr = reinterpret_cast<__ubuf__ D_T*>(outLocal.GetPhyAddr());

    const uint16_t repeatTime = static_cast<uint16_t>((laneN + kRepF32U - 1) / kRepF32U);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> f32Reg;
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = laneN;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);

            AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
            AscendC::Reg::Sqrt(f32Reg, f32Reg, mask);

            if constexpr (kIsFp32) {
                AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
            } else if constexpr (kIsB16) {
                AscendC::Reg::RegTensor<D_T> b16Reg;
                AscendC::Reg::Cast<D_T, float, kCastTraitFromFp32>(b16Reg, f32Reg, mask);
                AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outPtr + off, b16Reg, mask);
            } else { // int32
                AscendC::Reg::RegTensor<int32_t> iReg;
                AscendC::Reg::Cast<int32_t, float, kCastTraitFromFp32>(iReg, f32Reg, mask);
                AscendC::Reg::StoreAlign(outPtr + off, iReg, mask);
            }
        }
    }
    outQue_.EnQue(outLocal);

    // MTE3：DataCopyPad（按 isTailR 拆两路径，详见函数头注释）
    auto outDeq = outQue_.DeQue<D_T>();
    DataCopyExtParams outParams;
    if constexpr (isTailR) {
        outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd_ * static_cast<int64_t>(sizeof(D_T)));
        outParams.blockCount = 1;
    } else {
        const int32_t lastA = LastAAxis();
        const int64_t lastASize = axisShape_[lastA];
        if (aSplit_ == lastA) {
            outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(D_T)));
            outParams.blockCount = 1;
        } else {
            outParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(D_T)));
            outParams.blockCount = static_cast<uint16_t>(aLen * innerAProd_ / lastASize);
        }
    }
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(yGm_[outerOutOff], outDeq, outParams);
    outQue_.FreeTensor(outDeq);
}

// ════════════════════════════════════════════════════════════════════════════
// Group 模板：A×R 2D 分核 Phase 1 → SyncAll → Phase 2 RA mini-kernel
//
// 数据流：
//   Phase 1: xGm_ ──(MTE2 CopyIn)→ preIn ──(VF Cast+Square)→ tmpBuf
//            ──(VF cache-tree reduce)→ cacheBuf ──(MTE3 CopyOut)→ workspace
//   Phase 2: workspace ──(MTE2 CopyIn)→ preIn ──(ReduceSum)→ tmpBuf
//            ──(VF Sqrt+Cast)→ outBuf ──(MTE3 CopyOut)→ yGm_
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::InitGroup(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const EuclideanNormTilingData* td)
{
    Init(x, y, td); // 复用 normal 全套 UB/字段初始化
    wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    rGroupCnt_ = td->rGroupCnt; // 2D 网格的 R 分组数 = workspace 行数

    // aTotal_ = ∏ 所有 A 轴 axisShape = workspace 列数 = 输出元素总数
    // A 轴在偶数下标 (A R A R ...)，i+=2 跳过 R 轴
    aTotal_ = 1;
    for (int32_t i = 0; i < axisNum_; i += 2) {
        aTotal_ *= axisShape_[i];
    }
}

template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::ProcessGroup()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
        return; // idle 核早退
    }

    const int64_t aOuter = aLoopCntTotal_;
    const int64_t rOuter = rLoopCntTotal_;

    // ── 2D 网格坐标：blockIdx → (aChunkIdx, rChunkIdx) ──
    // 网格 [aOuter, rGroupCnt_]，aPerCore=1（host 端 numBlocks 对齐到 aOuter 保证）
    int64_t aChunkIdx = blockIdx / rGroupCnt_; // 0 .. aOuter-1
    int64_t rChunkIdx = blockIdx % rGroupCnt_; // 0 .. rGroupCnt_-1

    // ── 本核 R 区间 = rGroupCnt_ 等分 rOuter ──
    int64_t rPerCore = (rOuter + rGroupCnt_ - 1) / rGroupCnt_; // CeilDiv
    int64_t rStart = rChunkIdx * rPerCore;
    int64_t rEnd = rStart + rPerCore;
    if (rEnd > rOuter) {
        rEnd = rOuter; // 末核兜底
    }
    if (rStart >= rOuter) {
        return; // rGroupCnt_ > rOuter 时空闲核
    }

    int64_t aLoopIdx = aChunkIdx; // aPerCore=1，直接映射

    // ── unravel aLoopIdx → (外层 A 轴坐标, aSplit 轴 chunk 号) ──
    // 与 normal Process() 完全相同的解码逻辑
    int64_t rem = aLoopIdx;
    const int64_t aSplitChunkIdx = rem % aSplitChunkCnt_; // aSplit 轴上第几个 ub chunk
    rem /= aSplitChunkCnt_;

    // chunkGmOff:  输入 xGm_ 中的偏移（含 R 轴 stride），供 MTE2 CopyIn 用
    // chunkOutOff: 输出空间中的偏移（仅 A 轴 dense stride），供 workspace 列偏移用
    int64_t chunkGmOff = 0;
    int64_t chunkOutOff = 0;
    for (int32_t k = aSplit_ - 2; k >= 0; k -= 2) { // 遍历 aSplit 外侧的 A 轴（跨 R 轴步进 -2）
        const int64_t sz = axisShape_[k];
        const int64_t ix = rem % sz;
        rem /= sz;
        chunkGmOff += ix * axisStride_[k]; // axisStride_: 全量 stride（含 R 轴）
        chunkOutOff += ix * outStride_[k]; // outStride_: 仅 A 轴 dense stride
    }

    const int64_t aSplitAxisSize = axisShape_[aSplit_];
    const int64_t aSplitStride = axisStride_[aSplit_]; // aSplit 轴在 xGm_ 中的 stride
    const int64_t aSplitOutStr = outStride_[aSplit_];  // aSplit 轴在输出空间中的 stride
    const int64_t aChunkStart = aSplitChunkIdx * aUbFactor_;
    const int64_t aEndVal = aChunkStart + aUbFactor_;
    const int64_t aLen = (aEndVal > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : aUbFactor_;
    if (aLen <= 0) {
        return;
    }

    chunkGmOff += aChunkStart * aSplitStride;  // xGm_ 中 aSplit chunk 的起始偏移
    chunkOutOff += aChunkStart * aSplitOutStr; // workspace 中 aSplit chunk 的列偏移

    int64_t rCount = rEnd - rStart; // 本核实际处理的 R chunk 数

    // ── Phase 1: CopyIn → CastSquare → 局部 R 段 cache-tree reduce → 写 workspace ──
    DoOneAChunkGroup(chunkGmOff, aLen, rStart, rEnd);
    Phase1OutputToWorkspace(chunkOutOff, aLen, rChunkIdx, rCount);

    SyncAll(); // 所有核 workspace 写完成

    // ── Phase 2: workspace → ReduceSum(消解 rGroupCnt) → Sqrt → yGm ──
    Phase2Kernel();
}

// ────────────────────────────────────────────────────────────────────────────
// DoOneAChunkGroup: DoOneAChunk 变体 —— R 区间限制在 [rStart, rEnd)
//
// 每个 R chunk 的数据流：
//   xGm_[outerGmOff + rOff]  ──(MTE2 DataCopyPad)→ preIn
//   ──(VF Cast+Square)→ tmpBuf →(ReduceRPattern)→(DoCachingVf)→ cacheBuf
//
// outerGmOff: 外层 A 轴 + aSplit chunk 在 xGm_ 中的偏移（由 ProcessGroup 算好）
// rOff:       UnravelOuterR 解码 R idx → 对应 R chunk 在 xGm_ 中的偏移
//             二者相加 = 该 R chunk 的首个元素在 xGm_ 中的完整 GM 地址
// ────────────────────────────────────────────────────────────────────────────
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::DoOneAChunkGroup(
    int64_t outerGmOff, int64_t aLen, int64_t rStart, int64_t rEnd)
{
    int64_t rCount = rEnd - rStart;
    if (rCount <= 0) {
        return;
    }

    // 以本核 R chunk 数 rCount 建局部二分累加树（非全局 rLoopCntTotal_）
    int64_t bisectionPos = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rCount)));
    int64_t bisectionTail = rCount - bisectionPos; // 主-尾配对的主块数

    ClearCacheTreeVf(); // 清空 cacheBuf（填充 sum identity=0）

    // ── Phase A: 主-尾配对 reduce ──
    for (int64_t i = 0; i < bisectionTail; ++i) {
        // 主块 slot=0: R chunk rStart+i
        {
            int64_t rChunk = 0, rLen = 0;
            const int64_t rOff = UnravelOuterR(rStart + i, rChunk, rLen);
            auto preIn = preInQue_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, preIn);
            preInQue_.EnQue(preIn);
            auto deq = preInQue_.DeQue<D_T>();
            CastSquareVf(deq, 0U); // VF: D_T→fp32 Cast + Square → tmpBuf[0]
            preInQue_.FreeTensor(deq);
            if (rLen < rUbFactor_) {
                ClearChunkExtensionVf(0U, rLen);
            }
        }
        // 尾块 slot=1: R chunk rStart+i+bisectionPos
        {
            int64_t rChunk = 0, rLen = 0;
            const int64_t rOff = UnravelOuterR(rStart + i + bisectionPos, rChunk, rLen);
            auto preIn = preInQue_.AllocTensor<D_T>();
            DoCopyInTile(outerGmOff + rOff, aLen, rLen, preIn);
            preInQue_.EnQue(preIn);
            auto deq = preInQue_.DeQue<D_T>();
            CastSquareVf(deq, 1U); // VF: D_T→fp32 Cast + Square → tmpBuf[1]
            preInQue_.FreeTensor(deq);
            if (rLen < rUbFactor_) {
                ClearChunkExtensionVf(1U, rLen);
            }
        }
        MergeTmpBufVf();            // VF: tmpBuf[0] += tmpBuf[1]
        ReduceRPattern();           // ReduceSum<AR/RA>: tmpBuf[0]→tmpBuf[1]，消解内层 R
        DoCachingVf(GetCacheID(i)); // VF: tmpBuf[1] → cacheBuf[GetCacheID(i)]，二分累加
    }

    // ── Phase B: 单块 reduce（剩余 R chunk 逐个处理）──
    for (int64_t i = bisectionTail; i < bisectionPos; ++i) {
        int64_t rChunk = 0, rLen = 0;
        const int64_t rOff = UnravelOuterR(rStart + i, rChunk, rLen);
        auto preIn = preInQue_.AllocTensor<D_T>();
        DoCopyInTile(outerGmOff + rOff, aLen, rLen, preIn);
        preInQue_.EnQue(preIn);
        auto deq = preInQue_.DeQue<D_T>();
        CastSquareVf(deq, 0U);
        preInQue_.FreeTensor(deq);
        if (rLen < rUbFactor_) {
            ClearChunkExtensionVf(0U, rLen);
        }
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
    // 循环结束后 cache 树根 = 本核 R 段 [rStart,rEnd) 的 fp32 部分和
}

// ────────────────────────────────────────────────────────────────────────────
// Phase1OutputToWorkspace: 读 cache 树根 fp32 → MTE3 CopyOut 到 workspace
//
// workspace 布局: [rGroupCnt_, aTotal_]，fp32 dense，无 padding
// wsOff = rChunkIdx * aTotal_ + wsColOff
//   rChunkIdx: 本核所属 R 分组（workspace 行）
//   wsColOff:  本核 A chunk 在输出空间中的线性列偏移（= ProcessGroup 中的 chunkOutOff）
//
// 源数据: cache 树根在 cacheBuf(VECCALC) 中，VF 已写完，MTE3 直接读，PipeV→MTE3 fence 保证可见
// ────────────────────────────────────────────────────────────────────────────
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::Phase1OutputToWorkspace(
    int64_t wsColOff, int64_t aLen, int64_t rChunkIdx, int64_t rCount)
{
    // ── cache 树根地址 ──
    // cache 树每层 levelStride 个 fp32，共 cacheCount 层，根在第 cacheCount-1 层
    int64_t bisectionPos = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rCount)));
    int64_t cacheCount = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos))) + 1;
    int64_t laneN = aUbFactorAlign_ * innerAProdAlign_;                    // 每层有效 + padding 的 A 元素总数
    int64_t levelStride = (laneN + kBlockF32 - 1) / kBlockF32 * kBlockF32; // CeilAlign 到 32B
    int64_t rootOff = (cacheCount - 1) * levelStride;                      // 根地址 = cacheBuf 基址 + rootOff

    // PipeV → MTE3 fence: cache 树所有 VF 写 (DoCachingVf/ReduceRPattern 等) 对 MTE3 可见
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    auto cacheLocal = cacheBuf_.Get<float>(); // LocalTensor<float>，引用 cacheBuf

    // ── MTE3 CopyOut: cache 树根 → workspace ──
    // 参数模式与 PostElewiseAndCopyOut 完全一致，仅 sizeof(D_T) → sizeof(float)
    DataCopyExtParams ext;
    if constexpr (isTailR) {
        // tail-R: A bundle 在外层，无行尾 padding → 单 block 连续搬
        ext.blockLen = static_cast<uint32_t>(aLen * innerAProd_ * static_cast<int64_t>(sizeof(float)));
        ext.blockCount = 1;
        ext.srcStride = 0;
    } else {
        // tail-A: A bundle 在内层（burst tail），可能有多根 A 轴
        const int32_t lastA = LastAAxis();
        const int64_t lastASize = axisShape_[lastA];
        if (aSplit_ == lastA) {
            // 单 A 轴 → 单 block
            ext.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = 1;
            ext.srcStride = 0;
        } else {
            // 多 A 轴 bundle: 每 block = 最内 A 轴一行，blockCount = 总行数
            ext.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = static_cast<uint16_t>(aLen * innerAProd_ / lastASize);
            // CopyIn 时 pad 的个数是按 DType 计算的。当前是float类型，pad 的个数不变，占用字节变多，所以 srcStride
            // 可能为0，也可能为1
            int64_t bsElem = kBlockBytes / static_cast<int64_t>(sizeof(DType));
            int64_t lastASizeAlign = ((lastASize + bsElem - 1) / bsElem) * bsElem;
            ext.srcStride = (lastASizeAlign - lastASize) * static_cast<int64_t>(sizeof(float)) / kBlockBytes;
        }
    }

    ext.dstStride = 0; // workspace dense 排布

    // MTE3 UB→GM: cache 树根 fp32 → workspace[rChunkIdx][wsColOff]
    int64_t wsOff = rChunkIdx * aTotal_ + wsColOff;
    DataCopyPad(wsGm_[wsOff], cacheLocal[rootOff], ext);
}

// ────────────────────────────────────────────────────────────────────────────
// Phase2Kernel: RA mini-kernel —— R 全载于 preInBuf，消解 rGroupCnt_ 维
//
// 每核处理 aUbFactorP2 宽的一段 A 列，消解全部 rGroupCnt_ 行。
// 数据流（每个 aLoop）:
//   workspace[0:rGroupCnt_-1][a_off:a_off+aLen-1]
//     ──(MTE2 DataCopyPad)→ preInBuf [rGroupCnt_, a_len_ub]  fp32
//     ──(ReduceSum<RA, srcInnerPad=true>)→ tmpBuf[0]          fp32, shape [a_len_ub]
//     ──(VF Sqrt + Cast→D_T)→ outBuf                          D_T, shape [a_len]
//     ──(MTE3 DataCopyPad)→ yGm_[a_off:a_off+aLen-1]          D_T
// ────────────────────────────────────────────────────────────────────────────
template <typename DType, bool isTailR>
__aicore__ inline void EuclideanNormKernel<DType, isTailR>::Phase2Kernel()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());

    // ── 1) kernel 侧算 aUbFactor_p2 ──
    // preInBuf 复用 Phase1 开 DB 的那个物理槽（单路 preReduceUbSize_ 字节）
    // R 全载: rUbFactor_p2 = rGroupCnt_
    // aUbFactor_p2 = preInBuf 单路 fp32 容量 / rGroupCnt_，floor，再 FloorAlign/CeilAlign + 三上界
    const int64_t preInElems = preReduceUbSize_ / static_cast<int64_t>(sizeof(float));
    constexpr int64_t BS_FP32 = 32 / static_cast<int64_t>(sizeof(float)); // = 8

    int64_t aUbFactorP2 = preInElems / rGroupCnt_; // floor
    if (aUbFactorP2 >= BS_FP32) {
        aUbFactorP2 = (aUbFactorP2 / BS_FP32) * BS_FP32; // FloorAlign burst 尾轴
    }
    {
        // 三上界：outBuf 容量、tmpBuf 容量、aTotal
        int64_t limit1 = postReduceUbSize_ / static_cast<int64_t>(sizeof(D_T));
        aUbFactorP2 = (aUbFactorP2 < limit1) ? aUbFactorP2 : limit1;               // outBuf 容不下会越界
        aUbFactorP2 = (aUbFactorP2 < tmpSlotElems_) ? aUbFactorP2 : tmpSlotElems_; // tmpBuf slot 容不下会越界
        aUbFactorP2 = (aUbFactorP2 < aTotal_) ? aUbFactorP2 : aTotal_;             // 超过 aTotal 无意义
    }
    // ── 2) 大小核均衡（复用 normal §6.2 同款逻辑）──
    // outerAProd_p2 = 1（workspace 仅 2 轴 RA，A 即唯一外层）
    const int64_t aSplitChunkCntP2 = (aTotal_ + aUbFactorP2 - 1) / aUbFactorP2; // CeilDiv
    const int64_t aLoopCntTotalP2 = aSplitChunkCntP2;

    const int64_t aSmallCoreLoopCntP2 = aLoopCntTotalP2 / usedCoreNum_; // floor
    const int64_t aBigCoreCntP2 = aLoopCntTotalP2 % usedCoreNum_;       // 大核个数
    const int64_t aBigCoreLoopCntP2 = aSmallCoreLoopCntP2 + (aBigCoreCntP2 > 0 ? 1 : 0);
    const int64_t usedCoreNumP2 = (aSmallCoreLoopCntP2 > 0) ? usedCoreNum_ : aBigCoreCntP2;
    if (blockIdx >= usedCoreNumP2) {
        return;
    }

    // blockIdx → [aLoopStart, aLoopEnd)
    int64_t aLoopStart, aLoopEnd;
    if (blockIdx < aBigCoreCntP2) {
        aLoopStart = blockIdx * aBigCoreLoopCntP2;
        aLoopEnd = aLoopStart + aBigCoreLoopCntP2;
    } else {
        aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 + (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2;
        aLoopEnd = aLoopStart + aSmallCoreLoopCntP2;
    }

    // ── 3) aLoop 主循环 ──
    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t aSplitChunkIdx = aLoopIdx;            // outerAProd=1，直接映射
        int64_t a_off = aSplitChunkIdx * aUbFactorP2; // workspace 列偏移（fp32 元素）
        int64_t a_len = aUbFactorP2;
        if (a_off + a_len > aTotal_) {
            a_len = aTotal_ - a_off;                                    // 尾 chunk 兜底
        }
        int64_t a_len_ub = ((a_len + BS_FP32 - 1) / BS_FP32) * BS_FP32; // CeilAlign，preIn 行宽

        // ── 3a) MTE2 CopyIn: workspace → preInBuf ──
        // src: workspace[rGroupCnt_, aTotal_]，fp32
        // dst: preInBuf [rGroupCnt_, a_len_ub]，fp32 dense 排布，行间无 gap
        //      每行 a_len_ub 元素（a_len 有效 + rightPad 清零）
        auto preInLocal = preInQue_.AllocTensor<float>();
        {
            DataCopyExtParams ext;
            // blockLen:    每行搬 a_len 个 fp32 有效值 (字节)
            // blockCount: 搬 rGroupCnt_ 行（workspace 的 R 维）
            ext.blockLen = static_cast<uint32_t>(a_len * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = static_cast<uint16_t>(rGroupCnt_);
            // srcStride: workspace 行间 stride - blockLen = 下一行首与当前行尾的字节 gap
            //   workspace 每行 aTotal_ 个 fp32，gap = aTotal_*4 - a_len*4
            ext.srcStride = static_cast<uint32_t>(
                aTotal_ * static_cast<int64_t>(sizeof(float)) - static_cast<int64_t>(ext.blockLen));
            // dstStride=0: UB 内行间紧密排列，每行占 a_len_ub fp32，无额外 gap
            ext.dstStride = 0;

            // isPad=true, leftPad=0, rightPad=a_len_ub-a_len, padValue=0.0f(sum identity)
            DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(a_len_ub - a_len), 0.0f};

            DataCopyPad(preInLocal, wsGm_[a_off], ext, padParams);
        }
        preInQue_.EnQue(preInLocal);
        auto preInDeq = preInQue_.DeQue<float>();

        // ── 3b) ReduceSum<fp32, RA, srcInnerPad=true> ──
        // Pattern RA: srcShape = {outerR, innerA} = {rGroupCnt_, a_len_ub}
        //   沿 outer R 维 reduce，保留 inner A 维 → dst shape [a_len_ub]
        // srcInnerPad=true: inner A 维有 right-padding (a_len_ub ≥ a_len)，
        //   硬件按 sum identity(0) 处理 pad 段，不影响有效结果
        // isReuseSource=true: 同 tmpBuf slot，复用 src 内存（dst 不可与 src 重叠）
        {
            auto tmpAll = tmpBuf_.Get<float>();
            uint32_t srcShape[2] = {static_cast<uint32_t>(rGroupCnt_), static_cast<uint32_t>(a_len_ub)};
            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tmpAll, preInDeq, srcShape, true);
        }

        // ── 3c) Post-elewise: Sqrt(fp32) + Cast→D_T → outBuf → MTE3 CopyOut → yGm ──
        // 仅处理 a_len 个有效元素（跳过 UB 尾 padding a_len_ub - a_len）
        {
            auto tmpSrc = tmpBuf_.Get<float>();
            auto outLocal = outQue_.AllocTensor<D_T>();
            __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(tmpSrc.GetPhyAddr());
            __ubuf__ D_T* dstPtr = reinterpret_cast<__ubuf__ D_T*>(outLocal.GetPhyAddr());

            const uint16_t repeatTime = static_cast<uint16_t>((static_cast<uint32_t>(a_len) + kRepF32U - 1) / kRepF32U);

            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<float> f32Reg;
                AscendC::Reg::MaskReg mask;
                uint32_t remaining = static_cast<uint32_t>(a_len); // 只处理有效元素
                for (uint16_t i = 0; i < repeatTime; ++i) {
                    int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
                    mask = AscendC::Reg::UpdateMask<float>(remaining);

                    AscendC::Reg::LoadAlign(f32Reg, srcPtr + off); // VF: load fp32 from tmpBuf
                    AscendC::Reg::Sqrt(f32Reg, f32Reg, mask);      // VF: y = sqrt(x)

                    if constexpr (kIsFp32) {
                        AscendC::Reg::StoreAlign(dstPtr + off, f32Reg, mask);
                    } else if constexpr (kIsB16) {
                        AscendC::Reg::RegTensor<D_T> b16Reg;
                        AscendC::Reg::Cast<D_T, float, kCastTraitFromFp32>(b16Reg, f32Reg, mask);
                        AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            dstPtr + off, b16Reg, mask);
                    } else { // int32
                        AscendC::Reg::RegTensor<int32_t> iReg;
                        AscendC::Reg::Cast<int32_t, float, kCastTraitFromFp32>(iReg, f32Reg, mask);
                        AscendC::Reg::StoreAlign(dstPtr + off, iReg, mask);
                    }
                }
            }
            outQue_.EnQue(outLocal);

            // MTE3 CopyOut: outBuf(D_T) → yGm_[a_off]，a_len 个 D_T 单 block
            auto outDeq = outQue_.DeQue<D_T>();
            DataCopyExtParams outParams;
            outParams.blockLen = static_cast<uint32_t>(a_len * static_cast<int64_t>(sizeof(D_T)));
            outParams.blockCount = 1;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            DataCopyPad(yGm_[a_off], outDeq, outParams);
            outQue_.FreeTensor(outDeq);
        }

        preInQue_.FreeTensor(preInDeq);
    }
}

} // namespace NsEuclideanNorm

#endif // OPS_NORM_EUCLIDEAN_NORM_H_
