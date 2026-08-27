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
 * \file act_ulq_clamp_min_grad.h
 * \brief ActULQClampMinGrad 算子 kernel 类实现（arch35 / RegBase，normal + group 同一 class）。
 *
 * 数学：clamp_min_grad = Σ_全轴( y_grad × ((1 - clamp_min_mask) - x_clamped_loss) ) → 0 维标量
 *
 * 算子类别 = All Reduce（axis_source: implicit_all）：合轴后 pattern 恒为 AR（tail-R），
 *   isTailR 恒 true → kernel 类去掉 isTailR 模板参数，只保留 tail-R 路径。
 *
 * Reducer = sum：
 *   identity = 0.0f，combine = a + b，pad_value = 0，
 *   post_op = identity（无 mean 除法，仅最终缩位 Cast），
 *   needs_bisection = true（二分缓存树 Phase A/B，cacheBuf 16KB），
 *   is_fast_path = true（pre_elewise(pad=0) 在 fp32 视图等价 identity 0，不清行 pad）。
 *
 * pre-elewise 融合（fp32 视图）：signal = 1 - mask（Duplicate 1.0f + Sub）→ x_min_grad = signal -
 * x_clamped_loss（Sub，符号负） → prod = y_grad × x_min_grad（Mul）。⚠ 与同族 Max 的 |mask|(Abs)+vadd 相反，不可互换。
 *
 * dtype（编译期 if constexpr 分发）：
 *   DType（y_grad/x_clamped_loss/输出）：fp16→UNPACK_B16+Cast / fp32→直通
 *   MaskType（clamp_min_mask）：fp16→UNPACK_B16+Cast / fp32→直通 / uint8(bool)→UNPACK4_B8+Cast(→int32→fp32)
 */
#ifndef OPS_ACT_ULQ_CLAMP_MIN_GRAD_H_
#define OPS_ACT_ULQ_CLAMP_MIN_GRAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/reduce/reduce.h"
#include "act_ulq_clamp_min_grad_tiling_data.h"
#include "act_ulq_clamp_min_grad_tiling_key.h"

namespace NsActULQClampMinGrad {

using namespace AscendC;

// ─── 常量 ───
constexpr uint32_t kVlBytes = 256;                     // VL = 256B
constexpr uint32_t kRepF32 = kVlBytes / sizeof(float); // = 64
constexpr uint16_t kRepF32U = static_cast<uint16_t>(kRepF32);
constexpr uint32_t kBlockBytes = 32;                        // 32B
constexpr uint32_t kBlockF32 = kBlockBytes / sizeof(float); // = 8

// UB 内一根轴的描述（innermost-first 排列）
struct UBAxisDesc {
    int32_t gmIdx;      // 该轴在 axisShape / axisStride 中的下标
    int64_t ubSize;     // UB 中有效长度（split 轴为 aLen/rLen，其它为整根 axisShape）
    int64_t paddedSize; // UB 中占位长度（含 burst-tail 行宽 CeilAlign / split 轴 factor[Align]）
    int64_t gmStride;   // GM 上的 stride（元素）
};

// DType：y_grad / x_clamped_loss / 输出承载类型（fp16 / fp32）
// MaskType：clamp_min_mask 承载类型（fp16 / fp32 / uint8(bool)）
template <typename DType, typename MaskType>
class ActULQClampMinGradKernel {
public:
    using D_T = DType;
    using M_T = MaskType;
    using ComputeT = float;

    static constexpr bool kIsFp32 = std::is_same_v<D_T, float>;
    static constexpr bool kIsB16 = (sizeof(D_T) == 2); // fp16 / bf16
    static constexpr bool kNeedCast = !kIsFp32;

    static constexpr bool kMaskIsFp32 = std::is_same_v<M_T, float>;
    static constexpr bool kMaskIsB16 = (sizeof(M_T) == 2);
    // ⚠ DT_BOOL 在 Ascend C kernel 编译期映射为 `bool`（1 字节），非 uint8_t：
    //   原判据 std::is_same_v<M_T, uint8_t> 恒 false → bool mask 误落 b16 分支（按 2 字节 UNPACK_B16
    //   读 1 字节 bool）→ |mask| 全错（本次真机证伪的根因）。改用 1 字节判据覆盖 bool/int8/uint8。
    static constexpr bool kMaskIsBool = (sizeof(M_T) == 1); // bool mask 按 1 字节存储

    // CastTrait：b16 → fp32（扩位，硬件不读 sat / round）
    static constexpr AscendC::Reg::CastTrait kCastB16ToF32{
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_NONE};
    // CastTrait：fp32 → b16（缩位；仅 fp16 输出路径）
    static constexpr AscendC::Reg::CastTrait kCastF32ToB16{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};

    __aicore__ inline ActULQClampMinGradKernel() {}

    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR mask, GM_ADDR xLoss, GM_ADDR out,
                                const ActULQClampMinGradTilingData* td);
    __aicore__ inline void Process();

    // ─── group 模板（public：供 kernel 入口 <op>.cpp 调用）───
    __aicore__ inline void InitGroup(GM_ADDR yGrad, GM_ADDR mask, GM_ADDR xLoss, GM_ADDR out, GM_ADDR workspace,
                                     const ActULQClampMinGradTilingData* td);
    __aicore__ inline void ProcessGroup();

private:
    // ─── 索引解码 ───
    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t& aSplitChunkIdx, int64_t& chunkGmOff,
                                        int64_t& chunkOutOff);
    __aicore__ inline int64_t UnravelOuterR(int64_t rIdx, int64_t& rChunkIdxOut, int64_t& rLenOut) const;
    __aicore__ inline int64_t RLenOfChunk(int64_t rChunkIdx) const;

    // ─── 主流程 ───
    __aicore__ inline void DoOneAChunk(int64_t outerGmOff, int64_t aLen);
    __aicore__ inline void ProcessOneRChunk(int64_t outerGmOff, int64_t aLen, int64_t rIdx, uint32_t slot);
    __aicore__ inline void PostElewise(int64_t aLen);
    __aicore__ inline void CopyOut(int64_t outerOutOff, int64_t aLen);

    // ─── group 模板（内部 helper）───
    __aicore__ inline void Phase1Process();
    __aicore__ inline void Phase2Process();
    __aicore__ inline void DoOneAChunkGroup(int64_t outerGmOff, int64_t aLen, int64_t rStart, int64_t rEnd);
    __aicore__ inline void Phase1OutputToWorkspace(int64_t wsColOff, int64_t aLen, int64_t rChunkIdx, int64_t rCount);

    // ─── CopyIn ───
    __aicore__ inline int32_t LastAAxis() const;
    __aicore__ inline int32_t LastRAxis() const;
    template <typename EltT>
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]) const;
    template <typename EltT>
    __aicore__ inline void DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                        const AscendC::GlobalTensor<EltT>& srcGm, AscendC::LocalTensor<EltT>& preIn);

    // ─── VF ───
    __aicore__ inline void FusePreElewiseVf(AscendC::LocalTensor<D_T>& yGradIn, AscendC::LocalTensor<M_T>& maskIn,
                                            AscendC::LocalTensor<D_T>& xLossIn, uint32_t slot);
    // bool mask 专用：mask 已由 MemBase Cast 转为 fp32（maskF32Buf_），VF 内以 fp32 视图加载。
    __aicore__ inline void FusePreElewiseVfF32Mask(AscendC::LocalTensor<D_T>& yGradIn,
                                                   AscendC::LocalTensor<float>& maskF32In,
                                                   AscendC::LocalTensor<D_T>& xLossIn, uint32_t slot);
    __aicore__ inline void ClearChunkExtensionVf(uint32_t slot, int64_t rLen);
    __aicore__ inline void MergeTmpBufVf();
    __aicore__ inline void ReduceRPattern();
    __aicore__ inline void ClearCacheTreeVf();
    __aicore__ inline void DoCachingVf(uint16_t cacheID);

    // ─── 辅助 ───
    __aicore__ inline uint16_t GetCacheID(int64_t idx) const;
    __aicore__ inline uint64_t FindNearestPower2(uint64_t v) const;
    __aicore__ inline uint64_t CalLog2(uint64_t v) const;

    // ─── tilingdata 镜像 ───
    int32_t axisNum_ = 0;
    int64_t axisShape_[MAX_PATTERN_RANK] = {0};
    int64_t axisStride_[MAX_PATTERN_RANK] = {0};
    int32_t aSplit_ = 0;
    int32_t rSplit_ = 0;
    int64_t aLoopCntTotal_ = 0;
    int64_t aSplitChunkCnt_ = 0;
    int64_t aBigCoreLoopCnt_ = 0;
    int64_t aSmallCoreLoopCnt_ = 0;
    int32_t aBigCoreCnt_ = 0;
    int32_t usedCoreNum_ = 0;
    int64_t aUbFactor_ = 0;
    int64_t aUbFactorAlign_ = 0;
    int64_t rUbFactor_ = 0;
    int64_t rUbFactorAlign_ = 0;
    int64_t innerAProd_ = 0;
    int64_t innerAProdAlign_ = 0;
    int64_t innerRProd_ = 0;
    int64_t innerRProdAlign_ = 0;
    int64_t rLoopCntTotal_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheCount_ = 0;
    int64_t preReduceUbSize_ = 0;
    int64_t postReduceUbSize_ = 0;
    int64_t tmpSlotElems_ = 0;
    int64_t cacheBufElems_ = 0;

    // 输出端 stride（A 轴 dense 排列）
    int64_t outStride_[MAX_PATTERN_RANK] = {0};

    // ─── GM + UB ───
    GlobalTensor<D_T> yGradGm_;
    GlobalTensor<M_T> maskGm_;
    GlobalTensor<D_T> xLossGm_;
    GlobalTensor<D_T> outGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> yGradQue_; // 复用槽（DB=2）
    TQue<QuePosition::VECIN, 2> maskQue_;  // 并行槽（承载 mask，DB=2 以简化流水）
    TQue<QuePosition::VECIN, 2> xLossQue_; // 并行槽
    TBuf<QuePosition::VECCALC> tmpBuf_;    // fp32 主+尾 2 份
    TBuf<QuePosition::VECCALC> cacheBuf_;  // fp32 缓存树 16KB
    // bool(uint8) mask 专用：MemBase 两步 Cast（uint8→half→fp32）暂存 buffer。
    //   uint8 mask 在 UB 的自然 lane 宽（8/16bit）与 fp32 y/x（32bit）不一致，无法在同一 VF 内
    //   按同一 UpdateMask<float> 逐 lane 混读；故 bool 分支先用 MemBase 整块 Cast 到 fp32，
    //   再让融合 VF 以 fp32 视图统一加载（对齐 CANN 官方 cast op uint8→float TWO_CAST：mid=half）。
    TBuf<QuePosition::VECCALC> maskHalfBuf_; // uint8→half 中间（仅 bool 用）
    TBuf<QuePosition::VECCALC> maskF32Buf_;  // half→fp32 结果（仅 bool 用，VF 以 fp32 加载）
    TQue<QuePosition::VECOUT, 1> outQue_;

    // ─── group 模板 ───
    GlobalTensor<float> wsGm_;
    int64_t rGroupCnt_ = 0;
    int64_t aTotal_ = 0;
    // Phase 2 workspace CopyIn 复用 preIn 物理槽（fp32 视图）
    TQue<QuePosition::VECIN, 1> wsInQue_;
};

// ════════════════════════════════════════════════════════════════════════════
// 工具函数
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline int32_t ActULQClampMinGradKernel<DType, MaskType>::LastAAxis() const
{
    for (int32_t i = axisNum_ - 1; i >= 0; --i) {
        if (i % 2 == 0) {
            return i;
        }
    }
    return 0;
}

template <typename DType, typename MaskType>
__aicore__ inline int32_t ActULQClampMinGradKernel<DType, MaskType>::LastRAxis() const
{
    for (int32_t i = axisNum_ - 1; i >= 0; --i) {
        if (i % 2 == 1) {
            return i;
        }
    }
    return 1;
}

template <typename DType, typename MaskType>
__aicore__ inline uint64_t ActULQClampMinGradKernel<DType, MaskType>::FindNearestPower2(uint64_t v) const
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

template <typename DType, typename MaskType>
__aicore__ inline uint64_t ActULQClampMinGradKernel<DType, MaskType>::CalLog2(uint64_t v) const
{
    uint64_t res = 0;
    while (v > 1) {
        v >>= 1;
        ++res;
    }
    return res;
}

template <typename DType, typename MaskType>
__aicore__ inline uint16_t ActULQClampMinGradKernel<DType, MaskType>::GetCacheID(int64_t idx) const
{
    const uint64_t v = static_cast<uint64_t>(idx);
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(v ^ (v + 1)) - 1);
}

template <typename DType, typename MaskType>
__aicore__ inline int64_t ActULQClampMinGradKernel<DType, MaskType>::RLenOfChunk(int64_t rChunkIdx) const
{
    const int64_t rAxisSize = axisShape_[rSplit_];
    const int64_t start = rChunkIdx * rUbFactor_;
    return (start + rUbFactor_ > rAxisSize) ? (rAxisSize - start) : rUbFactor_;
}

// rIdx → (外层 R chunk idx, rLen)；返回外层 R 贡献的 GM 偏移
template <typename DType, typename MaskType>
__aicore__ inline int64_t ActULQClampMinGradKernel<DType, MaskType>::UnravelOuterR(int64_t rIdx, int64_t& rChunkIdxOut,
                                                                                   int64_t& rLenOut) const
{
    const int64_t rChunksOnSplit = (axisShape_[rSplit_] + rUbFactor_ - 1) / rUbFactor_;
    rChunkIdxOut = rIdx % rChunksOnSplit;
    int64_t cur = rIdx / rChunksOnSplit;
    int64_t gmOff = 0;
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
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::Init(GM_ADDR yGrad, GM_ADDR mask, GM_ADDR xLoss,
                                                                       GM_ADDR out,
                                                                       const ActULQClampMinGradTilingData* td)
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

    // 输出端 stride：output 上仅 A 轴 dense 排列（All Reduce 下输出为标量，A=1）
    {
        int64_t acc = 1;
        for (int32_t i = axisNum_ - 1; i >= 0; --i) {
            if (i % 2 == 0) {
                outStride_[i] = acc;
                acc *= axisShape_[i];
            }
        }
    }

    yGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(yGrad));
    maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ M_T*>(mask));
    xLossGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(xLoss));
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(out));

    // preIn 三路：y_grad / x_clamped_loss 按 D_T，mask 按 M_T（bool 更小），
    //   预算按 preReduceUbSize（host 按 max(D_T) 口径算）分配。
    pipe_.InitBuffer(yGradQue_, /*bufNum=*/2, td->preReduceUbSize);
    pipe_.InitBuffer(maskQue_, /*bufNum=*/2, td->preReduceUbSize);
    pipe_.InitBuffer(xLossQue_, /*bufNum=*/2, td->preReduceUbSize);
    pipe_.InitBuffer(tmpBuf_, td->tmpBufUbSize * 2);
    pipe_.InitBuffer(cacheBuf_, td->cacheBufUbSize);
    // bool mask 专用 Cast 暂存（half + fp32），仅 kMaskIsBool 时使用；tmpBufUbSize 与 mask tile 同容量口径。
    if constexpr (kMaskIsBool) {
        pipe_.InitBuffer(maskHalfBuf_, td->tmpBufUbSize); // half 占 tile*2B ≤ tmpBufUbSize(tile*4B)
        pipe_.InitBuffer(maskF32Buf_, td->tmpBufUbSize);  // fp32 tile
    }
    pipe_.InitBuffer(outQue_, /*bufNum=*/1, td->postReduceUbSize);
}

// ────────────────────────────────────────────────────────────────────────────
// UnravelALoop: aLoopIdx → (aSplitChunkIdx, chunkGmOff, chunkOutOff)
// ────────────────────────────────────────────────────────────────────────────
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::UnravelALoop(int64_t aLoopIdx,
                                                                               int64_t& aSplitChunkIdx,
                                                                               int64_t& chunkGmOff,
                                                                               int64_t& chunkOutOff)
{
    int64_t rem = aLoopIdx;
    aSplitChunkIdx = rem % aSplitChunkCnt_;
    rem /= aSplitChunkCnt_;

    chunkGmOff = 0;
    chunkOutOff = 0;
    for (int32_t k = aSplit_ - 2; k >= 0; k -= 2) {
        const int64_t sz = axisShape_[k];
        const int64_t ix = rem % sz;
        rem /= sz;
        chunkGmOff += ix * axisStride_[k];
        chunkOutOff += ix * outStride_[k];
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Process: fused aLoop 多核解码（All Reduce 下 aLoopCntTotal 常为 1 → usedCoreNum=1）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
        return;
    }

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
        int64_t aSplitChunkIdx;
        int64_t chunkGmOff;
        int64_t chunkOutOff;
        UnravelALoop(aLoopIdx, aSplitChunkIdx, chunkGmOff, chunkOutOff);

        const int64_t aChunkStart = aSplitChunkIdx * aUbFactor_;
        const int64_t aEnd = aChunkStart + aUbFactor_;
        const int64_t aLen = (aEnd > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : aUbFactor_;
        if (aLen <= 0) {
            continue;
        }

        chunkGmOff += aChunkStart * aSplitStride;
        chunkOutOff += aChunkStart * aSplitOutStr;

        DoOneAChunk(chunkGmOff, aLen);
        PostElewise(aLen);
        CopyOut(chunkOutOff, aLen);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ProcessOneRChunk: 三输入 CopyIn → 融合 pre-elewise 写 tmpBuf[slot] → partial 清零
// ────────────────────────────────────────────────────────────────────────────
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::ProcessOneRChunk(int64_t outerGmOff, int64_t aLen,
                                                                                   int64_t rIdx, uint32_t slot)
{
    int64_t rChunk = 0, rLen = 0;
    const int64_t rOff = UnravelOuterR(rIdx, rChunk, rLen);
    const int64_t baseGmOff = outerGmOff + rOff;

    auto yGradIn = yGradQue_.template AllocTensor<D_T>();
    auto maskIn = maskQue_.template AllocTensor<M_T>();
    auto xLossIn = xLossQue_.template AllocTensor<D_T>();
    DoCopyInTile<D_T>(baseGmOff, aLen, rLen, yGradGm_, yGradIn);
    DoCopyInTile<M_T>(baseGmOff, aLen, rLen, maskGm_, maskIn);
    DoCopyInTile<D_T>(baseGmOff, aLen, rLen, xLossGm_, xLossIn);
    yGradQue_.EnQue(yGradIn);
    maskQue_.EnQue(maskIn);
    xLossQue_.EnQue(xLossIn);
    auto yDeq = yGradQue_.template DeQue<D_T>();
    auto mDeq = maskQue_.template DeQue<M_T>();
    auto xDeq = xLossQue_.template DeQue<D_T>();

    if constexpr (kMaskIsBool) {
        // bool(uint8) mask：整块 MemBase 两步 Cast uint8→half→fp32（口径同官方 cast op uint8→float），
        //   转到 maskF32Buf_ 后，融合 VF 以 fp32 视图统一加载（与 y/x 同 32bit lane，消除 lane 宽不匹配）。
        const int64_t totalElems = aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ * innerRProdAlign_;
        auto mHalf = maskHalfBuf_.template Get<half>();
        auto mF32 = maskF32Buf_.template Get<float>();
        // DT_BOOL(kernel `bool`, 1B) reinterpret 为 int8_t 供 MemBase Cast（官方 cast 支持 int8→half→float）。
        auto mI8 = mDeq.template ReinterpretCast<int8_t>();
        AscendC::Cast<half, int8_t>(mHalf, mI8, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(totalElems));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast<float, half>(mF32, mHalf, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(totalElems));
        AscendC::PipeBarrier<PIPE_V>();
        FusePreElewiseVfF32Mask(yDeq, mF32, xDeq, slot);
    } else {
        FusePreElewiseVf(yDeq, mDeq, xDeq, slot);
    }

    yGradQue_.FreeTensor(yDeq);
    maskQue_.FreeTensor(mDeq);
    xLossQue_.FreeTensor(xDeq);

    if (rLen < rUbFactor_) {
        ClearChunkExtensionVf(slot, rLen);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 单 A chunk 完整流程：清缓存树 → Phase A 配对 → Phase B 单块
// ────────────────────────────────────────────────────────────────────────────
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::DoOneAChunk(int64_t outerGmOff, int64_t aLen)
{
    ClearCacheTreeVf();

    // Phase A：主-尾配对
    for (int64_t i = 0; i < bisectionTail_; ++i) {
        ProcessOneRChunk(outerGmOff, aLen, i, /*slot=*/0U);
        ProcessOneRChunk(outerGmOff, aLen, i + bisectionPos_, /*slot=*/1U);
        MergeTmpBufVf();
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
    // Phase B：单块
    for (int64_t i = bisectionTail_; i < bisectionPos_; ++i) {
        ProcessOneRChunk(outerGmOff, aLen, i, /*slot=*/0U);
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BuildUBAxes：All Reduce 恒 tail-R；返回 UB 内轴数 K（2 ≤ K ≤ axisNum）
//   out[0] 是 burst 尾轴（最内层 stride 1），out[K-1] 是 UB 内最外侧轴。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
template <typename EltT>
__aicore__ inline int32_t ActULQClampMinGradKernel<DType, MaskType>::BuildUBAxes(int64_t aLen, int64_t rLen,
                                                                                 UBAxisDesc out[]) const
{
    int32_t k = 0;
    const int32_t lastR = LastRAxis();
    const int64_t bsElem = static_cast<int64_t>(kBlockBytes) / static_cast<int64_t>(sizeof(EltT));

    // 内 bundle = R bundle（innermost first：从大 idx 到 rSplit）
    for (int32_t i = axisNum_ - 1; i >= rSplit_; --i) {
        if (i % 2 != 1) {
            continue;
        }
        int64_t actual;
        int64_t padded;
        if (i == rSplit_) {
            actual = rLen;
            padded = rUbFactorAlign_;
        } else if (i == lastR) {
            actual = axisShape_[i];
            padded = (actual + bsElem - 1) / bsElem * bsElem;
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
            padded = aUbFactor_;
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
    return k;
}

// ════════════════════════════════════════════════════════════════════════════
// DoCopyInTile：用 DataCopyPad + LoopMode（可选 host for）把一块 GM tile 装入 preIn
//   行宽按 paddedSize × sizeof(EltT)（与下游 VF 一致，按 fp32 视图 padded 布局）。
//   注：三输入行宽的 UB 元素布局必须一致（都以 rUbFactorAlign / innerRProdAlign 为步距），
//   VF 才能按同一 totalElems 逐 lane 对齐运算。EltT 不同（fp16/fp32/uint8）时 32B 对齐单位
//   随字节宽变化，但 UB 步距用 paddedSize 元素数保持逻辑一致。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
template <typename EltT>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::DoCopyInTile(int64_t baseGmOff, int64_t aLen,
                                                                               int64_t rLen,
                                                                               const AscendC::GlobalTensor<EltT>& srcGm,
                                                                               AscendC::LocalTensor<EltT>& preInLocal)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t K = BuildUBAxes<EltT>(aLen, rLen, ubAxes);
    if (K > MAX_PATTERN_RANK) {
        return;
    }

    DataCopyExtParams extParams;
    LoopModeParams loopParams;
    loopParams.loop1Size = 0;
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    loopParams.loop2Size = 0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;

    const int64_t dtBytes = static_cast<int64_t>(sizeof(EltT));
    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].ubSize * dtBytes);

    uint32_t misalign = extParams.blockLen & (kBlockBytes - 1u);
    uint8_t rPad = 0;
    if (misalign != 0) {
        uint32_t gapBytes = kBlockBytes - misalign;
        rPad = static_cast<uint8_t>(gapBytes / static_cast<uint32_t>(dtBytes));
    }
    DataCopyPadExtParams<EltT> padParams{true, 0, rPad, static_cast<EltT>(0)};

    const int64_t copyPadBytes = (static_cast<int64_t>(extParams.blockLen) + kBlockBytes - 1) / kBlockBytes *
                                 kBlockBytes;
    const int64_t target0Bytes = ubAxes[0].paddedSize * dtBytes;
    extParams.dstStride = static_cast<uint32_t>((target0Bytes - copyPadBytes) / static_cast<int64_t>(kBlockBytes));

    if (K >= 2) {
        extParams.blockCount = static_cast<uint16_t>(ubAxes[1].ubSize);
        const int64_t srcStrideBytes = ubAxes[1].gmStride * dtBytes - static_cast<int64_t>(extParams.blockLen);
        extParams.srcStride = static_cast<uint32_t>(srcStrideBytes > 0 ? srcStrideBytes : 0);
    } else {
        extParams.blockCount = 1;
        extParams.srcStride = 0;
    }

    int64_t ubStride[MAX_PATTERN_RANK];
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < K; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedSize;
    }

    // All Reduce 下 K = 2（A_bundle 外 + R_bundle 内），Loop 模式与 host for 均不触发。
    const bool useLoopMode = (K >= 3);
    if (K >= 3) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[2].ubSize);
        loopParams.loop1SrcStride = static_cast<uint64_t>(ubAxes[2].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop1DstStride = static_cast<uint64_t>(ubStride[2]);
        loopParams.loop2Size = 1;
    }
    if (K >= 4) {
        loopParams.loop2Size = static_cast<uint32_t>(ubAxes[3].ubSize);
        loopParams.loop2SrcStride = static_cast<uint64_t>(ubAxes[3].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop2DstStride = static_cast<uint64_t>(ubStride[3]);
    }
    if (useLoopMode) {
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    }

    int64_t outerProd = 1;
    for (int32_t kk = 4; kk < K; ++kk) {
        outerProd *= ubAxes[kk].ubSize;
    }
    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        int64_t addGmOffElem = 0;
        int64_t addUbOffBytes = 0;
        int64_t cur = outerFlat;
        for (int32_t kk = 4; kk < K; ++kk) {
            const int64_t sz = ubAxes[kk].ubSize;
            const int64_t ix = cur % sz;
            cur /= sz;
            addGmOffElem += ix * ubAxes[kk].gmStride;
            addUbOffBytes += ix * ubStride[kk];
        }
        const int64_t ubOffElems = addUbOffBytes / dtBytes;
        DataCopyPad(preInLocal[ubOffElems], srcGm[baseGmOff + addGmOffElem], extParams, padParams);
    }

    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// FusePreElewiseVf：Load 3 输入 → Cast→fp32 → signal=1-mask → x_min_grad=signal-xLoss
//   → prod=yGrad×x_min_grad → Store(tmpBuf[slot])。全程 fp32 视图。
//
// mask 两路径（编译期 if constexpr；bool 走 FusePreElewiseVfF32Mask，不进本函数）：
//   fp32：LoadAlign 直通
//   fp16/bf16：LoadAlign DIST_UNPACK_B16 + Cast→fp32
// ⚠ bool(uint8) mask 因 UB 自然 lane 宽（8/16bit）与 fp32 y/x（32bit）不一致，无法在同一 VF 内
//   按 UpdateMask<float> 逐 lane 混读（UNPACK4_B8 为稀疏 2/8 lane 布局，UNPACK_B8 需 16bit mask），
//   已在 ProcessOneRChunk 用 MemBase Cast 预转 fp32 → 走 FusePreElewiseVfF32Mask。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::FusePreElewiseVf(AscendC::LocalTensor<D_T>& yGradIn,
                                                                                   AscendC::LocalTensor<M_T>& maskIn,
                                                                                   AscendC::LocalTensor<D_T>& xLossIn,
                                                                                   uint32_t slot)
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto tmpSlot = tmpAll[static_cast<int32_t>(slot) * static_cast<int32_t>(tmpSlotElems_)];

    __ubuf__ D_T* yPtr = reinterpret_cast<__ubuf__ D_T*>(yGradIn.GetPhyAddr());
    __ubuf__ M_T* mPtr = reinterpret_cast<__ubuf__ M_T*>(maskIn.GetPhyAddr());
    __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xLossIn.GetPhyAddr());
    __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(tmpSlot.GetPhyAddr());

    const uint32_t totalElems = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ *
                                                      innerRProdAlign_);
    const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> yReg, mReg, xReg, onesReg;
        AscendC::Reg::Duplicate(onesReg, 1.0f); // signal = 1 - mask 的常量 1（提到 loop 外）
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = totalElems;

        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);

            // ── y_grad → fp32 ──
            if constexpr (kIsFp32) {
                AscendC::Reg::LoadAlign(yReg, reinterpret_cast<__ubuf__ float*>(yPtr) + off);
            } else {
                AscendC::Reg::RegTensor<D_T> yB16;
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(yB16, yPtr + off);
                AscendC::Reg::Cast<float, D_T, kCastB16ToF32>(yReg, yB16, mask);
            }

            // ── x_clamped_loss → fp32 ──
            if constexpr (kIsFp32) {
                AscendC::Reg::LoadAlign(xReg, reinterpret_cast<__ubuf__ float*>(xPtr) + off);
            } else {
                AscendC::Reg::RegTensor<D_T> xB16;
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xB16, xPtr + off);
                AscendC::Reg::Cast<float, D_T, kCastB16ToF32>(xReg, xB16, mask);
            }

            // ── clamp_min_mask → fp32（bool 不进本函数）──
            if constexpr (kMaskIsFp32) {
                AscendC::Reg::LoadAlign(mReg, reinterpret_cast<__ubuf__ float*>(mPtr) + off);
            } else { // fp16 / bf16 mask
                AscendC::Reg::RegTensor<M_T> mB16;
                AscendC::Reg::LoadAlign<M_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(mB16, mPtr + off);
                AscendC::Reg::Cast<float, M_T, kCastB16ToF32>(mReg, mB16, mask);
            }

            // ── Min 融合链：signal = 1 - mask ；x_min_grad = signal - x_clamped_loss ；prod = y_grad × x_min_grad ──
            AscendC::Reg::Sub(mReg, onesReg, mReg, mask); // signal = 1 - mask（ones 在前、mask 在后）
            AscendC::Reg::Sub(mReg, mReg, xReg, mask);    // x_min_grad = signal - x_clamped_loss（符号负）
            AscendC::Reg::Mul(yReg, yReg, mReg, mask);    // prod = y_grad × x_min_grad

            AscendC::Reg::StoreAlign(dstPtr + off, yReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// FusePreElewiseVfF32Mask：bool mask 专用。mask 已由 MemBase Cast 预转为 fp32（maskF32In），
//   VF 内 mask 以 fp32 直载（与 y/x 同 32bit lane），其余与 FusePreElewiseVf 完全一致。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::FusePreElewiseVfF32Mask(
    AscendC::LocalTensor<D_T>& yGradIn, AscendC::LocalTensor<float>& maskF32In, AscendC::LocalTensor<D_T>& xLossIn,
    uint32_t slot)
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto tmpSlot = tmpAll[static_cast<int32_t>(slot) * static_cast<int32_t>(tmpSlotElems_)];

    __ubuf__ D_T* yPtr = reinterpret_cast<__ubuf__ D_T*>(yGradIn.GetPhyAddr());
    __ubuf__ float* mPtr = reinterpret_cast<__ubuf__ float*>(maskF32In.GetPhyAddr());
    __ubuf__ D_T* xPtr = reinterpret_cast<__ubuf__ D_T*>(xLossIn.GetPhyAddr());
    __ubuf__ float* dstPtr = reinterpret_cast<__ubuf__ float*>(tmpSlot.GetPhyAddr());

    const uint32_t totalElems = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ *
                                                      innerRProdAlign_);
    const uint16_t repeatTime = static_cast<uint16_t>((totalElems + kRepF32U - 1) / kRepF32U);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> yReg, mReg, xReg, onesReg;
        AscendC::Reg::Duplicate(onesReg, 1.0f); // signal = 1 - mask 的常量 1（提到 loop 外）
        AscendC::Reg::MaskReg mask;
        uint32_t remaining = totalElems;

        for (uint16_t i = 0; i < repeatTime; ++i) {
            int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);

            // ── y_grad → fp32 ──
            if constexpr (kIsFp32) {
                AscendC::Reg::LoadAlign(yReg, reinterpret_cast<__ubuf__ float*>(yPtr) + off);
            } else {
                AscendC::Reg::RegTensor<D_T> yB16;
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(yB16, yPtr + off);
                AscendC::Reg::Cast<float, D_T, kCastB16ToF32>(yReg, yB16, mask);
            }

            // ── x_clamped_loss → fp32 ──
            if constexpr (kIsFp32) {
                AscendC::Reg::LoadAlign(xReg, reinterpret_cast<__ubuf__ float*>(xPtr) + off);
            } else {
                AscendC::Reg::RegTensor<D_T> xB16;
                AscendC::Reg::LoadAlign<D_T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xB16, xPtr + off);
                AscendC::Reg::Cast<float, D_T, kCastB16ToF32>(xReg, xB16, mask);
            }

            // ── clamp_min_mask（已 fp32）直载 ──
            AscendC::Reg::LoadAlign(mReg, mPtr + off);

            // ── Min 融合链：signal = 1 - mask ；x_min_grad = signal - x_clamped_loss ；prod = y_grad × x_min_grad ──
            AscendC::Reg::Sub(mReg, onesReg, mReg, mask); // signal = 1 - mask（ones 在前、mask 在后）
            AscendC::Reg::Sub(mReg, mReg, xReg, mask);    // x_min_grad = signal - x_clamped_loss（符号负）
            AscendC::Reg::Mul(yReg, yReg, mReg, mask);    // prod = y_grad × x_min_grad

            AscendC::Reg::StoreAlign(dstPtr + off, yReg, mask);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ClearChunkExtensionVf：tail-R；清 CeilAlign(rLen×innerRPA, bs) 之后的 tail（identity=0）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::ClearChunkExtensionVf(uint32_t slot, int64_t rLen)
{
    if (rLen >= rUbFactor_) {
        return;
    }
    auto tmpAll = tmpBuf_.Get<float>();
    auto tmpSlot = tmpAll[static_cast<int32_t>(slot) * static_cast<int32_t>(tmpSlotElems_)];
    __ubuf__ float* base = reinterpret_cast<__ubuf__ float*>(tmpSlot.GetPhyAddr());

    const uint32_t aBundleEntries = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t innerRPA = static_cast<uint32_t>(innerRProdAlign_);
    const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
    const uint32_t extStart = (rLenInner + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    const uint32_t aStride = static_cast<uint32_t>(rUbFactorAlign_) * innerRPA;
    if (extStart >= aStride) {
        return;
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
                int32_t off = aOff + static_cast<int32_t>(extStart) +
                              static_cast<int32_t>(r) * static_cast<int32_t>(kRepF32);
                auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::StoreAlign(base + off, idReg, mask);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MergeTmpBufVf：tmpBuf[0] += tmpBuf[1]
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::MergeTmpBufVf()
{
    auto tmpAll = tmpBuf_.Get<float>();
    __ubuf__ float* p0 = reinterpret_cast<__ubuf__ float*>(tmpAll.GetPhyAddr());
    __ubuf__ float* p1 = p0 + tmpSlotElems_;

    const uint32_t totalElems = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_ * rUbFactorAlign_ *
                                                      innerRProdAlign_);
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
// ReduceRPattern：ReduceSum<fp32, AR>（All Reduce 恒 tail-R）
//   src = tmpBuf[0]，dst = tmpBuf[1]（src/dst 不可重叠）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::ReduceRPattern()
{
    auto tmpAll = tmpBuf_.Get<float>();
    auto src = tmpAll;
    auto dst = tmpAll[static_cast<int32_t>(tmpSlotElems_)];

    const uint32_t aBundle = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t rBundle = static_cast<uint32_t>(rUbFactorAlign_ * innerRProdAlign_);

    uint32_t srcShape[2] = {aBundle, rBundle};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(dst, src, srcShape,
                                                                                    /*srcInnerPad=*/true);
}

// ════════════════════════════════════════════════════════════════════════════
// ClearCacheTreeVf：整个 cacheBuf 清 0（identity）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::ClearCacheTreeVf()
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
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::DoCachingVf(uint16_t cacheID)
{
    auto tmpAll = tmpBuf_.Get<float>();
    __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(
        tmpAll[static_cast<int32_t>(tmpSlotElems_)].GetPhyAddr());
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
// PostElewise：读树根 fp32 → post_op(identity，sum 无后处理) → Cast→D_T → outBuf → EnQue
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::PostElewise(int64_t aLen)
{
    const uint32_t laneN = static_cast<uint32_t>(aUbFactorAlign_ * innerAProdAlign_);
    const uint32_t levelStride = (laneN + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    const int32_t rootOff = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);

    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOff;

    auto outLocal = outQue_.template AllocTensor<D_T>();
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
            // post_op = identity（sum 无后处理）
            if constexpr (kIsFp32) {
                AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
            } else { // fp16 / bf16 输出：缩位 Cast
                AscendC::Reg::RegTensor<D_T> b16Reg;
                AscendC::Reg::Cast<D_T, float, kCastF32ToB16>(b16Reg, f32Reg, mask);
                AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outPtr + off, b16Reg, mask);
            }
        }
    }
    outQue_.EnQue(outLocal);
}

// ════════════════════════════════════════════════════════════════════════════
// CopyOut：All Reduce 恒 tail-R 路径 1（单 burst）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::CopyOut(int64_t outerOutOff, int64_t aLen)
{
    auto outDeq = outQue_.template DeQue<D_T>();
    DataCopyExtParams outParams;
    outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd_ * static_cast<int64_t>(sizeof(D_T)));
    outParams.blockCount = 1;
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(outGm_[outerOutOff], outDeq, outParams);
    outQue_.FreeTensor(outDeq);
}

// ════════════════════════════════════════════════════════════════════════════
// Group 模板：A×R 2D 分核 Phase 1 → SyncAll → Phase 2 RA mini-kernel
// ════════════════════════════════════════════════════════════════════════════
template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::InitGroup(GM_ADDR yGrad, GM_ADDR mask, GM_ADDR xLoss,
                                                                            GM_ADDR out, GM_ADDR workspace,
                                                                            const ActULQClampMinGradTilingData* td)
{
    Init(yGrad, mask, xLoss, out, td);
    wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
    rGroupCnt_ = td->rGroupCnt;

    aTotal_ = 1;
    for (int32_t i = 0; i < axisNum_; i += 2) {
        aTotal_ *= axisShape_[i];
    }
    // Phase 2 workspace CopyIn 用独立 fp32 VECIN 槽（复用 preReduceUbSize 预算）
    pipe_.InitBuffer(wsInQue_, /*bufNum=*/1, td->preReduceUbSize);
}

template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::ProcessGroup()
{
    Phase1Process();
    SyncAll();
    Phase2Process();
}

template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::Phase1Process()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(usedCoreNum_)) {
        return;
    }

    const int64_t rOuter = rLoopCntTotal_;

    int64_t aChunkIdx = blockIdx / rGroupCnt_;
    int64_t rChunkIdx = blockIdx % rGroupCnt_;

    // R 方向大小核式均匀分配：rGroupCnt_ ≤ rOuter 恒成立（tiling 保证），每组 ≥1 chunk，无空组
    int64_t rSmallGroupLoopCnt = rOuter / rGroupCnt_;
    int64_t rBigGroupCnt = rOuter % rGroupCnt_;
    int64_t rBigGroupLoopCnt = rSmallGroupLoopCnt + (rBigGroupCnt > 0 ? 1 : 0);
    int64_t rStart = 0;
    int64_t rCount = 0;
    if (rChunkIdx < rBigGroupCnt) {
        rStart = rChunkIdx * rBigGroupLoopCnt;
        rCount = rBigGroupLoopCnt;
    } else {
        rStart = rBigGroupCnt * rBigGroupLoopCnt + (rChunkIdx - rBigGroupCnt) * rSmallGroupLoopCnt;
        rCount = rSmallGroupLoopCnt;
    }
    int64_t rEnd = rStart + rCount;
    if (rStart >= rOuter) {
        return; // 防御性早退（理论上不可达）
    }

    int64_t aLoopIdx = aChunkIdx;
    int64_t aSplitChunkIdx;
    int64_t chunkGmOff;
    int64_t chunkOutOff;
    UnravelALoop(aLoopIdx, aSplitChunkIdx, chunkGmOff, chunkOutOff);

    const int64_t aSplitAxisSize = axisShape_[aSplit_];
    const int64_t aSplitStride = axisStride_[aSplit_];
    const int64_t aSplitOutStr = outStride_[aSplit_];
    const int64_t aChunkStart = aSplitChunkIdx * aUbFactor_;
    const int64_t aEndVal = aChunkStart + aUbFactor_;
    const int64_t aLen = (aEndVal > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : aUbFactor_;
    if (aLen <= 0) {
        return;
    }

    chunkGmOff += aChunkStart * aSplitStride;
    chunkOutOff += aChunkStart * aSplitOutStr;

    DoOneAChunkGroup(chunkGmOff, aLen, rStart, rEnd);
    Phase1OutputToWorkspace(chunkOutOff, aLen, rChunkIdx, rCount);
}

template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::DoOneAChunkGroup(int64_t outerGmOff, int64_t aLen,
                                                                                   int64_t rStart, int64_t rEnd)
{
    int64_t rCount = rEnd - rStart;
    if (rCount <= 0) {
        return;
    }

    int64_t bisectionPos = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rCount)));
    int64_t bisectionTail = rCount - bisectionPos;

    ClearCacheTreeVf();

    for (int64_t i = 0; i < bisectionTail; ++i) {
        ProcessOneRChunk(outerGmOff, aLen, rStart + i, 0U);
        ProcessOneRChunk(outerGmOff, aLen, rStart + i + bisectionPos, 1U);
        MergeTmpBufVf();
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
    for (int64_t i = bisectionTail; i < bisectionPos; ++i) {
        ProcessOneRChunk(outerGmOff, aLen, rStart + i, 0U);
        ReduceRPattern();
        DoCachingVf(GetCacheID(i));
    }
}

template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::Phase1OutputToWorkspace(int64_t wsColOff,
                                                                                          int64_t aLen,
                                                                                          int64_t rChunkIdx,
                                                                                          int64_t rCount)
{
    int64_t bisectionPos = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rCount)));
    int64_t cacheCount = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos))) + 1;
    int64_t laneN = aUbFactorAlign_ * innerAProdAlign_;
    int64_t levelStride = (laneN + kBlockF32 - 1) / kBlockF32 * kBlockF32;
    int64_t rootOff = (cacheCount - 1) * levelStride;

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    auto cacheLocal = cacheBuf_.Get<float>();

    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(aLen * innerAProd_ * static_cast<int64_t>(sizeof(float)));
    ext.blockCount = 1;
    ext.srcStride = 0;
    ext.dstStride = 0;

    int64_t wsOff = rChunkIdx * aTotal_ + wsColOff;
    DataCopyPad(wsGm_[wsOff], cacheLocal[rootOff], ext);
}

template <typename DType, typename MaskType>
__aicore__ inline void ActULQClampMinGradKernel<DType, MaskType>::Phase2Process()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());

    const int64_t preInElems = preReduceUbSize_ / static_cast<int64_t>(sizeof(float));
    constexpr int64_t BS_FP32 = 32 / static_cast<int64_t>(sizeof(float));

    int64_t aUbFactorP2 = preInElems / rGroupCnt_;
    if (aUbFactorP2 >= BS_FP32) {
        aUbFactorP2 = (aUbFactorP2 / BS_FP32) * BS_FP32;
    }
    {
        int64_t limit1 = postReduceUbSize_ / static_cast<int64_t>(sizeof(D_T));
        aUbFactorP2 = (aUbFactorP2 < limit1) ? aUbFactorP2 : limit1;
        aUbFactorP2 = (aUbFactorP2 < tmpSlotElems_) ? aUbFactorP2 : tmpSlotElems_;
        aUbFactorP2 = (aUbFactorP2 < aTotal_) ? aUbFactorP2 : aTotal_;
    }
    aUbFactorP2 = (aUbFactorP2 < 1) ? 1 : aUbFactorP2;

    const int64_t aSplitChunkCntP2 = (aTotal_ + aUbFactorP2 - 1) / aUbFactorP2;
    const int64_t aLoopCntTotalP2 = aSplitChunkCntP2;

    const int64_t aSmallCoreLoopCntP2 = aLoopCntTotalP2 / usedCoreNum_;
    const int64_t aBigCoreCntP2 = aLoopCntTotalP2 % usedCoreNum_;
    const int64_t aBigCoreLoopCntP2 = aSmallCoreLoopCntP2 + (aBigCoreCntP2 > 0 ? 1 : 0);
    const int64_t usedCoreNumP2 = (aSmallCoreLoopCntP2 > 0) ? usedCoreNum_ : aBigCoreCntP2;
    if (blockIdx >= usedCoreNumP2) {
        return;
    }

    int64_t aLoopStart, aLoopEnd;
    if (blockIdx < aBigCoreCntP2) {
        aLoopStart = blockIdx * aBigCoreLoopCntP2;
        aLoopEnd = aLoopStart + aBigCoreLoopCntP2;
    } else {
        aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 + (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2;
        aLoopEnd = aLoopStart + aSmallCoreLoopCntP2;
    }

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t aSplitChunkIdx = aLoopIdx;
        int64_t a_off = aSplitChunkIdx * aUbFactorP2;
        int64_t a_len = aUbFactorP2;
        if (a_off + a_len > aTotal_) {
            a_len = aTotal_ - a_off;
        }
        int64_t a_len_ub = ((a_len + BS_FP32 - 1) / BS_FP32) * BS_FP32;

        auto preInLocal = wsInQue_.template AllocTensor<float>();
        {
            DataCopyExtParams ext;
            ext.blockLen = static_cast<uint32_t>(a_len * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = static_cast<uint16_t>(rGroupCnt_);
            const int64_t srcStrideBytes = aTotal_ * static_cast<int64_t>(sizeof(float)) -
                                           static_cast<int64_t>(ext.blockLen);
            ext.srcStride = static_cast<uint32_t>(srcStrideBytes > 0 ? srcStrideBytes : 0);
            ext.dstStride = 0;
            DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(a_len_ub - a_len), 0.0f};
            DataCopyPad(preInLocal, wsGm_[a_off], ext, padParams);
        }
        wsInQue_.EnQue(preInLocal);
        auto preInDeq = wsInQue_.template DeQue<float>();

        {
            // ⚠ 507015 修复：workspace CopyIn 已按 DataCopyPad(dstStride=0) 把每个 rGroup 块
            //   物理补齐到 32B（= CeilAlign(a_len,8) = a_len_ub）行宽。此处 srcShape[1] 传入的
            //   已经是**物理行 stride** a_len_ub，故 srcInnerPad 必须为 false（对齐 CANN 官方
            //   reduce_var_twopass RA 用法：srcShape={RNum, 物理 ANum} + srcInnerPad=false）。
            //   原实现 srcInnerPad=true 会让 API 在 a_len_ub 之上再叠一层内层对齐，越界读 → 507015。
            auto tmpAll = tmpBuf_.Get<float>();
            uint32_t srcShape[2] = {static_cast<uint32_t>(rGroupCnt_), static_cast<uint32_t>(a_len_ub)};
            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tmpAll, preInDeq, srcShape, false);
        }
        wsInQue_.FreeTensor(preInDeq);

        {
            auto tmpSrc = tmpBuf_.Get<float>();
            auto outLocal = outQue_.template AllocTensor<D_T>();
            __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(tmpSrc.GetPhyAddr());
            __ubuf__ D_T* dstPtr = reinterpret_cast<__ubuf__ D_T*>(outLocal.GetPhyAddr());

            const uint16_t repeatTime = static_cast<uint16_t>((static_cast<uint32_t>(a_len) + kRepF32U - 1) / kRepF32U);

            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<float> f32Reg;
                AscendC::Reg::MaskReg mask;
                uint32_t remaining = static_cast<uint32_t>(a_len);
                for (uint16_t i = 0; i < repeatTime; ++i) {
                    int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
                    mask = AscendC::Reg::UpdateMask<float>(remaining);

                    AscendC::Reg::LoadAlign(f32Reg, srcPtr + off);
                    if constexpr (kIsFp32) {
                        AscendC::Reg::StoreAlign(dstPtr + off, f32Reg, mask);
                    } else {
                        AscendC::Reg::RegTensor<D_T> b16Reg;
                        AscendC::Reg::Cast<D_T, float, kCastF32ToB16>(b16Reg, f32Reg, mask);
                        AscendC::Reg::StoreAlign<D_T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstPtr + off, b16Reg,
                                                                                              mask);
                    }
                }
            }
            outQue_.EnQue(outLocal);

            auto outDeq = outQue_.template DeQue<D_T>();
            DataCopyExtParams outParams;
            outParams.blockLen = static_cast<uint32_t>(a_len * static_cast<int64_t>(sizeof(D_T)));
            outParams.blockCount = 1;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            DataCopyPad(outGm_[a_off], outDeq, outParams);
            outQue_.FreeTensor(outDeq);
        }
    }
}

} // namespace NsActULQClampMinGrad

#endif // OPS_ACT_ULQ_CLAMP_MIN_GRAD_H_
