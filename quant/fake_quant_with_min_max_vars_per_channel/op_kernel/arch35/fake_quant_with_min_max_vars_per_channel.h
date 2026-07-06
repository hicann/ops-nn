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
 * \file fake_quant_with_min_max_vars_per_channel.h
 * \brief FakeQuantWithMinMaxVarsPerChannel kernel 实现（arch35 / DAV_3510 / v2.2 方案 A）
 *
 * v2.2 关键变更（基于 DESIGN v2.2 §3.5）：
 *   1. 删除 7 个行无关 TBuf（scaleBuf_/nudgedMinBuf_/nudgedMaxBuf_/boolBothZeroBuf_/
 *      scratchBuf_/oneVecBuf_/floorTmpBuf_）—— 共 ~42 KB UB 释放
 *   2. 删除 PrepareChannelScalarsForSegment 私有方法 —— 计算完全下沉 VF
 *   3. UB Init 仅保留 4 个 DB Queue：inQueueX_/outQueueY_/inQueueMin_/inQueueMax_
 *   4. ProcessSegment 精简为：CopyInMin/Max → DeQue → for baseN { CopyInX → Compute(min,max) → CopyOut }
 *   5. VF Compute 三阶段（同一 __VEC_SCOPE__ 内）：
 *      阶段 1 Load (x/min/max → RegTensor)
 *      阶段 2 Prepare 重算 (scale/scaleSafe/both/zpFromMin/nudgedZp/nudgedMin/nudgedMax 全 RegTensor)
 *      阶段 3 主公式 (clip/shift/div/floor/mul/add/bothMul)
 *   6. invQuantRange_ host 算好下发，避免 VF 内重复 Div
 *   7. CastTrait 仍集中在 _base.h（v2.1 已就绪）
 */
#ifndef _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_
#define _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fake_quant_with_min_max_vars_per_channel_tiling_data.h"
#include "fake_quant_with_min_max_vars_per_channel_base.h"

namespace NsFakeQuantPerChannel {

using namespace AscendC;

template <typename T>
class FakeQuantPerChannelNativePC {
    static constexpr int32_t BUF_NUM = 2; // Double Buffer
    static constexpr int64_t VL = 64;     // VL = 64 (fp32 = 256B)

public:
    __aicore__ inline FakeQuantPerChannelNativePC() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR minIn, GM_ADDR maxIn, GM_ADDR y,
                                const FakeQuantWithMinMaxVarsPerChannelTilingData* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSegment(int64_t rowsThisBlk, int64_t baseRow);

    __aicore__ inline void CopyInMin(int64_t curLen, int64_t tailOff);
    __aicore__ inline void CopyInMax(int64_t curLen, int64_t tailOff);
    __aicore__ inline void CopyInX(int64_t curLen, int64_t xElemOff);
    __aicore__ inline void Compute(int64_t curLen, LocalTensor<T>& minSeg, LocalTensor<T>& maxSeg);
    __aicore__ inline void CopyOutY(int64_t curLen, int64_t yElemOff);

    TPipe pipe_;
    // 行相关 DB
    TQue<QuePosition::VECIN, BUF_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUF_NUM> outQueueY_;
    // 行无关 DB
    TQue<QuePosition::VECIN, BUF_NUM> inQueueMin_;
    TQue<QuePosition::VECIN, BUF_NUM> inQueueMax_;
    // [v2.2 删除] 7 个行无关 TBuf —— prepare 全部下沉 VF Compute
    //   scaleBuf_ / nudgedMinBuf_ / nudgedMaxBuf_ / boolBothZeroBuf_ /
    //   scratchBuf_ / oneVecBuf_ / floorTmpBuf_

    GlobalTensor<T> xGm_;
    GlobalTensor<T> minGm_;
    GlobalTensor<T> maxGm_;
    GlobalTensor<T> yGm_;

    // Tiling 缓存
    int64_t numCore_ = 0;
    int64_t blockAxis_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockTailFactor_ = 0;
    int64_t ubAxis_ = 0;
    int64_t baseN_ = 1;
    int64_t baseLen_ = 0;
    int64_t dim0_ = 0;
    int64_t dim1_ = 0;
    int64_t headNum_ = 0;
    int64_t tailDim_ = 0;
    float quantMin_ = 0.0f;
    float quantMax_ = 0.0f;
    float invQuantRange_ = 1.0f; // [v2.2 新增] 1.0f / (quantMax - quantMin)

    // 单核任务边界
    int64_t myHeadRowOffset_ = 0;
    int64_t myHeadRows_ = 0;
    int64_t myTailOffset_ = 0;
    int64_t myTailLen_ = 0;

    int64_t baseLenAligned64_ = 0; // align_up(baseLen, 64)
};

// =============================================================================
// Init（v2.2 方案 A：仅 4 个 Queue）
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::Init(GM_ADDR x, GM_ADDR minIn, GM_ADDR maxIn, GM_ADDR y,
                                                            const FakeQuantWithMinMaxVarsPerChannelTilingData* td)
{
    // 1. 缓存 tiling
    numCore_ = td->numCore;
    blockAxis_ = td->blockAxis;
    blockFactor_ = td->blockFactor;
    blockTailFactor_ = td->blockTailFactor;
    ubAxis_ = td->ubAxis;
    baseN_ = td->baseN;
    baseLen_ = td->baseLen;
    dim0_ = td->dim0;
    dim1_ = td->dim1;
    headNum_ = td->headNum;
    tailDim_ = td->tailDim;
    quantMin_ = td->quantMin;
    quantMax_ = td->quantMax;

    if (baseN_ < 1) {
        baseN_ = 1;
    }
    if (baseLen_ < 1) {
        baseLen_ = VL;
    }

    baseLenAligned64_ = (baseLen_ + VL - 1) / VL * VL;

    // invQuantRange = 1/(qMax - qMin)；qMax-qMin >= 1（quant_max = 2^numBits - 1 >= 3）
    float range = quantMax_ - quantMin_;
    if (range <= 0.0f) {
        range = 1.0f;
    }
    invQuantRange_ = 1.0f / range;

    // 2. 计算本核任务边界
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= numCore_) {
        myHeadRows_ = 0;
        myTailLen_ = 0;
        return;
    }

    if (blockAxis_ == 0) {
        myHeadRowOffset_ = blockIdx * blockFactor_;
        myHeadRows_ = (blockIdx == numCore_ - 1) ? blockTailFactor_ : blockFactor_;
        myTailOffset_ = 0;
        myTailLen_ = tailDim_;
    } else {
        myHeadRowOffset_ = 0;
        myHeadRows_ = headNum_;
        myTailOffset_ = blockIdx * blockFactor_;
        myTailLen_ = (blockIdx == numCore_ - 1) ? blockTailFactor_ : blockFactor_;
    }

    // 3. 绑定 GM
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(minIn));
    maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(maxIn));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    // 4. 分配 UB（v2.2 方案 A：仅 4 个 Queue，无任何行无关 TBuf）
    pipe_.InitBuffer(inQueueX_, BUF_NUM, baseN_ * baseLen_ * sizeof(T));
    pipe_.InitBuffer(outQueueY_, BUF_NUM, baseN_ * baseLen_ * sizeof(T));
    pipe_.InitBuffer(inQueueMin_, BUF_NUM, baseLenAligned64_ * sizeof(T));
    pipe_.InitBuffer(inQueueMax_, BUF_NUM, baseLenAligned64_ * sizeof(T));
    // [v2.2 全部删除] 7 个行无关 TBuf 与 1 个 floorTmpBuf_ —— prepare 全在 VF Compute 内重算
}

// =============================================================================
// Process —— 三层嵌套外层（v2.2 不变）
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::Process()
{
    if (myHeadRows_ <= 0 || myTailLen_ <= 0) {
        return;
    }

    int64_t nBlockNum = (myHeadRows_ + baseN_ - 1) / baseN_;
    for (int64_t nb = 0; nb < nBlockNum; ++nb) {
        int64_t rowsThisBlk = (nb == nBlockNum - 1) ? (myHeadRows_ - nb * baseN_) : baseN_;
        int64_t baseRow = myHeadRowOffset_ + nb * baseN_;
        ProcessSegment(rowsThisBlk, baseRow);
    }
}

// =============================================================================
// ProcessSegment（v2.2 方案 A）：精简为 CopyInMin/Max → DeQue → for baseN { ... }
// 删除：PrepareChannelScalarsForSegment 调用
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::ProcessSegment(int64_t rowsThisBlk, int64_t baseRow)
{
    int64_t lenLoopNum = myTailLen_ / baseLen_;
    int64_t lenTail = myTailLen_ - lenLoopNum * baseLen_;

    for (int64_t li = 0; li < lenLoopNum; ++li) {
        int64_t tailOff = myTailOffset_ + li * baseLen_;
        CopyInMin(baseLen_, tailOff);
        CopyInMax(baseLen_, tailOff);
        LocalTensor<T> minSeg = inQueueMin_.template DeQue<T>();
        LocalTensor<T> maxSeg = inQueueMax_.template DeQue<T>();

        for (int64_t n = 0; n < rowsThisBlk; ++n) {
            int64_t xElemOff = (baseRow + n) * tailDim_ + tailOff;
            CopyInX(baseLen_, xElemOff);
            Compute(baseLen_, minSeg, maxSeg);
            CopyOutY(baseLen_, xElemOff);
        }

        inQueueMin_.FreeTensor(minSeg);
        inQueueMax_.FreeTensor(maxSeg);
    }

    if (lenTail > 0) {
        int64_t tailOff = myTailOffset_ + lenLoopNum * baseLen_;
        CopyInMin(lenTail, tailOff);
        CopyInMax(lenTail, tailOff);
        LocalTensor<T> minSeg = inQueueMin_.template DeQue<T>();
        LocalTensor<T> maxSeg = inQueueMax_.template DeQue<T>();

        for (int64_t n = 0; n < rowsThisBlk; ++n) {
            int64_t xElemOff = (baseRow + n) * tailDim_ + tailOff;
            CopyInX(lenTail, xElemOff);
            Compute(lenTail, minSeg, maxSeg);
            CopyOutY(lenTail, xElemOff);
        }

        inQueueMin_.FreeTensor(minSeg);
        inQueueMax_.FreeTensor(maxSeg);
    }
}

// =============================================================================
// CopyIn / CopyOut（v2.2 不变）
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::CopyInX(int64_t curLen, int64_t xElemOff)
{
    LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    DataCopyExtParams cpyParams;
    cpyParams.blockCount = 1;
    cpyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    cpyParams.srcStride = 0;
    cpyParams.dstStride = 0;
    cpyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[xElemOff], cpyParams, padParams);
    inQueueX_.template EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::CopyInMin(int64_t curLen, int64_t tailOff)
{
    LocalTensor<T> mLocal = inQueueMin_.template AllocTensor<T>();
    DataCopyExtParams cpyParams;
    cpyParams.blockCount = 1;
    cpyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    cpyParams.srcStride = 0;
    cpyParams.dstStride = 0;
    cpyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, minGm_[tailOff], cpyParams, padParams);
    inQueueMin_.template EnQue<T>(mLocal);
}

template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::CopyInMax(int64_t curLen, int64_t tailOff)
{
    LocalTensor<T> mLocal = inQueueMax_.template AllocTensor<T>();
    DataCopyExtParams cpyParams;
    cpyParams.blockCount = 1;
    cpyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    cpyParams.srcStride = 0;
    cpyParams.dstStride = 0;
    cpyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, maxGm_[tailOff], cpyParams, padParams);
    inQueueMax_.template EnQue<T>(mLocal);
}

template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::CopyOutY(int64_t curLen, int64_t yElemOff)
{
    LocalTensor<T> yLocal = outQueueY_.template DeQue<T>();
    DataCopyExtParams cpyParams;
    cpyParams.blockCount = 1;
    cpyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    cpyParams.srcStride = 0;
    cpyParams.dstStride = 0;
    cpyParams.rsv = 0;
    DataCopyPad(yGm_[yElemOff], yLocal, cpyParams);
    outQueueY_.FreeTensor(yLocal);
}

// =============================================================================
// Compute（v2.2 方案 A）—— VF Compute 三阶段全链路
//   阶段 1 Load (x/min/max → RegTensor)
//   阶段 2 Prepare 重算 (scale/scaleSafe/both/zpFromMin/nudgedZp/nudgedMin/nudgedMax)
//   阶段 3 主公式 (clip/shift/div/floor/mul/add/bothMul)
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::Compute(int64_t curLen, LocalTensor<T>& minSeg,
                                                               LocalTensor<T>& maxSeg)
{
    LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    LocalTensor<T> yLocal = outQueueY_.template AllocTensor<T>();

    __ubuf__ T* xAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    __ubuf__ T* yAddr = reinterpret_cast<__ubuf__ T*>(yLocal.GetPhyAddr());
    __ubuf__ T* minAddr = reinterpret_cast<__ubuf__ T*>(minSeg.GetPhyAddr());
    __ubuf__ T* maxAddr = reinterpret_cast<__ubuf__ T*>(maxSeg.GetPhyAddr());

    constexpr uint32_t vfLen32 = VECTOR_REG_WIDTH / sizeof(float); // = 64 for DAV_3510
    uint32_t totalLen = static_cast<uint32_t>(curLen);
    uint32_t outerLoop = (totalLen + vfLen32 - 1) / vfLen32;
    uint32_t remain = totalLen;
    const float qMin = quantMin_;
    const float qMax = quantMax_;
    const float qRange = quantMax_ - quantMin_; // 0-ULP Div 用：scale = (max-min)/qRange
    static constexpr AscendC::MicroAPI::DivSpecificMode kDivMode = {AscendC::MicroAPI::MaskMergeMode::ZEROING,
                                                                    true}; // true = 0-ULP fp32 Div

    __VEC_SCOPE__
    {
        // === 输入 ===
        AscendC::Reg::RegTensor<float> vregX;
        AscendC::Reg::RegTensor<float> vregMin;
        AscendC::Reg::RegTensor<float> vregMax;
        // === prepare 常量 / 中间量 ===
        AscendC::Reg::RegTensor<float> vregZero;   // 全 0（Sub 自身得到）
        AscendC::Reg::RegTensor<float> vregOne;    // 全 1（Adds(zero, 1.0f)）
        AscendC::Reg::RegTensor<float> vregQRange; // 全 qMax-qMin（用作 0-ULP Div 的除数）
        AscendC::Reg::RegTensor<float> vregScale;
        AscendC::Reg::RegTensor<float> vregScaleSafe;
        AscendC::Reg::RegTensor<float> vregDiffSaved;
        AscendC::Reg::RegTensor<float> vregInvScaleSafe;
        AscendC::Reg::RegTensor<float> vregAbsMin;
        AscendC::Reg::RegTensor<float> vregAbsMax;
        AscendC::Reg::RegTensor<float> vregAbsSum;
        AscendC::Reg::RegTensor<float> vregBoth;
        AscendC::Reg::RegTensor<float> vregZpFromMin;
        AscendC::Reg::RegTensor<float> vregZpClipped;
        AscendC::Reg::RegTensor<int32_t> vregZpInt;
        AscendC::Reg::RegTensor<float> vregNudgedZp;
        AscendC::Reg::RegTensor<float> vregNudgedMin;
        AscendC::Reg::RegTensor<float> vregNudgedMax;
        // === 主公式 ===
        AscendC::Reg::RegTensor<float> vregClamped;
        AscendC::Reg::RegTensor<float> vregShift;
        AscendC::Reg::RegTensor<float> vregScaled;
        AscendC::Reg::RegTensor<int32_t> vregFlrInt;
        AscendC::Reg::RegTensor<float> vregFloored;
        AscendC::Reg::RegTensor<float> vregOut;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg maskBoth;
        AscendC::Reg::MaskReg maskScaleZero;

        for (uint16_t i = 0; i < static_cast<uint16_t>(outerLoop); ++i) {
            mask = AscendC::Reg::UpdateMask<float>(remain);

            // -----------------------------------------------------------------
            // 阶段 1：Load x / min / max
            // -----------------------------------------------------------------
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregX, xAddr + i * vfLen32);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregMin, minAddr + i * vfLen32);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregMax, maxAddr + i * vfLen32);

            // -----------------------------------------------------------------
            // 阶段 2：Prepare 重算（替代 PrepareChannelScalarsForSegment）
            // -----------------------------------------------------------------

            // 2.0 构造常量 vregZero / vregOne / vregQRange（寄存器内）
            AscendC::Reg::Sub<float>(vregZero, vregMin, vregMin, mask);
            AscendC::Reg::Adds<float>(vregOne, vregZero, 1.0f, mask);
            AscendC::Reg::Adds<float>(vregQRange, vregZero, qRange, mask);

            // 2.1 scale = (max - min) / qRange
            //     用 0-ULP Div 替代 (max-min)*invR：避免 invR=1/(qMax-qMin) 在 fp32 上的
            //     round-to-nearest 误差与 (max-min)*invR 的舍入叠加，使 scale 精确等于
            //     `(max-min)/(qMax-qMin)` 的 IEEE 单步除法结果（与 TF 实现一致）。
            AscendC::Reg::Sub<float>(vregScale, vregMax, vregMin, mask);
            AscendC::Reg::Copy<float>(vregDiffSaved, vregScale, mask);
            AscendC::Reg::Div<float, &kDivMode>(vregScale, vregScale, vregQRange, mask);

            // 2.2 bothZero = (|min| + |max| > 0) ? 1.0 : 0.0
            AscendC::Reg::Abs<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregAbsMin, vregMin, mask);
            AscendC::Reg::Abs<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregAbsMax, vregMax, mask);
            AscendC::Reg::Add<float>(vregAbsSum, vregAbsMin, vregAbsMax, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskBoth, vregAbsSum, 0.0f, mask);
            // Select: mask=1 → src0(vregOne); mask=0 → src1(vregZero)
            AscendC::Reg::Select<float>(vregBoth, vregOne, vregZero, maskBoth);

            // 2.3 scale==0 兜底：vregScaleSafe = (scale==0) ? 1.0 : scale
            AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(maskScaleZero, vregScale, 0.0f, mask);
            AscendC::Reg::Select<float>(vregScaleSafe, vregOne, vregScale, maskScaleZero);

            // 2.3b invScaleSafe = qRange / diffSafe
            //      TF Nudge() 独立计算 inv_scale = (qMax-qMin)/(max-min)，
            //      而非 1/scale，避免 scale 的 fp32 舍入误差在取倒数时被放大。
            //      diffSafe = (diff==0) ? 1.0 : diff（复用 maskScaleZero，因 scale==0 ⟺ diff==0）
            AscendC::Reg::Select<float>(vregDiffSaved, vregOne, vregDiffSaved, maskScaleZero);
            AscendC::Reg::Div<float, &kDivMode>(vregInvScaleSafe, vregQRange, vregDiffSaved, mask);

            // 2.4 zp_from_min = qMin - min / scaleSafe
            //     用 0-ULP Div：min/scaleSafe 在 zfm 落到 N+0.5 半整数边界时极易抖动 1 ULP，
            //     与 floor(zfm+0.5) 叠加会让 nudged_zp 偏离 TF。
            AscendC::Reg::Div<float, &kDivMode>(vregZpFromMin, vregMin, vregScaleSafe, mask);
            AscendC::Reg::Muls<float>(vregZpFromMin, vregZpFromMin, -1.0f, mask);
            AscendC::Reg::Adds<float>(vregZpFromMin, vregZpFromMin, qMin, mask);

            // 2.5 nudged_zp = floor(clip(zp_from_min, qMin, qMax) + 0.5)
            AscendC::Reg::Maxs<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vregZpClipped, vregZpFromMin, qMin,
                                                                                   mask);
            AscendC::Reg::Mins<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vregZpClipped, vregZpClipped, qMax,
                                                                                   mask);
            AscendC::Reg::Adds<float>(vregZpClipped, vregZpClipped, 0.5f, mask);
            AscendC::Reg::Cast<int32_t, float, kCastFp32ToInt32Floor>(vregZpInt, vregZpClipped, mask);
            AscendC::Reg::Cast<float, int32_t, kCastInt32ToFp32>(vregNudgedZp, vregZpInt, mask);

            // 2.6 nudged_min = (qMin - nudged_zp) * scaleSafe
            //     nudged_max = (qMax - nudged_zp) * scaleSafe
            AscendC::Reg::Muls<float>(vregNudgedMin, vregNudgedZp, -1.0f, mask);
            AscendC::Reg::Adds<float>(vregNudgedMin, vregNudgedMin, qMin, mask);
            AscendC::Reg::Mul<float>(vregNudgedMin, vregNudgedMin, vregScaleSafe, mask);

            AscendC::Reg::Muls<float>(vregNudgedMax, vregNudgedZp, -1.0f, mask);
            AscendC::Reg::Adds<float>(vregNudgedMax, vregNudgedMax, qMax, mask);
            AscendC::Reg::Mul<float>(vregNudgedMax, vregNudgedMax, vregScaleSafe, mask);

            // -----------------------------------------------------------------
            // 阶段 3：主公式
            // -----------------------------------------------------------------
            // clip：clamped = min(max(x, nudgedMin), nudgedMax)
            AscendC::Reg::Min<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregClamped, vregX, vregNudgedMax, mask);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregClamped, vregClamped, vregNudgedMin,
                                                                           mask);
            // shift = clamped - nudgedMin
            AscendC::Reg::Sub<float>(vregShift, vregClamped, vregNudgedMin, mask);
            // scaled = shift * invScaleSafe + 0.5
            //        与 TF 一致：使用独立计算的 inv_scale = qRange/(max-min)，
            //        而非 shift/scale（二次舍入误差）
            AscendC::Reg::Mul<float>(vregScaled, vregShift, vregInvScaleSafe, mask);
            AscendC::Reg::Adds<float>(vregScaled, vregScaled, 0.5f, mask);
            // floor 语义：fp32 → int32 (CAST_FLOOR) → fp32 (UNKNOWN)
            AscendC::Reg::Cast<int32_t, float, kCastFp32ToInt32Floor>(vregFlrInt, vregScaled, mask);
            AscendC::Reg::Cast<float, int32_t, kCastInt32ToFp32>(vregFloored, vregFlrInt, mask);
            // out = floored * scaleSafe + nudgedMin
            AscendC::Reg::Mul<float>(vregOut, vregFloored, vregScaleSafe, mask);
            AscendC::Reg::Add<float>(vregOut, vregOut, vregNudgedMin, mask);
            // out *= bothZero（兜底：当 |min|+|max|==0 时 out=0）
            AscendC::Reg::Mul<float>(vregOut, vregOut, vregBoth, mask);
            // Store
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + i * vfLen32, vregOut, mask);
        }
    } // __VEC_SCOPE__

    outQueueY_.template EnQue<T>(yLocal);
    inQueueX_.FreeTensor(xLocal);
}

} // namespace NsFakeQuantPerChannel

#endif // _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_
