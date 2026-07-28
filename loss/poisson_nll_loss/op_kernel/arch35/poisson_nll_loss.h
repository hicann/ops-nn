/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file poisson_nll_loss.h
 * \brief hand-written kernel. No DAG / reduce template.
 *
 * Elementwise loss:
 *   log_input=True : exp(x) - target * x
 *   log_input=False: x - target * log(x + eps)
 *   full=True adds Stirling term where target > 1:
 *       target*log(target) - target + 0.5*log(2*pi*target)
 * reduction:
 *   none : write loss elementwise (same shape as input)
 *   sum  : all elements summed to a scalar
 *   mean : sum * (1/totalNum)
 * All of log_input / full / reduction are runtime scalar flags from tilingData.
 *
 * sum/mean reduce over all axes into a single scalar via a deterministic two-phase
 * fp32 reduction: each core accumulates a fp32 partial sum into workspace[blockIdx],
 * SyncAll, then block 0 sums the partials and writes the scalar output. fp32 accumulation
 * matters for fp16 inputs (golden accumulates in fp32).
 */

#ifndef POISSON_NLL_LOSS_H
#define POISSON_NLL_LOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "poisson_nll_loss_tiling_def.h"
#ifndef __CCE_KT_TEST__
#include "poisson_nll_loss_tiling_key.h"
#endif

namespace NsPoissonNllLoss {

using namespace AscendC;

constexpr uint32_t PNLL_REDUCTION_NONE = 0;
constexpr uint32_t PNLL_REDUCTION_SUM = 1;
constexpr uint32_t PNLL_REDUCTION_MEAN = 2;
// Per-core workspace partial sums must be 32B(=8 fp32)-block aligned: GM write transactions
// are atomic at 32B granularity, so adjacent cores writing dense 4B slots into one block would
// race and corrupt each other. Give each core its own block. (aligns with norm/batch_norm_grad_v3)
constexpr int32_t PNLL_WS_CORE_STRIDE = 8;
// Max vector cores on Ascend950 (vector_core_cnt <= 64). Phase-2 reads usedCoreNum*8 fp32
// partials back into a fixed buffer of this capacity, decoupled from ubFactor so it never
// overflows regardless of shape/core split.
constexpr int32_t PNLL_MAX_CORE_NUM = 64;
constexpr int32_t PNLL_PARTIAL_BUF_ELEMS = PNLL_MAX_CORE_NUM * PNLL_WS_CORE_STRIDE; // 512 fp32 = 2KB

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
class KernelPoissonNllLoss {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr bool NEED_CAST = !std::is_same<T_IN, T_COMPUTE>::value;

public:
    __aicore__ inline KernelPoissonNllLoss() {}

    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                const PoissonNllLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void ComputeLoss(int64_t currentNum, LocalTensor<T_COMPUTE>& lossOut);
    __aicore__ inline void ComputeNone(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void ProcessNone();
    __aicore__ inline void ProcessReduce();
    __aicore__ inline float LocalReduceSum(LocalTensor<T_COMPUTE>& src, int64_t count);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueT_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    TBuf<QuePosition::VECCALC> workBuf1_;
    TBuf<QuePosition::VECCALC> workBuf2_;
    TBuf<QuePosition::VECCALC> workBuf3_;
    TBuf<QuePosition::VECCALC> reduceBuf_;  // per-tile reduce scratch + this-core partial staging
    TBuf<QuePosition::VECCALC> partialBuf_; // Phase-2: read back all cores' partials (fixed 2KB)
    TBuf<QuePosition::VECCALC> cmpBuf_;
    TBuf<QuePosition::VECCALC> castXBuf_;
    TBuf<QuePosition::VECCALC> castTBuf_;

    GlobalTensor<T_IN> xGm_;
    GlobalTensor<T_IN> tGm_;
    GlobalTensor<T_IN> yGm_;
    GlobalTensor<float> wsGm_; // partial sums, one fp32 per core (reduction path)

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t usedCoreNum_ = 0;
    float eps_ = 1e-8f;
    float meanCof_ = 1.0f;
    bool logInput_ = true;
    bool full_ = false;
    uint32_t reduction_ = PNLL_REDUCTION_NONE;
    int32_t blockIdx_ = 0;
};

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::Init(
    GM_ADDR inputX, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace, const PoissonNllLossTilingData* tilingData)
{
    eps_ = tilingData->eps;
    meanCof_ = tilingData->meanCof;
    logInput_ = (tilingData->logInput != 0);
    full_ = (tilingData->full != 0);
    reduction_ = tilingData->reduction;
    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    blockIdx_ = static_cast<int32_t>(AscendC::GetBlockIdx());
    // Empty tensor (blockFactor_==0): keep usedCoreNum_ at 1 so the reduction path stages/reads
    // exactly one (zero) partial instead of doing 0-length DataCopy in Phase 2. block_dim is 1 too.
    usedCoreNum_ = (blockFactor_ > 0) ? ((totalNum_ + blockFactor_ - 1) / blockFactor_) : 1;

    int64_t startOffset = blockFactor_ * static_cast<int64_t>(blockIdx_);
    int64_t remaining = totalNum_ - startOffset;
    blockLength_ = (remaining <= 0) ? 0 : ((remaining > blockFactor_) ? blockFactor_ : remaining);

    xGm_.SetGlobalBuffer((__gm__ T_IN*)inputX + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    tGm_.SetGlobalBuffer((__gm__ T_IN*)target + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    if (reduction_ == PNLL_REDUCTION_NONE) {
        yGm_.SetGlobalBuffer((__gm__ T_IN*)loss + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    } else {
        yGm_.SetGlobalBuffer((__gm__ T_IN*)loss, 1);
        int64_t wsElems = (usedCoreNum_ > 0) ? (usedCoreNum_ * PNLL_WS_CORE_STRIDE) : PNLL_WS_CORE_STRIDE;
        wsGm_.SetGlobalBuffer((__gm__ float*)workspace, wsElems);
    }

    constexpr int64_t COMPARE_MIN_BYTES = 256;
    constexpr int64_t COMPARE_MIN_ELEMS = COMPARE_MIN_BYTES / static_cast<int64_t>(sizeof(T_COMPUTE));
    int64_t allocElems = (ubLength_ < COMPARE_MIN_ELEMS) ? COMPARE_MIN_ELEMS : ubLength_;

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(inQueueT_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));

    pipe_.InitBuffer(workBuf1_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    pipe_.InitBuffer(workBuf2_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    pipe_.InitBuffer(workBuf3_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    pipe_.InitBuffer(reduceBuf_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    if (reduction_ != PNLL_REDUCTION_NONE) {
        pipe_.InitBuffer(partialBuf_, PNLL_PARTIAL_BUF_ELEMS * static_cast<int64_t>(sizeof(float)));
    }

    int64_t cmpBufSize = ((allocElems / 8 + 255) / 256) * 256;
    if (cmpBufSize < 256) {
        cmpBufSize = 256;
    }
    pipe_.InitBuffer(cmpBuf_, cmpBufSize);

    if constexpr (NEED_CAST) {
        pipe_.InitBuffer(castXBuf_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
        pipe_.InitBuffer(castTBuf_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    }
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T_IN> xLocal = inQueueX_.template AllocTensor<T_IN>();
    LocalTensor<T_IN> tLocal = inQueueT_.template AllocTensor<T_IN>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T_IN)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T_IN> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[progress * ubLength_], copyParams, padParams);
    DataCopyPad(tLocal, tGm_[progress * ubLength_], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
    inQueueT_.EnQue(tLocal);
}

// Compute the elementwise loss into lossOut (a fp32 buffer). Consumes queued x/t tensors.
template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::ComputeLoss(int64_t currentNum,
                                                                                       LocalTensor<T_COMPUTE>& lossOut)
{
    LocalTensor<T_IN> xLocal = inQueueX_.template DeQue<T_IN>();
    LocalTensor<T_IN> tLocal = inQueueT_.template DeQue<T_IN>();

    LocalTensor<T_COMPUTE> tmp = workBuf2_.Get<T_COMPUTE>();
    LocalTensor<T_COMPUTE> tmp2 = workBuf3_.Get<T_COMPUTE>();
    LocalTensor<uint8_t> cmpMask = cmpBuf_.Get<uint8_t>();

    uint32_t elemCount = static_cast<uint32_t>(currentNum);
    constexpr uint32_t COMPARE_ALIGN_ELEMENTS = 256 / static_cast<uint32_t>(sizeof(T_COMPUTE));
    uint32_t alignedCount = (elemCount + COMPARE_ALIGN_ELEMENTS - 1) / COMPARE_ALIGN_ELEMENTS * COMPARE_ALIGN_ELEMENTS;

    LocalTensor<T_COMPUTE> xF;
    LocalTensor<T_COMPUTE> tF;
    if constexpr (NEED_CAST) {
        xF = castXBuf_.Get<T_COMPUTE>();
        tF = castTBuf_.Get<T_COMPUTE>();
        Cast(xF, xLocal, RoundMode::CAST_NONE, elemCount);
        Cast(tF, tLocal, RoundMode::CAST_NONE, elemCount);
    } else {
        xF = xLocal.template ReinterpretCast<T_COMPUTE>();
        tF = tLocal.template ReinterpretCast<T_COMPUTE>();
    }

    // -- Base loss --
    if (logInput_) {
        // loss = exp(x) - target * x
        Exp(lossOut, xF, elemCount);
        Mul(tmp, tF, xF, elemCount);
        Sub(lossOut, lossOut, tmp, elemCount);
    } else {
        // loss = x - target * log(x + eps)
        Adds(tmp, xF, static_cast<T_COMPUTE>(eps_), elemCount);
        Ln(tmp, tmp, elemCount);
        Mul(tmp, tF, tmp, elemCount);
        Sub(lossOut, xF, tmp, elemCount);
    }

    // -- Optional Stirling term (full=True): add where target > 1, else 0 --
    if (full_) {
        Ln(tmp2, tF, elemCount);                                              // log(target)
        Mul(tmp, tF, tmp2, elemCount);                                        // target*log(target)
        Sub(tmp, tmp, tF, elemCount);                                         // -target
        Muls(tmp2, tF, static_cast<T_COMPUTE>(6.283185307179586), elemCount); // 2*pi*target
        Ln(tmp2, tmp2, elemCount);                                            // log(2*pi*target)
        Muls(tmp2, tmp2, static_cast<T_COMPUTE>(0.5), elemCount);             // 0.5*log(2*pi*target)
        Add(tmp, tmp, tmp2, elemCount);                                       // stirling
        Duplicate<T_COMPUTE>(tmp2, static_cast<T_COMPUTE>(1), elemCount);
        Compare(cmpMask, tF, tmp2, CMPMODE::GT, alignedCount); // target > 1
        Duplicate<T_COMPUTE>(tmp2, static_cast<T_COMPUTE>(0), elemCount);
        Select(tmp, cmpMask, tmp, tmp2, SELMODE::VSEL_TENSOR_TENSOR_MODE, elemCount);
        Add(lossOut, lossOut, tmp, elemCount);
    }

    inQueueX_.FreeTensor(xLocal);
    inQueueT_.FreeTensor(tLocal);
}

// reduction=none: compute loss and write it out elementwise (same shape as input).
template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::ComputeNone(int64_t currentNum)
{
    LocalTensor<T_COMPUTE> loss = workBuf1_.Get<T_COMPUTE>();
    ComputeLoss(currentNum, loss);

    uint32_t elemCount = static_cast<uint32_t>(currentNum);
    LocalTensor<T_IN> yLocal = outQueueY_.template AllocTensor<T_IN>();
    if constexpr (NEED_CAST) {
        Cast(yLocal, loss, RoundMode::CAST_ROUND, elemCount);
    } else {
        LocalTensor<T_COMPUTE> outF = yLocal.template ReinterpretCast<T_COMPUTE>();
        Adds(outF, loss, static_cast<T_COMPUTE>(0), elemCount);
    }
    outQueueY_.template EnQue<T_IN>(yLocal);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T_IN> yLocal = outQueueY_.template DeQue<T_IN>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T_IN)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(yGm_[progress * ubLength_], yLocal, copyParams);
    outQueueY_.FreeTensor(yLocal);
}

// Whole-reduce a fp32 LocalTensor of `count` elements to a single fp32 value.
// Uses ReduceSum(dst, src, work, count) — the proven arch35 whole-reduce form.
template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline float KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::LocalReduceSum(LocalTensor<T_COMPUTE>& src,
                                                                                           int64_t count)
{
    LocalTensor<T_COMPUTE> red = reduceBuf_.Get<T_COMPUTE>();
    ReduceSum(red, src, red, static_cast<int32_t>(count));
    event_t evtVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evtVS);
    WaitFlag<HardEvent::V_S>(evtVS);
    return red.GetValue(0);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::ProcessNone()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        ComputeNone(currentNum);
        CopyOut(i, currentNum);
    }
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::ProcessReduce()
{
    // Phase 1: this core accumulates a fp32 partial sum of its loss elements.
    float partial = 0.0f;
    if (blockLength_ > 0) {
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
            CopyIn(i, currentNum);
            LocalTensor<T_COMPUTE> loss = workBuf1_.Get<T_COMPUTE>();
            ComputeLoss(currentNum, loss);
            partial += LocalReduceSum(loss, currentNum);
        }
    }

    // Stage this core's partial sum into its own 32B-aligned workspace slot (blockIdx_*8),
    // so adjacent cores never share a 32B GM write block (see PNLL_WS_CORE_STRIDE).
    LocalTensor<float> stage = reduceBuf_.Get<float>();
    stage.SetValue(0, partial);
    event_t evtSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evtSMTE3);
    WaitFlag<HardEvent::S_MTE3>(evtSMTE3);
    DataCopyExtParams wsParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(wsGm_[blockIdx_ * PNLL_WS_CORE_STRIDE], stage, wsParams);

    SyncAll();

    // Phase 2: block 0 reads every core's partial (strided by 8) and sums them with a scalar
    // loop, then writes the scalar output. Scalar accumulation over usedCoreNum (<= core count)
    // is cheap and avoids reducing over the alignment holes.
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<float> partials = partialBuf_.Get<float>();
    DataCopyExtParams inParams{1, static_cast<uint32_t>(usedCoreNum_ * PNLL_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> inPad{false, 0, 0, 0};
    DataCopyPad(partials, wsGm_, inParams, inPad);
    event_t evtMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evtMTE2S);
    WaitFlag<HardEvent::MTE2_S>(evtMTE2S);

    float total = 0.0f;
    for (int64_t i = 0; i < usedCoreNum_; i++) {
        total += partials.GetValue(i * PNLL_WS_CORE_STRIDE);
    }
    if (reduction_ == PNLL_REDUCTION_MEAN) {
        total *= meanCof_;
    }

    LocalTensor<T_IN> yLocal = outQueueY_.template AllocTensor<T_IN>();
    if constexpr (NEED_CAST) {
        LocalTensor<float> scalarF = reduceBuf_.Get<float>();
        scalarF.SetValue(0, total);
        event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtSV);
        WaitFlag<HardEvent::S_V>(evtSV);
        Cast(yLocal, scalarF, RoundMode::CAST_ROUND, 1);
    } else {
        yLocal.SetValue(0, static_cast<T_IN>(total));
        event_t evtSMTE3b = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evtSMTE3b);
        WaitFlag<HardEvent::S_MTE3>(evtSMTE3b);
    }
    outQueueY_.template EnQue<T_IN>(yLocal);

    LocalTensor<T_IN> yOut = outQueueY_.template DeQue<T_IN>();
    DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(T_IN)), 0, 0, 0};
    DataCopyPad(yGm_, yOut, outParams);
    outQueueY_.FreeTensor(yOut);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::Process()
{
    if (reduction_ == PNLL_REDUCTION_NONE) {
        ProcessNone();
    } else {
        ProcessReduce();
    }
}

} // namespace NsPoissonNllLoss

#endif // POISSON_NLL_LOSS_H
