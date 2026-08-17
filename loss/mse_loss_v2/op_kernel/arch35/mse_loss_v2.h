/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mse_loss_v2.h
 * \brief MSELossV2 kernel class definition (arch35 / Ascend950)
 *
 * Computes the mean-squared-error loss:  l = (input - target)^2, then reduces:
 *   reduction=none : write l elementwise (same shape as input)
 *   reduction=sum  : sum of all l into a scalar
 *   reduction=mean : (1/N) * sum of all l into a scalar
 * reduction is a runtime scalar field from the tiling data (not a tiling key), so one template
 * instance per (dtype, buffer_mode) covers all three.
 *
 * For fp16/bf16 inputs the squared difference and the reduction accumulate in fp32 (the golden
 * torch.nn.functional.mse_loss accumulates in fp32), then the result is cast back to the input
 * dtype with round-to-nearest-even (CAST_RINT).
 *
 * sum/mean reduce over all axes into a single scalar via a deterministic two-phase fp32
 * reduction: each core accumulates a fp32 partial sum into its own 32B-aligned workspace slot,
 * SyncAll, then block 0 sums the partials and writes the scalar output.
 */

#ifndef MSE_LOSS_V2_ARCH35_H
#define MSE_LOSS_V2_ARCH35_H

#include "kernel_operator.h"
#include "mse_loss_v2_tiling_data.h"
#ifndef DTYPE_X
#include "kernel_tiling/kernel_tiling.h"
#include "mse_loss_v2_tiling_key.h"
#endif

namespace NsMseLossV2 {

using namespace AscendC;

constexpr uint32_t MSE_REDUCTION_NONE = 0;
constexpr uint32_t MSE_REDUCTION_SUM = 1;
constexpr uint32_t MSE_REDUCTION_MEAN = 2;
// Per-core workspace partial sums must be 32B(=8 fp32)-block aligned: GM write transactions are
// atomic at 32B granularity, so adjacent cores writing dense 4B slots into one block would race.
// Give each core its own block. (aligns with loss/poisson_nll_loss, norm/batch_norm_grad_v3)
constexpr int32_t MSE_WS_CORE_STRIDE = 8;
// Max vector cores on Ascend950 (vector_core_cnt <= 64). Phase-2 reads usedCoreNum*8 fp32 partials
// back into a fixed buffer of this capacity, decoupled from ubFactor so it never overflows.
constexpr int32_t MSE_MAX_CORE_NUM = 64;
constexpr int32_t MSE_PARTIAL_BUF_ELEMS = MSE_MAX_CORE_NUM * MSE_WS_CORE_STRIDE; // 512 fp32 = 2KB
// A single vector op needs at least 256B of lane coverage; size compute buffers to that minimum
// even when ubFactor is tiny, so ReduceSum/Sub/Mul never underflow a vector register.
constexpr int64_t MSE_MIN_COMPUTE_ELEMS = 256 / static_cast<int64_t>(sizeof(float));

template <typename T_IN, int BUFFER_MODE>
class MseLossV2 {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    // fp16/bf16 promote to fp32 for compute + reduction (golden accumulates in fp32).
    static constexpr bool NEED_CAST = !std::is_same<T_IN, float>::value;

public:
    __aicore__ inline MseLossV2() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR output, GM_ADDR workspace,
                                const MSELossV2Arch35TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void ComputeSquaredDiff(int64_t currentNum, LocalTensor<float>& dst);
    __aicore__ inline void ComputeNone(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void ProcessNone();
    __aicore__ inline void ProcessReduce();
    __aicore__ inline float LocalReduceSum(LocalTensor<float>& src, int64_t count);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueT_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    TBuf<QuePosition::VECCALC> lossBuf_;    // fp32 (input-target)^2
    TBuf<QuePosition::VECCALC> reduceBuf_;  // fp32 ReduceSum scratch + this-core partial staging
    TBuf<QuePosition::VECCALC> partialBuf_; // fp32 phase-2 read-back of all cores' partials (fixed 2KB)
    TBuf<QuePosition::VECCALC> castXBuf_;   // fp32 cast of input  (NEED_CAST only)
    TBuf<QuePosition::VECCALC> castTBuf_;   // fp32 cast of target (NEED_CAST only)

    GlobalTensor<T_IN> xGm_;
    GlobalTensor<T_IN> tGm_;
    GlobalTensor<T_IN> yGm_;
    GlobalTensor<float> wsGm_; // partial sums, one fp32 slot (stride 8) per core (reduction path)

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t usedCoreNum_ = 0;
    uint32_t reduction_ = MSE_REDUCTION_NONE;
    int32_t blockIdx_ = 0;
};

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::Init(GM_ADDR input, GM_ADDR target, GM_ADDR output,
                                                          GM_ADDR workspace,
                                                          const MSELossV2Arch35TilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    reduction_ = tilingData->reduction;
    blockIdx_ = static_cast<int32_t>(AscendC::GetBlockIdx());
    usedCoreNum_ = (blockFactor_ > 0) ? ((totalNum_ + blockFactor_ - 1) / blockFactor_) : 0;

    int64_t startOffset = blockFactor_ * static_cast<int64_t>(blockIdx_);
    int64_t remaining = totalNum_ - startOffset;
    blockLength_ = (remaining <= 0) ? 0 : ((remaining > blockFactor_) ? blockFactor_ : remaining);

    xGm_.SetGlobalBuffer((__gm__ T_IN*)input + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    tGm_.SetGlobalBuffer((__gm__ T_IN*)target + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    if (reduction_ == MSE_REDUCTION_NONE) {
        yGm_.SetGlobalBuffer((__gm__ T_IN*)output + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    } else {
        yGm_.SetGlobalBuffer((__gm__ T_IN*)output, 1);
        int64_t wsElems = (usedCoreNum_ > 0) ? (usedCoreNum_ * MSE_WS_CORE_STRIDE) : MSE_WS_CORE_STRIDE;
        wsGm_.SetGlobalBuffer((__gm__ float*)workspace, wsElems);
    }

    int64_t allocElems = (ubLength_ < MSE_MIN_COMPUTE_ELEMS) ? MSE_MIN_COMPUTE_ELEMS : ubLength_;

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(inQueueT_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(lossBuf_, allocElems * static_cast<int64_t>(sizeof(float)));
    pipe_.InitBuffer(reduceBuf_, allocElems * static_cast<int64_t>(sizeof(float)));
    if (reduction_ != MSE_REDUCTION_NONE) {
        pipe_.InitBuffer(partialBuf_, MSE_PARTIAL_BUF_ELEMS * static_cast<int64_t>(sizeof(float)));
    }
    if constexpr (NEED_CAST) {
        pipe_.InitBuffer(castXBuf_, allocElems * static_cast<int64_t>(sizeof(float)));
        pipe_.InitBuffer(castTBuf_, allocElems * static_cast<int64_t>(sizeof(float)));
    }
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
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

// Compute (input - target)^2 in fp32 into dst. Consumes the queued x/t tensors.
template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ComputeSquaredDiff(int64_t currentNum, LocalTensor<float>& dst)
{
    LocalTensor<T_IN> xLocal = inQueueX_.template DeQue<T_IN>();
    LocalTensor<T_IN> tLocal = inQueueT_.template DeQue<T_IN>();
    uint32_t n = static_cast<uint32_t>(currentNum);

    if constexpr (NEED_CAST) {
        LocalTensor<float> xF = castXBuf_.Get<float>();
        LocalTensor<float> tF = castTBuf_.Get<float>();
        Cast(xF, xLocal, RoundMode::CAST_NONE, n);
        Cast(tF, tLocal, RoundMode::CAST_NONE, n);
        Sub(dst, xF, tF, n);
        Mul(dst, dst, dst, n);
    } else {
        Sub(dst, xLocal, tLocal, n);
        Mul(dst, dst, dst, n);
    }

    inQueueX_.FreeTensor(xLocal);
    inQueueT_.FreeTensor(tLocal);
}

// reduction=none: compute the loss and write it out elementwise (same shape as input).
template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ComputeNone(int64_t currentNum)
{
    LocalTensor<float> loss = lossBuf_.Get<float>();
    ComputeSquaredDiff(currentNum, loss);

    uint32_t n = static_cast<uint32_t>(currentNum);
    LocalTensor<T_IN> yLocal = outQueueY_.template AllocTensor<T_IN>();
    if constexpr (NEED_CAST) {
        Cast(yLocal, loss, RoundMode::CAST_RINT, n);
    } else {
        LocalTensor<float> yF = yLocal.template ReinterpretCast<float>();
        Adds(yF, loss, 0.0f, n);
    }
    outQueueY_.template EnQue<T_IN>(yLocal);
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
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
template <typename T_IN, int BUFFER_MODE>
__aicore__ inline float MseLossV2<T_IN, BUFFER_MODE>::LocalReduceSum(LocalTensor<float>& src, int64_t count)
{
    LocalTensor<float> red = reduceBuf_.Get<float>();
    ReduceSum(red, src, red, static_cast<int32_t>(count));
    event_t evtVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evtVS);
    WaitFlag<HardEvent::V_S>(evtVS);
    return red.GetValue(0);
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ProcessNone()
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

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ProcessReduce()
{
    // Phase 1: this core accumulates a fp32 partial sum of its squared-diff elements.
    float partial = 0.0f;
    if (blockLength_ > 0) {
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
            CopyIn(i, currentNum);
            LocalTensor<float> loss = lossBuf_.Get<float>();
            ComputeSquaredDiff(currentNum, loss);
            partial += LocalReduceSum(loss, currentNum);
        }
    }

    // Stage this core's partial into its own 32B-aligned workspace slot (blockIdx_*8).
    LocalTensor<float> stage = reduceBuf_.Get<float>();
    stage.SetValue(0, partial);
    event_t evtSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evtSMTE3);
    WaitFlag<HardEvent::S_MTE3>(evtSMTE3);
    DataCopyExtParams wsParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(wsGm_[blockIdx_ * MSE_WS_CORE_STRIDE], stage, wsParams);

    SyncAll();

    // Phase 2: block 0 reads every core's partial (strided by 8) and sums them, then writes the scalar.
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<float> partials = partialBuf_.Get<float>();
    DataCopyExtParams inParams{1, static_cast<uint32_t>(usedCoreNum_ * MSE_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> inPad{false, 0, 0, 0};
    DataCopyPad(partials, wsGm_, inParams, inPad);
    event_t evtMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evtMTE2S);
    WaitFlag<HardEvent::MTE2_S>(evtMTE2S);

    // Kahan 补偿累加:裸的 total += p 每加一次都会把加数末位舍掉,total 越大丢得越多,
    // 且不会互相抵消。补偿法把每次丢掉的零头算出来,下一轮加回去:
    //     y = p - comp             先补上轮丢的
    //     t = total + y            这一步会丢零头
    //     comp = (t - total) - y   实际加进去的 减 本来要加的 = 这次丢的零头
    //     total = t
    float total = 0.0f;
    float comp = 0.0f;
    for (int64_t i = 0; i < usedCoreNum_; i++) {
        float y = partials.GetValue(i * MSE_WS_CORE_STRIDE) - comp;
        float t = total + y;
        comp = (t - total) - y;
        total = t;
    }
    if (reduction_ == MSE_REDUCTION_MEAN) {
        // 用除法而不是乘 1/N:N 不是 2 的幂时 1/N 在 fp32 里存不下,先舍 1/N 再舍乘积
        // 是双重舍入;IEEE754 除法只舍一次。本算子 tiling 已拒收空 tensor,N>0。
        total /= static_cast<float>(totalNum_);
    }

    LocalTensor<T_IN> yLocal = outQueueY_.template AllocTensor<T_IN>();
    if constexpr (NEED_CAST) {
        LocalTensor<float> scalarF = reduceBuf_.Get<float>();
        scalarF.SetValue(0, total);
        event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtSV);
        WaitFlag<HardEvent::S_V>(evtSV);
        Cast(yLocal, scalarF, RoundMode::CAST_RINT, 1);
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

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::Process()
{
    if (reduction_ == MSE_REDUCTION_NONE) {
        ProcessNone();
    } else {
        ProcessReduce();
    }
}

} // namespace NsMseLossV2

#endif // MSE_LOSS_V2_ARCH35_H
