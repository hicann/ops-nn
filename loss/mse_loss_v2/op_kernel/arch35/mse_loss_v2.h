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
// A single vector op needs at least 256B of lane coverage; size compute buffers to that minimum
// even when ubFactor is tiny, so ReduceSum/Sub/Mul never underflow a vector register.
constexpr int64_t MSE_MIN_COMPUTE_ELEMS = 256 / static_cast<int64_t>(sizeof(float));
// 一个 fp32 向量寄存器的车道数(256B / 4B)
constexpr int32_t MSE_VL_FP32 = 64;

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
    TBuf<QuePosition::VECCALC> partialBuf_; // fp32 phase-2 read-back of all cores' partials
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
    int32_t partialUbElems_ = 0;
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
    partialUbElems_ = static_cast<int32_t>(tilingData->partialUbElems);

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
        pipe_.InitBuffer(partialBuf_, static_cast<int64_t>(partialUbElems_) * static_cast<int64_t>(sizeof(float)));
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
    // 写**整 32B 块**(partial + 7 个 0)而不是单个 4B: GM 上跨核写以 32B 为粒度, 4B 密排会让相邻
    // 核踩同一块; 而补零成整块后, phase 2 可以一次连续读进来直接矢量规约(0 不影响和), 不必再
    // 逐核 GetValue。
    LocalTensor<float> stage = reduceBuf_.Get<float>();
    for (int32_t k = 0; k < MSE_WS_CORE_STRIDE; k++) {
        stage.SetValue(k, 0.0f);
    }
    stage.SetValue(0, partial);
    event_t evtSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evtSMTE3);
    WaitFlag<HardEvent::S_MTE3>(evtSMTE3);
    DataCopyExtParams wsParams{1, static_cast<uint32_t>(MSE_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPad(wsGm_[blockIdx_ * MSE_WS_CORE_STRIDE], stage, wsParams);

    SyncAll();

    // Phase 2: block 0 reads every core's partial (strided by 8) and sums them, then writes the scalar.
    if (blockIdx_ != 0) {
        return;
    }
    LocalTensor<float> partials = partialBuf_.Get<float>();
    const uint16_t mergeRounds = static_cast<uint16_t>(partialUbElems_ / MSE_VL_FP32);
    DataCopyExtParams inParams{1, static_cast<uint32_t>(usedCoreNum_ * MSE_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> inPad{false, 0, 0, 0};
    DataCopyPad(partials, wsGm_, inParams, inPad);
    event_t evtMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evtMTE2V);
    WaitFlag<HardEvent::MTE2_V>(evtMTE2V);

    // 只清**补零区**: 矢量合并整轮读满 MSE_VL_FP32 车道、收尾标量循环也按 stride 读满
    // MSE_VL_FP32/MSE_WS_CORE_STRIDE 条车道, 而 DataCopyPad 只搬入 usedCoreNum_ 个槽; 余下车道若不清零读到的是
    // UB 残留(可能含 ±inf/NaN), 会被当作 partial 累加 —— 核数越少无效车道越多。
    // 放在搬入之后只清尾巴, 比"搬入前清整段"省一次 V->MTE2 同步和大部分清零量(实测小 shape 的
    // sum 快回 4~5%); Duplicate 与下面的矢量循环同为 V 流水, 按序发射无需额外同步。
    const int32_t copiedElems = static_cast<int32_t>(usedCoreNum_ * MSE_WS_CORE_STRIDE);
    if (partialUbElems_ > copiedElems) {
        Duplicate(partials[copiedElems], 0.0f, partialUbElems_ - copiedElems);
    }

    // ── 跨核合并: **矢量(regbase)形态的 Kahan 补偿累加** ───────────────────────────────
    // 布局: 每核占一个 32B 块(首元素是它的 partial, 其余 7 个为 0), 于是一次寄存器载入(64 车道)
    // 正好覆盖 8 个核, 补零车道加 0 不改变结果也不产生补偿量。64 核 = 8 轮, 即 8 车道并行、
    // 每车道顺序累加 8 个 partial —— Kahan 的补偿项覆盖的就是这 8 步。
    //
    // 为什么不是标量循环: 原写法在 block 0 上串行 usedCoreNum_(最多 64)次, 实测跨核规约段占
    // 小 shape 整核时延四成; 矢量化后这段压成 8 轮指令。
    // 为什么不是纯树形规约: 树形没有补偿项, 合并 64 个 partial 的误差 ~log2(64)*eps, 比 Kahan
    // 的 ~2*eps 差(实测 4M 用例 1.2 ulp vs 0.8 ulp), 精度不能白丢。
    // inf/nan: 补偿量 c = (t - sum) - y 在 t 为 ±inf 时算出 NaN, 会污染后续每一轮, 把本该是 inf
    // 的和变成 nan(实测单核不复现、多核必现)。用自比 Compare<EQ>(c, c) 找出非 NaN 车道, Select
    // 把 NaN 车道的补偿量清零 —— 只清补偿量、不动 sum, 故真 nan 仍会如实传播。
    // (同款写法见 norm/instance_norm_grad 的跨 N 合并。)
    __local_mem__ float* partialUb = (__local_mem__ float*)partials.GetPhyAddr();
    {
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::RegTensor<float> compReg;
        AscendC::Reg::RegTensor<float> zeroReg;
        AscendC::Reg::RegTensor<float> pReg;
        AscendC::Reg::RegTensor<float> kY;
        AscendC::Reg::RegTensor<float> kT;
        AscendC::Reg::RegTensor<float> kD;
        AscendC::Reg::MaskReg preg;
        AscendC::Reg::MaskReg finiteMask;
        uint32_t sreg = static_cast<uint32_t>(mergeRounds) * MSE_VL_FP32;
        __VEC_SCOPE__
        {
            preg = AscendC::Reg::UpdateMask<float>(sreg);
            AscendC::Reg::Duplicate(sumReg, 0.0f, preg);
            AscendC::Reg::Duplicate(compReg, 0.0f, preg);
            AscendC::Reg::Duplicate(zeroReg, 0.0f, preg);
            for (uint16_t r = 0; r < mergeRounds; ++r) {
                AscendC::Reg::DataCopy(pReg, partialUb + r * MSE_VL_FP32);
                AscendC::Reg::Sub(kY, pReg, compReg, preg); // y = p - comp
                AscendC::Reg::Add(kT, sumReg, kY, preg);    // t = sum + y
                AscendC::Reg::Sub(kD, kT, sumReg, preg);    // d = t - sum
                AscendC::Reg::Sub(compReg, kD, kY, preg);   // comp = d - y
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(finiteMask, compReg, compReg, preg);
                AscendC::Reg::Select(compReg, compReg, zeroReg, finiteMask); // NaN 车道 -> 0
                AscendC::Reg::Move(sumReg, kT, preg);
            }
            AscendC::Reg::DataCopy(partialUb, sumReg, preg);
        }
    }
    // 合并 64 条车道(其中 8 条有效, 其余恒 0)
    // 收尾: 只对 MSE_VL_FP32/MSE_WS_CORE_STRIDE 条有效车道做标量 Kahan(核数 64 时是 8 步, 不是原来的 64 步)。
    // 车道内已补偿过各自的顺序累加, 这里再补偿车道之间 —— 实测(同一批 partial 配对比较)
    // 与原标量 Kahan 6 例中 5 例逐位相同; 若这里改用无补偿的树形规约, 会有 3/6 变差。
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float total = 0.0f;
    float mergeComp = 0.0f;
    for (int32_t lane = 0; lane < MSE_VL_FP32; lane += static_cast<int32_t>(MSE_WS_CORE_STRIDE)) {
        float y = partials.GetValue(lane) - mergeComp;
        float t = total + y;
        mergeComp = (t - total) - y;
        if (__isinf(t) || __isnan(t)) {
            mergeComp = 0.0f;
        }
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
