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
 *   mean : sum / totalNum
 * All of log_input / full / reduction are runtime scalar flags from tilingData.
 *
 * sum/mean reduce over all axes into a single scalar via a deterministic two-phase
 * fp32 reduction: each core accumulates a fp32 partial sum into workspace[blockIdx],
 * SyncAll, then block 0 sums the pnllPartials with Kahan compensation and writes the scalar
 * output. fp32 accumulation matters for fp16 inputs (golden accumulates in fp32).
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
// 一个 fp32 向量寄存器的车道数(256B / 4B)
constexpr int32_t PNLL_VL_FP32 = 64;

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
    __aicore__ inline void CopyIn(int64_t pnllSeg, int64_t pnllSegNum);
    __aicore__ inline void ComputeLoss(int64_t pnllSegNum, LocalTensor<T_COMPUTE>& lossOut);
    __aicore__ inline void ComputeNone(int64_t pnllSegNum);
    __aicore__ inline void CopyOut(int64_t pnllSeg, int64_t pnllSegNum);
    __aicore__ inline void ProcessNone();
    __aicore__ inline void ProcessReduce();
    __aicore__ inline float LocalReduceSum(LocalTensor<T_COMPUTE>& src, int64_t count);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> pnllQueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> pnllQueT_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> pnllQueY_;

    TBuf<QuePosition::VECCALC> workBuf1_;
    TBuf<QuePosition::VECCALC> workBuf2_;
    TBuf<QuePosition::VECCALC> workBuf3_;
    TBuf<QuePosition::VECCALC> pnllReduceBuf_;  // per-tile reduce scratch + this-core partial staging
    TBuf<QuePosition::VECCALC> pnllPartialBuf_; // Phase-2: read back all cores' pnllPartials
    TBuf<QuePosition::VECCALC> cmpBuf_;
    TBuf<QuePosition::VECCALC> pnllCastXBuf_;
    TBuf<QuePosition::VECCALC> pnllCastTBuf_;

    GlobalTensor<T_IN> pnllXGm_;
    GlobalTensor<T_IN> pnllTGm_;
    GlobalTensor<T_IN> pnllYGm_;
    GlobalTensor<float> pnllWsGm_; // partial sums, one fp32 per core (reduction path)

    int64_t pnllBlockLen_ = 0;
    int64_t pnllUbLen_ = 0;
    int64_t pnllTotalNum_ = 0;
    int64_t pnllBlockFactor_ = 0;
    int64_t pnllCoreNum_ = 0;
    int32_t pnllPartialElems_ = 0;
    float eps_ = 1e-8f;
    bool logInput_ = true;
    bool full_ = false;
    uint32_t pnllReduction_ = PNLL_REDUCTION_NONE;
    int32_t pnllBlockIdx_ = 0;
};

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::Init(
    GM_ADDR inputX, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace, const PoissonNllLossTilingData* tilingData)
{
    eps_ = tilingData->eps;
    logInput_ = (tilingData->logInput != 0);
    full_ = (tilingData->full != 0);
    pnllReduction_ = tilingData->reduction;
    pnllTotalNum_ = tilingData->totalNum;
    pnllBlockFactor_ = tilingData->blockFactor;
    pnllUbLen_ = tilingData->ubFactor;
    pnllBlockIdx_ = static_cast<int32_t>(AscendC::GetBlockIdx());
    // Empty tensor (pnllBlockFactor_==0): keep pnllCoreNum_ at 1 so the reduction path stages/reads
    // exactly one (zero) partial instead of doing 0-length DataCopy in Phase 2. block_dim is 1 too.
    pnllCoreNum_ = (pnllBlockFactor_ > 0) ? ((pnllTotalNum_ + pnllBlockFactor_ - 1) / pnllBlockFactor_) : 1;
    pnllPartialElems_ = static_cast<int32_t>(tilingData->partialUbElems);

    int64_t startOffset = pnllBlockFactor_ * static_cast<int64_t>(pnllBlockIdx_);
    int64_t remaining = pnllTotalNum_ - startOffset;
    pnllBlockLen_ = (remaining <= 0) ? 0 : ((remaining > pnllBlockFactor_) ? pnllBlockFactor_ : remaining);

    pnllXGm_.SetGlobalBuffer((__gm__ T_IN*)inputX + startOffset, (pnllBlockLen_ > 0) ? pnllBlockLen_ : 1);
    pnllTGm_.SetGlobalBuffer((__gm__ T_IN*)target + startOffset, (pnllBlockLen_ > 0) ? pnllBlockLen_ : 1);
    if (pnllReduction_ == PNLL_REDUCTION_NONE) {
        pnllYGm_.SetGlobalBuffer((__gm__ T_IN*)loss + startOffset, (pnllBlockLen_ > 0) ? pnllBlockLen_ : 1);
    } else {
        pnllYGm_.SetGlobalBuffer((__gm__ T_IN*)loss, 1);
        int64_t wsElems = (pnllCoreNum_ > 0) ? (pnllCoreNum_ * PNLL_WS_CORE_STRIDE) : PNLL_WS_CORE_STRIDE;
        pnllWsGm_.SetGlobalBuffer((__gm__ float*)workspace, wsElems);
    }

    constexpr int64_t COMPARE_MIN_BYTES = 256;
    constexpr int64_t COMPARE_MIN_ELEMS = COMPARE_MIN_BYTES / static_cast<int64_t>(sizeof(T_COMPUTE));
    int64_t allocElems = (pnllUbLen_ < COMPARE_MIN_ELEMS) ? COMPARE_MIN_ELEMS : pnllUbLen_;

    pipe_.InitBuffer(pnllQueX_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(pnllQueT_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(pnllQueY_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));

    pipe_.InitBuffer(workBuf1_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    pipe_.InitBuffer(workBuf2_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    pipe_.InitBuffer(workBuf3_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    pipe_.InitBuffer(pnllReduceBuf_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    if (pnllReduction_ != PNLL_REDUCTION_NONE) {
        pipe_.InitBuffer(pnllPartialBuf_,
                         static_cast<int64_t>(pnllPartialElems_) * static_cast<int64_t>(sizeof(float)));
    }

    int64_t cmpBufSize = ((allocElems / 8 + 255) / 256) * 256;
    if (cmpBufSize < 256) {
        cmpBufSize = 256;
    }
    pipe_.InitBuffer(cmpBuf_, cmpBufSize);

    if constexpr (NEED_CAST) {
        pipe_.InitBuffer(pnllCastXBuf_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
        pipe_.InitBuffer(pnllCastTBuf_, allocElems * static_cast<int64_t>(sizeof(T_COMPUTE)));
    }
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::CopyIn(int64_t pnllSeg, int64_t pnllSegNum)
{
    LocalTensor<T_IN> pnllXLocal = pnllQueX_.template AllocTensor<T_IN>();
    LocalTensor<T_IN> pnllTLocal = pnllQueT_.template AllocTensor<T_IN>();
    DataCopyExtParams pnllCopyParams;
    pnllCopyParams.blockCount = 1;
    pnllCopyParams.blockLen = static_cast<uint32_t>(pnllSegNum * static_cast<int64_t>(sizeof(T_IN)));
    pnllCopyParams.srcStride = 0;
    pnllCopyParams.dstStride = 0;
    DataCopyPadExtParams<T_IN> pnllPadParams{false, 0, 0, 0};
    DataCopyPad(pnllXLocal, pnllXGm_[pnllSeg * pnllUbLen_], pnllCopyParams, pnllPadParams);
    DataCopyPad(pnllTLocal, pnllTGm_[pnllSeg * pnllUbLen_], pnllCopyParams, pnllPadParams);
    pnllQueX_.EnQue(pnllXLocal);
    pnllQueT_.EnQue(pnllTLocal);
}

// Compute the elementwise loss into lossOut (a fp32 buffer). Consumes queued x/t tensors.
template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::ComputeLoss(int64_t pnllSegNum,
                                                                                       LocalTensor<T_COMPUTE>& lossOut)
{
    LocalTensor<T_IN> pnllXLocal = pnllQueX_.template DeQue<T_IN>();
    LocalTensor<T_IN> pnllTLocal = pnllQueT_.template DeQue<T_IN>();

    LocalTensor<T_COMPUTE> tmp = workBuf2_.Get<T_COMPUTE>();
    LocalTensor<T_COMPUTE> tmp2 = workBuf3_.Get<T_COMPUTE>();
    LocalTensor<uint8_t> cmpMask = cmpBuf_.Get<uint8_t>();

    uint32_t elemCount = static_cast<uint32_t>(pnllSegNum);
    constexpr uint32_t COMPARE_ALIGN_ELEMENTS = 256 / static_cast<uint32_t>(sizeof(T_COMPUTE));
    uint32_t alignedCount = (elemCount + COMPARE_ALIGN_ELEMENTS - 1) / COMPARE_ALIGN_ELEMENTS * COMPARE_ALIGN_ELEMENTS;

    LocalTensor<T_COMPUTE> xF;
    LocalTensor<T_COMPUTE> tF;
    if constexpr (NEED_CAST) {
        xF = pnllCastXBuf_.Get<T_COMPUTE>();
        tF = pnllCastTBuf_.Get<T_COMPUTE>();
        Cast(xF, pnllXLocal, RoundMode::CAST_NONE, elemCount);
        Cast(tF, pnllTLocal, RoundMode::CAST_NONE, elemCount);
    } else {
        xF = pnllXLocal.template ReinterpretCast<T_COMPUTE>();
        tF = pnllTLocal.template ReinterpretCast<T_COMPUTE>();
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

    pnllQueX_.FreeTensor(pnllXLocal);
    pnllQueT_.FreeTensor(pnllTLocal);
}

// reduction=none: compute loss and write it out elementwise (same shape as input).
template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::ComputeNone(int64_t pnllSegNum)
{
    LocalTensor<T_COMPUTE> loss = workBuf1_.Get<T_COMPUTE>();
    ComputeLoss(pnllSegNum, loss);

    uint32_t elemCount = static_cast<uint32_t>(pnllSegNum);
    LocalTensor<T_IN> pnllYLocal = pnllQueY_.template AllocTensor<T_IN>();
    if constexpr (NEED_CAST) {
        Cast(pnllYLocal, loss, RoundMode::CAST_ROUND, elemCount);
    } else {
        LocalTensor<T_COMPUTE> outF = pnllYLocal.template ReinterpretCast<T_COMPUTE>();
        Adds(outF, loss, static_cast<T_COMPUTE>(0), elemCount);
    }
    pnllQueY_.template EnQue<T_IN>(pnllYLocal);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::CopyOut(int64_t pnllSeg, int64_t pnllSegNum)
{
    LocalTensor<T_IN> pnllYLocal = pnllQueY_.template DeQue<T_IN>();
    DataCopyExtParams pnllCopyParams;
    pnllCopyParams.blockCount = 1;
    pnllCopyParams.blockLen = static_cast<uint32_t>(pnllSegNum * static_cast<int64_t>(sizeof(T_IN)));
    pnllCopyParams.srcStride = 0;
    pnllCopyParams.dstStride = 0;
    DataCopyPad(pnllYGm_[pnllSeg * pnllUbLen_], pnllYLocal, pnllCopyParams);
    pnllQueY_.FreeTensor(pnllYLocal);
}

// Whole-reduce a fp32 LocalTensor of `count` elements to a single fp32 value.
// Uses ReduceSum(dst, src, work, count) — the proven arch35 whole-reduce form.
template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline float KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::LocalReduceSum(LocalTensor<T_COMPUTE>& src,
                                                                                           int64_t count)
{
    LocalTensor<T_COMPUTE> red = pnllReduceBuf_.Get<T_COMPUTE>();
    ReduceSum(red, src, red, static_cast<int32_t>(count));
    event_t evtVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evtVS);
    WaitFlag<HardEvent::V_S>(evtVS);
    return red.GetValue(0);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::ProcessNone()
{
    if (pnllBlockLen_ <= 0) {
        return;
    }
    int64_t pnllLoops = (pnllBlockLen_ + pnllUbLen_ - 1) / pnllUbLen_;
    for (int64_t i = 0; i < pnllLoops; i++) {
        int64_t pnllSegNum = (i == (pnllLoops - 1)) ? (pnllBlockLen_ - pnllUbLen_ * i) : pnllUbLen_;
        CopyIn(i, pnllSegNum);
        ComputeNone(pnllSegNum);
        CopyOut(i, pnllSegNum);
    }
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::ProcessReduce()
{
    // Phase 1: this core accumulates a fp32 partial sum of its loss elements.
    float partial = 0.0f;
    if (pnllBlockLen_ > 0) {
        int64_t pnllLoops = (pnllBlockLen_ + pnllUbLen_ - 1) / pnllUbLen_;
        for (int64_t i = 0; i < pnllLoops; i++) {
            int64_t pnllSegNum = (i == (pnllLoops - 1)) ? (pnllBlockLen_ - pnllUbLen_ * i) : pnllUbLen_;
            CopyIn(i, pnllSegNum);
            LocalTensor<T_COMPUTE> loss = workBuf1_.Get<T_COMPUTE>();
            ComputeLoss(pnllSegNum, loss);
            partial += LocalReduceSum(loss, pnllSegNum);
        }
    }

    // Stage this core's partial sum into its own 32B-aligned workspace slot (pnllBlockIdx_*8),
    // so adjacent cores never share a 32B GM write block (see PNLL_WS_CORE_STRIDE).
    // 写整 32B 块(partial + 7 个 0): 跨核 GM 写本就是 32B 粒度, 补零后 phase 2 能一次连续读入
    // 直接做矢量累加(补零车道加 0 不改变结果), 不必逐核 GetValue。
    LocalTensor<float> pnllStage = pnllReduceBuf_.Get<float>();
    for (int32_t k = 0; k < PNLL_WS_CORE_STRIDE; k++) {
        pnllStage.SetValue(k, 0.0f);
    }
    pnllStage.SetValue(0, partial);
    event_t evtSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evtSMTE3);
    WaitFlag<HardEvent::S_MTE3>(evtSMTE3);
    DataCopyExtParams pnllWsParams{1, static_cast<uint32_t>(PNLL_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPad(pnllWsGm_[pnllBlockIdx_ * PNLL_WS_CORE_STRIDE], pnllStage, pnllWsParams);

    SyncAll();

    // Phase 2: block 0 reads every core's partial (strided by 8) and sums them with a scalar
    // Kahan loop, then writes the scalar output. Scalar accumulation over usedCoreNum
    // (<= core count) is cheap and avoids reducing over the alignment holes.
    if (pnllBlockIdx_ != 0) {
        return;
    }
    LocalTensor<float> pnllPartials = pnllPartialBuf_.Get<float>();
    const uint16_t pnllRounds = static_cast<uint16_t>(pnllPartialElems_ / PNLL_VL_FP32);
    DataCopyExtParams pnllInParams{1, static_cast<uint32_t>(pnllCoreNum_ * PNLL_WS_CORE_STRIDE * sizeof(float)), 0, 0,
                                   0};
    DataCopyPadExtParams<float> pnllInPad{false, 0, 0, 0};
    DataCopyPad(pnllPartials, pnllWsGm_, pnllInParams, pnllInPad);
    event_t evtMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evtMTE2V);
    WaitFlag<HardEvent::MTE2_V>(evtMTE2V);

    // 只清**补零区**: 矢量合并整轮读满 PNLL_VL_FP32 车道、收尾标量循环也按 stride 读满
    // PNLL_VL_FP32/PNLL_WS_CORE_STRIDE 条车道, 而 DataCopyPad 只搬入 pnllCoreNum_ 个槽; 余下车道若不清零读到的是
    // UB 残留(可能含 ±inf/NaN), 会被当作 partial 累加 —— 核数越少无效车道越多。
    // 放在搬入之后只清尾巴, 比"搬入前清整段"省一次 V->MTE2 同步和大部分清零量(实测小 shape 的
    // sum 快回 4~5%); Duplicate 与下面的矢量循环同为 V 流水, 按序发射无需额外同步。
    const int32_t copiedElems = static_cast<int32_t>(pnllCoreNum_ * PNLL_WS_CORE_STRIDE);
    if (pnllPartialElems_ > copiedElems) {
        Duplicate(pnllPartials[copiedElems], 0.0f, pnllPartialElems_ - copiedElems);
    }

    // ── 跨核合并: **矢量(regbase)形态的 Kahan 补偿累加** ───────────────────────────────
    // 每核占一个 32B 块(首元素有效、其余为 0), 一次寄存器载入(64 车道)覆盖 8 个核; 64 核 = 8 轮,
    // 即 8 车道并行、每车道顺序累加 8 个 partial, Kahan 的补偿项覆盖这 8 步。补零车道不产生补偿量。
    // 原写法是 block 0 上串行 pnllCoreNum_(最多 64)次的标量循环, 实测占小 shape 整核时延四成。
    // inf/nan: 补偿量 c = (t - sum) - y 在 t 为 ±inf 时算出 NaN, 会污染后续每一轮, 把本该 inf 的
    // 和变成 nan(单核不复现、多核必现)。自比 Compare<EQ>(c, c) + Select 把 NaN 车道的补偿量清零,
    // 只清补偿量、不动 sum, 真 nan 仍如实传播。(同款写法见 norm/instance_norm_grad。)
    __local_mem__ float* pnllUbAddr = (__local_mem__ float*)pnllPartials.GetPhyAddr();
    {
        AscendC::Reg::RegTensor<float> pnllSum;
        AscendC::Reg::RegTensor<float> pnllComp;
        AscendC::Reg::RegTensor<float> pnllZero;
        AscendC::Reg::RegTensor<float> pnllPartialReg;
        AscendC::Reg::RegTensor<float> pnllErr;
        AscendC::Reg::RegTensor<float> pnllAcc;
        AscendC::Reg::RegTensor<float> pnllDelta;
        AscendC::Reg::MaskReg pnllMask;
        AscendC::Reg::MaskReg pnllFiniteMask;
        uint32_t pnllLaneCnt = static_cast<uint32_t>(pnllRounds) * PNLL_VL_FP32;
        __VEC_SCOPE__
        {
            pnllMask = AscendC::Reg::UpdateMask<float>(pnllLaneCnt);
            AscendC::Reg::Duplicate(pnllSum, 0.0f, pnllMask);
            AscendC::Reg::Duplicate(pnllComp, 0.0f, pnllMask);
            AscendC::Reg::Duplicate(pnllZero, 0.0f, pnllMask);
            for (uint16_t r = 0; r < pnllRounds; ++r) {
                AscendC::Reg::DataCopy(pnllPartialReg, pnllUbAddr + r * PNLL_VL_FP32);
                AscendC::Reg::Sub(pnllErr, pnllPartialReg, pnllComp, pnllMask);
                AscendC::Reg::Add(pnllAcc, pnllSum, pnllErr, pnllMask);
                AscendC::Reg::Sub(pnllDelta, pnllAcc, pnllSum, pnllMask);
                AscendC::Reg::Sub(pnllComp, pnllDelta, pnllErr, pnllMask);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pnllFiniteMask, pnllComp, pnllComp, pnllMask);
                AscendC::Reg::Select(pnllComp, pnllComp, pnllZero, pnllFiniteMask);
                AscendC::Reg::Move(pnllSum, pnllAcc, pnllMask);
            }
            AscendC::Reg::DataCopy(pnllUbAddr, pnllSum, pnllMask);
        }
    }
    // 收尾: 只对 PNLL_VL_FP32/PNLL_WS_CORE_STRIDE 条有效车道做标量 Kahan(核数 64 时是 8 步, 不是原来的 64 步)。
    // 车道内已补偿过各自的顺序累加, 这里再补偿车道之间 —— 实测(同一批 partial 配对比较)
    // 与原标量 Kahan 6 例中 5 例逐位相同; 若这里改用无补偿的树形规约, 会有 3/6 变差。
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float total = 0.0f;
    float pnllCarry = 0.0f;
    for (int32_t lane = 0; lane < PNLL_VL_FP32; lane += static_cast<int32_t>(PNLL_WS_CORE_STRIDE)) {
        float y = pnllPartials.GetValue(lane) - pnllCarry;
        float t = total + y;
        pnllCarry = (t - total) - y;
        if (__isinf(t) || __isnan(t)) {
            pnllCarry = 0.0f;
        }
        total = t;
    }
    if (pnllReduction_ == PNLL_REDUCTION_MEAN) {
        // 用除法而不是乘 1/N:N 不是 2 的幂时 1/N 在 fp32 里存不下,先舍一次,
        // 乘的时候再舍一次,双重舍入会多丢半个 ULP。IEEE754 的除法只舍一次,
        // 保证正确舍入。空 tensor 时 pnllTotalNum_=0,0.0f/0.0f 仍得到 nan,语义不变。
        total /= static_cast<float>(pnllTotalNum_);
    }

    LocalTensor<T_IN> pnllYLocal = pnllQueY_.template AllocTensor<T_IN>();
    if constexpr (NEED_CAST) {
        LocalTensor<float> scalarF = pnllReduceBuf_.Get<float>();
        scalarF.SetValue(0, total);
        event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtSV);
        WaitFlag<HardEvent::S_V>(evtSV);
        Cast(pnllYLocal, scalarF, RoundMode::CAST_ROUND, 1);
    } else {
        pnllYLocal.SetValue(0, static_cast<T_IN>(total));
        event_t evtSMTE3b = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evtSMTE3b);
        WaitFlag<HardEvent::S_MTE3>(evtSMTE3b);
    }
    pnllQueY_.template EnQue<T_IN>(pnllYLocal);

    LocalTensor<T_IN> pnllYOut = pnllQueY_.template DeQue<T_IN>();
    DataCopyExtParams pnllOutParams{1, static_cast<uint32_t>(sizeof(T_IN)), 0, 0, 0};
    DataCopyPad(pnllYGm_, pnllYOut, pnllOutParams);
    pnllQueY_.FreeTensor(pnllYOut);
}

template <typename T_IN, typename T_COMPUTE, int BUFFER_MODE>
__aicore__ inline void KernelPoissonNllLoss<T_IN, T_COMPUTE, BUFFER_MODE>::Process()
{
    if (pnllReduction_ == PNLL_REDUCTION_NONE) {
        ProcessNone();
    } else {
        ProcessReduce();
    }
}

} // namespace NsPoissonNllLoss

#endif // POISSON_NLL_LOSS_H
