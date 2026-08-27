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
 * \file apply_came_part4.h
 * \brief ApplyCamePart4 kernel class definition (arch35)
 *
 * Aligned to canndev ApplyCamePart4 (apply_came_part4_float32.h / _float16.h / _pre.h),
 * merged into one template class:
 *   - T = float:            compute directly (canndev float32 path)
 *   - T = half / bfloat16:  cast to fp32 after CopyIn, compute in fp32, cast back before CopyOut
 *
 * Update formulas (n = len(r), m = len(c); N/M from global_shape if given, else n/m):
 *   sum_r  = sum(r_in)                       (computed in-kernel when the optional input is absent)
 *   r_out  = beta3 * r_in + (1-beta3)/M * sum_u_r
 *   c_out  = beta3 * c_in + (1-beta3)/N * sum_u_c
 *   denom  = beta3 * sum_r / N + (1-beta3) * sum_u_rc / (M*N)
 *   param_out = (1 - lr*weight_decay) * param_in - lr * m / sqrt(r_out * c_out / denom)
 *
 * v1 simplification vs canndev: the r*c outer product always uses the per-row
 * Muls fallback (canndev CalcRcCycleMode); the ConfusionTranspose fast path is
 * dropped (in canndev it is effectively unreachable because rRcNumPerLoop < 128).
 */
#ifndef APPLY_CAME_PART4_H
#define APPLY_CAME_PART4_H

#include "kernel_operator.h"
#include "apply_came_part4_tiling_data.h"
#include "apply_came_part4_tiling_key.h"

namespace NsApplyCamePart4 {

using namespace AscendC;

template <typename T>
class ApplyCamePart4 {
    static constexpr bool IS_REDUCED = !AscendC::IsSameType<T, float>::value;
    static constexpr int64_t ONE_BLK = 32;
    // Pre phase (sum_r reduction) chunk sizes, aligned to canndev
    static constexpr int64_t PRE_MAX_ONCE_NUM = IS_REDUCED ? (56 * 1024 / sizeof(T)) : (128 * 1024 / sizeof(float));
    static constexpr int64_t PRE_CAST_MAX_NUM = 112 * 1024 / sizeof(float);
    static constexpr int64_t PRE_TMPBUF_NUM = 2048;
    // arch35 reduced-precision VCONV operates on aligned lane groups
    static constexpr int64_t CAST_ALIGN = 16;

public:
    __aicore__ inline ApplyCamePart4() {}
    __aicore__ inline void Init(GM_ADDR paramIn, GM_ADDR m, GM_ADDR rIn, GM_ADDR cIn, GM_ADDR weightDecay, GM_ADDR lr,
                                GM_ADDR beta3, GM_ADDR sumUR, GM_ADDR sumUC, GM_ADDR sumURC, GM_ADDR sumR,
                                GM_ADDR globalShape, GM_ADDR paramOut, GM_ADDR rOut, GM_ADDR cOut, GM_ADDR workspace,
                                const ApplyCamePart4TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ApplyCamePart4TilingData* tilingData);
    __aicore__ inline void CalcScalarInput();

    // Pre phase: sum_r reduction on core 0 + SyncAll (only when sum_r input is absent)
    __aicore__ inline void ProcessPre();
    __aicore__ inline void ComputeSumR(uint64_t gmOffset, uint64_t num);
    __aicore__ inline void ComputeSumROnce(uint64_t gmOffset, uint64_t num);

    __aicore__ inline void ProcessR();
    __aicore__ inline void CopyInR(int64_t iter, int64_t num);
    __aicore__ inline void CopyInSumur(int64_t iter, int64_t num);
    __aicore__ inline void ComputeR(int64_t num);
    __aicore__ inline void CopyOutR(int64_t iter, int64_t num);

    __aicore__ inline void ProcessC();
    __aicore__ inline void CopyInC(int64_t iter, int64_t num);
    __aicore__ inline void CopyInSumuc(int64_t iter, int64_t num);
    __aicore__ inline void ComputeC(int64_t num);
    __aicore__ inline void CopyOutC(int64_t iter, int64_t num);

    __aicore__ inline void ProcessParam();
    __aicore__ inline void InitForCalcParam();
    __aicore__ inline void InitProcessParam();
    __aicore__ inline void ProcessPerCoreParam();
    __aicore__ inline void ProcessTailCoreParam();
    __aicore__ inline void CalcParam(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum, int64_t curCNum);
    __aicore__ inline void CopyInParamr(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInNotAlignedParamr(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInParamc(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInNotAlignedParamc(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyInParam(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum, int64_t curCNum);
    __aicore__ inline void CopyInNotAlignedParam(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum, int64_t curCNum);
    __aicore__ inline void CopyInm(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum, int64_t curCNum);
    __aicore__ inline void CopyInNotAlignedm(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum, int64_t curCNum);
    __aicore__ inline void ComputeParam(int64_t curRNum, int64_t curCNum);
    __aicore__ inline void CalcRcCycleMode(LocalTensor<float>& dst, LocalTensor<float>& src,
                                           LocalTensor<float>& srcScalar, int64_t curRNum, int64_t curCNum);
    __aicore__ inline void CopyOutParam(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum, int64_t curCNum);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return b == 0 ? 0 : (a + b - 1) / b;
    }

private:
    TPipe pipe_;
    TPipe prePipe_; // separate pipe: Pre-phase buffers overlap main-phase buffers (sequential phases)
    TQue<QuePosition::VECIN, 1> rcInQue_;   // r & c input
    TQue<QuePosition::VECIN, 1> sumurcQue_; // sum_u_r & sum_u_c input (float)
    TQue<QuePosition::VECOUT, 1> rcOutQue_; // r & c output
    TBuf<QuePosition::VECCALC> scalarBuf_;

    TQue<QuePosition::VECIN, 1> inQueR_;
    TQue<QuePosition::VECIN, 1> inQueC_;
    TQue<QuePosition::VECIN, 1> inQuem_;
    TQue<QuePosition::VECIN, 1> inQueParam_;   // tile of param_in, shape (rRcNumPerLoop, cRcNumPerLoop)
    TQue<QuePosition::VECOUT, 1> outQueParam_; // tile of param_out
    TBuf<QuePosition::VECCALC> ub1Buf_;        // rc * coefficient
    TBuf<QuePosition::VECCALC> ub2Buf_;        // sqrt / div tmp
    TBuf<QuePosition::VECCALC> ub3Buf_;        // r * c

    // fp32 compute buffers (reduced precision only)
    TBuf<QuePosition::VECCALC> castInRBuf_;
    TBuf<QuePosition::VECCALC> castOutRBuf_;
    TBuf<QuePosition::VECCALC> castInCBuf_;
    TBuf<QuePosition::VECCALC> castOutCBuf_;
    TBuf<QuePosition::VECCALC> castParamBuf_;
    TBuf<QuePosition::VECCALC> castmBuf_;

    // Pre phase buffers (allocated on prePipe_, only when sum_r input is absent)
    TQue<QuePosition::VECIN, 1> preInQue_;
    TBuf<QuePosition::VECCALC> preTmpBuf_;
    TBuf<QuePosition::VECCALC> preCastBuf_;

    GlobalTensor<T> rInGm_;
    GlobalTensor<T> rOutGm_;
    GlobalTensor<T> cInGm_;
    GlobalTensor<T> cOutGm_;
    GlobalTensor<T> mGm_;
    GlobalTensor<T> paramInGm_;
    GlobalTensor<T> paramOutGm_;
    GlobalTensor<float> sumurGm_;
    GlobalTensor<float> sumucGm_;
    GlobalTensor<float> sumRWorkspaceGm_;

    GlobalTensor<float> weightDecayGm_;
    GlobalTensor<float> lrGm_;
    GlobalTensor<float> beta3Gm_;
    GlobalTensor<float> sumurcGm_;
    GlobalTensor<float> sumRGm_;
    GlobalTensor<int64_t> globalShapeGm_;

    float beta3_ = 0.0f;
    float sumR_ = 0.0f;
    float sumURC_ = 0.0f;
    float N_ = 0.0f;
    float M_ = 0.0f;
    float lr_ = 0.0f;
    float weightDecay_ = 0.0f;
    float rcCoefficient_ = 0.0f;

    int64_t blockIdx_ = 0;

    GM_ADDR rIn_;
    GM_ADDR rOut_;
    GM_ADDR sumUR_;
    GM_ADDR cIn_;
    GM_ADDR cOut_;
    GM_ADDR sumUC_;
    GM_ADDR sumRAddr_;
    GM_ADDR globalShapeAddr_;

    LocalTensor<T> rRcLocalTensor_;

    int64_t nShape_ = 0;
    int64_t mShape_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t handleMax_ = 0;

    int64_t rNumPerCore_ = 0;
    int64_t rCoreNumToUse_ = 0;
    int64_t rNumPerLoop_ = 0;
    int64_t rLoopCount_ = 0;
    int64_t rNumTailPerLoop_ = 0;
    int64_t rLoopCountTailCore_ = 0;
    int64_t rNumTailLoopLast_ = 0;

    int64_t cNumPerCore_ = 0;
    int64_t cCoreNumToUse_ = 0;
    int64_t cNumPerLoop_ = 0;
    int64_t cLoopCount_ = 0;
    int64_t cNumTailPerLoop_ = 0;
    int64_t cLoopCountTailCore_ = 0;
    int64_t cNumTailLoopLast_ = 0;

    int64_t rRcNumPerCore_ = 0;
    int64_t rRcCoreNumToUse_ = 0;
    int64_t rRcNumOnTailCore_ = 0;
    int64_t rRcNumPerLoop_ = 0;
    int64_t rRcLoopCount_ = 0;
    int64_t rRcNumTailLoop_ = 0;
    int64_t rRcLoopCountTailCore_ = 0;
    int64_t rRcNumTailLoopTailCore_ = 0;
    int64_t cRcNumPerLoop_ = 0;
    int64_t cRcLoopCount_ = 0;
    int64_t cRcNumTailLoop_ = 0;

    int64_t rRcBlockOffset_ = 0;

    const int64_t NUM_PER_BLOCK = ONE_BLK / sizeof(T);
};

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ParseTilingData(const ApplyCamePart4TilingData* tilingData)
{
    nShape_ = tilingData->n;
    mShape_ = tilingData->m;
    totalCoreNum_ = tilingData->totalCoreNum;
    handleMax_ = tilingData->handleMax;

    rNumPerCore_ = tilingData->rNumPerCore;
    rCoreNumToUse_ = tilingData->rCoreNumToUse;
    rNumPerLoop_ = tilingData->rNumPerLoop;
    rLoopCount_ = tilingData->rLoopCount;
    rNumTailPerLoop_ = tilingData->rNumTailPerLoop;
    rLoopCountTailCore_ = tilingData->rLoopCountTailCore;
    rNumTailLoopLast_ = tilingData->rNumTailLoopLast;

    cNumPerCore_ = tilingData->cNumPerCore;
    cCoreNumToUse_ = tilingData->cCoreNumToUse;
    cNumPerLoop_ = tilingData->cNumPerLoop;
    cLoopCount_ = tilingData->cLoopCount;
    cNumTailPerLoop_ = tilingData->cNumTailPerLoop;
    cLoopCountTailCore_ = tilingData->cLoopCountTailCore;
    cNumTailLoopLast_ = tilingData->cNumTailLoopLast;

    rRcNumPerCore_ = tilingData->rRcNumPerCore;
    rRcCoreNumToUse_ = tilingData->rRcCoreNumToUse;
    rRcNumOnTailCore_ = tilingData->rRcNumOnTailCore;
    rRcNumPerLoop_ = tilingData->rRcNumPerLoop;
    rRcLoopCount_ = tilingData->rRcLoopCount;
    rRcNumTailLoop_ = tilingData->rRcNumTailLoop;
    rRcLoopCountTailCore_ = tilingData->rRcLoopCountTailCore;
    rRcNumTailLoopTailCore_ = tilingData->rRcNumTailLoopTailCore;
    cRcNumPerLoop_ = tilingData->cRcNumPerLoop;
    cRcLoopCount_ = tilingData->cRcLoopCount;
    cRcNumTailLoop_ = tilingData->cRcNumTailLoop;
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::Init(GM_ADDR paramIn, GM_ADDR m, GM_ADDR rIn, GM_ADDR cIn,
                                               GM_ADDR weightDecay, GM_ADDR lr, GM_ADDR beta3, GM_ADDR sumUR,
                                               GM_ADDR sumUC, GM_ADDR sumURC, GM_ADDR sumR, GM_ADDR globalShape,
                                               GM_ADDR paramOut, GM_ADDR rOut, GM_ADDR cOut, GM_ADDR workspace,
                                               const ApplyCamePart4TilingData* tilingData)
{
    ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();

    rInGm_.SetGlobalBuffer((__gm__ T*)rIn + blockIdx_ * rNumPerCore_);
    cInGm_.SetGlobalBuffer((__gm__ T*)cIn + blockIdx_ * cNumPerCore_);
    sumurGm_.SetGlobalBuffer((__gm__ float*)sumUR + blockIdx_ * rNumPerCore_);
    sumucGm_.SetGlobalBuffer((__gm__ float*)sumUC + blockIdx_ * cNumPerCore_);
    rOutGm_.SetGlobalBuffer((__gm__ T*)rOut + blockIdx_ * rNumPerCore_);
    cOutGm_.SetGlobalBuffer((__gm__ T*)cOut + blockIdx_ * cNumPerCore_);
    mGm_.SetGlobalBuffer((__gm__ T*)m);
    paramInGm_.SetGlobalBuffer((__gm__ T*)paramIn);
    paramOutGm_.SetGlobalBuffer((__gm__ T*)paramOut);

    weightDecayGm_.SetGlobalBuffer((__gm__ float*)weightDecay);
    lrGm_.SetGlobalBuffer((__gm__ float*)lr);
    beta3Gm_.SetGlobalBuffer((__gm__ float*)beta3);
    sumurcGm_.SetGlobalBuffer((__gm__ float*)sumURC);
    globalShapeGm_.SetGlobalBuffer((__gm__ int64_t*)globalShape);
    // workspace layout: [0, totalCoreNum*32) reserved (sync flags), then 32B sum_r slot
    sumRWorkspaceGm_.SetGlobalBuffer((__gm__ float*)((__gm__ uint8_t*)workspace + totalCoreNum_ * 32));

    rIn_ = rIn;
    rOut_ = rOut;
    sumUR_ = sumUR;
    cIn_ = cIn;
    cOut_ = cOut;
    sumUC_ = sumUC;
    sumRAddr_ = sumR;
    globalShapeAddr_ = globalShape;

    if (nShape_ <= 0 || mShape_ <= 0) {
        return;
    }

    // buffers for R / C phase
    pipe_.InitBuffer(rcInQue_, 1, (handleMax_ * sizeof(T) + ONE_BLK - 1) / ONE_BLK * ONE_BLK);
    pipe_.InitBuffer(sumurcQue_, 1, (handleMax_ * sizeof(float) + ONE_BLK - 1) / ONE_BLK * ONE_BLK);
    pipe_.InitBuffer(rcOutQue_, 1, (handleMax_ * sizeof(T) + ONE_BLK - 1) / ONE_BLK * ONE_BLK);
    pipe_.InitBuffer(scalarBuf_, ONE_BLK);

    int64_t rBufferLength = rNumPerLoop_ > rRcNumPerLoop_ ? rNumPerLoop_ : rRcNumPerLoop_;
    int64_t cBufferLength = cNumPerLoop_ > cRcNumPerLoop_ ? cNumPerLoop_ : cRcNumPerLoop_;
    rBufferLength = NUM_PER_BLOCK * CeilDiv(rBufferLength, NUM_PER_BLOCK);
    cBufferLength = NUM_PER_BLOCK * CeilDiv(cBufferLength, NUM_PER_BLOCK);
    pipe_.InitBuffer(inQueR_, 1, (rBufferLength * sizeof(T) + ONE_BLK - 1) / ONE_BLK * ONE_BLK);
    pipe_.InitBuffer(inQueC_, 1, (cBufferLength * sizeof(T) + ONE_BLK - 1) / ONE_BLK * ONE_BLK);

    if constexpr (IS_REDUCED) {
        pipe_.InitBuffer(castInRBuf_, rBufferLength * sizeof(float));
        pipe_.InitBuffer(castInCBuf_, cBufferLength * sizeof(float));
        pipe_.InitBuffer(castOutRBuf_, rBufferLength * sizeof(float));
        pipe_.InitBuffer(castOutCBuf_, cBufferLength * sizeof(float));
    }

    InitForCalcParam();

    if (sumRAddr_ == nullptr) {
        prePipe_.InitBuffer(preInQue_, 1, PRE_MAX_ONCE_NUM * sizeof(T));
        prePipe_.InitBuffer(preTmpBuf_, PRE_TMPBUF_NUM * sizeof(float));
        if constexpr (IS_REDUCED) {
            prePipe_.InitBuffer(preCastBuf_, PRE_CAST_MAX_NUM * sizeof(float));
        }
    }

    CalcScalarInput();
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::InitForCalcParam()
{
    int64_t num = rRcNumPerLoop_ * cRcNumPerLoop_;
    num = CeilDiv(num, NUM_PER_BLOCK) * NUM_PER_BLOCK;
    int64_t perLoopBytes = num * sizeof(float);
    pipe_.InitBuffer(inQuem_, 1, perLoopBytes);
    pipe_.InitBuffer(inQueParam_, 1, perLoopBytes);
    pipe_.InitBuffer(outQueParam_, 1, perLoopBytes);
    pipe_.InitBuffer(ub1Buf_, perLoopBytes);
    pipe_.InitBuffer(ub2Buf_, perLoopBytes);
    pipe_.InitBuffer(ub3Buf_, perLoopBytes);
    if constexpr (IS_REDUCED) {
        pipe_.InitBuffer(castParamBuf_, perLoopBytes);
        pipe_.InitBuffer(castmBuf_, perLoopBytes);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CalcScalarInput()
{
    LocalTensor<float> inputLocal = scalarBuf_.Get<float>();
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    DataCopyExtParams scalarCopyParams{1, sizeof(float), 0, 0, 0};
    DataCopyPadExtParams<float> scalarPadParams{false, 0, 0, 0};

    DataCopyPad(inputLocal, weightDecayGm_, scalarCopyParams, scalarPadParams);
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    weightDecay_ = inputLocal.GetValue(0);

    DataCopyPad(inputLocal, lrGm_, scalarCopyParams, scalarPadParams);
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    lr_ = inputLocal.GetValue(0);

    DataCopyPad(inputLocal, beta3Gm_, scalarCopyParams, scalarPadParams);
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    beta3_ = inputLocal.GetValue(0);

    DataCopyPad(inputLocal, sumurcGm_, scalarCopyParams, scalarPadParams);
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    sumURC_ = inputLocal.GetValue(0);

    if (sumRAddr_ != nullptr) {
        sumRGm_.SetGlobalBuffer((__gm__ float*)sumRAddr_);
        DataCopyPad(inputLocal, sumRGm_, scalarCopyParams, scalarPadParams);
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        sumR_ = inputLocal.GetValue(0);
    }

    constexpr int64_t INT64_PER_BLOCK = ONE_BLK / sizeof(int64_t);
    LocalTensor<int64_t> int64Local = scalarBuf_.Get<int64_t>();
    if (globalShapeAddr_ != nullptr) {
        DataCopy(int64Local, globalShapeGm_, INT64_PER_BLOCK);
    } else {
        int64Local.SetValue(0, nShape_);
        int64Local.SetValue(1, mShape_);
    }
    PipeBarrier<PIPE_ALL>();
    Cast(inputLocal, int64Local, RoundMode::CAST_ROUND, INT64_PER_BLOCK);
    PipeBarrier<PIPE_ALL>();
    N_ = inputLocal.GetValue(0);
    M_ = inputLocal.GetValue(1);
    PipeBarrier<PIPE_ALL>();
}

// ------------------------------- Pre phase (sum_r) -------------------------------

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ProcessPre()
{
    if (sumRAddr_ != nullptr) {
        return;
    }
    if (GetBlockIdx() == 0) {
        InitOutput<float>(sumRWorkspaceGm_, 1, 0.0f);
        uint64_t loopTime = (nShape_ + PRE_MAX_ONCE_NUM - 1) / PRE_MAX_ONCE_NUM;
        uint64_t preEleNum = (nShape_ + loopTime - 1) / loopTime;
        uint64_t lastEleNum = nShape_ - preEleNum * (loopTime - 1);

        if (loopTime == 1) {
            ComputeSumROnce(0, lastEleNum);
        } else {
            for (uint64_t i = 0; i + 1 < loopTime; i++) {
                ComputeSumR(i * preEleNum, preEleNum);
            }
            ComputeSumR((loopTime - 1) * preEleNum, lastEleNum);
        }
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ComputeSumR(uint64_t gmOffset, uint64_t num)
{
    LocalTensor<T> rcInUb = preInQue_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(num * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(rcInUb, rInGm_[gmOffset], copyParams, padParams);
    preInQue_.EnQue(rcInUb);
    preInQue_.DeQue<T>();

    LocalTensor<float> workLocal = preTmpBuf_.Get<float>();
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    if constexpr (IS_REDUCED) {
        LocalTensor<float> inputCast = preCastBuf_.Get<float>();
        Cast(inputCast, rcInUb, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        ReduceSum(inputCast, inputCast, workLocal, num);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        SetAtomicAdd<float>();
        DataCopyPad(sumRWorkspaceGm_, inputCast, {1, (uint16_t)sizeof(float), 0, 0});
        SetAtomicNone();
    } else {
        ReduceSum(rcInUb, rcInUb, workLocal, num);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        SetAtomicAdd<float>();
        DataCopyPad(sumRWorkspaceGm_, rcInUb, {1, (uint16_t)sizeof(float), 0, 0});
        SetAtomicNone();
    }
    preInQue_.FreeTensor(rcInUb);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ComputeSumROnce(uint64_t gmOffset, uint64_t num)
{
    LocalTensor<T> rcInUb = preInQue_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(num * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(rcInUb, rInGm_[gmOffset], copyParams, padParams);
    preInQue_.EnQue(rcInUb);
    preInQue_.DeQue<T>();

    LocalTensor<float> workLocal = preTmpBuf_.Get<float>();
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    if constexpr (IS_REDUCED) {
        LocalTensor<float> inputCast = preCastBuf_.Get<float>();
        Cast(inputCast, rcInUb, RoundMode::CAST_NONE, num);
        PipeBarrier<PIPE_V>();
        ReduceSum(inputCast, inputCast, workLocal, num);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopyPad(sumRWorkspaceGm_, inputCast, {1, (uint16_t)sizeof(float), 0, 0});
    } else {
        ReduceSum(rcInUb, rcInUb, workLocal, num);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopyPad(sumRWorkspaceGm_, rcInUb, {1, (uint16_t)sizeof(float), 0, 0});
    }
    preInQue_.FreeTensor(rcInUb);
}

// ------------------------------- R / C phase -------------------------------

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ProcessR()
{
    if (GetBlockIdx() >= rCoreNumToUse_) {
        return;
    }
    if (GetBlockIdx() != rCoreNumToUse_ - 1) {
        for (int64_t i = 0; i < rLoopCount_; i++) {
            CopyInR(i, rNumPerLoop_);
            CopyInSumur(i, rNumPerLoop_);
            ComputeR(rNumPerLoop_);
            CopyOutR(i, rNumPerLoop_);
        }
    } else {
        for (int64_t i = 0; i < rLoopCountTailCore_; i++) {
            CopyInR(i, rNumTailPerLoop_);
            CopyInSumur(i, rNumTailPerLoop_);
            ComputeR(rNumTailPerLoop_);
            CopyOutR(i, rNumTailPerLoop_);
        }
        // handle the unaligned tail separately
        if (rNumTailLoopLast_ != 0) {
            rInGm_.SetGlobalBuffer((__gm__ T*)rIn_ + rLoopCountTailCore_ * rNumTailPerLoop_);
            rOutGm_.SetGlobalBuffer((__gm__ T*)rOut_ + rLoopCountTailCore_ * rNumTailPerLoop_);
            sumurGm_.SetGlobalBuffer((__gm__ float*)sumUR_ + rLoopCountTailCore_ * rNumTailPerLoop_);
            CopyInR(0, rNumTailLoopLast_);
            CopyInSumur(0, rNumTailLoopLast_);
            ComputeR(rNumTailLoopLast_);
            CopyOutR(0, rNumTailLoopLast_);
        }
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInR(int64_t iter, int64_t num)
{
    LocalTensor<T> rInput = rcInQue_.AllocTensor<T>();
    DataCopy(rInput, rInGm_[iter * num], CeilDiv(num, NUM_PER_BLOCK) * NUM_PER_BLOCK);
    rcInQue_.EnQue(rInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInSumur(int64_t iter, int64_t num)
{
    LocalTensor<float> sumurInput = sumurcQue_.AllocTensor<float>();
    constexpr int64_t FLOAT_PER_BLOCK = ONE_BLK / sizeof(float);
    DataCopy(sumurInput, sumurGm_[iter * num], CeilDiv(num, FLOAT_PER_BLOCK) * FLOAT_PER_BLOCK);
    sumurcQue_.EnQue(sumurInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ComputeR(int64_t num)
{
    LocalTensor<T> rInput = rcInQue_.DeQue<T>();
    LocalTensor<float> sumurInput = sumurcQue_.DeQue<float>();
    LocalTensor<T> output = rcOutQue_.AllocTensor<T>();

    float scalarVal = (1.0f - beta3_) / M_;
    if constexpr (IS_REDUCED) {
        int64_t castNum = CeilDiv(num, CAST_ALIGN) * CAST_ALIGN;
        LocalTensor<float> rLocal = castInRBuf_.Get<float>();
        Cast(rLocal, rInput, RoundMode::CAST_NONE, castNum);
        LocalTensor<float> outCast = castOutRBuf_.Get<float>();
        PipeBarrier<PIPE_V>();
        Muls(outCast, rLocal, beta3_, num);
        PipeBarrier<PIPE_V>();
        Axpy(outCast, sumurInput, scalarVal, num);
        PipeBarrier<PIPE_V>();
        Cast(output, outCast, RoundMode::CAST_RINT, castNum);
        PipeBarrier<PIPE_V>();
    } else {
        Muls(output, rInput, beta3_, num);
        PipeBarrier<PIPE_V>();
        Axpy(output, sumurInput, scalarVal, num);
    }
    rcOutQue_.EnQue<T>(output);
    rcInQue_.FreeTensor(rInput);
    sumurcQue_.FreeTensor(sumurInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyOutR(int64_t iter, int64_t num)
{
    LocalTensor<T> output = rcOutQue_.DeQue<T>();
    DataCopyParams dataCopyParams{1, (uint16_t)(num * sizeof(T)), 0, 0};
    DataCopyPad(rOutGm_[iter * num], output, dataCopyParams);
    rcOutQue_.FreeTensor(output);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ProcessC()
{
    if (GetBlockIdx() >= cCoreNumToUse_) {
        return;
    }
    if (GetBlockIdx() != cCoreNumToUse_ - 1) {
        for (int64_t i = 0; i < cLoopCount_; i++) {
            CopyInC(i, cNumPerLoop_);
            CopyInSumuc(i, cNumPerLoop_);
            ComputeC(cNumPerLoop_);
            CopyOutC(i, cNumPerLoop_);
        }
    } else {
        for (int64_t i = 0; i < cLoopCountTailCore_; i++) {
            CopyInC(i, cNumTailPerLoop_);
            CopyInSumuc(i, cNumTailPerLoop_);
            ComputeC(cNumTailPerLoop_);
            CopyOutC(i, cNumTailPerLoop_);
        }
        if (cNumTailLoopLast_ != 0) {
            cInGm_.SetGlobalBuffer((__gm__ T*)cIn_ + cLoopCountTailCore_ * cNumTailPerLoop_);
            cOutGm_.SetGlobalBuffer((__gm__ T*)cOut_ + cLoopCountTailCore_ * cNumTailPerLoop_);
            sumucGm_.SetGlobalBuffer((__gm__ float*)sumUC_ + cLoopCountTailCore_ * cNumTailPerLoop_);
            CopyInC(0, cNumTailLoopLast_);
            CopyInSumuc(0, cNumTailLoopLast_);
            ComputeC(cNumTailLoopLast_);
            CopyOutC(0, cNumTailLoopLast_);
        }
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInC(int64_t iter, int64_t num)
{
    LocalTensor<T> cInput = rcInQue_.AllocTensor<T>();
    DataCopy(cInput, cInGm_[iter * num], CeilDiv(num, NUM_PER_BLOCK) * NUM_PER_BLOCK);
    rcInQue_.EnQue(cInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInSumuc(int64_t iter, int64_t num)
{
    LocalTensor<float> sumucInput = sumurcQue_.AllocTensor<float>();
    constexpr int64_t FLOAT_PER_BLOCK = ONE_BLK / sizeof(float);
    DataCopy(sumucInput, sumucGm_[iter * num], CeilDiv(num, FLOAT_PER_BLOCK) * FLOAT_PER_BLOCK);
    sumurcQue_.EnQue(sumucInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ComputeC(int64_t num)
{
    LocalTensor<T> cInput = rcInQue_.DeQue<T>();
    LocalTensor<float> sumucInput = sumurcQue_.DeQue<float>();
    LocalTensor<T> output = rcOutQue_.AllocTensor<T>();

    float scalarVal = (1.0f - beta3_) / N_;
    if constexpr (IS_REDUCED) {
        int64_t castNum = CeilDiv(num, CAST_ALIGN) * CAST_ALIGN;
        LocalTensor<float> cLocal = castInCBuf_.Get<float>();
        Cast(cLocal, cInput, RoundMode::CAST_NONE, castNum);
        LocalTensor<float> outCast = castOutCBuf_.Get<float>();
        PipeBarrier<PIPE_V>();
        Muls(outCast, cLocal, beta3_, num);
        PipeBarrier<PIPE_V>();
        Axpy(outCast, sumucInput, scalarVal, num);
        PipeBarrier<PIPE_V>();
        Cast(output, outCast, RoundMode::CAST_RINT, castNum);
        PipeBarrier<PIPE_V>();
    } else {
        Muls(output, cInput, beta3_, num);
        PipeBarrier<PIPE_V>();
        Axpy(output, sumucInput, scalarVal, num);
    }
    rcOutQue_.EnQue<T>(output);
    rcInQue_.FreeTensor(cInput);
    sumurcQue_.FreeTensor(sumucInput);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyOutC(int64_t iter, int64_t num)
{
    LocalTensor<T> output = rcOutQue_.DeQue<T>();
    DataCopyParams dataCopyParams{1, (uint16_t)(num * sizeof(T)), 0, 0};
    DataCopyPad(cOutGm_[iter * num], output, dataCopyParams);
    rcOutQue_.FreeTensor(output);
}

// ------------------------------- Param phase -------------------------------

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ProcessParam()
{
    if (blockIdx_ >= rRcCoreNumToUse_) {
        return;
    }

    InitProcessParam();

    // calc denominator once, not in cycle
    float denominator = beta3_ * sumR_ / N_ + (1.0f - beta3_) * sumURC_ / (M_ * N_);
    rcCoefficient_ = 1.0f / denominator;

    rRcBlockOffset_ = blockIdx_ * rRcNumPerCore_;

    const bool isTailCore = (blockIdx_ == rRcCoreNumToUse_ - 1);
    if (!isTailCore) {
        ProcessPerCoreParam();
    } else {
        ProcessTailCoreParam();
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::InitProcessParam()
{
    rOutGm_.SetGlobalBuffer((__gm__ T*)rOut_);
    cOutGm_.SetGlobalBuffer((__gm__ T*)cOut_);

    if (sumRAddr_ == nullptr) {
        LocalTensor<float> inputLocal = scalarBuf_.Get<float>();
        DataCopyExtParams scalarCopyParams{1, sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> scalarPadParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, sumRWorkspaceGm_, scalarCopyParams, scalarPadParams);
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        sumR_ = inputLocal.GetValue(0);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ProcessPerCoreParam()
{
    const int64_t loopCount = rRcLoopCount_;
    int64_t curRNum = rRcNumPerLoop_;
    int64_t curCNum = 0;
    int64_t rLoopIdx = 0;
    // non-tail r loop
    for (rLoopIdx = 0; rLoopIdx < loopCount - 1; ++rLoopIdx) {
        CopyInParamr(rLoopIdx, curRNum);
        int64_t cLoopIdx = 0;
        // non-tail c loop
        for (cLoopIdx = 0; cLoopIdx < cRcLoopCount_ - 1; ++cLoopIdx) {
            curCNum = cRcNumPerLoop_;
            CalcParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
        }
        // tail c loop
        curCNum = cRcNumTailLoop_;
        CalcParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
        inQueR_.FreeTensor(rRcLocalTensor_);
    }

    // tail r loop
    curRNum = rRcNumTailLoop_;
    if (curRNum % NUM_PER_BLOCK == 0) {
        CopyInParamr(rLoopIdx, curRNum);
    } else {
        CopyInNotAlignedParamr(rLoopIdx, curRNum);
    }
    int64_t cLoopIdx = 0;
    for (cLoopIdx = 0; cLoopIdx < cRcLoopCount_ - 1; ++cLoopIdx) {
        curCNum = cRcNumPerLoop_;
        CalcParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
    }
    // tail c loop
    curCNum = cRcNumTailLoop_;
    CalcParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
    inQueR_.FreeTensor(rRcLocalTensor_);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ProcessTailCoreParam()
{
    const int64_t loopCount = rRcLoopCountTailCore_;
    int64_t curRNum = rRcNumPerLoop_;
    int64_t curCNum = 0;
    int64_t rLoopIdx = 0;
    // non-tail r loop
    for (rLoopIdx = 0; rLoopIdx < loopCount - 1; ++rLoopIdx) {
        CopyInParamr(rLoopIdx, curRNum);
        int64_t cLoopIdx = 0;
        for (cLoopIdx = 0; cLoopIdx < cRcLoopCount_ - 1; ++cLoopIdx) {
            curCNum = cRcNumPerLoop_;
            CalcParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
        }
        curCNum = cRcNumTailLoop_;
        CalcParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
        inQueR_.FreeTensor(rRcLocalTensor_);
    }

    // tail r loop
    curRNum = rRcNumTailLoopTailCore_;
    if (curRNum % NUM_PER_BLOCK == 0) {
        CopyInParamr(rLoopIdx, curRNum);
    } else {
        CopyInNotAlignedParamr(rLoopIdx, curRNum);
    }
    int64_t cLoopIdx = 0;
    for (cLoopIdx = 0; cLoopIdx < cRcLoopCount_ - 1; ++cLoopIdx) {
        curCNum = cRcNumPerLoop_;
        CalcParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
    }
    curCNum = cRcNumTailLoop_;
    CalcParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
    inQueR_.FreeTensor(rRcLocalTensor_);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInParamr(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<T> rLocal = inQueR_.AllocTensor<T>();
    int64_t offset = rRcBlockOffset_ + loopIdx * rRcNumPerLoop_;
    DataCopy(rLocal, rOutGm_[offset], dataCount);
    inQueR_.EnQue(rLocal);
    rRcLocalTensor_ = inQueR_.DeQue<T>();
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInNotAlignedParamr(int64_t loopIdx, int64_t dataCount)
{
    DataCopyParams dataCopyParams{1, (uint16_t)(dataCount * sizeof(T)), 0, 0};
    uint8_t rightPadding = NUM_PER_BLOCK * CeilDiv(dataCount, NUM_PER_BLOCK) - dataCount;
    DataCopyPadParams padParams{true, 0, rightPadding, 0};

    LocalTensor<T> rLocal = inQueR_.AllocTensor<T>();
    int64_t offset = rRcBlockOffset_ + loopIdx * rRcNumPerLoop_;
    DataCopyPad(rLocal, rOutGm_[offset], dataCopyParams, padParams);
    inQueR_.EnQue(rLocal);
    rRcLocalTensor_ = inQueR_.DeQue<T>();
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInParamc(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<T> cLocal = inQueC_.AllocTensor<T>();
    int64_t offset = loopIdx * cRcNumPerLoop_;
    DataCopy(cLocal, cOutGm_[offset], dataCount);
    inQueC_.EnQue(cLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInNotAlignedParamc(int64_t loopIdx, int64_t dataCount)
{
    DataCopyParams dataCopyParams{1, (uint16_t)(dataCount * sizeof(T)), 0, 0};
    uint8_t rightPadding = NUM_PER_BLOCK * CeilDiv(dataCount, NUM_PER_BLOCK) - dataCount;
    DataCopyPadParams padParams{true, 0, rightPadding, 0};

    LocalTensor<T> cLocal = inQueC_.AllocTensor<T>();
    int64_t offset = loopIdx * cRcNumPerLoop_;
    DataCopyPad(cLocal, cOutGm_[offset], dataCopyParams, padParams);
    inQueC_.EnQue(cLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInParam(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum,
                                                      int64_t curCNum)
{
    LocalTensor<T> paramLocal = inQueParam_.AllocTensor<T>();
    int64_t startRow = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startCol = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRow * mShape_ + startCol;
    int64_t alignedCNum = CeilDiv(curCNum, NUM_PER_BLOCK) * NUM_PER_BLOCK;
    for (int64_t i = 0; i < curRNum; ++i) {
        DataCopy(paramLocal[i * alignedCNum], paramInGm_[startOffset + i * mShape_], alignedCNum);
    }
    inQueParam_.EnQue(paramLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInNotAlignedParam(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum,
                                                                int64_t curCNum)
{
    LocalTensor<T> paramLocal = inQueParam_.AllocTensor<T>();
    int64_t startRow = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startCol = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRow * mShape_ + startCol;
    int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNum, NUM_PER_BLOCK);
    DataCopyParams dataCopyParams{1, (uint16_t)(curCNum * sizeof(T)), 0, 0};
    uint8_t rightPadding = alignedCNum - curCNum;
    DataCopyPadParams padParams{true, 0, rightPadding, 0};
    for (int64_t i = 0; i < curRNum; ++i) {
        DataCopyPad(paramLocal[i * alignedCNum], paramInGm_[startOffset + i * mShape_], dataCopyParams, padParams);
    }
    inQueParam_.EnQue(paramLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInm(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum, int64_t curCNum)
{
    LocalTensor<T> mLocal = inQuem_.AllocTensor<T>();
    int64_t startRow = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startCol = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRow * mShape_ + startCol;
    int64_t alignedCNum = CeilDiv(curCNum, NUM_PER_BLOCK) * NUM_PER_BLOCK;
    for (int64_t i = 0; i < curRNum; ++i) {
        DataCopy(mLocal[i * alignedCNum], mGm_[startOffset + i * mShape_], alignedCNum);
    }
    inQuem_.EnQue(mLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyInNotAlignedm(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum,
                                                            int64_t curCNum)
{
    LocalTensor<T> mLocal = inQuem_.AllocTensor<T>();
    int64_t startRow = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startCol = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRow * mShape_ + startCol;
    int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNum, NUM_PER_BLOCK);
    DataCopyParams dataCopyParams{1, (uint16_t)(curCNum * sizeof(T)), 0, 0};
    uint8_t rightPadding = alignedCNum - curCNum;
    DataCopyPadParams padParams{true, 0, rightPadding, 0};
    for (int64_t i = 0; i < curRNum; ++i) {
        DataCopyPad(mLocal[i * alignedCNum], mGm_[startOffset + i * mShape_], dataCopyParams, padParams);
    }
    inQuem_.EnQue(mLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CalcParam(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum,
                                                    int64_t curCNum)
{
    if ((curRNum % NUM_PER_BLOCK == 0) && (curCNum % NUM_PER_BLOCK == 0)) {
        CopyInParamc(cLoopIdx, curCNum);
        CopyInParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
        CopyInm(rLoopIdx, cLoopIdx, curRNum, curCNum);
        ComputeParam(curRNum, curCNum);
        CopyOutParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
        return;
    }

    CopyInNotAlignedParamc(cLoopIdx, curCNum);
    CopyInNotAlignedm(rLoopIdx, cLoopIdx, curRNum, curCNum);
    CopyInNotAlignedParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
    int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNum, NUM_PER_BLOCK);
    ComputeParam(curRNum, alignedCNum);
    CopyOutParam(rLoopIdx, cLoopIdx, curRNum, curCNum);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::ComputeParam(int64_t curRNum, int64_t curCNum)
{
    int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNum, NUM_PER_BLOCK);
    curCNum = alignedCNum;
    LocalTensor<T> cLocal = inQueC_.DeQue<T>();
    LocalTensor<T> mLocal = inQuem_.DeQue<T>();
    LocalTensor<T> paramInLocal = inQueParam_.DeQue<T>();
    LocalTensor<T> paramOutLocal = outQueParam_.AllocTensor<T>();
    int64_t dataCount = curCNum * curRNum;

    LocalTensor<float> ub1 = ub1Buf_.Get<float>();
    LocalTensor<float> ub2 = ub2Buf_.Get<float>();
    LocalTensor<float> ub3 = ub3Buf_.Get<float>();

    // fp32 views of the inputs (cast for reduced precision)
    LocalTensor<float> rF;
    LocalTensor<float> cF;
    LocalTensor<float> mF;
    LocalTensor<float> pF;
    if constexpr (IS_REDUCED) {
        rF = castInRBuf_.Get<float>();
        cF = castInCBuf_.Get<float>();
        mF = castmBuf_.Get<float>();
        pF = castParamBuf_.Get<float>();
        Cast(rF, rRcLocalTensor_, RoundMode::CAST_NONE, CeilDiv(curRNum, CAST_ALIGN) * CAST_ALIGN);
        Cast(cF, cLocal, RoundMode::CAST_NONE, curCNum);
        Cast(mF, mLocal, RoundMode::CAST_NONE, dataCount);
        Cast(pF, paramInLocal, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    } else {
        rF = rRcLocalTensor_;
        cF = cLocal;
        mF = mLocal;
        pF = paramInLocal;
    }

    // ub3: r*c (per-row scalar Muls)
    CalcRcCycleMode(ub3, cF, rF, curRNum, curCNum);
    PipeBarrier<PIPE_V>();
    // ub1: rc*coefficient
    Muls(ub1, ub3, rcCoefficient_, dataCount);
    PipeBarrier<PIPE_V>();
    // ub2 = sqrt(1/S)
    Sqrt(ub2, ub1, dataCount);
    PipeBarrier<PIPE_V>();
    // lr*m
    Muls(mF, mF, lr_, dataCount);
    PipeBarrier<PIPE_V>();
    // (m*lr) / ub2
    Div(ub2, mF, ub2, dataCount);
    PipeBarrier<PIPE_V>();

    // (1 - lr * weight_decay) * param
    float mScalar = 1.0f - lr_ * weightDecay_;
    Muls(pF, pF, mScalar, dataCount);
    PipeBarrier<PIPE_V>();

    if constexpr (IS_REDUCED) {
        Sub(pF, pF, ub2, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(paramOutLocal, pF, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    } else {
        Sub(paramOutLocal, pF, ub2, dataCount);
        PipeBarrier<PIPE_V>();
    }

    outQueParam_.EnQue<T>(paramOutLocal);
    inQueC_.FreeTensor(cLocal);
    inQuem_.FreeTensor(mLocal);
    inQueParam_.FreeTensor(paramInLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CalcRcCycleMode(LocalTensor<float>& dst, LocalTensor<float>& src,
                                                          LocalTensor<float>& srcScalar, int64_t curRNum,
                                                          int64_t curCNum)
{
    // srcScalar lives in UB written by MTE2 (fp32 path) or by Vector cast (reduced path)
    if constexpr (IS_REDUCED) {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    } else {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
    }
    for (int64_t rLoopIdx = 0; rLoopIdx < curRNum; ++rLoopIdx) {
        int64_t dstOffset = rLoopIdx * curCNum;
        Muls(dst[dstOffset], src, srcScalar.GetValue(rLoopIdx), curCNum);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::CopyOutParam(int64_t rLoopIdx, int64_t cLoopIdx, int64_t curRNum,
                                                       int64_t curCNum)
{
    LocalTensor<T> paramLocal = outQueParam_.DeQue<T>();
    int64_t startRow = rRcBlockOffset_ + rLoopIdx * rRcNumPerLoop_;
    int64_t startCol = cLoopIdx * cRcNumPerLoop_;
    int64_t startOffset = startRow * mShape_ + startCol;
    if (curCNum % NUM_PER_BLOCK == 0) {
        for (int64_t i = 0; i < curRNum; ++i) {
            DataCopy(paramOutGm_[startOffset + i * mShape_], paramLocal[i * curCNum], curCNum);
        }
    } else {
        int64_t alignedCNum = NUM_PER_BLOCK * CeilDiv(curCNum, NUM_PER_BLOCK);
        DataCopyParams dataCopyParams{1, (uint16_t)(curCNum * sizeof(T)), 0, 0};
        for (int64_t i = 0; i < curRNum; ++i) {
            DataCopyPad(paramOutGm_[startOffset + i * mShape_], paramLocal[i * alignedCNum], dataCopyParams);
        }
    }
    outQueParam_.FreeTensor(paramLocal);
}

template <typename T>
__aicore__ inline void ApplyCamePart4<T>::Process()
{
    if (nShape_ <= 0 || mShape_ <= 0) {
        return;
    }

    ProcessPre();

    ProcessR();

    ProcessC();

    // multi-core sync: r_out / c_out are consumed by the param phase
    SyncAll();

    ProcessParam();
}

} // namespace NsApplyCamePart4
#endif // APPLY_CAME_PART4_H
