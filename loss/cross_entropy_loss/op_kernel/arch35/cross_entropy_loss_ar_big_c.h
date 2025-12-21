/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file cross_entropy_loss_ar_big_c.h
 * \brief
 */

#ifndef CROSS_ENTROPY_LOSS_BIG_C_H
#define CROSS_ENTROPY_LOSS_BIG_C_H
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"


namespace CrossEntropyLoss {
using namespace AscendC;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::CastTrait;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

static constexpr int32_t VF_LEN_B32 = platform::GetVRegSize() / sizeof(float);
static constexpr int32_t ONE_BLOCK_SIZE = platform::GetUbBlockSize();
static constexpr int32_t ONE_BLOCK_NUM_B32 = ONE_BLOCK_SIZE / sizeof(float);
static constexpr uint32_t REDUCE_NUM = 24; // loss weight smooth都需要reduce成1个数，所以需要预留 3*8=24这么多数
static constexpr int32_t CONST_3 = 3;   // loss weight smooth各1个数，总共3个
static constexpr int32_t CONST_32 = 32; // 避免出发cacheline 128 / 4 = 32
static constexpr int32_t CONST_64 = 64; // N方向固定成64
static constexpr int32_t CONST_96 = 96; // loss weight smooth reduce后的结果ub大小，3*8*4=96
static constexpr float MIN_FP32 = -3.402823466e+38;

constexpr static CastTrait castB32ToB16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr static CastTrait castB16ToB32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
class CrossEntropyLossBigC
{
public:
    __aicore__ inline CrossEntropyLossBigC(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR logProb, GM_ADDR zloss,
        GM_ADDR lseForZloss, GM_ADDR workspace, const CrossEntropyLossRegBaseTilingData* __restrict tilingData,
        TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeSubLast(
        LocalTensor<T1>& inputUB, LocalTensor<float>& tmpUb, LocalTensor<float>& maxUb, LocalTensor<float>& cacheUb,
        int32_t onceC);
    __aicore__ inline void GatherLoss(
        int32_t j, int32_t kk, int32_t onceC, LocalTensor<float>& yLocal, LocalTensor<T2>& targetUb);
    __aicore__ inline void ComputeMul(int32_t nSize, LocalTensor<T2>& targetUb, LocalTensor<float>& lossUb);
    __aicore__ inline void DataCopyTarget(int32_t nOnceCore, int64_t nOffset, LocalTensor<T2>& targetUb);
    __aicore__ inline void DataCopyInputx(int32_t onceC, int64_t offset, LocalTensor<T1>& inputUB);
    __aicore__ inline void DataCopyWeight(int32_t onceC, int64_t offset, LocalTensor<float>& inputUB);
    __aicore__ inline void ComputeReduceTail(
        int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<float>& sumUb);
    __aicore__ inline void ComputeReduceAfter(
        int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<float>& sumUb);
    __aicore__ inline void DataCopyOut1(int32_t onceC, int64_t offset, LocalTensor<T1>& yLocal);
    __aicore__ inline void ComputeSubAndLoss(
        int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<T2>& targetUb);
    __aicore__ inline void DataCopyOut0(int32_t nOnceCore, int64_t nOffset, LocalTensor<T1>& lossUbB16);
    __aicore__ inline void ComputeReduceN(LocalTensor<float>& lossUb, int32_t nOnce);
    __aicore__ inline void DataCopyWorkspace(int32_t offset);
    __aicore__ inline void ReductionSum(LocalTensor<float>& cleanUb, LocalTensor<T1>& cleanUbB16);
    __aicore__ inline void ReductionMean(LocalTensor<float>& cleanUb, LocalTensor<T1>& cleanUbB16);
    __aicore__ inline void ComputeMulAfterSum(LocalTensor<float>& cleanUb, LocalTensor<T1>& cleanUbB16);
    __aicore__ inline void ComputeNLast(LocalTensor<float>& tmpUb);
    __aicore__ inline void ComputeSubSmooth(
        LocalTensor<T1>& inputUB, LocalTensor<float>& tmpUb, LocalTensor<float>& weightUb, LocalTensor<float>& outUb,
        int32_t onceC);
    __aicore__ inline void ComputeEfSoftmaxTail(
        int64_t nOffset, int32_t j, LocalTensor<float>& tmpUb, LocalTensor<float>& tmpUb1, LocalTensor<T2>& targetUb);
    __aicore__ inline void ComputeEfSoftmaxAfter(
        int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<T2>& targetUb);
    __aicore__ inline void ComputeMulAdd(
        LocalTensor<float>& lossUb, LocalTensor<float>& smoothUb, LocalTensor<T1>& lossUbT1, int32_t nOnceCore);
    __aicore__ inline void ParserTilingData();
    __aicore__ inline void ComputeMax(int64_t nOffset, int32_t j, LocalTensor<float>& maxUb);
    __aicore__ inline void ComputeSoftAndSmooth(
        int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<T2>& targetUb);

private:
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T2> targetGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<T1> lossGm_;
    GlobalTensor<T1> logProbGm_;
    GlobalTensor<float> workspaceGm_;
    TPipe* pipe_;
    const CrossEntropyLossRegBaseTilingData* tilingData_;
    constexpr static int32_t bufferNum = 2;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECIN, 1> targetInQueue_;
    TQue<QuePosition::VECIN, 1> weightQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<QuePosition::VECCALC> tmpUb_;  // tmpub 二分累加时需要
    TBuf<QuePosition::VECCALC> tmpUb1_; // smooth场景下需要多一块ub
    TBuf<QuePosition::VECCALC> maxUb_;
    TBuf<QuePosition::VECCALC> sumUb_;
    TBuf<QuePosition::VECCALC> cacheUb_; // 做二分累加缓存更新时需要
    TBuf<QuePosition::VECCALC> lossUb_;
    TBuf<QuePosition::VECCALC> smoothUb_;
    TBuf<QuePosition::VECCALC> clearUb_;
    TBuf<QuePosition::VECCALC> weightUb_; // 此ub存放weight gather后的结果
    int32_t c_ = 0;
    int32_t blockIdx_ = 0;
    int32_t nLoopNum_ = 0;
    int32_t nOnceCore_ = 0;
    int32_t nOnceCoreTail_ = 0;
    int32_t cLoopTimes_ = 0;
    int32_t cTail_ = 0;
    int32_t onceC_ = 0;
    int64_t nOffsetStart_ = 0;
    int32_t kTimesTail_ = 0;
    int32_t kTimes_ = 0;
    int32_t realCoreNum_ = 0;
    T2 ignoreData_ = 0;
    int32_t cacheStart_ = 0;
    float label1_ = 0.0f; // 1-labelsmoothing
    float labelc_ = 0.0f; // labelsmoothing/c

    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParamsF_{false, 0, 0, 0};
    DataCopyExtParams copyParams{1, 32, 0, 0, 0};
};

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ParserTilingData()
{
    c_ = tilingData_->C;
    if (blockIdx_ < tilingData_->frontCoreNum) {
        nLoopNum_ = tilingData_->nLoopTimesB;
        nOnceCoreTail_ = tilingData_->nTailNumB;
        nOnceCore_ = tilingData_->onceNSize;
        nOffsetStart_ = blockIdx_ * (tilingData_->frontCoreNSize + 1);
    } else {
        nLoopNum_ = tilingData_->nLoopTimesT;
        nOnceCoreTail_ = tilingData_->nTailNumT;
        nOnceCore_ = tilingData_->onceNSizeT;
        nOffsetStart_ = blockIdx_ * tilingData_->frontCoreNSize + tilingData_->frontCoreNum;
    }
    cLoopTimes_ = tilingData_->cLoopTimes;
    cTail_ = tilingData_->cOnceNumTail;
    onceC_ = tilingData_->cOnceNum;
    kTimesTail_ = tilingData_->kTimesTail;
    kTimes_ = tilingData_->kTimes;
    realCoreNum_ = tilingData_->realCoreNum;
    ignoreData_ = tilingData_->ignoreIndex;
    cacheStart_ = tilingData_->cacheStart;
    label1_ = tilingData_->labelSmooth1;
    labelc_ = tilingData_->labelSmoothC;
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::Init(
    GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR logProb, GM_ADDR zloss, GM_ADDR lseForZloss,
    GM_ADDR workspace, const CrossEntropyLossRegBaseTilingData* __restrict tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tilingData_ = tilingData;
    inputGm_.SetGlobalBuffer((__gm__ T1*)input);
    targetGm_.SetGlobalBuffer((__gm__ T2*)target);
    if constexpr (isWeight == 1) {
        weightGm_.SetGlobalBuffer((__gm__ float*)weight);
    }
    lossGm_.SetGlobalBuffer((__gm__ T1*)loss);
    logProbGm_.SetGlobalBuffer((__gm__ T1*)logProb);
    ParserTilingData();
    pipe_->InitBuffer(inputQueue_, bufferNum, onceC_ * sizeof(T1));
    pipe_->InitBuffer(outQueue_, 1, onceC_ * sizeof(float));
    pipe_->InitBuffer(targetInQueue_, 1, CONST_64 * sizeof(T2));
    if constexpr (isWeight == 1 && labelS != 0) {
        pipe_->InitBuffer(weightQueue_, 1, onceC_ * sizeof(float));
    }
    if constexpr ((isWeight == 1) || (reduction == 1 && isWeight == 0)) {
        pipe_->InitBuffer(weightUb_, CONST_64 * sizeof(float));
    }
    pipe_->InitBuffer(maxUb_, ONE_BLOCK_SIZE);
    pipe_->InitBuffer(tmpUb_, onceC_ * sizeof(float));
    pipe_->InitBuffer(sumUb_, ONE_BLOCK_SIZE);
    pipe_->InitBuffer(cacheUb_, (cacheStart_ + 1) * ONE_BLOCK_NUM_B32 * sizeof(float));
    pipe_->InitBuffer(lossUb_, CONST_64 * sizeof(float));
    if constexpr (labelS != 0) {
        pipe_->InitBuffer(smoothUb_, CONST_64 * sizeof(float));
        pipe_->InitBuffer(tmpUb1_, onceC_ * sizeof(float));
    }
    if constexpr (reduction != 0) {
        workspaceGm_.SetGlobalBuffer((__gm__ float*)workspace);
        pipe_->InitBuffer(clearUb_, CONST_96);
    }
}

__aicore__ inline void CleanMaxUb(LocalTensor<float>& maxUb)
{
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    float minValue = MIN_FP32;
    __VEC_SCOPE__
    {
        MaskReg preg = CreateMask<float, MaskPattern::VL8>();
        RegTensor<float> srcReg0;
        AscendC::MicroAPI::Duplicate(srcReg0, minValue);
        AscendC::MicroAPI::DataCopy(maxUbAddr, srcReg0, preg);
    }
}

__aicore__ inline void VToMte3()
{
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
}

__aicore__ inline void SToV()
{
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

__aicore__ inline void Mte2ToV()
{
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
}

__aicore__ inline void VToS()
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <typename T1>
__aicore__ inline void GetRowMax(int32_t onceC, LocalTensor<float>& maxUb, LocalTensor<T1>& inputUb)
{
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T1*)inputUb.GetPhyAddr();
    int32_t vfLen = platform::GetVRegSize() / sizeof(T1);
    uint16_t repeatTimes1 = onceC / vfLen;
    uint32_t tailNum = onceC % vfLen;
    T1 minValue = MIN_FP32;
    if constexpr (IsSameType<T1, half>::value) {
        minValue = -65504;
    }
    __VEC_SCOPE__
    {
        RegTensor<T1> maxReg;
        RegTensor<T1> maxRegT;
        RegTensor<T1> srcReg;
        RegTensor<T1> srcRegT;
        RegTensor<T1> maxLowReg;
        RegTensor<float> maxLowRegFp32;
        RegTensor<T1> maxHighReg;
        RegTensor<float> maxHighRegFp32;
        RegTensor<float> maxRegFp32;
        RegTensor<float> maxRegLast;
        RegTensor<float> maxUbReg;
        RegTensor<float> maxUbRegLast;
        AscendC::MicroAPI::Duplicate(maxReg, minValue);
        MaskReg preg = CreateMask<T1, MaskPattern::ALL>();
        MaskReg preg2 = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t jj = 0; jj < repeatTimes1; jj++) {
            AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<T1>(jj, vfLen);
            AscendC::MicroAPI::DataCopy(srcReg, inputUbAddr, srcOffset);
            AscendC::MicroAPI::Max(maxReg, srcReg, maxReg, preg);
        }
        MaskReg preg1 = UpdateMask<T1>(tailNum);
        AscendC::MicroAPI::DataCopy(srcRegT, inputUbAddr + repeatTimes1 * vfLen);
        AscendC::MicroAPI::Max(maxRegT, maxReg, srcRegT, preg1);
        AscendC::MicroAPI::Copy<T1, AscendC::MicroAPI::MaskMergeMode::MERGING>(maxReg, maxRegT, preg1);
        if constexpr (sizeof(T1) == sizeof(half)) {
            // 再做reducemax操作，因为bf16不支持vcmax, FP32不用做这个unpack操作
            AscendC::MicroAPI::UnPack<int32_t, int16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                (RegTensor<int32_t>&)maxLowReg, (RegTensor<int16_t>&)maxReg);
            AscendC::MicroAPI::UnPack<int32_t, int16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(
                (RegTensor<int32_t>&)maxHighReg, (RegTensor<int16_t>&)maxReg);
            AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(maxLowRegFp32, maxLowReg, preg2);
            AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(maxHighRegFp32, maxHighReg, preg2);
            AscendC::MicroAPI::Max(maxRegFp32, maxLowRegFp32, maxHighRegFp32, preg2);
            AscendC::MicroAPI::ReduceMax(maxRegLast, maxRegFp32, preg2);
        } else {
            AscendC::MicroAPI::ReduceMax((RegTensor<T1>&)maxRegLast, maxReg, preg);
        }
        // 再和上一次max做比较
        AscendC::MicroAPI::DataCopy(maxUbReg, maxUbAddr);
        AscendC::MicroAPI::Max(maxUbRegLast, maxRegLast, maxUbReg, preg2);
        MaskReg maskOne = CreateMask<float, MaskPattern::VL1>();
        AscendC::MicroAPI::DataCopy(maxUbAddr, maxUbRegLast, maskOne);
    }
}

template <typename T1>
__aicore__ inline void ComputeSubExp(
    LocalTensor<T1>& x1Local, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, int32_t onceC)
{
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T1*)x1Local.GetPhyAddr();
    auto tmpUbAddr = (__ubuf__ float*)tmpUb.GetPhyAddr();
    int32_t vfLenB32 = VF_LEN_B32;
    uint16_t repeatTimes = CeilDivision(onceC, vfLenB32);
    uint32_t num = onceC;
    __VEC_SCOPE__
    {
        MaskReg preg;
        MaskReg preg1;
        MaskReg preg2;
        RegTensor<T1> vreg0;
        RegTensor<float> srcRegfp32;
        RegTensor<float> maxReg;
        RegTensor<float> subReg;
        RegTensor<float> expReg;
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxReg, maxUbAddr);
        for (uint16_t jj = 0; jj < repeatTimes; jj++) {
            preg = UpdateMask<float>(num);
            AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<T1>(jj, vfLenB32);
            AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<float>(jj, vfLenB32);
            if constexpr (sizeof(T1) == 2) {
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vreg0, inputUbAddr, srcOffset);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, vreg0, preg);
            } else {
                AscendC::MicroAPI::DataCopy((RegTensor<T1>&)srcRegfp32, inputUbAddr, srcOffset);
            }
            AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, preg);
            AscendC::MicroAPI::Exp(expReg, subReg, preg);
            AscendC::MicroAPI::DataCopy(tmpUbAddr, expReg, outOffset, preg);
        }
    }
}

template <typename T1>
__aicore__ inline void ComputeSubExpTail(
    LocalTensor<T1>& x1Local, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, int32_t onceC, int32_t alignC)
{
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T1*)x1Local.GetPhyAddr();
    auto tmpUbAddr = (__ubuf__ float*)tmpUb.GetPhyAddr();
    int32_t vfLenB32 = VF_LEN_B32;
    uint16_t repeatTimes = onceC / vfLenB32;
    uint32_t tailNumAlign = alignC - repeatTimes * vfLenB32;
    uint32_t tailNum = onceC % vfLenB32;
    uint32_t num = onceC;
    __VEC_SCOPE__
    {
        MaskReg preg = UpdateMask<float>(tailNum);
        MaskReg preg1 = UpdateMask<float>(tailNumAlign);
        RegTensor<T1> vreg0;
        RegTensor<float> srcRegfp32;
        RegTensor<float> maxReg;
        RegTensor<float> subReg;
        RegTensor<float> expReg;
        MaskReg copyOutReg = CreateMask<float, MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxReg, maxUbAddr);
        for (uint16_t jj = 0; jj < repeatTimes; jj++) {
            AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<T1>(jj, vfLenB32);
            AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<float>(jj, vfLenB32);
            if constexpr (sizeof(T1) == 2) {
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vreg0, inputUbAddr, srcOffset);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, vreg0, copyOutReg);
            } else {
                AscendC::MicroAPI::DataCopy((RegTensor<T1>&)srcRegfp32, inputUbAddr, srcOffset);
            }
            AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, copyOutReg);
            AscendC::MicroAPI::Exp(expReg, subReg, copyOutReg);
            AscendC::MicroAPI::DataCopy(tmpUbAddr, expReg, outOffset, copyOutReg);
        }
        if constexpr (sizeof(T1) == 2) {
            AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vreg0, inputUbAddr + repeatTimes * vfLenB32);
            AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, vreg0, preg);
        } else {
            AscendC::MicroAPI::DataCopy((RegTensor<T1>&)srcRegfp32, inputUbAddr + repeatTimes * vfLenB32);
        }

        AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, preg);
        AscendC::MicroAPI::Exp(expReg, subReg, preg);
        AscendC::MicroAPI::DataCopy(tmpUbAddr + repeatTimes * vfLenB32, expReg, preg1);
    }
}

__aicore__ inline void FoldAddVF(LocalTensor<float>& tmpUb, LocalTensor<float>& tmpUb1, int32_t tailC)
{
    auto tmpUbAddr = (__ubuf__ float*)tmpUb.GetPhyAddr();
    auto tmpUb1Addr = (__ubuf__ float*)tmpUb1.GetPhyAddr();
    int32_t vfLenB32 = VF_LEN_B32;
    uint16_t repeatTimes = CeilDivision(tailC, vfLenB32);
    uint32_t num = tailC;
    __VEC_SCOPE__
    {
        MaskReg preg;
        MaskReg preg1;
        MaskReg preg2;
        RegTensor<float> vreg0;
        RegTensor<float> vreg1;
        RegTensor<float> addReg;
        RegTensor<float> subReg;
        RegTensor<float> expReg;
        for (uint16_t jj = 0; jj < repeatTimes; jj++) {
            preg = UpdateMask<float>(num);
            AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(jj, vfLenB32);
            AscendC::MicroAPI::DataCopy(vreg0, tmpUbAddr, srcOffset);
            AscendC::MicroAPI::DataCopy(vreg1, tmpUb1Addr, srcOffset);
            AscendC::MicroAPI::Add(addReg, vreg0, vreg1, preg);
            AscendC::MicroAPI::DataCopy(tmpUbAddr, addReg, srcOffset, preg);
        }
    }
}

__aicore__ inline void UpdateCache(LocalTensor<float>& srcUb, LocalTensor<float>& dstUb, int32_t index)
{
    const int32_t cacheID = bcnt1(index ^ (index + 1)) - 1;
    int32_t stride = ONE_BLOCK_NUM_B32;
    auto dstUbAddr = (__ubuf__ float*)dstUb.GetPhyAddr();
    auto cahUbAddr = (__ubuf__ float*)dstUb.GetPhyAddr() + cacheID * stride;
    auto srcUbAddr = (__ubuf__ float*)srcUb.GetPhyAddr();

    const uint16_t innerLoopTimes = static_cast<uint16_t>(cacheID);

    __VEC_SCOPE__
    {
        RegTensor<float> aReg, bReg;
        MaskReg pMask = CreateMask<float, MaskPattern::VL1>();
        AscendC::MicroAPI::DataCopy(aReg, srcUbAddr);
        for (uint16_t j = 0; j < innerLoopTimes; ++j) {
            AddrReg srcOffsetJ = AscendC::MicroAPI::CreateAddrReg<float>(j, stride);
            AscendC::MicroAPI::DataCopy(bReg, dstUbAddr, srcOffsetJ);
            AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
        }
        AscendC::MicroAPI::DataCopy(cahUbAddr, aReg, pMask);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeSubLast(
    LocalTensor<T1>& inputUB, LocalTensor<float>& tmpUb, LocalTensor<float>& maxUb, LocalTensor<float>& cacheUb,
    int32_t onceC)
{
    int32_t vfLenB32 = VF_LEN_B32;
    uint16_t repeatTimes = CeilDivision(onceC, vfLenB32);
    uint32_t sreg = onceC;
    auto xMaxPtr = (__ubuf__ float*)maxUb.GetPhyAddr();
    auto xSumPtr = (__ubuf__ float*)cacheUb.GetPhyAddr() + cacheStart_ * ONE_BLOCK_NUM_B32;
    auto srcUbAddr = (__ubuf__ T1*)inputUB.GetPhyAddr();
    auto yAddr = (__ubuf__ float*)tmpUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> maxReg, sumReg, logReg, srcRegFp32, subMaxReg, dstReg;
        RegTensor<T1> srcReg, dstRegB16;
        MaskReg maskAll = MicroAPI::CreateMask<float, MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxReg, xMaxPtr);
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
        AscendC::MicroAPI::Log(logReg, sumReg, maskAll);
        MaskReg pMask;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            pMask = UpdateMask<float>(sreg);
            AddrReg srcOffsetJ = AscendC::MicroAPI::CreateAddrReg<T1>(j, vfLenB32);
            AddrReg srcOffsety = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLenB32);
            if constexpr (sizeof(T1) == 2) {
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    srcReg, srcUbAddr, srcOffsetJ);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegFp32, srcReg, pMask);
            } else {
                AscendC::MicroAPI::DataCopy((RegTensor<T1>&)srcRegFp32, srcUbAddr, srcOffsetJ);
            }
            AscendC::MicroAPI::Sub(subMaxReg, srcRegFp32, maxReg, pMask);
            AscendC::MicroAPI::Sub(dstReg, subMaxReg, logReg, pMask);
            AscendC::MicroAPI::DataCopy(yAddr, dstReg, srcOffsety, pMask);
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::GatherLoss(
    int32_t j, int32_t kk, int32_t onceC, LocalTensor<float>& yLocal, LocalTensor<T2>& targetUb)
{
    LocalTensor<float> lossUb = lossUb_.Get<float>();
    VToS();
    T2 index = targetUb.GetValue(j);
    ASSERT((0 <= index && index < c_) && "Target is not in [0, C)");
    if constexpr (isWeight == 1) {
        LocalTensor<float> weightUb = weightUb_.Get<float>();
        float weightValue = weightGm_.GetValue(index);
        if (index == ignoreData_) {
            weightValue = 0.0f;
        }
        weightUb.SetValue(j, weightValue); // weight Gather的结果
    }
    if constexpr (reduction == 1 && isWeight == 0) {
        float weightValue = 1.0f;
        LocalTensor<float> weightUb = weightUb_.Get<float>();
        if (index == ignoreData_) {
            weightValue = 0.0f;
        }
        weightUb.SetValue(j, weightValue);
    }
    if (index >= onceC_ * kk && index < onceC_ * kk + onceC) {
        float cValue = yLocal.GetValue(index - onceC_ * kk);
        lossUb.SetValue(j, cValue);
    }
    SToV();
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeMul(
    int32_t nSize, LocalTensor<T2>& targetUb, LocalTensor<float>& lossUb)
{
    auto targetUbAddr = (__ubuf__ T2*)targetUb.GetPhyAddr();
    auto lossOutUbAddr = (__ubuf__ float*)lossUb.GetPhyAddr();
    auto smoothUbAddr = lossOutUbAddr;
    if constexpr (labelS != 0) {
        LocalTensor<float> smoothUb = smoothUb_.Get<float>();
        smoothUbAddr = (__ubuf__ float*)smoothUb.GetPhyAddr();
    }
    auto weightInUbAddr = lossOutUbAddr;
    if constexpr (isWeight == 1) {
        LocalTensor<float> weightUb = weightUb_.Get<float>();
        weightInUbAddr = (__ubuf__ float*)weightUb.GetPhyAddr();
    }
    auto lossOutUbAddrB16 = (__ubuf__ T1*)lossUb.GetPhyAddr();
    uint32_t nSize2 = nSize;
    int32_t vfLen = platform::GetVRegSize() / sizeof(int64_t);
    if constexpr (sizeof(T2) == sizeof(float) || ignorex == 0) {
        vfLen = VF_LEN_B32;
    }
    T2 ignoreData = ignoreData_;
    uint16_t repeatTimes1 = CeilDivision(nSize, vfLen);
    __VEC_SCOPE__
    {
        MaskReg dstMask, dstMask1, preg1;
        RegTensor<T2> targetReg;
        RegTensor<float> srcReg0;
        MaskReg ignoreMaskReg;
        RegTensor<float> lossMulReg;
        RegTensor<float> smoothMulReg;
        RegTensor<float> weightReg0;
        RegTensor<float> lossMulReg1;
        RegTensor<float> lossMulRegS1;
        RegTensor<T1> lossMulRegB16;
        RegTensor<float> lossReg0;
        RegTensor<float> smoothReg0;
        RegTensor<float> weightReg;
        MaskReg preg = CreateMask<float, MaskPattern::ALL>();
        if constexpr (sizeof(T2) == sizeof(int64_t) && ignorex == 1) {
            preg = CreateMask<float, MaskPattern::VL32>();
        }
        if constexpr (ignorex == 1) {
            AscendC::MicroAPI::Duplicate(srcReg0, 0.0f);
        }
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
            AscendC::MicroAPI::DataCopy(lossReg0, lossOutUbAddr, srcOffset);
            AscendC::MicroAPI::Muls(lossMulReg, lossReg0, -1.0f, preg);
            if constexpr (isWeight == 1) {
                AscendC::MicroAPI::DataCopy(weightReg0, weightInUbAddr, srcOffset);
                AscendC::MicroAPI::Mul(lossMulReg, lossMulReg, weightReg0, preg);
            }
            if constexpr (labelS != 0) {
                AscendC::MicroAPI::DataCopy(smoothReg0, smoothUbAddr, srcOffset);
                AscendC::MicroAPI::Muls(smoothMulReg, smoothReg0, -1.0f, preg);
            }
            if constexpr (ignorex == 1) {
                preg1 = UpdateMask<T2>(nSize2);
                AddrReg maskOffset = AscendC::MicroAPI::CreateAddrReg<T2>(j, vfLen);
                AscendC::MicroAPI::DataCopy(targetReg, targetUbAddr, maskOffset);
                AscendC::MicroAPI::CompareScalar<T2, CMPMODE::NE>(dstMask, targetReg, ignoreData, preg1);
                if constexpr (sizeof(T2) == sizeof(int64_t)) {
                    AscendC::MicroAPI::MaskPack(dstMask1, dstMask);
                    AscendC::MicroAPI::Select(lossMulReg1, lossMulReg, srcReg0, dstMask1);
                    if constexpr (labelS != 0) {
                        AscendC::MicroAPI::Select(lossMulRegS1, smoothMulReg, srcReg0, dstMask1);
                    }
                } else {
                    AscendC::MicroAPI::Select(lossMulReg1, lossMulReg, srcReg0, dstMask);
                    if constexpr (labelS != 0) {
                        AscendC::MicroAPI::Select(lossMulRegS1, smoothMulReg, srcReg0, dstMask);
                    }
                }
                if constexpr (sizeof(T1) == sizeof(half) && reduction == 0 && labelS == 0) {
                    AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<T1>(j, vfLen);
                    AscendC::MicroAPI::Cast<T1, float, castB32ToB16>(lossMulRegB16, lossMulReg1, preg);
                    AscendC::MicroAPI::DataCopy<T1, StoreDist::DIST_PACK_B32>(
                        lossOutUbAddrB16, lossMulRegB16, outOffset, preg);
                } else {
                    AscendC::MicroAPI::DataCopy(lossOutUbAddr, lossMulReg1, srcOffset, preg);
                    if constexpr (labelS != 0) {
                        AscendC::MicroAPI::DataCopy(smoothUbAddr, lossMulRegS1, srcOffset, preg);
                    }
                }
            } else {
                if constexpr (sizeof(T1) == sizeof(half) && reduction == 0 && labelS == 0) {
                    AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<T1>(j, vfLen);
                    AscendC::MicroAPI::Cast<T1, float, castB32ToB16>(lossMulRegB16, lossMulReg, preg);
                    AscendC::MicroAPI::DataCopy<T1, StoreDist::DIST_PACK_B32>(
                        lossOutUbAddrB16, lossMulRegB16, outOffset, preg);
                } else {
                    AscendC::MicroAPI::DataCopy(lossOutUbAddr, lossMulReg, srcOffset, preg);
                    if constexpr (labelS != 0) {
                        AscendC::MicroAPI::DataCopy(smoothUbAddr, smoothMulReg, srcOffset, preg);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyTarget(
    int32_t nOnceCore, int64_t nOffset, LocalTensor<T2>& targetUb)
{
    copyParams.blockLen = nOnceCore * sizeof(T2);
    DataCopyPadExtParams<T2> padParams1{false, 0, 0, 0};
    AscendC::DataCopyPad(targetUb, targetGm_[nOffset], copyParams, padParams1);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyInputx(
    int32_t onceC, int64_t offset, LocalTensor<T1>& inputUB)
{
    copyParams.blockLen = onceC * sizeof(T1);
    AscendC::DataCopyPad(inputUB, inputGm_[offset], copyParams, padParams_);
}
template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyWeight(
    int32_t onceC, int64_t offset, LocalTensor<float>& inputUB)
{
    copyParams.blockLen = onceC * sizeof(float);
    AscendC::DataCopyPad(inputUB, weightGm_[offset], copyParams, padParamsF_);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeReduceTail(
    int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<float>& sumUb)
{
    LocalTensor<float> cacheUb = cacheUb_.Get<float>();
    for (int32_t ll = 0; ll < kTimesTail_; ++ll) { // 需要首尾块相加的地方
        int64_t offset = (nOffset + j) * c_ + ll * onceC_;
        int64_t offset1 = offset + kTimes_ * onceC_;
        int32_t tailC = (ll == (kTimesTail_ - 1) && cTail_ > 0) ? cTail_ : onceC_;
        LocalTensor<T1> x1Local = inputQueue_.AllocTensor<T1>();
        DataCopyInputx(onceC_, offset, x1Local);
        inputQueue_.EnQue<T1>(x1Local);
        x1Local = inputQueue_.DeQue<T1>();
        ComputeSubExp<T1>(x1Local, maxUb, tmpUb, onceC_);
        inputQueue_.FreeTensor(x1Local);

        // 加尾块部分
        LocalTensor<T1> x2Local = inputQueue_.AllocTensor<T1>();
        LocalTensor<float> tmpUb1 = outQueue_.AllocTensor<float>();
        DataCopyInputx(tailC, offset1, x2Local);

        inputQueue_.EnQue<T1>(x2Local);
        x2Local = inputQueue_.DeQue<T1>();
        ComputeSubExp<T1>(x2Local, maxUb, tmpUb1, tailC);
        FoldAddVF(tmpUb, tmpUb1, tailC); // ub做累加 结果放在tmpUb
        inputQueue_.FreeTensor(x2Local);
        outQueue_.FreeTensor(tmpUb1);
        uint32_t srcShape[2] = {1, static_cast<uint32_t>(onceC_)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, tmpUb, srcShape, false);
        UpdateCache(sumUb, cacheUb, ll);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeReduceAfter(
    int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<float>& sumUb)
{
    LocalTensor<float> cacheUb = cacheUb_.Get<float>();
    for (int32_t ci = kTimesTail_; ci < kTimes_; ++ci) {
        int64_t offset = (nOffset + j) * c_ + ci * onceC_;
        int32_t tailC = (ci == (kTimes_ - 1) && kTimesTail_ == 0 && cTail_ != 0) ? cTail_ : onceC_;
        int32_t alignC = ((tailC + ONE_BLOCK_NUM_B32 - 1) / ONE_BLOCK_NUM_B32) * ONE_BLOCK_NUM_B32;
        LocalTensor<T1> x3Local = inputQueue_.AllocTensor<T1>();
        DataCopyInputx(tailC, offset, x3Local);

        inputQueue_.EnQue<T1>(x3Local);
        x3Local = inputQueue_.DeQue<T1>();
        ComputeSubExpTail<T1>(x3Local, maxUb, tmpUb, tailC, alignC);
        inputQueue_.FreeTensor(x3Local);
        uint32_t srcShape[2] = {1, static_cast<uint32_t>(alignC)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, tmpUb, srcShape, false);
        UpdateCache(sumUb, cacheUb, ci);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyOut1(
    int32_t onceC, int64_t offset, LocalTensor<T1>& yLocal)
{
    copyParams.blockLen = onceC * sizeof(T1);
    DataCopyPad(logProbGm_[offset], yLocal, copyParams);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeSubAndLoss(
    int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<T2>& targetUb)
{
    LocalTensor<float> cacheUb = cacheUb_.Get<float>();
    for (int32_t kk = 0; kk < cLoopTimes_; kk++) {
        int32_t onceC = (kk == (cLoopTimes_ - 1) && cTail_ != 0) ? cTail_ : onceC_;
        int64_t offset = (nOffset + j) * c_ + kk * onceC_;
        LocalTensor<T1> inputUB = inputQueue_.AllocTensor<T1>();
        DataCopyInputx(onceC, offset, inputUB);
        inputQueue_.EnQue(inputUB);
        inputUB = inputQueue_.DeQue<T1>();
        if constexpr (sizeof(T1) == sizeof(float)) {
            LocalTensor<T1> yLocal = outQueue_.AllocTensor<T1>();
            LocalTensor<float> yLocalB32 = yLocal.template ReinterpretCast<float>();
            ComputeSubLast(inputUB, yLocalB32, maxUb, cacheUb, onceC);
            inputQueue_.FreeTensor(inputUB);
            GatherLoss(j, kk, onceC, yLocal, targetUb);
            outQueue_.EnQue<T1>(yLocal);
            yLocal = outQueue_.DeQue<T1>();
            DataCopyOut1(onceC, offset, yLocal);
            outQueue_.FreeTensor(yLocal);
        } else {
            ComputeSubLast(inputUB, tmpUb, maxUb, cacheUb, onceC);
            inputQueue_.FreeTensor(inputUB);
            GatherLoss(j, kk, onceC, tmpUb, targetUb);
            LocalTensor<T1> yLocal = outQueue_.AllocTensor<T1>();
            AscendC::Cast(yLocal, tmpUb, AscendC::RoundMode::CAST_RINT, onceC);
            outQueue_.EnQue<T1>(yLocal);
            yLocal = outQueue_.DeQue<T1>();
            DataCopyOut1(onceC, offset, yLocal);
            outQueue_.FreeTensor(yLocal);
        }
    }
}

__aicore__ inline void CleanRUb(LocalTensor<float>& clearUb)
{
    uint32_t aa = REDUCE_NUM;
    auto clearUbAddr = (__ubuf__ float*)clearUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg0;
        MaskReg preg = UpdateMask<float>(aa);
        AscendC::MicroAPI::Duplicate(srcReg0, 0.0f);
        AscendC::MicroAPI::DataCopy(clearUbAddr, srcReg0, preg);
    }
}
template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyOut0(
    int32_t nOnceCore, int64_t nOffset, LocalTensor<T1>& lossUbB16)
{
    VToMte3();
    copyParams.blockLen = nOnceCore * sizeof(T1);
    AscendC::DataCopyPad(lossGm_[nOffset], lossUbB16, copyParams);
}

__aicore__ inline void ReduceSumOne(
    MaskReg& preg, MaskReg& preg2, __local_mem__ float* clearAddr, __local_mem__ float* lossUbAddr)
{
    RegTensor<float> regClean;
    RegTensor<float> srcRegLoss;
    RegTensor<float> srcRegLossSum;
    AscendC::MicroAPI::DataCopy(regClean, clearAddr);
    AscendC::MicroAPI::DataCopy(srcRegLoss, lossUbAddr);
    AscendC::MicroAPI::ReduceSum(srcRegLossSum, srcRegLoss, preg);
    AscendC::MicroAPI::Add(srcRegLoss, srcRegLossSum, regClean, preg2);
    AscendC::MicroAPI::DataCopy(clearAddr, srcRegLoss, preg2);
}
template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeReduceN(
    LocalTensor<float>& lossUb, int32_t nOnce)
{
    LocalTensor<float> clearUb = clearUb_.Get<float>();
    auto clearAddr = (__ubuf__ float*)clearUb.GetPhyAddr();
    auto lossUbAddr = (__ubuf__ float*)lossUb.GetPhyAddr();
    auto weightAddr = lossUbAddr;
    auto smoothAddr = lossUbAddr;
    if constexpr (reduction == 1) {
        LocalTensor<float> weightUb = weightUb_.Get<float>();
        weightAddr = (__ubuf__ float*)weightUb.GetPhyAddr();
    }
    if constexpr (labelS != 0) {
        LocalTensor<float> smoothUb = smoothUb_.Get<float>();
        smoothAddr = (__ubuf__ float*)smoothUb.GetPhyAddr();
    }
    uint32_t nSize = nOnce;

    __VEC_SCOPE__
    {
        MaskReg preg = UpdateMask<float>(nSize);
        MaskReg preg2 = CreateMask<float, MaskPattern::VL1>();

        ReduceSumOne(preg, preg2, clearAddr, lossUbAddr);
        if constexpr (reduction == 1) {
            ReduceSumOne(preg, preg2, clearAddr + 8, weightAddr);
        }
        if constexpr (labelS != 0) {
            ReduceSumOne(preg, preg2, clearAddr + 16, smoothAddr);
        }
    }
}
template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyWorkspace(
    int32_t offset)
{
    LocalTensor<float> clearUb = clearUb_.Get<float>();
    DataCopyExtParams copyParams3;
    copyParams3.blockCount = CONST_3;
    copyParams3.blockLen = sizeof(float);
    copyParams3.srcStride = 0;
    copyParams3.dstStride = (realCoreNum_ * CONST_32 - 1) * sizeof(float);
    DataCopyPad(workspaceGm_[offset], clearUb, copyParams3);
}
__aicore__ inline void DataCopyAdd(
    RegTensor<float>& srcReg0, RegTensor<float>& srcReg1, MaskReg& regAllFp32, __local_mem__ float* srcAddr,
    AddrReg& srcOffset)
{
    AscendC::MicroAPI::DataCopy(srcReg1, srcAddr, srcOffset);
    AscendC::MicroAPI::Add(srcReg0, srcReg1, srcReg0, regAllFp32);
}

__aicore__ inline void DataCopyAddSumT(
    RegTensor<float>& srcReg0, RegTensor<float>& srcReg1, RegTensor<float>& srcRegT, MaskReg& preg, MaskReg& regAllFp32,
    MaskReg& preg2, uint16_t repeatTimes1, uint32_t vfLen, __local_mem__ float* srcAddr, __local_mem__ float* dstAddr)
{
    AscendC::MicroAPI::DataCopy(srcReg1, srcAddr + repeatTimes1 * vfLen);
    AscendC::MicroAPI::Add(srcRegT, srcReg1, srcReg0, preg);
    AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(srcReg0, srcRegT, preg);
    AscendC::MicroAPI::ReduceSum(srcRegT, srcReg0, regAllFp32);
    AscendC::MicroAPI::DataCopy(dstAddr, srcRegT, preg2);
}

template <uint64_t reduction, uint64_t labelS>
__aicore__ inline void VfAllSum(int32_t size, LocalTensor<float>& allReduceUb, LocalTensor<float>& cleanUb)
{
    int32_t vfLen = VF_LEN_B32;
    auto allReduceUbLddr = (__ubuf__ float*)allReduceUb.GetPhyAddr();
    auto allReduceUbWddr = (__ubuf__ float*)allReduceUb.GetPhyAddr() + size;
    auto allReduceUbSddr = (__ubuf__ float*)allReduceUb.GetPhyAddr() + size * 2;
    auto cleanAddr = (__ubuf__ float*)cleanUb.GetPhyAddr();
    uint16_t repeatTimes1 = (size) / vfLen;
    uint32_t tail = size % vfLen;
    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<float> srcReg0;
        RegTensor<float> srcReg1;
        RegTensor<float> srcRegT;
        RegTensor<float> sumReg;

        RegTensor<float> srcRegW0;
        RegTensor<float> srcRegW1;
        RegTensor<float> srcRegTW;
        RegTensor<float> sumRegW;

        RegTensor<float> srcRegS0;
        RegTensor<float> srcRegS1;
        RegTensor<float> srcRegTS;
        RegTensor<float> sumRegS;

        MaskReg regAllFp32 = CreateMask<float, MaskPattern::ALL>();
        preg = UpdateMask<float>(tail);
        MaskReg preg2 = CreateMask<float, MaskPattern::VL1>();

        AscendC::MicroAPI::Duplicate(srcReg0, 0.0f);
        if constexpr (reduction == 1) {
            AscendC::MicroAPI::Duplicate(srcRegW0, 0.0f);
        }
        if constexpr (labelS != 0) {
            AscendC::MicroAPI::Duplicate(srcRegS0, 0.0f);
        }
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
            DataCopyAdd(srcReg0, srcReg1, regAllFp32, allReduceUbLddr, srcOffset);
            if constexpr (reduction == 1) {
                DataCopyAdd(srcRegW0, srcRegW1, regAllFp32, allReduceUbWddr, srcOffset);
            }
            if constexpr (labelS != 0) {
                DataCopyAdd(srcRegS0, srcRegS1, regAllFp32, allReduceUbSddr, srcOffset);
            }
        }

        DataCopyAddSumT(
            srcReg0, srcReg1, srcRegT, preg, regAllFp32, preg2, repeatTimes1, vfLen, allReduceUbLddr, cleanAddr);
        if constexpr (reduction == 1) {
            DataCopyAddSumT(
                srcRegW0, srcRegW1, srcRegTW, preg, regAllFp32, preg2, repeatTimes1, vfLen, allReduceUbWddr,
                cleanAddr + 8);
        }
        if constexpr (labelS != 0) {
            DataCopyAddSumT(
                srcRegS0, srcRegS1, srcRegTS, preg, regAllFp32, preg2, repeatTimes1, vfLen, allReduceUbSddr,
                cleanAddr + 16);
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ReductionSum(
    LocalTensor<float>& clearUb, LocalTensor<T1>& cleanUbT1)
{
    auto srcAddr = (__ubuf__ float*)clearUb.GetPhyAddr();
    auto dstAddr = (__ubuf__ T1*)cleanUbT1.GetPhyAddr();
    float labelS1 = label1_;
    float labelSc = labelc_;

    __VEC_SCOPE__
    {
        RegTensor<float> srcReg0;
        RegTensor<float> srcReg1;
        RegTensor<float> srcRegMul;
        RegTensor<float> srcRegMulS;
        RegTensor<float> srcRegAdd;
        RegTensor<T1> srcRegAddT1;
        MaskReg preg = CreateMask<float, MaskPattern::VL1>();
        AscendC::MicroAPI::DataCopy(srcReg0, srcAddr);
        AscendC::MicroAPI::DataCopy(srcReg1, srcAddr + 16);
        AscendC::MicroAPI::Muls(srcRegMul, srcReg0, labelS1, preg);
        AscendC::MicroAPI::Muls(srcRegMulS, srcReg1, labelSc, preg);
        AscendC::MicroAPI::Add(srcRegAdd, srcRegMul, srcRegMulS, preg);
        if constexpr (sizeof(T1) == sizeof(half)) {
            AscendC::MicroAPI::Cast<T1, float, castB32ToB16>(srcRegAddT1, srcRegAdd, preg);
            AscendC::MicroAPI::DataCopy<T1, StoreDist::DIST_PACK_B32>(dstAddr, srcRegAddT1, preg);
        } else {
            AscendC::MicroAPI::DataCopy(dstAddr, (RegTensor<T1>&)srcRegAdd, preg);
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ReductionMean(
    LocalTensor<float>& clearUb, LocalTensor<T1>& cleanUbB16)
{
    auto srcAddr = (__ubuf__ float*)clearUb.GetPhyAddr();
    auto dstAddr = (__ubuf__ T1*)cleanUbB16.GetPhyAddr();
    float label1 = label1_;
    float labelc = labelc_;

    __VEC_SCOPE__
    {
        RegTensor<float> srcReg0;
        RegTensor<float> srcReg1;
        RegTensor<float> srcReg2;
        RegTensor<float> srcRegDiv;
        RegTensor<float> srcRegDiv1;
        RegTensor<float> srcRegMuls1;
        RegTensor<float> srcRegMulsc;
        RegTensor<T1> srcRegAddT1;
        MaskReg preg = CreateMask<float, MaskPattern::VL1>();
        AscendC::MicroAPI::DataCopy(srcReg0, srcAddr);
        AscendC::MicroAPI::DataCopy(srcReg1, srcAddr + 8);
        if constexpr (labelS != 0) {
            AscendC::MicroAPI::DataCopy(srcReg2, srcAddr + 16);
            AscendC::MicroAPI::Div(srcRegDiv1, srcReg2, srcReg1, preg);
        }
        AscendC::MicroAPI::Div(srcRegDiv, srcReg0, srcReg1, preg);
        if constexpr (labelS != 0) {
            AscendC::MicroAPI::Muls(srcRegMuls1, srcRegDiv, label1, preg);
            AscendC::MicroAPI::Muls(srcRegMulsc, srcRegDiv1, labelc, preg);
            AscendC::MicroAPI::Add(srcRegDiv, srcRegMuls1, srcRegMulsc, preg);
        }
        if constexpr (sizeof(T1) == sizeof(half)) {
            AscendC::MicroAPI::Cast<T1, float, castB32ToB16>(srcRegAddT1, srcRegDiv, preg);
            AscendC::MicroAPI::DataCopy<T1, StoreDist::DIST_PACK_B32>(dstAddr, srcRegAddT1, preg);
        } else {
            AscendC::MicroAPI::DataCopy(dstAddr, (RegTensor<T1>&)srcRegDiv, preg);
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeMulAfterSum(
    LocalTensor<float>& cleanUb, LocalTensor<T1>& cleanUbB16)
{
    if constexpr (reduction == 2 && labelS != 0) {
        ReductionSum(cleanUb, cleanUbB16);
        VToMte3();
        DataCopyOut0(1, 0, cleanUbB16);
    }
    if constexpr (reduction == 2 && labelS == 0) {
        if constexpr (sizeof(T1) == sizeof(half)) {
            AscendC::Cast(cleanUbB16, cleanUb, AscendC::RoundMode::CAST_RINT, 1);
        }
        VToMte3();
        DataCopyOut0(1, 0, cleanUbB16);
    }
    if constexpr (reduction == 1) {
        ReductionMean(cleanUb, cleanUbB16);
        VToMte3();
        DataCopyOut0(1, 0, cleanUbB16);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeNLast(
    LocalTensor<float>& tmpUb)
{
    if (blockIdx_ == 0) {
        DataCopyExtParams copyParams4;
        copyParams4.blockCount = realCoreNum_ * CONST_3;
        copyParams4.blockLen = sizeof(float);
        copyParams4.srcStride = (CONST_32 - 1) * sizeof(float);
        copyParams4.dstStride = 0;
        DataCopyPadExtParams<float> padParams2;
        padParams2.isPad = true;
        padParams2.leftPadding = 0;
        padParams2.rightPadding = (ONE_BLOCK_NUM_B32 - 1);
        padParams2.paddingValue = 0.0f;
        AscendC::DataCopyPad(tmpUb, workspaceGm_[0], copyParams4, padParams2);
        Mte2ToV();
        int32_t num = realCoreNum_ * ONE_BLOCK_NUM_B32;
        LocalTensor<float> cleanUb = clearUb_.Get<float>();
        LocalTensor<T1> cleanUbT1 = clearUb_.Get<T1>();
        // clearUb的0位置放loss，8位置放weight, 16位置放smooth
        VfAllSum<reduction, labelS>(num, tmpUb, cleanUb);
        ComputeMulAfterSum(cleanUb, cleanUbT1);
    }
}
template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeSubSmooth(
    LocalTensor<T1>& inputUB, LocalTensor<float>& tmpUb, LocalTensor<float>& weightUb, LocalTensor<float>& outUb,
    int32_t onceC)
{
    int32_t vfLenB32 = VF_LEN_B32;
    uint16_t repeatTimes = onceC / vfLenB32;
    uint32_t alignC = (onceC + ONE_BLOCK_NUM_B32 - 1) / ONE_BLOCK_NUM_B32 * ONE_BLOCK_NUM_B32;
    uint32_t sreg = onceC % vfLenB32;
    uint32_t sreg1 = alignC - repeatTimes * vfLenB32;
    LocalTensor<float> maxUb = maxUb_.Get<float>();
    auto xMaxPtr = (__ubuf__ float*)maxUb.GetPhyAddr();
    LocalTensor<float> cacheUb = cacheUb_.Get<float>();
    auto xSumPtr = (__ubuf__ float*)cacheUb.GetPhyAddr() + cacheStart_ * ONE_BLOCK_NUM_B32;
    auto srcUbAddr = (__ubuf__ T1*)inputUB.GetPhyAddr();
    auto srcUbAddrB32 = (__ubuf__ float*)inputUB.GetPhyAddr();
    auto tmpUbAddr = (__ubuf__ float*)tmpUb.GetPhyAddr();
    auto yAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    auto weightUbAddr = (__ubuf__ float*)weightUb.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> maxReg, sumReg, logReg, srcRegFp32, subMaxReg, dstReg, weightReg;
        RegTensor<T1> srcReg, dstRegB16;
        MaskReg maskAll = MicroAPI::CreateMask<float, MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(maxReg, xMaxPtr);
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumPtr);
        AscendC::MicroAPI::Log(logReg, sumReg, maskAll);
        MaskReg pMask = UpdateMask<float>(sreg);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            AddrReg srcOffsetJ = AscendC::MicroAPI::CreateAddrReg<T1>(j, vfLenB32);
            AddrReg srcOffsety = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLenB32);
            if constexpr (sizeof(T1) == sizeof(half)) {
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    srcReg, srcUbAddr, srcOffsetJ);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegFp32, srcReg, maskAll);
            } else {
                AscendC::MicroAPI::DataCopy(srcRegFp32, srcUbAddrB32, srcOffsety);
            }
            AscendC::MicroAPI::Sub(subMaxReg, srcRegFp32, maxReg, maskAll);
            AscendC::MicroAPI::Sub(dstReg, subMaxReg, logReg, maskAll);
            if constexpr (isWeight == 1) {
                AscendC::MicroAPI::DataCopy(weightReg, weightUbAddr, srcOffsety);
                AscendC::MicroAPI::Mul(weightReg, dstReg, weightReg, maskAll);
                AscendC::MicroAPI::DataCopy(tmpUbAddr, weightReg, srcOffsety, maskAll);
            } else {
                AscendC::MicroAPI::DataCopy(tmpUbAddr, dstReg, srcOffsety, maskAll);
            }

            AscendC::MicroAPI::DataCopy(yAddr, dstReg, srcOffsety, maskAll);
        }
        MaskReg pMask1 = UpdateMask<float>(sreg1);
        if constexpr (sizeof(T1) == sizeof(half)) {
            AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                srcReg, srcUbAddr + repeatTimes * vfLenB32);
            AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegFp32, srcReg, pMask);
        } else {
            AscendC::MicroAPI::DataCopy(srcRegFp32, srcUbAddrB32 + repeatTimes * vfLenB32);
        }
        AscendC::MicroAPI::Sub(subMaxReg, srcRegFp32, maxReg, pMask);
        AscendC::MicroAPI::Sub(dstReg, subMaxReg, logReg, pMask);
        if constexpr (isWeight == 1) {
            AscendC::MicroAPI::DataCopy(weightReg, weightUbAddr + repeatTimes * vfLenB32);
            AscendC::MicroAPI::Mul(weightReg, dstReg, weightReg, pMask);
            AscendC::MicroAPI::DataCopy(tmpUbAddr + repeatTimes * vfLenB32, weightReg, pMask1);
        } else {
            AscendC::MicroAPI::DataCopy(tmpUbAddr + repeatTimes * vfLenB32, dstReg, pMask1);
        }
        AscendC::MicroAPI::DataCopy(yAddr + repeatTimes * vfLenB32, dstReg, pMask);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeEfSoftmaxTail(
    int64_t nOffset, int32_t j, LocalTensor<float>& tmpUb, LocalTensor<float>& tmpUb1, LocalTensor<T2>& targetUb)
{
    LocalTensor<float> cacheUb = cacheUb_.Get<float>();
    LocalTensor<float> sumUb = sumUb_.Get<float>();
    for (int32_t ll = 0; ll < kTimesTail_; ++ll) { // 需要首尾块相加的地方
        int64_t offsetW = ll * onceC_;
        int64_t offset = (nOffset + j) * c_ + offsetW;
        int64_t offset1 = offset + kTimes_ * onceC_;
        int32_t tailC = (ll == (kTimesTail_ - 1) && cTail_ > 0) ? cTail_ : onceC_;
        LocalTensor<T1> x1Local = inputQueue_.AllocTensor<T1>();
        DataCopyInputx(onceC_, offset, x1Local);
        inputQueue_.EnQue<T1>(x1Local);
        x1Local = inputQueue_.DeQue<T1>();

        LocalTensor<T1> outUb = outQueue_.AllocTensor<T1>();
        LocalTensor<float> outUbB32 = outUb.template ReinterpretCast<float>();
        if constexpr (isWeight == 1) {
            LocalTensor<float> weightLocal = weightQueue_.AllocTensor<float>();
            DataCopyWeight(onceC_, offsetW, weightLocal);
            weightQueue_.EnQue<float>(weightLocal);
            weightLocal = weightQueue_.DeQue<float>();
            ComputeSubSmooth(x1Local, tmpUb, weightLocal, outUbB32, onceC_);
            weightQueue_.FreeTensor(weightLocal);
        } else {
            ComputeSubSmooth(x1Local, tmpUb, tmpUb, outUbB32, onceC_);
        }

        inputQueue_.FreeTensor(x1Local);
        GatherLoss(j, ll, onceC_, outUbB32, targetUb);
        if constexpr (sizeof(T1) == sizeof(half)) {
            AscendC::Cast(outUb, outUbB32, AscendC::RoundMode::CAST_RINT, onceC_);
        }
        outQueue_.EnQue<T1>(outUb);
        outUb = outQueue_.DeQue<T1>();
        DataCopyOut1(onceC_, offset, outUb);
        outQueue_.FreeTensor(outUb);

        // 加尾块部分
        LocalTensor<T1> x2Local = inputQueue_.AllocTensor<T1>();
        DataCopyInputx(tailC, offset1, x2Local);
        inputQueue_.EnQue<T1>(x2Local);
        x2Local = inputQueue_.DeQue<T1>();
        LocalTensor<T1> outUb2 = outQueue_.AllocTensor<T1>();
        LocalTensor<float> outUb2B32 = outUb2.template ReinterpretCast<float>();
        if constexpr (isWeight == 1) {
            LocalTensor<float> weightLocal2 = weightQueue_.AllocTensor<float>();
            int64_t offsetW1 = offsetW + kTimes_ * onceC_;
            DataCopyWeight(tailC, offsetW1, weightLocal2);
            weightQueue_.EnQue<float>(weightLocal2);
            weightLocal2 = weightQueue_.DeQue<float>();
            ComputeSubSmooth(x2Local, tmpUb1, weightLocal2, outUb2B32, tailC);
            weightQueue_.FreeTensor(weightLocal2);
        } else {
            ComputeSubSmooth(x2Local, tmpUb1, tmpUb1, outUb2B32, tailC);
        }
        inputQueue_.FreeTensor(x2Local);
        GatherLoss(j, ll + kTimes_, tailC, outUb2B32, targetUb);
        if constexpr (sizeof(T1) == sizeof(half)) {
            AscendC::Cast(outUb2, outUb2B32, AscendC::RoundMode::CAST_RINT, tailC);
        }
        outQueue_.EnQue<T1>(outUb2);
        outUb2 = outQueue_.DeQue<T1>();
        DataCopyOut1(tailC, offset1, outUb2);
        outQueue_.FreeTensor(outUb2);
        FoldAddVF(tmpUb, tmpUb1, tailC);
        uint32_t srcShape[2] = {1, static_cast<uint32_t>(onceC_)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, tmpUb, srcShape, false);
        UpdateCache(sumUb, cacheUb, ll);
    }
}
template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeEfSoftmaxAfter(
    int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<T2>& targetUb)
{
    LocalTensor<float> cacheUb = cacheUb_.Get<float>();
    LocalTensor<float> sumUb = sumUb_.Get<float>();
    for (int32_t ci = kTimesTail_; ci < kTimes_; ++ci) {
        int64_t offsetW1 = ci * onceC_;
        int64_t offset = (nOffset + j) * c_ + ci * onceC_;
        int32_t tailC = (ci == (kTimes_ - 1) && kTimesTail_ == 0 && cTail_ != 0) ? cTail_ : onceC_;
        int32_t alignC = ((tailC + ONE_BLOCK_NUM_B32 - 1) / ONE_BLOCK_NUM_B32) * ONE_BLOCK_NUM_B32;
        LocalTensor<T1> x3Local = inputQueue_.AllocTensor<T1>();
        DataCopyInputx(tailC, offset, x3Local);
        inputQueue_.EnQue<T1>(x3Local);
        x3Local = inputQueue_.DeQue<T1>();

        LocalTensor<T1> outUb = outQueue_.AllocTensor<T1>();
        LocalTensor<float> outUbB32 = outUb.template ReinterpretCast<float>();
        if constexpr (isWeight == 1) {
            LocalTensor<float> weightLocal2 = weightQueue_.AllocTensor<float>();
            DataCopyWeight(tailC, offsetW1, weightLocal2);
            weightQueue_.EnQue<float>(weightLocal2);
            weightLocal2 = weightQueue_.DeQue<float>();
            ComputeSubSmooth(x3Local, tmpUb, weightLocal2, outUbB32, tailC);
            weightQueue_.FreeTensor(weightLocal2);
        } else {
            ComputeSubSmooth(x3Local, tmpUb, tmpUb, outUbB32, tailC);
        }
        inputQueue_.FreeTensor(x3Local);
        GatherLoss(j, ci, tailC, outUbB32, targetUb);
        if constexpr (sizeof(T1) == sizeof(half)) {
            AscendC::Cast(outUb, outUbB32, AscendC::RoundMode::CAST_RINT, tailC);
        }
        outQueue_.EnQue<T1>(outUb);
        outUb = outQueue_.DeQue<T1>();
        DataCopyOut1(tailC, offset, outUb);
        outQueue_.FreeTensor(outUb);

        uint32_t srcShape[2] = {1, static_cast<uint32_t>(alignC)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, tmpUb, srcShape, false);
        UpdateCache(sumUb, cacheUb, ci);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeMulAdd(
    LocalTensor<float>& lossUb, LocalTensor<float>& smoothUb, LocalTensor<T1>& lossUbT1, int32_t nOnceCore)
{
    auto lossAddr = (__ubuf__ float*)lossUb.GetPhyAddr();
    auto lossAddrT1 = (__ubuf__ T1*)lossUbT1.GetPhyAddr();
    auto smoothAddr = (__ubuf__ float*)smoothUb.GetPhyAddr();
    float label1 = label1_;
    float labelc = labelc_;
    int32_t vfLenB32 = VF_LEN_B32;
    uint16_t times = CeilDivision(nOnceCore, vfLenB32);
    uint32_t size = nOnceCore;

    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<float> srcReg0;
        RegTensor<float> srcReg1;
        RegTensor<float> srcReg2;
        RegTensor<float> srcReg3;
        RegTensor<float> srcRegAdd;
        RegTensor<T1> srcRegAddT1;
        for (uint16_t t = 0; t < times; t++) {
            preg = UpdateMask<float>(size);
            AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(t, vfLenB32);
            AscendC::MicroAPI::DataCopy(srcReg0, lossAddr, srcOffset);
            AscendC::MicroAPI::DataCopy(srcReg1, smoothAddr, srcOffset);
            AscendC::MicroAPI::Muls(srcReg2, srcReg0, label1, preg);
            AscendC::MicroAPI::Muls(srcReg3, srcReg1, labelc, preg);
            AscendC::MicroAPI::Add(srcRegAdd, srcReg2, srcReg3, preg);
            if constexpr (sizeof(T1) == sizeof(half)) {
                AscendC::MicroAPI::Cast<T1, float, castB32ToB16>(srcRegAddT1, srcRegAdd, preg);
                AddrReg dstOffset = AscendC::MicroAPI::CreateAddrReg<T1>(t, vfLenB32);
                AscendC::MicroAPI::DataCopy<T1, StoreDist::DIST_PACK_B32>(lossAddrT1, srcRegAddT1, dstOffset, preg);
            } else {
                AscendC::MicroAPI::DataCopy(lossAddr, srcRegAdd, srcOffset, preg);
            }
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeMax(
    int64_t nOffset, int32_t j, LocalTensor<float>& maxUb)
{
    for (int32_t k = 0; k < cLoopTimes_; k++) {
        int32_t onceC = (k == (cLoopTimes_ - 1) && (cTail_ != 0)) ? cTail_ : onceC_;
        int64_t offset = (nOffset + j) * c_ + k * onceC_;
        LocalTensor<T1> inputUB = inputQueue_.AllocTensor<T1>();
        DataCopyInputx(onceC, offset, inputUB);
        inputQueue_.EnQue(inputUB);
        inputUB = inputQueue_.DeQue<T1>();
        GetRowMax<T1>(onceC, maxUb, inputUB);
        inputQueue_.FreeTensor(inputUB);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeSoftAndSmooth(
    int64_t nOffset, int32_t j, LocalTensor<float>& maxUb, LocalTensor<float>& tmpUb, LocalTensor<T2>& targetUb)
{
    if constexpr (labelS == 0) {
        ComputeSubAndLoss(nOffset, j, maxUb, tmpUb, targetUb);
    } else {
        LocalTensor<float> tmpUb1 = tmpUb1_.Get<float>();
        ComputeEfSoftmaxTail(nOffset, j, tmpUb, tmpUb1, targetUb);
        ComputeEfSoftmaxAfter(nOffset, j, maxUb, tmpUb, targetUb);
    }
    if constexpr (labelS != 0) {
        VToS();
        LocalTensor<float> cacheUb = cacheUb_.Get<float>();
        float sumSmooth = cacheUb.GetValue(cacheStart_ * ONE_BLOCK_NUM_B32);
        LocalTensor<float> smoothUb = smoothUb_.Get<float>();
        smoothUb.SetValue(j, sumSmooth);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossBigC<T1, T2, reduction, isWeight, labelS, ignorex>::Process()
{
    if (blockIdx_ > realCoreNum_) {
        return;
    }
    if constexpr (reduction != 0) {
        LocalTensor<float> clearUb = clearUb_.Get<float>();
        CleanRUb(clearUb);
    }
    LocalTensor<float> tmpUb = tmpUb_.Get<float>();
    LocalTensor<float> maxUb = maxUb_.Get<float>();
    LocalTensor<float> sumUb = sumUb_.Get<float>();
    LocalTensor<float> lossUb = lossUb_.Get<float>();
    LocalTensor<T1> lossUbT1 = lossUb_.Get<T1>();
    for (int32_t i = 0; i < nLoopNum_; i++) {
        int32_t nOnceCore = (i == (nLoopNum_ - 1) && (nOnceCoreTail_ != 0)) ? nOnceCoreTail_ : nOnceCore_;
        LocalTensor<T2> targetUb = targetInQueue_.AllocTensor<T2>();
        int64_t nOffset = nOffsetStart_ + i * nOnceCore_;
        DataCopyTarget(nOnceCore, nOffset, targetUb);
        targetInQueue_.EnQue(targetUb);
        targetUb = targetInQueue_.DeQue<T2>();
        for (int32_t j = 0; j < nOnceCore; j++) {
            CleanMaxUb(maxUb);
            ComputeMax(nOffset, j, maxUb);
            ComputeReduceTail(nOffset, j, maxUb, tmpUb, sumUb);
            ComputeReduceAfter(nOffset, j, maxUb, tmpUb, sumUb);
            ComputeSoftAndSmooth(nOffset, j, maxUb, tmpUb, targetUb);
        }
        if constexpr (labelS != 0) {
            SToV();
        }
        ComputeMul(nOnceCore, targetUb, lossUb);
        targetInQueue_.FreeTensor(targetUb);
        if constexpr (reduction != 0) {
            ComputeReduceN(lossUb, nOnceCore);
        } else {
            if constexpr (labelS != 0) {
                LocalTensor<float> smoothUb = smoothUb_.Get<float>();
                ComputeMulAdd(lossUb, smoothUb, lossUbT1, nOnceCore);
            }
            VToMte3();
            DataCopyOut0(nOnceCore, nOffset, lossUbT1);
        }
    }
    if constexpr (reduction != 0) {
        int32_t offsetWorkSpace = blockIdx_ * CONST_32;
        VToMte3();
        DataCopyWorkspace(offsetWorkSpace);
        SyncAll();
        ComputeNLast(tmpUb);
    }
}
} // namespace CrossEntropyLoss
#endif // namespace CrossEntropyLossBigC