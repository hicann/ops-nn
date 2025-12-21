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
 * \file cross_entropy_loss_full_load.h
 * \brief
 */

#ifndef CROSS_ENTROPY_LOSS_FULL_LOAD_H
#define CROSS_ENTROPY_LOSS_FULL_LOAD_H
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace CrossEntropyLoss {
using namespace AscendC;

const int32_t FP32_DTYPE = 4;
const int32_t BLOCK_UB = platform::GetUbBlockSize(); // 一个block的ub大小
const int32_t ONE_FP32 = BLOCK_UB / FP32_DTYPE;

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
class CrossEntropyLossFullLoad
{
public:
    __aicore__ inline CrossEntropyLossFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR logProb, GM_ADDR zloss,
        GM_ADDR lseForZloss, GM_ADDR workspace, const CrossEntropyLossRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ProcessEachCore(
        int64_t nLoopTimes, int64_t onceNSize, int64_t nTailNum, int64_t nCoreOffset);
    __aicore__ inline void ComputeEachCoreOnce(
        int32_t nLoopTimes, int32_t onceNSize, int32_t nTailNum, int64_t nUbOffset);
    __aicore__ inline void VfComputeNone(
        int32_t nTailNum, LocalTensor<float>& lossOutLocal, LocalTensor<float>& smoothLossLocal);
    __aicore__ inline void VfComputeSmoothLoss(
        int32_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float>& tempNCLocal,
        LocalTensor<float>& weightLocal, LocalTensor<float>& subUbLocal, LocalTensor<float>& sumUbLocal);
    __aicore__ inline void VfComputeLossOut(
        int32_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float>& lossOutLocal,
        LocalTensor<float>& weightYnLocal);
    __aicore__ inline void VfComputeLogSub(
        int64_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float> sumUbLocal, LocalTensor<float> subUbLocal,
        LocalTensor<float> tempLocal);
    __aicore__ inline void VfSubExp(
        int64_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float> maxUbLocal, LocalTensor<T1> inputUbLocal,
        LocalTensor<float> subUbLocal, LocalTensor<float> tempLocal);
    __aicore__ inline void VfReduceMax(
        int64_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float> maxUbLocal, LocalTensor<T1> inputUbLocal);
    __aicore__ inline void CopyInInput(int64_t nTailNum, int64_t offset);
    __aicore__ inline void CopyInTarget(int64_t nTailNum, int64_t offset);
    __aicore__ inline void CopyOutPadLoss(int64_t nTailNum, int64_t offset);
    __aicore__ inline void CopyOutPadLogProp(int64_t nTailNum, int64_t offset);
    __aicore__ inline void CopyInWeight(int64_t cTailNum, int64_t cAlgin);
    __aicore__ inline void CleanRUb(LocalTensor<float>& clearUb);
    __aicore__ inline void ComputePartSum(
        int32_t nTailNum, LocalTensor<float>& lossOutLocal, LocalTensor<float>& smoothLossLocal,
        LocalTensor<float>& weightYnLocal);
    __aicore__ inline void VfComputeIgnore(
        int32_t nSizeNum, int64_t nOffset, LocalTensor<T2>& targetUb, LocalTensor<float>& weightLocal,
        LocalTensor<float>& weightYnLocal, LocalTensor<float>& tempNCLocal, LocalTensor<float>& lossOutLocal,
        LocalTensor<float>& smoothLossLocal, LocalTensor<float>& sumUbLocal);
    __aicore__ inline void VToS();
    __aicore__ inline void MTE2ToS();
    __aicore__ inline void SToV();
    __aicore__ inline void MTE2ToV();
    __aicore__ inline void VToMTE3();
    __aicore__ inline void DataCopyWorkspace(int32_t offset);
    __aicore__ inline void ComputeNLast();
    __aicore__ inline void VfAllSum(int32_t size, LocalTensor<float>& allReduceUb, LocalTensor<float>& clearUb);
    __aicore__ inline void DataCopyOutLoss(LocalTensor<float>& clearUb, LocalTensor<T1>& clearUbT1);
    __aicore__ inline void DataCopyOutS0(LocalTensor<float>& clearUb, LocalTensor<T1>& clearUbT1);
    __aicore__ inline void DataCopyOutS1(LocalTensor<float>& clearUb, LocalTensor<T1>& clearUbT1);
    __aicore__ inline void VfAddAll(
        RegTensor<float>& srcReg0, RegTensor<float>& srcReg1, __local_mem__ float* srcAddr, MaskReg& regAllFp32,
        AddrReg& srcOffset);
    __aicore__ inline void VfAddTail(
        RegTensor<float>& srcReg0, RegTensor<float>& srcReg1, RegTensor<float>& srcRegT, MaskReg& preg,
        MaskReg& regAllFp32, MaskReg& preg2, uint16_t repeatTimes1, uint32_t vfLen, __local_mem__ float* srcAddr,
        __local_mem__ float* dstAddr);
    __aicore__ inline void DataCopyCommon(LocalTensor<float>& clearUb, LocalTensor<T1>& clearUbT1);

    constexpr static uint32_t DOUBLE_BUFFER_ONE = 1;
    constexpr static uint32_t WORKSPACE_SUM_NUM = 3;

private:
    TPipe* pipe_;
    const CrossEntropyLossRegBaseTilingData* tilingData_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECIN, 1> targetInQueue_;
    TQue<QuePosition::VECIN, 1> weightQueue_;
    TQue<QuePosition::VECOUT, 1> lossQueue_;
    TQue<QuePosition::VECOUT, 1> logPropQueue_;
    TBuf<QuePosition::VECCALC> subUb_;
    TBuf<QuePosition::VECCALC> tempUb_;
    TBuf<QuePosition::VECCALC> weightYnUb_;
    TBuf<QuePosition::VECCALC> lossOutUb_;
    TBuf<QuePosition::VECCALC> smoothLossUb_;
    TBuf<QuePosition::VECCALC> partSumUb_;
    TBuf<QuePosition::VECCALC> maxUb_;
    TBuf<QuePosition::VECCALC> sumUb_;
    TBuf<QuePosition::VECCALC> clearUb_;
    GlobalTensor<T1> inputGm_;
    GlobalTensor<T2> targetGm_;
    GlobalTensor<float> weightGm_;
    GlobalTensor<float> workspaceGm_;
    GlobalTensor<T1> lossGm_;
    GlobalTensor<T1> logProbGm_;

    int32_t blockIdx_ = 0;
    int32_t vfLenFp32_ = platform::GetVRegSize() / FP32_DTYPE;
    int32_t sumNum = 3; // loss + weight + smooth共3个需要n轴reduce
    int32_t clearUbNum = sumNum * ONE_FP32 * FP32_DTYPE;
    int32_t updateStart_ = 0;
    int32_t SIZE_32 = 32;
    float MIN_VALUE = -3.402823466e+38;

    constexpr static MicroAPI::CastTrait castTrait1 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait castTrait2 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
};

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::Init(
    GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR logProb, GM_ADDR zloss, GM_ADDR lseForZloss,
    GM_ADDR workspace, const CrossEntropyLossRegBaseTilingData* tilingData, TPipe* pipe)
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
    if constexpr (reduction != 0) {
        workspaceGm_.SetGlobalBuffer((__gm__ float*)workspace);
        pipe_->InitBuffer(partSumUb_, clearUbNum);
        pipe_->InitBuffer(clearUb_, clearUbNum);
    }
    int32_t ncOnce = static_cast<int32_t>(tilingData_->onceNSize * tilingData_->cOnceNum);
    int32_t perBlockF32 = BLOCK_UB / FP32_DTYPE;
    int32_t perBlockT1 = static_cast<int32_t>(BLOCK_UB / sizeof(T1));
    int32_t perBlockT2 = static_cast<int32_t>(BLOCK_UB / sizeof(T2));
    int32_t nAlignF32 = (tilingData_->onceNSize + perBlockF32 - 1) / perBlockF32 * perBlockF32;
    int32_t nAlignT1 = (tilingData_->onceNSize + perBlockT1 - 1) / perBlockT1 * perBlockT1;
    int32_t nAlignT2 = (tilingData_->onceNSize + perBlockT2 - 1) / perBlockT2 * perBlockT2;
    uint32_t db = tilingData_->db;
    pipe_->InitBuffer(inputQueue_, db, ncOnce * sizeof(T1));
    pipe_->InitBuffer(targetInQueue_, DOUBLE_BUFFER_ONE, nAlignT2 * sizeof(T2));
    pipe_->InitBuffer(logPropQueue_, DOUBLE_BUFFER_ONE, ncOnce * sizeof(T1));
    if constexpr (reduction == 0) {
        pipe_->InitBuffer(lossQueue_, DOUBLE_BUFFER_ONE, nAlignT1 * sizeof(T1));
    }
    if constexpr (isWeight == 1) {
        pipe_->InitBuffer(weightQueue_, DOUBLE_BUFFER_ONE, tilingData_->cOnceNum * FP32_DTYPE);
    }
    pipe_->InitBuffer(weightYnUb_, nAlignF32 * FP32_DTYPE);
    pipe_->InitBuffer(maxUb_, nAlignF32 * FP32_DTYPE);
    pipe_->InitBuffer(sumUb_, nAlignF32 * FP32_DTYPE);
    pipe_->InitBuffer(subUb_, ncOnce * FP32_DTYPE);
    pipe_->InitBuffer(tempUb_, ncOnce * FP32_DTYPE);
    pipe_->InitBuffer(smoothLossUb_, nAlignF32 * FP32_DTYPE);
    pipe_->InitBuffer(lossOutUb_, nAlignF32 * FP32_DTYPE);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::ProcessEachCore(
    int64_t nLoopTimes, int64_t onceNSize, int64_t nTailNum, int64_t nCoreOffset)
{
    if constexpr (reduction != 0) {
        LocalTensor<float> clearUb = clearUb_.Get<float>();
        CleanRUb(clearUb);
    }
    int64_t nUbOffset = 0;
    for (int32_t nLoopIndex = 0; nLoopIndex < nLoopTimes; nLoopIndex++) {
        nUbOffset = nCoreOffset + nLoopIndex * tilingData_->onceNSize;
        ComputeEachCoreOnce(nLoopIndex, onceNSize, onceNSize, nUbOffset);
    }
    if (nTailNum > 0) {
        nUbOffset = nCoreOffset + nLoopTimes * tilingData_->onceNSize;
        ComputeEachCoreOnce(nLoopTimes, onceNSize, nTailNum, nUbOffset);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }
    int64_t nCoreOffset = 0;
    if (blockIdx_ < tilingData_->frontCoreNum) {
        nCoreOffset = (tilingData_->frontCoreNSize + 1) * blockIdx_;
        ProcessEachCore(tilingData_->nLoopTimesB, tilingData_->onceNSize, tilingData_->nTailNumB, nCoreOffset);
    } else {
        nCoreOffset = tilingData_->frontCoreNSize * blockIdx_ + tilingData_->frontCoreNum;
        ProcessEachCore(tilingData_->nLoopTimesT, tilingData_->onceNSize, tilingData_->nTailNumT, nCoreOffset);
    }
    if constexpr (reduction != 0) {
        int32_t offset = blockIdx_ * SIZE_32;
        VToMTE3();
        DataCopyWorkspace(offset);
        SyncAll();
        ComputeNLast();
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeEachCoreOnce(
    int32_t nLoopTimes, int32_t onceNSize, int32_t nTailNum, int64_t nUbOffset)
{
    CopyInInput(nTailNum, nUbOffset * tilingData_->C);
    LocalTensor<T1> inputUbLocal = inputQueue_.DeQue<T1>();
    LocalTensor<float> maxUbLocal = maxUb_.Get<float>();
    VfReduceMax(nTailNum, tilingData_->C, tilingData_->cOnceNum, maxUbLocal, inputUbLocal);

    LocalTensor<float> sumUbLocal = sumUb_.Get<float>();
    LocalTensor<float> subUbLocal = subUb_.Get<float>();
    LocalTensor<float> tempNCLocal = tempUb_.Get<float>();
    VfSubExp(nTailNum, tilingData_->C, tilingData_->cOnceNum, maxUbLocal, inputUbLocal, subUbLocal, tempNCLocal);
    uint32_t srcShape[2] = {static_cast<uint32_t>(nTailNum), static_cast<uint32_t>(tilingData_->cOnceNum)};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(sumUbLocal, tempNCLocal, srcShape, false);
    VfComputeLogSub(nTailNum, tilingData_->C, tilingData_->cOnceNum, sumUbLocal, subUbLocal, tempNCLocal);

    LocalTensor<T1> logPropLocal = logPropQueue_.AllocTensor<T1>();
    if constexpr (sizeof(T1) == sizeof(half)) {
        AscendC::Cast(logPropLocal, tempNCLocal, AscendC::RoundMode::CAST_RINT, nTailNum * tilingData_->cOnceNum);
    } else {
        AscendC::Copy(logPropLocal, tempNCLocal, nTailNum * tilingData_->cOnceNum);
    }
    logPropQueue_.EnQue(logPropLocal);
    CopyOutPadLogProp(nTailNum, nUbOffset * tilingData_->C);
    inputQueue_.FreeTensor(inputUbLocal);

    CopyInTarget(nTailNum, nUbOffset);
    LocalTensor<T2> targetUbLocal = targetInQueue_.DeQue<T2>();
    if constexpr (isWeight == 1) {
        CopyInWeight(tilingData_->C, tilingData_->cOnceNum);
    }

    LocalTensor<float> weightLocal = weightQueue_.DeQue<float>();
    LocalTensor<float> lossOutLocal = lossOutUb_.Get<float>();
    LocalTensor<float> weightYnLocal = weightYnUb_.Get<float>();
    LocalTensor<float> smoothLossLocal = smoothLossUb_.Get<float>();
    if constexpr (labelS != 0) {
        VfComputeSmoothLoss(
            nTailNum, tilingData_->C, tilingData_->cOnceNum, tempNCLocal, weightLocal, subUbLocal, sumUbLocal);
    }
    VfComputeIgnore(
        nTailNum, nUbOffset, targetUbLocal, weightLocal, weightYnLocal, tempNCLocal, lossOutLocal, smoothLossLocal,
        sumUbLocal);
    targetInQueue_.FreeTensor(targetUbLocal);
    if constexpr (isWeight == 1) {
        weightQueue_.FreeTensor(weightLocal);
    }
    VfComputeLossOut(nTailNum, tilingData_->C, tilingData_->cOnceNum, lossOutLocal, weightYnLocal);
    if constexpr (reduction == 0) {
        VfComputeNone(nTailNum, lossOutLocal, smoothLossLocal);
        LocalTensor<T1> lossUbLocal = lossQueue_.AllocTensor<T1>();
        if constexpr (sizeof(T1) == sizeof(half)) {
            AscendC::Cast(lossUbLocal, lossOutLocal, AscendC::RoundMode::CAST_RINT, nTailNum);
        } else {
            AscendC::Copy(lossUbLocal, lossOutLocal, nTailNum);
        }
        lossQueue_.EnQue(lossUbLocal);
        CopyOutPadLoss(nTailNum, nUbOffset);
    } else {
        ComputePartSum(nTailNum, lossOutLocal, smoothLossLocal, weightYnLocal);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::ComputePartSum(
    int32_t nTailNum, LocalTensor<float>& lossOutLocal, LocalTensor<float>& smoothLossLocal,
    LocalTensor<float>& weightYnLocal)
{
    LocalTensor<float> clearUb = clearUb_.Get<float>();
    LocalTensor<float> partSumUb = partSumUb_.Get<float>();

    uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(nTailNum)};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(partSumUb[0], lossOutLocal, srcShape, false);
    AscendC::Add(clearUb[0], partSumUb[0], clearUb[0], 1);
    if constexpr (reduction == 1) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
            partSumUb[ONE_FP32], weightYnLocal, srcShape, false);
        AscendC::Add(clearUb[ONE_FP32], partSumUb[ONE_FP32], clearUb[ONE_FP32], 1);
    }
    if constexpr (labelS != 0) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
            partSumUb[ONE_FP32 * 2], smoothLossLocal, srcShape, false);
        AscendC::Add(clearUb[ONE_FP32 * 2], partSumUb[ONE_FP32 * 2], clearUb[ONE_FP32 * 2], 1);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfComputeNone(
    int32_t nTailNum, LocalTensor<float>& lossOutLocal, LocalTensor<float>& smoothLossLocal)
{
    auto smoothLossAddr = (__ubuf__ float*)smoothLossLocal.GetPhyAddr();
    auto lossOutAddr = (__ubuf__ float*)lossOutLocal.GetPhyAddr();
    float smooth1 = tilingData_->labelSmooth1;
    float sommthC = tilingData_->labelSmoothC;
    uint32_t vfLen = vfLenFp32_;
    uint32_t sreg = nTailNum;
    uint16_t repeatTimes = CeilDivision(nTailNum, vfLen);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> smoothLossReg;
        AscendC::MicroAPI::RegTensor<float> lossOutReg;
        AscendC::MicroAPI::MaskReg pregMain =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(sreg);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            if constexpr (labelS == 0) {
                AscendC::MicroAPI::DataCopy(lossOutReg, lossOutAddr + i * vfLen);
                AscendC::MicroAPI::Muls(lossOutReg, lossOutReg, smooth1, preg);
                AscendC::MicroAPI::DataCopy(lossOutAddr + i * vfLen, lossOutReg, preg);
            } else {
                AscendC::MicroAPI::DataCopy(smoothLossReg, smoothLossAddr + i * vfLen);
                AscendC::MicroAPI::DataCopy(lossOutReg, lossOutAddr + i * vfLen);
                AscendC::MicroAPI::Muls(lossOutReg, lossOutReg, smooth1, preg);
                AscendC::MicroAPI::Muls(smoothLossReg, smoothLossReg, sommthC, preg);
                AscendC::MicroAPI::Add(lossOutReg, lossOutReg, smoothLossReg, preg);
                AscendC::MicroAPI::DataCopy(lossOutAddr + i * vfLen, lossOutReg, preg);
            }
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfComputeSmoothLoss(
    int32_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float>& tempNCLocal, LocalTensor<float>& weightLocal,
    LocalTensor<float>& subUbLocal, LocalTensor<float>& sumUbLocal)
{
    auto weightAddr = (__ubuf__ float*)weightLocal.GetPhyAddr();
    auto logPropAddr = (__ubuf__ float*)tempNCLocal.GetPhyAddr();
    auto tempAddr = (__ubuf__ float*)subUbLocal.GetPhyAddr();

    uint16_t aTimes = nTailNum;
    uint32_t vfLen = vfLenFp32_;
    uint16_t repeatTimes = cNum / vfLen;
    uint32_t tailNum = cNum % vfLen;
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;
    uint32_t tailNumAlign = AlignC - repeatTimes * vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> logPropReg;
        AscendC::MicroAPI::RegTensor<float> weightReg;
        AscendC::MicroAPI::MaskReg pregMain =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
        AscendC::MicroAPI::MaskReg pregAlign = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);

        for (uint16_t i = 0; i < aTimes; i++) {
            for (uint16_t j = 0; j < repeatTimes; j++) {
                AscendC::MicroAPI::DataCopy(logPropReg, logPropAddr + i * AlignC + j * vfLen);
                if constexpr (isWeight == 1) {
                    AscendC::MicroAPI::DataCopy(weightReg, weightAddr + j * vfLen);
                    AscendC::MicroAPI::Mul(logPropReg, logPropReg, weightReg, pregMain);
                }
                AscendC::MicroAPI::DataCopy(tempAddr + i * AlignC + j * vfLen, logPropReg, pregMain);
            }
            for (uint16_t k = 0; k < tailLoop; k++) {
                AscendC::MicroAPI::DataCopy(logPropReg, logPropAddr + i * AlignC + repeatTimes * vfLen);
                if constexpr (isWeight == 1) {
                    AscendC::MicroAPI::DataCopy(weightReg, weightAddr + repeatTimes * vfLen);
                    AscendC::MicroAPI::Mul(logPropReg, logPropReg, weightReg, preg);
                }
                AscendC::MicroAPI::DataCopy(tempAddr + i * AlignC + repeatTimes * vfLen, logPropReg, pregAlign);
            }
        }
    }
    uint32_t srcShape[2] = {static_cast<uint32_t>(nTailNum), static_cast<uint32_t>(tilingData_->cOnceNum)};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(sumUbLocal, subUbLocal, srcShape, false);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfComputeLossOut(
    int32_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float>& lossOutLocal, LocalTensor<float>& weightYnLocal)
{
    auto lossOutAddr = (__ubuf__ float*)lossOutLocal.GetPhyAddr();
    auto weightYnAddr = (__ubuf__ float*)weightYnLocal.GetPhyAddr();

    uint32_t vfLen = vfLenFp32_;
    uint32_t sreg = nTailNum;
    uint16_t repeatTimes = CeilDivision(nTailNum, vfLen);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> lossOutReg;
        AscendC::MicroAPI::RegTensor<float> weightYnReg;
        AscendC::MicroAPI::MaskReg pregMain =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(sreg);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::MicroAPI::DataCopy(lossOutReg, lossOutAddr + i * vfLen);
            AscendC::MicroAPI::Muls(lossOutReg, lossOutReg, -1.0f, preg);
            if constexpr (isWeight == 1) {
                AscendC::MicroAPI::DataCopy(weightYnReg, weightYnAddr + i * vfLen);
                AscendC::MicroAPI::Mul(lossOutReg, lossOutReg, weightYnReg, preg);
            }
            AscendC::MicroAPI::DataCopy(lossOutAddr + i * vfLen, lossOutReg, preg);
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfComputeLogSub(
    int64_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float> sumUbLocal, LocalTensor<float> subUbLocal,
    LocalTensor<float> tempLocal)
{
    auto sumAddr = (__ubuf__ float*)sumUbLocal.GetPhyAddr();
    auto subAddr = (__ubuf__ float*)subUbLocal.GetPhyAddr();
    auto tempAddr = (__ubuf__ float*)tempLocal.GetPhyAddr();

    uint16_t aTimes = nTailNum;
    uint32_t vfLen = vfLenFp32_;
    uint16_t repeatTimes = cNum / vfLen;
    uint32_t tailNum = cNum % vfLen;
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;
    uint32_t tailNumAlign = AlignC - repeatTimes * vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> sumReg;
        AscendC::MicroAPI::RegTensor<float> logReg;
        AscendC::MicroAPI::RegTensor<float> subReg;
        AscendC::MicroAPI::RegTensor<float> tempReg;

        AscendC::MicroAPI::MaskReg pregMain =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
        AscendC::MicroAPI::MaskReg pregAlign = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);

        for (uint16_t i = 0; i < aTimes; i++) {
            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(sumReg, sumAddr + i);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<float>(i, AlignC, j, vfLen);
                AscendC::MicroAPI::DataCopy(subReg, subAddr, offset);
                AscendC::MicroAPI::Log(logReg, sumReg, pregMain);
                AscendC::MicroAPI::Sub(tempReg, subReg, logReg, pregMain);
                AscendC::MicroAPI::DataCopy(tempAddr, tempReg, offset, pregMain);
            }
            for (uint16_t k = 0; k < tailLoop; k++) {
                AscendC::MicroAPI::DataCopy(subReg, subAddr + i * AlignC + repeatTimes * vfLen);
                AscendC::MicroAPI::Log(logReg, sumReg, preg);
                AscendC::MicroAPI::Sub(tempReg, subReg, logReg, preg);
                AscendC::MicroAPI::DataCopy(tempAddr + i * AlignC + repeatTimes * vfLen, tempReg, preg);
            }
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfSubExp(
    int64_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float> maxUbLocal, LocalTensor<T1> inputUbLocal,
    LocalTensor<float> subUbLocal, LocalTensor<float> tempLocal)
{
    auto subAddr = (__ubuf__ float*)subUbLocal.GetPhyAddr();
    auto tempAddr = (__ubuf__ float*)tempLocal.GetPhyAddr();
    auto maxAddr = (__ubuf__ float*)maxUbLocal.GetPhyAddr();
    auto featuresAddr = (__ubuf__ T1*)inputUbLocal.GetPhyAddr();

    uint16_t aTimes = nTailNum;
    uint32_t vfLen = vfLenFp32_;
    uint16_t repeatTimes = cNum / vfLen;
    uint32_t tailNum = cNum % vfLen;
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;
    uint32_t tailNumAlign = AlignC - repeatTimes * vfLen;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> tempReg;
        AscendC::MicroAPI::RegTensor<float> subReg;

        AscendC::MicroAPI::RegTensor<T1> featuresReg;
        AscendC::MicroAPI::RegTensor<float> featuresReg32;
        AscendC::MicroAPI::RegTensor<float> maxReg32;

        AscendC::MicroAPI::MaskReg pregMain =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
        AscendC::MicroAPI::MaskReg pregAlign = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);

        for (uint16_t i = 0; i < aTimes; i++) {
            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(maxReg32, maxAddr + i);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                AscendC::MicroAPI::AddrReg offsetT = AscendC::MicroAPI::CreateAddrReg<T1>(i, AlignC, j, vfLen);
                AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<float>(i, AlignC, j, vfLen);
                if constexpr (sizeof(T1) == sizeof(half)) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        featuresReg, featuresAddr, offsetT);
                    AscendC::MicroAPI::Cast<float, T1, castTrait2>(featuresReg32, featuresReg, pregMain);
                } else {
                    AscendC::MicroAPI::DataCopy(featuresReg32, featuresAddr, offset);
                }
                AscendC::MicroAPI::Sub(subReg, featuresReg32, maxReg32, pregMain);
                AscendC::MicroAPI::Exp(tempReg, subReg, pregMain);
                AscendC::MicroAPI::DataCopy(tempAddr, tempReg, offset, pregMain);
                AscendC::MicroAPI::DataCopy(subAddr, subReg, offset, pregMain);
            }

            for (uint16_t k = 0; k < tailLoop; k++) {
                if constexpr (sizeof(T1) == sizeof(half)) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        featuresReg, featuresAddr + i * AlignC + repeatTimes * vfLen);
                    AscendC::MicroAPI::Cast<float, T1, castTrait2>(featuresReg32, featuresReg, preg);
                } else {
                    AscendC::MicroAPI::DataCopy(featuresReg32, featuresAddr + i * AlignC + repeatTimes * vfLen);
                }
                AscendC::MicroAPI::Sub(subReg, featuresReg32, maxReg32, preg);
                AscendC::MicroAPI::Exp(tempReg, subReg, preg);
                AscendC::MicroAPI::DataCopy(tempAddr + i * AlignC + repeatTimes * vfLen, tempReg, pregAlign);
                AscendC::MicroAPI::DataCopy(subAddr + i * AlignC + repeatTimes * vfLen, subReg, preg);
            }
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfReduceMax(
    int64_t nTailNum, int64_t cNum, int64_t AlignC, LocalTensor<float> maxUbLocal, LocalTensor<T1> inputUbLocal)
{
    uint16_t aTimes = nTailNum;
    uint32_t vfLen = platform::GetVRegSize() / sizeof(T1);
    uint16_t repeatTimes = cNum / vfLen;
    uint32_t tailNum = cNum % vfLen;
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;

    auto maxAddr = (__ubuf__ float*)maxUbLocal.GetPhyAddr();
    auto inputAddr = (__ubuf__ T1*)inputUbLocal.GetPhyAddr();
    auto inputAddr1 = (__ubuf__ T1*)inputUbLocal.GetPhyAddr();
    T1 minValue = MIN_VALUE;
    if constexpr (IsSameType<T1, half>::value) {
        minValue = -65504;
    }

    if (cNum < vfLen) {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> maxReg;
            AscendC::MicroAPI::RegTensor<T1> inputReg;
            AscendC::MicroAPI::RegTensor<T1> inputRegTemp;
            AscendC::MicroAPI::RegTensor<float> inputReg32;
            AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
            AscendC::MicroAPI::MaskReg mergePreg = AscendC::MicroAPI::CreateMask<float, MaskPattern::VL1>();

            for (uint16_t i = 0; i < aTimes; i++) {
                AscendC::MicroAPI::DataCopy(inputReg, inputAddr + i * AlignC);
                if constexpr (sizeof(T1) == sizeof(half)) {
                    AscendC::MicroAPI::UnPack(
                        (AscendC::MicroAPI::RegTensor<int32_t>&)inputRegTemp,
                        (AscendC::MicroAPI::RegTensor<int16_t>&)inputReg);
                    AscendC::MicroAPI::Cast<float, T1, castTrait2>(inputReg32, inputRegTemp, preg);
                    AscendC::MicroAPI::ReduceMax(maxReg, inputReg32, preg);
                } else {
                    AscendC::MicroAPI::ReduceMax(maxReg, inputReg, preg);
                }
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr + i, maxReg, mergePreg);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T1> inputRegT1;
            AscendC::MicroAPI::RegTensor<T1> inputReg;
            AscendC::MicroAPI::RegTensor<float> maxReg32;
            AscendC::MicroAPI::RegTensor<T1> maxRegTemp;
            AscendC::MicroAPI::RegTensor<float> maxRegOne32;
            AscendC::MicroAPI::RegTensor<T1> maxReg;
            AscendC::MicroAPI::RegTensor<T1> maxLowReg;
            AscendC::MicroAPI::RegTensor<float> maxLowReg32;
            AscendC::MicroAPI::RegTensor<T1> maxHighReg;
            AscendC::MicroAPI::RegTensor<float> maxHighReg32;

            AscendC::MicroAPI::MaskReg pregAllT1 =
                AscendC::MicroAPI::CreateMask<T1, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg pregAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T1>(tailNum);
            AscendC::MicroAPI::MaskReg mergePreg = AscendC::MicroAPI::CreateMask<float, MaskPattern::VL1>();

            for (uint16_t i = 0; i < aTimes; i++) {
                AscendC::MicroAPI::Duplicate(maxReg, minValue);
                for (uint16_t j = 0; j < repeatTimes; j++) {
                    AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<T1>(i, AlignC, j, vfLen);
                    AscendC::MicroAPI::DataCopy(inputReg, inputAddr, offset);
                    AscendC::MicroAPI::Max(maxReg, maxReg, inputReg, pregAllT1);
                }
                for (uint16_t k = 0; k < tailLoop; k++) {
                    AscendC::MicroAPI::DataCopy(inputRegT1, inputAddr + i * AlignC + repeatTimes * vfLen);
                    AscendC::MicroAPI::Max(maxRegTemp, maxReg, inputRegT1, preg);
                    AscendC::MicroAPI::Copy<T1, AscendC::MicroAPI::MaskMergeMode::MERGING>(maxReg, maxRegTemp, preg);
                }
                if constexpr (sizeof(T1) == sizeof(half)) {
                    AscendC::MicroAPI::UnPack<int32_t, int16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                        (RegTensor<int32_t>&)maxLowReg, (RegTensor<int16_t>&)maxReg);
                    AscendC::MicroAPI::UnPack<int32_t, int16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(
                        (RegTensor<int32_t>&)maxHighReg, (RegTensor<int16_t>&)maxReg);
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(maxLowReg32, maxLowReg, pregAll);
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(maxHighReg32, maxHighReg, pregAll);
                    AscendC::MicroAPI::Max(maxReg32, maxLowReg32, maxHighReg32, pregAll);
                    AscendC::MicroAPI::ReduceMax(maxRegOne32, maxReg32, pregAll);
                } else {
                    AscendC::MicroAPI::ReduceMax((RegTensor<T1>&)maxRegOne32, maxReg, pregAllT1);
                }
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    maxAddr + i, maxRegOne32, mergePreg);
            }
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::CopyInInput(
    int64_t nTailNum, int64_t offset)
{
    LocalTensor<T1> inputUbLocal = inputQueue_.AllocTensor<T1>();
    int64_t cNum = tilingData_->C;
    DataCopyExtParams copyParams;
    copyParams.blockCount = nTailNum;
    copyParams.blockLen = cNum * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    T1 minVal = MIN_VALUE;
    if constexpr (IsSameType<T1, half>::value) {
        minVal = -65504;
    }
    int64_t rNumAlign = tilingData_->cOnceNum;
    DataCopyPadExtParams<T1> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = rNumAlign - cNum;
    padParams.paddingValue = minVal;
    AscendC::DataCopyPad(inputUbLocal, inputGm_[offset], copyParams, padParams);
    inputQueue_.EnQue<T1>(inputUbLocal);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::CopyInTarget(
    int64_t nTailNum, int64_t offset)
{
    LocalTensor<T2> targetUbLocal = targetInQueue_.AllocTensor<T2>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = nTailNum * sizeof(T2);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPadExtParams<T2> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    AscendC::DataCopyPad(targetUbLocal, targetGm_[offset], copyParams, padParams);
    targetInQueue_.EnQue<T2>(targetUbLocal);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::CopyOutPadLogProp(
    int64_t nTailNum, int64_t offset)
{
    LocalTensor<T1> logPropLocal = logPropQueue_.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = nTailNum;
    copyParams.blockLen = tilingData_->C * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(logProbGm_[offset], logPropLocal, copyParams);
    logPropQueue_.FreeTensor(logPropLocal);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::CopyOutPadLoss(
    int64_t nTailNum, int64_t offset)
{
    LocalTensor<T1> lossUbLocal = lossQueue_.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = nTailNum * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(lossGm_[offset], lossUbLocal, copyParams);
    lossQueue_.FreeTensor(lossUbLocal);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::CopyInWeight(
    int64_t cTailNum, int64_t cAlgin)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = cTailNum * FP32_DTYPE;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    LocalTensor<float> weightUb = weightQueue_.AllocTensor<float>();
    AscendC::DataCopyPad(weightUb, weightGm_[0], copyParams, padParams);
    weightQueue_.EnQue(weightUb);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfComputeIgnore(
    int32_t nSizeNum, int64_t nOffset, LocalTensor<T2>& targetUb, LocalTensor<float>& weightLocal,
    LocalTensor<float>& weightYnLocal, LocalTensor<float>& tempNCLocal, LocalTensor<float>& lossOutLocal,
    LocalTensor<float>& smoothLossLocal, LocalTensor<float>& sumUbLocal)
{
    uint32_t batchNum = nSizeNum;
    uint32_t vfLen = platform::GetVRegSize() / sizeof(T2);
    uint16_t repeatTimes = batchNum / vfLen;
    uint32_t tailNum = batchNum % vfLen;
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;
    auto targetUbAddr = (__ubuf__ T2*)targetUb.GetPhyAddr();
    auto weightAddr = (__ubuf__ float*)weightLocal.GetPhyAddr();
    auto weightYnAddr = (__ubuf__ float*)weightYnLocal.GetPhyAddr();
    auto logPropAddr = (__ubuf__ float*)tempNCLocal.GetPhyAddr();
    auto lossOutAddr = (__ubuf__ float*)lossOutLocal.GetPhyAddr();
    auto smoothSumAddr = (__ubuf__ float*)sumUbLocal.GetPhyAddr();
    auto smoothLossAddr = (__ubuf__ float*)smoothLossLocal.GetPhyAddr();
    int32_t ignoreData = static_cast<int32_t>(tilingData_->ignoreIndex);
    int64_t rAlign = tilingData_->cOnceNum;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::MaskReg dstMask;
        AscendC::MicroAPI::RegTensor<T2> srcReg;
        preg = AscendC::MicroAPI::UpdateMask<uint32_t>(tailNum);
        AscendC::MicroAPI::RegTensor<int32_t> tempReg;
        AscendC::MicroAPI::RegTensor<int32_t> targetReg;
        AscendC::MicroAPI::MaskReg regGather;
        AscendC::MicroAPI::MaskReg regAll =
            AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<float> reg0;
        AscendC::MicroAPI::RegTensor<float> reg1;
        AscendC::MicroAPI::Duplicate(reg0, 0.0f);
        AscendC::MicroAPI::Duplicate(reg1, 1.0f);
        AscendC::MicroAPI::RegTensor<float> weightGatherReg;
        AscendC::MicroAPI::RegTensor<float> weightDestReg;
        AscendC::MicroAPI::RegTensor<float> lossOutGatherReg;
        AscendC::MicroAPI::RegTensor<float> lossOutDestReg;
        AscendC::MicroAPI::RegTensor<float> smoothLossReg;
        AscendC::MicroAPI::RegTensor<float> smoothLossDestReg;
        AscendC::MicroAPI::RegTensor<int32_t> indexReg0;
        AscendC::MicroAPI::RegTensor<int32_t> indexReg1;
        AscendC::MicroAPI::RegTensor<int32_t> indexReg2;
        AscendC::MicroAPI::RegTensor<int32_t> targetDstReg;
        AscendC::MicroAPI::Duplicate(indexReg0, static_cast<int32_t>(rAlign));
        AscendC::MicroAPI::Arange(indexReg1, 0);
        AscendC::MicroAPI::Mul(indexReg2, indexReg0, indexReg1, regAll);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            if constexpr (sizeof(T2) == sizeof(int64_t)) {
                MicroAPI::DataCopy(srcReg, targetUbAddr + i * vfLen);
                MicroAPI::Cast<int32_t, T2, castTrait1>(tempReg, srcReg, regAll);
                MicroAPI::Pack((RegTensor<uint32_t>&)targetReg, (RegTensor<uint64_t>&)tempReg);
                regGather = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::H>();
            } else {
                MicroAPI::DataCopy(targetReg, targetUbAddr + i * vfLen);
                regGather = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            }
            MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(dstMask, targetReg, ignoreData, regAll);

            if constexpr (isWeight == 1) {
                MicroAPI::DataCopyGather(
                    weightGatherReg, weightAddr, (MicroAPI::RegTensor<uint32_t>&)targetReg, regGather);
                MicroAPI::Select(weightDestReg, reg0, weightGatherReg, dstMask);
                MicroAPI::DataCopy(weightYnAddr + i * vfLen, weightDestReg, regGather);
            } else {
                MicroAPI::Select(weightDestReg, reg0, reg1, dstMask);
                MicroAPI::DataCopy(weightYnAddr + i * vfLen, weightDestReg, regGather);
            }
            if constexpr (labelS != 0) {
                MicroAPI::DataCopy(smoothLossReg, smoothSumAddr + i * vfLen);
                MicroAPI::Muls(smoothLossReg, smoothLossReg, -1.0f, regAll);
                MicroAPI::Select(smoothLossDestReg, reg0, smoothLossReg, dstMask);
                MicroAPI::DataCopy(smoothLossAddr + i * vfLen, smoothLossDestReg, regGather);
            }
            MicroAPI::Add(targetDstReg, indexReg2, targetReg, regAll);
            MicroAPI::DataCopyGather(
                lossOutGatherReg, logPropAddr + i * vfLen * rAlign, (MicroAPI::RegTensor<uint32_t>&)targetDstReg,
                regGather);
            MicroAPI::Select(lossOutDestReg, reg0, lossOutGatherReg, dstMask);
            MicroAPI::DataCopy(lossOutAddr + i * vfLen, lossOutDestReg, regGather);
        }
        for (uint16_t k = 0; k < tailLoop; k++) {
            if constexpr (sizeof(T2) == sizeof(int64_t)) {
                MicroAPI::DataCopy(srcReg, targetUbAddr + repeatTimes * vfLen);
                MicroAPI::Cast<int32_t, T2, castTrait1>(tempReg, srcReg, regAll);
                MicroAPI::Pack((RegTensor<uint32_t>&)targetReg, (RegTensor<uint64_t>&)tempReg);
            } else {
                MicroAPI::DataCopy(targetReg, targetUbAddr + repeatTimes * vfLen);
            }
            MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(dstMask, targetReg, ignoreData, regAll);

            if constexpr (isWeight == 1) {
                MicroAPI::DataCopyGather(weightGatherReg, weightAddr, (MicroAPI::RegTensor<uint32_t>&)targetReg, preg);
                MicroAPI::Select(weightDestReg, reg0, weightGatherReg, dstMask);
                MicroAPI::DataCopy(weightYnAddr + repeatTimes * vfLen, weightDestReg, preg);
            } else {
                MicroAPI::Select(weightDestReg, reg0, reg1, dstMask);
                MicroAPI::DataCopy(weightYnAddr + repeatTimes * vfLen, weightDestReg, preg);
            }
            if constexpr (labelS != 0) {
                MicroAPI::DataCopy(smoothLossReg, smoothSumAddr + repeatTimes * vfLen);
                MicroAPI::Muls(smoothLossReg, smoothLossReg, -1.0f, regAll);
                MicroAPI::Select(smoothLossDestReg, reg0, smoothLossReg, dstMask);
                MicroAPI::DataCopy(smoothLossAddr + repeatTimes * vfLen, smoothLossDestReg, preg);
            }
            MicroAPI::Add(targetDstReg, indexReg2, targetReg, regAll);
            MicroAPI::DataCopyGather(
                lossOutGatherReg, logPropAddr + repeatTimes * vfLen * rAlign,
                (MicroAPI::RegTensor<uint32_t>&)targetDstReg, preg);
            MicroAPI::Select(lossOutDestReg, reg0, lossOutGatherReg, dstMask);
            MicroAPI::DataCopy(lossOutAddr + repeatTimes * vfLen, lossOutDestReg, preg);
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::CleanRUb(
    LocalTensor<float>& clearUb)
{
    uint32_t vfLen = vfLenFp32_;

    auto clearUbAddr = (__ubuf__ float*)clearUb.GetPhyAddr();
    uint32_t aa = 24;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> srcReg0;
        for (uint16_t j = 0; j < 1; j++) {
            AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(aa);
            AscendC::MicroAPI::Duplicate(srcReg0, 0.0f);
            AscendC::MicroAPI::DataCopy(clearUbAddr, srcReg0, preg);
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VToS()
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::SToV()
{
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::MTE2ToV()
{
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VToMTE3()
{
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::MTE2ToS()
{
    event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyWorkspace(
    int32_t offset)
{
    LocalTensor<float> clearUb = clearUb_.Get<float>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = sumNum;
    copyParams.blockLen = FP32_DTYPE;
    copyParams.srcStride = 0;
    copyParams.dstStride = (tilingData_->realCoreNum * SIZE_32 - 1) * FP32_DTYPE;
    DataCopyPad(workspaceGm_[offset], clearUb, copyParams);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::ComputeNLast()
{
    if (blockIdx_ == 0) {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<int32_t>(tilingData_->realCoreNum) * sumNum;
        copyParams.blockLen = FP32_DTYPE;
        copyParams.srcStride = (SIZE_32 - 1) * FP32_DTYPE;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (ONE_FP32 - 1);
        padParams.paddingValue = 0.0f;
        LocalTensor<float> allReduceUb = inputQueue_.AllocTensor<float>();
        AscendC::DataCopyPad(allReduceUb, workspaceGm_[0], copyParams, padParams);
        MTE2ToV();
        int32_t num = static_cast<int32_t>(tilingData_->realCoreNum) * ONE_FP32;
        LocalTensor<float> clearUb = clearUb_.Get<float>();
        LocalTensor<T1> clearUbT1 = clearUb_.Get<T1>();
        VfAllSum(num, allReduceUb, clearUb);
        DataCopyOutLoss(clearUb, clearUbT1);
        inputQueue_.FreeTensor(allReduceUb);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfAllSum(
    int32_t size, LocalTensor<float>& allReduceUb, LocalTensor<float>& clearUb)
{
    int32_t vfLen = vfLenFp32_;
    auto allReduceUbLddr = (__ubuf__ float*)allReduceUb.GetPhyAddr();
    auto allReduceUbWddr = (__ubuf__ float*)allReduceUb.GetPhyAddr() + size;
    auto allReduceUbSddr = (__ubuf__ float*)allReduceUb.GetPhyAddr() + size * 2;
    auto clearAddr = (__ubuf__ float*)clearUb.GetPhyAddr();
    uint16_t repeatTimes1 = size / vfLen;
    uint32_t tail = size % vfLen;
    uint16_t tailLoop = tail > 0 ? 1 : 0;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::RegTensor<float> srcReg0;
        AscendC::MicroAPI::RegTensor<float> srcReg1;
        AscendC::MicroAPI::RegTensor<float> srcRegT;
        AscendC::MicroAPI::RegTensor<float> srcRegW0;
        AscendC::MicroAPI::RegTensor<float> srcRegW1;
        AscendC::MicroAPI::RegTensor<float> srcRegWT;
        AscendC::MicroAPI::RegTensor<float> srcRegS0;
        AscendC::MicroAPI::RegTensor<float> srcRegS1;
        AscendC::MicroAPI::RegTensor<float> srcRegST;
        AscendC::MicroAPI::MaskReg regAllFp32 =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg2 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::Duplicate(srcReg0, 0.0f);
        AscendC::MicroAPI::Duplicate(srcRegW0, 0.0f);
        AscendC::MicroAPI::Duplicate(srcRegS0, 0.0f);
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
            VfAddAll(srcReg0, srcReg1, allReduceUbLddr, regAllFp32, srcOffset);
            if constexpr (reduction == 1) {
                VfAddAll(srcRegW0, srcRegW1, allReduceUbWddr, regAllFp32, srcOffset);
            }
            if constexpr (labelS != 0) {
                VfAddAll(srcRegS0, srcRegS1, allReduceUbSddr, regAllFp32, srcOffset);
            }
        }
        preg = AscendC::MicroAPI::UpdateMask<float>(tail);
        VfAddTail(srcReg0, srcReg1, srcRegT, preg, regAllFp32, preg2, repeatTimes1, vfLen, allReduceUbLddr, clearAddr);
        if constexpr (reduction == 1) {
            VfAddTail(
                srcRegW0, srcRegW1, srcRegWT, preg, regAllFp32, preg2, repeatTimes1, vfLen, allReduceUbWddr,
                clearAddr + 8);
        }
        if constexpr (labelS != 0) {
            VfAddTail(
                srcRegS0, srcRegS1, srcRegST, preg, regAllFp32, preg2, repeatTimes1, vfLen, allReduceUbSddr,
                clearAddr + 16);
        }
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfAddAll(
    RegTensor<float>& srcReg0, RegTensor<float>& srcReg1, __local_mem__ float* srcAddr, MaskReg& regAllFp32,
    AddrReg& srcOffset)
{
    AscendC::MicroAPI::DataCopy(srcReg1, srcAddr, srcOffset);
    AscendC::MicroAPI::Add(srcReg0, srcReg1, srcReg0, regAllFp32);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::VfAddTail(
    RegTensor<float>& srcReg0, RegTensor<float>& srcReg1, RegTensor<float>& srcRegT, MaskReg& preg, MaskReg& regAllFp32,
    MaskReg& preg2, uint16_t repeatTimes1, uint32_t vfLen, __local_mem__ float* srcAddr, __local_mem__ float* dstAddr)
{
    AscendC::MicroAPI::DataCopy(srcReg1, srcAddr + repeatTimes1 * vfLen);
    AscendC::MicroAPI::Add(srcRegT, srcReg1, srcReg0, preg);
    AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(srcReg0, srcRegT, preg);
    AscendC::MicroAPI::ReduceSum(srcRegT, srcReg0, regAllFp32);
    AscendC::MicroAPI::DataCopy(dstAddr, srcRegT, preg2);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyOutLoss(
    LocalTensor<float>& clearUb, LocalTensor<T1>& clearUbT1)
{
    if constexpr (labelS == 0) {
        DataCopyOutS0(clearUb, clearUbT1);
    } else {
        DataCopyOutS1(clearUb, clearUbT1);
    }
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyOutS0(
    LocalTensor<float>& clearUb, LocalTensor<T1>& clearUbT1)
{
    if constexpr (reduction == 1) {
        AscendC::Div(clearUb[0], clearUb, clearUb[ONE_FP32], 1);
    }
    DataCopyCommon(clearUb, clearUbT1);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyOutS1(
    LocalTensor<float>& clearUb, LocalTensor<T1>& clearUbT1)
{
    float lableS1 = tilingData_->labelSmooth1;
    float lableSc = tilingData_->labelSmoothC;
    if constexpr (reduction == 2) {
        AscendC::Muls(clearUb, clearUb, lableS1, 1);
        AscendC::Muls(clearUb[ONE_FP32 * 2], clearUb[ONE_FP32 * 2], lableSc, 1);
        AscendC::Add(clearUb, clearUb, clearUb[ONE_FP32 * 2], 1);
    } else {
        AscendC::Div(clearUb[ONE_FP32 * 2], clearUb[ONE_FP32 * 2], clearUb[ONE_FP32], 1);
        AscendC::Div(clearUb[0], clearUb, clearUb[ONE_FP32], 1);
        AscendC::Muls(clearUb[ONE_FP32 * 2], clearUb[ONE_FP32 * 2], lableSc, 1);
        AscendC::Muls(clearUb[0], clearUb[0], lableS1, 1);
        AscendC::Add(clearUb, clearUb, clearUb[ONE_FP32 * 2], 1);
    }
    DataCopyCommon(clearUb, clearUbT1);
}

template <typename T1, typename T2, uint64_t reduction, uint64_t isWeight, uint64_t labelS, uint64_t ignorex>
__aicore__ inline void CrossEntropyLossFullLoad<T1, T2, reduction, isWeight, labelS, ignorex>::DataCopyCommon(
    LocalTensor<float>& clearUb, LocalTensor<T1>& clearUbT1)
{
    if constexpr (sizeof(T1) == sizeof(half)) {
        AscendC::Cast(clearUbT1[ONE_FP32 * 2], clearUb, AscendC::RoundMode::CAST_RINT, 1);
        VToMTE3();
        DataCopyPad(lossGm_, clearUbT1[ONE_FP32 * 2], {1, 2, 0, 0});
    } else {
        VToMTE3();
        DataCopyPad(lossGm_, clearUb[0], {1, 4, 0, 0});
    }
}

} // namespace CrossEntropyLoss
#endif // namespace CrossEntropyLossFullLoad
