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
 * \file cross_entropy_loss_grad_full_load.h
 * \brief
 */
#ifndef CROSS_ENTROPY_LOSS_GRAD_FULL_LOAD_H
#define CROSS_ENTROPY_LOSS_GRAD_FULL_LOAD_H

#include "cross_entropy_loss_grad_base.h"

namespace CrossEntropyLossGradRegbase {

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
class CELossGradFullLoad : protected CrossEntropyLossGradBase<T1, T2> {
public:
    __aicore__ inline CELossGradFullLoad(){};
    __aicore__ inline void Init(GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target, GM_ADDR weight, GM_ADDR grad_zloss,
                                GM_ADDR lse_for_zloss, GM_ADDR x_grad, GM_ADDR workspace,
                                const CrossEntropyLossGradRegbaseTilingData& tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputerEachBatch(uint64_t peerLoopN, uint64_t nLoopNum);
    __aicore__ inline void ComputeLogSoftmax(LocalTensor<T1>& logProbLocal, uint64_t nLoopNum);
    __aicore__ inline void ComputePredictGradLoss(LocalTensor<T2>& targetLocal, uint64_t nLoopNum);
    __aicore__ inline void SmoothLabel(uint64_t nLoopNum, uint64_t calcLen, LocalTensor<T1>& logProbLocal);
    __aicore__ inline void ProcessPerCore(uint64_t peerLoopNIdx, uint64_t nLoopNum);
    __aicore__ inline void GetGradLoss(float& gradLossValue, LocalTensor<T1>& gradLossLocal, uint64_t targetNum,
                                       uint64_t peerLoopNIdx);
    __aicore__ inline void GradReduction(float factor, uint64_t targetNum, uint64_t peerLoopNIdx);
    __aicore__ inline void GradReductionSmooth(float factor, uint64_t targetNum, uint64_t peerLoopNIdx);
    __aicore__ inline void ComputeLogSmoothLoss(uint64_t nLoopNum, uint64_t cOnceNumTail, uint64_t cOnceNumAlign,
                                                LocalTensor<float>& sumUb);
    __aicore__ inline void ComputeIgnoreAndTarget(uint64_t targetNum, uint64_t peerLoopNIndex);
    __aicore__ inline void TargetWeightReduceSumFullLoad();

private:
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> fp32Buf7;
    TBuf<TPosition::VECCALC> targetWeightBuf;

    LocalTensor<T1> xGradLocal;
    LocalTensor<float> fp32Buf4Local;
    LocalTensor<float> targetWeightLocal;
    LocalTensor<float> weightLocal;
    constexpr static Reg::DivSpecificMode divMode = {Reg::MaskMergeMode::ZEROING, false,
                                                     DivAlgo::PRECISION_1ULP_FTZ_FALSE};

    constexpr static Reg::CastTrait castS642S32 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                   Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static Reg::CastTrait castTrait1 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    int64_t peerLoopNum = 0;
};

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::Init(
    GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target, GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
    GM_ADDR x_grad, GM_ADDR workspace, const CrossEntropyLossGradRegbaseTilingData& tilingData)
{
    CrossEntropyLossGradBase<T1, T2>::Init(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad,
                                           workspace, tilingData);
    peerLoopNum = this->alignNCNum / this->alignColLoopNum;
    this->nLoopTimes = this->nPeerCoreNum / peerLoopNum;
    this->nLoopTail = this->nPeerCoreNum % peerLoopNum;
    this->pipe.InitBuffer(tmpBuf, this->alignNCNum * sizeof(float));
    fp32Buf4Local = tmpBuf.Get<float>();
    if constexpr (isWeight == 1) {
        this->pipe.InitBuffer(this->inQueWeight_, BUFFER_NUM, this->alignColLoopNum * sizeof(float));
        this->pipe.InitBuffer(targetWeightBuf, this->targetWeightSize);
        targetWeightLocal = targetWeightBuf.Get<float>();
        if constexpr (labelS == 1) {
            this->pipe.InitBuffer(sumBuf, this->targetCastSize);
            this->pipe.InitBuffer(fp32Buf7, this->alignNCNum * sizeof(float));
        }
    }
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::ComputeLogSmoothLoss(
    uint64_t nLoopNum, uint64_t cOnceNumTail, uint64_t cOnceNumAlign, LocalTensor<float>& sumUb)
{
    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t repeatTimes1 = cOnceNumTail / vfLen;
    uint32_t tailNum = cOnceNumTail % vfLen;
    uint32_t tailNumAlign = cOnceNumAlign - repeatTimes1 * vfLen;
    uint16_t nNum = nLoopNum;
    uint16_t cAlign = cOnceNumAlign;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;
    LocalTensor<float> tmpLocal = fp32Buf7.Get<float>();
    auto smGradAddr = (__ubuf__ float*)tmpLocal.GetPhyAddr();
    auto weightAddr = (__ubuf__ float*)weightLocal.GetPhyAddr();
    auto smGradTailAddr = (__ubuf__ float*)tmpLocal.GetPhyAddr() + repeatTimes1 * vfLen;
    auto weightTailAddr = (__ubuf__ float*)weightLocal.GetPhyAddr() + repeatTimes1 * vfLen;
    auto smoothAddr = (__ubuf__ float*)this->smoothLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::MaskReg copyOutReg = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> regWeight;
        Reg::RegTensor<float> regOut;
        Reg::RegTensor<float> regSmoothGrad;
        Reg::RegTensor<float> regOutAdd;
        Reg::MaskReg preg = Reg::UpdateMask<float>(tailNum);
        Reg::MaskReg preg1 = Reg::UpdateMask<float>(tailNumAlign);

        for (uint16_t ni = 0; ni < nNum; ni++) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(regSmoothGrad, smoothAddr + ni);
            for (uint16_t loop = 0; loop < repeatTimes1; loop++) {
                Reg::AddrReg srcOffset = Reg::CreateAddrReg<float>(loop, vfLen);
                Reg::AddrReg srcOffset1 = Reg::CreateAddrReg<float>(ni, cAlign, loop, vfLen);
                Reg::LoadAlign(regWeight, weightAddr, srcOffset);
                Reg::Mul(regWeight, regWeight, regSmoothGrad, copyOutReg);
                Reg::StoreAlign(smGradAddr, regWeight, srcOffset1, copyOutReg);
            }
            for (uint16_t loopT = 0; loopT < tailLoopTimes; loopT++) {
                Reg::LoadAlign(regWeight, weightTailAddr);
                Reg::Mul(regWeight, regWeight, regSmoothGrad, preg);
                Reg::StoreAlign(smGradTailAddr + ni * cAlign, regWeight, preg1);
            }
        }
    }
    uint32_t srcShape[2] = {static_cast<uint32_t>(nLoopNum), static_cast<uint32_t>(cOnceNumAlign)};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, tmpLocal, srcShape, false);
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::SmoothLabel(
    uint64_t nLoopNum, uint64_t calcLen, LocalTensor<T1>& logProbLocal)
{
    float smoothGradreduceSum = this->colVal;
    __ubuf__ float* weightAddr;
    __ubuf__ float* sumAddr;
    if constexpr (isWeight == 1) {
        LocalTensor<float> sumUb = sumBuf.template Get<float>();
        ComputeLogSmoothLoss(nLoopNum, calcLen, this->alignColLoopNum, sumUb);
        weightAddr = (__ubuf__ float*)weightLocal.GetPhyAddr();
        sumAddr = (__ubuf__ float*)sumUb.GetPhyAddr();
    }
    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint32_t totalCLen = calcLen;
    uint16_t nNum = nLoopNum;
    uint32_t totalCLenAlign = this->alignColLoopNum;
    uint16_t repeatTimes = totalCLen / vfLen;
    uint32_t tailNum = totalCLen % vfLen;
    uint32_t tailNumAlign = totalCLenAlign - repeatTimes * vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;
    auto logProbAddr = (__ubuf__ T1*)logProbLocal.GetPhyAddr();
    auto xGradTmpAddr = (__ubuf__ float*)fp32Buf4Local.GetPhyAddr();
    auto xGradAddr = (__ubuf__ T1*)xGradLocal.GetPhyAddr();
    auto smoothGradAttr = (__ubuf__ float*)this->smoothLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::RegTensor<T1> regLogProb;
        Reg::RegTensor<float> regLogProbB32;
        Reg::RegTensor<float> regXGrad;
        Reg::RegTensor<T1> regSmooth;
        Reg::RegTensor<float> regSmoothB32;
        Reg::RegTensor<float> regSmoothGrad;
        Reg::RegTensor<float> regSmoothGradSum;
        Reg::RegTensor<float> regWeight;
        Reg::MaskReg regAllFp32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregUp = Reg::UpdateMask<float>(tailNum);
        Reg::MaskReg pregUp2 = Reg::UpdateMask<T1>(tailNumAlign);
        for (uint16_t ni = 0; ni < nNum; ni++) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regSmoothGrad, smoothGradAttr + ni);
            if constexpr (isWeight == 1) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regSmoothGradSum, sumAddr + ni);
            }
            for (uint16_t loop = 0; loop < repeatTimes; loop++) {
                Reg::AddrReg srcOffset = Reg::CreateAddrReg<T1>(ni, totalCLenAlign, loop, vfLen);
                Reg::AddrReg outOffset = Reg::CreateAddrReg<float>(ni, totalCLenAlign, loop, vfLen);
                Reg::AddrReg srcOffset1 = Reg::CreateAddrReg<float>(loop, vfLen);
                if constexpr (!IsSameType<T1, float>::value) {
                    Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(regLogProb, logProbAddr, srcOffset);
                    Reg::Cast<float, T1, castTrait0>(regLogProbB32, regLogProb, regAllFp32);
                } else {
                    Reg::LoadAlign(regLogProbB32, logProbAddr, srcOffset);
                }
                Reg::LoadAlign(regXGrad, xGradTmpAddr, outOffset);
                Reg::Exp(regLogProbB32, regLogProbB32, regAllFp32);
                if constexpr (isWeight == 1) {
                    Reg::Mul(regSmoothB32, regLogProbB32, regSmoothGradSum, regAllFp32);
                    Reg::LoadAlign(regWeight, weightAddr, srcOffset1);
                    Reg::Mul(regWeight, regSmoothGrad, regWeight, regAllFp32);
                    Reg::Sub(regSmoothB32, regSmoothB32, regWeight, regAllFp32);
                } else {
                    Reg::Muls(regSmoothGradSum, regSmoothGrad, smoothGradreduceSum, regAllFp32);
                    Reg::Mul(regSmoothB32, regLogProbB32, regSmoothGradSum, regAllFp32);
                    Reg::Sub(regSmoothB32, regSmoothB32, regSmoothGrad, regAllFp32);
                }
                Reg::Add(regSmoothB32, regSmoothB32, regXGrad, regAllFp32);
                if constexpr (!IsSameType<T1, float>::value) {
                    Reg::Cast<T1, float, castTrait1>(regSmooth, regSmoothB32, regAllFp32);
                    Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B32>(xGradAddr, regSmooth, srcOffset, regAllFp32);
                } else {
                    Reg::StoreAlign(xGradAddr, regSmoothB32, srcOffset, regAllFp32);
                }
            }
            for (uint16_t k = 0; k < tailLoopTimes; k++) {
                if constexpr (!IsSameType<T1, float>::value) {
                    Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(
                        regLogProb, logProbAddr + ni * totalCLenAlign + vfLen * repeatTimes);
                    Reg::Cast<float, T1, castTrait0>(regLogProbB32, regLogProb, pregUp);
                } else {
                    Reg::LoadAlign(regLogProbB32, logProbAddr + ni * totalCLenAlign + vfLen * repeatTimes);
                }
                Reg::LoadAlign(regXGrad, xGradTmpAddr + ni * totalCLenAlign + vfLen * repeatTimes);
                Reg::Exp(regLogProbB32, regLogProbB32, pregUp);
                if constexpr (isWeight == 1) {
                    Reg::Mul(regSmoothB32, regLogProbB32, regSmoothGradSum, regAllFp32);
                    Reg::LoadAlign(regWeight, weightAddr + vfLen * repeatTimes);
                    Reg::Mul(regWeight, regSmoothGrad, regWeight, pregUp);
                    Reg::Sub(regSmoothB32, regSmoothB32, regWeight, pregUp);
                } else {
                    Reg::Muls(regSmoothGradSum, regSmoothGrad, smoothGradreduceSum, pregUp);
                    Reg::Mul(regSmoothB32, regLogProbB32, regSmoothGradSum, pregUp);
                    Reg::Sub(regSmoothB32, regSmoothB32, regSmoothGrad, pregUp);
                }
                Reg::Add(regSmoothB32, regSmoothB32, regXGrad, pregUp);
                if constexpr (!IsSameType<T1, float>::value) {
                    Reg::Cast<T1, float, castTrait1>(regSmooth, regSmoothB32, pregUp);
                    Reg::StoreAlign<T1, Reg::StoreDist::DIST_PACK_B32>(
                        xGradAddr + ni * totalCLenAlign + vfLen * repeatTimes, regSmooth, pregUp);
                } else {
                    Reg::StoreAlign(xGradAddr + ni * totalCLenAlign + vfLen * repeatTimes, regSmoothB32, pregUp2);
                }
            }
        }
    }
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::ComputeLogSoftmax(
    LocalTensor<T1>& logProbLocal, uint64_t nLoopNum)
{
    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint32_t totalCLen = this->colVal;
    uint16_t nNum = nLoopNum;
    uint32_t totalCLenAlign = this->alignColLoopNum;
    uint16_t repeatTimes = totalCLen / vfLen;
    uint32_t tailNum = totalCLen % vfLen;
    uint32_t tailNumAlign = totalCLenAlign - repeatTimes * vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;
    auto logProbAddr = (__ubuf__ T1*)logProbLocal.GetPhyAddr();
    auto nllLossGradAddr = (__ubuf__ float*)this->targetCast.GetPhyAddr();
    auto xGradTmpAddr = (__ubuf__ float*)fp32Buf4Local.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::RegTensor<T1> regLogProb;
        Reg::RegTensor<float> regLogProbB32;
        Reg::RegTensor<float> regNllLossGrad;
        Reg::MaskReg regAllFp32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregUp = Reg::UpdateMask<float>(tailNum);
        Reg::MaskReg pregUp1 = Reg::UpdateMask<float>(tailNumAlign);
        for (uint16_t ni = 0; ni < nNum; ni++) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regNllLossGrad, nllLossGradAddr + ni);
            for (uint16_t loop = 0; loop < repeatTimes; loop++) {
                Reg::AddrReg srcOffset = Reg::CreateAddrReg<T1>(ni, totalCLenAlign, loop, vfLen);
                Reg::AddrReg outOffset = Reg::CreateAddrReg<float>(ni, totalCLenAlign, loop, vfLen);
                if constexpr (!IsSameType<T1, float>::value) {
                    Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(regLogProb, logProbAddr, srcOffset);
                    Reg::Cast<float, T1, castTrait0>(regLogProbB32, regLogProb, regAllFp32);
                } else {
                    Reg::LoadAlign(regLogProbB32, logProbAddr, srcOffset);
                }
                Reg::Exp(regLogProbB32, regLogProbB32, regAllFp32);
                Reg::Mul(regLogProbB32, regLogProbB32, regNllLossGrad, regAllFp32);
                Reg::StoreAlign(xGradTmpAddr, regLogProbB32, outOffset, regAllFp32);
            }
            for (uint16_t k = 0; k < tailLoopTimes; k++) {
                if constexpr (!IsSameType<T1, float>::value) {
                    Reg::LoadAlign<T1, Reg::LoadDist::DIST_UNPACK_B16>(
                        regLogProb, logProbAddr + ni * totalCLenAlign + vfLen * repeatTimes);
                    Reg::Cast<float, T1, castTrait0>(regLogProbB32, regLogProb, pregUp);
                } else {
                    Reg::LoadAlign(regLogProbB32, logProbAddr + ni * totalCLenAlign + vfLen * repeatTimes);
                }
                Reg::Exp(regLogProbB32, regLogProbB32, pregUp);
                Reg::Mul(regLogProbB32, regLogProbB32, regNllLossGrad, pregUp);
                Reg::StoreAlign(xGradTmpAddr + ni * totalCLenAlign + vfLen * repeatTimes, regLogProbB32, pregUp1);
            }
        }
    }
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::ComputePredictGradLoss(
    LocalTensor<T2>& targetLocal, uint64_t nLoopNum)
{
    uint32_t vfLen = platform::GetVRegSize() / sizeof(T2);
    uint32_t nNum = nLoopNum;
    uint32_t totalCLenAlign = this->alignColLoopNum;
    auto targetAddr = (__ubuf__ T2*)targetLocal.GetPhyAddr();
    auto nllLossGradAddr = (__ubuf__ float*)this->targetCast.GetPhyAddr();
    auto xGradTmpAddr = (__ubuf__ float*)fp32Buf4Local.GetPhyAddr();
    uint16_t repeatTimes = nNum / vfLen;
    uint32_t tailNum = nNum % vfLen;
    uint16_t tailLoopNums = tailNum != 0 ? 1 : 0;
    auto targetAddrT = (__ubuf__ T2*)targetLocal.GetPhyAddr() + repeatTimes * vfLen;
    auto nllLossGradAddrT = (__ubuf__ float*)this->targetCast.GetPhyAddr() + repeatTimes * vfLen;
    auto xGradTmpAddrT = (__ubuf__ float*)fp32Buf4Local.GetPhyAddr() + repeatTimes * vfLen * totalCLenAlign;
    __VEC_SCOPE__
    {
        Reg::MaskReg regAllGather;
        Reg::RegTensor<T2> regTarget;
        Reg::RegTensor<int32_t> regTargetB32Tmp;
        Reg::RegTensor<int32_t> regTargetB32;
        Reg::RegTensor<int32_t> regIndex1;
        Reg::RegTensor<int32_t> regIndex0;
        Reg::RegTensor<float> regGather1;
        Reg::RegTensor<float> regGather;
        Reg::RegTensor<float> regLogProbB32;
        Reg::RegTensor<float> regNllLossGrad;
        Reg::MaskReg regAllB32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregUp = Reg::UpdateMask<uint32_t>(tailNum);
        Reg::Arange(regIndex1, 0);
        Reg::Duplicate(regIndex0, totalCLenAlign, regAllB32);
        Reg::Mul(regIndex1, regIndex0, regIndex1, regAllB32);

        for (uint16_t loop = 0; loop < repeatTimes; loop++) {
            Reg::AddrReg srcOffset = Reg::CreateAddrReg<T2>(loop, vfLen);
            Reg::AddrReg outOffset = Reg::CreateAddrReg<float>(loop, vfLen);
            if constexpr (IsSameType<T2, int64_t>::value) {
                Reg::LoadAlign(regTarget, targetAddr, srcOffset);
                Reg::Cast<int32_t, T2, castS642S32>(regTargetB32Tmp, regTarget, regAllB32);
                Reg::Pack((Reg::RegTensor<uint32_t>&)regTargetB32, (Reg::RegTensor<uint64_t>&)regTargetB32Tmp);
                regAllGather = Reg::CreateMask<int32_t, Reg::MaskPattern::H>();
            } else {
                Reg::LoadAlign(regTargetB32, targetAddr, srcOffset);
                regAllGather = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
            }
            Reg::Add(regTargetB32, regTargetB32, regIndex1, regAllB32);
            Reg::LoadAlign(regNllLossGrad, nllLossGradAddr, outOffset);
            Reg::Gather(regGather, xGradTmpAddr + loop * vfLen * totalCLenAlign,
                        (Reg::RegTensor<uint32_t>&)regTargetB32, regAllGather);
            Reg::Sub(regGather1, regGather, regNllLossGrad, regAllB32);
            Reg::Scatter(xGradTmpAddr + loop * vfLen * totalCLenAlign, regGather1,
                         (Reg::RegTensor<uint32_t>&)regTargetB32, regAllGather);
        }
        for (uint16_t k = 0; k < tailLoopNums; k++) {
            if constexpr (IsSameType<T2, int64_t>::value) {
                Reg::LoadAlign(regTarget, targetAddrT);
                Reg::Cast<int32_t, T2, castS642S32>(regTargetB32Tmp, regTarget, regAllB32);
                Reg::Pack((Reg::RegTensor<uint32_t>&)regTargetB32, (Reg::RegTensor<uint64_t>&)regTargetB32Tmp);
            } else {
                Reg::LoadAlign(regTargetB32, targetAddrT);
            }
            Reg::Add(regTargetB32, regTargetB32, regIndex1, regAllB32);
            Reg::LoadAlign(regNllLossGrad, nllLossGradAddrT);
            Reg::Gather(regGather, xGradTmpAddrT, (Reg::RegTensor<uint32_t>&)regTargetB32, pregUp);
            Reg::Sub(regGather1, regGather, regNllLossGrad, regAllB32);
            Reg::Scatter(xGradTmpAddrT, regGather1, (Reg::RegTensor<uint32_t>&)regTargetB32, pregUp);
        }
    }
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::ComputerEachBatch(
    uint64_t peerLoopN, uint64_t nLoopNum)
{
    uint32_t cacleCLen = this->colVal;
    uint16_t nNum = nLoopNum;
    LocalTensor<T1> logProbLocal = this->inQueLogProb_.template AllocTensor<T1>();
    DataCopyExtParams copyParams{nNum, static_cast<uint32_t>(cacleCLen * sizeof(T1)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPad(logProbLocal, this->logProbGm[peerLoopN * cacleCLen], copyParams, padParams);
    this->inQueLogProb_.template EnQue(logProbLocal);
    logProbLocal = this->inQueLogProb_.template DeQue<T1>();
    ComputeLogSoftmax(logProbLocal, nLoopNum);

    LocalTensor<T2> targetLocal = this->inQueTarget_.template AllocTensor<T2>();
    DataCopyExtParams copyParams1{1, static_cast<uint32_t>(nLoopNum * sizeof(T2)), 0, 0, 0};
    DataCopyPadExtParams<T2> padParams1{false, 0, 0, 0};
    DataCopyPad(targetLocal, this->targetGm[this->targetOffset + peerLoopN], copyParams1, padParams1);
    this->inQueTarget_.template EnQue(targetLocal);
    targetLocal = this->inQueTarget_.template DeQue<T2>();
    ComputePredictGradLoss(targetLocal, nLoopNum);
    this->inQueTarget_.template FreeTensor(targetLocal);
    xGradLocal = this->outQueXGrad_.template AllocTensor<T1>();
    if constexpr (labelS == 0) {
        if constexpr (IsSameType<T1, float>::value) {
            Copy(xGradLocal, fp32Buf4Local, this->alignColLoopNum * nLoopNum);
        } else {
            Cast(xGradLocal, fp32Buf4Local, RoundMode::CAST_RINT, this->alignColLoopNum * nLoopNum);
        }
    } else {
        SmoothLabel(nLoopNum, this->colVal, logProbLocal);
    }
    this->inQueLogProb_.template FreeTensor(logProbLocal);
    this->outQueXGrad_.template EnQue<T1>(xGradLocal);
    xGradLocal = this->outQueXGrad_.template DeQue<T1>();
    DataCopyExtParams copyParams2{nNum, static_cast<uint32_t>(cacleCLen * sizeof(T1)), 0, 0, 0};
    DataCopyPad(this->xGradGm[peerLoopN * cacleCLen], xGradLocal, copyParams2);
    this->outQueXGrad_.template FreeTensor(xGradLocal);
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void
CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::TargetWeightReduceSumFullLoad()
{
    Duplicate(this->blockBuf2Local[0], 0.0f, 1);
    for (uint64_t n = 0; n < this->nLoopTimes; n++) {
        uint64_t nIdx = n * peerLoopNum;
        ComputeIgnoreAndTarget(peerLoopNum, nIdx);
        uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(peerLoopNum)};
        if constexpr (isWeight == 1) {
            ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, targetWeightLocal, srcShape,
                                                                 false);
        } else {
            ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, this->ignoreSelect, srcShape,
                                                                 false);
        }
        Add(this->blockBuf2Local[0], this->blockBuf2Local[0], this->blockBuf3Local[0], 1);
    }
    if (this->nLoopTail != 0) {
        uint64_t nIdx = this->nLoopTimes * peerLoopNum;
        ComputeIgnoreAndTarget(this->nLoopTail, nIdx);
        uint32_t srcShape[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->nLoopTail)};
        if constexpr (isWeight == 1) {
            ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, targetWeightLocal, srcShape,
                                                                 false);
        } else {
            ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(this->blockBuf3Local, this->ignoreSelect, srcShape,
                                                                 false);
        }
        Add(this->blockBuf2Local[0], this->blockBuf2Local[0], this->blockBuf3Local[0], 1);
    }
    this->PipeV2M();
    this->CopyOutWsAndReduce();
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::Process()
{
    if constexpr (isWeight == 1) {
        this->CopyInWeight(0, this->colVal);
        weightLocal = this->inQueWeight_.template DeQue<float>();
    }
    if constexpr (reductionKey == REDUCTION_MEAN && (isWeight == 1 || (isIgnore == 1 && isWeight == 0))) {
        TargetWeightReduceSumFullLoad();
    }
    for (uint64_t n = 0; n < this->nLoopTimes; n++) {
        ProcessPerCore(n * peerLoopNum, peerLoopNum);
    }
    if (this->nLoopTail != 0) {
        ProcessPerCore(this->nLoopTimes * peerLoopNum, this->nLoopTail);
    }
    if constexpr (isWeight == 1) {
        this->inQueWeight_.template FreeTensor(weightLocal);
    }
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::ComputeIgnoreAndTarget(
    uint64_t targetNum, uint64_t peerLoopNIndex)
{
    LocalTensor<T2> targetLocal = this->inQueTarget_.template AllocTensor<T2>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(targetNum * sizeof(T2)), 0, 0, 0};
    DataCopyPadExtParams<T2> padParams{false, 0, 0, 0};
    DataCopyPad(targetLocal, this->targetGm[this->targetOffset + peerLoopNIndex], copyParams, padParams);
    this->inQueTarget_.template EnQue(targetLocal);
    targetLocal = this->inQueTarget_.template DeQue<T2>();
    __ubuf__ float* weightUbAddr;
    __ubuf__ float* targetWeightAddr;
    if constexpr (isWeight == 1) {
        weightUbAddr = (__ubuf__ float*)weightLocal.GetPhyAddr();
        targetWeightAddr = (__ubuf__ float*)targetWeightLocal.GetPhyAddr();
    }
    uint32_t batchNum = targetNum;
    uint32_t colNum = this->colVal;
    uint32_t vfLen = platform::GetVRegSize() / sizeof(T2);
    uint16_t repeatTimes = batchNum / vfLen;
    uint32_t tailNum = batchNum % vfLen;
    uint16_t tailLoopTimes = tailNum == 0 ? 0 : 1;
    auto targetUbAddr = (__ubuf__ T2*)targetLocal.GetPhyAddr();
    auto ignoreMaskUbAddr = (__ubuf__ float*)this->ignoreSelect.GetPhyAddr();
    int32_t ignore = this->ignoreIndex;
    __VEC_SCOPE__
    {
        Reg::MaskReg preg;
        Reg::MaskReg dstMask;
        Reg::RegTensor<T2> srcReg;
        Reg::RegTensor<float> dstReg1;
        Reg::RegTensor<float> srcReg0;
        Reg::RegTensor<float> dstReg0;
        Reg::RegTensor<float> dstReg;
        Reg::RegTensor<float> dstIgReg;
        Reg::RegTensor<float> gatherReg;
        Reg::RegTensor<int32_t> targetB32Reg;
        Reg::RegTensor<int32_t> targetB32TmpReg;
        Reg::MaskReg regAll;
        Reg::MaskReg regAllFp32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Duplicate(dstReg1, 1.0f, regAllFp32);
        Reg::Duplicate(dstReg0, 0.0f, regAllFp32);
        preg = Reg::UpdateMask<int32_t>(tailNum);
        for (uint16_t loop = 0; loop < repeatTimes; loop++) {
            Reg::AddrReg srcOffset = Reg::CreateAddrReg<T2>(loop, vfLen);
            Reg::AddrReg addrReg = Reg::CreateAddrReg<float>(loop, vfLen);
            if constexpr (sizeof(T2) == sizeof(int64_t)) {
                Reg::LoadAlign(srcReg, targetUbAddr, srcOffset);
                Reg::Cast<int32_t, T2, castS642S32>(targetB32TmpReg, srcReg, regAllFp32);
                Reg::Pack((RegTensor<uint32_t>&)targetB32Reg, (RegTensor<uint64_t>&)targetB32TmpReg);
                regAll = Reg::CreateMask<float, Reg::MaskPattern::H>();
            } else {
                Reg::LoadAlign(targetB32Reg, targetUbAddr, srcOffset);
                regAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            }
            if constexpr (isIgnore == 1) {
                Reg::Compares<int32_t, CMPMODE::EQ>(dstMask, targetB32Reg, ignore, regAllFp32);
                Reg::Select(dstIgReg, dstReg0, dstReg1, dstMask);
                Reg::StoreAlign(ignoreMaskUbAddr, dstIgReg, addrReg, regAll);
            }
            if constexpr (isWeight == 1) {
                Reg::Gather(gatherReg, weightUbAddr, (Reg::RegTensor<uint32_t>&)targetB32Reg, regAll);
                if constexpr (isIgnore == 1) {
                    Reg::Select(dstReg, dstReg0, dstReg1, dstMask);
                    Reg::Mul(dstReg, dstReg, gatherReg, regAll);
                    Reg::StoreAlign(targetWeightAddr, dstReg, addrReg, regAll);
                } else {
                    Reg::StoreAlign(targetWeightAddr, gatherReg, addrReg, regAll);
                }
            }
        }
        for (uint16_t t = 0; t < tailLoopTimes; t++) {
            if constexpr (sizeof(T2) == sizeof(int64_t)) {
                Reg::LoadAlign(srcReg, targetUbAddr + vfLen * repeatTimes);
                Reg::Cast<int32_t, T2, castS642S32>(targetB32TmpReg, srcReg, regAllFp32);
                Reg::Pack((RegTensor<uint32_t>&)targetB32Reg, (RegTensor<uint64_t>&)targetB32TmpReg);
            } else {
                Reg::LoadAlign(targetB32Reg, targetUbAddr + vfLen * repeatTimes);
            }
            if constexpr (isIgnore == 1) {
                Reg::Compares<int32_t, CMPMODE::EQ>(dstMask, targetB32Reg, ignore, preg);
                Reg::Select(dstIgReg, dstReg0, dstReg1, dstMask);
                Reg::StoreAlign(ignoreMaskUbAddr + vfLen * repeatTimes, dstIgReg, preg);
            }
            if constexpr (isWeight == 1) {
                Reg::Gather(gatherReg, weightUbAddr, (Reg::RegTensor<uint32_t>&)targetB32Reg, preg);
                if constexpr (isIgnore == 1) {
                    Reg::Select(dstReg, dstReg0, dstReg1, dstMask);
                    Reg::Mul(dstReg, dstReg, gatherReg, preg);
                    Reg::StoreAlign(targetWeightAddr + vfLen * repeatTimes, dstReg, preg);
                } else {
                    Reg::StoreAlign(targetWeightAddr + vfLen * repeatTimes, gatherReg, preg);
                }
            }
        }
    }
    this->inQueTarget_.template FreeTensor(targetLocal);
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::ProcessPerCore(
    uint64_t peerLoopNIdx, uint64_t nLoopNum)
{
    float reduceFactor = this->rowVal;
    if constexpr (isIgnore == 1 || isWeight == 1) {
        ComputeIgnoreAndTarget(nLoopNum, peerLoopNIdx);
        if constexpr (reductionKey == REDUCTION_MEAN) {
            reduceFactor = this->blockBuf2Local.GetValue(0);
            this->PipeS2V();
        }
    }
    if constexpr (labelS == 0) {
        GradReduction(reduceFactor, nLoopNum, peerLoopNIdx);
    } else {
        GradReductionSmooth(reduceFactor, nLoopNum, peerLoopNIdx);
    }
    ComputerEachBatch(peerLoopNIdx, nLoopNum);
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::GetGradLoss(
    float& gradLossValue, LocalTensor<T1>& gradLossLocal, uint64_t targetNum, uint64_t peerLoopNIdx)
{
    if constexpr (reductionKey == REDUCTION_NONE) {
        this->CopyInGradLoss(targetNum, peerLoopNIdx);
        gradLossLocal = this->inQueGradLoss_.template DeQue<T1>();
    } else {
        if constexpr (std::is_same<T1, bfloat16_t>::value) {
            gradLossValue = ToFloat(this->gradLossGm.GetValue(0));
        } else if constexpr (std::is_same<T1, half>::value) {
            gradLossValue = static_cast<float>(this->gradLossGm.GetValue(0));
        } else if constexpr (std::is_same<T1, float>::value) {
            gradLossValue = this->gradLossGm.GetValue(0);
        }
        this->PipeS2V();
    }
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::GradReduction(
    float factor, uint64_t targetNum, uint64_t peerLoopNIdx)
{
    float gradLossValue = 0.0f;
    LocalTensor<T1> gradLossLocal;
    GetGradLoss(gradLossValue, gradLossLocal, targetNum, peerLoopNIdx);

    uint32_t totalCLen = targetNum;
    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint32_t repeatTimes = CeilDivision(totalCLen, vfLen);

    float labelSm = this->labelSmFactor;
    auto gradLossAddr = (__ubuf__ T1*)gradLossLocal.GetPhyAddr();
    auto ignoreSelectAddr = (__ubuf__ float*)this->ignoreSelect.GetPhyAddr();
    auto targetCastAddr = (__ubuf__ float*)this->targetCast.GetPhyAddr();
    __ubuf__ float* targetWeightAddr;
    if constexpr (isWeight == 1) {
        targetWeightAddr = (__ubuf__ float*)targetWeightLocal.GetPhyAddr();
    }
    __VEC_SCOPE__
    {
        Reg::MaskReg pregUp;
        Reg::RegTensor<float> regGradLoss;
        Reg::RegTensor<float> regIgnoreSelect;
        Reg::RegTensor<float> regtargetWeight;
        Reg::RegTensor<float> regFactor;
        Reg::RegTensor<float> regSmoothGrad;
        for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
            pregUp = Reg::UpdateMask<float>(totalCLen);
            if constexpr (reductionKey == REDUCTION_NONE) {
                this->LoadRegTensor(regGradLoss, gradLossAddr, pregUp, (int32_t)vfLen);
                Reg::Muls(regGradLoss, regGradLoss, labelSm, pregUp);
            } else if constexpr (reductionKey == REDUCTION_MEAN) {
                Reg::Duplicate(regGradLoss, gradLossValue, pregUp);
                Reg::Muls(regGradLoss, regGradLoss, labelSm, pregUp);
                Reg::Duplicate(regFactor, factor, pregUp);
                Reg::Div<float, &divMode>(regGradLoss, regGradLoss, regFactor, pregUp);
            } else {
                Reg::Duplicate(regGradLoss, gradLossValue, pregUp);
                Reg::Muls(regGradLoss, regGradLoss, labelSm, pregUp);
            }
            if constexpr (isIgnore == 1) {
                Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(regIgnoreSelect, ignoreSelectAddr,
                                                                          (int32_t)vfLen);
                Reg::Mul(regGradLoss, regIgnoreSelect, regGradLoss, pregUp);
            }
            if constexpr (isWeight == 1) {
                Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(regtargetWeight, targetWeightAddr,
                                                                          (int32_t)vfLen);
                Reg::Mul(regGradLoss, regGradLoss, regtargetWeight, pregUp);
            }
            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(targetCastAddr, regGradLoss, (int32_t)vfLen,
                                                                       pregUp);
        }
    }
    if constexpr (reductionKey == REDUCTION_NONE) {
        this->inQueGradLoss_.template FreeTensor(gradLossLocal);
    }
}

template <typename T1, typename T2, uint64_t reductionKey, uint64_t isWeight, uint64_t labelS, uint64_t isIgnore>
__aicore__ inline void CELossGradFullLoad<T1, T2, reductionKey, isWeight, labelS, isIgnore>::GradReductionSmooth(
    float factor, uint64_t targetNum, uint64_t peerLoopNIdx)
{
    float gradLossValue = 0.0f;
    LocalTensor<T1> gradLossLocal;
    GetGradLoss(gradLossValue, gradLossLocal, targetNum, peerLoopNIdx);
    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint32_t totalCLen = targetNum;
    float smFactor = this->smoothFactor;
    float labelSm = this->labelSmFactor;
    uint32_t repeatTimes = CeilDivision(totalCLen, vfLen);

    __ubuf__ float* targetWeightAddr;
    if constexpr (isWeight == 1) {
        targetWeightAddr = (__ubuf__ float*)targetWeightLocal.GetPhyAddr();
    }
    auto gradLossAddr = (__ubuf__ T1*)gradLossLocal.GetPhyAddr();
    auto ignoreSelectAddr = (__ubuf__ float*)this->ignoreSelect.GetPhyAddr();
    auto targetCastAddr = (__ubuf__ float*)this->targetCast.GetPhyAddr();
    auto smoothAddr = (__ubuf__ float*)this->smoothLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::MaskReg pregUp;
        Reg::RegTensor<float> regGradLoss;
        Reg::RegTensor<float> regIgnoreSelect;
        Reg::RegTensor<float> regtargetWeight;
        Reg::RegTensor<float> regSmoothGrad;
        Reg::RegTensor<float> regFactor;
        Reg::RegTensor<float> regTmp;
        for (uint16_t loop = 0; loop < (uint16_t)repeatTimes; loop++) {
            pregUp = Reg::UpdateMask<float>(totalCLen);

            if constexpr (reductionKey == REDUCTION_NONE) {
                this->LoadRegTensor(regGradLoss, gradLossAddr, pregUp, (int32_t)vfLen);
                Reg::Muls(regTmp, regGradLoss, labelSm, pregUp);
                Reg::Muls(regSmoothGrad, regGradLoss, smFactor, pregUp);
            } else if constexpr (reductionKey == REDUCTION_MEAN) {
                Reg::Duplicate(regGradLoss, gradLossValue, pregUp);
                Reg::Duplicate(regFactor, factor, pregUp);
                Reg::Muls(regTmp, regGradLoss, labelSm, pregUp);
                Reg::Div<float, &divMode>(regTmp, regTmp, regFactor, pregUp);
                Reg::Div<float, &divMode>(regSmoothGrad, regGradLoss, regFactor, pregUp);
                Reg::Muls(regSmoothGrad, regSmoothGrad, smFactor, pregUp);
            } else {
                Reg::Duplicate(regGradLoss, gradLossValue, pregUp);
                Reg::Muls(regTmp, regGradLoss, labelSm, pregUp);
                Reg::Muls(regSmoothGrad, regGradLoss, smFactor, pregUp);
            }
            if constexpr (isIgnore == 1) {
                Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(regIgnoreSelect, ignoreSelectAddr,
                                                                          (int32_t)vfLen);
                Reg::Mul(regTmp, regIgnoreSelect, regTmp, pregUp);
                Reg::Mul(regSmoothGrad, regIgnoreSelect, regSmoothGrad, pregUp);
            }
            if constexpr (isWeight == 1) {
                Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(regtargetWeight, targetWeightAddr,
                                                                          (int32_t)vfLen);
                Reg::Mul(regTmp, regtargetWeight, regTmp, pregUp);
            }
            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(targetCastAddr, regTmp, (int32_t)vfLen, pregUp);
            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(smoothAddr, regSmoothGrad, (int32_t)vfLen,
                                                                       pregUp);
        }
    }
    if constexpr (reductionKey == REDUCTION_NONE) {
        this->inQueGradLoss_.template FreeTensor(gradLossLocal);
    }
}
} // namespace CrossEntropyLossGradRegbase
#endif // CROSS_ENTROPY_LOSS_GRAD_FULL_LOAD_H
