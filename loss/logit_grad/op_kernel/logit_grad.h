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
 * \file logit_grad.h
 * \brief logit_grad head file
 */
#ifndef LOGIT_GRAD_H
#define LOGIT_GRAD_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace LogitGrad {

using namespace AscendC;

constexpr int64_t MAX_UB_SIZE = 192 * 1024;
constexpr int64_t PP_ELEMENT_NUM = 8 * 1024;
constexpr int64_t ONE_REPEAT_ELE_NUM_FP32 = 64;
constexpr int64_t ALIGN = 16;

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
constexpr AscendC::MicroAPI::CastTrait G5_FP16_TO_FP32_CAST_TRAIT = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
#endif

template <typename T>
class LogitGradND {
public:
    TPipe pipe;
    __aicore__ inline LogitGradND(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace,
                                const LogitGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAndCast(int64_t inputOffset, int64_t dataCount);
    __aicore__ inline void ComputeStepOne(int64_t dataCount);
    __aicore__ inline void ComputeStepTwo(int64_t dataCount);
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
    __aicore__ inline void ComputeFusedFp16(int64_t dataCount);
    __aicore__ inline void ComputeFusedBf16(int64_t dataCount);
#endif
    __aicore__ inline void CastAndCopyOut(int64_t outputOffset, int64_t dataCount);

private:
    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<uint8_t> selMaskOne;
    LocalTensor<uint8_t> selMaskTwo;

    LocalTensor<T> x1Tmp;
    LocalTensor<T> x2Tmp;

    LocalTensor<T> x1Tensor;
    LocalTensor<T> x2Tensor;

    LocalTensor<float> x1TensorFp32;
    LocalTensor<float> x2TensorFp32;
    LocalTensor<float> tmpTensorFp32;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> dyGm;
    GlobalTensor<T> outputGm;

    int64_t elementNum;
    uint64_t needCoreNumber;
    int64_t blockIdx;
    float eps;

    // 准备compare参数
    float epslion;
    float selectValue;

    event_t eventId = EVENT_ID0;
    int64_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void LogitGradND<T>::Init(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace,
                                            const LogitGradTilingData* tilingData)
{
    inputGm.SetGlobalBuffer((__gm__ T*)x);
    dyGm.SetGlobalBuffer((__gm__ T*)dy);
    outputGm.SetGlobalBuffer((__gm__ T*)dx);

    eps = tilingData->eps;

    if (eps >= 0) {
        epslion = eps;
        selectValue = 0.0;
    } else {
        epslion = 0.0;
        selectValue = sqrt(static_cast<float>(-1.0));
    }
    elementNum = tilingData->elementNum;
    needCoreNumber = tilingData->needCoreNum;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void LogitGradND<T>::Process()
{
    if (blockIdx >= needCoreNumber) {
        return;
    }

    int64_t totalTimes = elementNum / PP_ELEMENT_NUM;
    int64_t remain = elementNum % PP_ELEMENT_NUM;
    if (remain > 0) {
        totalTimes++;
    }
    int64_t loopNum = totalTimes / needCoreNumber;
    int64_t loopRemain = totalTimes % needCoreNumber;

    if (loopRemain > 0 && blockIdx < loopRemain) {
        loopNum++;
    }
    int64_t eachCoreStartOffset = loopNum * blockIdx * PP_ELEMENT_NUM;
    if (loopRemain > 0) {
        if (blockIdx >= loopRemain) {
            eachCoreStartOffset += elementNum % (PP_ELEMENT_NUM * needCoreNumber);
        }
    }

    int64_t calNum = PP_ELEMENT_NUM;

    int64_t lastCoreNum = loopRemain == 0 ? needCoreNumber - 1 : loopRemain - 1;
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t localOffset = i * PP_ELEMENT_NUM;

        // 最后一轮的最后一个核处理余数
        if (remain > 0 && i == loopNum - 1 && blockIdx == lastCoreNum) {
            calNum = remain;
        }
        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        CopyInAndCast(eachCoreStartOffset + localOffset, calNum);

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
        if constexpr (std::is_same_v<T, half>) {
            ComputeFusedFp16(calNum);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            ComputeFusedBf16(calNum);
        } else {
            ComputeStepOne(calNum);
            ComputeStepTwo(calNum);
        }
#else
        ComputeStepOne(calNum);
        ComputeStepTwo(calNum);
#endif

        CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);

        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void LogitGradND<T>::CopyInAndCast(int64_t inputOffset, int64_t dataCount)
{
    x1Tensor = pingPongFlag ? tmpTensor[MAX_UB_SIZE / 2].ReinterpretCast<T>() : tmpTensor[0].ReinterpretCast<T>();
    x2Tensor = pingPongFlag ? tmpTensor[PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                              tmpTensor[PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int64_t elementByte = PP_ELEMENT_NUM * sizeof(T);
        x1Tmp = pingPongFlag ? tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                               tmpTensor[elementByte].ReinterpretCast<T>();
        x2Tmp = pingPongFlag ?
                    tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                    tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
        DataCopyPad(x1Tmp, inputGm[inputOffset], dataCopyParams, padParams);
        DataCopyPad(x2Tmp, dyGm[inputOffset], dataCopyParams, padParams);

    } else {
        DataCopyPad(x1Tensor, inputGm[inputOffset], dataCopyParams, padParams);
        DataCopyPad(x2Tensor, dyGm[inputOffset], dataCopyParams, padParams);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    x1TensorFp32 = x1Tensor.template ReinterpretCast<float>();
    x2TensorFp32 = x2Tensor.template ReinterpretCast<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
    if constexpr (std::is_same_v<T, half>) {
        return;
    }
#endif
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        Cast(x1TensorFp32, x1Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(x2TensorFp32, x2Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void LogitGradND<T>::ComputeStepOne(int64_t dataCount)
{
    int64_t elementByte = PP_ELEMENT_NUM * sizeof(float);
    selMaskOne = pingPongFlag ? tmpTensor[elementByte * 2 + MAX_UB_SIZE / 2] : tmpTensor[elementByte * 2];
    selMaskTwo = pingPongFlag ? tmpTensor[elementByte * 2 + elementByte / 2 + MAX_UB_SIZE / 2] :
                                tmpTensor[elementByte * 2 + elementByte / 2];
    tmpTensorFp32 = selMaskOne.template ReinterpretCast<float>();

    Muls(tmpTensorFp32, x1TensorFp32, float(-1.0), dataCount);
    PipeBarrier<PIPE_V>();
    Adds(tmpTensorFp32, tmpTensorFp32, float(1.0), dataCount);
    PipeBarrier<PIPE_V>();
    Mul(tmpTensorFp32, x1TensorFp32, tmpTensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
    Div(x2TensorFp32, x2TensorFp32, tmpTensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
template <typename T>
__aicore__ inline void LogitGradND<T>::ComputeFusedFp16(int64_t dataCount)
{
    float lo = epslion;
    float hi = static_cast<float>(1.0) - epslion;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<half> regXHalf;
        AscendC::MicroAPI::RegTensor<half> regDyHalf;
        AscendC::MicroAPI::RegTensor<float> regX;
        AscendC::MicroAPI::RegTensor<float> regDy;
        AscendC::MicroAPI::RegTensor<float> regTmp;
        AscendC::MicroAPI::RegTensor<float> regOut;
        AscendC::MicroAPI::RegTensor<float> regLo;
        AscendC::MicroAPI::RegTensor<float> regHi;
        AscendC::MicroAPI::RegTensor<float> regInvalid;
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg maskGE;
        AscendC::MicroAPI::MaskReg maskLE;
        AscendC::MicroAPI::MaskReg maskValid;
        constexpr uint32_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t count = static_cast<uint32_t>(dataCount);
        uint16_t vfLoopNum = static_cast<uint16_t>((count + vfLen - 1) / vfLen);
        __local_mem__ half* xAddr = (__local_mem__ half*)x1Tmp.GetPhyAddr();
        __local_mem__ half* dyAddr = (__local_mem__ half*)x2Tmp.GetPhyAddr();
        __local_mem__ float* outAddr = (__local_mem__ float*)x1TensorFp32.GetPhyAddr();

        AscendC::MicroAPI::Duplicate<float>(regLo, lo);
        AscendC::MicroAPI::Duplicate<float>(regHi, hi);
        AscendC::MicroAPI::Duplicate<float>(regInvalid, selectValue);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            uint32_t rem = count - static_cast<uint32_t>(i) * vfLen;
            preg0 = AscendC::MicroAPI::UpdateMask<float>(rem);
            AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regXHalf,
                                                                                            xAddr + i * vfLen);
            AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regDyHalf,
                                                                                            dyAddr + i * vfLen);
            AscendC::MicroAPI::Cast<float, half, G5_FP16_TO_FP32_CAST_TRAIT>(regX, regXHalf, preg0);
            AscendC::MicroAPI::Cast<float, half, G5_FP16_TO_FP32_CAST_TRAIT>(regDy, regDyHalf, preg0);
            AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                regTmp, regX, static_cast<float>(-1.0), preg0);
            AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                regTmp, regTmp, static_cast<float>(1.0), preg0);
            AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(regTmp, regX, regTmp, preg0);
            AscendC::MicroAPI::Div<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(regDy, regDy, regTmp, preg0);
            AscendC::MicroAPI::Compare<float, AscendC::CMPMODE::GE>(maskGE, regX, regLo, preg0);
            AscendC::MicroAPI::Compare<float, AscendC::CMPMODE::LE>(maskLE, regX, regHi, preg0);
            AscendC::MicroAPI::MaskAnd(maskValid, maskGE, maskLE, preg0);
            AscendC::MicroAPI::Select<float>(regOut, regDy, regInvalid, maskValid);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(outAddr + i * vfLen, regOut,
                                                                                            preg0);
        }
    }
    PipeBarrier<PIPE_V>();
}
#endif

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
template <typename T>
__aicore__ inline void LogitGradND<T>::ComputeFusedBf16(int64_t dataCount)
{
    float lo = epslion;
    float hi = static_cast<float>(1.0) - epslion;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> regX;
        AscendC::MicroAPI::RegTensor<float> regDy;
        AscendC::MicroAPI::RegTensor<float> regTmp;
        AscendC::MicroAPI::RegTensor<float> regOut;
        AscendC::MicroAPI::RegTensor<float> regLo;
        AscendC::MicroAPI::RegTensor<float> regHi;
        AscendC::MicroAPI::RegTensor<float> regInvalid;
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg maskGE;
        AscendC::MicroAPI::MaskReg maskLE;
        AscendC::MicroAPI::MaskReg maskValid;
        constexpr uint32_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t count = static_cast<uint32_t>(dataCount);
        uint16_t vfLoopNum = static_cast<uint16_t>((count + vfLen - 1) / vfLen);
        __local_mem__ float* xAddr = (__local_mem__ float*)x1TensorFp32.GetPhyAddr();
        __local_mem__ float* dyAddr = (__local_mem__ float*)x2TensorFp32.GetPhyAddr();

        AscendC::MicroAPI::Duplicate<float>(regLo, lo);
        AscendC::MicroAPI::Duplicate<float>(regHi, hi);
        AscendC::MicroAPI::Duplicate<float>(regInvalid, selectValue);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            uint32_t rem = count - static_cast<uint32_t>(i) * vfLen;
            preg0 = AscendC::MicroAPI::UpdateMask<float>(rem);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(regX, xAddr + i * vfLen);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(regDy, dyAddr + i * vfLen);
            AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                regTmp, regX, static_cast<float>(-1.0), preg0);
            AscendC::MicroAPI::Adds<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                regTmp, regTmp, static_cast<float>(1.0), preg0);
            AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(regTmp, regX, regTmp, preg0);
            AscendC::MicroAPI::Div<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(regDy, regDy, regTmp, preg0);
            AscendC::MicroAPI::Compare<float, AscendC::CMPMODE::GE>(maskGE, regX, regLo, preg0);
            AscendC::MicroAPI::Compare<float, AscendC::CMPMODE::LE>(maskLE, regX, regHi, preg0);
            AscendC::MicroAPI::MaskAnd(maskValid, maskGE, maskLE, preg0);
            AscendC::MicroAPI::Select<float>(regOut, regDy, regInvalid, maskValid);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(xAddr + i * vfLen, regOut,
                                                                                            preg0);
        }
    }
    PipeBarrier<PIPE_V>();
}
#endif

template <typename T>
__aicore__ inline void LogitGradND<T>::ComputeStepTwo(int64_t dataCount)
{
    float lo = epslion;
    float hi = static_cast<float>(1.0) - epslion;

    auto tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;
    CompareScalar(selMaskOne, x1TensorFp32, (float)lo, CMPMODE::GE, tmpDataCount);
    PipeBarrier<PIPE_V>();

    CompareScalar(selMaskTwo, x1TensorFp32, (float)hi, CMPMODE::LE, tmpDataCount);
    PipeBarrier<PIPE_V>();

    auto tmpMaskOne = selMaskOne.template ReinterpretCast<uint16_t>();
    auto tmpMaskTwo = selMaskTwo.template ReinterpretCast<uint16_t>();
    And(tmpMaskOne, tmpMaskOne, tmpMaskTwo, tmpDataCount / ALIGN);
    PipeBarrier<PIPE_V>();

    Select(x1TensorFp32, selMaskOne, x2TensorFp32, (float)selectValue, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LogitGradND<T>::CastAndCopyOut(int64_t outputOffset, int64_t dataCount)
{
    if (std::is_same_v<T, half>) {
        Cast(x1Tensor, x1TensorFp32, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    } else if (std::is_same_v<T, bfloat16_t>) {
        Cast(x1Tensor, x1TensorFp32, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[outputOffset], x1Tensor, dataCopyParams);
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

} // namespace LogitGrad
#endif
