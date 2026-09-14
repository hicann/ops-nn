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
 * \file logit.h
 * \brief
 */

#ifndef LOGIT_H
#define LOGIT_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace Logit {

using namespace AscendC;

constexpr int64_t MAX_UB_SIZE = 192 * 1024;
constexpr int64_t PP_ELEMENT_NUM = 8 * 1024;
constexpr int64_t ONE_REPEAT_ELE_NUM_FP32 = 64;

template <typename T>
class LogitND {
public:
    TPipe pipe;
    __aicore__ inline LogitND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const LogitTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAndCast(int64_t inputOffset, int64_t dataCount);
    __aicore__ inline void ComputeStepOne(int64_t dataCount);
    __aicore__ inline void ComputeStepTwo(int64_t dataCount);
    __aicore__ inline void CastAndCopyOut(int64_t outputOffset, int64_t dataCount);

private:
    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<uint8_t> selMaskOne;
    LocalTensor<uint8_t> selMaskTwo;
    LocalTensor<uint8_t> selMaskThree;

    LocalTensor<T> x1Tmp;
    LocalTensor<T> x2Tmp;

    LocalTensor<T> x1Tensor;
    LocalTensor<T> x2Tensor;

    LocalTensor<float> x1TensorFp32;
    LocalTensor<float> x2TensorFp32;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;

    int64_t elementNum;
    uint64_t needCoreNumber;
    int64_t blockIdx;
    float eps;

    event_t eventId = EVENT_ID0;
    int64_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void LogitND<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                        const LogitTilingData* tilingData)
{
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    outputGm.SetGlobalBuffer((__gm__ T*)output);

    eps = tilingData->eps;
    elementNum = tilingData->elementNum;
    needCoreNumber = tilingData->needCoreNum;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void LogitND<T>::Process()
{
    if (blockIdx >= needCoreNumber) {
        return;
    }

    int64_t perCore = elementNum / needCoreNumber;
    int64_t coreRemain = elementNum % needCoreNumber;
    int64_t myLen = perCore + (blockIdx < coreRemain ? 1 : 0);
    int64_t myStart = blockIdx * perCore + (blockIdx < coreRemain ? blockIdx : coreRemain);

    int64_t myTimes = (myLen + PP_ELEMENT_NUM - 1) / PP_ELEMENT_NUM;
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t i = 0; i < myTimes; i++) {
        int64_t localOffset = i * PP_ELEMENT_NUM;
        int64_t calNum = (myLen - localOffset < PP_ELEMENT_NUM) ? (myLen - localOffset) : PP_ELEMENT_NUM;

        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        CopyInAndCast(myStart + localOffset, calNum);

        if (eps >= 0) {
            ComputeStepOne(calNum);
        }
        ComputeStepTwo(calNum);
        CastAndCopyOut(myStart + localOffset, calNum);
        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void LogitND<T>::CopyInAndCast(int64_t inputOffset, int64_t dataCount)
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

    } else {
        DataCopyPad(x1Tensor, inputGm[inputOffset], dataCopyParams, padParams);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    x1TensorFp32 = x1Tensor.template ReinterpretCast<float>();
    x2TensorFp32 = x2Tensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        Cast(x1TensorFp32, x1Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void LogitND<T>::ComputeStepOne(int64_t dataCount)
{
    // 对x用eps进行处理
    int64_t elementByte = PP_ELEMENT_NUM * sizeof(float);
    float nanValue = sqrt(static_cast<float>(-1.0));
    selMaskOne = pingPongFlag ? tmpTensor[elementByte * 2 + MAX_UB_SIZE / 2] : tmpTensor[elementByte * 2];
    selMaskTwo = pingPongFlag ? tmpTensor[elementByte * 2 + elementByte / 2 + MAX_UB_SIZE / 2] :
                                tmpTensor[elementByte * 2 + elementByte / 2];
    selMaskThree = x2TensorFp32.template ReinterpretCast<uint8_t>();

    float hi = static_cast<float>(1.0) - eps;
    auto tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;
    Compare(selMaskThree, x1TensorFp32, x1TensorFp32, CMPMODE::EQ, tmpDataCount);
    PipeBarrier<PIPE_V>();

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
    if (eps > static_cast<float>(0.5)) {
        CompareScalar(selMaskOne, x1TensorFp32, (float)eps, CMPMODE::GE, tmpDataCount);
        PipeBarrier<PIPE_V>();

        CompareScalar(selMaskTwo, x1TensorFp32, (float)hi, CMPMODE::LE, tmpDataCount);
        PipeBarrier<PIPE_V>();

        Select(x1TensorFp32, selMaskTwo, x1TensorFp32, (float)hi, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
        PipeBarrier<PIPE_V>();

        Select(x1TensorFp32, selMaskOne, x1TensorFp32, (float)eps, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
        PipeBarrier<PIPE_V>();
    } else {
        Mins(x1TensorFp32, x1TensorFp32, static_cast<float>(hi), tmpDataCount);
        PipeBarrier<PIPE_V>();

        Maxs(x1TensorFp32, x1TensorFp32, static_cast<float>(eps), tmpDataCount);
        PipeBarrier<PIPE_V>();
    }
#else
    CompareScalar(selMaskOne, x1TensorFp32, (float)eps, CMPMODE::GE, tmpDataCount);
    PipeBarrier<PIPE_V>();

    CompareScalar(selMaskTwo, x1TensorFp32, (float)hi, CMPMODE::LE, tmpDataCount);
    PipeBarrier<PIPE_V>();

    Select(x1TensorFp32, selMaskTwo, x1TensorFp32, (float)hi, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();

    Select(x1TensorFp32, selMaskOne, x1TensorFp32, (float)eps, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();
#endif

    Select(x1TensorFp32, selMaskThree, x1TensorFp32, (float)nanValue, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LogitND<T>::ComputeStepTwo(int64_t dataCount)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> regX;
        AscendC::Reg::RegTensor<float> regX2;
        AscendC::Reg::RegTensor<float> regTmp;
        AscendC::Reg::MaskReg preg0;
        constexpr uint32_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t count = static_cast<uint32_t>(dataCount);
        uint16_t vfLoopNum = static_cast<uint16_t>((count + vfLen - 1) / vfLen);
        __local_mem__ float* x1Addr = (__local_mem__ float*)x1TensorFp32.GetPhyAddr();
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            uint32_t rem = count - static_cast<uint32_t>(i) * vfLen;
            preg0 = AscendC::Reg::UpdateMask<float>(rem);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(regX, x1Addr + i * vfLen);
            // ln(x) - ln(1-x) = ln(2x) - ln(2-2x)：整体乘 2 精确无舍入，补偿在相减中相消。
            // 次正规 x 经乘 2 后必不触及库 FTZ_FALSE 缩放判定的上界（最大次正规数的
            // 一半不可表示），从而完全绕开 FTZ 冲刷；对可表示的 x < 1，2-2x 恒为正规数
            AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(regX2, regX, static_cast<float>(2.0),
                                                                                   preg0);
            AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(regTmp, regX,
                                                                                   static_cast<float>(-2.0), preg0);
            AscendC::Reg::Adds<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(regTmp, regTmp,
                                                                                   static_cast<float>(2.0), preg0);
            static constexpr AscendC::Reg::LogSpecificMode logNoFtz = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                                       AscendC::LogAlgo::PRECISION_1ULP_FTZ_FALSE};
            AscendC::Reg::Log<float, &logNoFtz>(regTmp, regTmp, preg0);
            AscendC::Reg::Log<float, &logNoFtz>(regX, regX2, preg0);
            AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(regX, regX, regTmp, preg0);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(x1Addr + i * vfLen, regX, preg0);
        }
    }
    PipeBarrier<PIPE_V>();
#else
    Muls(x2TensorFp32, x1TensorFp32, float(-1.0), dataCount);
    PipeBarrier<PIPE_V>();
    Adds(x2TensorFp32, x2TensorFp32, float(1.0), dataCount);
    PipeBarrier<PIPE_V>();
    Div(x1TensorFp32, x1TensorFp32, x2TensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
    Ln(x1TensorFp32, x1TensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
#endif
}

template <typename T>
__aicore__ inline void LogitND<T>::CastAndCopyOut(int64_t outputOffset, int64_t dataCount)
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

} // namespace Logit

// 增加int16 int8 uint8的logit实现
namespace NsLogit {

using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t SINGLE_BUFFER_NUM = 1;

template <typename T>
class KernelLogit {
public:
    __aicore__ inline KernelLogit(){};

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const LogitTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue0, tmpQueue2, tmpQueueMask1, tmpQueueMask2, tmpQueueMask3;
    AscendC::LocalTensor<uint8_t> mask1Local;
    AscendC::LocalTensor<uint8_t> mask2Local;
    AscendC::LocalTensor<uint8_t> mask3Local;
    AscendC::LocalTensor<float> tmp0Local;
    AscendC::LocalTensor<half> tmp2Local;
    AscendC::LocalTensor<T> xLocal;
    AscendC::LocalTensor<float> outLocal;

    AscendC::GlobalTensor<T> inputGm;
    AscendC::GlobalTensor<float> outGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;

    float eps = 0.0f;
    float hi = 0.0f;
    float nanValue = sqrt(static_cast<float>(-1.0));
};

template <typename T>
__aicore__ inline void KernelLogit<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                            const LogitTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tilingData->bigCoreDataNum * coreId;
    this->tileDataNum = tilingData->tileDataNum;
    // default open double buffer
    uint64_t BUFFER_NUM = DOUBLE_BUFFER_NUM;
    if (tilingData->bufferOpen == 0) {
        BUFFER_NUM = SINGLE_BUFFER_NUM;
    }
    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (coreId - tilingData->tailBlockNum);
    }
    inputGm.SetGlobalBuffer((__gm__ T*)input + globalBufferIndex, this->coreDataNum);
    outGm.SetGlobalBuffer((__gm__ float*)output + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(float));

    pipe.InitBuffer(tmpQueueMask1, this->tileDataNum * sizeof(uint8_t));
    pipe.InitBuffer(tmpQueueMask2, this->tileDataNum * sizeof(uint8_t));
    pipe.InitBuffer(tmpQueueMask3, this->tileDataNum * sizeof(uint8_t));

    pipe.InitBuffer(tmpQueue0, this->tileDataNum * sizeof(float));

    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        pipe.InitBuffer(tmpQueue2, this->tileDataNum * sizeof(half));
    }
    this->eps = tilingData->eps;
    this->hi = static_cast<float>(1.0) - (this->eps);
}

template <typename T>
__aicore__ inline void KernelLogit<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, inputGm[progress * this->tileDataNum], this->processDataNum);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void KernelLogit<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<float> outLocal = outQueueY.DeQue<float>();
    AscendC::DataCopy(outGm[progress * this->tileDataNum], outLocal, this->processDataNum);
    outQueueY.FreeTensor(outLocal);
}
template <typename T>
__aicore__ inline void KernelLogit<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<uint8_t> mask1Local = tmpQueueMask1.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint8_t> mask2Local = tmpQueueMask2.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint8_t> mask3Local = tmpQueueMask3.AllocTensor<uint8_t>();
    AscendC::LocalTensor<float> tmp0Local = tmpQueue0.AllocTensor<float>();

    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        tmp2Local = tmpQueue2.AllocTensor<half>();
    }

    if constexpr (std::is_same_v<T, int16_t>) {
        AscendC::Cast(tmp0Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        AscendC::Cast(tmp2Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(tmp0Local, tmp2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        AscendC::Cast(tmp2Local, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        PipeBarrier<PIPE_V>();
        AscendC::Cast(tmp0Local, tmp2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
    }

    AscendC::Compare(mask1Local, tmp0Local, tmp0Local, CMPMODE::EQ, this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Duplicate(outLocal, static_cast<float>(eps), this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Compare(mask2Local, tmp0Local, outLocal, CMPMODE::GE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Duplicate(outLocal, static_cast<float>(hi), this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Compare(mask3Local, tmp0Local, outLocal, CMPMODE::LE, this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Select(tmp0Local, mask3Local, tmp0Local, (float)hi, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Select(tmp0Local, mask2Local, tmp0Local, (float)eps, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Select(tmp0Local, mask1Local, tmp0Local, (float)nanValue, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Muls(outLocal, tmp0Local, float(-1.0), this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Adds(outLocal, outLocal, float(1.0), this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Div(tmp0Local, tmp0Local, outLocal, this->processDataNum);
    PipeBarrier<PIPE_V>();
    AscendC::Ln(outLocal, tmp0Local, this->processDataNum);
    PipeBarrier<PIPE_V>();

    outQueueY.EnQue<float>(outLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void KernelLogit<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsLogit

#endif
