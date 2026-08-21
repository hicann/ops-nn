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
 * \file foreach_implict_output_level_zero_api.h
 * \brief
 */

#ifndef FOREACH_IMPLICT_OUTPUT_LEVEL_ZERO_API
#define FOREACH_IMPLICT_OUTPUT_LEVEL_ZERO_API

#include "kernel_foreach_unary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr int16_t MAX_REPEATS = 255;
constexpr int16_t BYTES_PER_REPEAT = 256;
constexpr int8_t BYTES_PER_BLOCK = 32;
constexpr int8_t STRIDES_PER_REPEAT = 8;

template <typename T>
using ImplictOutputLevelZeroApiOp = void(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, uint64_t,
                                         const uint8_t, const BinaryRepeatParams&);

template <typename T, typename P, ImplictOutputLevelZeroApiOp<P>* op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(LocalTensor<T>& dataLocal, LocalTensor<float>& float32Tensor,
                                   uint32_t maxCastDataCount, int64_t dataCount, LocalTensor<P>& oneBlockData,
                                   uint64_t elementsPerRepeat)
    {
        uint32_t totalRepeats = 0;
        uint32_t divisible = 0;
        if (elementsPerRepeat == 0) {
            totalRepeats = -1;
            divisible = -1;
        } else {
            totalRepeats = dataCount / elementsPerRepeat;
            divisible = dataCount % elementsPerRepeat;
        }
        uint32_t outerRepeats = totalRepeats / MAX_REPEATS;

        uint32_t offset = 0;
        for (uint32_t i = 0; i < outerRepeats; i++) {
            op(dataLocal[offset], oneBlockData, dataLocal[offset], elementsPerRepeat, MAX_REPEATS,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (dataCount - (outerRepeats * MAX_REPEATS * elementsPerRepeat) > 0) {
            uint8_t curRepeat = totalRepeats - outerRepeats * MAX_REPEATS;
            if (curRepeat > 0) {
                op(dataLocal[offset], oneBlockData, dataLocal[offset], elementsPerRepeat, curRepeat,
                   {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
                offset += curRepeat * elementsPerRepeat;
            }

            if (divisible > 0) {
                uint32_t remain = dataCount - elementsPerRepeat * totalRepeats;
                op(dataLocal[offset], oneBlockData, dataLocal[offset], remain, 1,
                   {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            }
        }
    }
};

#if __CCE_AICORE__ >= 220
template <ImplictOutputLevelZeroApiOp<float>* op, uint8_t paramsCount>
class InnerComputer<bfloat16_t, float, op, paramsCount> {
public:
    __aicore__ inline void Compute(LocalTensor<bfloat16_t>& dataLocal, LocalTensor<float>& float32Tensor,
                                   uint32_t maxCastDataCount, int64_t dataCount, LocalTensor<float> oneBlockData,
                                   uint64_t elementsPerRepeat)
    {
        uint32_t castTimes = dataCount / maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount, oneBlockData,
                           elementsPerRepeat);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, castTimes, castTimesRemainder, oneBlockData,
                           elementsPerRepeat);
        }
    }

private:
    __aicore__ inline void ComputePerCast(LocalTensor<bfloat16_t>& dataLocal, LocalTensor<float>& float32Tensor,
                                          uint32_t maxCastDataCount, uint32_t index, int64_t dataCount,
                                          LocalTensor<float> oneBlockData, uint64_t elementsPerRepeat)
    {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);

        uint32_t totalRepeatCnt = dataCount / elementsPerRepeat;
        uint32_t totalRepeatCntRemainder = dataCount % elementsPerRepeat; // should calc
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;           // limit by L0 API, should calc
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;  // should calc

        uint32_t offset = 0;

        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], elementsPerRepeat, MAX_REPEATS,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (repeatBatchCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], elementsPerRepeat, repeatBatchCntRemainder,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * elementsPerRepeat;
        }

        if (totalRepeatCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], totalRepeatCntRemainder, 1,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        Cast(dataLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
};
#endif

#if __CCE_AICORE__ >= 220
template <ImplictOutputLevelZeroApiOp<float>* op, uint8_t paramsCount>
class InnerComputer<int16_t, float, op, paramsCount> {
public:
    __aicore__ inline void Compute(LocalTensor<int16_t>& dataLocal, LocalTensor<float>& float32Tensor,
                                   uint32_t maxCastDataCount, int64_t dataCount, LocalTensor<float> oneBlockData,
                                   uint64_t elementsPerRepeat)
    {
        uint32_t castTimes = dataCount / maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount, oneBlockData,
                           elementsPerRepeat);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, castTimes, castTimesRemainder, oneBlockData,
                           elementsPerRepeat);
        }
    }

private:
    __aicore__ inline void ComputePerCast(LocalTensor<int16_t>& dataLocal, LocalTensor<float>& float32Tensor,
                                          uint32_t maxCastDataCount, uint32_t index, int64_t dataCount,
                                          LocalTensor<float> oneBlockData, uint64_t elementsPerRepeat)
    {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);

        uint32_t totalRepeatCnt = dataCount / elementsPerRepeat;
        uint32_t totalRepeatCntRemainder = dataCount % elementsPerRepeat;
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;

        uint32_t offset = 0;

        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], elementsPerRepeat, MAX_REPEATS,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (repeatBatchCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], elementsPerRepeat, repeatBatchCntRemainder,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * elementsPerRepeat;
        }

        if (totalRepeatCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(float32Tensor[offset], oneBlockData, float32Tensor[offset], totalRepeatCntRemainder, 1,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        Cast(dataLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

template <typename T, ImplictOutputLevelZeroApiOp<half>* op, uint8_t paramsCount>
class InnerComputer8BitToHalf {
public:
    __aicore__ inline void Compute(LocalTensor<T>& dataLocal, LocalTensor<float>& float32Tensor,
                                   uint32_t maxCastDataCount, int64_t dataCount, LocalTensor<half> oneBlockData,
                                   uint64_t elementsPerRepeat)
    {
        LocalTensor<half> halfTensor = float32Tensor.template ReinterpretCast<half>();
        uint32_t castTimes = dataCount / maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(dataLocal, halfTensor, maxCastDataCount, i, maxCastDataCount, oneBlockData,
                           elementsPerRepeat);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(dataLocal, halfTensor, maxCastDataCount, castTimes, castTimesRemainder, oneBlockData,
                           elementsPerRepeat);
        }
    }

private:
    __aicore__ inline void ComputePerCast(LocalTensor<T>& dataLocal, LocalTensor<half>& halfTensor,
                                          uint32_t maxCastDataCount, uint32_t index, int64_t dataCount,
                                          LocalTensor<half> oneBlockData, uint64_t elementsPerRepeat)
    {
        PipeBarrier<PIPE_V>();
        Cast(halfTensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);

        uint32_t totalRepeatCnt = dataCount / elementsPerRepeat;
        uint32_t totalRepeatCntRemainder = dataCount % elementsPerRepeat;
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;

        uint32_t offset = 0;

        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PipeBarrier<PIPE_V>();
            op(halfTensor[offset], oneBlockData, halfTensor[offset], elementsPerRepeat, MAX_REPEATS,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (repeatBatchCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(halfTensor[offset], oneBlockData, halfTensor[offset], elementsPerRepeat, repeatBatchCntRemainder,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * elementsPerRepeat;
        }

        if (totalRepeatCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(halfTensor[offset], oneBlockData, halfTensor[offset], totalRepeatCntRemainder, 1,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        Cast(dataLocal[index * maxCastDataCount], halfTensor, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

template <ImplictOutputLevelZeroApiOp<half>* op, uint8_t paramsCount>
class InnerComputer<int8_t, half, op, paramsCount> : public InnerComputer8BitToHalf<int8_t, op, paramsCount> {};

template <ImplictOutputLevelZeroApiOp<half>* op, uint8_t paramsCount>
class InnerComputer<uint8_t, half, op, paramsCount> : public InnerComputer8BitToHalf<uint8_t, op, paramsCount> {};
#endif

template <typename T, typename P, ImplictOutputLevelZeroApiOp<P>* op, int32_t bufferNum = BUFFER_NUM,
          uint8_t paramsCount = INPUT_PARAMETER_COUNT>
class ForeachNegIntWrapLevelZeroApi
    : public KernelForeachUnary<T, ForeachNegIntWrapLevelZeroApi<T, P, op, bufferNum, paramsCount>, bufferNum,
                                paramsCount, false> {
public:
    using Base = KernelForeachUnary<T, ForeachNegIntWrapLevelZeroApi<T, P, op, bufferNum, paramsCount>, bufferNum,
                                    paramsCount, false>;

    __aicore__ inline ForeachNegIntWrapLevelZeroApi() : Base(*this){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData,
                                P duplicatedNum);
    using Base::Process;

protected:
    LocalTensor<P> scalarOneBlockUB;
    TQue<QuePosition::VECIN, 1> scalarOneBlockQueue;
    TBuf<QuePosition::VECCALC> maskBuf;
    uint64_t elementsPerRepeat = BYTES_PER_REPEAT / sizeof(P);

private:
    __aicore__ inline P GetSelectValue()
    {
        if constexpr (std::is_same_v<T, int16_t>) {
            return static_cast<P>(-32768);
        } else {
            return static_cast<P>(-128);
        }
    }

    __aicore__ inline P GetCompareValue()
    {
        if constexpr (std::is_same_v<T, uint8_t>) {
            return static_cast<P>(0);
        } else {
            return GetSelectValue();
        }
    }

    __aicore__ inline uint32_t AlignCompareCount(int64_t dataCount)
    {
        return (static_cast<uint32_t>(dataCount) + elementsPerRepeat - 1) / elementsPerRepeat * elementsPerRepeat;
    }

    __aicore__ inline void ApplyOp(LocalTensor<P>& computeTensor, int64_t dataCount)
    {
        uint32_t totalRepeatCnt = dataCount / elementsPerRepeat;
        uint32_t totalRepeatCntRemainder = dataCount % elementsPerRepeat;
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;
        uint32_t offset = 0;

        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            PipeBarrier<PIPE_V>();
            op(computeTensor[offset], scalarOneBlockUB, computeTensor[offset], elementsPerRepeat, MAX_REPEATS,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * elementsPerRepeat;
        }

        if (repeatBatchCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(computeTensor[offset], scalarOneBlockUB, computeTensor[offset], elementsPerRepeat,
               repeatBatchCntRemainder, {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * elementsPerRepeat;
        }

        if (totalRepeatCntRemainder > 0) {
            PipeBarrier<PIPE_V>();
            op(computeTensor[offset], scalarOneBlockUB, computeTensor[offset], totalRepeatCntRemainder, 1,
               {1, 0, 1, STRIDES_PER_REPEAT, 0, STRIDES_PER_REPEAT});
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputePerCast(LocalTensor<T>& dataLocal, LocalTensor<P>& computeTensor,
                                          LocalTensor<uint8_t>& maskTensor, uint32_t maxCastDataCount, uint32_t index,
                                          int64_t dataCount)
    {
        PipeBarrier<PIPE_V>();
        Cast(computeTensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();

        CompareScalar(maskTensor, computeTensor, GetCompareValue(), CMPMODE::NE, AlignCompareCount(dataCount));
        PipeBarrier<PIPE_V>();

        ApplyOp(computeTensor, dataCount);

        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<T, uint8_t>) {
            Select(computeTensor, maskTensor, computeTensor, static_cast<P>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   dataCount);
        } else {
            Select(computeTensor, maskTensor, computeTensor, GetSelectValue(), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   dataCount);
        }
        PipeBarrier<PIPE_V>();

        Cast(dataLocal[index * maxCastDataCount], computeTensor, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float>& float32Tensor,
                                   bool isRemainder)
    {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        LocalTensor<P> computeTensor = float32Tensor.template ReinterpretCast<P>();
        LocalTensor<uint8_t> maskTensor = maskBuf.Get<uint8_t>();
        uint32_t castTimes = dataCount / Base::maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % Base::maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(dataLocal, computeTensor, maskTensor, Base::maxCastDataCount, i, Base::maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(dataLocal, computeTensor, maskTensor, Base::maxCastDataCount, castTimes, castTimesRemainder);
        }

        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(Base::outTensorsGM[1ULL * index * Base::maxDataCount], dataLocal, copyParams);
        } else {
            DataCopy(Base::outTensorsGM[1ULL * index * Base::maxDataCount], dataLocal, dataCount);
        }
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        Base::dataQueue.FreeTensor(dataLocal);
    }

    __aicore__ inline void BeforeProcess() { scalarOneBlockQueue.DeQue<P>(); }

    __aicore__ inline void AfterProcess() { scalarOneBlockQueue.FreeTensor(scalarOneBlockUB); }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) { return false; }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

template <typename T, typename P, ImplictOutputLevelZeroApiOp<P>* op, int32_t bufferNum, uint8_t paramsCount>
__aicore__ inline void ForeachNegIntWrapLevelZeroApi<T, P, op, bufferNum, paramsCount>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData, P duplicatedNum)
{
    Base::Init(x, y, workspace, tilingData);
    Base::pipe.InitBuffer(scalarOneBlockQueue, 1, BYTES_PER_BLOCK);
    scalarOneBlockUB = scalarOneBlockQueue.AllocTensor<P>();
    Duplicate(scalarOneBlockUB, duplicatedNum, BYTES_PER_BLOCK / sizeof(P));
    scalarOneBlockQueue.EnQue(scalarOneBlockUB);

    uint32_t maskBufSize = ((Base::maxCastDataCount + BYTES_PER_REPEAT - 1) / BYTES_PER_REPEAT) * BYTES_PER_BLOCK;
    if (maskBufSize < BYTES_PER_BLOCK) {
        maskBufSize = BYTES_PER_BLOCK;
    }
    Base::pipe.InitBuffer(maskBuf, maskBufSize);
}

template <typename T, typename P, ImplictOutputLevelZeroApiOp<P>* op, int32_t bufferNum = BUFFER_NUM,
          uint8_t paramsCount = INPUT_PARAMETER_COUNT>
class ForeachImplictOutputLevelZeroApi
    : public KernelForeachUnary<T, ForeachImplictOutputLevelZeroApi<T, P, op, bufferNum, paramsCount>, bufferNum,
                                paramsCount, false> {
public:
    using Base = KernelForeachUnary<T, ForeachImplictOutputLevelZeroApi<T, P, op, bufferNum, paramsCount>, bufferNum,
                                    paramsCount, false>;
    using Operator = ImplictOutputLevelZeroApiOp<P>;

    __aicore__ inline ForeachImplictOutputLevelZeroApi() : Base(*this){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData,
                                P duplicatedNum);
    using Base::Process;

protected:
    LocalTensor<P> scalarOneBlockUB;
    // for repeat in one block
    TQue<QuePosition::VECIN, 1> scalarOneBlockQueue;
    uint64_t elementsPerRepeat = BYTES_PER_REPEAT / sizeof(P);

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float>& float32Tensor,
                                   bool isRemainder)
    {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();

        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(dataLocal, float32Tensor, Base::maxCastDataCount, dataCount, scalarOneBlockUB,
                         elementsPerRepeat);

        // Transport can be performed only after the Muls is complete.
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(Base::outTensorsGM[1ULL * index * Base::maxDataCount], dataLocal, copyParams);
        } else {
            DataCopy(Base::outTensorsGM[1ULL * index * Base::maxDataCount], dataLocal, dataCount);
        }
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        Base::dataQueue.FreeTensor(dataLocal);
    }

    __aicore__ inline void BeforeProcess() { scalarOneBlockQueue.DeQue<P>(); }

    __aicore__ inline void AfterProcess() { scalarOneBlockQueue.FreeTensor(scalarOneBlockUB); }

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) { return false; }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

template <typename T, typename P, ImplictOutputLevelZeroApiOp<P>* op, int32_t bufferNum, uint8_t paramsCount>
__aicore__ inline void ForeachImplictOutputLevelZeroApi<T, P, op, bufferNum, paramsCount>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData, P duplicatedNum)
{
    Base::Init(x, y, workspace, tilingData);
    Base::pipe.InitBuffer(scalarOneBlockQueue, 1, BYTES_PER_BLOCK);
    scalarOneBlockUB = scalarOneBlockQueue.AllocTensor<P>();
    Duplicate(scalarOneBlockUB, duplicatedNum, BYTES_PER_BLOCK / sizeof(P));
    scalarOneBlockQueue.EnQue(scalarOneBlockUB);
}

} // namespace OpKernel
} // namespace Common

#endif // KERNEL_FOREACH_UNARY_H
