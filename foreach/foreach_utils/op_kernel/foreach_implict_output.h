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
 * \file foreach_implict_output.h
 * \brief
 */

#ifndef FOREACH_IMPLICT_OUTPUT
#define FOREACH_IMPLICT_OUTPUT

#include "kernel_foreach_unary.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

template <typename T>
using ImplictOutputOp = void(const LocalTensor<T>&, const LocalTensor<T>&, const int32_t&);

template <typename T, typename P, ImplictOutputOp<P>* op, uint8_t paramsCount>
class InnerComputer {
public:
    __aicore__ inline void Compute(LocalTensor<T>& dataLocal, LocalTensor<float>& float32Tensor,
                                   uint32_t maxCastDataCount, int64_t dataCount)
    {
        op(dataLocal, dataLocal, dataCount);
    }
};

#if __CCE_AICORE__ >= 220
template <typename T, typename P, ImplictOutputOp<P>* op, uint8_t paramsCount>
class InnerComputerCastBase {
public:
    __aicore__ inline void Compute(LocalTensor<T>& dataLocal, LocalTensor<float>& float32Tensor,
                                   uint32_t maxCastDataCount, int64_t dataCount)
    {
        uint32_t castTimes = dataCount / maxCastDataCount;
        uint32_t castTimesRemainder = dataCount % maxCastDataCount;

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, i, maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(dataLocal, float32Tensor, maxCastDataCount, castTimes, castTimesRemainder);
        }
    }

private:
    __aicore__ inline void ComputePerCast(LocalTensor<T>& dataLocal, LocalTensor<float>& float32Tensor,
                                          uint32_t maxCastDataCount, uint32_t index, int64_t dataCount)
    {
        if constexpr (std::is_same_v<P, float>) {
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
            op(float32Tensor[offset], float32Tensor, dataCount);
            PipeBarrier<PIPE_V>();
            Cast(dataLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
            PipeBarrier<PIPE_V>();
        } else if constexpr (std::is_same_v<P, half>) {
            PipeBarrier<PIPE_V>();
            LocalTensor<half> halfTensor = float32Tensor.template ReinterpretCast<half>();
            Cast(halfTensor, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
            PipeBarrier<PIPE_V>();
            uint32_t offset = (paramsCount == 1) ? 0 : maxCastDataCount;
            op(halfTensor[offset], halfTensor, dataCount);
            PipeBarrier<PIPE_V>();
            Cast(dataLocal[index * maxCastDataCount], halfTensor[offset], RoundMode::CAST_RINT, dataCount);
            PipeBarrier<PIPE_V>();
        }
    }
};

template <ImplictOutputOp<float>* op, uint8_t paramsCount>
class InnerComputer<bfloat16_t, float, op, paramsCount>
    : public InnerComputerCastBase<bfloat16_t, float, op, paramsCount> {};

template <ImplictOutputOp<float>* op, uint8_t paramsCount>
class InnerComputer<int16_t, float, op, paramsCount> : public InnerComputerCastBase<int16_t, float, op, paramsCount> {};

template <ImplictOutputOp<half>* op, uint8_t paramsCount>
class InnerComputer<int8_t, half, op, paramsCount> : public InnerComputerCastBase<int8_t, half, op, paramsCount> {};

template <ImplictOutputOp<half>* op, uint8_t paramsCount>
class InnerComputer<uint8_t, half, op, paramsCount> : public InnerComputerCastBase<uint8_t, half, op, paramsCount> {};

#endif

template <typename T, typename P, ImplictOutputOp<P>* op, int32_t bufferNum = BUFFER_NUM,
          uint8_t paramsCount = INPUT_PARAMETER_COUNT>
class ForeachImplictOutput : public KernelForeachUnary<T, ForeachImplictOutput<T, P, op, bufferNum, paramsCount>,
                                                       bufferNum, paramsCount, false> {
public:
    using Base = KernelForeachUnary<T, ForeachImplictOutput<T, P, op, bufferNum, paramsCount>, bufferNum, paramsCount,
                                    false>;
    using Operator = ImplictOutputOp<P>;

    __aicore__ inline ForeachImplictOutput() : Base(*this){};
    using Base::Init;
    using Base::Process;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float>& float32Tensor,
                                   bool isRemainder)
    {
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        InnerComputer<T, P, op, paramsCount> computer;
        computer.Compute(dataLocal, float32Tensor, Base::maxCastDataCount, dataCount);

        // Transport can be performed only after the Muls is complete.
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        if (isRemainder) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(Base::outTensorsGM[index * Base::maxDataCount], dataLocal, copyParams);
        } else {
            DataCopy(Base::outTensorsGM[index * Base::maxDataCount], dataLocal, dataCount);
        }
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        Base::dataQueue.FreeTensor(dataLocal);
    }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) { return false; }

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

} // namespace OpKernel
} // namespace Common

#endif // KERNEL_FOREACH_UNARY_H
