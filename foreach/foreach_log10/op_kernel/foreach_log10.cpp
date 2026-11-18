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
 * \file foreach_log10.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "../foreach_utils/foreach_implict_output.h"

using namespace AscendC;
using namespace Common::OpKernel;

// Log10 rejects aliased dst/src tensors (TensorReuseLoopCheck), so float/half must be computed
// out of place: read from the input queue and write into a separate output queue buffer.
// bufferNum is 1 for both queues to keep the total UB footprint equal to the previous in-place layout.
template <typename T>
class ForeachLog10OutOfPlace : public KernelForeachUnary<T, ForeachLog10OutOfPlace<T>, 1, INPUT_PARAMETER_COUNT, true> {
public:
    using Base = KernelForeachUnary<T, ForeachLog10OutOfPlace<T>, 1, INPUT_PARAMETER_COUNT, true>;

    __aicore__ inline ForeachLog10OutOfPlace() : Base(*this){};
    using Base::Init;
    using Base::Process;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float>& float32Tensor,
                                   bool isRemainder)
    {
        (void)index;
        (void)float32Tensor;
        (void)isRemainder;
        LocalTensor<T> dataLocal = Base::dataQueue.template DeQue<T>();
        LocalTensor<T> outLocal = Base::outQueue.template AllocTensor<T>();
        PipeBarrier<PIPE_V>();
        Log10<T>(outLocal, dataLocal, static_cast<uint32_t>(dataCount));
        PipeBarrier<PIPE_V>();
        Base::dataQueue.FreeTensor(dataLocal);
        Base::outQueue.template EnQue<T>(outLocal);
    }

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) { return false; }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

// bfloat16 path: cast to float32 first, compute Log10 out of place inside the float buffer
// (second region, enabled by paramsCount = 2) and cast the result back into the output queue.
class ForeachLog10Bf16OutOfPlace
    : public KernelForeachUnary<bfloat16_t, ForeachLog10Bf16OutOfPlace, 1, INPUT_PARAMETER_COUNT, true> {
public:
    using Base = KernelForeachUnary<bfloat16_t, ForeachLog10Bf16OutOfPlace, 1, INPUT_PARAMETER_COUNT, true>;

    __aicore__ inline ForeachLog10Bf16OutOfPlace() : Base(*this){};
    using Base::Init;
    using Base::Process;

private:
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float>& float32Tensor,
                                   bool isRemainder)
    {
        (void)index;
        (void)isRemainder;
        LocalTensor<bfloat16_t> dataLocal = Base::dataQueue.template DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> outLocal = Base::outQueue.template AllocTensor<bfloat16_t>();
        uint32_t count = static_cast<uint32_t>(dataCount);
        uint32_t chunk = Base::maxCastDataCount;
        for (uint32_t processed = 0; processed < count; processed += chunk) {
            uint32_t cur = (count - processed) > chunk ? chunk : (count - processed);
            PipeBarrier<PIPE_V>();
            Cast(float32Tensor, dataLocal[processed], RoundMode::CAST_NONE, cur);
            PipeBarrier<PIPE_V>();
            Log10<float>(float32Tensor[chunk], float32Tensor, cur);
            PipeBarrier<PIPE_V>();
            Cast(outLocal[processed], float32Tensor[chunk], RoundMode::CAST_RINT, cur);
            PipeBarrier<PIPE_V>();
        }
        Base::dataQueue.FreeTensor(dataLocal);
        Base::outQueue.template EnQue<bfloat16_t>(outLocal);
    }

    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder) { return false; }

    __aicore__ inline void BeforeProcess() {}

    __aicore__ inline void AfterProcess() {}

    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder) {}

    __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart) {}

    friend Base;
};

extern "C" __global__ __aicore__ void foreach_log10(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachLog10OutOfPlace<half> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachLog10OutOfPlace<float> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(4)) {
        ForeachLog10Bf16OutOfPlace op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
#endif
    }
}
