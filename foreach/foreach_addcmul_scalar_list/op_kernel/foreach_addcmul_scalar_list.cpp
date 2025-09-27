/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_addcmul_scalar_list.cpp
 * \brief
 */

#include "kernel_operator.h"

#define DTYPE_SCALAR DTYPE_SCALARS

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_one_scalar_list_quaternary_implict_output.h"

using namespace AscendC;
using namespace Common::OpKernel;

template <typename T>
__aicore__ void AddcMulScalarListAdapterForFloat(
    const LocalTensor<T>& tensor1Local, const LocalTensor<T>& tensor2Local, const LocalTensor<T>& tensor3Local,
    const LocalTensor<T>& float32Tensor, const T scalarValue, const uint32_t maxCastDataCount, const int64_t dataCount)
{
    Mul(tensor2Local, tensor2Local, tensor3Local, dataCount);
    PipeBarrier<PIPE_V>();
    Axpy<T, T>(tensor1Local, tensor2Local, scalarValue, dataCount);
}

template <typename T>
__aicore__ void AddcMulScalarListAdapterForInt(
    const LocalTensor<T>& tensor1Local, const LocalTensor<T>& tensor2Local, const LocalTensor<T>& tensor3Local,
    const LocalTensor<float>& float32Tensor, const float scalarValue, const uint32_t maxCastDataCount,
    const int64_t dataCount)
{
    Mul(tensor2Local, tensor2Local, tensor3Local, dataCount);
    PipeBarrier<PIPE_V>();
    Muls(tensor2Local, tensor2Local, (int32_t)scalarValue, dataCount);
    PipeBarrier<PIPE_V>();
    Add(tensor1Local, tensor1Local, tensor2Local, dataCount);
}

template <typename T>
__aicore__ void ComputerPerCastForAddcMulScalarList(
    const LocalTensor<T>& tensor1Local, const LocalTensor<T>& tensor2Local, const LocalTensor<T>& tensor3Local,
    const LocalTensor<float>& float32Tensor, const float scalarValue, const uint32_t maxCastDataCount,
    const uint32_t index, const int64_t dataCount)
{
    Cast(float32Tensor, tensor2Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(float32Tensor[maxCastDataCount], tensor3Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(float32Tensor[maxCastDataCount * 2], tensor1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    // input + scalar_tensor * (tensor1 / tensor2)
    PipeBarrier<PIPE_V>();
    Mul(float32Tensor, float32Tensor, float32Tensor[maxCastDataCount], dataCount);
    PipeBarrier<PIPE_V>();
    Axpy<float, float>(float32Tensor[maxCastDataCount * 2], float32Tensor, scalarValue, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(tensor1Local[index * maxCastDataCount], float32Tensor[maxCastDataCount * 2], RoundMode::CAST_RINT, dataCount);
}

template <typename T>
__aicore__ void AddcMulScalarListAdapter(
    const LocalTensor<T>& tensor1Local, const LocalTensor<T>& tensor2Local, const LocalTensor<T>& tensor3Local,
    const LocalTensor<float>& float32Tensor, const float scalarValue, const uint32_t maxCastDataCount,
    const int64_t dataCount)
{
    uint32_t castTimes = dataCount / maxCastDataCount;
    uint32_t castTimesRemainder = dataCount % maxCastDataCount;
    for (uint32_t i = 0; i < castTimes; i++) {
        ComputerPerCastForAddcMulScalarList<T>(
            tensor1Local, tensor2Local, tensor3Local, float32Tensor, scalarValue, maxCastDataCount, i,
            maxCastDataCount);
    }
    if (castTimesRemainder > 0) {
        ComputerPerCastForAddcMulScalarList<T>(
            tensor1Local, tensor2Local, tensor3Local, float32Tensor, scalarValue, maxCastDataCount, castTimes,
            castTimesRemainder);
    }
}

extern "C" __global__ __aicore__ void foreach_addcmul_scalar_list(
    GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR tensor3, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachOneScalarListQuaternaryImplictOutput<half, AddcMulScalarListAdapter<half>, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachOneScalarListQuaternaryImplictOutput<float, AddcMulScalarListAdapterForFloat<float>, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    }
#if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(3)) {
        ForeachOneScalarListQuaternaryImplictOutput<int, AddcMulScalarListAdapterForInt<int>, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        ForeachOneScalarListQuaternaryImplictOutput<bfloat16_t, AddcMulScalarListAdapter<bfloat16_t>, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    }
#endif
}