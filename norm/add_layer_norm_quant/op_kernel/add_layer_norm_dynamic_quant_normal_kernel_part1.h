/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Dynamic quantization for zero, one, or two smooth inputs.
 * Include only from add_layer_norm_dynamic_quant_normal_kernel.h. */

__aicore__ inline void ComputeDynamicQuant(int32_t nums, int32_t elementCount)
{
    if (!this->smooth1Exist) {
        ComputeZeroDynamicQuant(nums, elementCount);
    } else if (!this->smooth2Exist) {
        ComputeSoleDynamicQuant(nums, elementCount);
    } else {
        ComputeDualDynamicQuant(nums, elementCount);
    }
}

__aicore__ inline void ComputeZeroDynamicQuant(int32_t nums, int32_t elementCount)
{
    LocalTensor<float> scaleLocal = scales1Que.template AllocTensor<float>();
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>(); // xLocalFp32 <-- y
    LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
    ScaleTensor(xLocalFp32, yLocalFp32, scaleLocal, elementCount, nums);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> yLocal = outRowQue.template AllocTensor<int8_t>();

    Cast(xLocalFp32.ReinterpretCast<int32_t>(), xLocalFp32, RoundMode::CAST_RINT, elementCount);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(xLocalFp32.ReinterpretCast<half>(), xLocalFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < nums; ++i) {
        Cast(yLocal[i * this->numLastDimRoundUp32], xLocalFp32.ReinterpretCast<half>()[i * this->numLastDimAligned],
             RoundMode::CAST_TRUNC, this->numLastDim);
    }

    PipeBarrier<PIPE_V>();
    outRowQue.EnQue(yLocal);
    scales1Que.EnQue(scaleLocal);
}

__aicore__ inline void ComputeSoleDynamicQuant(int32_t nums, int32_t elementCount)
{
    LocalTensor<float> scaleLocal = scales1Que.template AllocTensor<float>();
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>(); // xLocalFp32 <-- y
    LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

    LocalTensor<T> smoothLocal = inRowsQue.template DeQue<T>();

    // compute smooth1
    auto smoothFp32 = yLocalFp32[(nums - 1) * this->numLastDimAligned];
    if constexpr (!is_same<T, float>::value) {
        Cast(smoothFp32, smoothLocal, RoundMode::CAST_NONE, this->numLastDim);
    } else {
        Adds(smoothFp32, smoothLocal, (float)0.0, this->numLastDim);
    }
    PipeBarrier<PIPE_V>();
    inRowsQue.FreeTensor(smoothLocal);

    for (int32_t rid = 0; rid < nums; ++rid) {
        Mul(yLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], smoothFp32,
            this->numLastDim);
    }
    PipeBarrier<PIPE_V>();

    ScaleTensor(yLocalFp32, xLocalFp32, scaleLocal, elementCount, nums);
    PipeBarrier<PIPE_V>();

    LocalTensor<int8_t> yLocal = outRowQue.template AllocTensor<int8_t>();

    Cast(yLocalFp32.ReinterpretCast<int32_t>(), yLocalFp32, RoundMode::CAST_RINT, elementCount);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(yLocalFp32.ReinterpretCast<half>(), yLocalFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < nums; ++i) {
        Cast(yLocal[i * this->numLastDimRoundUp32], yLocalFp32.ReinterpretCast<half>()[i * this->numLastDimAligned],
             RoundMode::CAST_TRUNC, this->numLastDim);
    }

    outRowQue.EnQue(yLocal);
    scales1Que.EnQue(scaleLocal);
}

__aicore__ inline void ComputeDualDynamicQuant(int32_t nums, int32_t elementCount)
{
    LocalTensor<float> scale1Local = scales1Que.template AllocTensor<float>();
    LocalTensor<float> scale2Local = scales2Que.template AllocTensor<float>();
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>(); // xLocalFp32 <-- y
    LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

    LocalTensor<T> smooth12Local = inRowsQue.template DeQue<T>();
    auto smooth1Local = smooth12Local[0];
    auto smooth2Local = smooth12Local[this->numLastDimAligned];

    // compute smooth1
    LocalTensor<float> tmpLocal = smooth12Local.template ReinterpretCast<float>();
    auto smooth1Fp32 = yLocalFp32[(nums - 1) * this->numLastDimAligned];
    auto smooth2Fp32 = tmpLocal[0];
    if constexpr (!is_same<T, float>::value) {
        Cast(smooth1Fp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Cast(smooth2Fp32, smooth2Local, RoundMode::CAST_NONE, this->numLastDim);
    } else {
        Adds(smooth1Fp32, smooth1Local, (float)0.0, this->numLastDim);
        Adds(smooth2Fp32, smooth2Local, (float)0.0, this->numLastDim);
    }
    PipeBarrier<PIPE_V>();

    for (int32_t rid = 0; rid < nums; ++rid) {
        Mul(yLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], smooth1Fp32,
            this->numLastDim); // yLocalFp32 <-- y * smooth1
        Mul(xLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], smooth2Fp32,
            this->numLastDim); // xLocalFp32 <-- y * smooth2
    }
    PipeBarrier<PIPE_V>();

    ScaleTensor(yLocalFp32, tmpLocal, scale1Local, elementCount, nums);
    PipeBarrier<PIPE_V>();
    ScaleTensor(xLocalFp32, tmpLocal, scale2Local, elementCount, nums);
    PipeBarrier<PIPE_V>();

    inRowsQue.FreeTensor(smooth12Local);

    LocalTensor<int8_t> y12Local = outRowQue.template AllocTensor<int8_t>();
    auto y1Local = y12Local[0];

    Cast(yLocalFp32.ReinterpretCast<int32_t>(), yLocalFp32, RoundMode::CAST_RINT, elementCount);
    Cast(xLocalFp32.ReinterpretCast<int32_t>(), xLocalFp32, RoundMode::CAST_RINT, elementCount);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(yLocalFp32.ReinterpretCast<half>(), yLocalFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, elementCount);
    Cast(xLocalFp32.ReinterpretCast<half>(), xLocalFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, elementCount);
    PipeBarrier<PIPE_V>();

    for (int i = 0; i < nums; ++i) {
        Cast(y1Local[i * this->numLastDimRoundUp32], yLocalFp32.ReinterpretCast<half>()[i * this->numLastDimAligned],
             RoundMode::CAST_TRUNC, this->numLastDim);
        Cast(xLocalFp32.ReinterpretCast<int8_t>()[i * this->numLastDimRoundUp32],
             xLocalFp32.ReinterpretCast<half>()[i * this->numLastDimAligned], RoundMode::CAST_TRUNC, this->numLastDim);
    }
    PipeBarrier<PIPE_V>();

    outRowQue.EnQue(y12Local);
    scales1Que.EnQue(scale1Local);
    scales2Que.EnQue(scale2Local);
}
