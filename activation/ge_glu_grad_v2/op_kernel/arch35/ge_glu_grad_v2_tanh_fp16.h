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
 * \file ge_glu_grad_v2_tanh_fp16.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_TANH_FP16_H_
#define GE_GLU_GRAD_V2_TANH_FP16_H_

#include "ge_glu_grad_v2_tanh_base.h"

namespace GeGluGradV2Tanh {
using namespace AscendC;

class GeGluGradV2TanhFP16 : public GeGluGradV2TanhBase<half>
{
public:
    __aicore__ inline GeGluGradV2TanhFP16(
        GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx, const GeGluGradV2TilingData* tilingDataPtr, TPipe* tPipe)
        : GeGluGradV2TanhBase<half>(dy, x, gelu, dx, tilingDataPtr, tPipe){};
    __aicore__ inline void Init();

    __aicore__ inline void Process(bool perfMode = false)
    {
        if (perfMode) {
            ProcessPerf<
                GeGluGradV2TanhFP16, &GeGluGradV2TanhFP16::ComputeLeftHalf, &GeGluGradV2TanhFP16::ComputeRightHalf>(
                this);
            return;
        }

        if (valueM <= maxProcCount) {
            ProcessLessEqual<
                GeGluGradV2TanhFP16, &GeGluGradV2TanhFP16::ComputeLeftHalf, &GeGluGradV2TanhFP16::ComputeRightHalf>(
                this);
        } else {
            ProcessGreater<
                GeGluGradV2TanhFP16, &GeGluGradV2TanhFP16::ComputeLeftHalf, &GeGluGradV2TanhFP16::ComputeRightHalf>(
                this);
        }
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
};

__aicore__ inline void GeGluGradV2TanhFP16::Init()
{
    pipe->InitBuffer(inQueueDY, DB_BUFFER, maxProcCount * sizeof(half));
    pipe->InitBuffer(inQueueGelu, DB_BUFFER, maxProcCount * sizeof(half));
    pipe->InitBuffer(inQueueX1, DB_BUFFER, maxProcCount * sizeof(half));
    pipe->InitBuffer(inQueueX2, DB_BUFFER, maxProcCount * sizeof(half));

    pipe->InitBuffer(outQueueDX1, DB_BUFFER, maxProcCount * sizeof(half));
    pipe->InitBuffer(outQueueDX2, DB_BUFFER, maxProcCount * sizeof(half));

    pipe->InitBuffer(resultTempBuf0, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf1, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf2, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf3, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf4, maxProcCount * sizeof(float));
}

__aicore__ inline void GeGluGradV2TanhFP16::ComputeLeftHalf(const int64_t& realProcCount)
{
    LocalTensor<half> ubDY = inQueueDY.DeQue<half>();
    LocalTensor<half> ubGelu = inQueueGelu.DeQue<half>();
    LocalTensor<half> outLocalLeft = outQueueDX1.AllocTensor<half>();
    LocalTensor<half> ubX1 = inQueueX1.DeQue<half>();
    LocalTensor<float> xBufLeft = resultTempBuf0.Get<float>();
    Mul(outLocalLeft, ubGelu, ubDY, realProcCount); // dx1 = gelu * dy
    Mul(ubX1, ubX1, ubDY, realProcCount);           // x1 = x1 * dy
    Cast(xBufLeft, ubX1, RoundMode::CAST_NONE, realProcCount);
    inQueueDY.FreeTensor(ubDY);
    inQueueGelu.FreeTensor(ubGelu);
    inQueueX1.FreeTensor(ubX1);
    outQueueDX1.EnQue(outLocalLeft);
}

__aicore__ inline void GeGluGradV2TanhFP16::ComputeRightHalf(const int64_t& realProcCount)
{
    LocalTensor<half> ubX2 = inQueueX2.DeQue<half>();
    LocalTensor<float> xBufRight = resultTempBuf4.Get<float>();
    Cast(xBufRight, ubX2, RoundMode::CAST_NONE, realProcCount);
    inQueueX2.FreeTensor(ubX2);

    LocalTensor<float> xBufLeft = resultTempBuf0.Get<float>();
    ComputeGeluGrad(xBufLeft, xBufLeft, xBufRight, realProcCount);

    LocalTensor<half> outLocalRight = outQueueDX2.AllocTensor<half>();
    Cast(outLocalRight, xBufLeft, RoundMode::CAST_RINT, realProcCount);
    outQueueDX2.EnQue(outLocalRight);
}

} // namespace GeGluGradV2Tanh

#endif // GE_GLU_GRAD_V2_TANH_FP16_H_