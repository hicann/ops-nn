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
 * \file ge_glu_grad_v2_erf_bfp16.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_ERF_BFP16_H_
#define GE_GLU_GRAD_V2_ERF_BFP16_H_

#include "ge_glu_grad_v2_erf_base.h"

namespace GeGluGradV2Erf {
using namespace AscendC;

class GeGluGradV2ErfBFP16 : public GeGluGradV2ErfBase<bfloat16_t>
{
public:
    __aicore__ inline GeGluGradV2ErfBFP16(
        GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx, const GeGluGradV2TilingData* tilingDataPtr, TPipe* tPipe)
        : GeGluGradV2ErfBase<bfloat16_t>(dy, x, gelu, dx, tilingDataPtr, tPipe){};
    __aicore__ inline void Init();

    __aicore__ inline void Process(bool perfMode = false)
    {
        if (perfMode) {
            ProcessPerf<
                GeGluGradV2ErfBFP16, &GeGluGradV2ErfBFP16::ComputeLeftHalf, &GeGluGradV2ErfBFP16::ComputeRightHalf>(
                this);
            return;
        }

        if (valueM <= maxProcCount) {
            ProcessLessEqual<
                GeGluGradV2ErfBFP16, &GeGluGradV2ErfBFP16::ComputeLeftHalf, &GeGluGradV2ErfBFP16::ComputeRightHalf>(
                this);
        } else {
            ProcessGreater<
                GeGluGradV2ErfBFP16, &GeGluGradV2ErfBFP16::ComputeLeftHalf, &GeGluGradV2ErfBFP16::ComputeRightHalf>(
                this);
        }
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
};

__aicore__ inline void GeGluGradV2ErfBFP16::Init()
{
    pipe->InitBuffer(inQueueX1, DB_BUFFER, maxProcCount * sizeof(bfloat16_t));
    pipe->InitBuffer(inQueueX2, DB_BUFFER, maxProcCount * sizeof(bfloat16_t));
    pipe->InitBuffer(inQueueDY, DB_BUFFER, maxProcCount * sizeof(bfloat16_t));
    pipe->InitBuffer(inQueueGelu, DB_BUFFER, maxProcCount * sizeof(bfloat16_t));

    pipe->InitBuffer(outQueueDX1, DB_BUFFER, maxProcCount * sizeof(bfloat16_t));
    pipe->InitBuffer(outQueueDX2, DB_BUFFER, maxProcCount * sizeof(bfloat16_t));

    pipe->InitBuffer(resultTempBuf0, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf1, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf2, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf4, maxProcCount * sizeof(float));
}

__aicore__ inline void GeGluGradV2ErfBFP16::ComputeLeftHalf(const int64_t& realProcCount)
{
    LocalTensor<float> ubDY = resultTempBuf4.Get<float>();
    LocalTensor<bfloat16_t> ubDYbf16 = inQueueDY.DeQue<bfloat16_t>();
    LocalTensor<float> ubGelu = resultTempBuf1.Get<float>();
    LocalTensor<bfloat16_t> ubGelubf16 = inQueueGelu.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> outLocalLeft = outQueueDX1.AllocTensor<bfloat16_t>();
    LocalTensor<float> xBufLeft = resultTempBuf0.Get<float>();
    LocalTensor<bfloat16_t> ubX1 = inQueueX1.DeQue<bfloat16_t>();
    LocalTensor<bfloat16_t> ubX2 = inQueueX2.DeQue<bfloat16_t>();

    Cast(ubDY, ubDYbf16, RoundMode::CAST_NONE, realProcCount);
    Cast(ubGelu, ubGelubf16, RoundMode::CAST_NONE, realProcCount);
    Mul(ubGelu, ubGelu, ubDY, realProcCount); // dx1 = gelu * dy
    Cast(outLocalLeft, ubGelu, RoundMode::CAST_RINT, realProcCount);
    Cast(xBufLeft, ubX1, RoundMode::CAST_NONE, realProcCount);
    Mul(xBufLeft, xBufLeft, ubDY, realProcCount); // x1 = x1 * dy
    Cast(ubDY, ubX2, RoundMode::CAST_NONE, realProcCount);

    inQueueDY.FreeTensor(ubDYbf16);
    inQueueGelu.FreeTensor(ubGelubf16);
    inQueueX1.FreeTensor(ubX1);
    inQueueX2.FreeTensor(ubX2);
    outQueueDX1.EnQue(outLocalLeft);
}

__aicore__ inline void GeGluGradV2ErfBFP16::ComputeRightHalf(const int64_t& realProcCount)
{
    LocalTensor<float> xBufRight = resultTempBuf4.Get<float>();
    LocalTensor<float> xBufLeft = resultTempBuf0.Get<float>();
    ComputeGeluGrad(xBufLeft, xBufLeft, xBufRight, realProcCount);

    LocalTensor<bfloat16_t> outLocalRight = outQueueDX2.AllocTensor<bfloat16_t>();
    Cast(outLocalRight, xBufLeft, RoundMode::CAST_RINT, realProcCount);
    outQueueDX2.EnQue(outLocalRight);
}

} // namespace GeGluGradV2Erf

#endif // GE_GLU_GRAD_V2_ERF_BFP16_H_