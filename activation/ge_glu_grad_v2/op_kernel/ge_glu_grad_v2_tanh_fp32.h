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
 * \file ge_glu_grad_v2_tanh_fp32.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_TANH_FP32_H_
#define GE_GLU_GRAD_V2_TANH_FP32_H_

#include "ge_glu_grad_v2_tanh_base.h"

namespace GeGluGradV2Tanh {
using namespace AscendC;

class GeGluGradV2TanhFP32 : public GeGluGradV2TanhBase<float>
{
public:
    __aicore__ inline GeGluGradV2TanhFP32(
        GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx, const GeGluGradV2TilingData* tilingDataPtr)
        : GeGluGradV2TanhBase<float>(dy, x, gelu, dx, tilingDataPtr){};
    __aicore__ inline void Init();

    __aicore__ inline void Process(bool perfMode = false)
    {
        if (perfMode) {
            ProcessPerf<
                GeGluGradV2TanhFP32, &GeGluGradV2TanhFP32::ComputeLeftHalf, &GeGluGradV2TanhFP32::ComputeRightHalf>(
                this);
            return;
        }

        if (valueM <= maxProcCount) {
            ProcessLessEqual<
                GeGluGradV2TanhFP32, &GeGluGradV2TanhFP32::ComputeLeftHalf, &GeGluGradV2TanhFP32::ComputeRightHalf>(
                this);
        } else {
            ProcessGreater<
                GeGluGradV2TanhFP32, &GeGluGradV2TanhFP32::ComputeLeftHalf, &GeGluGradV2TanhFP32::ComputeRightHalf>(
                this);
        }
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
};

__aicore__ inline void GeGluGradV2TanhFP32::Init()
{
    pipe.InitBuffer(inQueueDY, NO_DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(inQueueGelu, DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(inQueueX1, NO_DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(inQueueX2, NO_DB_BUFFER, maxProcCount * sizeof(float));

    pipe.InitBuffer(outQueueDX1, NO_DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(outQueueDX2, NO_DB_BUFFER, maxProcCount * sizeof(float));

    pipe.InitBuffer(resultTempBuf, FP32_TEMP_BUF_CNT * maxProcCount * sizeof(float));
}

__aicore__ inline void GeGluGradV2TanhFP32::ComputeLeftHalf(const int64_t& realProcCount)
{
    LocalTensor<float> ubDY = inQueueDY.DeQue<float>();
    LocalTensor<float> ubGelu = inQueueGelu.DeQue<float>();
    LocalTensor<float> outLocalLeft = outQueueDX1.AllocTensor<float>();
    Mul(outLocalLeft, ubGelu, ubDY, realProcCount); // dx1 = gelu * dy
    outQueueDX1.EnQue(outLocalLeft);
    inQueueGelu.FreeTensor(ubGelu);

    LocalTensor<float> ubX1 = inQueueX1.DeQue<float>();
    LocalTensor<float> xBufLeft = GetTempBuf<float>(0);
    Mul(xBufLeft, ubX1, ubDY, realProcCount); // x1 = x1 * dy
    inQueueDY.FreeTensor(ubDY);
    inQueueX1.FreeTensor(ubX1);
}

__aicore__ inline void GeGluGradV2TanhFP32::ComputeRightHalf(const int64_t& realProcCount)
{
    LocalTensor<float> ubX2 = inQueueX2.DeQue<float>();
    LocalTensor<float> outLocalRight = outQueueDX2.AllocTensor<float>();
    LocalTensor<float> xBufLeft = GetTempBuf<float>(0);
    ComputeGeluGrad(outLocalRight, xBufLeft, ubX2, realProcCount);
    outQueueDX2.EnQue(outLocalRight);
    inQueueX2.FreeTensor(ubX2);
}

} // namespace GeGluGradV2Tanh

#endif // GE_GLU_GRAD_V2_TANH_FP32_H_