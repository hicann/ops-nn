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
 * \file ge_glu_grad_v2_fp32_310p.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_FP32_310P_H_
#define GE_GLU_GRAD_V2_FP32_310P_H_

#include "ge_glu_grad_v2_base_310p.h"

namespace GeGluGradV2For310P {
using namespace AscendC;

class GeGluGradV2FP32By310p : public GeGluGradV2Base310p<float>
{
public:
    __aicore__ inline GeGluGradV2FP32By310p(
        GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx, GM_ADDR workspace, const GeGluGradV2TilingData* tilingDataPtr)
        : GeGluGradV2Base310p<float>(dy, x, gelu, dx, workspace, tilingDataPtr){};
    __aicore__ inline void Init();
    __aicore__ inline void Process()
    {
        if (valueM <= maxProcCount) {
            ProcessLessEqual<
                GeGluGradV2FP32By310p, &GeGluGradV2FP32By310p::ComputeLeftHalf,
                &GeGluGradV2FP32By310p::ComputeRightHalf>(this);
        } else {
            ProcessGreater<
                GeGluGradV2FP32By310p, &GeGluGradV2FP32By310p::ComputeLeftHalf,
                &GeGluGradV2FP32By310p::ComputeRightHalf>(this);
        }
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
};

__aicore__ inline void GeGluGradV2FP32By310p::Init()
{
    BaseInit();
    pipe.InitBuffer(inQueueDY, NO_DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(inQueueGelu, NO_DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(inQueueX1, NO_DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(inQueueX2, NO_DB_BUFFER, maxProcCount * sizeof(float)); // enable DB buffer

    pipe.InitBuffer(outQueueDX1, NO_DB_BUFFER, maxProcCount * sizeof(float));
    pipe.InitBuffer(outQueueDX2, NO_DB_BUFFER, maxProcCount * sizeof(float));

    pipe.InitBuffer(resultTempBuf1, maxProcCount * sizeof(float));
    pipe.InitBuffer(resultTempBuf3, maxProcCount * sizeof(float));
    pipe.InitBuffer(resultTempBuf4, maxProcCount * sizeof(float));
}

__aicore__ inline void GeGluGradV2FP32By310p::ComputeLeftHalf(const int64_t& realProcCount)
{
    LocalTensor<float> ubDY = inQueueDY.DeQue<float>();
    LocalTensor<float> ubGelu = inQueueGelu.DeQue<float>();
    LocalTensor<float> outLocalLeft = outQueueDX1.AllocTensor<float>();
    Mul(outLocalLeft, ubGelu, ubDY, realProcCount); // dx1 = gelu * dy
    outQueueDX1.EnQue(outLocalLeft);
    inQueueGelu.FreeTensor(ubGelu);

    LocalTensor<float> ubX1 = inQueueX1.DeQue<float>();
    LocalTensor<float> xBufLeft = resultTempBuf1.Get<float>();
    Mul(xBufLeft, ubX1, ubDY, realProcCount); // x1 = x1 * dy
    inQueueDY.FreeTensor(ubDY);
    inQueueX1.FreeTensor(ubX1);
}

__aicore__ inline void GeGluGradV2FP32By310p::ComputeRightHalf(const int64_t& realProcCount)
{
    LocalTensor<float> ubX2 = inQueueX2.DeQue<float>();
    LocalTensor<float> outLocalRight = outQueueDX2.AllocTensor<float>();
    LocalTensor<float> xBufLeft = resultTempBuf1.Get<float>();
    ComputeGeluGrad(outLocalRight, xBufLeft, ubX2, realProcCount);
    outQueueDX2.EnQue(outLocalRight);
    inQueueX2.FreeTensor(ubX2);
}

} // namespace GeGluGradV2For310P

#endif // GE_GLU_GRAD_V2_FP32_310P_H_