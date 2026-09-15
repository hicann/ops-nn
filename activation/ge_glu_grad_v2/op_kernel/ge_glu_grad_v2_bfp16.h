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
 * \file ge_glu_grad_v2_bfp16.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_BFP16_H_
#define GE_GLU_GRAD_V2_BFP16_H_

#include "ge_glu_grad_v2_base.h"

namespace GeGluGradV2 {
using namespace AscendC;

template <bool IS_ERF>
class GeGluGradV2BFP16 : public GeGluGradV2Base<bfloat16_t, IS_ERF> {
public:
    __aicore__ inline GeGluGradV2BFP16(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx,
                                       const GeGluGradV2TilingData* tilingDataPtr)
        : GeGluGradV2Base<bfloat16_t, IS_ERF>(dy, x, gelu, dx, tilingDataPtr){};
    __aicore__ inline void Init();

    __aicore__ inline void Process(bool perfMode = false)
    {
        if (perfMode) {
            this->template ProcessPerf<GeGluGradV2BFP16<IS_ERF>, &GeGluGradV2BFP16<IS_ERF>::ComputeLeftHalf,
                                       &GeGluGradV2BFP16<IS_ERF>::ComputeRightHalf>(this);
            return;
        }

        if (this->valueM <= this->maxProcCount) {
            this->template ProcessLessEqual<GeGluGradV2BFP16<IS_ERF>, &GeGluGradV2BFP16<IS_ERF>::ComputeLeftHalf,
                                            &GeGluGradV2BFP16<IS_ERF>::ComputeRightHalf>(this);
        } else {
            this->template ProcessGreater<GeGluGradV2BFP16<IS_ERF>, &GeGluGradV2BFP16<IS_ERF>::ComputeLeftHalf,
                                          &GeGluGradV2BFP16<IS_ERF>::ComputeRightHalf>(this);
        }
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
};

template <bool IS_ERF>
__aicore__ inline void GeGluGradV2BFP16<IS_ERF>::Init()
{
    this->pipe.InitBuffer(this->inQueueX1, NO_DB_BUFFER, this->maxProcCount * sizeof(bfloat16_t));
    this->pipe.InitBuffer(this->inQueueX2, NO_DB_BUFFER, this->maxProcCount * sizeof(bfloat16_t));
    this->pipe.InitBuffer(this->inQueueDY, NO_DB_BUFFER, this->maxProcCount * sizeof(float));
    this->pipe.InitBuffer(this->inQueueGelu, NO_DB_BUFFER, this->maxProcCount * sizeof(float));

    this->pipe.InitBuffer(this->outQueueDX1, NO_DB_BUFFER, this->maxProcCount * sizeof(bfloat16_t));
    this->pipe.InitBuffer(this->outQueueDX2, NO_DB_BUFFER, this->maxProcCount * sizeof(bfloat16_t));

    this->pipe.InitBuffer(this->resultTempBuf, BFP16_TEMP_BUF_CNT * this->maxProcCount * sizeof(float));
}

template <bool IS_ERF>
__aicore__ inline void GeGluGradV2BFP16<IS_ERF>::ComputeLeftHalf(const int64_t& realProcCount)
{
    LocalTensor<float> ubDY = this->inQueueDY.template DeQue<float>();
    LocalTensor<bfloat16_t> ubDYbf16 = ubDY.ReinterpretCast<bfloat16_t>()[this->maxProcCount];
    Cast(ubDY, ubDYbf16, RoundMode::CAST_NONE, realProcCount);

    LocalTensor<float> ubGelu = this->inQueueGelu.template DeQue<float>();
    LocalTensor<bfloat16_t> ubGelubf16 = ubGelu.ReinterpretCast<bfloat16_t>()[this->maxProcCount];
    Cast(ubGelu, ubGelubf16, RoundMode::CAST_NONE, realProcCount);

    LocalTensor<bfloat16_t> outLocalLeft = this->outQueueDX1.template AllocTensor<bfloat16_t>();

    Mul(ubGelu, ubGelu, ubDY, realProcCount); // dx1 = gelu * dy
    Cast(outLocalLeft, ubGelu, RoundMode::CAST_RINT, realProcCount);
    this->outQueueDX1.EnQue(outLocalLeft);
    this->inQueueGelu.FreeTensor(ubGelu);

    LocalTensor<bfloat16_t> ubX1 = this->inQueueX1.template DeQue<bfloat16_t>();
    LocalTensor<float> xBufLeft = this->template GetTempBuf<float>(0);
    Cast(xBufLeft, ubX1, RoundMode::CAST_NONE, realProcCount);
    this->inQueueX1.FreeTensor(ubX1);
    Mul(xBufLeft, xBufLeft, ubDY, realProcCount); // x1 = x1 * dy
    this->inQueueDY.FreeTensor(ubDY);
}

template <bool IS_ERF>
__aicore__ inline void GeGluGradV2BFP16<IS_ERF>::ComputeRightHalf(const int64_t& realProcCount)
{
    LocalTensor<bfloat16_t> ubX2 = this->inQueueX2.template DeQue<bfloat16_t>();
    LocalTensor<float> xBufRight = this->template GetTempBuf<float>(4);
    Cast(xBufRight, ubX2, RoundMode::CAST_NONE, realProcCount);
    this->inQueueX2.FreeTensor(ubX2);

    LocalTensor<float> xBufLeft = this->template GetTempBuf<float>(0);
    this->ComputeGeluGrad(xBufLeft, xBufLeft, xBufRight, realProcCount);

    LocalTensor<bfloat16_t> outLocalRight = this->outQueueDX2.template AllocTensor<bfloat16_t>();
    Cast(outLocalRight, xBufLeft, RoundMode::CAST_RINT, realProcCount);
    this->outQueueDX2.EnQue(outLocalRight);
}

} // namespace GeGluGradV2

#endif // GE_GLU_GRAD_V2_BFP16_H_
