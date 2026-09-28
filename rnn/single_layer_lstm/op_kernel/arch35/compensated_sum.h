/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_COMPENSATED_SUM_H
#define OPS_RNN_SINGLE_LAYER_LSTM_COMPENSATED_SUM_H
#include "kernel_operator.h"
#include "onchip_budget.h"

namespace SingleLayerLstmVec {
/* Consumes (and destroys) a drained partial AFTER reading it. No GM traffic.
 * sum/correction/y/tmp/partial must be distinct, compare-padded FP32 planes.
 * A nonfinite correction carries no meaningful roundoff; reset that correction
 * alone. The actual sum retains IEEE NaN/+Inf/-Inf, including Inf + -Inf. */
__aicore__ inline void FoldCubePartial(const AscendC::LocalTensor<float>& sum,
                                       const AscendC::LocalTensor<float>& correction,
                                       const AscendC::LocalTensor<float>& partial, const AscendC::LocalTensor<float>& y,
                                       const AscendC::LocalTensor<float>& tmp,
                                       const AscendC::LocalTensor<uint8_t>& mask, uint32_t n)
{
    AscendC::Sub(y, partial, correction, n);
    AscendC::Add(tmp, sum, y, n);
    AscendC::Sub(correction, tmp, sum, n);
    AscendC::Sub(correction, correction, y, n);
    AscendC::Abs(y, correction, n);
    const uint32_t cmpN = SingleLayerLstmCube::CeilAlign(n, SingleLayerLstmCube::CMP_REPEAT_ELEMS);
    if (cmpN > n) {
        AscendC::Duplicate(y[n], 0.0f, cmpN - n);
    }
    AscendC::Duplicate(partial, 0.0f, n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compares(mask, y, 3.40282347e+38F, AscendC::CMPMODE::LT, cmpN);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select(correction, mask, correction, partial, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    AscendC::Adds(sum, tmp, 0.0f, n);
}
} // namespace SingleLayerLstmVec
#endif
