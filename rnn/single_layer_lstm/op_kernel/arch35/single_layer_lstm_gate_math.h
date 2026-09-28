/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_SINGLE_LAYER_LSTM_GATE_MATH_H
#define OPS_NN_SINGLE_LAYER_LSTM_GATE_MATH_H
#include "kernel_operator.h"

// Ascend950 FP32 vector gate arithmetic shared by forward and backward replay.
// Scratch t1/t2/t3 must hold ceil(n/64)*64 floats; msk holds at least that many bits.
// The small-x polynomial avoids cancellation in the exp-based tanh formula.
namespace SingleLayerLstmVec {
constexpr uint32_t CMP_REPEAT_ELEMS = 256U / sizeof(float);
__aicore__ inline uint32_t LstmGateCeilAlign(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

// dst may alias poly; src and squared must be distinct scratch tensors.
__aicore__ inline void TanhSmallVec(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                                    const AscendC::LocalTensor<float>& squared, const AscendC::LocalTensor<float>& poly,
                                    uint32_t n)
{
    AscendC::Mul(squared, src, src, n);
    AscendC::Muls(poly, squared, static_cast<float>(-0.006274179), n);
    AscendC::Adds(poly, poly, static_cast<float>(0.021071639), n);
    AscendC::Mul(poly, poly, squared, n);
    AscendC::Adds(poly, poly, static_cast<float>(-0.0538523), n);
    AscendC::Mul(poly, poly, squared, n);
    AscendC::Adds(poly, poly, static_cast<float>(0.13332586), n);
    AscendC::Mul(poly, poly, squared, n);
    AscendC::Adds(poly, poly, static_cast<float>(-0.33333316), n);
    AscendC::Mul(poly, poly, squared, n);
    AscendC::Mul(poly, poly, src, n);
    AscendC::Add(dst, poly, src, n);
}

// Near zero, sigmoid(z) = 0.5 + 0.5 * tanh(z / 2) avoids rounding exp(-z) + 1.
// Keep |z / 2| below 0.25, inside the existing small-x tanh polynomial domain.
// Other lanes retain the exp/div result, including infinities and NaNs.
constexpr float SIGMOID_CENTERED_LIMIT = 0.5f;
__aicore__ inline void SigmoidVec(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& t1,
                                  const AscendC::LocalTensor<float>& t2, const AscendC::LocalTensor<float>& t3,
                                  const AscendC::LocalTensor<uint8_t>& msk, uint32_t n)
{
    AscendC::Muls(t1, dst, static_cast<float>(-1.0), n);
    AscendC::Exp(t1, t1, n);
    AscendC::Adds(t1, t1, static_cast<float>(1.0), n);
    AscendC::Duplicate(t2, static_cast<float>(1.0), n);
    AscendC::Div(t2, t2, t1, n);

    AscendC::Abs(t1, dst, n);
    const uint32_t cmpCount = LstmGateCeilAlign(n, CMP_REPEAT_ELEMS);
    if (cmpCount > n) {
        AscendC::Duplicate(t1[n], static_cast<float>(0.0), cmpCount - n);
    }
    AscendC::Compares(msk, t1, SIGMOID_CENTERED_LIMIT, AscendC::CMPMODE::LT, cmpCount);
    AscendC::Muls(t1, dst, static_cast<float>(0.5), n);
    TanhSmallVec(dst, t1, t3, dst, n);
    AscendC::Muls(dst, dst, static_cast<float>(0.5), n);
    AscendC::Adds(dst, dst, static_cast<float>(0.5), n);
    AscendC::Select(dst, msk, dst, t2, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
}

__aicore__ inline void TanhVec(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                               const AscendC::LocalTensor<float>& t1, const AscendC::LocalTensor<float>& t2,
                               const AscendC::LocalTensor<float>& t3, const AscendC::LocalTensor<uint8_t>& msk,
                               uint32_t n)
{
    TanhSmallVec(t2, src, t1, t2, n);
    AscendC::Muls(t3, src, static_cast<float>(2.0), n);
    AscendC::Mins(t3, t3, static_cast<float>(20.0), n);
    AscendC::Exp(t3, t3, n);
    AscendC::Adds(t1, t3, static_cast<float>(1.0), n);
    AscendC::Adds(t3, t3, static_cast<float>(-1.0), n);
    AscendC::Div(t3, t3, t1, n);
    AscendC::Abs(t1, src, n);
    const uint32_t cmpCount = LstmGateCeilAlign(n, CMP_REPEAT_ELEMS);
    if (cmpCount > n) {
        AscendC::Duplicate(t1[n], static_cast<float>(0.0), cmpCount - n);
    }
    AscendC::Compares(msk, t1, static_cast<float>(0.55), AscendC::CMPMODE::LT, cmpCount);
    AscendC::Select(dst, msk, t2, t3, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
}

__aicore__ inline void SingleLayerLstmGates(
    const AscendC::LocalTensor<float>& gi, const AscendC::LocalTensor<float>& gf, const AscendC::LocalTensor<float>& gg,
    const AscendC::LocalTensor<float>& go, const AscendC::LocalTensor<float>& cc, const AscendC::LocalTensor<float>& hh,
    const AscendC::LocalTensor<float>& co, const AscendC::LocalTensor<float>& t1, const AscendC::LocalTensor<float>& t2,
    const AscendC::LocalTensor<float>& t3, const AscendC::LocalTensor<uint8_t>& msk, uint32_t n)
{
    SigmoidVec(gi, t1, t2, t3, msk, n);
    SigmoidVec(gf, t1, t2, t3, msk, n);
    TanhVec(gg, gg, t1, t2, t3, msk, n);
    SigmoidVec(go, t1, t2, t3, msk, n);
    AscendC::Mul(t1, gi, gg, n);
    AscendC::Mul(cc, gf, cc, n);
    AscendC::Add(cc, cc, t1, n);
    TanhVec(co, cc, t1, t2, t3, msk, n);
    AscendC::Mul(hh, go, co, n);
}

__aicore__ inline void SingleLayerLstmGates(
    const AscendC::LocalTensor<float>& gi, const AscendC::LocalTensor<float>& gf, const AscendC::LocalTensor<float>& gg,
    const AscendC::LocalTensor<float>& go, const AscendC::LocalTensor<float>& cc, const AscendC::LocalTensor<float>& hh,
    const AscendC::LocalTensor<float>& t1, const AscendC::LocalTensor<float>& t2, const AscendC::LocalTensor<float>& t3,
    const AscendC::LocalTensor<uint8_t>& msk, uint32_t n)
{
    SingleLayerLstmGates(gi, gf, gg, go, cc, hh, hh, t1, t2, t3, msk, n);
}
} // namespace SingleLayerLstmVec
#endif
