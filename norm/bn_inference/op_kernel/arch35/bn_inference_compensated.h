/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BN_INFERENCE_COMPENSATED_MATH_H
#define BN_INFERENCE_COMPENSATED_MATH_H

#include "kernel_operator.h"

namespace BNInferenceOps {
namespace Compensated {
using namespace AscendC::Reg;

// Only the residual is sanitized. The leading value keeps IEEE NaN/Inf
// propagation when the compensated path cannot form a finite result.
__aicore__ inline void FiniteResidual(RegTensor<float>& value, MaskReg& mask)
{
    RegTensor<float> magnitude, zeros;
    MaskReg finite;
    Abs(magnitude, value, mask);
    CompareScalar<float, AscendC::CMPMODE::LE>(finite, magnitude, 0x1.fffffep127f, mask);
    Duplicate(zeros, 0.0f, mask);
    Select(value, value, zeros, finite);
}

__aicore__ inline void TwoSum(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& a, RegTensor<float>& b,
                              MaskReg& mask)
{
    RegTensor<float> sum, v, error, temp;
    Add(sum, a, b, mask);
    Sub(v, sum, a, mask);
    Sub(temp, sum, v, mask);
    Sub(error, a, temp, mask);
    Sub(temp, b, v, mask);
    Add(error, error, temp, mask);
    FiniteResidual(error, mask);
    // Multiplication by one preserves signed zero; addition of +0 does not.
    Muls(hi, sum, 1.0f, mask);
    Muls(lo, error, 1.0f, mask);
}

__aicore__ inline void MulPair(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& ah, RegTensor<float>& al,
                               RegTensor<float>& bh, RegTensor<float>& bl, MaskReg& mask)
{
    RegTensor<float> product, error, low;
    Mul(product, ah, bh, mask);
    Muls(error, product, -1.0f, mask);
    MulAddDst(error, ah, bh, mask);
    MulAddDst(error, ah, bl, mask);
    MulAddDst(error, al, bh, mask);
    MulAddDst(error, al, bl, mask);
    FiniteResidual(error, mask);
    TwoSum(hi, lo, product, error, mask);
}

__aicore__ inline void ComputeInverseRoot(AscendC::Reg::RegTensor<float>& rh, AscendC::Reg::RegTensor<float>& rl,
                                          AscendC::Reg::RegTensor<float>& hi, AscendC::Reg::RegTensor<float>& lo,
                                          AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    static constexpr DivSpecificMode divMode{MaskMergeMode::ZEROING, false, AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    RegTensor<float> root, one, zero, ph, pl, error;
    Duplicate(one, 1.0f, mask);
    Duplicate(zero, 0.0f, mask);
    // Refine a full-range root/reciprocal seed with a compensated Newton step.
    static constexpr SqrtSpecificMode sqrtMode{MaskMergeMode::ZEROING, false,
                                               AscendC::SqrtAlgo::PRECISION_1ULP_FTZ_FALSE};
    Sqrt<float, &sqrtMode>(root, hi, mask);
    Div<float, &divMode>(rh, one, root, mask);
    MulPair(ph, pl, hi, lo, rh, zero, mask);
    MulPair(ph, pl, ph, pl, rh, zero, mask);
    Sub(error, one, ph, mask);
    Sub(error, error, pl, mask);
    Mul(rl, rh, error, mask);
    Muls(rl, rl, 0.5f, mask);
    FiniteResidual(rl, mask);
    TwoSum(rh, rl, rh, rl, mask);
}

__aicore__ inline void ComputeWideRstd(RegTensor<float>& variance, RegTensor<float>& high, RegTensor<float>& low,
                                       MaskReg& mask, float epsilon)
{
    RegTensor<float> eps, magnitude, varScaled, epsScaled, factor, hi, lo;
    MaskReg large;
    Duplicate(eps, epsilon, mask);
    Abs(magnitude, variance, mask);
    Abs(factor, eps, mask);
    Max(magnitude, magnitude, factor, mask);
    CompareScalar<float, AscendC::CMPMODE::GT>(large, magnitude, 0x1p124f, mask);
    Duplicate(factor, 1.0f, mask);
    Duplicate(hi, 0.25f, mask);
    Select(factor, hi, factor, large);
    Mul(varScaled, variance, factor, mask);
    Mul(epsScaled, eps, factor, mask);
    TwoSum(hi, lo, varScaled, epsScaled, mask);
    ComputeInverseRoot(high, low, hi, lo, mask);
    Duplicate(factor, 1.0f, mask);
    Duplicate(hi, 0.5f, mask);
    Select(factor, hi, factor, large);
    Mul(high, high, factor, mask);
    Mul(low, low, factor, mask);
}

} // namespace Compensated
} // namespace BNInferenceOps
#endif // BN_INFERENCE_COMPENSATED_MATH_H
