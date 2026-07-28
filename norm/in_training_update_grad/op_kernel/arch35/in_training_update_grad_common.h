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
 * \file in_training_update_grad_common.h
 * \brief Shared regbase helpers: dtype-aware register load, Newton-Raphson rstd, and the
 *        C0-kept vertical accumulation (accGamma += dy*x_norm, accBeta += dy) over spatial rows.
 */
#ifndef IN_TRAINING_UPDATE_GRAD_COMMON_H_
#define IN_TRAINING_UPDATE_GRAD_COMMON_H_

#include "kernel_operator.h"
#include "in_training_update_grad_tiling_data.h"

namespace InTrainingUpdateGrad {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr uint32_t VECTOR_REG_WIDTH = 256U;                    // DAV_3510 vector register width in bytes
constexpr uint32_t VL_FP32 = VECTOR_REG_WIDTH / sizeof(float); // 64
constexpr uint32_t BUFFER_NUM = 2;

// eps and IEEE guards for rsqrt(variance + eps)
constexpr float IN_EPSILON = 1e-6f;
constexpr float RSTD_POS_INF = 3.40282366920938E+38f;
constexpr float RSTD_ZERO = 0.0f;

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }

// Load one register of `T` from UB at `offset`, promoting fp16 -> fp32 (fp32 loads directly).
template <typename T>
__aicore__ inline void LoadTensorT(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src + offset);
        Cast<float, T, castTraitB162B32>(dst, raw, preg);
    }
}

// rstd = rsqrt(variance + eps), Newton-Raphson (copied from instance_norm CalculateRstd),
// with the IEEE guards var==+inf -> rstd=0 and var==0 -> rstd=+inf. Operates on `count` fp32 elements.
__aicore__ inline void ComputeRstd(__local_mem__ float* varUb, __local_mem__ float* rstdUb, uint32_t count, float eps)
{
    uint16_t aLoop = static_cast<uint16_t>(CeilDiv(count, VL_FP32));
    __VEC_SCOPE__
    {
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> var;
        RegTensor<float> one;
        RegTensor<float> r;
        RegTensor<float> y;
        RegTensor<float> s;
        RegTensor<float> t;
        RegTensor<float> scalar1;
        RegTensor<float> scalarInf;
        RegTensor<float> scalarZero;
        RegTensor<float> t1;
        RegTensor<float> t3;
        RegTensor<float> t4;
        RegTensor<float> rstd;

        MaskReg cmpRegZero;
        MaskReg cmpRegInf;
        MaskReg pregLoop;

        Duplicate(one, 1.0f, pregMain);
        uint32_t sreg0 = count;
        for (uint16_t a = 0; a < aLoop; a++) {
            pregLoop = UpdateMask<float>(sreg0);
            Duplicate(scalar1, float(0.5), pregLoop);
            Duplicate(scalarInf, RSTD_POS_INF, pregLoop);
            Duplicate(scalarZero, RSTD_ZERO, pregLoop);
            Duplicate(t1, float(1.5), pregLoop);
            Duplicate(s, float(1.0), pregLoop);

            DataCopy(var, varUb + a * VL_FP32);
            Adds(var, var, eps, pregLoop);
            Div(r, one, var, pregLoop);
            Sqrt(y, r, pregLoop);
            Muls(t, var, float(-0.5), pregLoop);
            Mul(t, t, y, pregLoop);
            Mula(t1, t, y, pregLoop);
            Mul(rstd, y, t1, pregLoop);
            Muls(t3, var, float(-1.0), pregLoop);
            Mula(s, t3, r, pregLoop);
            Muls(t4, rstd, float(-1.0), pregLoop);
            Mula(r, t4, rstd, pregLoop);
            Mula(s, var, r, pregLoop);
            Mul(s, s, rstd, pregLoop);
            Mula(rstd, s, scalar1, pregLoop);
            CompareScalar(cmpRegZero, var, RSTD_POS_INF, pregLoop);
            Select(rstd, scalarZero, rstd, cmpRegZero);
            CompareScalar(cmpRegInf, var, RSTD_ZERO, pregLoop);
            Select(rstd, scalarInf, rstd, cmpRegInf);
            DataCopy(rstdUb + a * VL_FP32, rstd, pregLoop);
        }
    }
}

// Zero out `c0` fp32 lanes of a C0-wide accumulator in UB.
__aicore__ inline void ZeroC0(__local_mem__ float* accUb, uint32_t c0)
{
    __VEC_SCOPE__
    {
        RegTensor<float> zero;
        uint32_t sreg = c0;
        MaskReg preg = UpdateMask<float>(sreg);
        Duplicate(zero, 0.0f, preg);
        DataCopy(accUb, zero, preg);
    }
}

// C0-kept vertical accumulation over `rowCount` spatial rows of a (rows, C0) UB block:
//   x_norm = (x - mean) * rstd   (mean/rstd broadcast per C0 column)
//   accGamma[c0] += x_norm[r,c0] * dy[r,c0]
//   accBeta[c0]  += dy[r,c0]
// C0 is the KEPT axis (16 lanes), rows are reduced vertically (register-to-register add), so there is
// NO horizontal ReduceSum here. When initAcc is false the accumulators are reloaded from UB first
// (Stream reuses them across chunks). fp16 inputs are promoted to fp32; all math is fp32.
template <typename T>
__aicore__ inline void AccumulateGroupC0(__local_mem__ T* dyUb, __local_mem__ T* xUb, __local_mem__ float* meanUb,
                                         __local_mem__ float* rstdUb, __local_mem__ float* accGammaUb,
                                         __local_mem__ float* accBetaUb, __local_mem__ float* cGammaUb,
                                         __local_mem__ float* cBetaUb, uint32_t rowCount, uint32_t c0, bool initAcc)
{
    __VEC_SCOPE__
    {
        RegTensor<float> accGamma;
        RegTensor<float> accBeta;
        RegTensor<float> meanReg;
        RegTensor<float> rstdReg;
        RegTensor<float> dyReg;
        RegTensor<float> xReg;
        RegTensor<float> xNorm;
        // Kahan compensated-summation state. The spatial reduction sums many large
        // dy*x_norm (and dy) terms that cancel (Sum|terms|/|result| ~ 1e4-1e5), so a
        // naive fp32 accumulation has error ~eps*N and is ~5x worse than a pairwise
        // competitor. Kahan keeps the error ~eps regardless of N.
        // ★ The compensation is carried across calls (cGammaUb/cBetaUb) the same way the
        // running sum is (accGammaUb/accBetaUb): for the Stream path a large reduction is
        // split into many chunks, and resetting the compensation per chunk would drop it
        // cross-chunk and regress a huge-M reduction back to naive error (verified: bigD
        // 66000 was ~50-96x worse than the GPU competitor). Reload when initAcc == false.
        RegTensor<float> cGamma;
        RegTensor<float> cBeta;
        RegTensor<float> termReg;
        RegTensor<float> yReg;
        RegTensor<float> tReg;
        RegTensor<float> dReg;
        // nan-guard: on inf/nan input the running sum overflows to inf (matches A2), but the
        // Kahan compensation (t-sum)-y = inf-inf = nan then poisons the next row's sum -> nan.
        // Zero the compensation whenever it is nan so the sum stays inf, aligned with A2/golden.
        RegTensor<float> zeroReg;
        MaskReg nanMask;

        uint32_t sregC0 = c0;
        MaskReg preg = UpdateMask<float>(sregC0); // C0 (=16) active lanes
        Duplicate(zeroReg, 0.0f, preg);
        DataCopy(meanReg, meanUb);
        DataCopy(rstdReg, rstdUb);
        if (initAcc) {
            Duplicate(accGamma, 0.0f, preg);
            Duplicate(accBeta, 0.0f, preg);
            Duplicate(cGamma, 0.0f, preg);
            Duplicate(cBeta, 0.0f, preg);
        } else {
            DataCopy(accGamma, accGammaUb);
            DataCopy(accBeta, accBetaUb);
            DataCopy(cGamma, cGammaUb);
            DataCopy(cBeta, cBetaUb);
        }
        for (uint16_t r = 0; r < static_cast<uint16_t>(rowCount); r++) {
            uint32_t off = r * c0;
            LoadTensorT<T>(dyUb, dyReg, preg, off);
            LoadTensorT<T>(xUb, xReg, preg, off);
            Sub(xReg, xReg, meanReg, preg);  // x - mean
            Mul(xNorm, xReg, rstdReg, preg); // (x - mean) * rstd
            // Kahan accGamma += x_norm*dy :  y = term - c;  t = sum + y;  c = (t - sum) - y;  sum = t
            Mul(termReg, xNorm, dyReg, preg);
            Sub(yReg, termReg, cGamma, preg);
            Add(tReg, accGamma, yReg, preg);
            Sub(dReg, tReg, accGamma, preg);
            Sub(cGamma, dReg, yReg, preg);
            Compare<float, CMPMODE::EQ>(nanMask, cGamma, cGamma, preg); // false only where cGamma is nan
            Select(cGamma, cGamma, zeroReg, nanMask);                   // nan -> 0, keep sum at inf
            Move(accGamma, tReg, preg);
            // Kahan accBeta += dy
            Sub(yReg, dyReg, cBeta, preg);
            Add(tReg, accBeta, yReg, preg);
            Sub(dReg, tReg, accBeta, preg);
            Sub(cBeta, dReg, yReg, preg);
            Compare<float, CMPMODE::EQ>(nanMask, cBeta, cBeta, preg);
            Select(cBeta, cBeta, zeroReg, nanMask);
            Move(accBeta, tReg, preg);
        }
        DataCopy(accGammaUb, accGamma, preg);
        DataCopy(accBetaUb, accBeta, preg);
        DataCopy(cGammaUb, cGamma, preg);
        DataCopy(cBetaUb, cBeta, preg);
    }
}
} // namespace InTrainingUpdateGrad
#endif // IN_TRAINING_UPDATE_GRAD_COMMON_H_
