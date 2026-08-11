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
 * \file instance_norm_grad_common.h
 * \brief regbase VF helpers for InstanceNormGrad (NDHWC channel-last, column-wise reduction over M).
 */
#ifndef INSTANCE_NORM_GRAD_COMMON_H
#define INSTANCE_NORM_GRAD_COMMON_H
#pragma once

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace InstanceNormGrad {
using namespace AscendC;
using AscendC::MicroAPI::Compare;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::Select;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using ops::CeilAlign;

constexpr int DOUBLE_BUFFER = 2;
constexpr float INSTANCE_NORM_GRAD_EPS = 1e-6f;

__aicore__ inline constexpr uint32_t GetVRegSizeBytes()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
__aicore__ inline constexpr uint32_t GetUbBlockSize() { return 32U; }

constexpr uint32_t VL_FP32 = GetVRegSizeBytes() / sizeof(float);
constexpr uint32_t BLK_B32 = GetUbBlockSize() / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// Load T (fp16/fp32) from UB into an fp32 register (fp16 is unpacked+cast).
template <typename T>
__aicore__ inline void LoadTAsF32(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T> tmp;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(tmp, src + offset);
        Cast<float, T, castTraitB162B32>(dst, tmp, preg);
    }
}

// Store an fp32 register into UB as T (fp16 is cast+packed).
template <typename T>
__aicore__ inline void StoreF32AsT(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16>(tmp, src, preg);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dst + offset, tmp, preg);
    }
}

// rstd = (variance + eps)^(-1/2) over cLen fp32 elements (var/rstd are separate fp32 UB buffers).
__aicore__ inline void ComputeRstd(__local_mem__ float* varUb, __local_mem__ float* rstdUb, uint32_t cLen)
{
    uint16_t loopCnt = (cLen + VL_FP32 - 1) / VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> var;
        RegTensor<float> s;
        RegTensor<float> one;
        RegTensor<float> rstd;
        MaskReg preg;
        uint32_t sreg = cLen;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            preg = UpdateMask<float>(sreg);
            DataCopy(var, varUb + i * VL_FP32);
            Adds(var, var, INSTANCE_NORM_GRAD_EPS, preg);
            Sqrt(s, var, preg);
            Duplicate(one, 1.0f, preg);
            Div(rstd, one, s, preg);
            DataCopy(rstdUb + i * VL_FP32, rstd, preg);
        }
    }
}

// Zero a length-cLen fp32 accumulator buffer.
__aicore__ inline void ZeroF32(__local_mem__ float* buf, uint32_t cLen)
{
    uint16_t loopCnt = (cLen + VL_FP32 - 1) / VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> zero;
        MaskReg preg;
        uint32_t sreg = cLen;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            preg = UpdateMask<float>(sreg);
            Duplicate(zero, 0.0f, preg);
            DataCopy(buf + i * VL_FP32, zero, preg);
        }
    }
}

/*
 * Pass1: fold `rows` M-rows into length-cLen fp32 column accumulators.
 *   accPdVar  += -0.5 * (dy*gamma) * (x-mean) * rstd^3
 *   accPdMean += -1.0 * (dy*gamma) * rstd
 *   accDgamma += dy * ((x-mean)*rstd)
 *   accDbeta  += dy
 * x/dy are T tiles laid out [rows, rowStride] (rowStride in T elements, >= cLen, 32B aligned).
 * mean/rstd/gamma and the four accumulators are fp32 UB buffers of length cAlign(>= cLen).
 */
template <typename T>
__aicore__ inline void Pass1Accumulate(__local_mem__ T* xUb, __local_mem__ T* dyUb, __local_mem__ float* meanUb,
                                       __local_mem__ float* rstdUb, __local_mem__ float* gammaUb,
                                       __local_mem__ float* accPdVarUb, __local_mem__ float* accPdMeanUb,
                                       __local_mem__ float* accDgammaUb, __local_mem__ float* accDbetaUb,
                                       __local_mem__ float* cDgammaUb, __local_mem__ float* cDbetaUb,
                                       __local_mem__ float* cPdVarUb, __local_mem__ float* cPdMeanUb, uint32_t rows,
                                       uint32_t cLen, uint32_t rowStride)
{
    uint16_t cLoop = (cLen + VL_FP32 - 1) / VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> accPdVar, accPdMean, accDgamma, accDbeta;
        RegTensor<float> meanR, rstdR, gammaR, rstd3R;
        RegTensor<float> xR, dyR, xcR, tmpR;
        // Kahan compensated-summation state for the two reduction OUTPUTS (dgamma/dbeta).
        // The spatial sum of dy*xhat (and dy) cancels heavily (Sum|terms|/|result| ~ 1e4-1e5),
        // so a naive fp32 accumulation is ~5x worse than a pairwise competitor. Kahan keeps the
        // error ~eps regardless of the row count.
        RegTensor<float> cDgamma, cDbeta, kY, kT, kD;
        // pdVar/pdMean 同为跨 M-tile 的空间规约,且 pd_x 由二者线性组合得到,
        // 裸 fp32 累加的误差会直接进入 pd_x,故同样施加 Kahan 补偿。
        RegTensor<float> cPdVar, cPdMean;
        // nan-guard: on inf/nan input the running sum overflows to inf (matches ascend910b), but the Kahan
        // compensation (t-sum)-y = inf-inf = nan then poisons the next row's sum -> nan. Zero the
        // compensation whenever it is nan so the sum stays inf, aligned with ascend910b/golden.
        RegTensor<float> zeroReg;
        MaskReg preg;
        MaskReg nanMask;
        uint32_t sregC = cLen;
        for (uint16_t c = 0; c < cLoop; ++c) {
            preg = UpdateMask<float>(sregC);
            Duplicate(zeroReg, 0.0f, preg);
            uint32_t cOff = c * VL_FP32;
            DataCopy(meanR, meanUb + cOff);
            DataCopy(rstdR, rstdUb + cOff);
            DataCopy(gammaR, gammaUb + cOff);
            Mul(rstd3R, rstdR, rstdR, preg);
            Mul(rstd3R, rstd3R, rstdR, preg); // rstd^3 = (var+eps)^(-3/2)
            DataCopy(accPdVar, accPdVarUb + cOff);
            DataCopy(accPdMean, accPdMeanUb + cOff);
            DataCopy(accDgamma, accDgammaUb + cOff);
            DataCopy(accDbeta, accDbetaUb + cOff);
            // Reload the Kahan compensation (persisted like the accumulators): the recompute path
            // calls Pass1Accumulate once per M-tile, so resetting it per call would drop the
            // compensation cross-tile and regress a huge-M reduction back to naive error.
            DataCopy(cDgamma, cDgammaUb + cOff);
            DataCopy(cDbeta, cDbetaUb + cOff);
            DataCopy(cPdVar, cPdVarUb + cOff);
            DataCopy(cPdMean, cPdMeanUb + cOff);
            for (uint16_t m = 0; m < static_cast<uint16_t>(rows); ++m) {
                uint32_t off = m * rowStride + cOff;
                LoadTAsF32<T>(xUb, xR, preg, off);
                LoadTAsF32<T>(dyUb, dyR, preg, off);
                Sub(xcR, xR, meanR, preg); // x - mean
                // Kahan dbeta += dy :  y = dy - c; t = sum + y; c = (t - sum) - y; sum = t
                Sub(kY, dyR, cDbeta, preg);
                Add(kT, accDbeta, kY, preg);
                Sub(kD, kT, accDbeta, preg);
                Sub(cDbeta, kD, kY, preg);
                Compare<float, CMPMODE::EQ>(nanMask, cDbeta, cDbeta, preg); // false only where nan
                Select(cDbeta, cDbeta, zeroReg, nanMask);                   // nan -> 0, keep sum at inf
                Move(accDbeta, kT, preg);
                Mul(xR, dyR, gammaR, preg);  // pd_xl = dy * gamma  (reuse xR)
                Mul(tmpR, xcR, rstdR, preg); // xhat = (x-mean)*rstd
                // Kahan dgamma += dy*xhat :  term = dy*xhat (in kD), then compensate
                Mul(kD, dyR, tmpR, preg);
                Sub(kY, kD, cDgamma, preg);
                Add(kT, accDgamma, kY, preg);
                Sub(kD, kT, accDgamma, preg);
                Sub(cDgamma, kD, kY, preg);
                Compare<float, CMPMODE::EQ>(nanMask, cDgamma, cDgamma, preg);
                Select(cDgamma, cDgamma, zeroReg, nanMask);
                Move(accDgamma, kT, preg);
                Mul(tmpR, xR, rstdR, preg); // pd_xl * rstd
                // Kahan pdMean += -pd_xl*rstd
                Muls(kD, tmpR, -1.0f, preg);
                Sub(kY, kD, cPdMean, preg);
                Add(kT, accPdMean, kY, preg);
                Sub(kD, kT, accPdMean, preg);
                Sub(cPdMean, kD, kY, preg);
                Compare<float, CMPMODE::EQ>(nanMask, cPdMean, cPdMean, preg);
                Select(cPdMean, cPdMean, zeroReg, nanMask);
                Move(accPdMean, kT, preg);
                Mul(tmpR, xR, xcR, preg);      // pd_xl * (x-mean)
                Mul(tmpR, tmpR, rstd3R, preg); // * rstd^3
                // Kahan pdVar += -0.5 * pd_xl*(x-mean)*rstd^3
                Muls(kD, tmpR, -0.5f, preg);
                Sub(kY, kD, cPdVar, preg);
                Add(kT, accPdVar, kY, preg);
                Sub(kD, kT, accPdVar, preg);
                Sub(cPdVar, kD, kY, preg);
                Compare<float, CMPMODE::EQ>(nanMask, cPdVar, cPdVar, preg);
                Select(cPdVar, cPdVar, zeroReg, nanMask);
                Move(accPdVar, kT, preg);
            }
            DataCopy(accPdVarUb + cOff, accPdVar, preg);
            DataCopy(accPdMeanUb + cOff, accPdMean, preg);
            DataCopy(accDgammaUb + cOff, accDgamma, preg);
            DataCopy(accDbetaUb + cOff, accDbeta, preg);
            DataCopy(cDgammaUb + cOff, cDgamma, preg);
            DataCopy(cDbetaUb + cOff, cDbeta, preg);
            DataCopy(cPdVarUb + cOff, cPdVar, preg);
            DataCopy(cPdMeanUb + cOff, cPdMean, preg);
        }
    }
}

/*
 * Pass2: pd_x[m,c] = (dy*gamma)*rstd + pdVar*(2/M)*(x-mean) + pdMean*(1/M)
 * pdVar/pdMean are the FINAL per-channel reductions (fp32 UB); pd_x written as T.
 */
template <typename T>
__aicore__ inline void ComputePdx(__local_mem__ T* xUb, __local_mem__ T* dyUb, __local_mem__ T* pdxUb,
                                  __local_mem__ float* meanUb, __local_mem__ float* rstdUb,
                                  __local_mem__ float* gammaUb, __local_mem__ float* pdVarUb,
                                  __local_mem__ float* pdMeanUb, uint32_t rows, uint32_t cLen, uint32_t rowStride,
                                  float twoOverM, float oneOverM)
{
    uint16_t cLoop = (cLen + VL_FP32 - 1) / VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> meanR, rstdR, gammaR, cVarR, cMeanR;
        RegTensor<float> xR, dyR, xcR, resR;
        MaskReg preg;
        uint32_t sregC = cLen;
        for (uint16_t c = 0; c < cLoop; ++c) {
            preg = UpdateMask<float>(sregC);
            uint32_t cOff = c * VL_FP32;
            DataCopy(meanR, meanUb + cOff);
            DataCopy(rstdR, rstdUb + cOff);
            DataCopy(gammaR, gammaUb + cOff);
            DataCopy(cVarR, pdVarUb + cOff);
            DataCopy(cMeanR, pdMeanUb + cOff);
            Muls(cVarR, cVarR, twoOverM, preg);   // pdVar * (2/M)
            Muls(cMeanR, cMeanR, oneOverM, preg); // pdMean * (1/M)
            for (uint16_t m = 0; m < static_cast<uint16_t>(rows); ++m) {
                uint32_t off = m * rowStride + cOff;
                LoadTAsF32<T>(xUb, xR, preg, off);
                LoadTAsF32<T>(dyUb, dyR, preg, off);
                Sub(xcR, xR, meanR, preg);         // x - mean
                Mul(dyR, dyR, gammaR, preg);       // pd_xl = dy * gamma
                Mul(resR, dyR, rstdR, preg);       // pd_xl * rstd
                MulAddDst(resR, cVarR, xcR, preg); // + pdVar*(2/M)*(x-mean)
                Add(resR, resR, cMeanR, preg);     // + pdMean*(1/M)
                StoreF32AsT<T>(pdxUb, resR, preg, off);
            }
        }
    }
}

// Cast a length-cLen fp32 buffer to T (used for writing dgamma/dbeta directly when N == 1).
template <typename T>
__aicore__ inline void CastF32ToT(__local_mem__ float* srcUb, __local_mem__ T* dstUb, uint32_t cLen)
{
    uint16_t loopCnt = (cLen + VL_FP32 - 1) / VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> v;
        MaskReg preg;
        uint32_t sreg = cLen;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            preg = UpdateMask<float>(sreg);
            DataCopy(v, srcUb + i * VL_FP32);
            StoreF32AsT<T>(dstUb, v, preg, i * VL_FP32);
        }
    }
}
} // namespace InstanceNormGrad
#endif // INSTANCE_NORM_GRAD_COMMON_H
