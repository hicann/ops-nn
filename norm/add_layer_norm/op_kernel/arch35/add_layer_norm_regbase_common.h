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
 * \file add_layer_norm_regbase_common.cpp
 * \brief
 */

#ifndef ADD_LAYER_NORM_REGBASE_COMMON_H
#define ADD_LAYER_NORM_REGBASE_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AddLayerNorm {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

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

constexpr int64_t TWO = 2;
constexpr int32_t BINARY_ADD_COEFF = 2;
constexpr int32_t VL_FP32 = VECTOR_REG_WIDTH / sizeof(float);
constexpr float SCALAR1 = -0.5f;
constexpr float SCALAR2 = 1.5f;
constexpr float SCALAR3 = 0.5f;
constexpr float POS_INF = 3.40282366920938E+38;

#define IS_BIAS_ELEWISE ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)

__aicore__ inline int32_t CEIL_DIV(int32_t x, int32_t y)
{
    return (y > 0) ? (x + y - 1) / y : 0;
}

__aicore__ inline uint32_t BLOCK_ALIGN(uint32_t x, uint32_t blockSize)
{
    return (blockSize > 0) ? (x + blockSize - 1) / blockSize * blockSize : 0;
}

template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE>
__aicore__ inline void LoadInputsToRegWithBiasElewise(
    __local_mem__ X1_TYPE* x1Addr, __local_mem__ X2_TYPE* x2Addr, __local_mem__ BIAS_TYPE* biasAddr,
    RegTensor<float>& dst, MaskReg& preg, uint32_t offset0, uint32_t offset1, uint32_t offset2)
{
    RegTensor<float> x1Fp32, x2Fp32, biasFp32;

    if constexpr (IsSameType<X2_TYPE, float>::value) {
        DataCopy(x2Fp32, (__local_mem__ X2_TYPE*)x2Addr + offset1);
    } else {
        RegTensor<X2_TYPE> x2;
        DataCopy<X2_TYPE, LoadDist::DIST_UNPACK_B16>(x2, (__local_mem__ X2_TYPE*)x2Addr + offset1);
        Cast<float, X2_TYPE, castTraitB162B32>(x2Fp32, x2, preg);
    }

    if constexpr (IsSameType<BIAS_TYPE, float>::value) {
        DataCopy(biasFp32, (__local_mem__ BIAS_TYPE*)biasAddr + offset2);
    } else {
        RegTensor<BIAS_TYPE> bias;
        DataCopy<BIAS_TYPE, LoadDist::DIST_UNPACK_B16>(bias, (__local_mem__ BIAS_TYPE*)biasAddr + offset2);
        Cast<float, BIAS_TYPE, castTraitB162B32>(biasFp32, bias, preg);
    }
    Add<float>(dst, x2Fp32, biasFp32, preg);

    if constexpr (IsSameType<X1_TYPE, float>::value) {
        DataCopy(x1Fp32, (__local_mem__ X1_TYPE*)x1Addr + offset0);
    } else {
        RegTensor<X1_TYPE> x1;
        DataCopy<X1_TYPE, LoadDist::DIST_UNPACK_B16>(x1, (__local_mem__ X1_TYPE*)x1Addr + offset0);
        Cast<float, X1_TYPE, castTraitB162B32>(x1Fp32, x1, preg);
    }
    Add<float>(dst, dst, x1Fp32, preg);
}

template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE>
__aicore__ inline void LoadInputsToRegWithBiasBroadcast(
    __local_mem__ X1_TYPE* x1Addr, __local_mem__ X2_TYPE* x2Addr, __local_mem__ BIAS_TYPE* biasAddr,
    RegTensor<float>& dst, MaskReg& preg, uint32_t offset0, uint32_t offset1, uint32_t offset2)
{
    RegTensor<float> x1Fp32, x2Fp32, biasFp32;

    if constexpr (IsSameType<X2_TYPE, float>::value) {
        DataCopy(x2Fp32, (__local_mem__ X2_TYPE*)x2Addr + offset1);
    } else {
        RegTensor<X2_TYPE> x2;
        DataCopy<X2_TYPE, LoadDist::DIST_UNPACK_B16>(x2, (__local_mem__ X2_TYPE*)x2Addr + offset1);
        Cast<float, X2_TYPE, castTraitB162B32>(x2Fp32, x2, preg);
    }

    if constexpr (IsSameType<BIAS_TYPE, float>::value) {
        DataCopy(biasFp32, (__local_mem__ BIAS_TYPE*)biasAddr + offset2);
    } else {
        RegTensor<BIAS_TYPE> bias;
        DataCopy<BIAS_TYPE, LoadDist::DIST_UNPACK_B16>(bias, (__local_mem__ BIAS_TYPE*)biasAddr + offset2);
        Cast<float, BIAS_TYPE, castTraitB162B32>(biasFp32, bias, preg);
    }
    Add<float>(dst, x2Fp32, biasFp32, preg);

    if constexpr (IsSameType<X1_TYPE, float>::value) {
        DataCopy(x1Fp32, (__local_mem__ X1_TYPE*)x1Addr + offset0);
    } else {
        RegTensor<X1_TYPE> x1;
        DataCopy<X1_TYPE, LoadDist::DIST_UNPACK_B16>(x1, (__local_mem__ X1_TYPE*)x1Addr + offset0);
        Cast<float, X1_TYPE, castTraitB162B32>(x1Fp32, x1, preg);
    }
    Add<float>(dst, dst, x1Fp32, preg);
}

template <typename X1_TYPE, typename X2_TYPE>
__aicore__ inline void LoadInputsToRegWithBiasNone(
    __local_mem__ X1_TYPE* x1Addr, __local_mem__ X2_TYPE* x2Addr, RegTensor<float>& dst, MaskReg& preg,
    uint32_t offset0, uint32_t offset1)
{
    RegTensor<float> x1Fp32, x2Fp32;

    if constexpr (IsSameType<X1_TYPE, float>::value) {
        DataCopy(x1Fp32, (__local_mem__ X1_TYPE*)x1Addr + offset0);
    } else {
        RegTensor<X1_TYPE> x1;
        DataCopy<X1_TYPE, LoadDist::DIST_UNPACK_B16>(x1, (__local_mem__ X1_TYPE*)x1Addr + offset0);
        Cast<float, X1_TYPE, castTraitB162B32>(x1Fp32, x1, preg);
    }

    if constexpr (IsSameType<X2_TYPE, float>::value) {
        DataCopy(x2Fp32, (__local_mem__ X2_TYPE*)x2Addr + offset1);
    } else {
        RegTensor<X2_TYPE> x2;
        DataCopy<X2_TYPE, LoadDist::DIST_UNPACK_B16>(x2, (__local_mem__ X2_TYPE*)x2Addr + offset1);
        Cast<float, X2_TYPE, castTraitB162B32>(x2Fp32, x2, preg);
    }
    Add<float>(dst, x1Fp32, x2Fp32, preg);
}

template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE, int TILING_KEY>
__aicore__ inline void LoadInputsToReg(
    __local_mem__ X1_TYPE* x1Addr, __local_mem__ X2_TYPE* x2Addr, __local_mem__ BIAS_TYPE* biasAddr,
    RegTensor<float>& dst, MaskReg& preg, uint32_t offset0, uint32_t offset1, uint32_t offset2)
{
    if constexpr (IS_BIAS_ELEWISE) {
        LoadInputsToRegWithBiasElewise(x1Addr, x2Addr, biasAddr, dst, preg, offset0, offset1, offset2);
    } else if constexpr (IS_BIAS_BROADCAST) {
        LoadInputsToRegWithBiasBroadcast(x1Addr, x2Addr, biasAddr, dst, preg, offset0, offset1, offset2);
    } else {
        LoadInputsToRegWithBiasNone(x1Addr, x2Addr, dst, preg, offset0, offset1);
    }
}

template <typename T>
__aicore__ inline void LoadGammaBeta(
    __local_mem__ T* gammaAddr, __local_mem__ T* betaAddr, RegTensor<float>& gamma, RegTensor<float>& beta,
    MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> gammaFp16;
        RegTensor<half> betaFp16;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(gammaFp16, (__local_mem__ half*)gammaAddr + offset);
        Cast<float, half, castTraitB162B32>(gamma, gammaFp16, preg);
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(betaFp16, (__local_mem__ half*)betaAddr + offset);
        Cast<float, half, castTraitB162B32>(beta, betaFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> gammaBf16;
        RegTensor<bfloat16_t> betaBf16;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gammaBf16, (__local_mem__ bfloat16_t*)gammaAddr + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(gamma, gammaBf16, preg);
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(betaBf16, (__local_mem__ bfloat16_t*)betaAddr + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(beta, betaBf16, preg);
    } else {
        DataCopy(gamma, (__local_mem__ float*)gammaAddr + offset);
        DataCopy(beta, (__local_mem__ float*)betaAddr + offset);
    }
}

template <typename T>
__aicore__ inline void StoreRegToOutput(__local_mem__ T* dstAddr, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> dst;
        Cast<half, float, castTraitB322B16>(dst, src, preg);
        DataCopy<half, StoreDist::DIST_PACK_B32>((__local_mem__ half*)dstAddr + offset, dst, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> dst;
        Cast<bfloat16_t, float, castTraitB322B16>(dst, src, preg);
        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>((__local_mem__ bfloat16_t*)dstAddr + offset, dst, preg);
    } else {
        DataCopy((__local_mem__ float*)dstAddr + offset, src, preg);
    }
}

__aicore__ inline void DichotomyAdd(
    RegTensor<float>& dstReg, __local_mem__ float* src, uint16_t outerLoop, uint16_t innerLoop, uint32_t lastNum)
{
    RegTensor<float> tmpReg1;
    RegTensor<float> tmpReg2;
    RegTensor<float> tmpReg3;
    LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t k = 0; k < outerLoop; k++) {
        innerLoop = innerLoop / BINARY_ADD_COEFF;
        for (uint16_t i = 0; i < innerLoop; i++) {
            DataCopy(tmpReg1, src + i * VL_FP32);
            DataCopy(tmpReg2, src + (i + innerLoop) * VL_FP32);
            Add(tmpReg3, tmpReg1, tmpReg2, pregMain);
            DataCopy(src + i * VL_FP32, tmpReg3, pregMain);
        }
        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    }
    uint32_t sreg0 = lastNum;
    MaskReg pregLoop = UpdateMask<float>(sreg0);
    DataCopy(tmpReg3, src);
    ReduceSum(dstReg, tmpReg3, pregLoop);
}

__aicore__ inline void CalRstdByHighPrecision(RegTensor<float>& var, RegTensor<float>& rstd, float epsilon)
{
    RegTensor<float> r;
    RegTensor<float> y;
    RegTensor<float> s;
    RegTensor<float> t;
    RegTensor<float> e;
    RegTensor<float> one;
    RegTensor<float> scalar1;
    RegTensor<float> scalar2;
    RegTensor<float> scalar3;
    RegTensor<float> t1;
    RegTensor<float> t2;
    RegTensor<float> t3;
    RegTensor<float> t4;
    MaskReg cmpReg1;
    MaskReg cmpReg2;
    MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();

    Duplicate(one, float(1.0), pregMerge);
    Duplicate(scalar1, SCALAR3, pregMerge);
    Duplicate(scalar2, POS_INF, pregMerge);
    Duplicate(scalar3, float(0.0), pregMerge);
    Duplicate(t1, SCALAR2, pregMerge);
    Duplicate(s, float(1.0), pregMerge);

    Adds(var, var, epsilon, pregMerge);
    Div(r, one, var, pregMerge);
    Sqrt(y, r, pregMerge);
    Muls(t, var, SCALAR1, pregMerge);
    Mul(t, t, y, pregMerge);                // -0.5 * x * y
    Mula(t1, t, y, pregMerge);              // 1.5 + (-0.5 * x * y) * y
    Mul(rstd, y, t1, pregMerge);            // y = y * (1.5 - 0.5 * x * y)
    Muls(t3, var, float(-1.0), pregMerge);  // -1 * x
    Mula(s, t3, r, pregMerge);              // 1 + (-1) * x * r
    Muls(t4, rstd, float(-1.0), pregMerge); // (-1) * y
    Mula(r, t4, rstd, pregMerge);           // r + (-1) * y * y
    Mula(s, var, r, pregMerge);             // s + x * t
    Mul(s, s, rstd, pregMerge);             // e * y
    Mula(rstd, s, scalar1, pregMerge);      // y + y * e * 0.5
    CompareScalar(cmpReg1, var, POS_INF, pregMerge);
    Select(rstd, scalar3, rstd, cmpReg1);
    CompareScalar(cmpReg2, var, float(0.0), pregMerge);
    Select(rstd, scalar2, rstd, cmpReg2);
}
} // namespace AddLayerNorm
#endif // ADD_LAYER_NORM_REGBASE_COMMON_H
