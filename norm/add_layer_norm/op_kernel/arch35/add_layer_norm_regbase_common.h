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
#include "../../norm_common/reduce_common_regbase.h"

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
constexpr int32_t VL_FP32 = VECTOR_REG_WIDTH / sizeof(float);

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

} // namespace AddLayerNorm
#endif // ADD_LAYER_NORM_REGBASE_COMMON_H
