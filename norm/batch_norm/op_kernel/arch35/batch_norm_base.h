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
 * \file batch_norm_base.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_BASE_H
#define NORM_BATCH_NORM_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace BatchNormOps
{
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

constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int32_t BUFFER_DEPTH = 1;
static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static float POS_INF = 3.40282366920938E+38;
constexpr static float zero = 0.0f;
constexpr static int64_t NDDMA_THRESHOLD = 32;
constexpr static int64_t NDDMA_SECOND_DIM = 1;
constexpr static int64_t NDDMA_THIRD_DIM = 2;
constexpr static int64_t NDDMA_DIM_NUM = 3;
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

static constexpr float SCALAR1 = -0.5;
static constexpr float SCALAR2 = 1.5;
static constexpr float SCALAR3 = 0.5;
static constexpr float SCALAR0 = -99.99;

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
    RegTensor<float> t1;
    RegTensor<float> t2;
    RegTensor<float> t3;
    RegTensor<float> t4;
    RegTensor<float> scalarInf;
    RegTensor<float> scalarZero;
    MaskReg cmpRegZero;
    MaskReg cmpRegInf;
    MaskReg pregMerge = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    Duplicate(scalarInf, POS_INF, pregMerge);
    Duplicate(scalarZero, zero, pregMerge);
    Duplicate(one, float(1.0), pregMerge);
    Duplicate(scalar1, SCALAR3, pregMerge);
    Duplicate(t1, SCALAR2, pregMerge);
    Duplicate(s, float(1.0), pregMerge);

    Adds(var, var, epsilon, pregMerge);
    // we need sqrt(1/var) = nan, when var < 0.
    // But div donot support subnormal(when var is less -1e38, 1/var will be 0), then sqrt(1/var) is 0.
    // So we do maxs to avoid the subnormal problem, sqrt(1/var) = nan
    Maxs(var, var, SCALAR0, pregMerge);
    Div(r, one, var, pregMerge);
    Sqrt(y, r, pregMerge);
    Muls(t, var, SCALAR1, pregMerge);
    Mul(t, t, y, pregMerge);                 // -0.5 * x * y
    Mula(t1, t, y, pregMerge);               // 1.5 + (-0.5 * x * y) * y
    Mul(rstd, y, t1, pregMerge);             // y = y * (1.5 - 0.5 * x * y)
    Muls(t3, var, float(-1.0), pregMerge);   // -1 * x
    Mula(s, t3, r, pregMerge);               // 1 + (-1) * x * r
    Muls(t4, rstd, float(-1.0), pregMerge);  // (-1) * y
    Mula(r, t4, rstd, pregMerge);            // r + (-1) * y * y
    Mula(s, var, r, pregMerge);              // s + x * t
    Mul(s, s, rstd, pregMerge);              // e * y
    Mula(rstd, s, scalar1, pregMerge);       // y + y * e * 0.5
    CompareScalar(cmpRegZero, var, POS_INF, pregMerge);
    Select(rstd, scalarZero, rstd, cmpRegZero);
    CompareScalar(cmpRegInf, var, zero, pregMerge);
    Select(rstd, scalarInf, rstd, cmpRegInf);
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(AscendC::MicroAPI::RegTensor<float>& dst, __local_mem__ T* src,
                                           AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
    } else {  // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeTBrc(RegTensor<float>& dst, __local_mem__ T* src, MaskReg& preg,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst, (__local_mem__ float*)src + offset);
    } else {  // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_BRC_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(RegTensor<float>& dst1, RegTensor<float>& dst2, __local_mem__ T* src1,
                                              __local_mem__ T* src2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                              uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16Q;
        RegTensor<bfloat16_t> xFp16R;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        DataCopy(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
        DataCopy(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeTBrc(RegTensor<float>& dst1, RegTensor<float>& dst2, __local_mem__ T* src1,
                                                 __local_mem__ T* src2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                                 uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
        DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16Q;
        RegTensor<bfloat16_t> xFp16R;
        DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
        DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeT(__local_mem__ T* dst, AscendC::MicroAPI::RegTensor<float>& src,
                                               AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xFp16;
        Cast<T, float, castTraitB322B16>(xFp16, src, preg);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

}  // namespace BatchNormOps

#endif // NORM_BATCH_NORM_BASE_H