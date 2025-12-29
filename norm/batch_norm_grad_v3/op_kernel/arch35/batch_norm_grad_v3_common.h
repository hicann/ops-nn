/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_v3_common.h
 * \brief
 */
#ifndef __BATCH_NORM_GRAD_V3_COMMON_H__
#define __BATCH_NORM_GRAD_V3_COMMON_H__

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"

namespace BatchNormGradV3 {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

static constexpr uint32_t ONE = 1;
static constexpr uint32_t TWO = 2;
static constexpr uint32_t THREE = 3;
static constexpr uint32_t FOUR = 4;
static constexpr uint32_t ONE_BLK_SIZE = platform::GetUbBlockSize();
static constexpr uint32_t TWO_BLK_SIZE = ONE_BLK_SIZE * TWO;
static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
static constexpr uint64_t MOVE_ALIGN_LOOP_SIZE_SHIFT = 21;
static constexpr uint64_t MOVE_ALIGN_LOOP_STRIDE_SHIFT = 40;
static constexpr uint64_t MOVE_ALIGN_LOOP_ONCE = 2097153ULL;
static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

__aicore__ inline uint32_t RoundUpOneBlock(uint32_t x)
{
    return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
}

__aicore__ inline uint32_t RoundUpTwoBlock(uint32_t x)
{
    return (x + TWO_BLK_SIZE - 1) / TWO_BLK_SIZE * TWO_BLK_SIZE;
}

static constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void LoadOneTensor(
    const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst, MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, (__local_mem__ half*)(input) + offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xBf16, (__local_mem__ bfloat16_t*)(input) + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, (__local_mem__ float*)(input) + offset);
    }
}

template <typename T>
__aicore__ inline void LoadsTensorForDtypeT(
    const __local_mem__ void* src, MicroAPI::RegTensor<float>& dst, MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst, (__local_mem__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_BRC_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void StoreOneTensor(
    const __local_mem__ void* output, MicroAPI::RegTensor<float>& src, MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>((__local_mem__ half*)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(
            (__local_mem__ bfloat16_t*)(output) + offset, xBf16, preg);
    } else {
        DataCopy((__local_mem__ float*)(output) + offset, src, preg);
    }
}

template <typename T>
__aicore__ inline void LoadOneElement(
    const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst, MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(xFp16, (__local_mem__ half*)(input) + offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(xBf16, (__local_mem__ bfloat16_t*)(input) + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dst, ((__local_mem__ float*)(input)) + offset);
    }
}

template <typename T>
__aicore__ inline void StoreOneElement(
    const __local_mem__ void* output, MicroAPI::RegTensor<float>& src, MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        DataCopy<half, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
            (__local_mem__ half*)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
            (__local_mem__ bfloat16_t*)(output) + offset, xBf16, preg);
    } else {
        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            ((__local_mem__ float*)output) + offset, src, preg);
    }
}

__aicore__ inline void LoadTwoTensorSum(
    const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst, MicroAPI::MaskReg& preg, uint32_t offset0,
    uint32_t offset1)
{
    MicroAPI::RegTensor<float> a, b;
    DataCopy(a, (__local_mem__ float*)(input) + offset0);
    DataCopy(b, (__local_mem__ float*)(input) + offset1);
    Add<float, MicroAPI::MaskMergeMode::ZEROING>(dst, a, b, preg);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, int64_t n)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = n * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, int64_t n)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = n * sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void CopyOut(
    const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, int64_t rowSize, int64_t colSize,
    int64_t dstStride, int64_t srcStride)
{
    // CopyOut
    DataCopyExtParams params;
    params.blockCount = rowSize;
    params.blockLen = colSize * sizeof(T);
    params.dstStride = dstStride * sizeof(T) - params.blockLen;
    params.srcStride = (srcStride * sizeof(T) - RoundUpOneBlock(params.blockLen)) / ONE_BLK_SIZE;
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void CopyIn(
    const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, int64_t rowSize, int64_t colSize,
    int64_t dstStride, int64_t srcStride)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = rowSize;
    params.blockLen = colSize * sizeof(T);
    params.srcStride = srcStride * sizeof(T) - params.blockLen;
    params.dstStride = (dstStride * sizeof(T) - RoundUpOneBlock(params.blockLen)) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

__aicore__ inline int64_t GetCacheID(const int64_t idx)
{
    return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
}

constexpr float POS_INF = 3.40282366920938E+38;
constexpr float SCALAR1 = -0.5;
constexpr float SCALAR2 = 1.5;
constexpr float SCALAR3 = 0.5;
constexpr float SCALAR0 = -99.99;

__aicore__ inline void CalRstdByHighPrecision(RegTensor<float>& var, RegTensor<float>& rstd, const float epsilon)
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
    Duplicate(scalarZero, float(0.0), pregMerge);
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
    CompareScalar(cmpRegZero, var, POS_INF, pregMerge);
    Select(rstd, scalarZero, rstd, cmpRegZero);
    CompareScalar(cmpRegInf, var, float(0.0), pregMerge);
    Select(rstd, scalarInf, rstd, cmpRegInf);
}

} // namespace BatchNormGradV3
#endif
