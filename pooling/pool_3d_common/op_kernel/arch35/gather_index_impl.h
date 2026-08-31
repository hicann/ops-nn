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
 * \file gather_index_impl.h
 * \brief
 */
#ifndef GATHER_INDEX_IMPL_H_
#define GATHER_INDEX_IMPL_H_

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "kernel_operator.h"
#include "pool_utils/pool_type_traits.h"

namespace GatherIndexImpl {
using namespace AscendC;

constexpr int32_t ONE = 1;
constexpr int32_t TWO = 2;
constexpr int32_t THREE = 3;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;

struct ShapeInfo {
    uint32_t inStride[FIVE] = {1};
    uint32_t gatherSize[FIVE] = {1};
    uint32_t gatherStride[FIVE] = {1};
};

template <typename U>
__aicore__ inline void GenGatherFiveDim(const ShapeInfo& param, LocalTensor<U>& indexLocal, uint16_t loopNum = 1)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    U fourDimElemtsIn = param.inStride[4];
    U fourDimElemtsOut = param.gatherSize[0] * param.gatherSize[1] * param.gatherSize[2] * param.gatherSize[3];
    U threeDimElemtsIn = param.inStride[3];
    U threeDimElemtsOut = param.gatherSize[0] * param.gatherSize[1] * param.gatherSize[2];
    U twoDimElemtsIn = param.inStride[2];
    U twoDimElemtsOut = param.gatherSize[0] * param.gatherSize[1];
    constexpr U oneRegLength = platform::GetVRegSize() / sizeof(U);
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;
        Reg::RegTensor<U> v4;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd7;
        Reg::RegTensor<U> vd8;
        Reg::RegTensor<U> vd9;
        Reg::RegTensor<U> vd10;
        Reg::RegTensor<U> vd11;
        Reg::RegTensor<U> vd12;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        Reg::Duplicate(v1, (U)param.gatherSize[0], p0);
        Reg::Duplicate(v2, (U)twoDimElemtsOut, p0);
        Reg::Duplicate(v3, (U)threeDimElemtsOut, p0);
        Reg::Duplicate(v4, (U)fourDimElemtsOut, p0);

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange((Reg::RegTensor<regType>&)v0, i * oneRegLength);
            Reg::Div(vd1, v0, v4, p0); // i / (dim3 * dim2 * dim1 * dim0)
            Reg::Muls(vd2, vd1, (U)fourDimElemtsIn,
                      p0); // i / (dim3 * dim2 * dim1 * dim0) * (dim3in * dim2in * dim1in * dim0in)
            Reg::Muls(vd2, vd2, (U)param.gatherStride[4], p0);
            Reg::Mul(vd3, vd1, v4, p0); // i / dim3 * dim2 * dim1 * dim0) * (dim3 * dim2 * dim1 * dim0)
            Reg::Sub(vd4, v0, vd3, p0); // i % (dim3 * dim2 * dim1 * dim0)

            Reg::Div(vd1, vd4, v3, p0);                   // i / (dim2 * dim1 * dim0)
            Reg::Muls(vd3, vd1, (U)threeDimElemtsIn, p0); // i / (dim2 * dim1 * dim0) * (dim2in * dim1in * dim0in)
            Reg::Muls(vd3, vd3, (U)param.gatherStride[3], p0);
            Reg::Mul(vd1, vd1, v3, p0);  // i / dim2 * dim1 * dim0) * (dim2 * dim1 * dim0)
            Reg::Sub(vd4, vd4, vd1, p0); // i % (dim2 * dim1 * dim0)

            Reg::Div(vd5, vd4, v2, p0);                 // i / ( dim1 * dim0)
            Reg::Muls(vd6, vd5, (U)twoDimElemtsIn, p0); // i / (dim1 * dim0) * (dim1in * dim0in)
            Reg::Muls(vd6, vd6, (U)param.gatherStride[2], p0);
            Reg::Mul(vd7, vd5, v2, p0);  // i / dim1 * dim0) * (dim1 * dim0)
            Reg::Sub(vd8, vd4, vd7, p0); // i % (dim1 * dim0)

            Reg::Div(vd9, vd8, v1, p0);                     // i / ( dim1 * dim0)
            Reg::Muls(vd10, vd9, (U)param.inStride[1], p0); // i / (dim1 * dim0) * (dim1in * dim0in)
            Reg::Muls(vd10, vd10, (U)param.gatherStride[1], p0);
            Reg::Mul(vd11, vd9, v1, p0);   // i / dim1 * dim0) * (dim1 * dim0)
            Reg::Sub(vd12, vd8, vd11, p0); // i % (dim1 * dim0)
            Reg::Muls(vd12, vd12, (U)param.gatherStride[0], p0);

            Reg::Add(vd2, vd2, vd3, p0);
            Reg::Add(vd6, vd6, vd10, p0);
            Reg::Add(vd6, vd2, vd6, p0);
            Reg::Add(vd12, vd12, vd6, p0);
            Reg::StoreAlign(dstAddr + i * oneRegLength, vd12, p0);
        }
    }
}

template <typename U>
__aicore__ inline void GenGatherFourDim(const ShapeInfo& param, LocalTensor<U>& indexLocal, uint16_t loopNum = 1)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U threeDimElemtsIn = param.inStride[3];
    U threeDimElemtsOut = param.gatherSize[0] * param.gatherSize[1] * param.gatherSize[2];
    U twoDimElemtsIn = param.inStride[2];
    U twoDimElemtsOut = param.gatherSize[0] * param.gatherSize[1];
    constexpr U oneRegLength = platform::GetVRegSize() / sizeof(U);
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd7;
        Reg::RegTensor<U> vd8;
        Reg::RegTensor<U> vd9;
        Reg::RegTensor<U> vd10;
        Reg::RegTensor<U> vd11;
        Reg::RegTensor<U> vd12;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        Reg::Duplicate(v1, (U)param.gatherSize[0], p0);
        Reg::Duplicate(v2, (U)twoDimElemtsOut, p0);
        Reg::Duplicate(v3, (U)threeDimElemtsOut, p0);

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange((Reg::RegTensor<regType>&)v0, i * oneRegLength);
            Reg::Div(vd1, v0, v3, p0);                    // i / (dim2 * dim1 * dim0)
            Reg::Muls(vd2, vd1, (U)threeDimElemtsIn, p0); // i / (dim2 * dim1 * dim0) * (dim2in * dim1in * dim0in)
            Reg::Muls(vd2, vd2, (U)param.gatherStride[3], p0);
            Reg::Mul(vd3, vd1, v3, p0); // i / dim2 * dim1 * dim0) * (dim2 * dim1 * dim0)
            Reg::Sub(vd4, v0, vd3, p0); // i % (dim2 * dim1 * dim0)

            Reg::Div(vd5, vd4, v2, p0);                 // i / ( dim1 * dim0)
            Reg::Muls(vd6, vd5, (U)twoDimElemtsIn, p0); // i / (dim1 * dim0) * (dim1in * dim0in)
            Reg::Muls(vd6, vd6, (U)param.gatherStride[2], p0);
            Reg::Mul(vd7, vd5, v2, p0);  // i / dim1 * dim0) * (dim1 * dim0)
            Reg::Sub(vd8, vd4, vd7, p0); // i % (dim1 * dim0)

            Reg::Div(vd9, vd8, v1, p0);                     // i / ( dim1 * dim0)
            Reg::Muls(vd10, vd9, (U)param.inStride[1], p0); // i / (dim1 * dim0) * (dim1in * dim0in)
            Reg::Muls(vd10, vd10, (U)param.gatherStride[1], p0);
            Reg::Mul(vd11, vd9, v1, p0);   // i / dim1 * dim0) * (dim1 * dim0)
            Reg::Sub(vd12, vd8, vd11, p0); // i % (dim1 * dim0)
            Reg::Muls(vd12, vd12, (U)param.gatherStride[0], p0);

            Reg::Add(vd2, vd2, vd6, p0);
            Reg::Add(vd12, vd12, vd10, p0);
            Reg::Add(vd12, vd12, vd2, p0);
            Reg::StoreAlign(dstAddr + i * oneRegLength, vd12, p0);
        }
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexThreeDim(const ShapeInfo& param, LocalTensor<U>& indexLocal, uint16_t loopNum = 1)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U twoDimElemtsIn = param.inStride[2];
    U twoDimElemtsOut = param.gatherSize[0] * param.gatherSize[1];
    constexpr U oneRegLength = platform::GetVRegSize() / sizeof(U);
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd7;
        Reg::RegTensor<U> vd8;
        Reg::RegTensor<U> vd9;
        Reg::RegTensor<U> vd10;
        Reg::RegTensor<U> vd11;
        Reg::RegTensor<U> vd12;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        Reg::Duplicate(v1, (U)param.gatherSize[0], p0);
        Reg::Duplicate(v2, (U)twoDimElemtsOut, p0);

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange((Reg::RegTensor<regType>&)v0, i * oneRegLength);
            Reg::Div(vd1, v0, v2, p0);                  // i / (rows * cols)
            Reg::Muls(vd2, vd1, (U)twoDimElemtsIn, p0); // i / (rows * cols) * batchElemtsIn
            Reg::Muls(vd2, vd2, (U)param.gatherStride[2], p0);
            Reg::Mul(vd3, vd1, v2, p0); // (i / wFactorOut * wIn * hStride)
            Reg::Sub(vd4, v0, vd3, p0); // i % (rows * cols)

            Reg::Div(vd5, vd4, v1, p0);                         // hwoffset / cols
            Reg::Muls(vd6, vd5, (U)param.inStride[1], p0);      // hwoffset / cols * wIn
            Reg::Muls(vd7, vd6, (U)param.gatherStride[1], p0);  // hwoffset / cols * wIn * hStride
            Reg::Mul(vd8, vd5, v1, p0);                         // hwoffset / cols * cols
            Reg::Sub(vd9, vd4, vd8, p0);                        // hwoffset % cols
            Reg::Muls(vd10, vd9, (U)param.gatherStride[0], p0); // hwoffset % cols * wStride
            Reg::Add(vd11, vd7, vd10, p0); // hwoffset / cols * wIn * hStride + hwoffset % cols * wStride
            Reg::Add(vd12, vd2, vd11, p0);
            Reg::StoreAlign(dstAddr + i * oneRegLength, vd12, p0);
        }
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexTwoDim(uint32_t wFactorOut, uint32_t wIn, uint32_t hStride, uint32_t wStride,
                                            LocalTensor<U>& indexLocal, uint16_t loopNum = 1)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    constexpr U oneRegLength = platform::GetVRegSize() / sizeof(U);
    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;
        Reg::RegTensor<U> v4;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd7;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        Reg::Duplicate(v1, (U)wFactorOut, p0);
        Reg::Duplicate(v2, (U)wIn, p0);
        Reg::Duplicate(v3, (U)hStride, p0);
        Reg::Duplicate(v4, (U)wStride, p0);

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange((Reg::RegTensor<regType>&)v0, i * oneRegLength);
            Reg::Div(vd1, v0, v1, p0);   // i / wFactorOut
            Reg::Mul(vd2, vd1, v2, p0);  // (i / wFactorOut * wIn)
            Reg::Mul(vd3, vd2, v3, p0);  // (i / wFactorOut * wIn * hStride)
            Reg::Mul(vd4, vd1, v1, p0);  // (i / wFactorOut * wFactorOut)
            Reg::Sub(vd5, v0, vd4, p0);  // i % wFactor
            Reg::Mul(vd6, vd5, v4, p0);  // i % wFactorOut * wStride
            Reg::Add(vd7, vd3, vd6, p0); // (i / wFactorOut * wIn * hStride + i % wFactorOut * wStride)
            Reg::StoreAlign(dstAddr + i * oneRegLength, vd7, p0);
        }
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexOneDim(uint32_t wStride, LocalTensor<U>& indexLocal, uint16_t loopNum = 1)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    constexpr U oneRegLength = platform::GetVRegSize() / sizeof(U);
    // i * wStride
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;

        Reg::RegTensor<U> vd0;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange((Reg::RegTensor<regType>&)v0, i * oneRegLength);
            Reg::Muls(vd0, v0, (U)wStride, p0); // (i / wFactorOut * wIn)
            Reg::StoreAlign(dstAddr + i * oneRegLength, vd0, p0);
        }
    }
}

template <typename U, uint8_t DIM>
__aicore__ inline void GenGatherIndex(const ShapeInfo& param, LocalTensor<U>& indexLocal, uint16_t loopNum = 1)
{
    if constexpr (DIM == ONE) {
        GenGatherIndexOneDim(param.gatherStride[0], indexLocal, loopNum);
    } else if constexpr (DIM == TWO) {
        GenGatherIndexTwoDim(param.gatherSize[0], param.inStride[1], param.gatherStride[1], param.gatherStride[0],
                             indexLocal, loopNum);
    } else if constexpr (DIM == THREE) {
        GenGatherIndexThreeDim(param, indexLocal, loopNum);
    } else if constexpr (DIM == FOUR) {
        GenGatherFourDim(param, indexLocal, loopNum);
    } else if constexpr (DIM == FIVE) {
        GenGatherFiveDim(param, indexLocal, loopNum);
    }
}

/**
#include "gather_index_impl.h"
// input [a, b, c]  gather_size [5, 2, 4] stride [as, bs, cs]
GatherIndexImpl::ShapeInfo info = {
    {1, c, b*c, a*b*c, a*b*c},     //inStride
    {4, 2, 5, 1, 1},    // gather size
    {cs, bs, as, 1, 1}   //stride
};
GatherIndexImpl::GenGatherIndex(info, indexLocal);

**/

} // namespace GatherIndexImpl
#endif // GATHER_INDEX_IMPL_H_
