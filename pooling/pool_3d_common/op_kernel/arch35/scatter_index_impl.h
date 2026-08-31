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
 * \file scatter_index_impl.h
 * \brief
 */

#ifndef _SCATTER_INDEX_IMPL_H_
#define _SCATTER_INDEX_IMPL_H_

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "kernel_operator.h"
#include "pool_utils/pool_type_traits.h"

namespace ScatterIndexImpl {
using namespace AscendC;

constexpr int32_t ONE = 1;
constexpr int32_t TWO = 2;
constexpr int32_t THREE = 3;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;

const int64_t DIM0 = 0; // 第一个非连续轴从这里开始
const int64_t DIM1 = 1;
const int64_t DIM2 = 2;
const int64_t DIM3 = 3;
const int64_t DIM4 = 4;

struct ScatterShapeInfo {
    uint32_t inSize[FIVE] = {1};
    uint32_t dstStride[FIVE] = {1};
};

template <typename U>
__aicore__ inline void GenScatterIndexOneDim(const ScatterShapeInfo& params, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::StoreAlign(dstAddr, v0, p0);
    }
}

template <typename U>
__aicore__ inline void GenScatterIndexTwoDim(const ScatterShapeInfo& params, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
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

        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, (U)params.inSize[DIM0], p0);

        Reg::Div(vd1, v0, v1, p0);                          // i / wIn
        Reg::Muls(vd2, vd1, (U)params.dstStride[DIM1], p0); // i / wIn * winDst
        Reg::Mul(vd3, vd1, v1, p0);                         // i / wIn * win
        Reg::Sub(vd4, v0, vd3, p0);                         // i % win
        Reg::Add(vd4, vd4, vd2, p0);

        Reg::StoreAlign(dstAddr, vd4, p0);
    }
}

template <typename U>
__aicore__ inline void GenScatterIndexThreeDim(const ScatterShapeInfo& params, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    U twoDimsElmsIn = params.inSize[DIM0] * params.inSize[DIM1];
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

        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, twoDimsElmsIn, p0);
        Reg::Duplicate(v2, (U)params.inSize[DIM0], p0);

        Reg::Div(vd1, v0, v1, p0);                          // i / (dim1 * dim2)
        Reg::Muls(vd2, vd1, (U)params.dstStride[DIM2], p0); // i / dim1 / dim2 * dstdim2size

        Reg::Mul(vd3, vd1, v1, p0);                         // i / dim1 / dim2 * dim2
        Reg::Sub(vd4, v0, vd3, p0);                         // i mod dim1 * dim2
        Reg::Div(vd5, vd4, v2, p0);                         // i mod dim1 * dim2 / dim1
        Reg::Muls(vd6, vd5, (U)params.dstStride[DIM1], p0); // i mod dim1 * dim2 / dim1 * dim1

        Reg::Div(vd7, v0, v2, p0); // i / (dim1 )
        Reg::Mul(vd7, vd7, v2, p0);
        Reg::Sub(vd7, v0, vd7, p0); // i mod dim1

        Reg::Add(vd8, vd6, vd7, p0); //
        Reg::Add(vd8, vd2, vd8, p0); //
        Reg::StoreAlign(dstAddr, vd8, p0);
    }
}

template <typename U>
__aicore__ inline void GenScatterIndexFourDim(const ScatterShapeInfo& params, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    U twoDimsElmsIn = params.inSize[DIM0] * params.inSize[DIM1];
    U threeDimsElmsIn = params.inSize[DIM0] * params.inSize[DIM1] * params.inSize[DIM2];
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

        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, threeDimsElmsIn, p0);
        Reg::Duplicate(v2, twoDimsElmsIn, p0);
        Reg::Duplicate(v3, (U)params.inSize[DIM0], p0);

        Reg::Div(vd1, v0, v1, p0);                       // i / (3dimInSize)
        Reg::Muls(vd2, vd1, params.dstStride[DIM3], p0); // i / (dim3InSize) * dstdim3size

        Reg::Mul(vd3, vd1, v1, p0);                      // i / (dim3InSize) * dstdim3size
        Reg::Sub(vd4, v0, vd3, p0);                      // i mod (dim3InSize)
        Reg::Div(vd3, vd4, v2, p0);                      // i / (2dimInSize )
        Reg::Muls(vd5, vd3, params.dstStride[DIM2], p0); // i / (2dimInSize )* dstdim2size

        Reg::Mul(vd6, vd3, v2, p0);                      // i / (2dimInSize )* dstdim2size
        Reg::Sub(vd3, vd4, vd6, p0);                     // i mod dim2
        Reg::Div(vd6, vd3, v3, p0);                      // i / (dim1InSize )
        Reg::Muls(vd6, vd6, params.dstStride[DIM1], p0); // i / (dim1InSize ) * dim1InSize

        Reg::Div(vd7, v0, v3, p0); // i / (dim1 )
        Reg::Mul(vd7, vd7, v3, p0);
        Reg::Sub(vd7, v0, vd7, p0); // i mod dim1

        Reg::Add(vd2, vd2, vd5, p0);
        Reg::Add(vd6, vd7, vd6, p0);
        Reg::Add(vd2, vd2, vd6, p0);
        Reg::StoreAlign(dstAddr, vd2, p0);
    }
}

template <typename U>
__aicore__ inline void GenScatterIndexFiveDim(const ScatterShapeInfo& params, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    U twoDimsElmsIn = params.inSize[DIM0] * params.inSize[DIM1];
    U threeDimsElmsIn = params.inSize[DIM0] * params.inSize[DIM1] * params.inSize[DIM2];
    U fourDimsElmsIn = params.inSize[DIM0] * params.inSize[DIM1] * params.inSize[DIM2] * params.inSize[DIM3];
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

        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, fourDimsElmsIn, p0);
        Reg::Duplicate(v2, threeDimsElmsIn, p0);
        Reg::Duplicate(v3, twoDimsElmsIn, p0);
        Reg::Duplicate(v4, (U)params.inSize[DIM0], p0);

        Reg::Div(vd1, v0, v1, p0);                       // i / (4dimInSize)
        Reg::Muls(vd2, vd1, params.dstStride[DIM4], p0); // i / (dim4InSize) * dstdim4size

        Reg::Mul(vd3, vd1, v1, p0);                      // i / (dim4InSize) * 4dimInSize
        Reg::Sub(vd4, v0, vd3, p0);                      // i mod (4dimInSize)
        Reg::Div(vd3, vd4, v2, p0);                      // i / (dim3InSize )
        Reg::Muls(vd5, vd3, params.dstStride[DIM3], p0); // i / (3dimInSize )* dstdim3dstsize

        Reg::Mul(vd6, vd3, v2, p0);                      // i / dim3InSize * dim3InSize
        Reg::Sub(vd6, vd4, vd6, p0);                     // i mod dim3
        Reg::Div(vd7, vd6, v3, p0);                      // i / (dim2InSize )
        Reg::Muls(vd8, vd7, params.dstStride[DIM2], p0); // i / (dim2InSize ) * dim2dstSize

        Reg::Mul(vd9, vd7, v3, p0);                      // i / dim12nSize * dim2InSize
        Reg::Sub(vd9, vd6, vd9, p0);                     // i mod dim2
        Reg::Div(vd9, vd9, v3, p0);                      // i / (dim1InSize )
        Reg::Muls(vd9, vd9, params.dstStride[DIM1], p0); // i / (dim1InSize ) * dim1dstSize

        Reg::Div(vd10, v0, v4, p0); // i / (dim1 )
        Reg::Mul(vd10, vd10, v4, p0);
        Reg::Sub(vd10, v0, vd10, p0); // i mod dim1

        Reg::Add(vd2, vd2, vd5, p0);
        Reg::Add(vd8, vd8, vd9, p0);
        Reg::Add(vd8, vd8, vd10, p0);
        Reg::Add(vd2, vd2, vd8, p0);
        Reg::StoreAlign(dstAddr, vd2, p0);
    }
}

template <typename U, uint8_t DIM>
__aicore__ inline void GenScatterIndex(const ScatterShapeInfo& params, LocalTensor<U>& indexLocal)
{
    if constexpr (DIM == ONE) {
        GenScatterIndexOneDim(params, indexLocal);
    } else if constexpr (DIM == TWO) {
        GenScatterIndexTwoDim(params, indexLocal);
    } else if constexpr (DIM == THREE) {
        GenScatterIndexThreeDim(params, indexLocal);
    } else if constexpr (DIM == FOUR) {
        GenScatterIndexFourDim(params, indexLocal);
    } else if constexpr (DIM == FIVE) {
        GenScatterIndexFiveDim(params, indexLocal);
    }
}

// eg:
// two dims
//  ScatterShapeInfo info;
//  info.inSize[DIM0] = win
//  info.dstStride[DIM1] = winDst;
//  GenScatterIndex<U, TWO>(info, indexLocal)

// three dims
// ScatterShapeInfo info;
// info.inSize[DIM0] = channels
// info.inSize[DIM1] = win
// info.dstStride[DIM1] = channels;
// info.dstStride[DIM2] = winDst;
// GenScatterIndex<U, Three>(info, indexLocal)
} // namespace ScatterIndexImpl
#endif //_SCATTER_INDEX_IMPL_H_
