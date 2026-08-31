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
 * \file pool_2d_gather_scatter_index.h
 * \brief AvgPool/MaxPoolV3 二维池化共用的 NCHW/NHWC gather 与 scatter 索引生成接口。
 */

#ifndef POOL_UTILS_ARCH35_INDEX_POOL_2D_GATHER_SCATTER_INDEX_H_
#define POOL_UTILS_ARCH35_INDEX_POOL_2D_GATHER_SCATTER_INDEX_H_

#include <cstdint>
#include <type_traits>

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "pool_utils/pool_type_traits.h"

namespace PoolUtils {
namespace Index {

template <typename U>
__aicore__ inline void GenGatherIndexMultiBatch(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElemtsIn,
                                                uint32_t wIn, uint32_t hStride, uint32_t wStride,
                                                AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U batchElemtsOut = hFactorOut * wFactorOut;
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> v3;
        AscendC::Reg::RegTensor<U> v4;
        AscendC::Reg::RegTensor<U> v5;
        AscendC::Reg::RegTensor<U> v6;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd7;
        AscendC::Reg::RegTensor<U> vd8;
        AscendC::Reg::RegTensor<U> vd9;
        AscendC::Reg::RegTensor<U> vd10;
        AscendC::Reg::RegTensor<U> vd11;
        AscendC::Reg::RegTensor<U> vd12;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, 0);
        AscendC::Reg::Duplicate(v1, (U)wFactorOut, p0);
        AscendC::Reg::Duplicate(v2, (U)wIn, p0);
        AscendC::Reg::Duplicate(v3, (U)hStride, p0);
        AscendC::Reg::Duplicate(v4, (U)wStride, p0);
        AscendC::Reg::Duplicate(v5, (U)batchElemtsIn, p0);
        AscendC::Reg::Duplicate(v6, (U)batchElemtsOut, p0);

        AscendC::Reg::Div(vd1, v0, v6, p0);  // i / (rows * cols)
        AscendC::Reg::Mul(vd2, vd1, v5, p0); // i / (rows * cols) * batchElemtsIn
        AscendC::Reg::Mul(vd3, vd1, v6, p0); // (i / wFactorOut * wIn * hStride)
        AscendC::Reg::Sub(vd4, v0, vd3, p0); // i % (rows * cols)

        AscendC::Reg::Div(vd5, vd4, v1, p0);    // hwoffset / cols
        AscendC::Reg::Mul(vd6, vd5, v2, p0);    // hwoffset / cols * wIn
        AscendC::Reg::Mul(vd7, vd6, v3, p0);    // hwoffset / cols * wIn * hStride
        AscendC::Reg::Mul(vd8, vd5, v1, p0);    // hwoffset / cols * cols
        AscendC::Reg::Sub(vd9, vd4, vd8, p0);   // hwoffset % cols
        AscendC::Reg::Mul(vd10, vd9, v4, p0);   // hwoffset % cols * wStride
        AscendC::Reg::Add(vd11, vd7, vd10, p0); // hwoffset / cols * wIn * hStride + hwoffset % cols * wStride
        AscendC::Reg::Add(vd12, vd2, vd11, p0);
        AscendC::Reg::StoreAlign(dstAddr, vd12, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexMultiRow(uint32_t wFactorOut, uint32_t wIn, uint32_t hStride, uint32_t wStride,
                                              AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> v3;
        AscendC::Reg::RegTensor<U> v4;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd7;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, 0);
        AscendC::Reg::Duplicate(v1, (U)wFactorOut, p0);
        AscendC::Reg::Duplicate(v2, (U)wIn, p0);
        AscendC::Reg::Duplicate(v3, (U)hStride, p0);
        AscendC::Reg::Duplicate(v4, (U)wStride, p0);

        AscendC::Reg::Div(vd1, v0, v1, p0);   // i / wFactorOut
        AscendC::Reg::Mul(vd2, vd1, v2, p0);  // (i / wFactorOut * wIn)
        AscendC::Reg::Mul(vd3, vd2, v3, p0);  // (i / wFactorOut * wIn * hStride)
        AscendC::Reg::Mul(vd4, vd1, v1, p0);  // (i / wFactorOut * wFactorOut)
        AscendC::Reg::Sub(vd5, v0, vd4, p0);  // i % wFactor
        AscendC::Reg::Mul(vd6, vd5, v4, p0);  // i % wFactorOut * wStride
        AscendC::Reg::Add(vd7, vd3, vd6, p0); // (i / wFactorOut * wIn * hStride + i % wFactorOut * wStride)
        AscendC::Reg::StoreAlign(dstAddr, vd7, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexSingleRow(uint32_t wStride, AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    // i * wStride
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, 0);
        AscendC::Reg::Duplicate(v1, (U)wStride, p0);
        AscendC::Reg::Mul(vd0, v0, v1, p0); // (i / wFactorOut * wIn)
        AscendC::Reg::StoreAlign(dstAddr, vd0, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexSingleKernel(uint32_t wIn, uint32_t kW, uint32_t kH,
                                                  AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    uint16_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t loopNum = (kW * kH + repeatNum - 1) / repeatNum;
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, i * repeatNum);
            AscendC::Reg::Duplicate(v1, (U)kW, p0);
            AscendC::Reg::Duplicate(v2, (U)wIn, p0);

            AscendC::Reg::Div(vd1, v0, v1, p0);
            AscendC::Reg::Mul(vd2, vd1, v2, p0);
            AscendC::Reg::Mul(vd3, vd1, v1, p0);
            AscendC::Reg::Sub(vd4, v0, vd3, p0);
            AscendC::Reg::Add(vd5, vd2, vd4, p0);
            AscendC::Reg::StoreAlign(dstAddr + i * repeatNum, vd5, p0);
        }
    }
}

template <typename U, bool SingleRow>
__aicore__ inline void GenScatterIndex(uint32_t wIn, uint32_t wInDst, AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;

        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, 0);
        if constexpr (SingleRow) {
            AscendC::Reg::StoreAlign(dstAddr, v0, p0);
        } else {
            AscendC::Reg::Duplicate(v1, (U)wIn, p0);
            AscendC::Reg::Duplicate(v2, (U)wInDst, p0);

            AscendC::Reg::Div(vd1, v0, v1, p0);
            AscendC::Reg::Mul(vd2, vd1, v2, p0);
            AscendC::Reg::Mul(vd3, vd1, v1, p0);
            AscendC::Reg::Sub(vd4, v0, vd3, p0);
            AscendC::Reg::Add(vd5, vd2, vd4, p0);
            AscendC::Reg::StoreAlign(dstAddr, vd5, p0);
        }
    }
}

template <typename U, bool SingleRow>
__aicore__ inline void NHWCGenScatterIndex(uint32_t wIn, uint32_t wInDstElms, uint32_t channels,
                                           AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> v3;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd7;
        AscendC::Reg::RegTensor<U> vd8;
        AscendC::Reg::RegTensor<U> vd9;
        AscendC::Reg::RegTensor<U> vd10;

        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, 0);
        if constexpr (SingleRow) {
            AscendC::Reg::StoreAlign(dstAddr, v0, p0);
        } else {
            AscendC::Reg::Duplicate(v1, (U)wIn, p0);
            AscendC::Reg::Duplicate(v2, (U)wInDstElms, p0);
            AscendC::Reg::Duplicate(v3, (U)channels, p0);

            AscendC::Reg::Div(vd1, v0, v3, p0);  // i / channels
            AscendC::Reg::Div(vd2, vd1, v1, p0); // i / channels / win
            AscendC::Reg::Mul(vd3, vd2, v2, p0); // i / channels / win * winDst

            AscendC::Reg::Mul(vd4, vd2, v1, p0);  // i / channels / win * win
            AscendC::Reg::Sub(vd5, vd1, vd4, p0); // i / channels mod win
            AscendC::Reg::Mul(vd6, vd5, v3, p0);  // ( i / channels mod win) * channels
            AscendC::Reg::Add(vd7, vd3, vd6, p0); // i / channels / win * winDst + i / channels mod win * channels

            AscendC::Reg::Mul(vd8, vd1, v3, p0);
            AscendC::Reg::Sub(vd9, v0, vd8, p0); // i mod channels

            AscendC::Reg::Add(vd10, vd9, vd7,
                              p0); // (i / channels / win * winDst + i / channels mod win) * channels + i mod channels
            AscendC::Reg::StoreAlign(dstAddr, vd10, p0);
        }
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexSingleRow(uint32_t wStride, uint32_t channels,
                                                   AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    // i * wStride
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<regType> tmp;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;

        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, 0);
        AscendC::Reg::Duplicate(v1, (U)wStride, p0);
        AscendC::Reg::Duplicate(v2, (U)channels, p0); // channels
        AscendC::Reg::Div(vd0, v0, v2, p0);           // i / channels
        AscendC::Reg::Mul(vd1, vd0, v2, p0);
        AscendC::Reg::Sub(vd5, v0, vd1, p0);  // i % channel
        AscendC::Reg::Mul(vd2, vd0, v1, p0);  // (i / channel * wstride)
        AscendC::Reg::Mul(vd3, vd2, v2, p0);  // (i / channel * wstride * channels)
        AscendC::Reg::Add(vd4, vd3, vd5, p0); // (i / channel * wstride * channels) + i % channel
        AscendC::Reg::StoreAlign(dstAddr, vd4, p0);
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexMultiRow(uint32_t wFactorOut, uint32_t wInElms, uint32_t hStride,
                                                  uint32_t wStride, uint32_t channels,
                                                  AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> v3;
        AscendC::Reg::RegTensor<U> v4;
        AscendC::Reg::RegTensor<U> v5;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd7;
        AscendC::Reg::RegTensor<U> vd8;
        AscendC::Reg::RegTensor<U> vd9;
        AscendC::Reg::RegTensor<U> vd10;
        AscendC::Reg::RegTensor<U> vd11;
        AscendC::Reg::RegTensor<U> vd12;
        AscendC::Reg::RegTensor<U> vd13;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, 0);
        AscendC::Reg::Duplicate(v1, (U)wFactorOut, p0);
        AscendC::Reg::Duplicate(v2, (U)wInElms, p0);
        AscendC::Reg::Duplicate(v3, (U)hStride, p0);
        AscendC::Reg::Duplicate(v4, (U)wStride, p0);
        AscendC::Reg::Duplicate(v5, (U)channels, p0);

        AscendC::Reg::Div(vd1, v0, v5, p0);  // i / channels
        AscendC::Reg::Div(vd2, vd1, v1, p0); // i / channels / wFactorOut
        AscendC::Reg::Mul(vd3, vd2, v2, p0); // (i  / channels / wFactorOut * wIn)
        AscendC::Reg::Mul(vd4, vd3, v3, p0); // (i / channels / wFactorOut * wIn * hStride

        AscendC::Reg::Mul(vd5, vd2, v1, p0);  // (i / channels / wFactorOut * wFactorOut)
        AscendC::Reg::Sub(vd6, vd1, vd5, p0); // (i  / channels) % wFactor
        AscendC::Reg::Mul(vd7, vd6, v4, p0);  // (i  / channels) % wFactorOut * wStride
        AscendC::Reg::Mul(vd8, vd7, v5, p0);  // ( i  / channels) % wFactorOut * wStride) * channels

        AscendC::Reg::Add(
            vd9, vd8, vd4,
            p0); // (i  / channels) / wFactorOut * wIn * hStride + (i  / channels) % wFactorOut * wStride* channels)
        AscendC::Reg::Mul(vd11, vd1, v5, p0);  // i / channels * channels
        AscendC::Reg::Sub(vd12, v0, vd11, p0); // i mod channel
        AscendC::Reg::Add(vd13, vd9, vd12, p0);
        AscendC::Reg::StoreAlign(dstAddr, vd13, p0);
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexMultiBatch(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t hIn,
                                                    uint32_t wInElms, uint32_t hStride, uint32_t wStride,
                                                    uint32_t channels, AscendC::LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U batchElemtsIn = hIn * wInElms;
    U batchElemtsOut = hFactorOut * wFactorOut * channels;
    __VEC_SCOPE__
    {
        using regType = typename PoolUtils::TypeTraits::VciTypeGet<U>::type;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> v3;
        AscendC::Reg::RegTensor<U> v4;
        AscendC::Reg::RegTensor<U> v5;
        AscendC::Reg::RegTensor<U> v6;
        AscendC::Reg::RegTensor<U> v7;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd8;
        AscendC::Reg::RegTensor<U> vd12;
        AscendC::Reg::RegTensor<U> vd14;
        AscendC::Reg::RegTensor<U> vd17;
        AscendC::Reg::RegTensor<U> vd18;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Arange((AscendC::Reg::RegTensor<regType>&)v0, 0);
        AscendC::Reg::Duplicate(v1, (U)wFactorOut, p0);
        AscendC::Reg::Duplicate(v2, (U)wInElms, p0);
        AscendC::Reg::Duplicate(v3, (U)hStride, p0);
        AscendC::Reg::Duplicate(v4, (U)wStride, p0);
        AscendC::Reg::Duplicate(v5, (U)channels, p0);
        AscendC::Reg::Duplicate(v6, (U)batchElemtsIn, p0);
        AscendC::Reg::Duplicate(v7, (U)batchElemtsOut, p0);

        AscendC::Reg::Div(vd1, v0, v7, p0);  // i / (rows * cols * channels)
        AscendC::Reg::Mul(vd2, vd1, v6, p0); // i / (rows * cols * channels) * batchElemtsIn       n

        AscendC::Reg::Mul(vd4, vd1, v7, p0); // (i / (rows * cols * channels) * (rows * cols * channels)
        AscendC::Reg::Sub(vd4, v0, vd4, p0); // i % (rows * cols *channels)

        AscendC::Reg::Div(vd5, vd4, v5, p0); // hwoffset / channels
        AscendC::Reg::Div(vd6, vd5, v1, p0); // hwoffset / channels / wfout
        AscendC::Reg::Mul(vd8, vd6, v2, p0); // hwoffset / channels / wfout * win
        AscendC::Reg::Mul(vd8, vd8, v3, p0); // hwoffset / channels / wfout * hstride  h

        AscendC::Reg::Mul(vd12, vd6, v1, p0);   // hwoffset / channels / wfout * wfout
        AscendC::Reg::Sub(vd12, vd5, vd12, p0); // hwoffset / channels % wfout
        AscendC::Reg::Mul(vd12, vd12, v4, p0);  // hwoffset / channels % wfout * wstride
        AscendC::Reg::Mul(vd12, vd12, v5, p0);  // (hwoffset / channels % wfout * wstride) * channels

        AscendC::Reg::Add(vd14, vd12, vd8,
                          p0); // hwoffset / channels / wfout * hstride + hwoffset / channels % wfout * wstride
        AscendC::Reg::Add(vd14, vd14, vd2, p0); // (hwoffset / channels / wfout * hstride + hwoffset / channels / wfout
                                                // * wstride) * channels + i / (rows * cols * channels) * batchElemtsIn

        AscendC::Reg::Div(vd17, v0, v5, p0);   // i / channels
        AscendC::Reg::Mul(vd17, vd17, v5, p0); // i / channels * channels
        AscendC::Reg::Sub(vd17, v0, vd17, p0); // i % channels

        AscendC::Reg::Add(vd18, vd14, vd17, p0);
        AscendC::Reg::StoreAlign(dstAddr, vd18, p0);
    }
}

} // namespace Index
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_INDEX_POOL_2D_GATHER_SCATTER_INDEX_H_
