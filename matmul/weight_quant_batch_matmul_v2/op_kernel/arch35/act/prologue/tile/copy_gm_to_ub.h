/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_COPY_GM_TO_UB_H
#define PROLOGUE_TILE_COPY_GM_TO_UB_H

#include "../../utils/arch.h"
#include "../../utils/tensor_traits.h"
#include "../../constant.h"
#include "act/utils/tuple_utils.h"

namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Tile::detail {
// ND 转置场景(kn)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyImpl<
    Arch::Ascend910_95, AscendC::LocalTensor<DstTrait>, AscendC::GlobalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_gm<SrcTrait> && in_ub<DstTrait> && IsColumnMajor2D<decltype(DstTrait{}.GetLayout())>::value &&
        IsColumnMajor2D<decltype(SrcTrait{}.GetLayout())>::value>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::GlobalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyPadExtParams<PrimType> padParams;
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = ::Act::Gemm::Get<1>(shape);
        intriParams.blockLen = ElemToByte<PrimType>(::Act::Gemm::Get<0>(shape));
        intriParams.srcStride = ElemToByte<PrimType>(::Act::Gemm::Get<1>(srcTensor.GetStride())) - intriParams.blockLen;
        intriParams.dstStride =
            (ElemToByte<PrimType>(::Act::Gemm::Get<1>(dstTensor.GetStride())) - intriParams.blockLen) / BYTE_PER_BLK;
        DataCopyPadGm2UBImpl(
            (__ubuf__ PrimType*)(dstTensor.GetPhyAddr()), (__gm__ PrimType*)(srcTensor.GetPhyAddr()), intriParams,
            padParams, static_cast<uint8_t>(CacheMode));
    }
};

// ND 转置场景(nk)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyImpl<
    Arch::Ascend910_95, AscendC::LocalTensor<DstTrait>, AscendC::GlobalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_gm<SrcTrait> && in_ub<DstTrait> && IsRowMajor2D<decltype(DstTrait{}.GetLayout())>::value &&
        IsRowMajor2D<decltype(SrcTrait{}.GetLayout())>::value>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::GlobalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyPadExtParams<PrimType> padParams;
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = ::Act::Gemm::Get<0>(shape);
        intriParams.blockLen = ElemToByte<PrimType>(::Act::Gemm::Get<1>(shape));
        intriParams.srcStride = ElemToByte<PrimType>(::Act::Gemm::Get<0>(srcTensor.GetStride())) - intriParams.blockLen;
        intriParams.dstStride =
            (ElemToByte<PrimType>(::Act::Gemm::Get<0>(dstTensor.GetStride())) - intriParams.blockLen) / BYTE_PER_BLK;
        DataCopyPadGm2UBImpl(
            (__ubuf__ PrimType*)(dstTensor.GetPhyAddr()), (__gm__ PrimType*)(srcTensor.GetPhyAddr()), intriParams,
            padParams, static_cast<uint8_t>(CacheMode));
    }
};

// Zn 转置场景(kn) (n1,k1,k0,n0)
// layout ((n0,n1),(k0,k1)):((_1,k1*k0*n0),(_16,_256))
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyImpl<
    Arch::Ascend910_95, AscendC::LocalTensor<DstTrait>, AscendC::GlobalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_gm<SrcTrait> && in_ub<DstTrait> && is_2d_zn<decltype(DstTrait{}.GetLayout())>::value &&
        is_2d_zn<decltype(SrcTrait{}.GetLayout())>::value>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::GlobalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyPadExtParams<PrimType> padParams;
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = AscendC::CeilDiv(::Act::Gemm::Get<0>(shape), BLOCK_CUBE);
        intriParams.blockLen =
            ElemToByte<PrimType>(AscendC::CeilAlign(::Act::Gemm::Get<1>(shape), BLOCK_CUBE) * BLOCK_CUBE);
        intriParams.srcStride =
            ElemToByte<PrimType>(::Act::Gemm::Get<0, 1>(srcTensor.GetStride())) - intriParams.blockLen;
        intriParams.dstStride =
            (ElemToByte<PrimType>(::Act::Gemm::Get<0, 1>(dstTensor.GetStride())) - intriParams.blockLen) / BYTE_PER_BLK;
        DataCopyPadGm2UBImpl(
            (__ubuf__ PrimType*)(dstTensor.GetPhyAddr()), (__gm__ PrimType*)(srcTensor.GetPhyAddr()), intriParams,
            padParams, static_cast<uint8_t>(CacheMode));
    }
};

// ND 2维表达，实际一维，有效维为0 场景
// PS per-channel antiquant_scale/offset
// layout (n,1):(1,n)
template <class DstTrait, class SrcTrait, class Shape, CacheMode CacheMode>
struct CopyImpl<
    Arch::Ascend910_95, AscendC::LocalTensor<DstTrait>, AscendC::GlobalTensor<SrcTrait>, Shape, CacheMode,
    typename AscendC::Std::enable_if_t<
        in_gm<SrcTrait> && in_ub<DstTrait> && is_2d_dim1_1<decltype(SrcTrait{}.GetLayout())>::value>> {
    __aicore__ inline static void Run(
        AscendC::LocalTensor<DstTrait> const& dstTensor, AscendC::GlobalTensor<SrcTrait> const& srcTensor,
        const Shape& shape)
    {
        using PrimType = AscendC::PrimT<DstTrait>;
        AscendC::DataCopyPadExtParams<PrimType> padParams;
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = ElemToByte<PrimType>(::Act::Gemm::Get<0>(shape));
        intriParams.srcStride = ElemToByte<PrimType>(::Act::Gemm::Get<1>(srcTensor.GetStride())) - intriParams.blockLen;
        intriParams.dstStride =
            (ElemToByte<PrimType>(::Act::Gemm::Get<1>(dstTensor.GetStride())) - intriParams.blockLen) / BYTE_PER_BLK;
        DataCopyPadGm2UBImpl(
            (__ubuf__ PrimType*)(dstTensor.GetPhyAddr()), (__gm__ PrimType*)(srcTensor.GetPhyAddr()), intriParams,
            padParams, static_cast<uint8_t>(CacheMode));
    }
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Tile::detail
#endif // PROLOGUE_TILE_COPY_GM_TO_UB_H