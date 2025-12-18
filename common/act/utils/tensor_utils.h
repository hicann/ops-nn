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
 * \file tensor_utils.h
 * \brief
 */

#ifndef UTILS_TENSOR_UTILS_H
#define UTILS_TENSOR_UTILS_H
#include "integral_constant.h"
#include "tuple_utils.h"
#include "kernel_operator_list_tensor_intf.h"

namespace AscendC {
template <typename Tp>
struct is_global_tensor : public Std::false_type {};

template <typename Tp>
struct is_global_tensor<GlobalTensor<Tp>> : public Std::true_type {};

template <typename Tp>
constexpr bool is_global_tensor_v = is_global_tensor<Tp>::value;

template <typename Tp>
struct is_local_tensor : public Std::false_type {};

template <typename Tp>
struct is_local_tensor<LocalTensor<Tp>> : public Std::true_type {};

template <typename Tp>
constexpr bool is_local_tensor_v = is_local_tensor<Tp>::value;

template <typename Tp>
struct tensor_trait {
    static_assert(Std::always_false_v<Tp>, "Unsupported tensor type");
};

template <typename Tp>
struct tensor_trait<GlobalTensor<Tp>> {
    using trait_type = Tp;
};

template <typename Tp>
struct tensor_trait<LocalTensor<Tp>> {
    using trait_type = Tp;
};
} // namespace AscendC

namespace Act {
namespace Gemm {

template <class Layout_, class AGlobalTensor_, class ATensorTrait_, class AType_>
__aicore__ inline void InitGlobalTensorA(AGlobalTensor_& aGlobal, GM_ADDR aGmAddr, bool transA, int64_t m, int64_t k)
{
    Layout_ aLayout;
    if (!transA) {
        aLayout = AscendC::MakeLayout(AscendC::MakeShape(m, k), AscendC::MakeStride(k, int64_t(1)));
    } else {
        aLayout = AscendC::MakeLayout(AscendC::MakeShape(k, m), AscendC::MakeStride(m, int64_t(1)));
    }
    aGlobal.SetTensorTrait(ATensorTrait_(aLayout));
#if defined(__DAV_C310__)
    AscendC::GlobalTensor<AType_> aTmp;
    aTmp.SetGlobalBuffer(reinterpret_cast<__gm__ AType_*>(aGmAddr), m * k);
    aGlobal.address_ = aTmp.address_;
#else
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType_*>(aGmAddr), m * k);
#endif
}

template <class Layout_, class BGlobalTensor_, class BTensorTrait_, class BType_>
__aicore__ inline void InitGlobalTensorB(BGlobalTensor_& bGlobal, GM_ADDR bGmAddr, bool transB, int64_t n, int64_t k)
{
    Layout_ bLayout;
    if (!transB) {
        bLayout = AscendC::MakeLayout(AscendC::MakeShape(k, n), AscendC::MakeStride(n, int64_t(1)));
    } else {
        bLayout = AscendC::MakeLayout(AscendC::MakeShape(n, k), AscendC::MakeStride(k, int64_t(1)));
    }
    bGlobal.SetTensorTrait(BTensorTrait_(bLayout));
#if defined(__DAV_C310__)
    AscendC::GlobalTensor<BType_> bTmp;
    bTmp.SetGlobalBuffer(reinterpret_cast<__gm__ BType_*>(bGmAddr), k * n);
    bGlobal.address_ = bTmp.address_;
#else
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType_*>(bGmAddr), k * n);
#endif
}

template <class Layout_, class CGlobalTensor_, class CTensorTrait_, class CType_>
__aicore__ inline void InitGlobalTensorC(CGlobalTensor_& cGlobal, GM_ADDR cGmAddr, int64_t m, int64_t n)
{
    Layout_ cLayout = AscendC::MakeLayout(AscendC::MakeShape(m, n), AscendC::MakeStride(n, int64_t(1)));
    cGlobal.SetTensorTrait(CTensorTrait_(cLayout));
#if defined(__DAV_C310__)
    AscendC::GlobalTensor<CType_> cTmp;
    cTmp.SetGlobalBuffer(reinterpret_cast<__gm__ CType_*>(cGmAddr), m * n);
    cGlobal.address_ = cTmp.address_;
#else
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType_*>(cGmAddr), m * n);
#endif
}

template <class Layout_, class CTensorTrait_, class CType_, class BlockShape_>
__aicore__ inline AscendC::GlobalTensor<CTensorTrait_> GetWorkSpaceGlobal(BlockShape_ blockShape, GM_ADDR cGmAddr)
{
    int64_t blockShapeM = Get<0>(blockShape);
    int64_t blockShapeN = Get<1>(blockShape);
    Layout_ cLayout =
        AscendC::MakeLayout(AscendC::MakeShape(blockShapeM, blockShapeN), AscendC::MakeStride(blockShapeN, int64_t(1)));
    CTensorTrait_ cTensorTrait = AscendC::MakeTensorTrait<CType_, AscendC::TPosition::GM>(cLayout);
    AscendC::GlobalTensor<CTensorTrait_> workspaceGlobal;
    workspaceGlobal.SetTensorTrait(cTensorTrait);
#if defined(__DAV_C310__)
    AscendC::GlobalTensor<CType_> cTmp;
    cTmp.SetGlobalBuffer(reinterpret_cast<__gm__ CType_*>(cGmAddr));
    workspaceGlobal.address_ = cTmp.address_;
#else
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType_*>(cGmAddr));
#endif
    return workspaceGlobal;
}

template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint64_t index, GM_ADDR tensorPtr)
{
    AscendC::ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void*>(tensorPtr));
    return listTensorDesc.GetDataPtr<T>(index);
}

template <class T, AscendC::TPosition Pos, class Layout, class Coord, class Shape>
__aicore__ inline constexpr auto GetTile(AscendC::GlobalTensor<AscendC::TensorTrait<T, Pos, Layout>> const &tensor,
                                         Coord const &coord, Shape const &shape)
{
    auto layout = tensor.GetTensorTrait().GetLayout();
    auto offset = layout(coord);
    typename AscendC::Std::remove_cvref_t<decltype(tensor)> newTensor;
#if defined(__CCE_AICORE__) && __CCE_AICORE__>=310
    if constexpr (AscendC::IsSameTypeV<T, fp4x2_e2m1_t> || AscendC::IsSameTypeV<T, fp4x2_e1m2_t>) {
        newTensor.address_ = (__gm__ T *)((__gm__ uint8_t *)tensor.address_ + (offset >> 1));
    } else {
        newTensor.address_ = (__gm__ T *)((__gm__ uint8_t *)tensor.address_ + offset * sizeof(T));
    }
#else
    newTensor.address_ = (__gm__ T *)((__gm__ uint8_t *)tensor.address_ + offset * sizeof(T));
#endif
    newTensor.SetTensorTrait(
        AscendC::MakeTensorTrait<T, Pos>(AscendC::MakeLayout(shape, tensor.GetTensorTrait().GetLayout().GetStride())));
    return newTensor;
}

template <class T, class Layout, AscendC::TPosition Pos = AscendC::TPosition::GM>
__aicore__ inline constexpr auto MakeTensor(__gm__ T *addr, Layout const &layout)
{
    using TensorTraitType = AscendC::TensorTrait<T, Pos, Layout>;
    using TensorType =
        AscendC::Std::conditional_t<Pos == AscendC::TPosition::GM, AscendC::GlobalTensor<TensorTraitType>,
                                    AscendC::LocalTensor<TensorTraitType>>;
    TensorType tensor;
    tensor.address_ = addr;
    tensor.SetTensorTrait(AscendC::MakeTensorTrait<T, Pos>(layout));
    return tensor;
}
} // namespace Gemm
} // namespace Act
#endif