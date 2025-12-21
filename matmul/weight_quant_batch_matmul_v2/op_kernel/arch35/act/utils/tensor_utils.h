/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_ACT_UTILS_TENSOR_UTILS_H
#define ARCH35_ACT_UTILS_TENSOR_UTILS_H

#include "kernel_operator_intf.h"
#include "tensor_traits.h"
namespace WeightQuantBatchMatmulV2::Arch35::Act {

template <class T, class Layout, AscendC::TPosition Pos = AscendC::TPosition::GM>
__aicore__ inline constexpr auto MakeTensor(__gm__ T* addr, Layout const& layout)
{
    using TensorTraitType = AscendC::TensorTrait<T, Pos, Layout>;
    using TensorType = AscendC::Std::conditional_t<
        Pos == AscendC::TPosition::GM, AscendC::GlobalTensor<TensorTraitType>, AscendC::LocalTensor<TensorTraitType>>;
    TensorType tensor;
    tensor.address_ = addr;
    tensor.SetTensorTrait(AscendC::MakeTensorTrait<T, Pos>(layout));
    return tensor;
}

template <typename T, AscendC::TPosition Pos, typename Layout, typename Stride>
__aicore__ inline auto SetStride(
    AscendC::LocalTensor<AscendC::TensorTrait<T, Pos, Layout>>& tensor, Stride const& stride)
{
    tensor.SetTensorTrait(AscendC::MakeTensorTrait<T, Pos>(AscendC::MakeLayout(tensor.GetShape(), stride)));
    return tensor;
}

template <typename TensorTrait>
__aicore__ inline auto CreateTensor(uint32_t addr)
{
    static_assert(AscendC::is_tensorTrait_v<TensorTrait>, "only support TensorTrait type Tensor!");
    AscendC::LocalTensor<TensorTrait> tensor;
    tensor.address_.dataLen = 32;
    tensor.address_.bufferAddr = addr;
    tensor.address_.logicPos = static_cast<uint8_t>(TensorTrait::tPos);
    return tensor;
}

template <class T, AscendC::TPosition Pos, class Layout, class Coord, class Shape>
__aicore__ inline constexpr auto GetTile(
    AscendC::GlobalTensor<AscendC::TensorTrait<T, Pos, Layout>> const& tensor, Coord const& coord, Shape const& shape)
{
    auto layout = tensor.GetTensorTrait().GetLayout();
    auto offset = layout(coord);
    typename AscendC::Std::remove_cvref_t<decltype(tensor)> newTensor;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ >= 310
    if constexpr (
        AscendC::IsSameTypeV<T, fp4x2_e2m1_t> || AscendC::IsSameTypeV<T, fp4x2_e1m2_t> ||
        AscendC::IsSameTypeV<T, int4b_t>) {
        newTensor.address_ = (__gm__ T*)((__gm__ uint8_t*)tensor.address_ + (offset >> 1));
    } else {
        newTensor.address_ = (__gm__ T*)((__gm__ uint8_t*)tensor.address_ + offset * sizeof(T));
    }
#else
    newTensor.address_ = (__gm__ T*)((__gm__ uint8_t*)tensor.address_ + offset * sizeof(T));
#endif
    newTensor.SetTensorTrait(
        AscendC::MakeTensorTrait<T, Pos>(AscendC::MakeLayout(shape, tensor.GetTensorTrait().GetLayout().GetStride())));
    return newTensor;
}
// 不设置layout，需要手动更新
template <typename CAST_T, typename T>
__aicore__ inline auto ReinerpretCast(const AscendC::LocalTensor<T>& tensorIn)
{
    static_assert(
        AscendC::is_tensorTrait_v<CAST_T> && AscendC::is_tensorTrait_v<T>, "only support TensorTrait type Tensor!");
    if constexpr (AscendC::Std::is_same_v<CAST_T, T>) {
        return tensorIn;
    }
    static_assert(all_static_v<decltype(CAST_T{}.GetStride())> && all_static_v<decltype(T{}.GetStride())>);
    AscendC::LocalTensor<CAST_T> tensorOut;
    tensorOut.address_.logicPos = static_cast<uint8_t>(tensorIn.GetPosition());
    tensorOut.address_.bufferHandle = tensorIn.GetBufferHandle();
    tensorOut.address_.bufferAddr = tensorIn.address_.bufferAddr;
    return tensorOut;
}
} // namespace WeightQuantBatchMatmulV2::Arch35::Act
#endif