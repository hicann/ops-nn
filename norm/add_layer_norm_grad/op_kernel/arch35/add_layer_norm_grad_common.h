/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_LAYER_NORM_GRAD_COMMON_H
#define ADD_LAYER_NORM_GRAD_COMMON_H

#include "../add_layer_norm_grad_utils.h"

namespace AddLayerNormGrad {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

constexpr CastTrait castTraitB16ToB32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait castTraitB32ToB16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                         RoundMode::CAST_RINT};
template <typename T>
__simd_callee__ inline void LoadTensor(RegTensor<float>& dst, __ubuf__ T* srcAddr, MaskReg& pregLoop)
{
    if constexpr (std::is_same_v<T, float>) {
        DataCopy(dst, srcAddr);
    } else {
        RegTensor<T> tmpB16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(tmpB16, srcAddr);
        Cast<float, T, castTraitB16ToB32>(dst, tmpB16, pregLoop);
    }
}

template <typename T>
__simd_callee__ inline void CopyToTensor(__ubuf__ T* dstAddr, RegTensor<float>& src, MaskReg& pregLoop)
{
    if constexpr (std::is_same_v<T, float>) {
        DataCopy(dstAddr, src, pregLoop);
    } else {
        RegTensor<T> tmpB16;
        Cast<T, float, castTraitB32ToB16>(tmpB16, src, pregLoop);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr, tmpB16, pregLoop);
    }
}

__simd_callee__ inline void ReduceSumToScalar(RegTensor<float>& acc, __ubuf__ float* tmpAddr, MaskReg& pregReduce,
                                              MaskReg& pregMerge)
{
    ReduceSum(acc, acc, pregReduce);
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpAddr, acc, pregMerge);
}

__simd_callee__ inline void BroadcastScalar(RegTensor<float>& dst, __ubuf__ float* srcAddr)
{
    DataCopy<float, LoadDist::DIST_BRC_B32>(dst, srcAddr);
}

} // namespace AddLayerNormGrad

#endif // ADD_LAYER_NORM_GRAD_COMMON_H
