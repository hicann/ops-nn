/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_offset_utils.h
 * \brief Utility functions for computing scatter output offsets, reusable across scatter-like operators.
 */

#ifndef SCATTER_OFFSET_UTILS_H
#define SCATTER_OFFSET_UTILS_H

#include "kernel_operator.h"

namespace ScatterOffsetUtils {
using namespace AscendC;

constexpr Reg::CastTrait castTraitB322B64 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};
template <typename T>
using IndexRegType = typename std::conditional<IsSameType<T, int64_t>::value,
                                               typename AscendC::Reg::RegTensor<uint64_t, AscendC::Reg::RegTraitNumTwo>,
                                               typename AscendC::Reg::RegTensor<uint32_t>>::type;
template <typename T>
using InnerRegType = typename std::conditional<IsSameType<T, int64_t>::value,
                                               typename AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo>,
                                               typename AscendC::Reg::RegTensor<int32_t>>::type;

template <typename INDICES_T, typename TYPE_T>
__simd_vf__ inline void ComputeOutOfsetVF(__ubuf__ INDICES_T* indicesLocalPtr, __ubuf__ TYPE_T* outOfstLocalPtr,
                                          __ubuf__ TYPE_T* strideLocalPtr, __ubuf__ TYPE_T* outputShapeLocalPtr,
                                          int32_t rankSize, uint32_t dataLen, uint32_t vfLen, uint16_t loopCnt,
                                          uint16_t rankSizeLoops, uint32_t sliceSize)
{
    // 计算ub中indicesLocal的偏移，收集ub中的indices，所以一定是uint，
    InnerRegType<INDICES_T> orderReg;
    IndexRegType<INDICES_T> indexReg;
    InnerRegType<TYPE_T> inReg;
    InnerRegType<TYPE_T> outReg;
    InnerRegType<TYPE_T> selectReg;

    AscendC::Reg::MaskReg pregLoop;
    AscendC::Reg::MaskReg cmpMask;
    AscendC::Reg::MaskReg invalidMask;

    for (uint16_t i = 0; i < loopCnt; i++) {
        if constexpr (IsSameType<TYPE_T, int64_t>::value) {
            pregLoop = AscendC::Reg::UpdateMask<TYPE_T, AscendC::Reg::RegTraitNumTwo>(dataLen);
            invalidMask = AscendC::Reg::CreateMask<TYPE_T, Reg::MaskPattern::ALLF, AscendC::Reg::RegTraitNumTwo>();
        } else {
            pregLoop = AscendC::Reg::UpdateMask<TYPE_T>(dataLen);
            invalidMask = AscendC::Reg::CreateMask<TYPE_T, Reg::MaskPattern::ALLF>();
        }
        AscendC::Reg::Duplicate(outReg, 0, pregLoop);
        AscendC::Reg::Arange(orderReg, i * vfLen);
        AscendC::Reg::Muls(orderReg, orderReg, rankSize, pregLoop);
        for (uint16_t dim = 0; dim < rankSizeLoops; dim++) {
            TYPE_T strideValue = strideLocalPtr[dim];
            TYPE_T outputShapeValue = outputShapeLocalPtr[dim];
            indexReg = (IndexRegType<INDICES_T>&)orderReg;

            if constexpr (IsSameType<INDICES_T, int32_t>::value && IsSameType<TYPE_T, int64_t>::value) {
                AscendC::Reg::RegTensor<int32_t> castReg;
                AscendC::Reg::Gather(castReg, indicesLocalPtr, indexReg, pregLoop);
                Reg::Cast<int64_t, int32_t, castTraitB322B64>(inReg, castReg, pregLoop);
            } else {
                AscendC::Reg::Gather(inReg, indicesLocalPtr, indexReg, pregLoop);
            }
            AscendC::Reg::Compares<TYPE_T, CMPMODE::LT>(cmpMask, inReg, static_cast<TYPE_T>(0), pregLoop);
            AscendC::Reg::Or(invalidMask, invalidMask, cmpMask, pregLoop);
            AscendC::Reg::Compares<TYPE_T, CMPMODE::GE>(cmpMask, inReg, static_cast<TYPE_T>(outputShapeValue),
                                                        pregLoop);
            AscendC::Reg::Or(invalidMask, invalidMask, cmpMask, pregLoop);

            AscendC::Reg::Muls(inReg, inReg, strideValue, pregLoop);
            AscendC::Reg::Add(outReg, inReg, outReg, pregLoop);
            AscendC::Reg::Adds(orderReg, orderReg, static_cast<INDICES_T>(1), pregLoop);
        }
        AscendC::Reg::Duplicate(selectReg, static_cast<TYPE_T>(-1), pregLoop);
        AscendC::Reg::Muls(outReg, outReg, static_cast<TYPE_T>(sliceSize), pregLoop);
        AscendC::Reg::Select(outReg, selectReg, outReg, invalidMask);
        auto outOfstAddr = outOfstLocalPtr + i * vfLen;
        AscendC::Reg::StoreAlign(outOfstAddr, outReg, pregLoop);
    }
}

template <typename INDICES_T, typename TYPE_T>
__aicore__ inline void ComputeOutOfset(const LocalTensor<INDICES_T>& indicesLocal,
                                       const LocalTensor<TYPE_T>& outOfstLocal, const LocalTensor<TYPE_T>& strideLocal,
                                       const LocalTensor<TYPE_T>& outputShapeLocal, uint32_t dataLen, int32_t rankSize,
                                       uint32_t sliceSize)
{
    __ubuf__ INDICES_T* indicesLocalPtr = ((__ubuf__ INDICES_T*)indicesLocal.GetPhyAddr());
    __ubuf__ TYPE_T* outOfstLocalPtr = ((__ubuf__ TYPE_T*)outOfstLocal.GetPhyAddr());
    __ubuf__ TYPE_T* strideLocalPtr = ((__ubuf__ TYPE_T*)strideLocal.GetPhyAddr());
    __ubuf__ TYPE_T* outputShapeLocalPtr = ((__ubuf__ TYPE_T*)outputShapeLocal.GetPhyAddr());

    uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
    uint32_t indicesLenTimes = ops::CeilDiv(dataLen, vfLen);
    uint16_t loopCnt = static_cast<uint16_t>(indicesLenTimes);
    uint16_t rankSizeLoops = static_cast<uint16_t>(rankSize);

    ComputeOutOfsetVF<INDICES_T, TYPE_T>(indicesLocalPtr, outOfstLocalPtr, strideLocalPtr, outputShapeLocalPtr,
                                         rankSize, dataLen, vfLen, loopCnt, rankSizeLoops, sliceSize);
}

} // namespace ScatterOffsetUtils

#endif // SCATTER_OFFSET_UTILS_H
