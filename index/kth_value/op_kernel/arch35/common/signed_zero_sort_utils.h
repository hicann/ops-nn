/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIGNED_ZERO_SORT_UTILS_H
#define SIGNED_ZERO_SORT_UTILS_H

#include <type_traits>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace SignedZeroSortCommon {
using namespace AscendC;

constexpr uint32_t SIGNED_ZERO_INDEX_FLAG = 0x80000000U;

template <typename T>
constexpr bool IS_FLOATING_POINT_V = std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
                                     std::is_same_v<T, float>;

template <typename T>
using FloatBitsT = std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>;

// Detect exact -0 before entering the signed-zero preservation path. Most
// inputs do not contain -0, so they can keep the original Sort fast path and
// avoid materializing/restoring source-order flags for every element.
template <typename T>
__aicore__ inline bool HasNegativeZeroVec(LocalTensor<T>& values, LocalTensor<uint32_t>& scalar, uint32_t totalElems)
{
    using BitsT = FloatBitsT<T>;
    constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(BitsT);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, vlSize));
    __ubuf__ BitsT* valueAddr = reinterpret_cast<__ubuf__ BitsT*>(values.GetPhyAddr());
    __ubuf__ uint32_t* scalarAddr = reinterpret_cast<__ubuf__ uint32_t*>(scalar.GetPhyAddr());
    __VEC_SCOPE__
    {
        Reg::RegTensor<BitsT> valueReg;
        Reg::MaskReg foundMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALLF>();
        Reg::MaskReg fullMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALL>();
        for (uint16_t i = 0U; i < repeatTimes; ++i) {
            uint32_t curCount = totalElems - static_cast<uint32_t>(i) * vlSize;
            curCount = curCount > vlSize ? vlSize : curCount;
            Reg::MaskReg valueMask = Reg::UpdateMask<BitsT>(curCount);
            Reg::LoadAlign(valueReg, valueAddr + static_cast<uint32_t>(i) * vlSize);
            Reg::MaskReg negativeZeroMask;
            Reg::Compares<BitsT, CMPMODE::EQ>(negativeZeroMask, valueReg, signBit, valueMask);
            Reg::Or(foundMask, foundMask, negativeZeroMask, fullMask);
        }
        if constexpr (sizeof(BitsT) == sizeof(uint16_t)) {
            Reg::RegTensor<uint16_t> oneReg;
            Reg::RegTensor<uint16_t> resultReg;
            Reg::RegTensor<uint32_t> resultU32Reg;
            Reg::MaskReg oneMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
            Reg::Duplicate(oneReg, static_cast<uint16_t>(1U));
            Reg::Reduce<Reg::ReduceType::MAX>(resultReg, oneReg, foundMask);
            Reg::UnPack<uint32_t, uint16_t, Reg::HighLowPart::LOWEST>(resultU32Reg, resultReg);
            Reg::StoreAlign(scalarAddr, resultU32Reg, oneMask);
        } else {
            Reg::RegTensor<uint32_t> oneReg;
            Reg::RegTensor<uint32_t> resultReg;
            Reg::MaskReg oneMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL1>();
            Reg::Duplicate(oneReg, 1U);
            Reg::Reduce<Reg::ReduceType::MAX>(resultReg, oneReg, foundMask);
            Reg::StoreAlign(scalarAddr, resultReg, oneMask);
        }
    }
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    return scalar.GetValue(0) != 0U;
}

// Normalize exact -0 to +0 before radix sort and preserve the original sign
// in the opaque source-index payload.  Keep this path in Vector-Reg code: a
// new asc_vf_call changes the binary-wide SIMT UB reservation and can force
// unrelated two-stage schedules to retile.
template <typename T>
__aicore__ inline void PrepareSignedZeroKeysVec(LocalTensor<T>& values, LocalTensor<uint32_t>& sourceOrder,
                                                uint32_t totalElems)
{
    using BitsT = FloatBitsT<T>;
    constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
    // Process 64 elements per repeat so one predicate maps one-to-one to both
    // B16/B32 values and the B32 source-index payload.
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, vlSize));
    __ubuf__ BitsT* valueAddr = (__ubuf__ BitsT*)values.GetPhyAddr();
    __ubuf__ uint32_t* sourceOrderAddr = (__ubuf__ uint32_t*)sourceOrder.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::RegTensor<BitsT> valueReg;
        Reg::RegTensor<BitsT> zeroReg;
        Reg::RegTensor<BitsT> normalizedReg;
        Reg::RegTensor<int32_t> baseIndexReg;
        Reg::RegTensor<int32_t> indexReg;
        Reg::RegTensor<uint32_t> indexFlagReg;
        Reg::RegTensor<uint32_t> flaggedIndexReg;
        Reg::RegTensor<uint32_t> sourceOrderReg;
        Reg::MaskReg fullValueMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALL>();
        Reg::MaskReg fullIndexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(zeroReg, static_cast<BitsT>(0U), fullValueMask);
        Reg::Duplicate(indexFlagReg, SIGNED_ZERO_INDEX_FLAG, fullIndexMask);
        Reg::Arange(baseIndexReg, 0);
        for (uint16_t i = 0U; i < repeatTimes; ++i) {
            uint32_t curCount = totalElems - static_cast<uint32_t>(i) * vlSize;
            if (curCount > vlSize) {
                curCount = vlSize;
            }
            uint32_t valueCount = curCount;
            uint32_t indexCount = curCount;
            Reg::MaskReg valueMask = Reg::UpdateMask<BitsT>(valueCount);
            Reg::MaskReg indexMask = Reg::UpdateMask<uint32_t>(indexCount);
            Reg::LoadAlign(valueReg, valueAddr + static_cast<uint32_t>(i) * vlSize);
            Reg::MaskReg negativeZeroMask;
            Reg::Compares<BitsT, CMPMODE::EQ>(negativeZeroMask, valueReg, signBit, valueMask);
            Reg::Select(normalizedReg, zeroReg, valueReg, negativeZeroMask);
            Reg::StoreAlign(valueAddr + static_cast<uint32_t>(i) * vlSize, normalizedReg, valueMask);

            Reg::Adds(indexReg, baseIndexReg, static_cast<int32_t>(i * vlSize), indexMask);
            Reg::Or(flaggedIndexReg, (Reg::RegTensor<uint32_t>&)indexReg, indexFlagReg, indexMask);
            Reg::MaskReg negativeZeroIndexMask;
            if constexpr (sizeof(BitsT) == sizeof(uint16_t)) {
                // A B16 predicate uses bit 2*i for lane i, while a B32 predicate
                // uses bit 4*i. Expand the low half before applying it to indices.
                Reg::UnPack<Reg::HighLowPart::LOWEST>(negativeZeroIndexMask, negativeZeroMask);
            } else {
                negativeZeroIndexMask = negativeZeroMask;
            }
            Reg::Select(sourceOrderReg, flaggedIndexReg, (Reg::RegTensor<uint32_t>&)indexReg, negativeZeroIndexMask);
            Reg::StoreAlign(sourceOrderAddr + static_cast<uint32_t>(i) * vlSize, sourceOrderReg, indexMask);
        }
    }
}

// Two-stage sort already produces source positions.  Preserve only the -0
// marker in a dead B32-sized output buffer and let the implicit-index Sort overload
// build clean positions.  This avoids generating and sorting a full custom
// source-index vector on that path.
template <typename T>
__aicore__ inline void PrepareSignedZeroKeysAndFlagsVec(LocalTensor<T>& values, LocalTensor<uint32_t>& signFlags,
                                                        uint32_t totalElems)
{
    using BitsT = FloatBitsT<T>;
    constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(BitsT);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, vlSize));
    __ubuf__ BitsT* valueAddr = (__ubuf__ BitsT*)values.GetPhyAddr();
    __ubuf__ BitsT* signFlagAddr = reinterpret_cast<__ubuf__ BitsT*>(signFlags.GetPhyAddr());
    __VEC_SCOPE__
    {
        Reg::RegTensor<BitsT> valueReg;
        Reg::RegTensor<BitsT> zeroReg;
        Reg::RegTensor<BitsT> normalizedReg;
        Reg::RegTensor<BitsT> zeroFlagReg;
        Reg::RegTensor<BitsT> negativeZeroFlagReg;
        Reg::RegTensor<BitsT> signFlagReg;
        Reg::MaskReg fullValueMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALL>();
        Reg::Duplicate(zeroReg, static_cast<BitsT>(0U), fullValueMask);
        Reg::Duplicate(zeroFlagReg, static_cast<BitsT>(0U), fullValueMask);
        Reg::Duplicate(negativeZeroFlagReg, signBit, fullValueMask);
        for (uint16_t i = 0U; i < repeatTimes; ++i) {
            uint32_t curCount = totalElems - static_cast<uint32_t>(i) * vlSize;
            if (curCount > vlSize) {
                curCount = vlSize;
            }
            Reg::MaskReg valueMask = Reg::UpdateMask<BitsT>(curCount);
            Reg::LoadAlign(valueReg, valueAddr + static_cast<uint32_t>(i) * vlSize);
            Reg::MaskReg negativeZeroMask;
            Reg::Compares<BitsT, CMPMODE::EQ>(negativeZeroMask, valueReg, signBit, valueMask);
            Reg::Select(normalizedReg, zeroReg, valueReg, negativeZeroMask);
            // The rank-inverse scatter only needs a non-zero marker. Store it
            // at value width so the Compare predicate can be reused directly;
            // the backing allocation remains B32-sized and is therefore ample.
            Reg::Select(signFlagReg, negativeZeroFlagReg, zeroFlagReg, negativeZeroMask);
            Reg::StoreAlign(valueAddr + static_cast<uint32_t>(i) * vlSize, normalizedReg, valueMask);
            Reg::StoreAlign(signFlagAddr + static_cast<uint32_t>(i) * vlSize, signFlagReg, valueMask);
        }
    }
}

// Complete the prepare/restore loop by gathering each sorted element's source-order
// marker through indices, then restoring canonical +0 to -0 only for values that
// originally carried the negative-zero marker.
template <typename T>
__aicore__ inline void RestoreSignedZeroValuesByIndexVec(LocalTensor<T>& values, LocalTensor<uint32_t>& indices,
                                                         LocalTensor<uint32_t>& sourceOrder, uint32_t totalElems)
{
    using BitsT = FloatBitsT<T>;
    constexpr BitsT signBit = static_cast<BitsT>(BitsT{1} << (sizeof(T) * 8U - 1U));
    constexpr uint32_t vlSize = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    uint16_t repeatTimes = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, vlSize));
    __ubuf__ BitsT* valueAddr = reinterpret_cast<__ubuf__ BitsT*>(values.GetPhyAddr());
    __ubuf__ uint32_t* indexAddr = reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr());
    __ubuf__ uint32_t* sourceOrderAddr = reinterpret_cast<__ubuf__ uint32_t*>(sourceOrder.GetPhyAddr());
    __VEC_SCOPE__
    {
        Reg::RegTensor<BitsT> valueReg, negativeZeroReg, restoredValueReg;
        Reg::RegTensor<uint32_t> indexReg, sourceOrderReg, flagReg, indexFlagReg;
        Reg::MaskReg fullValueMask = Reg::CreateMask<BitsT, Reg::MaskPattern::ALL>();
        Reg::MaskReg fullIndexMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(negativeZeroReg, signBit, fullValueMask);
        Reg::Duplicate(indexFlagReg, SIGNED_ZERO_INDEX_FLAG, fullIndexMask);
        for (uint16_t i = 0U; i < repeatTimes; ++i) {
            uint32_t curCount = totalElems - static_cast<uint32_t>(i) * vlSize;
            curCount = curCount > vlSize ? vlSize : curCount;
            Reg::MaskReg valueMask = Reg::UpdateMask<BitsT>(curCount);
            Reg::MaskReg indexMask = Reg::UpdateMask<uint32_t>(curCount);
            Reg::LoadAlign(valueReg, valueAddr + static_cast<uint32_t>(i) * vlSize);
            Reg::LoadAlign(indexReg, indexAddr + static_cast<uint32_t>(i) * vlSize);
            Reg::Gather(sourceOrderReg, sourceOrderAddr, indexReg, indexMask);
            Reg::And(flagReg, sourceOrderReg, indexFlagReg, indexMask);
            Reg::MaskReg negativeZeroIndexMask;
            Reg::Compares<uint32_t, CMPMODE::NE>(negativeZeroIndexMask, flagReg, 0U, indexMask);
            Reg::MaskReg negativeZeroValueMask;
            if constexpr (sizeof(BitsT) == sizeof(uint16_t)) {
                Reg::Pack<Reg::HighLowPart::LOWEST>(negativeZeroValueMask, negativeZeroIndexMask);
            } else {
                negativeZeroValueMask = negativeZeroIndexMask;
            }
            Reg::Select(restoredValueReg, negativeZeroReg, valueReg, negativeZeroValueMask);
            Reg::StoreAlign(valueAddr + static_cast<uint32_t>(i) * vlSize, restoredValueReg, valueMask);
        }
    }
}

} // namespace SignedZeroSortCommon

#endif // SIGNED_ZERO_SORT_UTILS_H
