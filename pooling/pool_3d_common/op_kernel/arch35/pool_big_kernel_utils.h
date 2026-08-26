/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_BIG_KERNEL_UTILS_H_
#define POOL_BIG_KERNEL_UTILS_H_

#include "kernel_operator.h"

namespace PoolBigKernelUtils {
using namespace AscendC;

constexpr int32_t FOUR = 4;

constexpr Reg::CastTrait castTraitB322B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::CAST_RINT};

constexpr Reg::CastTrait castTraitB162B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

constexpr Reg::CastTrait castTraitB322B64 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

template <typename T, typename U>
__aicore__ inline void StoreOneElement(const __ubuf__ void* output, Reg::RegTensor<U>& src, Reg::MaskReg& preg,
                                       uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>((__ubuf__ half*)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>((__ubuf__ bfloat16_t*)(output) + offset,
                                                                            xBf16, preg);
    } else if constexpr (sizeof(T) == FOUR) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(((__ubuf__ float*)output) + offset,
                                                                       (Reg::RegTensor<float>&)src, preg);
    } else {
        Reg::UnalignRegForStore u0;
        auto dstAddr = (__ubuf__ T*)(output) + offset;
        Reg::StoreUnAlign(dstAddr, src, u0, 1);
        Reg::StoreUnAlignPost(dstAddr, u0, 0);
    }
}

template <typename T, typename U>
__aicore__ inline void LoadOneElement(const __ubuf__ void* input, Reg::RegTensor<U>& dst, Reg::MaskReg& preg,
                                      uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_BRC_B16>(xFp16, (__ubuf__ half*)(input) + offset);
        Reg::Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_BRC_B16>(xBf16, (__ubuf__ bfloat16_t*)(input) + offset);
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else if constexpr (sizeof(T) == FOUR) {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(dst, ((__ubuf__ T*)(input)) + offset);
    } else {
        Reg::UnalignRegForLoad u0;
        auto srcAddr = (__ubuf__ T*)(input) + offset;
        Reg::LoadUnAlignPre(u0, srcAddr);
        Reg::LoadUnAlign(dst, u0, srcAddr, 1);
    }
}

template <typename T>
__aicore__ inline void LoadOneTensor(const __ubuf__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                     Reg::AddrReg& offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16, (__ubuf__ half*)(input), offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, (__ubuf__ bfloat16_t*)(input), offset);
        Reg::Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        Reg::LoadAlign(dst, (__ubuf__ float*)(input), offset);
    }
}

template <typename T, bool SPLITKW>
__aicore__ inline void CalcRealIndex(Reg::RegTensor<T>& resIndex, Reg::RegTensor<int32_t>& index, int64_t curKw,
                                     int64_t inputW, int64_t offset)
{
    Reg::MaskReg pregOneIndex = Reg::CreateMask<int32_t, Reg::MaskPattern::VL1>();

    Reg::RegTensor<T> indexCast;
    if constexpr (IsSameType<T, int64_t>::value) {
        Reg::Cast<int64_t, int32_t, castTraitB322B64>(indexCast, index, pregOneIndex);
    } else {
        Reg::Move(indexCast, index, pregOneIndex);
    }
    if constexpr (SPLITKW) {
        Reg::Adds(resIndex, indexCast, static_cast<T>(offset), pregOneIndex);
    } else {
        Reg::RegTensor<T> wLen;
        Reg::RegTensor<T> v0;
        Reg::RegTensor<T> v1;
        Reg::Duplicate(wLen, static_cast<T>(curKw), pregOneIndex);
        Reg::Div(v0, indexCast, wLen, pregOneIndex);
        Reg::Muls(resIndex, v0, inputW, pregOneIndex);
        Reg::Adds(resIndex, resIndex, static_cast<T>(offset), pregOneIndex);
        Reg::Mul(wLen, wLen, v0, pregOneIndex);
        Reg::Sub(v0, indexCast, wLen, pregOneIndex);
        Reg::Add(resIndex, resIndex, v0, pregOneIndex);
    }
}

template <typename T>
__aicore__ inline void DuplicateNegInfReg(Reg::RegTensor<T>& negInfReg)
{
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, FLOAT32_NEG_INF);
    } else if constexpr (std::is_same<T, half>::value) {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, FLOAT16_NEG_INF);
    } else {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, BFLOAT16_NEG_INF);
    }
}

template <typename T>
__aicore__ inline void DuplicateNegInf(const __ubuf__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    Reg::RegTensor<T> v0;
    Reg::UnalignRegForStore u0;
    DuplicateNegInfReg<T>(v0);
    __ubuf__ T* addr = (__ubuf__ T*)dstAddr + offset;
    Reg::StoreUnAlign(addr, v0, u0, calNum);
    Reg::StoreUnAlignPost(addr, u0, 0);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T>
__aicore__ inline void ReduceMaxWithIndex(Reg::RegTensor<T>& dst, Reg::RegTensor<int32_t>& dstIndex,
                                          Reg::RegTensor<T>& src, Reg::RegTensor<int32_t>& srcIndex,
                                          int32_t indexPadValue)
{
    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg notNanMaskReg;
    Reg::MaskReg nanMaskReg;
    Reg::RegTensor<T> vd1;
    Reg::RegTensor<T> vd2;
    Reg::RegTensor<int32_t> nanIndex;
    Reg::Duplicate(nanIndex, indexPadValue);
    Reg::Compare<T, CMPMODE::NE>(nanMaskReg, src, src, maskAll);
    Reg::Not(notNanMaskReg, nanMaskReg, maskAll);
    Reg::Select(nanIndex, srcIndex, nanIndex, nanMaskReg);
    Reg::Reduce<Reg::ReduceType::MAX>(nanIndex, nanIndex, maskAll);
    Reg::Reduce<Reg::ReduceType::MAX>(vd1, src, notNanMaskReg);
    Reg::Duplicate(vd2, vd1, maskAll);
    Reg::Compare<T, CMPMODE::EQ>(notNanMaskReg, src, vd2, maskAll);
    Reg::Reduce<Reg::ReduceType::MIN>(dstIndex, srcIndex, notNanMaskReg);
    Reg::Compares<int32_t, CMPMODE::NE>(nanMaskReg, nanIndex, indexPadValue, maskAll);
    Reg::Select(dstIndex, nanIndex, dstIndex, nanMaskReg);
    Reg::Duplicate(dstIndex, dstIndex, maskAll);
    Reg::Compare<int32_t, CMPMODE::EQ>(notNanMaskReg, dstIndex, srcIndex, maskAll);
    Reg::Reduce<Reg::ReduceType::MAX>(dst, src, notNanMaskReg);
    Reg::Compares<int32_t, CMPMODE::EQ>(notNanMaskReg, dstIndex, indexPadValue, maskAll);
    Reg::Duplicate(nanIndex, static_cast<int32_t>(0));
    Reg::Select(dstIndex, nanIndex, dstIndex, notNanMaskReg);
}

} // namespace PoolBigKernelUtils
#endif // POOL_BIG_KERNEL_UTILS_H_
