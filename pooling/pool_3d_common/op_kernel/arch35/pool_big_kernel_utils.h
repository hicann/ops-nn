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

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr MicroAPI::CastTrait castTraitB322B64 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T, typename U>
__aicore__ inline void StoreOneElement(
    const __local_mem__ void* output, MicroAPI::RegTensor<U>& src, MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
            (__local_mem__ half*)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
            (__local_mem__ bfloat16_t*)(output) + offset, xBf16, preg);
    } else if constexpr (sizeof(T) == FOUR) {
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            ((__local_mem__ float*)output) + offset, (MicroAPI::RegTensor<float>&)src, preg);
    } else {
        MicroAPI::UnalignReg u0;
        auto dstAddr = (__local_mem__ T*)(output) + offset;
        MicroAPI::DataCopyUnAlign(dstAddr, src, u0, 1);
        MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
    }
}

template <typename T, typename U>
__aicore__ inline void LoadOneElement(
    const __local_mem__ void* input, MicroAPI::RegTensor<U>& dst, MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(xFp16, (__local_mem__ half*)(input) + offset);
        MicroAPI::Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_BRC_B16>(
            xBf16, (__local_mem__ bfloat16_t*)(input) + offset);
        MicroAPI::Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else if constexpr (sizeof(T) == FOUR) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(dst, ((__local_mem__ T*)(input)) + offset);
    } else {
        MicroAPI::UnalignReg u0;
        auto srcAddr = (__local_mem__ T*)(input) + offset;
        MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
        MicroAPI::DataCopyUnAlign(dst, u0, srcAddr, 1);
    }
}

template <typename T>
__aicore__ inline void LoadOneTensor(
    const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst, MicroAPI::MaskReg& preg,
    MicroAPI::AddrReg& offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, (__local_mem__ half*)(input), offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<bfloat16_t> xBf16;
        MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            xBf16, (__local_mem__ bfloat16_t*)(input), offset);
        MicroAPI::Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        MicroAPI::DataCopy(dst, (__local_mem__ float*)(input), offset);
    }
}

template <typename T, bool SPLITKW>
__aicore__ inline void CalcRealIndex(
    MicroAPI::RegTensor<T>& resIndex, MicroAPI::RegTensor<int32_t>& index, int64_t curKw, int64_t inputW,
    int64_t offset)
{
    MicroAPI::MaskReg pregOneIndex = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::VL1>();

    MicroAPI::RegTensor<T> indexCast;
    if constexpr (IsSameType<T, int64_t>::value) {
        MicroAPI::Cast<int64_t, int32_t, castTraitB322B64>(indexCast, index, pregOneIndex);
    } else {
        MicroAPI::Copy(indexCast, index, pregOneIndex);
    }
    if constexpr (SPLITKW) {
        MicroAPI::Adds(resIndex, indexCast, static_cast<T>(offset), pregOneIndex);
    } else {
        MicroAPI::RegTensor<T> wLen;
        MicroAPI::RegTensor<T> v0;
        MicroAPI::RegTensor<T> v1;
        MicroAPI::Duplicate(wLen, static_cast<T>(curKw), pregOneIndex);
        MicroAPI::Div(v0, indexCast, wLen, pregOneIndex);
        MicroAPI::Muls(resIndex, v0, inputW, pregOneIndex);
        MicroAPI::Adds(resIndex, resIndex, static_cast<T>(offset), pregOneIndex);
        MicroAPI::Mul(wLen, wLen, v0, pregOneIndex);
        MicroAPI::Sub(v0, indexCast, wLen, pregOneIndex);
        MicroAPI::Add(resIndex, resIndex, v0, pregOneIndex);
    }
}

template <typename T>
__aicore__ inline void DuplicateNegInfReg(MicroAPI::RegTensor<T>& negInfReg)
{
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<computeType>&)negInfReg, FLOAT32_NEG_INF);
    } else if constexpr (std::is_same<T, half>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<computeType>&)negInfReg, FLOAT16_NEG_INF);
    } else {
        MicroAPI::Duplicate((MicroAPI::RegTensor<computeType>&)negInfReg, BFLOAT16_NEG_INF);
    }
}

template <typename T>
__aicore__ inline void DuplicateNegInf(const __local_mem__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    MicroAPI::RegTensor<T> v0;
    MicroAPI::UnalignReg u0;
    DuplicateNegInfReg<T>(v0);
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr + offset;
    MicroAPI::DataCopyUnAlign(addr, v0, u0, calNum);
    MicroAPI::DataCopyUnAlignPost(addr, u0, 0);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T>
__aicore__ inline void ReduceMaxWithIndex(
    MicroAPI::RegTensor<T>& dst, MicroAPI::RegTensor<int32_t>& dstIndex, MicroAPI::RegTensor<T>& src,
    MicroAPI::RegTensor<int32_t>& srcIndex, int32_t indexPadValue)
{
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg notNanMaskReg;
    MicroAPI::MaskReg nanMaskReg;
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::RegTensor<T> vd2;
    MicroAPI::RegTensor<int32_t> nanIndex;
    MicroAPI::Duplicate(nanIndex, indexPadValue);
    MicroAPI::Compare<T, CMPMODE::NE>(nanMaskReg, src, src, maskAll);
    MicroAPI::MaskNot(notNanMaskReg, nanMaskReg, maskAll);
    MicroAPI::Select(nanIndex, srcIndex, nanIndex, nanMaskReg);
    MicroAPI::ReduceMax(nanIndex, nanIndex, maskAll);
    MicroAPI::ReduceMax(vd1, src, notNanMaskReg);
    MicroAPI::Duplicate(vd2, vd1, maskAll);
    MicroAPI::Compare<T, CMPMODE::EQ>(notNanMaskReg, src, vd2, maskAll);
    MicroAPI::ReduceMin(dstIndex, srcIndex, notNanMaskReg);
    MicroAPI::CompareScalar<int32_t, CMPMODE::NE>(nanMaskReg, nanIndex, indexPadValue, maskAll);
    MicroAPI::Select(dstIndex, nanIndex, dstIndex, nanMaskReg);
    MicroAPI::Duplicate(dstIndex, dstIndex, maskAll);
    MicroAPI::Compare<int32_t, CMPMODE::EQ>(notNanMaskReg, dstIndex, srcIndex, maskAll);
    MicroAPI::ReduceMax(dst, src, notNanMaskReg);
    MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(notNanMaskReg, dstIndex, indexPadValue, maskAll);
    MicroAPI::Duplicate(nanIndex, static_cast<int32_t>(0));
    MicroAPI::Select(dstIndex, nanIndex, dstIndex, notNanMaskReg);
}

} // namespace PoolBigKernelUtils
#endif // POOL_BIG_KERNEL_UTILS_H_
