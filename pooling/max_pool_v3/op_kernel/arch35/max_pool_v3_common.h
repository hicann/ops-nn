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
 * \file max_pool_v3_common.h
 * \brief
 */
#ifndef MAX_POOL_V3_COMMON_H_
#define MAX_POOL_V3_COMMON_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MaxPoolV3 {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;     // -inf 0xFC00
constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000; // -inf 0xFF800000
constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;    // -inf 0xFF80

constexpr int32_t ONE = 1;
constexpr int32_t TWO = 2;
constexpr int32_t THREE = 3;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;
constexpr int32_t SIX = 6;
constexpr int32_t SEVEN = 7;
constexpr int32_t EIGHT = 8;
constexpr int32_t NINE = 9;
constexpr int32_t TEN = 10;
constexpr int32_t ELEVEN = 11;
constexpr int32_t TWELVE = 12;
constexpr int32_t THIRTEEN = 13;
constexpr int32_t FOURTEEN = 14;
constexpr int32_t FIFTEEN = 15;
constexpr int32_t SIXTEEN = 16;

constexpr int32_t MIN_INT32 = -2147483648;
constexpr int64_t MIN_INT64 = -9223372036854775807LL - 1;
constexpr uint8_t MIN_UINT8 = 0;
constexpr int16_t MIN_INT16 = -32768;
constexpr int8_t MIN_INT8 = -128;
constexpr uint16_t MIN_UINT16 = 0;
constexpr int32_t INDEX_SIZE = 256;
constexpr int32_t B64 = 8;
constexpr int32_t B8 = 1;
constexpr int32_t B16 = 2;
constexpr int32_t B32 = 4;

constexpr int32_t GATHER_SINGLE_ROW = 0;
constexpr int32_t GATHER_MULTI_ROW = 1;
constexpr int32_t GATHER_MULTI_BATCH = 2;
constexpr int32_t GATHER_SINGLE_KERNEL = 3;
constexpr int32_t NOT_GATHER = 1001;

constexpr int32_t SCATTER_SINGLE_ROW = 0;
constexpr int32_t SCATTER_MULTI_ROW = 1;
constexpr int32_t COPY_SINGLE_ROW = 2;

constexpr int32_t SPLIT_COLS = 1;
constexpr int32_t SPLIT_ROWS = 2;
constexpr int32_t SPLIT_BATCHS = 3;
constexpr uint16_t INT64_MAXREGNUM = 8;

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

template <typename T>
struct GetComputeType {
    using type = typename std::conditional<IsSame<T, bool>::value, int8_t, T>::type;
};

template <typename T>
struct GetGatherType {
    using type = typename std::conditional<
        IsSame<T, int8_t>::value, int16_t,
        typename std::conditional<IsSame<T, uint8_t>::value, uint16_t, T>::type>::type;
};

template <typename T>
struct VciTypeGet {
    using type = typename std::conditional<
        IsSame<T, uint32_t>::value, int32_t,
        typename std::conditional<
            IsSame<T, uint16_t>::value, int16_t,
            typename std::conditional<IsSame<T, uint64_t>::value, int64_t, T>::type>::type>::type;
};

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == B8 || sizeof(T) == B16, uint16_t, uint32_t>::type;
};

constexpr MicroAPI::CastTrait castTraitB82B16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr AscendC::MicroAPI::CastTrait castTraitB162B8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void StoreElement(
    const __local_mem__ void* output, MicroAPI::RegTensor<T>& src, uint32_t offset, uint32_t element)
{
    MicroAPI::UnalignReg u0;
    auto dstAddr = (__local_mem__ T*)(output) + offset;
    MicroAPI::DataCopyUnAlign(dstAddr, src, u0, element);
    MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename RegDstT>
__aicore__ inline void LoadOneElement(const __local_mem__ void* input, RegDstT& dst, uint32_t offset)
{
    MicroAPI::UnalignReg u0;
    auto srcAddr = (__local_mem__ T*)(input) + offset;
    MicroAPI::DataCopyUnAlignPre(u0, srcAddr);
    MicroAPI::DataCopyUnAlign(dst, u0, srcAddr, 1);
}

template <typename T>
__aicore__ inline constexpr T GetNegInf()
{
    T negInf = 0;
    if constexpr (IsSame<T, int32_t>::value) {
        negInf = MIN_INT32;
    } else if constexpr (IsSame<T, int64_t>::value) {
        negInf = MIN_INT64;
    } else if constexpr (IsSame<T, uint8_t>::value) {
        negInf = MIN_UINT8;
    } else if constexpr (IsSame<T, int16_t>::value) {
        negInf = MIN_INT16;
    } else if constexpr (IsSame<T, int8_t>::value) {
        negInf = MIN_INT8;
    } else if constexpr (IsSame<T, uint16_t>::value) {
        negInf = MIN_UINT16;
    } else if constexpr (IsSame<T, float>::value) {
        negInf = *reinterpret_cast<const float*>(&FLOAT32_NEG_INF);
    } else {
        negInf = *reinterpret_cast<const half*>(&FLOAT16_NEG_INF);
    }
    return negInf;
}

template <typename T>
__aicore__ inline void DuplicateNegInf(const __local_mem__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    uint32_t num = calNum;
    MicroAPI::RegTensor<T> v0;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T>(num);
    MicroAPI::UnalignReg u0;
    if constexpr (IsSame<T, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF, p0);
    } else {
        T value = GetNegInf<T>();
        MicroAPI::Duplicate(v0, value, p0);
    }
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr + offset;
    MicroAPI::DataCopyUnAlign(addr, v0, u0, calNum);
    MicroAPI::DataCopyUnAlignPost(addr, u0, 0);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T>
__aicore__ inline void CustomDuplicate(__local_mem__ T* dstAddr, uint32_t calNum, uint16_t loop)
{
    uint32_t sreg = calNum;
    using RegDstT = typename std::conditional<
        sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<T>>::type;
    RegDstT v0;
    if constexpr (IsSame<T, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        MicroAPI::Duplicate(v0, value);
    }
    if constexpr (sizeof(T) == B64) {
        constexpr uint16_t repeatElm = TWO * Ops::Base::GetVRegSize() / sizeof(T);
        for (uint16_t i = 0; i < loop; i++) {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
            MicroAPI::DataCopy(dstAddr + i * repeatElm, v0, preg);
        }
    } else {
        constexpr uint16_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
        for (uint16_t i = 0; i < loop; i++) {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatElm);
            MicroAPI::DataCopy(dstAddr, v0, offset, preg);
        }
    }
}

template <typename T>
__aicore__ inline void CustomCopy(
    const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr, uint32_t srcBatchStride, uint32_t srcRowStride,
    uint32_t dstBatchStride, uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset, uint16_t batch,
    uint16_t rows, uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm)
{
    using RegDstT = typename std::conditional<
        sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<T>>::type;
    RegDstT v0;
    MicroAPI::UnalignReg u0;

    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __local_mem__ T* curSrcAddr = (__local_mem__ T*)srcAddr + i * srcBatchStride + j * srcRowStride;
            __local_mem__ T* curDstAddr =
                (__local_mem__ T*)dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride + dstColOffset;
            for (uint16_t k = 0; k < loopCols; k++) {
                MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
            }
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
            MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, tailCols);
            MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterSingleRow(
    const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr, uint16_t srcBatchStride, uint16_t srcRowStride,
    uint16_t dstBatchStride, uint16_t dstRowStride, uint16_t dstRowOffset, uint16_t dstColOffset, uint16_t batch,
    uint16_t rows, uint16_t loopCols, uint16_t cols, uint16_t repeatElm)
{
    using M = typename GetGatherType<T>::type;
    using RegDstT = typename std::conditional<
        sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<T>>::type;
    RegDstT v0;
    MicroAPI::RegTensor<U> sIndex;
    MicroAPI::MaskReg preg;
    using regType = typename VciTypeGet<U>::type;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)sIndex, 0);
    auto dstAddr1 = (__local_mem__ T*)dstAddr + dstRowOffset * dstRowStride + dstColOffset;
    for (uint16_t i = 0; i < batch; i++) {
        auto dstAddr2 = dstAddr1 + i * dstBatchStride;
        auto srcAddr1 = (__local_mem__ T*)srcAddr + i * srcBatchStride;
        uint32_t sreg = cols;
        for (uint16_t j = 0; j < loopCols; j++) {
            auto curDstAddr = dstAddr2 + j * repeatElm;
            auto curSrcAddr = srcAddr1 + j * repeatElm;
            if constexpr (sizeof(T) == B64) {
                preg = MicroAPI::UpdateMask<U, MicroAPI::RegTraitNumTwo>(sreg);
            } else {
                preg = MicroAPI::UpdateMask<U>(sreg);
            }
            for (uint16_t k = 0; k < rows; k++) {
                if constexpr (sizeof(T) == B8) {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B8>(v0, curSrcAddr + k * srcRowStride);
                } else {
                    MicroAPI::DataCopy(v0, curSrcAddr + k * srcRowStride);
                }
                MicroAPI::DataCopyScatter(curDstAddr + k * dstRowStride, v0, sIndex, preg);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterMultiRows(
    const __local_mem__ T* dstAddr, const __local_mem__ T* srcAddr, MicroAPI::RegTensor<U> index,
    uint32_t srcBatchStride, uint32_t srcRowStride, uint32_t dstBatchStride, uint32_t dstRowStride, uint32_t dstOffset,
    uint16_t batch, uint16_t loopRows, uint32_t repeatElm, uint32_t tailElm)
{
    using M = typename GetGatherType<T>::type;
    using RegDstT = typename std::conditional<
        sizeof(M) == B64, MicroAPI::RegTensor<M, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<M>>::type;
    RegDstT vd1;
    MicroAPI::RegTensor<U> v1, v2, v3, v4;
    MicroAPI::RegTensor<U> gIndex;
    uint32_t sreg = repeatElm;
    uint32_t tailSreg = tailElm;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg tailPreg;
    if constexpr (sizeof(T) == B64) {
        preg = MicroAPI::UpdateMask<M, MicroAPI::RegTraitNumTwo>(sreg);
        tailPreg = MicroAPI::UpdateMask<M, MicroAPI::RegTraitNumTwo>(tailSreg);
    } else {
        preg = MicroAPI::UpdateMask<U>(sreg);
        tailPreg = MicroAPI::UpdateMask<U>(tailSreg);
    }
    using regType = typename VciTypeGet<U>::type;
    MicroAPI::RegTensor<U> vd0;
    MicroAPI::Arange((MicroAPI::RegTensor<regType>&)gIndex, 0);
    __local_mem__ T* curDstAddr = (__local_mem__ T*)dstAddr + dstOffset;
    for (uint16_t i = 0; i < batch; i++) {
        MicroAPI::Adds(v1, index, i * dstBatchStride, maskAll);
        MicroAPI::Adds(v3, gIndex, i * srcBatchStride, maskAll);
        for (uint16_t j = 0; j < loopRows; j++) {
            MicroAPI::Adds(v2, v1, j * dstRowStride, preg);
            MicroAPI::Adds(v4, v3, j * srcRowStride, preg);
            MicroAPI::DataCopyGather(vd1, srcAddr, v4, preg);
            if constexpr (sizeof(T) == B8) {
                MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v2, preg);
            } else {
                MicroAPI::DataCopyScatter(curDstAddr, vd1, v2, preg);
            }
        }
        MicroAPI::Adds(v2, v1, loopRows * dstRowStride, tailPreg);
        MicroAPI::Adds(v4, v3, loopRows * srcRowStride, tailPreg);
        MicroAPI::DataCopyGather(vd1, srcAddr, v4, tailPreg);
        if constexpr (sizeof(T) == B8) {
            MicroAPI::DataCopyScatter(curDstAddr, (MicroAPI::RegTensor<T>&)vd1, v2, tailPreg);
        } else {
            MicroAPI::DataCopyScatter(curDstAddr, vd1, v2, tailPreg);
        }
    }
}

template <typename T, typename U, typename RegDstT>
__aicore__ inline void MaxPoolImpl(
    RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kH, uint16_t kW, U rowStrideInub,
    MicroAPI::MaskReg& pMask, uint16_t channels = 1)
{
    using gatherType = typename GetGatherType<T>::type;
    MicroAPI::RegTensor<gatherType> vd0;
    RegDstT vd1;
    MicroAPI::RegTensor<U> v0;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    if constexpr (IsSame<T, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        MicroAPI::Duplicate(res, value);
    }
    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        MicroAPI::Adds(v0, index, hIdx * rowStrideInub, pMask);
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            MicroAPI::Adds(v1, v0, wIdx * channels, pMask);
            if constexpr (sizeof(T) == 1) {
                MicroAPI::DataCopyGather(vd0, srcAddr, v1, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint8_t>&)vd1, vd0);
                MicroAPI::Max(res, vd1, res, maskAll);
            } else {
                MicroAPI::DataCopyGather(vd1, srcAddr, v1, pMask);
                MicroAPI::Max(res, vd1, res, pMask);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MaxPoolSingleChannel(
    __local_mem__ T* dstLocalAddr, __local_mem__ T* srcLocalAddr, uint16_t kH, uint16_t kW, U rowStrideInub,
    uint16_t alignChannels, uint16_t repeatElms)
{
    using RegDstT = typename std::conditional<
        sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<T>>::type;
    RegDstT res;
    RegDstT vd0;
    uint32_t num = repeatElms;
    MicroAPI::MaskReg p0;
    MicroAPI::UnalignReg u0;
    __local_mem__ T* curSrcAddr = srcLocalAddr;

    if constexpr (IsSame<T, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        MicroAPI::Duplicate(res, value);
    }
    if constexpr (sizeof(T) == B64) {
        p0 = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(num);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                auto srcAddr = curSrcAddr + hIdx * rowStrideInub + wIdx * alignChannels;
                MicroAPI::DataCopy(vd0, srcAddr);
                MicroAPI::Max(res, vd0, res, p0);
            }
        }
    } else {
        p0 = MicroAPI::UpdateMask<T>(num);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                auto aReg = MicroAPI::CreateAddrReg<T>(hIdx, rowStrideInub, wIdx, alignChannels);
                MicroAPI::DataCopy(vd0, curSrcAddr, aReg);
                MicroAPI::Max(res, vd0, res, p0);
            }
        }
    }
    MicroAPI::DataCopy(dstLocalAddr, res, p0);
}

template <typename T, typename U>
__aicore__ inline void MaxPoolSplitW(
    __local_mem__ T* dstLocalAddr, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kH, uint16_t kW,
    uint16_t loopH, uint16_t loopW, U oneLoopStrideH, U oneLoopStrideW, U rowStrideInub, uint16_t oneLoopElements,
    uint16_t tailLoopElements, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<
        sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<T>>::type;
    RegDstT res;
    MicroAPI::RegTensor<U> v0;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<U> v2;
    MicroAPI::RegTensor<U> v3;
    MicroAPI::RegTensor<U> v4;
    MicroAPI::UnalignReg u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);
    __local_mem__ T* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopH; i++) {
        MicroAPI::Adds(v0, index, i * oneLoopStrideH, p0);
        for (uint16_t j = 0; j < loopW; j++) {
            MicroAPI::Adds(v2, v0, j * oneLoopStrideW, p0);
            MaxPoolImpl<T, U>(res, srcAddr, v2, kH, kW, rowStrideInub, p0, channels);
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, oneLoopElements);
        }
        MicroAPI::Adds(v2, v0, loopW * oneLoopStrideW, pTail);
        MaxPoolImpl<T, U>(res, srcAddr, v2, kH, kW, rowStrideInub, pTail, channels);
        MicroAPI::DataCopyUnAlign(dstAddr, res, u0, tailLoopElements);
    }
    MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename U>
__aicore__ inline void MaxPoolSplitH(
    __local_mem__ T* dstLocalAddr, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kH, uint16_t kW,
    uint16_t loopN, uint16_t loopH, U oneChannelElements, U rowStrideInub, U oneLoopStride, uint16_t oneLoopElements,
    uint16_t tailLoopElements, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<
        sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<T>>::type;
    RegDstT res;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<U> v2;
    MicroAPI::RegTensor<U> v3;
    MicroAPI::RegTensor<U> v4;
    MicroAPI::UnalignReg u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);
    __local_mem__ T* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopN; i++) {
        MicroAPI::Adds(v1, index, i * oneChannelElements, p0);
        for (uint16_t j = 0; j < loopH; j++) {
            MicroAPI::Adds(v2, v1, j * oneLoopStride, p0);
            MaxPoolImpl<T, U>(res, srcAddr, v2, kH, kW, rowStrideInub, p0, channels);
            MicroAPI::DataCopyUnAlign(dstAddr, res, u0, oneLoopElements);
        }
        MicroAPI::Adds(v2, v1, loopH * oneLoopStride, pTail);
        MaxPoolImpl<T, U>(res, srcAddr, v2, kH, kW, rowStrideInub, pTail, channels);
        MicroAPI::DataCopyUnAlign(dstAddr, res, u0, tailLoopElements);
    }
    MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename U>
__aicore__ inline void MaxPoolSplitBatch(
    __local_mem__ T* dstLocalAddr, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kH, uint16_t kW,
    uint16_t loopN, U rowStrideInub, U oneLoopStride, uint16_t oneLoopElements, uint16_t tailLoopElements,
    uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<
        sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<T>>::type;
    RegDstT res;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::UnalignReg u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);
    __local_mem__ T* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopN; i++) {
        MicroAPI::Adds(v1, index, i * oneLoopStride, p0);
        MaxPoolImpl<T, U>(res, srcAddr, v1, kH, kW, rowStrideInub, p0, channels);
        MicroAPI::DataCopyUnAlign(dstAddr, res, u0, oneLoopElements);
    }
    MicroAPI::Adds(v1, index, loopN * oneLoopStride, pTail);
    MaxPoolImpl<T, U>(res, srcAddr, v1, kH, kW, rowStrideInub, pTail, channels);
    MicroAPI::DataCopyUnAlign(dstAddr, res, u0, tailLoopElements);
    MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
}

template <typename U>
__aicore__ inline void GenGatherIndexMultiBatch(
    uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElemtsIn, uint32_t wIn, uint32_t hStride, uint32_t wStride,
    LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U batchElemtsOut = hFactorOut * wFactorOut;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> v4;
        MicroAPI::RegTensor<U> v5;
        MicroAPI::RegTensor<U> v6;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::RegTensor<U> vd8;
        MicroAPI::RegTensor<U> vd9;
        MicroAPI::RegTensor<U> vd10;
        MicroAPI::RegTensor<U> vd11;
        MicroAPI::RegTensor<U> vd12;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wFactorOut, p0);
        MicroAPI::Duplicate(v2, (U)wIn, p0);
        MicroAPI::Duplicate(v3, (U)hStride, p0);
        MicroAPI::Duplicate(v4, (U)wStride, p0);
        MicroAPI::Duplicate(v5, (U)batchElemtsIn, p0);
        MicroAPI::Duplicate(v6, (U)batchElemtsOut, p0);

        MicroAPI::Div(vd1, v0, v6, p0);  // i / (rows * cols)
        MicroAPI::Mul(vd2, vd1, v5, p0); // i / (rows * cols) * batchElemtsIn
        MicroAPI::Mul(vd3, vd1, v6, p0); // (i / wFactorOut * wIn * hStride)
        MicroAPI::Sub(vd4, v0, vd3, p0); // i % (rows * cols)

        MicroAPI::Div(vd5, vd4, v1, p0);    // hwoffset / cols
        MicroAPI::Mul(vd6, vd5, v2, p0);    // hwoffset / cols * wIn
        MicroAPI::Mul(vd7, vd6, v3, p0);    // hwoffset / cols * wIn * hStride
        MicroAPI::Mul(vd8, vd5, v1, p0);    // hwoffset / cols * cols
        MicroAPI::Sub(vd9, vd4, vd8, p0);   // hwoffset % cols
        MicroAPI::Mul(vd10, vd9, v4, p0);   // hwoffset % cols * wStride
        MicroAPI::Add(vd11, vd7, vd10, p0); // hwoffset / cols * wIn * hStride + hwoffset % cols * wStride
        MicroAPI::Add(vd12, vd2, vd11, p0);
        MicroAPI::DataCopy(dstAddr, vd12, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexMultiRow(
    uint32_t wFactorOut, uint32_t wIn, uint32_t hStride, uint32_t wStride, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> v4;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wFactorOut, p0);
        MicroAPI::Duplicate(v2, (U)wIn, p0);
        MicroAPI::Duplicate(v3, (U)hStride, p0);
        MicroAPI::Duplicate(v4, (U)wStride, p0);

        MicroAPI::Div(vd1, v0, v1, p0);   // i / wFactorOut
        MicroAPI::Mul(vd2, vd1, v2, p0);  // (i / wFactorOut * wIn)
        MicroAPI::Mul(vd3, vd2, v3, p0);  // (i / wFactorOut * wIn * hStride)
        MicroAPI::Mul(vd4, vd1, v1, p0);  // (i / wFactorOut * wFactorOut)
        MicroAPI::Sub(vd5, v0, vd4, p0);  // i % wFactor
        MicroAPI::Mul(vd6, vd5, v4, p0);  // i % wFactorOut * wStride
        MicroAPI::Add(vd7, vd3, vd6, p0); // (i / wFactorOut * wIn * hStride + i % wFactorOut * wStride)
        MicroAPI::DataCopy(dstAddr, vd7, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexSingleRow(uint32_t wStride, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    // i * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wStride, p0);
        MicroAPI::Mul(vd0, v0, v1, p0); // (i / wFactorOut * wIn)
        MicroAPI::DataCopy(dstAddr, vd0, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexSingleKernel(uint32_t wIn, uint32_t kW, uint32_t kH, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    uint16_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t loopNum = (kW * kH + repeatNum - 1) / repeatNum;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, i * repeatNum);
            MicroAPI::Duplicate(v1, (U)kW, p0);
            MicroAPI::Duplicate(v2, (U)wIn, p0);

            MicroAPI::Div(vd1, v0, v1, p0);
            MicroAPI::Mul(vd2, vd1, v2, p0);
            MicroAPI::Mul(vd3, vd1, v1, p0);
            MicroAPI::Sub(vd4, v0, vd3, p0);
            MicroAPI::Add(vd5, vd2, vd4, p0);
            MicroAPI::DataCopy(dstAddr + i * repeatNum, vd5, p0);
        }
    }
}

template <typename U, bool SingleRow>
__aicore__ inline void GenScatterIndex(uint32_t wIn, uint32_t wInDst, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;

        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        if constexpr (SingleRow) {
            MicroAPI::DataCopy(dstAddr, v0, p0);
        } else {
            MicroAPI::Duplicate(v1, (U)wIn, p0);
            MicroAPI::Duplicate(v2, (U)wInDst, p0);

            MicroAPI::Div(vd1, v0, v1, p0);
            MicroAPI::Mul(vd2, vd1, v2, p0);
            MicroAPI::Mul(vd3, vd1, v1, p0);
            MicroAPI::Sub(vd4, v0, vd3, p0);
            MicroAPI::Add(vd5, vd2, vd4, p0);
            MicroAPI::DataCopy(dstAddr, vd5, p0);
        }
    }
}

template <typename U, bool SingleRow>
__aicore__ inline void NHWCGenScatterIndex(
    uint32_t wIn, uint32_t wInDstElms, uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::RegTensor<U> vd8;
        MicroAPI::RegTensor<U> vd9;
        MicroAPI::RegTensor<U> vd10;

        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        if constexpr (SingleRow) {
            MicroAPI::DataCopy(dstAddr, v0, p0);
        } else {
            MicroAPI::Duplicate(v1, (U)wIn, p0);
            MicroAPI::Duplicate(v2, (U)wInDstElms, p0);
            MicroAPI::Duplicate(v3, (U)channels, p0);

            MicroAPI::Div(vd1, v0, v3, p0);  // i / channels
            MicroAPI::Div(vd2, vd1, v1, p0); // i / channels / win
            MicroAPI::Mul(vd3, vd2, v2, p0); // i / channels / win * winDst

            MicroAPI::Mul(vd4, vd2, v1, p0);  // i / channels / win * win
            MicroAPI::Sub(vd5, vd1, vd4, p0); // i / channels mod win
            MicroAPI::Mul(vd6, vd5, v3, p0);  // ( i / channels mod win) * channels
            MicroAPI::Add(vd7, vd3, vd6, p0); // i / channels / win * winDst + i / channels mod win * channels

            MicroAPI::Mul(vd8, vd1, v3, p0);
            MicroAPI::Sub(vd9, v0, vd8, p0); // i mod channels

            MicroAPI::Add(
                vd10, vd9, vd7, p0); // (i / channels / win * winDst + i / channels mod win) * channels + i mod channels
            MicroAPI::DataCopy(dstAddr, vd10, p0);
        }
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexSingleRow(uint32_t wStride, uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    // i * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<regType> tmp;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;

        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wStride, p0);
        MicroAPI::Duplicate(v2, (U)channels, p0); // channels
        MicroAPI::Div(vd0, v0, v2, p0);           // i / channels
        MicroAPI::Mul(vd1, vd0, v2, p0);
        MicroAPI::Sub(vd5, v0, vd1, p0);  // i % channel
        MicroAPI::Mul(vd2, vd0, v1, p0);  // (i / channel * wstride)
        MicroAPI::Mul(vd3, vd2, v2, p0);  // (i / channel * wstride * channels)
        MicroAPI::Add(vd4, vd3, vd5, p0); // (i / channel * wstride * channels) + i % channel
        MicroAPI::DataCopy(dstAddr, vd4, p0);
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexMultiRow(
    uint32_t wFactorOut, uint32_t wInElms, uint32_t hStride, uint32_t wStride, uint32_t channels,
    LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> v4;
        MicroAPI::RegTensor<U> v5;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd3;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd7;
        MicroAPI::RegTensor<U> vd8;
        MicroAPI::RegTensor<U> vd9;
        MicroAPI::RegTensor<U> vd10;
        MicroAPI::RegTensor<U> vd11;
        MicroAPI::RegTensor<U> vd12;
        MicroAPI::RegTensor<U> vd13;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wFactorOut, p0);
        MicroAPI::Duplicate(v2, (U)wInElms, p0);
        MicroAPI::Duplicate(v3, (U)hStride, p0);
        MicroAPI::Duplicate(v4, (U)wStride, p0);
        MicroAPI::Duplicate(v5, (U)channels, p0);

        MicroAPI::Div(vd1, v0, v5, p0);  // i / channels
        MicroAPI::Div(vd2, vd1, v1, p0); // i / channels / wFactorOut
        MicroAPI::Mul(vd3, vd2, v2, p0); // (i  / channels / wFactorOut * wIn)
        MicroAPI::Mul(vd4, vd3, v3, p0); // (i / channels / wFactorOut * wIn * hStride

        MicroAPI::Mul(vd5, vd2, v1, p0);  // (i / channels / wFactorOut * wFactorOut)
        MicroAPI::Sub(vd6, vd1, vd5, p0); // (i  / channels) % wFactor
        MicroAPI::Mul(vd7, vd6, v4, p0);  // (i  / channels) % wFactorOut * wStride
        MicroAPI::Mul(vd8, vd7, v5, p0);  // ( i  / channels) % wFactorOut * wStride) * channels

        MicroAPI::Add(
            vd9, vd8, vd4,
            p0); // (i  / channels) / wFactorOut * wIn * hStride + (i  / channels) % wFactorOut * wStride* channels)
        MicroAPI::Mul(vd11, vd1, v5, p0);  // i / channels * channels
        MicroAPI::Sub(vd12, v0, vd11, p0); // i mod channel
        MicroAPI::Add(vd13, vd9, vd12, p0);
        MicroAPI::DataCopy(dstAddr, vd13, p0);
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexMultiBatch(
    uint32_t hFactorOut, uint32_t wFactorOut, uint32_t hIn, uint32_t wInElms, uint32_t hStride, uint32_t wStride,
    uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U batchElemtsIn = hIn * wInElms;
    U batchElemtsOut = hFactorOut * wFactorOut * channels;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        MicroAPI::RegTensor<U> v0;
        MicroAPI::RegTensor<U> v1;
        MicroAPI::RegTensor<U> v2;
        MicroAPI::RegTensor<U> v3;
        MicroAPI::RegTensor<U> v4;
        MicroAPI::RegTensor<U> v5;
        MicroAPI::RegTensor<U> v6;
        MicroAPI::RegTensor<U> v7;

        MicroAPI::RegTensor<U> vd0;
        MicroAPI::RegTensor<U> vd1;
        MicroAPI::RegTensor<U> vd2;
        MicroAPI::RegTensor<U> vd4;
        MicroAPI::RegTensor<U> vd5;
        MicroAPI::RegTensor<U> vd6;
        MicroAPI::RegTensor<U> vd8;
        MicroAPI::RegTensor<U> vd12;
        MicroAPI::RegTensor<U> vd14;
        MicroAPI::RegTensor<U> vd17;
        MicroAPI::RegTensor<U> vd18;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Arange((MicroAPI::RegTensor<regType>&)v0, 0);
        MicroAPI::Duplicate(v1, (U)wFactorOut, p0);
        MicroAPI::Duplicate(v2, (U)wInElms, p0);
        MicroAPI::Duplicate(v3, (U)hStride, p0);
        MicroAPI::Duplicate(v4, (U)wStride, p0);
        MicroAPI::Duplicate(v5, (U)channels, p0);
        MicroAPI::Duplicate(v6, (U)batchElemtsIn, p0);
        MicroAPI::Duplicate(v7, (U)batchElemtsOut, p0);

        MicroAPI::Div(vd1, v0, v7, p0);  // i / (rows * cols * channels)
        MicroAPI::Mul(vd2, vd1, v6, p0); // i / (rows * cols * channels) * batchElemtsIn       n

        MicroAPI::Mul(vd4, vd1, v7, p0); // (i / (rows * cols * channels) * (rows * cols * channels)
        MicroAPI::Sub(vd4, v0, vd4, p0); // i % (rows * cols *channels)

        MicroAPI::Div(vd5, vd4, v5, p0); // hwoffset / channels
        MicroAPI::Div(vd6, vd5, v1, p0); // hwoffset / channels / wfout
        MicroAPI::Mul(vd8, vd6, v2, p0); // hwoffset / channels / wfout * win
        MicroAPI::Mul(vd8, vd8, v3, p0); // hwoffset / channels / wfout * hstride  h

        MicroAPI::Mul(vd12, vd6, v1, p0);   // hwoffset / channels / wfout * wfout
        MicroAPI::Sub(vd12, vd5, vd12, p0); // hwoffset / channels % wfout
        MicroAPI::Mul(vd12, vd12, v4, p0);  // hwoffset / channels % wfout * wstride
        MicroAPI::Mul(vd12, vd12, v5, p0);  // (hwoffset / channels % wfout * wstride) * channels

        MicroAPI::Add(
            vd14, vd12, vd8, p0); // hwoffset / channels / wfout * hstride + hwoffset / channels % wfout * wstride
        MicroAPI::Add(vd14, vd14, vd2, p0); // (hwoffset / channels / wfout * hstride + hwoffset / channels / wfout *
                                            // wstride) * channels + i / (rows * cols * channels) * batchElemtsIn

        MicroAPI::Div(vd17, v0, v5, p0);   // i / channels
        MicroAPI::Mul(vd17, vd17, v5, p0); // i / channels * channels
        MicroAPI::Sub(vd17, v0, vd17, p0); // i % channels

        MicroAPI::Add(vd18, vd14, vd17, p0);
        MicroAPI::DataCopy(dstAddr, vd18, p0);
    }
}

template <typename T, typename RegDstT>
__aicore__ inline void MergeMaxRes(RegDstT& res, const __local_mem__ T* dstLocalAddr, int32_t offset)
{
    // merge cur result with pre result
    MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
    RegDstT lastRes;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    LoadOneElement<T>(dstLocalAddr, lastRes, offset);
    MicroAPI::Max(res, res, lastRes, pregOne); // nan index
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

template <typename T, typename RegDstT>
__aicore__ inline void ReduceMaxAll(RegDstT& res, RegDstT& src, MicroAPI::MaskReg& maskAll)
{
    if constexpr (sizeof(T) == 1) {
        MicroAPI::RegTensor<T> left;
        MicroAPI::RegTensor<T> right;
        MicroAPI::RegTensor<half> dst1;
        MicroAPI::RegTensor<half> dst2;
        MicroAPI::Interleave(left, right, src, src);
        MicroAPI::Cast<half, T, castTraitB82B16>(dst1, left, maskAll);
        MicroAPI::Cast<half, T, castTraitB82B16>(dst2, right, maskAll);
        MicroAPI::Max(dst1, dst1, dst2, maskAll);
        MicroAPI::ReduceMax(dst1, dst1, maskAll);
        MicroAPI::Cast<T, half, castTraitB162B8>(res, dst1, maskAll);
    } else if constexpr (IsSame<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<T> left;
        MicroAPI::RegTensor<T> right;
        MicroAPI::RegTensor<float> dst1;
        MicroAPI::RegTensor<float> dst2;
        MicroAPI::Interleave(left, right, src, src);
        MicroAPI::Cast<float, T, castTraitB82B16>(dst1, left, maskAll);
        MicroAPI::Cast<float, T, castTraitB82B16>(dst2, right, maskAll);
        MicroAPI::Max(dst1, dst1, dst2, maskAll);
        MicroAPI::ReduceMax(dst1, dst1, maskAll);
        MicroAPI::Cast<T, float, castTraitB162B8>(res, dst1, maskAll);
    } else {
        MicroAPI::ReduceMax(res, src, maskAll);
    }
}

template <bool MaskMergeMode, typename T, typename U, typename RegDstT>
__aicore__ inline void MaxWithGather(
    RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, MicroAPI::MaskReg& mask)
{
    RegDstT vd1;
    if constexpr (sizeof(T) == 1) {
        using gatherType = typename GetGatherType<T>::type;
        MicroAPI::RegTensor<gatherType> vd0;
        MicroAPI::DataCopyGather(vd0, srcAddr, index, mask);
        MicroAPI::Pack((MicroAPI::RegTensor<uint8_t>&)vd1, vd0);
        MicroAPI::MaskReg pMask;
        MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(pMask, mask);
        if constexpr (MaskMergeMode) {
            RegDstT tmp;
            MicroAPI::Max(tmp, vd1, res, pMask);
            MicroAPI::Copy<T, MicroAPI::MaskMergeMode::MERGING>(res, tmp, pMask);
        } else {
            MicroAPI::Max(res, vd1, res, pMask);
        }
    } else {
        MicroAPI::DataCopyGather(vd1, srcAddr, index, mask);
        if constexpr (MaskMergeMode) {
            RegDstT tmp;
            MicroAPI::Max(tmp, vd1, res, mask);
            MicroAPI::Copy<T, MicroAPI::MaskMergeMode::MERGING>(res, tmp, mask);
        } else {
            MicroAPI::Max(res, vd1, res, mask);
        }
    }
}

template <typename T, typename RegDstT>
__aicore__ inline void DuplicateNegInf(RegDstT& v0)
{
    if constexpr (IsSame<T, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        MicroAPI::Duplicate(v0, value);
    }
}

template <uint16_t REG_NUM, uint16_t IDX, typename U>
__aicore__ inline void LoadIndex(__local_mem__ U* indexAddr, MicroAPI::RegTensor<U>& index)
{
    constexpr uint32_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
    if constexpr (REG_NUM > IDX) {
        MicroAPI::DataCopy(index, indexAddr + IDX * repeatNum);
    }
}

template <uint16_t REG_NUM, uint16_t IDX, typename U, typename T, typename RegDstT>
__aicore__ inline void ComputeMaxWithGather(
    RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, MicroAPI::MaskReg& mask)
{
    if constexpr (REG_NUM > IDX) {
        MaxWithGather<false>(res, srcAddr, index, mask);
    }
}

template <typename T, typename U, typename RegDstT>
__aicore__ inline void GatherCommon(
    RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, MicroAPI::MaskReg& mask)
{
    if constexpr (sizeof(T) == 1) {
        using gatherType = typename GetGatherType<T>::type;
        MicroAPI::RegTensor<gatherType> vd0;
        MicroAPI::DataCopyGather(vd0, srcAddr, index, mask);
        MicroAPI::Pack((MicroAPI::RegTensor<uint8_t>&)res, vd0);
    } else {
        MicroAPI::DataCopyGather(res, srcAddr, index, mask);
    }
}

template <typename T, typename U, uint16_t REG_NUM>
__aicore__ inline void MaxPoolSingleKernelCommon(
    __local_mem__ T* dstLocalAddr, __local_mem__ T* xLocalAddr, __local_mem__ U* indexAddr, uint16_t loopN,
    uint16_t loopH, uint16_t loopW, U oneChannelElements, U oneLoopStrideH, U oneLoopStrideW, uint16_t tailLoopElements)
{
    if constexpr (sizeof(T) == sizeof(int64_t) && REG_NUM > INT64_MAXREGNUM) {
        return;
    }
    using RegDstT = typename std::conditional<
        sizeof(T) == B64, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTensor<T>>::type;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> index[SIXTEEN];
        MicroAPI::UnalignReg u0;
        uint32_t tailNum = tailLoopElements;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);

        MicroAPI::DataCopy(index[0], indexAddr);
        LoadIndex<REG_NUM, ONE>(indexAddr, index[ONE]);
        LoadIndex<REG_NUM, TWO>(indexAddr, index[TWO]);
        LoadIndex<REG_NUM, THREE>(indexAddr, index[THREE]);
        LoadIndex<REG_NUM, FOUR>(indexAddr, index[FOUR]);
        LoadIndex<REG_NUM, FIVE>(indexAddr, index[FIVE]);
        LoadIndex<REG_NUM, SIX>(indexAddr, index[SIX]);
        LoadIndex<REG_NUM, SEVEN>(indexAddr, index[SEVEN]);
        LoadIndex<REG_NUM, EIGHT>(indexAddr, index[EIGHT]);
        LoadIndex<REG_NUM, NINE>(indexAddr, index[NINE]);
        LoadIndex<REG_NUM, TEN>(indexAddr, index[TEN]);
        LoadIndex<REG_NUM, ELEVEN>(indexAddr, index[ELEVEN]);
        LoadIndex<REG_NUM, TWELVE>(indexAddr, index[TWELVE]);
        LoadIndex<REG_NUM, THIRTEEN>(indexAddr, index[THIRTEEN]);
        LoadIndex<REG_NUM, FOURTEEN>(indexAddr, index[FOURTEEN]);
        LoadIndex<REG_NUM, FIFTEEN>(indexAddr, index[FIFTEEN]);
        __local_mem__ T* dstAddr = dstLocalAddr;
        for (uint16_t i = 0; i < loopN; i++) {
            __local_mem__ T* srcAddr = xLocalAddr + i * oneChannelElements;
            for (uint16_t j = 0; j < loopH; j++) {
                __local_mem__ T* srcAddrH = srcAddr + j * oneLoopStrideH;
                for (uint16_t k = 0; k < loopW; k++) {
                    __local_mem__ T* srcAddrW = srcAddrH + k * oneLoopStrideW;
                    RegDstT res;
                    RegDstT tmp;
                    if constexpr (REG_NUM == 1) {
                        DuplicateNegInf<T, RegDstT>(res);
                        MaxWithGather<true>(res, srcAddrW, index[0], pTail);
                        ReduceMaxAll<T>(tmp, res, maskAll);
                    } else {
                        GatherCommon(res, srcAddrW, index[0], maskAll);
                        ComputeMaxWithGather<REG_NUM, TWO>(res, srcAddrW, index[ONE], maskAll);
                        ComputeMaxWithGather<REG_NUM, THREE>(res, srcAddrW, index[TWO], maskAll);
                        ComputeMaxWithGather<REG_NUM, FOUR>(res, srcAddrW, index[THREE], maskAll);
                        ComputeMaxWithGather<REG_NUM, FIVE>(res, srcAddrW, index[FOUR], maskAll);
                        ComputeMaxWithGather<REG_NUM, SIX>(res, srcAddrW, index[FIVE], maskAll);
                        ComputeMaxWithGather<REG_NUM, SEVEN>(res, srcAddrW, index[SIX], maskAll);
                        ComputeMaxWithGather<REG_NUM, EIGHT>(res, srcAddrW, index[SEVEN], maskAll);
                        ComputeMaxWithGather<REG_NUM, NINE>(res, srcAddrW, index[EIGHT], maskAll);
                        ComputeMaxWithGather<REG_NUM, TEN>(res, srcAddrW, index[NINE], maskAll);
                        ComputeMaxWithGather<REG_NUM, ELEVEN>(res, srcAddrW, index[TEN], maskAll);
                        ComputeMaxWithGather<REG_NUM, TWELVE>(res, srcAddrW, index[ELEVEN], maskAll);
                        ComputeMaxWithGather<REG_NUM, THIRTEEN>(res, srcAddrW, index[TWELVE], maskAll);
                        ComputeMaxWithGather<REG_NUM, FOURTEEN>(res, srcAddrW, index[THIRTEEN], maskAll);
                        ComputeMaxWithGather<REG_NUM, FIFTEEN>(res, srcAddrW, index[FOURTEEN], maskAll);
                        MaxWithGather<true>(res, srcAddrW, index[REG_NUM - 1], pTail);
                        ReduceMaxAll<T>(tmp, res, maskAll);
                    }

                    MicroAPI::DataCopyUnAlign(dstAddr, tmp, u0, 1);
                }
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstAddr, u0, 0);
    }
}

} // namespace MaxPoolV3
#endif // MAX_POOL_V3_COMMON_H_
