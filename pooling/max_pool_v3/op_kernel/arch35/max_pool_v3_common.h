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
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
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
        typename std::conditional<IsSame<T, uint16_t>::value, int16_t,
                                  typename std::conditional<IsSame<T, uint64_t>::value, int64_t, T>::type>::type>::type;
};

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == B8 || sizeof(T) == B16, uint16_t, uint32_t>::type;
};

constexpr Reg::CastTrait castTraitB82B16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait castTraitB162B8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void StoreElement(const __ubuf__ void* output, Reg::RegTensor<T>& src, uint32_t offset,
                                    uint32_t element)
{
    Reg::UnalignRegForStore u0;
    auto dstAddr = (__ubuf__ T*)(output) + offset;
    Reg::StoreUnAlign(dstAddr, src, u0, element);
    Reg::StoreUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename RegDstT>
__aicore__ inline void LoadOneElement(const __ubuf__ void* input, RegDstT& dst, uint32_t offset)
{
    Reg::UnalignRegForLoad u0;
    auto srcAddr = (__ubuf__ T*)(input) + offset;
    Reg::LoadUnAlignPre(u0, srcAddr);
    Reg::LoadUnAlign(dst, u0, srcAddr, 1);
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
__aicore__ inline void DuplicateNegInf(const __ubuf__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    uint32_t num = calNum;
    Reg::RegTensor<T> v0;
    Reg::MaskReg p0 = Reg::UpdateMask<T>(num);
    Reg::UnalignRegForStore u0;
    if constexpr (IsSame<T, bfloat16_t>::value) {
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF, p0);
    } else {
        T value = GetNegInf<T>();
        Reg::Duplicate(v0, value, p0);
    }
    __ubuf__ T* addr = (__ubuf__ T*)dstAddr + offset;
    Reg::StoreUnAlign(addr, v0, u0, calNum);
    Reg::StoreUnAlignPost(addr, u0, 0);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T>
__aicore__ inline void CustomDuplicate(__ubuf__ T* dstAddr, uint32_t calNum, uint16_t loop)
{
    uint32_t sreg = calNum;
    using RegDstT = typename std::conditional<sizeof(T) == B64, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<T>>::type;
    RegDstT v0;
    if constexpr (IsSame<T, bfloat16_t>::value) {
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        Reg::Duplicate(v0, value);
    }
    if constexpr (sizeof(T) == B64) {
        constexpr uint16_t repeatElm = TWO * Ops::Base::GetVRegSize() / sizeof(T);
        for (uint16_t i = 0; i < loop; i++) {
            Reg::MaskReg preg = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(sreg);
            Reg::StoreAlign(dstAddr + i * repeatElm, v0, preg);
        }
    } else {
        constexpr uint16_t repeatElm = Ops::Base::GetVRegSize() / sizeof(T);
        for (uint16_t i = 0; i < loop; i++) {
            Reg::MaskReg preg = Reg::UpdateMask<T>(sreg);
            Reg::AddrReg offset = Reg::CreateAddrReg<T>(i, repeatElm);
            Reg::StoreAlign(dstAddr, v0, offset, preg);
        }
    }
}

template <typename T>
__aicore__ inline void CustomCopy(const __ubuf__ T* dstAddr, const __ubuf__ T* srcAddr, uint32_t srcBatchStride,
                                  uint32_t srcRowStride, uint32_t dstBatchStride, uint32_t dstRowStride,
                                  uint32_t dstRowOffset, uint32_t dstColOffset, uint16_t batch, uint16_t rows,
                                  uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<T>>::type;
    RegDstT v0;
    Reg::UnalignRegForStore u0;

    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __ubuf__ T* curSrcAddr = (__ubuf__ T*)srcAddr + i * srcBatchStride + j * srcRowStride;
            __ubuf__ T* curDstAddr = (__ubuf__ T*)dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride +
                                     dstColOffset;
            for (uint16_t k = 0; k < loopCols; k++) {
                Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
            }
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
            Reg::StoreUnAlign(curDstAddr, v0, u0, tailCols);
            Reg::StoreUnAlignPost(curDstAddr, u0, 0);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterSingleRow(const __ubuf__ T* dstAddr, const __ubuf__ T* srcAddr,
                                                    uint16_t srcBatchStride, uint16_t srcRowStride,
                                                    uint16_t dstBatchStride, uint16_t dstRowStride,
                                                    uint16_t dstRowOffset, uint16_t dstColOffset, uint16_t batch,
                                                    uint16_t rows, uint16_t loopCols, uint16_t cols, uint16_t repeatElm)
{
    using M = typename GetGatherType<T>::type;
    using RegDstT = typename std::conditional<sizeof(T) == B64, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<T>>::type;
    RegDstT v0;
    Reg::RegTensor<U> sIndex;
    Reg::MaskReg preg;
    using regType = typename VciTypeGet<U>::type;
    Reg::Arange((Reg::RegTensor<regType>&)sIndex, 0);
    auto dstAddr1 = (__ubuf__ T*)dstAddr + dstRowOffset * dstRowStride + dstColOffset;
    for (uint16_t i = 0; i < batch; i++) {
        auto dstAddr2 = dstAddr1 + i * dstBatchStride;
        auto srcAddr1 = (__ubuf__ T*)srcAddr + i * srcBatchStride;
        uint32_t sreg = cols;
        for (uint16_t j = 0; j < loopCols; j++) {
            auto curDstAddr = dstAddr2 + j * repeatElm;
            auto curSrcAddr = srcAddr1 + j * repeatElm;
            if constexpr (sizeof(T) == B64) {
                preg = Reg::UpdateMask<U, Reg::RegTraitNumTwo>(sreg);
            } else {
                preg = Reg::UpdateMask<U>(sreg);
            }
            for (uint16_t k = 0; k < rows; k++) {
                if constexpr (sizeof(T) == B8) {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B8>(v0, curSrcAddr + k * srcRowStride);
                } else {
                    Reg::LoadAlign(v0, curSrcAddr + k * srcRowStride);
                }
                Reg::Scatter(curDstAddr + k * dstRowStride, v0, sIndex, preg);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void CustomCopyByScatterMultiRows(const __ubuf__ T* dstAddr, const __ubuf__ T* srcAddr,
                                                    Reg::RegTensor<U> index, uint32_t srcBatchStride,
                                                    uint32_t srcRowStride, uint32_t dstBatchStride,
                                                    uint32_t dstRowStride, uint32_t dstOffset, uint16_t batch,
                                                    uint16_t loopRows, uint32_t repeatElm, uint32_t tailElm)
{
    using M = typename GetGatherType<T>::type;
    using RegDstT = typename std::conditional<sizeof(M) == B64, Reg::RegTensor<M, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<M>>::type;
    RegDstT vd1;
    Reg::RegTensor<U> v1, v2, v3, v4;
    Reg::RegTensor<U> gIndex;
    uint32_t sreg = repeatElm;
    uint32_t tailSreg = tailElm;
    Reg::MaskReg maskAll = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
    Reg::MaskReg preg;
    Reg::MaskReg tailPreg;
    if constexpr (sizeof(T) == B64) {
        preg = Reg::UpdateMask<M, Reg::RegTraitNumTwo>(sreg);
        tailPreg = Reg::UpdateMask<M, Reg::RegTraitNumTwo>(tailSreg);
    } else {
        preg = Reg::UpdateMask<U>(sreg);
        tailPreg = Reg::UpdateMask<U>(tailSreg);
    }
    using regType = typename VciTypeGet<U>::type;
    Reg::RegTensor<U> vd0;
    Reg::Arange((Reg::RegTensor<regType>&)gIndex, 0);
    __ubuf__ T* curDstAddr = (__ubuf__ T*)dstAddr + dstOffset;
    for (uint16_t i = 0; i < batch; i++) {
        Reg::Adds(v1, index, i * dstBatchStride, maskAll);
        Reg::Adds(v3, gIndex, i * srcBatchStride, maskAll);
        for (uint16_t j = 0; j < loopRows; j++) {
            Reg::Adds(v2, v1, j * dstRowStride, preg);
            Reg::Adds(v4, v3, j * srcRowStride, preg);
            Reg::Gather(vd1, srcAddr, v4, preg);
            if constexpr (sizeof(T) == B8) {
                Reg::Scatter(curDstAddr, (Reg::RegTensor<T>&)vd1, v2, preg);
            } else {
                Reg::Scatter(curDstAddr, vd1, v2, preg);
            }
        }
        Reg::Adds(v2, v1, loopRows * dstRowStride, tailPreg);
        Reg::Adds(v4, v3, loopRows * srcRowStride, tailPreg);
        Reg::Gather(vd1, srcAddr, v4, tailPreg);
        if constexpr (sizeof(T) == B8) {
            Reg::Scatter(curDstAddr, (Reg::RegTensor<T>&)vd1, v2, tailPreg);
        } else {
            Reg::Scatter(curDstAddr, vd1, v2, tailPreg);
        }
    }
}

template <typename T, typename U, typename RegDstT>
__aicore__ inline void MaxPoolImpl(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index, uint16_t kH,
                                   uint16_t kW, U rowStrideInub, Reg::MaskReg& pMask, uint16_t channels = 1)
{
    using gatherType = typename GetGatherType<T>::type;
    Reg::RegTensor<gatherType> vd0;
    RegDstT vd1;
    Reg::RegTensor<U> v0;
    Reg::RegTensor<U> v1;
    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    if constexpr (IsSame<T, bfloat16_t>::value) {
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        Reg::Duplicate(res, value);
    }
    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        Reg::Adds(v0, index, hIdx * rowStrideInub, pMask);
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            Reg::Adds(v1, v0, wIdx * channels, pMask);
            if constexpr (sizeof(T) == 1) {
                Reg::Gather(vd0, srcAddr, v1, pMask);
                Reg::Pack((Reg::RegTensor<uint8_t>&)vd1, vd0);
                Reg::Max(res, vd1, res, maskAll);
            } else {
                Reg::Gather(vd1, srcAddr, v1, pMask);
                Reg::Max(res, vd1, res, pMask);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MaxPoolSingleChannel(__ubuf__ T* dstLocalAddr, __ubuf__ T* srcLocalAddr, uint16_t kH,
                                            uint16_t kW, U rowStrideInub, uint16_t alignChannels, uint16_t repeatElms)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<T>>::type;
    RegDstT res;
    RegDstT vd0;
    uint32_t num = repeatElms;
    Reg::MaskReg p0;
    Reg::UnalignRegForStore u0;
    __ubuf__ T* curSrcAddr = srcLocalAddr;

    if constexpr (IsSame<T, bfloat16_t>::value) {
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        Reg::Duplicate(res, value);
    }
    if constexpr (sizeof(T) == B64) {
        p0 = Reg::UpdateMask<T, Reg::RegTraitNumTwo>(num);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                auto srcAddr = curSrcAddr + hIdx * rowStrideInub + wIdx * alignChannels;
                Reg::LoadAlign(vd0, srcAddr);
                Reg::Max(res, vd0, res, p0);
            }
        }
    } else {
        p0 = Reg::UpdateMask<T>(num);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                auto aReg = Reg::CreateAddrReg<T>(hIdx, rowStrideInub, wIdx, alignChannels);
                Reg::LoadAlign(vd0, curSrcAddr, aReg);
                Reg::Max(res, vd0, res, p0);
            }
        }
    }
    Reg::StoreAlign(dstLocalAddr, res, p0);
}

template <typename T, typename U>
__aicore__ inline void MaxPoolSplitW(__ubuf__ T* dstLocalAddr, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                     uint16_t kH, uint16_t kW, uint16_t loopH, uint16_t loopW, U oneLoopStrideH,
                                     U oneLoopStrideW, U rowStrideInub, uint16_t oneLoopElements,
                                     uint16_t tailLoopElements, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<T>>::type;
    RegDstT res;
    Reg::RegTensor<U> v0;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<U> v2;
    Reg::RegTensor<U> v3;
    Reg::RegTensor<U> v4;
    Reg::UnalignRegForStore u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);
    __ubuf__ T* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopH; i++) {
        Reg::Adds(v0, index, i * oneLoopStrideH, p0);
        for (uint16_t j = 0; j < loopW; j++) {
            Reg::Adds(v2, v0, j * oneLoopStrideW, p0);
            MaxPoolImpl<T, U>(res, srcAddr, v2, kH, kW, rowStrideInub, p0, channels);
            Reg::StoreUnAlign(dstAddr, res, u0, oneLoopElements);
        }
        Reg::Adds(v2, v0, loopW * oneLoopStrideW, pTail);
        MaxPoolImpl<T, U>(res, srcAddr, v2, kH, kW, rowStrideInub, pTail, channels);
        Reg::StoreUnAlign(dstAddr, res, u0, tailLoopElements);
    }
    Reg::StoreUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename U>
__aicore__ inline void MaxPoolSplitH(__ubuf__ T* dstLocalAddr, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                     uint16_t kH, uint16_t kW, uint16_t loopN, uint16_t loopH, U oneChannelElements,
                                     U rowStrideInub, U oneLoopStride, uint16_t oneLoopElements,
                                     uint16_t tailLoopElements, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<T>>::type;
    RegDstT res;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<U> v2;
    Reg::RegTensor<U> v3;
    Reg::RegTensor<U> v4;
    Reg::UnalignRegForStore u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);
    __ubuf__ T* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopN; i++) {
        Reg::Adds(v1, index, i * oneChannelElements, p0);
        for (uint16_t j = 0; j < loopH; j++) {
            Reg::Adds(v2, v1, j * oneLoopStride, p0);
            MaxPoolImpl<T, U>(res, srcAddr, v2, kH, kW, rowStrideInub, p0, channels);
            Reg::StoreUnAlign(dstAddr, res, u0, oneLoopElements);
        }
        Reg::Adds(v2, v1, loopH * oneLoopStride, pTail);
        MaxPoolImpl<T, U>(res, srcAddr, v2, kH, kW, rowStrideInub, pTail, channels);
        Reg::StoreUnAlign(dstAddr, res, u0, tailLoopElements);
    }
    Reg::StoreUnAlignPost(dstAddr, u0, 0);
}

template <typename T, typename U>
__aicore__ inline void MaxPoolSplitBatch(__ubuf__ T* dstLocalAddr, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                         uint16_t kH, uint16_t kW, uint16_t loopN, U rowStrideInub, U oneLoopStride,
                                         uint16_t oneLoopElements, uint16_t tailLoopElements, uint16_t channels = 1)
{
    using RegDstT = typename std::conditional<sizeof(T) == B64, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<T>>::type;
    RegDstT res;
    Reg::RegTensor<U> v1;
    Reg::UnalignRegForStore u0;
    uint32_t num = oneLoopElements;
    uint32_t tailNum = tailLoopElements;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);
    __ubuf__ T* dstAddr = dstLocalAddr;
    for (uint16_t i = 0; i < loopN; i++) {
        Reg::Adds(v1, index, i * oneLoopStride, p0);
        MaxPoolImpl<T, U>(res, srcAddr, v1, kH, kW, rowStrideInub, p0, channels);
        Reg::StoreUnAlign(dstAddr, res, u0, oneLoopElements);
    }
    Reg::Adds(v1, index, loopN * oneLoopStride, pTail);
    MaxPoolImpl<T, U>(res, srcAddr, v1, kH, kW, rowStrideInub, pTail, channels);
    Reg::StoreUnAlign(dstAddr, res, u0, tailLoopElements);
    Reg::StoreUnAlignPost(dstAddr, u0, 0);
}

template <typename U>
__aicore__ inline void GenGatherIndexMultiBatch(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t batchElemtsIn,
                                                uint32_t wIn, uint32_t hStride, uint32_t wStride,
                                                LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U batchElemtsOut = hFactorOut * wFactorOut;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;
        Reg::RegTensor<U> v4;
        Reg::RegTensor<U> v5;
        Reg::RegTensor<U> v6;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd7;
        Reg::RegTensor<U> vd8;
        Reg::RegTensor<U> vd9;
        Reg::RegTensor<U> vd10;
        Reg::RegTensor<U> vd11;
        Reg::RegTensor<U> vd12;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, (U)wFactorOut, p0);
        Reg::Duplicate(v2, (U)wIn, p0);
        Reg::Duplicate(v3, (U)hStride, p0);
        Reg::Duplicate(v4, (U)wStride, p0);
        Reg::Duplicate(v5, (U)batchElemtsIn, p0);
        Reg::Duplicate(v6, (U)batchElemtsOut, p0);

        Reg::Div(vd1, v0, v6, p0);  // i / (rows * cols)
        Reg::Mul(vd2, vd1, v5, p0); // i / (rows * cols) * batchElemtsIn
        Reg::Mul(vd3, vd1, v6, p0); // (i / wFactorOut * wIn * hStride)
        Reg::Sub(vd4, v0, vd3, p0); // i % (rows * cols)

        Reg::Div(vd5, vd4, v1, p0);    // hwoffset / cols
        Reg::Mul(vd6, vd5, v2, p0);    // hwoffset / cols * wIn
        Reg::Mul(vd7, vd6, v3, p0);    // hwoffset / cols * wIn * hStride
        Reg::Mul(vd8, vd5, v1, p0);    // hwoffset / cols * cols
        Reg::Sub(vd9, vd4, vd8, p0);   // hwoffset % cols
        Reg::Mul(vd10, vd9, v4, p0);   // hwoffset % cols * wStride
        Reg::Add(vd11, vd7, vd10, p0); // hwoffset / cols * wIn * hStride + hwoffset % cols * wStride
        Reg::Add(vd12, vd2, vd11, p0);
        Reg::StoreAlign(dstAddr, vd12, p0);
    }
}

template <typename U>
__aicore__ inline void GenGatherIndexMultiRow(uint32_t wFactorOut, uint32_t wIn, uint32_t hStride, uint32_t wStride,
                                              LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;
        Reg::RegTensor<U> v4;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd7;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, (U)wFactorOut, p0);
        Reg::Duplicate(v2, (U)wIn, p0);
        Reg::Duplicate(v3, (U)hStride, p0);
        Reg::Duplicate(v4, (U)wStride, p0);

        Reg::Div(vd1, v0, v1, p0);   // i / wFactorOut
        Reg::Mul(vd2, vd1, v2, p0);  // (i / wFactorOut * wIn)
        Reg::Mul(vd3, vd2, v3, p0);  // (i / wFactorOut * wIn * hStride)
        Reg::Mul(vd4, vd1, v1, p0);  // (i / wFactorOut * wFactorOut)
        Reg::Sub(vd5, v0, vd4, p0);  // i % wFactor
        Reg::Mul(vd6, vd5, v4, p0);  // i % wFactorOut * wStride
        Reg::Add(vd7, vd3, vd6, p0); // (i / wFactorOut * wIn * hStride + i % wFactorOut * wStride)
        Reg::StoreAlign(dstAddr, vd7, p0);
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
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;

        Reg::RegTensor<U> vd0;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, (U)wStride, p0);
        Reg::Mul(vd0, v0, v1, p0); // (i / wFactorOut * wIn)
        Reg::StoreAlign(dstAddr, vd0, p0);
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
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange((Reg::RegTensor<regType>&)v0, i * repeatNum);
            Reg::Duplicate(v1, (U)kW, p0);
            Reg::Duplicate(v2, (U)wIn, p0);

            Reg::Div(vd1, v0, v1, p0);
            Reg::Mul(vd2, vd1, v2, p0);
            Reg::Mul(vd3, vd1, v1, p0);
            Reg::Sub(vd4, v0, vd3, p0);
            Reg::Add(vd5, vd2, vd4, p0);
            Reg::StoreAlign(dstAddr + i * repeatNum, vd5, p0);
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
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;

        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        if constexpr (SingleRow) {
            Reg::StoreAlign(dstAddr, v0, p0);
        } else {
            Reg::Duplicate(v1, (U)wIn, p0);
            Reg::Duplicate(v2, (U)wInDst, p0);

            Reg::Div(vd1, v0, v1, p0);
            Reg::Mul(vd2, vd1, v2, p0);
            Reg::Mul(vd3, vd1, v1, p0);
            Reg::Sub(vd4, v0, vd3, p0);
            Reg::Add(vd5, vd2, vd4, p0);
            Reg::StoreAlign(dstAddr, vd5, p0);
        }
    }
}

template <typename U, bool SingleRow>
__aicore__ inline void NHWCGenScatterIndex(uint32_t wIn, uint32_t wInDstElms, uint32_t channels,
                                           LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd7;
        Reg::RegTensor<U> vd8;
        Reg::RegTensor<U> vd9;
        Reg::RegTensor<U> vd10;

        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        if constexpr (SingleRow) {
            Reg::StoreAlign(dstAddr, v0, p0);
        } else {
            Reg::Duplicate(v1, (U)wIn, p0);
            Reg::Duplicate(v2, (U)wInDstElms, p0);
            Reg::Duplicate(v3, (U)channels, p0);

            Reg::Div(vd1, v0, v3, p0);  // i / channels
            Reg::Div(vd2, vd1, v1, p0); // i / channels / win
            Reg::Mul(vd3, vd2, v2, p0); // i / channels / win * winDst

            Reg::Mul(vd4, vd2, v1, p0);  // i / channels / win * win
            Reg::Sub(vd5, vd1, vd4, p0); // i / channels mod win
            Reg::Mul(vd6, vd5, v3, p0);  // ( i / channels mod win) * channels
            Reg::Add(vd7, vd3, vd6, p0); // i / channels / win * winDst + i / channels mod win * channels

            Reg::Mul(vd8, vd1, v3, p0);
            Reg::Sub(vd9, v0, vd8, p0); // i mod channels

            Reg::Add(vd10, vd9, vd7,
                     p0); // (i / channels / win * winDst + i / channels mod win) * channels + i mod channels
            Reg::StoreAlign(dstAddr, vd10, p0);
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
        Reg::RegTensor<regType> tmp;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;

        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, (U)wStride, p0);
        Reg::Duplicate(v2, (U)channels, p0); // channels
        Reg::Div(vd0, v0, v2, p0);           // i / channels
        Reg::Mul(vd1, vd0, v2, p0);
        Reg::Sub(vd5, v0, vd1, p0);  // i % channel
        Reg::Mul(vd2, vd0, v1, p0);  // (i / channel * wstride)
        Reg::Mul(vd3, vd2, v2, p0);  // (i / channel * wstride * channels)
        Reg::Add(vd4, vd3, vd5, p0); // (i / channel * wstride * channels) + i % channel
        Reg::StoreAlign(dstAddr, vd4, p0);
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexMultiRow(uint32_t wFactorOut, uint32_t wInElms, uint32_t hStride,
                                                  uint32_t wStride, uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    // i / wFactorOut * wIn * hStride + i % wFactorOut * wStride
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;
        Reg::RegTensor<U> v4;
        Reg::RegTensor<U> v5;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd7;
        Reg::RegTensor<U> vd8;
        Reg::RegTensor<U> vd9;
        Reg::RegTensor<U> vd10;
        Reg::RegTensor<U> vd11;
        Reg::RegTensor<U> vd12;
        Reg::RegTensor<U> vd13;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, (U)wFactorOut, p0);
        Reg::Duplicate(v2, (U)wInElms, p0);
        Reg::Duplicate(v3, (U)hStride, p0);
        Reg::Duplicate(v4, (U)wStride, p0);
        Reg::Duplicate(v5, (U)channels, p0);

        Reg::Div(vd1, v0, v5, p0);  // i / channels
        Reg::Div(vd2, vd1, v1, p0); // i / channels / wFactorOut
        Reg::Mul(vd3, vd2, v2, p0); // (i  / channels / wFactorOut * wIn)
        Reg::Mul(vd4, vd3, v3, p0); // (i / channels / wFactorOut * wIn * hStride

        Reg::Mul(vd5, vd2, v1, p0);  // (i / channels / wFactorOut * wFactorOut)
        Reg::Sub(vd6, vd1, vd5, p0); // (i  / channels) % wFactor
        Reg::Mul(vd7, vd6, v4, p0);  // (i  / channels) % wFactorOut * wStride
        Reg::Mul(vd8, vd7, v5, p0);  // ( i  / channels) % wFactorOut * wStride) * channels

        Reg::Add(
            vd9, vd8, vd4,
            p0); // (i  / channels) / wFactorOut * wIn * hStride + (i  / channels) % wFactorOut * wStride* channels)
        Reg::Mul(vd11, vd1, v5, p0);  // i / channels * channels
        Reg::Sub(vd12, v0, vd11, p0); // i mod channel
        Reg::Add(vd13, vd9, vd12, p0);
        Reg::StoreAlign(dstAddr, vd13, p0);
    }
}

template <typename U>
__aicore__ inline void NHWCGenGatherIndexMultiBatch(uint32_t hFactorOut, uint32_t wFactorOut, uint32_t hIn,
                                                    uint32_t wInElms, uint32_t hStride, uint32_t wStride,
                                                    uint32_t channels, LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    U batchElemtsIn = hIn * wInElms;
    U batchElemtsOut = hFactorOut * wFactorOut * channels;
    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::type;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;
        Reg::RegTensor<U> v4;
        Reg::RegTensor<U> v5;
        Reg::RegTensor<U> v6;
        Reg::RegTensor<U> v7;

        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        Reg::RegTensor<U> vd8;
        Reg::RegTensor<U> vd12;
        Reg::RegTensor<U> vd14;
        Reg::RegTensor<U> vd17;
        Reg::RegTensor<U> vd18;
        Reg::MaskReg p0 = Reg::CreateMask<U, Reg::MaskPattern::ALL>();

        Reg::Arange((Reg::RegTensor<regType>&)v0, 0);
        Reg::Duplicate(v1, (U)wFactorOut, p0);
        Reg::Duplicate(v2, (U)wInElms, p0);
        Reg::Duplicate(v3, (U)hStride, p0);
        Reg::Duplicate(v4, (U)wStride, p0);
        Reg::Duplicate(v5, (U)channels, p0);
        Reg::Duplicate(v6, (U)batchElemtsIn, p0);
        Reg::Duplicate(v7, (U)batchElemtsOut, p0);

        Reg::Div(vd1, v0, v7, p0);  // i / (rows * cols * channels)
        Reg::Mul(vd2, vd1, v6, p0); // i / (rows * cols * channels) * batchElemtsIn       n

        Reg::Mul(vd4, vd1, v7, p0); // (i / (rows * cols * channels) * (rows * cols * channels)
        Reg::Sub(vd4, v0, vd4, p0); // i % (rows * cols *channels)

        Reg::Div(vd5, vd4, v5, p0); // hwoffset / channels
        Reg::Div(vd6, vd5, v1, p0); // hwoffset / channels / wfout
        Reg::Mul(vd8, vd6, v2, p0); // hwoffset / channels / wfout * win
        Reg::Mul(vd8, vd8, v3, p0); // hwoffset / channels / wfout * hstride  h

        Reg::Mul(vd12, vd6, v1, p0);   // hwoffset / channels / wfout * wfout
        Reg::Sub(vd12, vd5, vd12, p0); // hwoffset / channels % wfout
        Reg::Mul(vd12, vd12, v4, p0);  // hwoffset / channels % wfout * wstride
        Reg::Mul(vd12, vd12, v5, p0);  // (hwoffset / channels % wfout * wstride) * channels

        Reg::Add(vd14, vd12, vd8,
                 p0);                  // hwoffset / channels / wfout * hstride + hwoffset / channels % wfout * wstride
        Reg::Add(vd14, vd14, vd2, p0); // (hwoffset / channels / wfout * hstride + hwoffset / channels / wfout *
                                       // wstride) * channels + i / (rows * cols * channels) * batchElemtsIn

        Reg::Div(vd17, v0, v5, p0);   // i / channels
        Reg::Mul(vd17, vd17, v5, p0); // i / channels * channels
        Reg::Sub(vd17, v0, vd17, p0); // i % channels

        Reg::Add(vd18, vd14, vd17, p0);
        Reg::StoreAlign(dstAddr, vd18, p0);
    }
}

template <typename T, typename RegDstT>
__aicore__ inline void MergeMaxRes(RegDstT& res, const __ubuf__ T* dstLocalAddr, int32_t offset)
{
    // merge cur result with pre result
    Reg::MaskReg pregOne = Reg::CreateMask<T, Reg::MaskPattern::VL1>();
    RegDstT lastRes;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    LoadOneElement<T>(dstLocalAddr, lastRes, offset);
    Reg::Max(res, res, lastRes, pregOne); // nan index
    Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
}

template <typename T, typename RegDstT>
__aicore__ inline void ReduceMaxAll(RegDstT& res, RegDstT& src, Reg::MaskReg& maskAll)
{
    if constexpr (sizeof(T) == 1) {
        Reg::RegTensor<T> left;
        Reg::RegTensor<T> right;
        Reg::RegTensor<half> dst1;
        Reg::RegTensor<half> dst2;
        Reg::Interleave(left, right, src, src);
        Reg::Cast<half, T, castTraitB82B16>(dst1, left, maskAll);
        Reg::Cast<half, T, castTraitB82B16>(dst2, right, maskAll);
        Reg::Max(dst1, dst1, dst2, maskAll);
        Reg::Reduce<Reg::ReduceType::MAX>(dst1, dst1, maskAll);
        Reg::Cast<T, half, castTraitB162B8>(res, dst1, maskAll);
    } else if constexpr (IsSame<T, bfloat16_t>::value) {
        Reg::RegTensor<T> left;
        Reg::RegTensor<T> right;
        Reg::RegTensor<float> dst1;
        Reg::RegTensor<float> dst2;
        Reg::Interleave(left, right, src, src);
        Reg::Cast<float, T, castTraitB82B16>(dst1, left, maskAll);
        Reg::Cast<float, T, castTraitB82B16>(dst2, right, maskAll);
        Reg::Max(dst1, dst1, dst2, maskAll);
        Reg::Reduce<Reg::ReduceType::MAX>(dst1, dst1, maskAll);
        Reg::Cast<T, float, castTraitB162B8>(res, dst1, maskAll);
    } else {
        Reg::Reduce<Reg::ReduceType::MAX>(res, src, maskAll);
    }
}

template <bool MaskMergeMode, typename T, typename U, typename RegDstT>
__aicore__ inline void MaxWithGather(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index, Reg::MaskReg& mask)
{
    RegDstT vd1;
    if constexpr (sizeof(T) == 1) {
        using gatherType = typename GetGatherType<T>::type;
        Reg::RegTensor<gatherType> vd0;
        Reg::Gather(vd0, srcAddr, index, mask);
        Reg::Pack((Reg::RegTensor<uint8_t>&)vd1, vd0);
        Reg::MaskReg pMask;
        Reg::Pack<Reg::HighLowPart::LOWEST>(pMask, mask);
        if constexpr (MaskMergeMode) {
            RegDstT tmp;
            Reg::Max(tmp, vd1, res, pMask);
            Reg::Move<T, Reg::MaskMergeMode::MERGING>(res, tmp, pMask);
        } else {
            Reg::Max(res, vd1, res, pMask);
        }
    } else {
        Reg::Gather(vd1, srcAddr, index, mask);
        if constexpr (MaskMergeMode) {
            RegDstT tmp;
            Reg::Max(tmp, vd1, res, mask);
            Reg::Move<T, Reg::MaskMergeMode::MERGING>(res, tmp, mask);
        } else {
            Reg::Max(res, vd1, res, mask);
        }
    }
}

template <typename T, typename RegDstT>
__aicore__ inline void DuplicateNegInf(RegDstT& v0)
{
    if constexpr (IsSame<T, bfloat16_t>::value) {
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        Reg::Duplicate(v0, value);
    }
}

template <uint16_t REG_NUM, uint16_t IDX, typename U>
__aicore__ inline void LoadIndex(__ubuf__ U* indexAddr, Reg::RegTensor<U>& index)
{
    constexpr uint32_t repeatNum = Ops::Base::GetVRegSize() / sizeof(U);
    if constexpr (REG_NUM > IDX) {
        Reg::LoadAlign(index, indexAddr + IDX * repeatNum);
    }
}

template <uint16_t REG_NUM, uint16_t IDX, typename U, typename T, typename RegDstT>
__aicore__ inline void ComputeMaxWithGather(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                            Reg::MaskReg& mask)
{
    if constexpr (REG_NUM > IDX) {
        MaxWithGather<false>(res, srcAddr, index, mask);
    }
}

template <typename T, typename U, typename RegDstT>
__aicore__ inline void GatherCommon(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index, Reg::MaskReg& mask)
{
    if constexpr (sizeof(T) == 1) {
        using gatherType = typename GetGatherType<T>::type;
        Reg::RegTensor<gatherType> vd0;
        Reg::Gather(vd0, srcAddr, index, mask);
        Reg::Pack((Reg::RegTensor<uint8_t>&)res, vd0);
    } else {
        Reg::Gather(res, srcAddr, index, mask);
    }
}

template <typename T, typename U, uint16_t REG_NUM>
__aicore__ inline void MaxPoolSingleKernelCommon(__ubuf__ T* dstLocalAddr, __ubuf__ T* xLocalAddr,
                                                 __ubuf__ U* indexAddr, uint16_t loopN, uint16_t loopH, uint16_t loopW,
                                                 U oneChannelElements, U oneLoopStrideH, U oneLoopStrideW,
                                                 uint16_t tailLoopElements)
{
    if constexpr (sizeof(T) == sizeof(int64_t) && REG_NUM > INT64_MAXREGNUM) {
        return;
    }
    using RegDstT = typename std::conditional<sizeof(T) == B64, Reg::RegTensor<T, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<T>>::type;
    __VEC_SCOPE__
    {
        Reg::RegTensor<U> index[SIXTEEN];
        Reg::UnalignRegForStore u0;
        uint32_t tailNum = tailLoopElements;
        Reg::MaskReg maskAll = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);

        Reg::LoadAlign(index[0], indexAddr);
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
        __ubuf__ T* dstAddr = dstLocalAddr;
        for (uint16_t i = 0; i < loopN; i++) {
            __ubuf__ T* srcAddr = xLocalAddr + i * oneChannelElements;
            for (uint16_t j = 0; j < loopH; j++) {
                __ubuf__ T* srcAddrH = srcAddr + j * oneLoopStrideH;
                for (uint16_t k = 0; k < loopW; k++) {
                    __ubuf__ T* srcAddrW = srcAddrH + k * oneLoopStrideW;
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

                    Reg::StoreUnAlign(dstAddr, tmp, u0, 1);
                }
            }
        }
        Reg::StoreUnAlignPost(dstAddr, u0, 0);
    }
}

} // namespace MaxPoolV3
#endif // MAX_POOL_V3_COMMON_H_
