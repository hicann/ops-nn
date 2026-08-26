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
 * \file pool_3d_common.h
 * \brief
 */
#ifndef POOL_3D_COMMON_H_
#define POOL_3D_COMMON_H_

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace Pool3D {
using namespace AscendC;

constexpr int32_t DIM_D = 0;
constexpr int32_t DIM_H = 1;
constexpr int32_t DIM_W = 2;
constexpr int32_t NUM128 = 128;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;     // -inf 0xFC00
constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000; // -inf 0xFF800000
constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;    // -inf 0xFF80
constexpr int32_t MIN_INT32 = -2147483648;
constexpr int64_t MIN_INT64 = -9223372036854775807LL - 1;
constexpr uint8_t MIN_UINT8 = 0;
constexpr int16_t MIN_INT16 = -32768;
constexpr int8_t MIN_INT8 = -128;
constexpr uint16_t MIN_UINT16 = 0;
constexpr int32_t EIGHT = 8;
constexpr int32_t ZERO = 0;
constexpr int32_t ONE = 1;
constexpr int32_t TWO = 2;
constexpr int32_t THREE = 3;
constexpr int32_t FOUR = 4;
constexpr int32_t FIVE = 5;
constexpr int32_t INDEX_SIZE = 256;
constexpr int32_t B64 = 8;
constexpr int32_t B8 = 1;
constexpr int32_t B16 = 2;
constexpr int32_t B32 = 4;

constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;
constexpr int32_t DIM3 = 3;
constexpr int32_t DIM4 = 4;

constexpr int32_t OP_TYPE_MAX_POOL_3D = 0;
constexpr int32_t OP_TYPE_AVG_POOL_3D = 1;

constexpr int32_t SPLIT_COLS = 1;
constexpr int32_t SPLIT_ROWS = 2;
constexpr int32_t SPLIT_DEPTHS = 3;
constexpr int32_t SPLIT_BATCHS = 4;

constexpr int32_t COPY_SINGLE_ROW = 0;
constexpr int32_t SCATTER_SINGLE_ROW = 1;
constexpr int32_t SCATTER_MULTI_ROW = 2;
constexpr int32_t SCATTER_MULTI_DEPTH = 3;
constexpr int32_t SCATTER_MULTI_BATCH = 4;

constexpr int32_t GATHER_SINGLE_ROW = 0;
constexpr int32_t GATHER_MULTI_ROW = 1;
constexpr int32_t GATHER_MULTI_PLANE = 2;
constexpr int32_t GATHER_MULTI_BATCH = 3;

constexpr int32_t SPARSE_W = 1;
constexpr int32_t SPARSE_H = 2;
constexpr int32_t SPARSE_D = 4;
constexpr int32_t SPARSE_WH = 3;
constexpr int32_t SPARSE_WD = 5;
constexpr int32_t SPARSE_HD = 6;

struct TensorDescInfo {
    uint32_t size[5] = {1};
    uint32_t dstStride[5] = {1};
    int64_t srcStride[5] = {1};
};

struct Pool3dParam {
    uint16_t kSize[3] = {1};
    uint16_t stride[3] = {1};
    uint16_t dilation[3] = {1};
    float divisor = 1;
};

struct ShapeInfo {
    uint32_t n = 0;
    uint32_t depth = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t channel = 0;
};

struct CalcDivisorParam {
    int64_t kD = 0;
    int64_t kH = 0;
    int64_t kW = 0;
    int64_t sD = 0;
    int64_t sH = 0;
    int64_t sW = 0;
    int64_t frontPad = 0;
    int64_t backendPad = 0;
    int64_t topPad = 0;
    int64_t bottomPad = 0;
    int64_t leftPad = 0;
    int64_t rightPad = 0;
    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t dIn = 0;
    int64_t hIn = 0;
    int64_t wIn = 0;
};

struct ParamsForDim {
    int64_t in = 0;
    int64_t o = 0;
    int64_t k = 0;
    int64_t s = 0;
    int64_t d = 0;
    int64_t pl = 0;
    int64_t pr = 0;
};

template <typename T>
struct GetComputeType {
    using type = typename std::conditional<std::is_same<T, bool>::value, int8_t, T>::type;
};

template <typename T>
struct GetGatherType {
    using type = typename std::conditional<
        std::is_same<T, int8_t>::value, int16_t,
        typename std::conditional<std::is_same<T, uint8_t>::value, uint16_t, T>::type>::type;
};

template <typename T>
struct VciTypeGet {
    using type = typename std::conditional<
        std::is_same<T, uint32_t>::value, int32_t,
        typename std::conditional<
            std::is_same<T, uint16_t>::value, int16_t,
            typename std::conditional<std::is_same<T, uint64_t>::value, int64_t, T>::type>::type>::type;
};

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == B8 || sizeof(T) == B16, uint16_t, uint32_t>::type;
};

constexpr Reg::CastTrait castTraitB16ToB32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait castTraitB32ToB16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait CAST_INT32_TO_FP32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait CAST_INT64_TO_FP32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};

template <typename T, int32_t OP_TYPE>
__aicore__ inline constexpr uint32_t GetVFLen()
{
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        return platform::GetVRegSize() / sizeof(T);
    } else {
        return platform::GetVRegSize() / sizeof(float);
    }
}

__aicore__ inline void CalcKernelSizeCore(const ParamsForDim& paramsInfo, int64_t& curk, int64_t& curkWithPad,
                                          int64_t& curOrigin)
{
    curOrigin = paramsInfo.s * paramsInfo.o - paramsInfo.pl; // left
    int64_t leftInvaild = 0;
    if (curOrigin < 0) {
        leftInvaild = (-curOrigin + paramsInfo.d - 1) / paramsInfo.d; // 0 左侧有几个无效k
    }
    // min(in - origin - leftinvaild, k)
    curk = min((paramsInfo.in - curOrigin + paramsInfo.d - 1) / paramsInfo.d - leftInvaild, paramsInfo.k - leftInvaild);
    // min (in + pr - origin, k)
    curkWithPad = min(paramsInfo.in + paramsInfo.pr - curOrigin, paramsInfo.k);
    curOrigin += leftInvaild * paramsInfo.d; // 矫正到curOrigin +轴位置
}

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

template <typename T, int32_t OP_TYPE, typename RegDstT>
__aicore__ inline void MergeMaxRes(RegDstT& res, const __ubuf__ T* dstLocalAddr, int32_t offset)
{
    // merge cur result with pre result
    Reg::MaskReg pregOne = Reg::CreateMask<T, Reg::MaskPattern::VL1>();
    RegDstT lastRes;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    LoadOneElement<T>(dstLocalAddr, lastRes, offset);
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        Reg::Max(res, res, lastRes, pregOne);
    } else {
        Reg::Add(res, res, lastRes, pregOne);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void MergeMaxParaRes(Reg::RegTensor<T>& res, __ubuf__ T* dstLocalAddr, int32_t num)
{
    // merge cur result with pre result
    Reg::MaskReg pregAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::RegTensor<T> lastRes;
    AscendC::Reg::UnalignRegForLoad u0;
    auto curSrcAddr = dstLocalAddr;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadUnAlignPre(u0, curSrcAddr);
    AscendC::Reg::LoadUnAlign(lastRes, u0, curSrcAddr, num);
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        Reg::Max(res, res, lastRes, pregAll);
    } else {
        Reg::Add(res, res, lastRes, pregAll);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
}

template <typename T>
__aicore__ inline constexpr T GetNegInf()
{
    T negInf = 0;
    if constexpr (std::is_same<T, int32_t>::value) {
        negInf = MIN_INT32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
        negInf = MIN_INT64;
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        negInf = MIN_UINT8;
    } else if constexpr (std::is_same<T, int16_t>::value) {
        negInf = MIN_INT16;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        negInf = MIN_INT8;
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        negInf = MIN_UINT16;
    } else if constexpr (std::is_same<T, float>::value) {
        negInf = *reinterpret_cast<const float*>(&FLOAT32_NEG_INF);
    } else {
        negInf = *reinterpret_cast<const half*>(&FLOAT16_NEG_INF);
    }
    return negInf;
}

constexpr Reg::CastTrait castTraitT2Fp32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::UNKNOWN};

constexpr Reg::CastTrait castTraitFp322T = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::CAST_ROUND};

template <typename T, int32_t OP_TYPE, typename RegDstT>
__aicore__ inline void ReduceAll(RegDstT& res, RegDstT& src, Reg::MaskReg& maskAll)
{
    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        Reg::Reduce<Reg::ReduceType::SUM>(res, src, maskAll);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        Reg::RegTensor<T> left;
        Reg::RegTensor<T> right;
        Reg::RegTensor<float> dst1;
        Reg::RegTensor<float> dst2;
        Reg::Interleave(left, right, src, src);
        Reg::Cast<float, T, castTraitT2Fp32>(dst1, left, maskAll);
        Reg::Cast<float, T, castTraitT2Fp32>(dst2, right, maskAll);
        Reg::Max(dst1, dst1, dst2, maskAll);
        Reg::Reduce<Reg::ReduceType::MAX>(dst1, dst1, maskAll);
        Reg::Cast<T, float, castTraitFp322T>(res, dst1, maskAll);
    } else {
        Reg::Reduce<Reg::ReduceType::MAX>(res, src, maskAll);
    }
}

template <bool MaskMergeMode, int32_t OP_TYPE, typename T, typename U>
__aicore__ inline void MaxWithGather(Reg::RegTensor<T>& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                     Reg::MaskReg& mask)
{
    Reg::RegTensor<T> vd1;
    Reg::Gather(vd1, srcAddr, index, mask);
    if constexpr (MaskMergeMode) {
        Reg::RegTensor<T> tmp;
        if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
            Reg::Max(tmp, vd1, res, mask);
        } else {
            Reg::Add(tmp, vd1, res, mask);
        }
        Reg::Move<T, Reg::MaskMergeMode::MERGING>(res, tmp, mask);
    } else {
        if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
            Reg::Max(res, vd1, res, mask);
        } else {
            Reg::Add(res, vd1, res, mask);
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void DuplicateReg(Reg::RegTensor<T>& reg, Reg::MaskReg mask)
{
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Reg::Duplicate((Reg::RegTensor<uint16_t>&)reg, BFLOAT16_NEG_INF, mask);
        } else {
            T value = GetNegInf<T>();
            Reg::Duplicate(reg, value, mask);
        }
    } else {
        Reg::Duplicate(reg, 0, mask);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void DuplicateValue(const __ubuf__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    uint32_t num = calNum;
    Reg::RegTensor<T> v0;
    Reg::MaskReg p0 = Reg::UpdateMask<T>(num);
    Reg::UnalignRegForStore u0;
    DuplicateReg<T, OP_TYPE>(v0, p0);
    __ubuf__ T* addr = (__ubuf__ T*)dstAddr + offset;
    Reg::StoreUnAlign(addr, v0, u0, calNum);
    Reg::StoreUnAlignPost(addr, u0, 0);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void CustomDuplicate(__ubuf__ T* dstAddr, uint32_t calNum, uint16_t loop)
{
    uint32_t sreg = calNum;
    Reg::RegTensor<T> v0;
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Reg::Duplicate((Reg::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF);
        } else {
            T value = GetNegInf<T>();
            Reg::Duplicate(v0, value);
        }
    } else {
        Reg::Duplicate(v0, 0);
    }
    constexpr uint16_t repeatElm = platform::GetVRegSize() / sizeof(T);
    for (uint16_t i = 0; i < loop; i++) {
        Reg::MaskReg preg = Reg::UpdateMask<T>(sreg);
        Reg::AddrReg offset = Reg::CreateAddrReg<T>(i, repeatElm);
        Reg::StoreAlign(dstAddr, v0, offset, preg);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline constexpr T GetPadValue()
{
    if (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        return GetNegInf<T>();
    }
    return T(0);
}

template <typename T, typename U, typename RegDstT>
__aicore__ inline void MaxPool3DImpl(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index, uint16_t kD,
                                     uint16_t kH, uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub,
                                     Reg::MaskReg& pMask)
{
    using gatherType = typename GetGatherType<T>::type;
    Reg::RegTensor<gatherType> vd0;
    RegDstT vd1;
    Reg::RegTensor<U> v0;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<U> v2;

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        Reg::Duplicate(res, value);
    }
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        Reg::Adds(v0, index, dIdx * depthStrideInub, pMask);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            Reg::Adds(v1, v0, hIdx * rowStrideInub, pMask);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                Reg::Adds(v2, v1, wIdx * colStrideInub, pMask);
                Reg::Gather(vd1, srcAddr, v2, pMask);
                Reg::Max(res, vd1, res, pMask);
            }
        }
    }
}

template <typename T, typename U, typename RegDstT>
__aicore__ inline void MaxPool3DTraitTwoImpl(RegDstT& res0, RegDstT& res1, __ubuf__ T* srcAddr,
                                             Reg::RegTensor<U>& index0, Reg::RegTensor<U>& index1, uint16_t kD,
                                             uint16_t kH, uint16_t kW, U depthStrideInub, U rowStrideInub,
                                             U colStrideInub, Reg::MaskReg& pMask0, Reg::MaskReg& pMask1)
{
    using gatherType = typename GetGatherType<T>::type;
    Reg::RegTensor<gatherType> vd0One;
    RegDstT vd1One;
    Reg::RegTensor<U> v0One;
    Reg::RegTensor<U> v1One;
    Reg::RegTensor<U> v2One;
    Reg::RegTensor<gatherType> vd0Two;
    RegDstT vd1Two;
    Reg::RegTensor<U> v0Two;
    Reg::RegTensor<U> v1Two;
    Reg::RegTensor<U> v2Two;
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)res0, BFLOAT16_NEG_INF);
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)res1, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        Reg::Duplicate(res0, value);
        Reg::Duplicate(res1, value);
    }
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        Reg::Adds(v0One, index0, dIdx * depthStrideInub, pMask0);
        Reg::Adds(v0Two, index1, dIdx * depthStrideInub, pMask1);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            Reg::Adds(v1One, v0One, hIdx * rowStrideInub, pMask0);
            Reg::Adds(v1Two, v0Two, hIdx * rowStrideInub, pMask1);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                Reg::Adds(v2One, v1One, wIdx * colStrideInub, pMask0);
                Reg::Adds(v2Two, v1Two, wIdx * colStrideInub, pMask1);
                Reg::Gather(vd1One, srcAddr, v2One, pMask0);
                Reg::Gather(vd1Two, srcAddr, v2Two, pMask1);
                Reg::Max(res0, vd1One, res0, pMask0);
                Reg::Max(res1, vd1Two, res1, pMask1);
            }
        }
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorB32Impl(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                                   uint16_t kD, uint16_t kH, uint16_t kW, U depthStrideInub,
                                                   U rowStrideInub, U colStrideInub, float32_t divisor,
                                                   Reg::MaskReg& pMask)
{
    using gatherType = typename GetGatherType<T>::type;
    Reg::RegTensor<gatherType> vd0;
    RegDstT vd1;
    Reg::RegTensor<U> v0;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<U> v2;
    Reg::RegTensor<float32_t> divisorReg;
    Reg::Duplicate(res, (T)0);
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        Reg::Adds(v0, index, dIdx * depthStrideInub, pMask);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            Reg::Adds(v1, v0, hIdx * rowStrideInub, pMask);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                Reg::Adds(v2, v1, wIdx * colStrideInub, pMask);
                Reg::Gather(vd1, srcAddr, v2, pMask);
                Reg::Add(res, vd1, res, pMask);
            }
        }
    }
    if constexpr (!NO_DIV) {
        Reg::Duplicate(divisorReg, divisor);
        Reg::Div(res, res, divisorReg, pMask);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorB32TraitTwoImpl(RegDstT& res0, RegDstT& res1, __ubuf__ T* srcAddr,
                                                           Reg::RegTensor<U>& index0, Reg::RegTensor<U>& index1,
                                                           uint16_t kD, uint16_t kH, uint16_t kW, U depthStrideInub,
                                                           U rowStrideInub, U colStrideInub, float32_t divisor,
                                                           Reg::MaskReg& pMask0, Reg::MaskReg& pMask1)
{
    using gatherType = typename GetGatherType<T>::type;
    Reg::RegTensor<gatherType> vd0One;
    RegDstT vd1One;
    Reg::RegTensor<U> v0One;
    Reg::RegTensor<U> v1One;
    Reg::RegTensor<U> v2One;
    Reg::RegTensor<gatherType> vd0Two;
    RegDstT vd1Two;
    Reg::RegTensor<U> v0Two;
    Reg::RegTensor<U> v1Two;
    Reg::RegTensor<U> v2Two;
    Reg::RegTensor<float32_t> divisorReg;
    Reg::Duplicate(res0, (T)0);
    Reg::Duplicate(res1, (T)0);
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        Reg::Adds(v0One, index0, dIdx * depthStrideInub, pMask0);
        Reg::Adds(v0Two, index1, dIdx * depthStrideInub, pMask1);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            Reg::Adds(v1One, v0One, hIdx * rowStrideInub, pMask0);
            Reg::Adds(v1Two, v0Two, hIdx * rowStrideInub, pMask1);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                Reg::Adds(v2One, v1One, wIdx * colStrideInub, pMask0);
                Reg::Adds(v2Two, v1Two, wIdx * colStrideInub, pMask1);
                Reg::Gather(vd1One, srcAddr, v2One, pMask0);
                Reg::Gather(vd1Two, srcAddr, v2Two, pMask1);
                Reg::Add(res0, vd1One, res0, pMask0);
                Reg::Add(res1, vd1Two, res1, pMask1);
            }
        }
    }
    if constexpr (!NO_DIV) {
        Reg::Duplicate(divisorReg, divisor);
        Reg::Div(res0, res0, divisorReg, pMask0);
        Reg::Div(res1, res1, divisorReg, pMask1);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorB16Impl(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                                   uint16_t kD, uint16_t kH, uint16_t kW, U depthStrideInub,
                                                   U rowStrideInub, U colStrideInub, float32_t divisor,
                                                   Reg::MaskReg& pMask)
{
    using gatherType = typename GetGatherType<T>::type;
    Reg::RegTensor<gatherType> vd0;
    Reg::RegTensor<T> vd1;
    Reg::RegTensor<T> zero;
    Reg::RegTensor<U> v0;
    Reg::RegTensor<U> v1;
    Reg::RegTensor<U> v2;
    Reg::RegTensor<float32_t> tmpRes1;
    Reg::RegTensor<float32_t> tmpRes2;
    Reg::RegTensor<float32_t> left;
    Reg::RegTensor<float32_t> right;
    Reg::RegTensor<float32_t> divisorReg;
    Reg::RegTensor<T> tmpLeft;
    Reg::RegTensor<T> tmpRight;
    Reg::Duplicate(tmpRes1, (float32_t)0);
    Reg::Duplicate(tmpRes2, (float32_t)0);
    Reg::MaskReg defaultMask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Duplicate((Reg::RegTensor<float16_t>&)zero, (float16_t)0);
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        Reg::Adds(v0, index, dIdx * depthStrideInub, pMask);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            Reg::Adds(v1, v0, hIdx * rowStrideInub, pMask);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                Reg::Adds(v2, v1, wIdx * colStrideInub, pMask);
                Reg::Gather(vd1, srcAddr, v2, pMask);
                Reg::Interleave(tmpLeft, tmpRight, vd1, zero);
                Reg::Cast<float32_t, T, castTraitB16ToB32>(left, tmpLeft, defaultMask);
                Reg::Cast<float32_t, T, castTraitB16ToB32>(right, tmpRight, defaultMask);
                Reg::Add(tmpRes1, tmpRes1, left, defaultMask);
                Reg::Add(tmpRes2, tmpRes2, right, defaultMask);
            }
        }
    }
    if constexpr (NO_DIV) {
        Reg::Move((Reg::RegTensor<float32_t>&)res.reg[0], tmpRes1);
        Reg::Move((Reg::RegTensor<float32_t>&)res.reg[1], tmpRes2);
    } else {
        Reg::Duplicate(divisorReg, divisor);
        Reg::Div(tmpRes1, tmpRes1, divisorReg, defaultMask);
        Reg::Div(tmpRes2, tmpRes2, divisorReg, defaultMask);
        Reg::Cast<T, float32_t, castTraitB32ToB16>(tmpLeft, tmpRes1, defaultMask);
        Reg::Cast<T, float32_t, castTraitB32ToB16>(tmpRight, tmpRes2, defaultMask);
        Reg::DeInterleave(res, zero, tmpLeft, tmpRight);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorB16TraitTwoImpl(RegDstT& res0, RegDstT& res1, __ubuf__ T* srcAddr,
                                                           Reg::RegTensor<U>& index0, Reg::RegTensor<U>& index1,
                                                           uint16_t kD, uint16_t kH, uint16_t kW, U depthStrideInub,
                                                           U rowStrideInub, U colStrideInub, float32_t divisor,
                                                           Reg::MaskReg& pMask0, Reg::MaskReg& pMask1)
{
    using gatherType = typename GetGatherType<T>::type;
    Reg::RegTensor<T> zero;
    Reg::RegTensor<float32_t> divisorReg;
    Reg::RegTensor<gatherType> vd0One;
    Reg::RegTensor<T> vd1One;
    Reg::RegTensor<U> v0One;
    Reg::RegTensor<U> v1One;
    Reg::RegTensor<U> v2One;
    Reg::RegTensor<float32_t> tmpRes1One;
    Reg::RegTensor<float32_t> tmpRes2One;
    Reg::RegTensor<float32_t> leftOne;
    Reg::RegTensor<float32_t> rightOne;
    Reg::RegTensor<T> tmpLeftOne;
    Reg::RegTensor<T> tmpRightOne;
    Reg::RegTensor<gatherType> vd0Two;
    Reg::RegTensor<T> vd1Two;
    Reg::RegTensor<U> v0Two;
    Reg::RegTensor<U> v1Two;
    Reg::RegTensor<U> v2Two;
    Reg::RegTensor<float32_t> tmpRes1Two;
    Reg::RegTensor<float32_t> tmpRes2Two;
    Reg::RegTensor<float32_t> leftTwo;
    Reg::RegTensor<float32_t> rightTwo;
    Reg::RegTensor<T> tmpLeftTwo;
    Reg::RegTensor<T> tmpRightTwo;
    Reg::Duplicate(tmpRes1One, (float32_t)0);
    Reg::Duplicate(tmpRes2One, (float32_t)0);
    Reg::Duplicate(tmpRes1Two, (float32_t)0);
    Reg::Duplicate(tmpRes2Two, (float32_t)0);
    Reg::MaskReg defaultMask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Duplicate((Reg::RegTensor<float16_t>&)zero, (float16_t)0);
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        Reg::Adds(v0One, index0, dIdx * depthStrideInub, pMask0);
        Reg::Adds(v0Two, index1, dIdx * depthStrideInub, pMask1);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            Reg::Adds(v1One, v0One, hIdx * rowStrideInub, pMask0);
            Reg::Adds(v1Two, v0Two, hIdx * rowStrideInub, pMask1);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                Reg::Adds(v2One, v1One, wIdx * colStrideInub, pMask0);
                Reg::Adds(v2Two, v1Two, wIdx * colStrideInub, pMask1);
                Reg::Gather(vd1One, srcAddr, v2One, pMask0);
                Reg::Gather(vd1Two, srcAddr, v2Two, pMask1);
                Reg::Interleave(tmpLeftOne, tmpRightOne, vd1One, zero);
                Reg::Interleave(tmpLeftTwo, tmpRightTwo, vd1Two, zero);
                Reg::Cast<float32_t, T, castTraitB16ToB32>(leftOne, tmpLeftOne, defaultMask);
                Reg::Cast<float32_t, T, castTraitB16ToB32>(leftTwo, tmpLeftTwo, defaultMask);
                Reg::Cast<float32_t, T, castTraitB16ToB32>(rightOne, tmpRightOne, defaultMask);
                Reg::Cast<float32_t, T, castTraitB16ToB32>(rightTwo, tmpRightTwo, defaultMask);
                Reg::Add(tmpRes1One, tmpRes1One, leftOne, defaultMask);
                Reg::Add(tmpRes1Two, tmpRes1Two, leftTwo, defaultMask);
                Reg::Add(tmpRes2One, tmpRes2One, rightOne, defaultMask);
                Reg::Add(tmpRes2Two, tmpRes2Two, rightTwo, defaultMask);
            }
        }
    }
    if constexpr (NO_DIV) {
        Reg::Move((Reg::RegTensor<float32_t>&)res0.reg[0], tmpRes1One);
        Reg::Move((Reg::RegTensor<float32_t>&)res0.reg[1], tmpRes2One);
        Reg::Move((Reg::RegTensor<float32_t>&)res1.reg[0], tmpRes1Two);
        Reg::Move((Reg::RegTensor<float32_t>&)res1.reg[1], tmpRes2Two);
    } else {
        Reg::Duplicate(divisorReg, divisor);
        Reg::Div(tmpRes1One, tmpRes1One, divisorReg, defaultMask);
        Reg::Div(tmpRes1Two, tmpRes1Two, divisorReg, defaultMask);
        Reg::Div(tmpRes2One, tmpRes2One, divisorReg, defaultMask);
        Reg::Div(tmpRes2Two, tmpRes2Two, divisorReg, defaultMask);
        Reg::Cast<T, float32_t, castTraitB32ToB16>(tmpLeftOne, tmpRes1One, defaultMask);
        Reg::Cast<T, float32_t, castTraitB32ToB16>(tmpLeftTwo, tmpRes1Two, defaultMask);
        Reg::Cast<T, float32_t, castTraitB32ToB16>(tmpRightOne, tmpRes2One, defaultMask);
        Reg::Cast<T, float32_t, castTraitB32ToB16>(tmpRightTwo, tmpRes2Two, defaultMask);
        Reg::DeInterleave(res0, zero, tmpLeftOne, tmpRightOne);
        Reg::DeInterleave(res1, zero, tmpLeftTwo, tmpRightTwo);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorImpl(RegDstT& res, __ubuf__ T* srcAddr, Reg::RegTensor<U>& index,
                                                uint16_t kD, uint16_t kH, uint16_t kW, U depthStrideInub,
                                                U rowStrideInub, U colStrideInub, float32_t divisor,
                                                Reg::MaskReg& pMask)
{
    if constexpr (sizeof(T) == TWO) {
        AvgPool3DWithDivisorB16Impl<T, U, NO_DIV>(res, srcAddr, index, kD, kH, kW, depthStrideInub, rowStrideInub,
                                                  colStrideInub, divisor, pMask);
    } else {
        AvgPool3DWithDivisorB32Impl<T, U, NO_DIV>(res, srcAddr, index, kD, kH, kW, depthStrideInub, rowStrideInub,
                                                  colStrideInub, divisor, pMask);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorTraitTwoImpl(RegDstT& res0, RegDstT& res1, __ubuf__ T* srcAddr,
                                                        Reg::RegTensor<U>& index0, Reg::RegTensor<U>& index1,
                                                        uint16_t kD, uint16_t kH, uint16_t kW, U depthStrideInub,
                                                        U rowStrideInub, U colStrideInub, float32_t divisor,
                                                        Reg::MaskReg& pMask0, Reg::MaskReg& pMask1)
{
    if constexpr (sizeof(T) == TWO) {
        AvgPool3DWithDivisorB16TraitTwoImpl<T, U, NO_DIV>(res0, res1, srcAddr, index0, index1, kD, kH, kW,
                                                          depthStrideInub, rowStrideInub, colStrideInub, divisor,
                                                          pMask0, pMask1);
    } else {
        AvgPool3DWithDivisorB32TraitTwoImpl<T, U, NO_DIV>(res0, res1, srcAddr, index0, index1, kD, kH, kW,
                                                          depthStrideInub, rowStrideInub, colStrideInub, divisor,
                                                          pMask0, pMask1);
    }
}

template <typename T, typename U, typename Z, int32_t OP_TYPE, bool NO_DIV = false>
__aicore__ inline void Pool3DWithOneLoopTraitTwo(__ubuf__ Z* dstAddr, __ubuf__ T* srcAddr, __ubuf__ U* indexAddr,
                                                 uint16_t kD, uint16_t kH, uint16_t kW, U depthStrideInub,
                                                 U rowStrideInub, U colStrideInub, U oneLoopOutElements,
                                                 U tailLoopOutElements, U oneLoopStride, uint16_t loopNum,
                                                 float32_t divisor = 1)
{
    constexpr U oneRegNum = platform::GetVRegSize() / sizeof(U);
    U oneLoopOutElements0 = oneLoopOutElements > oneRegNum ? oneRegNum : oneLoopOutElements;
    U oneLoopOutElements1 = oneLoopOutElements > oneRegNum ? oneLoopOutElements - oneRegNum : 0;
    U tailLoopOutElements0 = tailLoopOutElements > oneRegNum ? oneRegNum : tailLoopOutElements;
    U tailLoopOutElements1 = tailLoopOutElements > oneRegNum ? tailLoopOutElements - oneRegNum : 0;
    constexpr U oneRegNumFp32 = platform::GetVRegSize() / sizeof(float32_t);

    U halfLoopOut00 = oneLoopOutElements0 > oneRegNumFp32 ? oneRegNumFp32 : oneLoopOutElements0;
    U halfLoopOut01 = oneLoopOutElements0 > oneRegNumFp32 ? oneLoopOutElements0 - oneRegNumFp32 : 0;
    U halfLoopOut10 = oneLoopOutElements1 > oneRegNumFp32 ? oneRegNumFp32 : oneLoopOutElements1;
    U halfLoopOut11 = oneLoopOutElements1 > oneRegNumFp32 ? oneLoopOutElements1 - oneRegNumFp32 : 0;
    U tailHalfLoopOut00 = tailLoopOutElements0 > oneRegNumFp32 ? oneRegNumFp32 : tailLoopOutElements0;
    U tailHalfLoopOut01 = tailLoopOutElements0 > oneRegNumFp32 ? tailLoopOutElements0 - oneRegNumFp32 : 0;
    U tailHalfLoopOut10 = tailLoopOutElements1 > oneRegNumFp32 ? oneRegNumFp32 : tailLoopOutElements1;
    U tailHalfLoopOut11 = tailLoopOutElements1 > oneRegNumFp32 ? tailLoopOutElements1 - oneRegNumFp32 : 0;
    __VEC_SCOPE__
    {
        using RegDstT = typename std::conditional<sizeof(T) == B16 && std::is_same<Z, float32_t>::value,
                                                  Reg::RegTensor<Z, Reg::RegTraitNumTwo>, Reg::RegTensor<T>>::type;

        RegDstT res0;
        RegDstT res1;
        Reg::RegTensor<U> v1One;
        Reg::RegTensor<U> v2One;
        Reg::RegTensor<U> v3One;
        Reg::RegTensor<U> v4One;
        Reg::RegTensor<U> v1Two;
        Reg::RegTensor<U> v2Two;
        Reg::RegTensor<U> v3Two;
        Reg::RegTensor<U> v4Two;
        Reg::UnalignRegForStore u0;
        uint32_t num = oneLoopOutElements;
        uint32_t tailNum = tailLoopOutElements;
        Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
        Reg::MaskReg p1 = Reg::UpdateMask<U>(num);
        Reg::MaskReg pTail0 = Reg::UpdateMask<U>(tailNum);
        Reg::MaskReg pTail1 = Reg::UpdateMask<U>(tailNum);
        Reg::RegTensor<U> index0;
        Reg::RegTensor<U> index1;
        Reg::LoadAlign(index0, indexAddr);
        Reg::LoadAlign(index1, indexAddr + oneRegNum);
        __ubuf__ Z* tmpDstAddr = dstAddr;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Adds(v1One, index0, i * oneLoopStride, p0);
            Reg::Adds(v1Two, index1, i * oneLoopStride, p1);
            if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
                MaxPool3DTraitTwoImpl<T, U>(res0, res1, srcAddr, v1One, v1Two, kD, kH, kW, depthStrideInub,
                                            rowStrideInub, colStrideInub, p0, p1);
            } else {
                AvgPool3DWithDivisorTraitTwoImpl<T, U, NO_DIV>(res0, res1, srcAddr, v1One, v1Two, kD, kH, kW,
                                                               depthStrideInub, rowStrideInub, colStrideInub, divisor,
                                                               p0, p1);
            }
            if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
                Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res0.reg[0], u0, halfLoopOut00);
                Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res0.reg[1], u0, halfLoopOut01);
                Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res1.reg[0], u0, halfLoopOut10);
                Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res1.reg[1], u0, halfLoopOut11);
            } else {
                Reg::StoreUnAlign(tmpDstAddr, res0, u0, oneLoopOutElements0);
                Reg::StoreUnAlign(tmpDstAddr, res1, u0, oneLoopOutElements1);
            }
        }
        Reg::Adds(v1One, index0, loopNum * oneLoopStride, pTail0);
        Reg::Adds(v1Two, index1, loopNum * oneLoopStride, pTail1);
        if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
            MaxPool3DTraitTwoImpl<T, U>(res0, res1, srcAddr, v1One, v1Two, kD, kH, kW, depthStrideInub, rowStrideInub,
                                        colStrideInub, pTail0, pTail1);
        } else {
            AvgPool3DWithDivisorTraitTwoImpl<T, U, NO_DIV>(res0, res1, srcAddr, v1One, v1Two, kD, kH, kW,
                                                           depthStrideInub, rowStrideInub, colStrideInub, divisor,
                                                           pTail0, pTail1);
        }
        if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
            Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res0.reg[0], u0, tailHalfLoopOut00);
            Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res0.reg[1], u0, tailHalfLoopOut01);
            Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res1.reg[0], u0, tailHalfLoopOut10);
            Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res1.reg[1], u0, tailHalfLoopOut11);
        } else {
            Reg::StoreUnAlign(tmpDstAddr, res0, u0, tailLoopOutElements0);
            Reg::StoreUnAlign(tmpDstAddr, res1, u0, tailLoopOutElements1);
        }
        Reg::StoreUnAlignPost(tmpDstAddr, u0, 0);
    }
}

template <typename T, typename U, typename Z, int32_t OP_TYPE, bool NO_DIV = false, bool USE_TRAIT_TWO = false>
__aicore__ inline void Pool3DWithOneLoop(__ubuf__ Z* dstAddr, __ubuf__ T* srcAddr, __ubuf__ U* indexAddr, uint16_t kD,
                                         uint16_t kH, uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub,
                                         U oneLoopOutElements, U tailLoopOutElements, U oneLoopStride, uint16_t loopNum,
                                         float32_t divisor = 1)
{
    if constexpr (USE_TRAIT_TWO) {
        return Pool3DWithOneLoopTraitTwo<T, U, Z, OP_TYPE, NO_DIV>(
            dstAddr, srcAddr, indexAddr, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, oneLoopOutElements,
            tailLoopOutElements, oneLoopStride, loopNum, divisor);
    }
    constexpr U oneRegNumFp32 = platform::GetVRegSize() / sizeof(float32_t);
    U halfLoopOut0 = oneLoopOutElements > oneRegNumFp32 ? oneRegNumFp32 : oneLoopOutElements;
    U halfLoopOut1 = oneLoopOutElements > oneRegNumFp32 ? oneLoopOutElements - oneRegNumFp32 : 0;
    U tailHalfLoopOut0 = tailLoopOutElements > oneRegNumFp32 ? oneRegNumFp32 : tailLoopOutElements;
    U tailHalfLoopOut1 = tailLoopOutElements > oneRegNumFp32 ? tailLoopOutElements - oneRegNumFp32 : 0;
    __VEC_SCOPE__
    {
        using RegDstT = typename std::conditional<sizeof(T) == B16 && std::is_same<Z, float32_t>::value,
                                                  Reg::RegTensor<Z, Reg::RegTraitNumTwo>, Reg::RegTensor<T>>::type;
        Reg::RegTensor<U> v1;
        Reg::RegTensor<U> v2;
        Reg::RegTensor<U> v3;
        Reg::RegTensor<U> v4;
        Reg::UnalignRegForStore u0;
        RegDstT res;
        uint32_t num = oneLoopOutElements;
        uint32_t tailNum = tailLoopOutElements;
        Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
        Reg::MaskReg pTail = Reg::UpdateMask<U>(tailNum);
        Reg::RegTensor<U> index;
        Reg::LoadAlign(index, indexAddr);
        __ubuf__ Z* tmpDstAddr = dstAddr;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Adds(v1, index, i * oneLoopStride, p0);
            if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
                MaxPool3DImpl<T, U>(res, srcAddr, v1, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, p0);
            } else {
                AvgPool3DWithDivisorImpl<T, U, NO_DIV>(res, srcAddr, v1, kD, kH, kW, depthStrideInub, rowStrideInub,
                                                       colStrideInub, divisor, p0);
            }
            if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
                Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res.reg[0], u0, halfLoopOut0);
                Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res.reg[1], u0, halfLoopOut1);
            } else {
                Reg::StoreUnAlign(tmpDstAddr, res, u0, oneLoopOutElements);
            }
        }
        Reg::Adds(v1, index, loopNum * oneLoopStride, pTail);
        if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
            MaxPool3DImpl<T, U>(res, srcAddr, v1, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, pTail);
        } else {
            AvgPool3DWithDivisorImpl<T, U, NO_DIV>(res, srcAddr, v1, kD, kH, kW, depthStrideInub, rowStrideInub,
                                                   colStrideInub, divisor, pTail);
        }
        if constexpr (sizeof(T) == B16 && std::is_same<Z, float32_t>::value) {
            Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res.reg[0], u0, tailHalfLoopOut0);
            Reg::StoreUnAlign(tmpDstAddr, (Reg::RegTensor<float32_t>&)res.reg[1], u0, tailHalfLoopOut1);
        } else {
            Reg::StoreUnAlign(tmpDstAddr, res, u0, tailLoopOutElements);
        }
        Reg::StoreUnAlignPost(tmpDstAddr, u0, 0);
    }
}

__aicore__ inline void FastDivImpl(Reg::RegTensor<uint32_t>& res, Reg::RegTensor<uint32_t>& src,
                                   Reg::RegTensor<uint32_t>& magic, int16_t shift, Reg::MaskReg& mask)
{
    Reg::RegTensor<uint32_t> tmp;
    Reg::Mull(tmp, res, src, magic, mask);
    Reg::Add(tmp, src, res, mask);
    Reg::ShiftRights(res, tmp, shift, mask);
}

template <typename T, bool INCLUDE_PAD, typename RegT>
__aicore__ inline void CalcWindowSize(Reg::RegTensor<float>& res, RegT& src, T kD, T sD, T negFrontPad, T dIn,
                                      T dInAndBackendPad, Reg::MaskReg& mask)
{
    RegT tmp1, tmp2;
    Reg::Muls(tmp1, src, sD, mask);           // (didx * sd)
    Reg::Adds(tmp2, tmp1, negFrontPad, mask); // (didx * sd - fPad)
    Reg::Adds(tmp1, tmp2, kD, mask);          // dstart + kD
    if constexpr (INCLUDE_PAD) {
        Reg::Mins(tmp1, tmp1, dInAndBackendPad, mask);
    } else {
        Reg::Maxs(tmp2, tmp2, (T)0, mask);
        Reg::Mins(tmp1, tmp1, dIn, mask);
    }
    Reg::Sub(tmp1, tmp1, tmp2, mask);
    if constexpr (std::is_same<T, int32_t>::value) {
        Reg::Cast<float, T, CAST_INT32_TO_FP32>(res, tmp1, mask);
    } else {
        Reg::Cast<float, T, CAST_INT64_TO_FP32>(res, tmp1, mask);
    }
}

template <bool countIncludePad, bool PAD_MULTI_BATCH>
__aicore__ inline void ComputeDivisorImplB32(__ubuf__ float* divAddr, const CalcDivisorParam& param, int32_t start,
                                             int32_t total)
{
    __ubuf__ float* dstAddr = divAddr;
    int32_t oneRegLength = platform::GetVRegSize() / sizeof(float32_t);
    int32_t oneBatchOut = param.outD * param.outH * param.outW;
    int32_t outPlane = param.outH * param.outW;
    int32_t totalNum = total;
    uint16_t loopNum = ops::CeilDiv(totalNum, oneRegLength);
    int32_t kD = param.kD;
    int32_t kH = param.kH;
    int32_t kW = param.kW;
    int32_t sD = param.sD;
    int32_t sH = param.sH;
    int32_t sW = param.sW;

    int32_t negFrontPad = -1 * param.frontPad;
    int32_t dInAndBackendPad = param.dIn + param.backendPad;
    int32_t negTopPad = -1 * param.topPad;
    int32_t hInAndBottomPad = param.hIn + param.bottomPad;
    int32_t negLeftPad = -1 * param.leftPad;
    int32_t wInAndRightPad = param.wIn + param.rightPad;
    int32_t dIn = param.dIn;
    int32_t hIn = param.hIn;
    int32_t wIn = param.wIn;
    uint32_t m0, m1, m2, m3;
    uint32_t shift0, shift1, shift2;

    GetUintDivMagicAndShift<uint32_t>(m0, shift0, outPlane);
    GetUintDivMagicAndShift<uint32_t>(m1, shift1, param.outW);
    GetUintDivMagicAndShift<uint32_t>(m2, shift2, oneBatchOut);
    int32_t outW = param.outW;
    int32_t outH = param.outH;
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> v0;
        Reg::RegTensor<int32_t> v1;
        Reg::RegTensor<int32_t> v2;
        Reg::RegTensor<int32_t> v3;
        Reg::RegTensor<int32_t> v4;
        Reg::RegTensor<uint32_t> magic0;
        Reg::RegTensor<uint32_t> magic1;
        Reg::RegTensor<uint32_t> magic2;
        Reg::RegTensor<int32_t> vd0;
        Reg::RegTensor<int32_t> vd1;
        Reg::RegTensor<int32_t> vd2;
        Reg::RegTensor<int32_t> vd3;
        Reg::RegTensor<int32_t> vd4;
        Reg::RegTensor<int32_t> vd5;
        Reg::RegTensor<int32_t> vd6;

        Reg::RegTensor<float32_t> res;
        Reg::RegTensor<float32_t> dWindow;
        Reg::RegTensor<float32_t> hWindow;
        Reg::RegTensor<float32_t> wWindow;
        Reg::MaskReg p0 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(v1, outPlane, p0);
        Reg::Duplicate(v2, outW, p0);
        Reg::Duplicate(v3, outH, p0);

        Reg::Duplicate(magic0, m0, p0);
        Reg::Duplicate(magic1, m1, p0);
        Reg::Duplicate(magic2, m2, p0);

        uint32_t sreg = totalNum;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange(v0, i * oneRegLength + start);
            Reg::AddrReg resOffset = Reg::CreateAddrReg<float32_t>(i, oneRegLength);
            Reg::MaskReg pWrite = Reg::UpdateMask<float32_t>(sreg);
            if constexpr (PAD_MULTI_BATCH) {
                Reg::Duplicate(v4, oneBatchOut, p0);
                FastDivImpl((Reg::RegTensor<uint32_t>&)vd1, (Reg::RegTensor<uint32_t>&)v0, magic2, shift2, p0);
                Reg::Mul(vd2, vd1, v4, p0);
                Reg::Sub(v0, v0, vd2, p0);
            }
            FastDivImpl((Reg::RegTensor<uint32_t>&)vd1, (Reg::RegTensor<uint32_t>&)v0, magic0, shift0,
                        p0);            // (i / outhw) -> didx
            Reg::Mul(vd2, vd1, v1, p0); //
            Reg::Sub(vd3, v0, vd2, p0); // (i % outhw)
            FastDivImpl((Reg::RegTensor<uint32_t>&)vd4, (Reg::RegTensor<uint32_t>&)vd3, magic1, shift1,
                        p0); // (i % outhw / outw) ->hidx

            Reg::Mul(vd6, vd4, v2, p0);  //
            Reg::Sub(vd5, vd3, vd6, p0); // i % outw  ->widx(vd5)
            CalcWindowSize<int32_t, countIncludePad>(dWindow, vd1, kD, sD, negFrontPad, dIn, dInAndBackendPad, p0);
            CalcWindowSize<int32_t, countIncludePad>(hWindow, vd4, kH, sH, negTopPad, hIn, hInAndBottomPad, p0);
            CalcWindowSize<int32_t, countIncludePad>(wWindow, vd5, kW, sW, negLeftPad, wIn, wInAndRightPad, p0);
            Reg::Mul(res, dWindow, hWindow, p0);
            Reg::Mul(res, res, wWindow, p0);
            Reg::StoreAlign(dstAddr, res, resOffset, pWrite);
        }
    }
}

template <bool countIncludePad, bool PAD_MULTI_BATCH>
__aicore__ inline void ComputeDivisorImplB64(__ubuf__ float* divAddr, const CalcDivisorParam& param, int32_t start,
                                             int32_t total)
{
    __ubuf__ float* dstAddr = divAddr;
    int64_t oneRegLength = platform::GetVRegSize() / sizeof(float32_t);
    int32_t oneBatchOut = param.outD * param.outH * param.outW;
    int32_t outPlane = param.outH * param.outW;
    int64_t totalNum = total;
    uint16_t loopNum = ops::CeilDiv(totalNum, oneRegLength);
    int64_t kD = param.kD;
    int64_t kH = param.kH;
    int64_t kW = param.kW;
    int64_t sD = param.sD;
    int64_t sH = param.sH;
    int64_t sW = param.sW;

    int64_t negFrontPad = -1 * param.frontPad;
    int64_t dInAndBackendPad = param.dIn + param.backendPad;
    int64_t negTopPad = -1 * param.topPad;
    int64_t hInAndBottomPad = param.hIn + param.bottomPad;
    int64_t negLeftPad = -1 * param.leftPad;
    int64_t wInAndRightPad = param.wIn + param.rightPad;
    int64_t dIn = param.dIn;
    int64_t hIn = param.hIn;
    int64_t wIn = param.wIn;
    int64_t outW = param.outW;
    int64_t outH = param.outH;
    __VEC_SCOPE__
    {
        using RegDstT = typename Reg::RegTensor<int64_t, Reg::RegTraitNumTwo>;
        RegDstT v0;
        RegDstT v1;
        RegDstT v2;
        RegDstT v3;
        RegDstT v4;

        RegDstT vd0;
        RegDstT vd1;
        RegDstT vd2;
        RegDstT vd3;

        Reg::RegTensor<float32_t> dWindow;
        Reg::RegTensor<float32_t> hWindow;
        Reg::RegTensor<float32_t> wWindow;
        Reg::RegTensor<float32_t> res;
        Reg::MaskReg p0 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(v1, outPlane, p0);
        Reg::Duplicate(v2, outW, p0);
        Reg::Duplicate(v3, outH, p0);

        uint32_t sreg = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange(v0, i * oneRegLength + start);
            Reg::AddrReg resOffset = Reg::CreateAddrReg<float32_t>(i, oneRegLength);
            Reg::MaskReg pWrite = Reg::UpdateMask<float32_t>(sreg);
            Reg::Div(vd1, v0, v1, p0); // i / outhw  -> didx(vd1)
            CalcWindowSize<int64_t, countIncludePad>(dWindow, vd1, kD, sD, negFrontPad, dIn, dInAndBackendPad, p0);
            Reg::StoreAlign(dstAddr, dWindow, resOffset, pWrite);
        }
        uint32_t sreg1 = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange(v0, i * oneRegLength + start);
            Reg::AddrReg resOffset = Reg::CreateAddrReg<float32_t>(i, oneRegLength);
            Reg::MaskReg pWrite = Reg::UpdateMask<float32_t>(sreg1);
            Reg::Div(vd1, v0, v1, p0);  // i / outhw
            Reg::Mul(vd2, vd1, v1, p0); // (i / outhw * outhw)
            Reg::Sub(vd3, v0, vd2, p0); // i % outhw
            Reg::Div(vd2, vd3, v2, p0); // i % outhw / outh
            CalcWindowSize<int64_t, countIncludePad>(hWindow, vd2, kH, sH, negTopPad, hIn, hInAndBottomPad, p0);
            Reg::LoadAlign(res, dstAddr, resOffset);
            Reg::Mul(res, res, hWindow, p0);
            Reg::StoreAlign(dstAddr, res, resOffset, pWrite);
        }
        uint32_t sreg2 = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Arange(v0, i * oneRegLength + start);
            Reg::AddrReg resOffset = Reg::CreateAddrReg<float32_t>(i, oneRegLength);
            Reg::MaskReg pWrite = Reg::UpdateMask<float32_t>(sreg2);
            Reg::Div(vd3, v0, v2, p0);  // i / outw
            Reg::Mul(vd3, vd3, v2, p0); // (i / outhw * outhw)
            Reg::Sub(vd3, v0, vd3, p0); // i % outhw

            CalcWindowSize<int64_t, countIncludePad>(wWindow, vd3, kW, sW, negLeftPad, wIn, wInAndRightPad, p0);
            Reg::LoadAlign(res, dstAddr, resOffset);
            Reg::Mul(res, res, wWindow, p0);
            Reg::StoreAlign(dstAddr, res, resOffset, pWrite);
        }
    }

    if (PAD_MULTI_BATCH && (oneBatchOut < total)) {
        uint32_t diff = (total - oneBatchOut);
        uint16_t loopNum = diff / oneBatchOut;
        auto startAddr = dstAddr;
        auto writeAddr = dstAddr + oneBatchOut;
        if (oneBatchOut < oneRegLength) {
            uint32_t repeatElm = oneBatchOut;
            __VEC_SCOPE__
            {
                auto curDstAddr = writeAddr;
                Reg::UnalignRegForStore u0;
                Reg::RegTensor<float32_t> v0;
                Reg::LoadAlign(v0, startAddr);
                for (uint16_t k = 0; k < loopNum; k++) {
                    Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                Reg::StoreUnAlignPost(curDstAddr, u0, 0);
            }
        } else {
            uint32_t repeatElm = oneRegLength;
            uint16_t loopInner = oneBatchOut / oneRegLength;
            uint16_t tailInner = oneBatchOut - loopInner * oneRegLength;
            if (tailInner == 0) {
                loopInner -= 1;
                tailInner = oneRegLength;
            }
            __VEC_SCOPE__
            {
                auto curDstAddr = writeAddr;
                Reg::UnalignRegForLoad u0;
                Reg::UnalignRegForStore u1;
                Reg::RegTensor<float32_t> v0;
                Reg::LoadAlign(v0, startAddr);
                for (uint16_t i = 0; i < loopNum; i++) {
                    auto curSrcAddr = startAddr;
                    Reg::LoadUnAlignPre(u0, curSrcAddr);
                    for (uint16_t k = 0; k < loopInner; k++) {
                        Reg::LoadUnAlign(v0, u0, curSrcAddr, repeatElm);
                        Reg::StoreUnAlign(curDstAddr, v0, u1, repeatElm);
                    }
                    Reg::LoadUnAlign(v0, u0, curSrcAddr, tailInner);
                    Reg::StoreUnAlign(curDstAddr, v0, u1, tailInner);
                }
                Reg::StoreUnAlignPost(curDstAddr, u1, 0);
            }
        }
    }
}

__aicore__ inline void ComputeDivisorCommon(int64_t computeMode, __ubuf__ float* dstAddr, const CalcDivisorParam& param,
                                            int64_t start, int64_t num)
{
    switch (computeMode) {
        case 0:
            ComputeDivisorImplB32<false, false>(dstAddr, param, start, num);
            break;
        case 1:
            ComputeDivisorImplB32<false, true>(dstAddr, param, start, num);
            break;
        case 2:
            ComputeDivisorImplB32<true, false>(dstAddr, param, start, num);
            break;
        case 3:
            ComputeDivisorImplB32<true, true>(dstAddr, param, start, num);
            break;
        case 4:
            ComputeDivisorImplB64<false, false>(dstAddr, param, start, num);
            break;
        case 5:
            ComputeDivisorImplB64<false, true>(dstAddr, param, start, num);
            break;
        case 6:
            ComputeDivisorImplB64<true, false>(dstAddr, param, start, num);
            break;
        case 7:
            ComputeDivisorImplB64<true, true>(dstAddr, param, start, num);
            break;
    }
}

template <typename T>
__aicore__ inline void AvgPoolDivNormChannelBroadCast(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr,
                                                      __ubuf__ float32_t* divAddr, uint32_t num, uint32_t channel = 1)
{
    uint32_t oneRegChannel = platform::GetVRegSize() / sizeof(float32_t) / channel;
    uint16_t oneRegNum = oneRegChannel * channel;

    uint16_t loopNum = num / oneRegChannel;
    uint16_t tailNum = (num - loopNum * oneRegChannel) * channel;
    if (tailNum == 0) {
        loopNum -= 1;
        tailNum = oneRegNum;
    }
    __VEC_SCOPE__
    {
        Reg::RegTensor<float32_t> src;
        Reg::RegTensor<float32_t> div;
        Reg::RegTensor<float32_t> tmp;
        Reg::RegTensor<T> res;
        Reg::RegTensor<uint32_t> index;
        Reg::UnalignRegForLoad u0;
        Reg::UnalignRegForStore u1;
        auto curDstAddr = dstAddr;
        auto curSrcAddr = srcAddr;
        uint32_t mainSreg = oneRegNum;
        uint32_t tailSreg = tailNum;
        Reg::MaskReg pMask = Reg::UpdateMask<float32_t>(mainSreg);
        Reg::MaskReg pMaskTail = Reg::UpdateMask<float32_t>(tailSreg);
        Reg::Arange((Reg::RegTensor<int32_t>&)index, 0);
        Reg::RegTensor<uint32_t> channelDiv;
        Reg::MaskReg p0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate(channelDiv, channel, p0);
        Reg::Div(index, index, channelDiv, p0);
        Reg::LoadUnAlignPre(u0, curSrcAddr);
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadUnAlign(src, u0, curSrcAddr, oneRegNum);
            Reg::Gather(div, divAddr + i * oneRegChannel, index, pMask);

            if constexpr (std::is_same<T, float32_t>::value) {
                Reg::Div(res, src, div, pMask);
                Reg::StoreUnAlign(curDstAddr, res, u1, oneRegNum);
            } else {
                Reg::Div(tmp, src, div, pMask);
                Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                Reg::StoreUnAlign(curDstAddr, res, u1, oneRegNum);
            }
        }

        Reg::LoadUnAlign(src, u0, curSrcAddr, tailNum);
        Reg::Gather(div, divAddr + loopNum * oneRegChannel, index, pMaskTail);
        if constexpr (std::is_same<T, float32_t>::value) {
            Reg::Div(res, src, div, pMask);
            Reg::StoreUnAlign(curDstAddr, res, u1, tailNum);
        } else {
            Reg::Div(tmp, src, div, pMask);
            Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
            Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
            Reg::StoreUnAlign(curDstAddr, res, u1, tailNum);
        }
        Reg::StoreUnAlignPost(curDstAddr, u0, 0);
    }
}

template <typename T, bool CHANNEL_BROADACAST = false>
__aicore__ inline void AvgPoolDivNorm(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr, __ubuf__ float32_t* divAddr,
                                      uint32_t num, uint32_t channel = 1)
{
    if constexpr (CHANNEL_BROADACAST) {
        return AvgPoolDivNormChannelBroadCast(dstAddr, srcAddr, divAddr, num, channel);
    }
    uint16_t oneRegNum = platform::GetVRegSize() / sizeof(float32_t);
    uint16_t loopNum = (num + oneRegNum - 1) / oneRegNum;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float32_t> src;
        Reg::RegTensor<float32_t> div;
        Reg::RegTensor<float32_t> tmp;
        Reg::RegTensor<T> res;
        Reg::UnalignRegForLoad u0;
        Reg::LoadUnAlignPre(u0, divAddr);
        uint32_t sreg = num;
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::AddrReg srcOffset = Reg::CreateAddrReg<float32_t>(i, oneRegNum);
            Reg::AddrReg dstOffset = Reg::CreateAddrReg<T>(i, oneRegNum);
            Reg::MaskReg pMask = Reg::UpdateMask<float32_t>(sreg);

            Reg::LoadAlign(src, srcAddr, srcOffset);
            Reg::LoadUnAlign(div, u0, divAddr, oneRegNum);

            if constexpr (std::is_same<T, float32_t>::value) {
                Reg::Div(res, src, div, pMask);
                Reg::StoreAlign(dstAddr, res, dstOffset, pMask);
            } else {
                Reg::Div(tmp, src, div, pMask);
                Reg::MaskReg newMask;
                Reg::Pack<Reg::HighLowPart::LOWEST>(newMask, pMask);
                Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                Reg::StoreAlign(dstAddr, res, dstOffset, newMask);
            }
        }
    }
}

template <typename T, bool CHANNEL_BROADACAST = false>
__aicore__ inline void AvgPoolDivBatchV1(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr, __ubuf__ float32_t* divAddr,
                                         uint32_t batchNum, uint32_t batchElement, uint32_t channel = 1)
{
    uint32_t oneRegChannel = platform::GetVRegSize() / sizeof(float32_t) / channel;
    uint16_t oneRegNum = oneRegChannel * channel;
    uint16_t loopNum = batchElement / oneRegChannel;
    uint16_t tailNum = (batchElement - loopNum * oneRegChannel) * channel;
    uint16_t loopBatch = batchNum;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float32_t> src;
        Reg::RegTensor<float32_t> div;
        Reg::RegTensor<float32_t> tmp;
        Reg::RegTensor<T> res;
        Reg::RegTensor<uint32_t> index;
        Reg::UnalignRegForLoad u0;
        Reg::UnalignRegForStore u1;
        auto curSrcAddr = srcAddr;
        auto curDstAddr = dstAddr;

        uint32_t mainSreg = oneRegNum;
        uint32_t tailSreg = tailNum;
        Reg::MaskReg pMask = Reg::UpdateMask<float32_t>(mainSreg);
        Reg::MaskReg pMaskTail = Reg::UpdateMask<float32_t>(tailSreg);
        Reg::LoadUnAlignPre(u0, curSrcAddr);
        if constexpr (CHANNEL_BROADACAST) {
            Reg::Arange((Reg::RegTensor<int32_t>&)index, 0);
            Reg::RegTensor<uint32_t> channelDiv;
            Reg::MaskReg p0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
            Reg::Duplicate(channelDiv, channel, p0);
            Reg::Div(index, index, channelDiv, p0);
        }
        for (uint16_t i = 0; i < loopBatch; i++) {
            uint32_t sreg = batchElement;
            for (uint16_t j = 0; j < loopNum; j++) {
                Reg::AddrReg divOffset = Reg::CreateAddrReg<float32_t>(j, oneRegNum);
                Reg::LoadUnAlign(src, u0, curSrcAddr, oneRegNum);
                if constexpr (CHANNEL_BROADACAST) {
                    Reg::Gather(div, divAddr + j * oneRegChannel, index, pMask);
                } else {
                    Reg::LoadAlign(div, divAddr, divOffset);
                }
                if constexpr (std::is_same<T, float32_t>::value) {
                    Reg::Div(res, src, div, pMask);
                    Reg::StoreUnAlign(curDstAddr, res, u1, oneRegNum);
                } else {
                    Reg::Div(tmp, src, div, pMask);
                    Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                    Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                    Reg::StoreUnAlign(curDstAddr, res, u1, oneRegNum);
                }
            }
            Reg::LoadUnAlign(src, u0, curSrcAddr, tailNum);
            if constexpr (CHANNEL_BROADACAST) {
                Reg::Gather(div, divAddr + loopNum * oneRegChannel, index, pMaskTail);
            } else {
                Reg::LoadAlign(div, divAddr + loopNum * oneRegNum);
            }
            if constexpr (std::is_same<T, float32_t>::value) {
                Reg::Div(res, src, div, pMask);
                Reg::StoreUnAlign(curDstAddr, res, u1, tailNum);
            } else {
                Reg::Div(tmp, src, div, pMask);
                Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                Reg::StoreUnAlign(curDstAddr, res, u1, tailNum);
            }
        }
        Reg::StoreUnAlignPost(curDstAddr, u1, 0);
    }
}

template <typename T, bool CHANNEL_BROADACAST = false>
__aicore__ inline void AvgPoolDivBatchV2(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr, __ubuf__ float32_t* divAddr,
                                         uint32_t batchNum, uint32_t batchElement, uint32_t channel = 1)
{
    constexpr uint16_t oneRegNum = platform::GetVRegSize() / sizeof(float32_t);
    uint16_t onceRepeatBatch = oneRegNum / (batchElement * channel);
    uint16_t loopNum = batchNum / onceRepeatBatch;
    uint16_t onceRepeatNum = onceRepeatBatch * batchElement * channel;
    uint16_t tailRepeatNum = (batchNum - loopNum * onceRepeatBatch) * batchElement * channel;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float32_t> src;
        Reg::RegTensor<float32_t> div;
        Reg::RegTensor<float32_t> tmp;
        Reg::RegTensor<T> res;
        Reg::RegTensor<uint32_t> index;
        Reg::UnalignRegForLoad u0;
        Reg::UnalignRegForStore u1;
        auto curSrcAddr = srcAddr;
        auto curDstAddr = dstAddr;
        uint32_t mainSreg = onceRepeatNum;
        Reg::MaskReg pMask = Reg::UpdateMask<float32_t>(mainSreg);
        if constexpr (CHANNEL_BROADACAST) {
            Reg::Arange((Reg::RegTensor<int32_t>&)index, 0);
            Reg::RegTensor<uint32_t> channelDiv;
            Reg::MaskReg p0 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
            Reg::Duplicate(channelDiv, channel, p0);
            Reg::Div(index, index, channelDiv, p0);
        }
        Reg::LoadUnAlignPre(u0, curSrcAddr);
        if constexpr (CHANNEL_BROADACAST) {
            Reg::Gather(div, divAddr, index, pMask);
        } else {
            Reg::LoadAlign(div, divAddr);
        }
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadUnAlign(src, u0, curSrcAddr, onceRepeatNum);
            if constexpr (std::is_same<T, float32_t>::value) {
                Reg::Div(res, src, div, pMask);
                Reg::StoreUnAlign(curDstAddr, res, u1, onceRepeatNum);
            } else {
                Reg::Div(tmp, src, div, pMask);
                Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
                Reg::StoreUnAlign(curDstAddr, res, u1, onceRepeatNum);
            }
        }
        Reg::LoadUnAlign(src, u0, curSrcAddr, tailRepeatNum);
        if constexpr (std::is_same<T, float32_t>::value) {
            Reg::Div(res, src, div, pMask);
            Reg::StoreUnAlign(curDstAddr, res, u1, tailRepeatNum);
        } else {
            Reg::Div(tmp, src, div, pMask);
            Reg::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
            Reg::Pack((Reg::RegTensor<uint16_t>&)res, (Reg::RegTensor<uint32_t>&)res);
            Reg::StoreUnAlign(curDstAddr, res, u1, tailRepeatNum);
        }
        Reg::StoreUnAlignPost(curDstAddr, u1, 0);
    }
}

template <typename T, bool CHANNEL_BROADACAST = false>
__aicore__ inline void AvgPoolDivMultiBatch(__ubuf__ T* dstAddr, __ubuf__ float32_t* srcAddr,
                                            __ubuf__ float32_t* divAddr, uint32_t batchNum, uint32_t batchElement,
                                            uint32_t channel = 1)
{
    uint32_t oneVL = platform::GetVRegSize() / sizeof(float32_t);
    if (batchElement > oneVL) {
        AvgPoolDivBatchV1<T, CHANNEL_BROADACAST>(dstAddr, srcAddr, divAddr, batchNum, batchElement, channel);
    } else {
        AvgPoolDivBatchV2<T, CHANNEL_BROADACAST>(dstAddr, srcAddr, divAddr, batchNum, batchElement, channel);
    }
}

template <typename M, typename U>
__aicore__ inline void MaxPoolSingleChannel(__ubuf__ M* dstLocalAddr, __ubuf__ M* srcLocalAddr, uint16_t kD,
                                            uint16_t kH, uint16_t kW, uint32_t depStrideInUb, uint32_t rowStrideInub,
                                            uint16_t alignChannels, uint16_t repeatElms)
{
    using RegDstT = typename std::conditional<sizeof(M) == B64, Reg::RegTensor<M, Reg::RegTraitNumTwo>,
                                              Reg::RegTensor<M>>::type;
    RegDstT res;
    RegDstT vd0;
    uint32_t num = repeatElms;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::UnalignRegForStore u0;
    __ubuf__ M* curSrcAddr = srcLocalAddr;

    if constexpr (std::is_same<M, bfloat16_t>::value) {
        Reg::Duplicate((Reg::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
    } else {
        M value = GetNegInf<M>();
        Reg::Duplicate(res, value);
    }
    if constexpr (sizeof(M) == B64) {
        for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
            for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
                for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                    auto srcAddr = curSrcAddr + dIdx * depStrideInUb + hIdx * rowStrideInub + wIdx * alignChannels;
                    Reg::LoadAlign(vd0, srcAddr);
                    Reg::Max(res, vd0, res, p0);
                }
            }
        }
    } else {
        for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
            for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
                for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                    auto aReg = Reg::CreateAddrReg<U>(dIdx, depStrideInUb, hIdx, rowStrideInub, wIdx, alignChannels);
                    Reg::LoadAlign(vd0, curSrcAddr, aReg);
                    Reg::Max(res, vd0, res, p0);
                }
            }
        }
    }
    Reg::StoreAlign(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelB32(__ubuf__ M* dstLocalAddr, __ubuf__ M* srcLocalAddr, uint16_t kD,
                                               uint16_t kH, uint16_t kW, uint32_t depStrideInUb, uint32_t rowStrideInub,
                                               uint16_t alignChannels, uint16_t repeatElms, float32_t divisor)
{
    Reg::RegTensor<M> res;
    Reg::RegTensor<M> vd0;
    Reg::RegTensor<M> divRegs;
    uint32_t num = repeatElms;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::UnalignRegForStore u0;
    __ubuf__ M* curSrcAddr = srcLocalAddr;

    Reg::Duplicate(res, (float32_t)0);

    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                auto aReg = Reg::CreateAddrReg<U>(dIdx, depStrideInUb, hIdx, rowStrideInub, wIdx, alignChannels);
                Reg::LoadAlign(vd0, curSrcAddr, aReg);
                Reg::Add(res, vd0, res, p0);
            }
        }
    }
    Reg::Duplicate(divRegs, divisor);
    Reg::Div(res, res, divRegs, p0);
    Reg::StoreAlign(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelB16(__ubuf__ M* dstLocalAddr, __ubuf__ M* srcLocalAddr, uint16_t kD,
                                               uint16_t kH, uint16_t kW, uint32_t depStrideInUb, uint32_t rowStrideInub,
                                               uint16_t alignChannels, uint16_t repeatElms, float32_t divisor)
{
    Reg::RegTensor<M> res;
    Reg::RegTensor<M> vd0;
    Reg::RegTensor<M> zero;
    Reg::RegTensor<float32_t> tmpRes1;
    Reg::RegTensor<float32_t> tmpRes2;
    Reg::RegTensor<float32_t> left;
    Reg::RegTensor<float32_t> right;
    Reg::RegTensor<float32_t> divisorReg;
    Reg::RegTensor<M> tmpLeft;
    Reg::RegTensor<M> tmpRight;

    Reg::Duplicate(tmpRes1, (float32_t)0);
    Reg::Duplicate(tmpRes2, (float32_t)0);
    Reg::Duplicate((Reg::RegTensor<float16_t>&)zero, (float16_t)0);
    uint32_t num = repeatElms;
    Reg::MaskReg p0 = Reg::UpdateMask<U>(num);
    Reg::UnalignRegForStore u0;
    __ubuf__ M* curSrcAddr = srcLocalAddr;
    Reg::MaskReg defaultMask = Reg::CreateMask<M, Reg::MaskPattern::ALL>();

    Reg::Duplicate(res, (float32_t)0);

    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                auto aReg = Reg::CreateAddrReg<U>(dIdx, depStrideInUb, hIdx, rowStrideInub, wIdx, alignChannels);
                Reg::LoadAlign(vd0, curSrcAddr, aReg);
                Reg::Interleave(tmpLeft, tmpRight, vd0, zero);
                Reg::Cast<float32_t, M, castTraitB16ToB32>(left, tmpLeft, defaultMask);
                Reg::Cast<float32_t, M, castTraitB16ToB32>(right, tmpRight, defaultMask);
                Reg::Add(tmpRes1, tmpRes1, left, defaultMask);
                Reg::Add(tmpRes2, tmpRes2, right, defaultMask);
            }
        }
    }
    Reg::Duplicate(divisorReg, divisor);
    Reg::Div(tmpRes1, tmpRes1, divisorReg, defaultMask);
    Reg::Div(tmpRes2, tmpRes2, divisorReg, defaultMask);
    Reg::Cast<M, float32_t, castTraitB32ToB16>(tmpLeft, tmpRes1, defaultMask);
    Reg::Cast<M, float32_t, castTraitB32ToB16>(tmpRight, tmpRes2, defaultMask);
    Reg::DeInterleave(res, zero, tmpLeft, tmpRight);
    Reg::StoreAlign(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelImpl(__ubuf__ M* dstLocalAddr, __ubuf__ M* srcLocalAddr, uint16_t kD,
                                                uint16_t kH, uint16_t kW, uint32_t depStrideInUb,
                                                uint32_t rowStrideInub, uint16_t alignChannels, uint16_t repeatElms,
                                                float32_t divisor)
{
    if constexpr (sizeof(M) == TWO) {
        AvgPoolSingleChannelB16<M, U>(dstLocalAddr, srcLocalAddr, kD, kH, kW, depStrideInUb, rowStrideInub,
                                      alignChannels, repeatElms, divisor);
    } else {
        AvgPoolSingleChannelB32<M, U>(dstLocalAddr, srcLocalAddr, kD, kH, kW, depStrideInUb, rowStrideInub,
                                      alignChannels, repeatElms, divisor);
    }
}

} // namespace Pool3D
#endif // POOL_3D_COMMON_H_
