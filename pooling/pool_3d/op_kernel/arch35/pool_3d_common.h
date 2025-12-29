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

namespace Pool3D
{
using namespace AscendC;

constexpr int32_t DIM_D = 0;
constexpr int32_t DIM_H = 1;
constexpr int32_t DIM_W = 2;
constexpr int32_t NUM128 = 128;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;      // -inf 0xFC00
constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;  // -inf 0xFF800000
constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;     // -inf 0xFF80
constexpr int32_t MIN_INT32 = -2147483648;
constexpr int64_t MIN_INT64 = -9223372036854775807LL-1;
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
    using type =
        typename std::conditional<std::is_same<T, int8_t>::value, int16_t,
                                  typename std::conditional<std::is_same<T, uint8_t>::value, uint16_t, T>::type>::type;
};

template <typename T>
struct VciTypeGet {
    using type = typename std::conditional<
        std::is_same<T, uint32_t>::value, int32_t,
        typename std::conditional<std::is_same<T, uint16_t>::value, int16_t,
                                  typename std::conditional<std::is_same<T, uint64_t>::value, int64_t, T>::type>::type>::type;
};

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == B8 || sizeof(T) == B16, uint16_t, uint32_t>::type;
};

constexpr MicroAPI::CastTrait castTraitB16ToB32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr AscendC::MicroAPI::CastTrait castTraitB32ToB16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait CAST_INT32_TO_FP32 = {
  AscendC::MicroAPI::RegLayout::UNKNOWN,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};

constexpr AscendC::MicroAPI::CastTrait CAST_INT64_TO_FP32 = {
  AscendC::MicroAPI::RegLayout::UNKNOWN,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};

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
    curOrigin = paramsInfo.s * paramsInfo.o - paramsInfo.pl;  // left
    int64_t leftInvaild = 0;
    if (curOrigin < 0) {
        leftInvaild = (-curOrigin + paramsInfo.d - 1) / paramsInfo.d;  // 0 左侧有几个无效k
    }
    // min(in - origin - leftinvaild, k)
    curk = min((paramsInfo.in - curOrigin + paramsInfo.d - 1) / paramsInfo.d - leftInvaild,
        paramsInfo.k - leftInvaild);
    // min (in + pr - origin, k)
    curkWithPad = min(paramsInfo.in + paramsInfo.pr - curOrigin, paramsInfo.k);
    curOrigin += leftInvaild * paramsInfo.d;  // 矫正到curOrigin +轴位置
}

template <typename T>
__aicore__ inline void StoreElement(const __local_mem__ void* output, MicroAPI::RegTensor<T>& src, uint32_t offset,
                                    uint32_t element)
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

template <typename T, int32_t OP_TYPE, typename RegDstT>
__aicore__ inline void MergeMaxRes(RegDstT& res, const __local_mem__ T* dstLocalAddr, int32_t offset)
{
    // merge cur result with pre result
    MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
    RegDstT lastRes;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    LoadOneElement<T>(dstLocalAddr, lastRes, offset);
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        MicroAPI::Max(res, res, lastRes, pregOne);
    } else {
        MicroAPI::Add(res, res, lastRes, pregOne);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void MergeMaxParaRes(MicroAPI::RegTensor<T>& res, __local_mem__ T* dstLocalAddr, int32_t num)
{
    // merge cur result with pre result
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<T> lastRes;
    AscendC::MicroAPI::UnalignReg u0;
    auto curSrcAddr = dstLocalAddr;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    AscendC::MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
    AscendC::MicroAPI::DataCopyUnAlign(lastRes, u0, curSrcAddr, num);
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        MicroAPI::Max(res, res, lastRes, pregAll);
    } else {
        MicroAPI::Add(res, res, lastRes, pregAll);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
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

constexpr MicroAPI::CastTrait castTraitT2Fp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr MicroAPI::CastTrait castTraitFp322T = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

template <typename T, int32_t OP_TYPE, typename RegDstT>
__aicore__ inline void ReduceAll(RegDstT& res,  RegDstT& src, MicroAPI::MaskReg& maskAll)
{
    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        MicroAPI::ReduceSum(res, src, maskAll);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<T> left;
        MicroAPI::RegTensor<T> right;
        MicroAPI::RegTensor<float> dst1;
        MicroAPI::RegTensor<float> dst2;
        MicroAPI::Interleave(left, right, src, src);
        MicroAPI::Cast<float, T, castTraitT2Fp32>(dst1, left, maskAll);
        MicroAPI::Cast<float, T, castTraitT2Fp32>(dst2, right, maskAll);
        MicroAPI::Max(dst1, dst1, dst2, maskAll);
        MicroAPI::ReduceMax(dst1, dst1, maskAll);
        MicroAPI::Cast<T, float, castTraitFp322T>(res, dst1, maskAll);
    } else {
        MicroAPI::ReduceMax(res, src, maskAll);
    }
}

template <bool MaskMergeMode, int32_t OP_TYPE, typename T, typename U>
__aicore__ inline void MaxWithGather(MicroAPI::RegTensor<T>& res, __local_mem__ T* srcAddr,
    MicroAPI::RegTensor<U>& index, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::DataCopyGather(vd1, srcAddr, index, mask);
    if constexpr (MaskMergeMode) {
        MicroAPI::RegTensor<T> tmp;
        if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
            MicroAPI::Max(tmp, vd1, res, mask);
        } else {
            MicroAPI::Add(tmp, vd1, res, mask);
        }
        MicroAPI::Copy<T, MicroAPI::MaskMergeMode::MERGING>(res, tmp, mask);
    } else {
        if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
            MicroAPI::Max(res, vd1, res, mask);
        } else {
            MicroAPI::Add(res, vd1, res, mask);
        }
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void DuplicateReg(MicroAPI::RegTensor<T>& reg, MicroAPI::MaskReg mask)
{
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)reg, BFLOAT16_NEG_INF, mask);
        } else {
            T value = GetNegInf<T>();
            MicroAPI::Duplicate(reg, value, mask);
        }
    } else {
        MicroAPI::Duplicate(reg, 0, mask);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void DuplicateValue(const __local_mem__ void* dstAddr, uint32_t calNum, uint32_t offset)
{
    uint32_t num = calNum;
    MicroAPI::RegTensor<T> v0;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<T>(num);
    MicroAPI::UnalignReg u0;
    DuplicateReg<T, OP_TYPE>(v0, p0);
    __local_mem__ T* addr = (__local_mem__ T*)dstAddr + offset;
    MicroAPI::DataCopyUnAlign(addr, v0, u0, calNum);
    MicroAPI::DataCopyUnAlignPost(addr, u0, 0);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void CustomDuplicate(__local_mem__ T* dstAddr, uint32_t calNum, uint16_t loop)
{
    uint32_t sreg = calNum;
    MicroAPI::RegTensor<T> v0;
    if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)v0, BFLOAT16_NEG_INF);
        } else {
            T value = GetNegInf<T>();
            MicroAPI::Duplicate(v0, value);
        }
    } else {
        MicroAPI::Duplicate(v0, 0);
    }
    constexpr uint16_t repeatElm = platform::GetVRegSize() / sizeof(T);
    for (uint16_t i = 0; i < loop; i++) {
        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<T>(i, repeatElm);
        MicroAPI::DataCopy(dstAddr, v0, offset, preg);
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
__aicore__ inline void MaxPool3DImpl(RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kD, uint16_t kH,
                                   uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub, MicroAPI::MaskReg& pMask)
{
    using gatherType = typename GetGatherType<T>::type;
    MicroAPI::RegTensor<gatherType> vd0;
    RegDstT vd1;
    MicroAPI::RegTensor<U> v0;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<U> v2;

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        MicroAPI::Duplicate(res, value);
    }
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        MicroAPI::Adds(v0, index, dIdx * depthStrideInub, pMask);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            MicroAPI::Adds(v1, v0, hIdx * rowStrideInub, pMask);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                MicroAPI::Adds(v2, v1, wIdx * colStrideInub, pMask);
                MicroAPI::DataCopyGather(vd1, srcAddr, v2, pMask);
                MicroAPI::Max(res, vd1, res, pMask);
            }
        }
    }
}

template <typename T, typename U, typename RegDstT>
__aicore__ inline void MaxPool3DTraitTwoImpl(RegDstT& res0, RegDstT& res1, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index0, MicroAPI::RegTensor<U>& index1, uint16_t kD, uint16_t kH,
                                   uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub, MicroAPI::MaskReg& pMask0, MicroAPI::MaskReg& pMask1)
{
    using gatherType = typename GetGatherType<T>::type;
    MicroAPI::RegTensor<gatherType> vd0One;
    RegDstT vd1One;
    MicroAPI::RegTensor<U> v0One;
    MicroAPI::RegTensor<U> v1One;
    MicroAPI::RegTensor<U> v2One;
    MicroAPI::RegTensor<gatherType> vd0Two;
    RegDstT vd1Two;
    MicroAPI::RegTensor<U> v0Two;
    MicroAPI::RegTensor<U> v1Two;
    MicroAPI::RegTensor<U> v2Two;
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res0, BFLOAT16_NEG_INF);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res1, BFLOAT16_NEG_INF);
    } else {
        T value = GetNegInf<T>();
        MicroAPI::Duplicate(res0, value);
        MicroAPI::Duplicate(res1, value);
    }
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        MicroAPI::Adds(v0One, index0, dIdx * depthStrideInub, pMask0);
        MicroAPI::Adds(v0Two, index1, dIdx * depthStrideInub, pMask1);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            MicroAPI::Adds(v1One, v0One, hIdx * rowStrideInub, pMask0);
            MicroAPI::Adds(v1Two, v0Two, hIdx * rowStrideInub, pMask1);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                MicroAPI::Adds(v2One, v1One, wIdx * colStrideInub, pMask0);
                MicroAPI::Adds(v2Two, v1Two, wIdx * colStrideInub, pMask1);
                MicroAPI::DataCopyGather(vd1One, srcAddr, v2One, pMask0);
                MicroAPI::DataCopyGather(vd1Two, srcAddr, v2Two, pMask1);
                MicroAPI::Max(res0, vd1One, res0, pMask0);
                MicroAPI::Max(res1, vd1Two, res1, pMask1);
            }
        }
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorB32Impl(RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kD, uint16_t kH,
                                   uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask)
{
    using gatherType = typename GetGatherType<T>::type;
    MicroAPI::RegTensor<gatherType> vd0;
    RegDstT vd1;
    MicroAPI::RegTensor<U> v0;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<U> v2;
    MicroAPI::RegTensor<float32_t> divisorReg;
    MicroAPI::Duplicate(res, (T)0);
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        MicroAPI::Adds(v0, index, dIdx * depthStrideInub, pMask);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            MicroAPI::Adds(v1, v0, hIdx * rowStrideInub, pMask);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                MicroAPI::Adds(v2, v1, wIdx * colStrideInub, pMask);
                MicroAPI::DataCopyGather(vd1, srcAddr, v2, pMask);
                MicroAPI::Add(res, vd1, res, pMask); 
            }
        }
    }
    if constexpr (!NO_DIV) {
        MicroAPI::Duplicate(divisorReg, divisor);
        MicroAPI::Div(res, res, divisorReg, pMask);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorB32TraitTwoImpl(RegDstT& res0, RegDstT& res1, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index0, MicroAPI::RegTensor<U>& index1, uint16_t kD, uint16_t kH,
                                   uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask0, MicroAPI::MaskReg& pMask1)
{
    using gatherType = typename GetGatherType<T>::type;
    MicroAPI::RegTensor<gatherType> vd0One;
    RegDstT vd1One;
    MicroAPI::RegTensor<U> v0One;
    MicroAPI::RegTensor<U> v1One;
    MicroAPI::RegTensor<U> v2One;
    MicroAPI::RegTensor<gatherType> vd0Two;
    RegDstT vd1Two;
    MicroAPI::RegTensor<U> v0Two;
    MicroAPI::RegTensor<U> v1Two;
    MicroAPI::RegTensor<U> v2Two;
    MicroAPI::RegTensor<float32_t> divisorReg;
    MicroAPI::Duplicate(res0, (T)0);
    MicroAPI::Duplicate(res1, (T)0);
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        MicroAPI::Adds(v0One, index0, dIdx * depthStrideInub, pMask0);
        MicroAPI::Adds(v0Two, index1, dIdx * depthStrideInub, pMask1);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            MicroAPI::Adds(v1One, v0One, hIdx * rowStrideInub, pMask0);
            MicroAPI::Adds(v1Two, v0Two, hIdx * rowStrideInub, pMask1);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                MicroAPI::Adds(v2One, v1One, wIdx * colStrideInub, pMask0);
                MicroAPI::Adds(v2Two, v1Two, wIdx * colStrideInub, pMask1);
                MicroAPI::DataCopyGather(vd1One, srcAddr, v2One, pMask0);
                MicroAPI::DataCopyGather(vd1Two, srcAddr, v2Two, pMask1);
                MicroAPI::Add(res0, vd1One, res0, pMask0); 
                MicroAPI::Add(res1, vd1Two, res1, pMask1);
            }
        }
    }
    if constexpr (!NO_DIV) {
        MicroAPI::Duplicate(divisorReg, divisor);
        MicroAPI::Div(res0, res0, divisorReg, pMask0);
        MicroAPI::Div(res1, res1, divisorReg, pMask1);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorB16Impl(RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kD, uint16_t kH,
                                   uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask)
{
    using gatherType = typename GetGatherType<T>::type;
    MicroAPI::RegTensor<gatherType> vd0;
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::RegTensor<T> zero;
    MicroAPI::RegTensor<U> v0;
    MicroAPI::RegTensor<U> v1;
    MicroAPI::RegTensor<U> v2;
    MicroAPI::RegTensor<float32_t> tmpRes1;
    MicroAPI::RegTensor<float32_t> tmpRes2;
    MicroAPI::RegTensor<float32_t> left;
    MicroAPI::RegTensor<float32_t> right;
    MicroAPI::RegTensor<float32_t> divisorReg;
    MicroAPI::RegTensor<T> tmpLeft;
    MicroAPI::RegTensor<T> tmpRight;
    MicroAPI::Duplicate(tmpRes1, (float32_t)0);
    MicroAPI::Duplicate(tmpRes2, (float32_t)0);
    MicroAPI::MaskReg defaultMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate((MicroAPI::RegTensor<float16_t>&)zero, (float16_t)0);
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        MicroAPI::Adds(v0, index, dIdx * depthStrideInub, pMask);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            MicroAPI::Adds(v1, v0, hIdx * rowStrideInub, pMask);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                MicroAPI::Adds(v2, v1, wIdx * colStrideInub, pMask);
                MicroAPI::DataCopyGather(vd1, srcAddr, v2, pMask);
                MicroAPI::Interleave(tmpLeft, tmpRight, vd1, zero);
                MicroAPI::Cast<float32_t, T, castTraitB16ToB32>(left, tmpLeft, defaultMask);
                MicroAPI::Cast<float32_t, T, castTraitB16ToB32>(right, tmpRight, defaultMask);
                MicroAPI::Add(tmpRes1, tmpRes1, left, defaultMask); 
                MicroAPI::Add(tmpRes2, tmpRes2, right, defaultMask); 
            }
        }
    }
    if constexpr (NO_DIV) {
        MicroAPI::Copy((MicroAPI::RegTensor<float32_t>&)res.reg[0], tmpRes1);
        MicroAPI::Copy((MicroAPI::RegTensor<float32_t>&)res.reg[1], tmpRes2);
    } else {
        MicroAPI::Duplicate(divisorReg, divisor);
        MicroAPI::Div(tmpRes1, tmpRes1, divisorReg, defaultMask);
        MicroAPI::Div(tmpRes2, tmpRes2, divisorReg, defaultMask);
        MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(tmpLeft, tmpRes1, defaultMask);
        MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(tmpRight, tmpRes2, defaultMask);
        MicroAPI::DeInterleave(res, zero, tmpLeft, tmpRight);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorB16TraitTwoImpl(RegDstT& res0, RegDstT& res1, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index0, MicroAPI::RegTensor<U>& index1, uint16_t kD, uint16_t kH,
                                   uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask0, MicroAPI::MaskReg& pMask1)
{
    using gatherType = typename GetGatherType<T>::type;
    MicroAPI::RegTensor<T> zero;
    MicroAPI::RegTensor<float32_t> divisorReg;
    MicroAPI::RegTensor<gatherType> vd0One;
    MicroAPI::RegTensor<T> vd1One;
    MicroAPI::RegTensor<U> v0One;
    MicroAPI::RegTensor<U> v1One;
    MicroAPI::RegTensor<U> v2One;
    MicroAPI::RegTensor<float32_t> tmpRes1One;
    MicroAPI::RegTensor<float32_t> tmpRes2One;
    MicroAPI::RegTensor<float32_t> leftOne;
    MicroAPI::RegTensor<float32_t> rightOne;
    MicroAPI::RegTensor<T> tmpLeftOne;
    MicroAPI::RegTensor<T> tmpRightOne;
    MicroAPI::RegTensor<gatherType> vd0Two;
    MicroAPI::RegTensor<T> vd1Two;
    MicroAPI::RegTensor<U> v0Two;
    MicroAPI::RegTensor<U> v1Two;
    MicroAPI::RegTensor<U> v2Two;
    MicroAPI::RegTensor<float32_t> tmpRes1Two;
    MicroAPI::RegTensor<float32_t> tmpRes2Two;
    MicroAPI::RegTensor<float32_t> leftTwo;
    MicroAPI::RegTensor<float32_t> rightTwo;
    MicroAPI::RegTensor<T> tmpLeftTwo;
    MicroAPI::RegTensor<T> tmpRightTwo;
    MicroAPI::Duplicate(tmpRes1One, (float32_t)0);
    MicroAPI::Duplicate(tmpRes2One, (float32_t)0);
    MicroAPI::Duplicate(tmpRes1Two, (float32_t)0);
    MicroAPI::Duplicate(tmpRes2Two, (float32_t)0);
    MicroAPI::MaskReg defaultMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate((MicroAPI::RegTensor<float16_t>&)zero, (float16_t)0);
    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        MicroAPI::Adds(v0One, index0, dIdx * depthStrideInub, pMask0);
        MicroAPI::Adds(v0Two, index1, dIdx * depthStrideInub, pMask1);
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            MicroAPI::Adds(v1One, v0One, hIdx * rowStrideInub, pMask0);
            MicroAPI::Adds(v1Two, v0Two, hIdx * rowStrideInub, pMask1);
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                MicroAPI::Adds(v2One, v1One, wIdx * colStrideInub, pMask0);
                MicroAPI::Adds(v2Two, v1Two, wIdx * colStrideInub, pMask1);
                MicroAPI::DataCopyGather(vd1One, srcAddr, v2One, pMask0);
                MicroAPI::DataCopyGather(vd1Two, srcAddr, v2Two, pMask1);
                MicroAPI::Interleave(tmpLeftOne, tmpRightOne, vd1One, zero);
                MicroAPI::Interleave(tmpLeftTwo, tmpRightTwo, vd1Two, zero);
                MicroAPI::Cast<float32_t, T, castTraitB16ToB32>(leftOne, tmpLeftOne, defaultMask);
                MicroAPI::Cast<float32_t, T, castTraitB16ToB32>(leftTwo, tmpLeftTwo, defaultMask);
                MicroAPI::Cast<float32_t, T, castTraitB16ToB32>(rightOne, tmpRightOne, defaultMask);
                MicroAPI::Cast<float32_t, T, castTraitB16ToB32>(rightTwo, tmpRightTwo, defaultMask);
                MicroAPI::Add(tmpRes1One, tmpRes1One, leftOne, defaultMask); 
                MicroAPI::Add(tmpRes1Two, tmpRes1Two, leftTwo, defaultMask); 
                MicroAPI::Add(tmpRes2One, tmpRes2One, rightOne, defaultMask); 
                MicroAPI::Add(tmpRes2Two, tmpRes2Two, rightTwo, defaultMask); 
            }
        }
    }
    if constexpr (NO_DIV) {
        MicroAPI::Copy((MicroAPI::RegTensor<float32_t>&)res0.reg[0], tmpRes1One);
        MicroAPI::Copy((MicroAPI::RegTensor<float32_t>&)res0.reg[1], tmpRes2One);
        MicroAPI::Copy((MicroAPI::RegTensor<float32_t>&)res1.reg[0], tmpRes1Two);
        MicroAPI::Copy((MicroAPI::RegTensor<float32_t>&)res1.reg[1], tmpRes2Two);
    } else {
        MicroAPI::Duplicate(divisorReg, divisor);
        MicroAPI::Div(tmpRes1One, tmpRes1One, divisorReg, defaultMask);
        MicroAPI::Div(tmpRes1Two, tmpRes1Two, divisorReg, defaultMask);
        MicroAPI::Div(tmpRes2One, tmpRes2One, divisorReg, defaultMask);
        MicroAPI::Div(tmpRes2Two, tmpRes2Two, divisorReg, defaultMask);
        MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(tmpLeftOne, tmpRes1One, defaultMask);
        MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(tmpLeftTwo, tmpRes1Two, defaultMask);
        MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(tmpRightOne, tmpRes2One, defaultMask);
        MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(tmpRightTwo, tmpRes2Two, defaultMask);
        MicroAPI::DeInterleave(res0, zero, tmpLeftOne, tmpRightOne);
        MicroAPI::DeInterleave(res1, zero, tmpLeftTwo, tmpRightTwo);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorImpl(RegDstT& res, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index, uint16_t kD, uint16_t kH,
                                   uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask)
{
    if constexpr (sizeof(T) == TWO) {
        AvgPool3DWithDivisorB16Impl<T, U, NO_DIV>(res, srcAddr, index, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, divisor, pMask);
    } else {
        AvgPool3DWithDivisorB32Impl<T, U, NO_DIV>(res, srcAddr, index, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, divisor, pMask);
    }
}

template <typename T, typename U, bool NO_DIV, typename RegDstT>
__aicore__ inline void AvgPool3DWithDivisorTraitTwoImpl(RegDstT& res0, RegDstT& res1, __local_mem__ T* srcAddr, MicroAPI::RegTensor<U>& index0, MicroAPI::RegTensor<U>& index1, uint16_t kD, uint16_t kH,
                                   uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub, float32_t divisor, MicroAPI::MaskReg& pMask0, MicroAPI::MaskReg& pMask1)
{
    if constexpr (sizeof(T) == TWO) {
        AvgPool3DWithDivisorB16TraitTwoImpl<T, U, NO_DIV>(res0, res1, srcAddr, index0, index1, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, divisor, pMask0, pMask1);
    } else {
        AvgPool3DWithDivisorB32TraitTwoImpl<T, U, NO_DIV>(res0, res1, srcAddr, index0, index1, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, divisor, pMask0, pMask1);
    }
}

template <typename T, typename U, typename Z, int32_t OP_TYPE, bool NO_DIV=false>
__aicore__ inline void Pool3DWithOneLoopTraitTwo(__local_mem__ Z* dstAddr, __local_mem__ T* srcAddr,
                                         __ubuf__ U* indexAddr, uint16_t kD, uint16_t kH,
                                         uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub,
                                         U oneLoopOutElements, U tailLoopOutElements, U oneLoopStride, uint16_t loopNum, float32_t divisor=1)                                
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
            using RegDstT = typename std::conditional<sizeof(T)==B16  && std::is_same<Z, float32_t>::value, MicroAPI::RegTensor<Z, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
                                              
            RegDstT res0;
            RegDstT res1;
            MicroAPI::RegTensor<U> v1One;
            MicroAPI::RegTensor<U> v2One;
            MicroAPI::RegTensor<U> v3One;
            MicroAPI::RegTensor<U> v4One;
            MicroAPI::RegTensor<U> v1Two;
            MicroAPI::RegTensor<U> v2Two;
            MicroAPI::RegTensor<U> v3Two;
            MicroAPI::RegTensor<U> v4Two;
            MicroAPI::UnalignReg u0;
            uint32_t num = oneLoopOutElements;
            uint32_t tailNum = tailLoopOutElements;
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
            MicroAPI::MaskReg p1 = MicroAPI::UpdateMask<U>(num);
            MicroAPI::MaskReg pTail0 = MicroAPI::UpdateMask<U>(tailNum);
            MicroAPI::MaskReg pTail1 = MicroAPI::UpdateMask<U>(tailNum);
            MicroAPI::RegTensor<U> index0;
            MicroAPI::RegTensor<U> index1;
            MicroAPI::DataCopy(index0, indexAddr);
            MicroAPI::DataCopy(index1, indexAddr + oneRegNum);
            __local_mem__ Z* tmpDstAddr = dstAddr;
            for (uint16_t i = 0; i < loopNum; i++) {
                MicroAPI::Adds(v1One, index0, i * oneLoopStride, p0);       
                MicroAPI::Adds(v1Two, index1, i * oneLoopStride, p1);     
                if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
                    MaxPool3DTraitTwoImpl<T, U>(res0, res1, srcAddr, v1One, v1Two, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, p0, p1);
                } else {
                    AvgPool3DWithDivisorTraitTwoImpl<T, U, NO_DIV>(res0, res1, srcAddr, v1One, v1Two, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, divisor, p0, p1);
                }
                if constexpr (sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res0.reg[0], u0, halfLoopOut00);
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res0.reg[1], u0, halfLoopOut01);
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res1.reg[0], u0, halfLoopOut10);
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res1.reg[1], u0, halfLoopOut11);
                } else {
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, res0, u0, oneLoopOutElements0);
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, res1, u0, oneLoopOutElements1);
                }
            }
            MicroAPI::Adds(v1One, index0, loopNum * oneLoopStride, pTail0);
            MicroAPI::Adds(v1Two, index1, loopNum * oneLoopStride, pTail1);
            if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
                MaxPool3DTraitTwoImpl<T, U>(res0, res1, srcAddr, v1One, v1Two, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, pTail0, pTail1);
            } else {
                AvgPool3DWithDivisorTraitTwoImpl<T, U, NO_DIV>(res0, res1, srcAddr, v1One, v1Two, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, divisor, pTail0, pTail1);
            }
            if constexpr (sizeof(T)==B16 && std::is_same<Z, float32_t>::value) {
                MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res0.reg[0], u0, tailHalfLoopOut00);
                MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res0.reg[1], u0, tailHalfLoopOut01);
                MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res1.reg[0], u0, tailHalfLoopOut10);
                MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res1.reg[1], u0, tailHalfLoopOut11);
            } else {
                MicroAPI::DataCopyUnAlign(tmpDstAddr, res0, u0, tailLoopOutElements0);
                MicroAPI::DataCopyUnAlign(tmpDstAddr, res1, u0, tailLoopOutElements1);
            }
            MicroAPI::DataCopyUnAlignPost(tmpDstAddr, u0, 0);
        }
}

template <typename T, typename U, typename Z, int32_t OP_TYPE, bool NO_DIV=false, bool USE_TRAIT_TWO=false>
__aicore__ inline void Pool3DWithOneLoop(__local_mem__ Z* dstAddr, __local_mem__ T* srcAddr,
                                         __ubuf__ U* indexAddr, uint16_t kD, uint16_t kH,
                                         uint16_t kW, U depthStrideInub, U rowStrideInub, U colStrideInub,
                                         U oneLoopOutElements, U tailLoopOutElements, U oneLoopStride, uint16_t loopNum, float32_t divisor=1)                                
{
    if constexpr (USE_TRAIT_TWO) {
        return Pool3DWithOneLoopTraitTwo<T, U, Z, OP_TYPE, NO_DIV>(dstAddr, srcAddr, indexAddr, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub,
                                         oneLoopOutElements, tailLoopOutElements, oneLoopStride, loopNum, divisor);   
    }
    constexpr U oneRegNumFp32 = platform::GetVRegSize() / sizeof(float32_t);
    U halfLoopOut0 = oneLoopOutElements > oneRegNumFp32 ? oneRegNumFp32 : oneLoopOutElements;
    U halfLoopOut1 = oneLoopOutElements > oneRegNumFp32 ? oneLoopOutElements - oneRegNumFp32 : 0;
    U tailHalfLoopOut0 = tailLoopOutElements > oneRegNumFp32 ? oneRegNumFp32 : tailLoopOutElements;
    U tailHalfLoopOut1 = tailLoopOutElements > oneRegNumFp32 ? tailLoopOutElements - oneRegNumFp32 : 0;
    __VEC_SCOPE__
        {
            using RegDstT = typename std::conditional<sizeof(T)==B16 && std::is_same<Z, float32_t>::value, MicroAPI::RegTensor<Z, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<T>>::type;
            MicroAPI::RegTensor<U> v1;
            MicroAPI::RegTensor<U> v2;
            MicroAPI::RegTensor<U> v3;
            MicroAPI::RegTensor<U> v4;
            MicroAPI::UnalignReg u0;
            RegDstT res;
            uint32_t num = oneLoopOutElements;
            uint32_t tailNum = tailLoopOutElements;
            MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
            MicroAPI::MaskReg pTail = MicroAPI::UpdateMask<U>(tailNum);
            MicroAPI::RegTensor<U> index;
            MicroAPI::DataCopy(index, indexAddr);
            __local_mem__ Z* tmpDstAddr = dstAddr;
            for (uint16_t i = 0; i < loopNum; i++) {
                MicroAPI::Adds(v1, index, i * oneLoopStride, p0);           
                if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
                    MaxPool3DImpl<T, U>(res, srcAddr, v1, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, p0);
                } else {
                    AvgPool3DWithDivisorImpl<T, U, NO_DIV>(res, srcAddr, v1, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, divisor, p0);
                }
                if constexpr (sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[0], u0, halfLoopOut0);
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[1], u0, halfLoopOut1);
                } else {
                    MicroAPI::DataCopyUnAlign(tmpDstAddr, res, u0, oneLoopOutElements);
                }
            }
            MicroAPI::Adds(v1, index, loopNum * oneLoopStride, pTail);
            if constexpr (OP_TYPE == OP_TYPE_MAX_POOL_3D) {
                MaxPool3DImpl<T, U>(res, srcAddr, v1, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, pTail);
            } else {
                AvgPool3DWithDivisorImpl<T, U, NO_DIV>(res, srcAddr, v1, kD, kH, kW, depthStrideInub, rowStrideInub, colStrideInub, divisor, pTail);
            }
            if constexpr (sizeof(T)==B16  && std::is_same<Z, float32_t>::value) {
                MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[0], u0, tailHalfLoopOut0);
                MicroAPI::DataCopyUnAlign(tmpDstAddr, (MicroAPI::RegTensor<float32_t>&)res.reg[1], u0, tailHalfLoopOut1);
            } else {
                MicroAPI::DataCopyUnAlign(tmpDstAddr, res, u0, tailLoopOutElements);
            }
            MicroAPI::DataCopyUnAlignPost(tmpDstAddr, u0, 0);
        }
}

__aicore__ inline void FastDivImpl(MicroAPI::RegTensor<uint32_t>& res, MicroAPI::RegTensor<uint32_t>& src, MicroAPI::RegTensor<uint32_t>& magic, int16_t shift, MicroAPI::MaskReg &mask) 
{
    MicroAPI::RegTensor<uint32_t> tmp;
    MicroAPI::Mull(tmp, res, src, magic, mask);                                                      
    MicroAPI::Add(tmp, src, res, mask);                                                                
    MicroAPI::ShiftRights(res, tmp, shift, mask);   
}

template <typename T, bool INCLUDE_PAD, typename RegT>
__aicore__ inline void CalcWindowSize(MicroAPI::RegTensor<float>& res, RegT& src, T kD, T sD, T negFrontPad, T dIn, T dInAndBackendPad, MicroAPI::MaskReg &mask) 
{
    RegT tmp1, tmp2;
    MicroAPI::Muls(tmp1, src, sD, mask);   // (didx * sd)
    MicroAPI::Adds(tmp2, tmp1, negFrontPad, mask);   // (didx * sd - fPad)
    MicroAPI::Adds(tmp1, tmp2, kD, mask);   // dstart + kD 
    if constexpr (INCLUDE_PAD) {       
        MicroAPI::Mins(tmp1, tmp1, dInAndBackendPad, mask);     
    } else {
        MicroAPI::Maxs(tmp2, tmp2, (T)0, mask);     
        MicroAPI::Mins(tmp1, tmp1, dIn, mask);   
    }
    MicroAPI::Sub(tmp1, tmp1, tmp2, mask);
    if constexpr (std::is_same<T, int32_t>::value) {
        MicroAPI::Cast<float, T, CAST_INT32_TO_FP32>(res, tmp1, mask);
    } else {
        MicroAPI::Cast<float, T, CAST_INT64_TO_FP32>(res, tmp1, mask);
    }    
}

template <bool countIncludePad, bool PAD_MULTI_BATCH>
__aicore__ inline void ComputeDivisorImplB32(__local_mem__ float* divAddr, const CalcDivisorParam &param, int32_t start, int32_t total)   {
    __local_mem__ float*  dstAddr = divAddr;
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
        MicroAPI::RegTensor<int32_t> v0;
        MicroAPI::RegTensor<int32_t> v1;
        MicroAPI::RegTensor<int32_t> v2;
        MicroAPI::RegTensor<int32_t> v3;
        MicroAPI::RegTensor<int32_t> v4;
        MicroAPI::RegTensor<uint32_t> magic0;
        MicroAPI::RegTensor<uint32_t> magic1;
        MicroAPI::RegTensor<uint32_t> magic2;
        MicroAPI::RegTensor<int32_t> vd0;
        MicroAPI::RegTensor<int32_t> vd1;
        MicroAPI::RegTensor<int32_t> vd2;
        MicroAPI::RegTensor<int32_t> vd3;
        MicroAPI::RegTensor<int32_t> vd4;
        MicroAPI::RegTensor<int32_t> vd5;
        MicroAPI::RegTensor<int32_t> vd6;

        MicroAPI::RegTensor<float32_t> res;
        MicroAPI::RegTensor<float32_t> dWindow;
        MicroAPI::RegTensor<float32_t> hWindow;
        MicroAPI::RegTensor<float32_t> wWindow;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(v1, outPlane, p0);
        MicroAPI::Duplicate(v2, outW, p0);
        MicroAPI::Duplicate(v3, outH, p0);

        MicroAPI::Duplicate(magic0, m0, p0);
        MicroAPI::Duplicate(magic1, m1, p0);
        MicroAPI::Duplicate(magic2, m2, p0);
     
        uint32_t sreg = totalNum;
        for (uint16_t i = 0; i < loopNum; i++)
        {
            MicroAPI::Arange(v0, i * oneRegLength + start);
            MicroAPI::AddrReg resOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegLength);
            MicroAPI::MaskReg pWrite = MicroAPI::UpdateMask<float32_t>(sreg);
            if constexpr (PAD_MULTI_BATCH) {
                MicroAPI::Duplicate(v4, oneBatchOut, p0);
                FastDivImpl((MicroAPI::RegTensor<uint32_t> &)vd1, (MicroAPI::RegTensor<uint32_t> &)v0, magic2, shift2, p0);
                MicroAPI::Mul(vd2, vd1, v4, p0);  
                MicroAPI::Sub(v0, v0, vd2, p0);                 
            }
            FastDivImpl((MicroAPI::RegTensor<uint32_t> &)vd1, (MicroAPI::RegTensor<uint32_t> &)v0, magic0, shift0, p0); // (i / outhw) -> didx
            MicroAPI::Mul(vd2, vd1, v1, p0);   // 
            MicroAPI::Sub(vd3, v0, vd2, p0);   // (i % outhw)
            FastDivImpl((MicroAPI::RegTensor<uint32_t> &)vd4, (MicroAPI::RegTensor<uint32_t> &)vd3, magic1, shift1, p0); // (i % outhw / outw) ->hidx

            MicroAPI::Mul(vd6, vd4, v2, p0);   // 
            MicroAPI::Sub(vd5, vd3, vd6, p0);   // i % outw  ->widx(vd5)
            CalcWindowSize<int32_t, countIncludePad>(dWindow, vd1, kD, sD, negFrontPad, dIn, dInAndBackendPad, p0);
            CalcWindowSize<int32_t, countIncludePad>(hWindow, vd4, kH, sH, negTopPad, hIn, hInAndBottomPad, p0);
            CalcWindowSize<int32_t, countIncludePad>(wWindow, vd5, kW, sW, negLeftPad, wIn, wInAndRightPad, p0);
            MicroAPI::Mul(res, dWindow, hWindow, p0);
            MicroAPI::Mul(res, res, wWindow, p0);
            MicroAPI::DataCopy(dstAddr, res, resOffset, pWrite);             
        }
    }  
}


template <bool countIncludePad, bool PAD_MULTI_BATCH>
__aicore__ inline void ComputeDivisorImplB64(__local_mem__ float* divAddr, const CalcDivisorParam &param, int32_t start, int32_t total)   {

    __local_mem__ float*  dstAddr = divAddr;
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
        using RegDstT = typename MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo>;
        RegDstT v0;
        RegDstT v1;
        RegDstT v2;
        RegDstT v3;
        RegDstT v4;

        RegDstT vd0;
        RegDstT vd1;
        RegDstT vd2;
        RegDstT vd3;

        MicroAPI::RegTensor<float32_t> dWindow;
        MicroAPI::RegTensor<float32_t> hWindow;
        MicroAPI::RegTensor<float32_t> wWindow;
        MicroAPI::RegTensor<float32_t> res;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(v1, outPlane, p0);
        MicroAPI::Duplicate(v2, outW, p0);
        MicroAPI::Duplicate(v3, outH, p0);
    
        uint32_t sreg = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++)
        {
            MicroAPI::Arange(v0, i * oneRegLength + start);
            MicroAPI::AddrReg resOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegLength);
            MicroAPI::MaskReg pWrite = MicroAPI::UpdateMask<float32_t>(sreg);
            MicroAPI::Div(vd1, v0, v1, p0);    // i / outhw  -> didx(vd1)
            CalcWindowSize<int64_t, countIncludePad>(dWindow, vd1, kD, sD, negFrontPad, dIn, dInAndBackendPad, p0);
            MicroAPI::DataCopy(dstAddr, dWindow, resOffset, pWrite);             
        }
        uint32_t sreg1 = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++)
        {
            MicroAPI::Arange(v0, i * oneRegLength + start);
            MicroAPI::AddrReg resOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegLength);
            MicroAPI::MaskReg pWrite = MicroAPI::UpdateMask<float32_t>(sreg1);
            MicroAPI::Div(vd1, v0, v1, p0);    // i / outhw 
            MicroAPI::Mul(vd2, vd1, v1, p0);   // (i / outhw * outhw)
            MicroAPI::Sub(vd3, v0, vd2, p0);   // i % outhw  
            MicroAPI::Div(vd2, vd3, v2, p0);    // i % outhw / outh
            CalcWindowSize<int64_t, countIncludePad>(hWindow, vd2, kH, sH, negTopPad, hIn, hInAndBottomPad, p0);
            MicroAPI::DataCopy(res, dstAddr, resOffset);
            MicroAPI::Mul(res, res, hWindow, p0);
            MicroAPI::DataCopy(dstAddr, res, resOffset, pWrite);             
        }
        uint32_t sreg2 = oneBatchOut;
        for (uint16_t i = 0; i < loopNum; i++)
        {
            MicroAPI::Arange(v0, i * oneRegLength + start);
            MicroAPI::AddrReg resOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegLength);
            MicroAPI::MaskReg pWrite = MicroAPI::UpdateMask<float32_t>(sreg2);
            MicroAPI::Div(vd3, v0, v2, p0);    // i / outw 
            MicroAPI::Mul(vd3, vd3, v2, p0);   // (i / outhw * outhw)
            MicroAPI::Sub(vd3, v0, vd3, p0);   // i % outhw  
     
            CalcWindowSize<int64_t, countIncludePad>(wWindow, vd3, kW, sW, negLeftPad, wIn, wInAndRightPad, p0);
            MicroAPI::DataCopy(res, dstAddr, resOffset);
            MicroAPI::Mul(res, res, wWindow, p0);
            MicroAPI::DataCopy(dstAddr, res, resOffset, pWrite);             
        }
    } 

    if  (PAD_MULTI_BATCH && (oneBatchOut < total)) {
        uint32_t diff = (total - oneBatchOut);
        uint16_t loopNum = diff / oneBatchOut;
        auto startAddr = dstAddr;
        auto writeAddr = dstAddr + oneBatchOut;
        if (oneBatchOut < oneRegLength) {
            uint32_t repeatElm = oneBatchOut;
            __VEC_SCOPE__
            {
                auto curDstAddr = writeAddr;
                MicroAPI::UnalignReg u0;
                MicroAPI::RegTensor<float32_t> v0;
                MicroAPI::DataCopy(v0, startAddr);
                for (uint16_t k = 0; k < loopNum; k++) {
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);  
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
                MicroAPI::UnalignReg u0, u1;
                MicroAPI::RegTensor<float32_t> v0;
                MicroAPI::DataCopy(v0, startAddr);
                for (uint16_t i = 0; i < loopNum; i++) {
                    auto curSrcAddr = startAddr;
                    MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
                    for (uint16_t k = 0; k < loopInner; k++) {
                        MicroAPI::DataCopyUnAlign(v0, u0, curSrcAddr, repeatElm);
                        MicroAPI::DataCopyUnAlign(curDstAddr, v0, u1, repeatElm);
                    }
                    MicroAPI::DataCopyUnAlign(v0, u0, curSrcAddr, tailInner);
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u1, tailInner);       
                } 
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u1, 0);             
            }            
        }
    }
}

__aicore__ inline void ComputeDivisorCommon(int64_t computeMode, __local_mem__ float* dstAddr, const CalcDivisorParam &param, int64_t start, int64_t num)   {
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
__aicore__ inline void AvgPoolDivNormChannelBroadCast(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t num, uint32_t channel=1)
{
    uint32_t oneRegChannel = platform::GetVRegSize() / sizeof(float32_t) / channel;
    uint16_t oneRegNum = oneRegChannel * channel;

    uint16_t loopNum = num  / oneRegChannel;
    uint16_t tailNum = (num - loopNum * oneRegChannel) * channel;
    if (tailNum == 0) {
        loopNum -= 1;
        tailNum = oneRegNum;
    }
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<float32_t> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;
        MicroAPI::RegTensor<uint32_t> index;
        MicroAPI::UnalignReg u0, u1;
        auto curDstAddr = dstAddr;
        auto curSrcAddr = srcAddr;
        uint32_t mainSreg = oneRegNum;
        uint32_t tailSreg = tailNum;
        MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(mainSreg);
        MicroAPI::MaskReg pMaskTail = MicroAPI::UpdateMask<float32_t>(tailSreg);
        MicroAPI::Arange((MicroAPI::RegTensor<int32_t>&)index, 0);
        MicroAPI::RegTensor<uint32_t> channelDiv;
        MicroAPI::MaskReg p0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(channelDiv, channel, p0);
        MicroAPI::Div(index, index, channelDiv, p0); 
        MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
        for (uint16_t i = 0; i < loopNum; i++) {
            MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, oneRegNum); 
            MicroAPI::DataCopyGather(div, divAddr + i * oneRegChannel, index, pMask);  

            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, oneRegNum); 
            } else {
                MicroAPI::Div(tmp, src, div, pMask); 
                MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, oneRegNum); 
            }
        }

        MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, tailNum); 
        MicroAPI::DataCopyGather(div, divAddr + loopNum * oneRegChannel, index, pMaskTail); 
        if constexpr(std::is_same<T, float32_t>::value) {
            MicroAPI::Div(res, src, div, pMask); 
            MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailNum); 
        } else {
            MicroAPI::Div(tmp, src, div, pMask); 
            MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
            MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
            MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailNum); 
        }
        MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);   
    } 
}

template <typename T, bool CHANNEL_BROADACAST=false>
__aicore__ inline void AvgPoolDivNorm(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t num, uint32_t channel=1)
{
    if constexpr (CHANNEL_BROADACAST) {
       return AvgPoolDivNormChannelBroadCast(dstAddr, srcAddr, divAddr, num, channel);
    }
    uint16_t oneRegNum = platform::GetVRegSize() / sizeof(float32_t);
    uint16_t loopNum = (num + oneRegNum - 1) / oneRegNum;
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<float32_t> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;
        MicroAPI::UnalignReg u0;
        MicroAPI::DataCopyUnAlignPre(u0, divAddr);
        uint32_t sreg = num;
        for (uint16_t i = 0; i < loopNum; i++) {
            MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<float32_t>(i, oneRegNum);
            MicroAPI::AddrReg dstOffset = MicroAPI::CreateAddrReg<T>(i, oneRegNum);
            MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(sreg);

            MicroAPI::DataCopy(src, srcAddr, srcOffset);  
            MicroAPI::DataCopyUnAlign(div, u0, divAddr, oneRegNum); 

            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopy(dstAddr, res, dstOffset, pMask);
            } else {
                MicroAPI::Div(tmp, src, div, pMask); 
                MicroAPI::MaskReg newMask;
                MicroAPI::MaskPack<MicroAPI::HighLowPart::LOWEST>(newMask, pMask);
                MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                MicroAPI::DataCopy(dstAddr, res, dstOffset, newMask);
            }
        }
    } 
}

template <typename T, bool CHANNEL_BROADACAST=false>
__aicore__ inline void AvgPoolDivBatchV1(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t batchNum, uint32_t batchElement, uint32_t channel=1)
{
    uint32_t oneRegChannel = platform::GetVRegSize() / sizeof(float32_t) / channel;
    uint16_t oneRegNum = oneRegChannel * channel;
    uint16_t loopNum = batchElement / oneRegChannel;
    uint16_t tailNum = (batchElement - loopNum * oneRegChannel) * channel;
    uint16_t loopBatch = batchNum;
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<float32_t> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;
        MicroAPI::RegTensor<uint32_t> index;
        MicroAPI::UnalignReg u0, u1;
        auto curSrcAddr = srcAddr;
        auto curDstAddr = dstAddr;

        uint32_t mainSreg = oneRegNum;
        uint32_t tailSreg = tailNum;
        MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(mainSreg);
        MicroAPI::MaskReg pMaskTail = MicroAPI::UpdateMask<float32_t>(tailSreg);        
        MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
        if constexpr(CHANNEL_BROADACAST) {
            MicroAPI::Arange((MicroAPI::RegTensor<int32_t>&)index, 0);
            MicroAPI::RegTensor<uint32_t> channelDiv;
            MicroAPI::MaskReg p0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(channelDiv, channel, p0);
            MicroAPI::Div(index, index, channelDiv, p0); 
        }
        for (uint16_t i = 0; i < loopBatch; i++)
        {
            uint32_t sreg = batchElement;
            for (uint16_t j = 0; j < loopNum; j++) {
                MicroAPI::AddrReg divOffset = MicroAPI::CreateAddrReg<float32_t>(j, oneRegNum);
                MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, oneRegNum); 
                if constexpr(CHANNEL_BROADACAST) { 
                   MicroAPI::DataCopyGather(div, divAddr + j * oneRegChannel, index, pMask); 
                } else {
                    MicroAPI::DataCopy(div, divAddr, divOffset); 
                } 
                if constexpr(std::is_same<T, float32_t>::value) {
                    MicroAPI::Div(res, src, div, pMask); 
                    MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, oneRegNum);  
                } else {
                    MicroAPI::Div(tmp, src, div, pMask); 
                    MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                    MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                    MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, oneRegNum);  
                }
            }
            MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, tailNum); 
            if constexpr(CHANNEL_BROADACAST) { 
                MicroAPI::DataCopyGather(div, divAddr + loopNum * oneRegChannel, index, pMaskTail); 
            } else {
                MicroAPI::DataCopy(div, divAddr + loopNum * oneRegNum);  
            }        
            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailNum);  
            } else {
                MicroAPI::Div(tmp, src, div, pMask); 
                MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailNum);  
            }
        }
        MicroAPI::DataCopyUnAlignPost(curDstAddr, u1, 0);
    } 
}

template <typename T, bool CHANNEL_BROADACAST=false>
__aicore__ inline void AvgPoolDivBatchV2(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t batchNum, uint32_t batchElement, uint32_t channel=1)
{
    constexpr uint16_t oneRegNum = platform::GetVRegSize() / sizeof(float32_t);
    uint16_t onceRepeatBatch = oneRegNum / (batchElement * channel);
    uint16_t loopNum = batchNum / onceRepeatBatch;
    uint16_t onceRepeatNum = onceRepeatBatch * batchElement * channel;
    uint16_t tailRepeatNum = (batchNum - loopNum * onceRepeatBatch) * batchElement * channel;
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<float32_t> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;
        MicroAPI::RegTensor<uint32_t> index;
        MicroAPI::UnalignReg u0, u1;
        auto curSrcAddr = srcAddr;
        auto curDstAddr = dstAddr;
        uint32_t mainSreg = onceRepeatNum;
        MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(mainSreg);
        if constexpr(CHANNEL_BROADACAST) {
            MicroAPI::Arange((MicroAPI::RegTensor<int32_t>&)index, 0);
            MicroAPI::RegTensor<uint32_t> channelDiv;
            MicroAPI::MaskReg p0 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(channelDiv, channel, p0);
            MicroAPI::Div(index, index, channelDiv, p0); 
        }
        MicroAPI::DataCopyUnAlignPre(u0, curSrcAddr);
        if constexpr(CHANNEL_BROADACAST) { 
            MicroAPI::DataCopyGather(div, divAddr, index, pMask); 
        } else {
            MicroAPI::DataCopy(div, divAddr); 
        } 
        for (uint16_t i = 0; i < loopNum; i++) { 
            MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, onceRepeatNum); 
            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, onceRepeatNum);  
            } else {
                MicroAPI::Div(tmp, src, div, pMask); 
                MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
                MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, onceRepeatNum);  
            }
        }
        MicroAPI::DataCopyUnAlign(src, u0, curSrcAddr, tailRepeatNum); 
        if constexpr(std::is_same<T, float32_t>::value) {
            MicroAPI::Div(res, src, div, pMask); 
            MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailRepeatNum);  
        } else {
            MicroAPI::Div(tmp, src, div, pMask); 
            MicroAPI::Cast<T, float32_t, castTraitB32ToB16>(res, tmp, pMask);
            MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)res, (MicroAPI::RegTensor<uint32_t>&)res);
            MicroAPI::DataCopyUnAlign(curDstAddr, res, u1, tailRepeatNum);  
        }    
        MicroAPI::DataCopyUnAlignPost(curDstAddr, u1, 0);    
    } 
}

template <typename T, bool CHANNEL_BROADACAST=false>
__aicore__ inline void AvgPoolDivMultiBatch(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, __local_mem__ float32_t* divAddr, uint32_t batchNum, uint32_t batchElement, uint32_t channel=1) {
    uint32_t oneVL = platform::GetVRegSize() / sizeof(float32_t);
    if (batchElement > oneVL) {
        AvgPoolDivBatchV1<T, CHANNEL_BROADACAST>(dstAddr, srcAddr, divAddr, batchNum, batchElement, channel);
    } else {
        AvgPoolDivBatchV2<T, CHANNEL_BROADACAST>(dstAddr, srcAddr, divAddr, batchNum, batchElement, channel);
    }
}

template <typename M, typename U>
__aicore__ inline void MaxPoolSingleChannel(__local_mem__ M* dstLocalAddr, __local_mem__ M* srcLocalAddr, uint16_t kD,
                                            uint16_t kH, uint16_t kW, uint32_t depStrideInUb, uint32_t rowStrideInub,
                                            uint16_t alignChannels, uint16_t repeatElms)
{
    using RegDstT = typename std::conditional<sizeof(M) == B64, MicroAPI::RegTensor<M, MicroAPI::RegTraitNumTwo>,
                                              MicroAPI::RegTensor<M>>::type;
    RegDstT res;
    RegDstT vd0;
    uint32_t num = repeatElms;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::UnalignReg u0;
    __local_mem__ M* curSrcAddr = srcLocalAddr;

    if constexpr (std::is_same<M, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)res, BFLOAT16_NEG_INF);
    } else {
        M value = GetNegInf<M>();
        MicroAPI::Duplicate(res, value);
    }
    if constexpr (sizeof(M) == B64) {
        for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
            for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
                for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                    auto srcAddr = curSrcAddr + dIdx * depStrideInUb + hIdx * rowStrideInub + wIdx * alignChannels;
                    MicroAPI::DataCopy(vd0, srcAddr);
                    MicroAPI::Max(res, vd0, res, p0);
                }
            }
        }
    } else {
        for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
            for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
                for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                    auto aReg =
                        MicroAPI::CreateAddrReg<U>(dIdx, depStrideInUb, hIdx, rowStrideInub, wIdx, alignChannels);
                    MicroAPI::DataCopy(vd0, curSrcAddr, aReg);
                    MicroAPI::Max(res, vd0, res, p0);
                }
            }
        }
    }
    MicroAPI::DataCopy(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelB32(__local_mem__ M* dstLocalAddr, __local_mem__ M* srcLocalAddr, uint16_t kD,
                                            uint16_t kH, uint16_t kW, uint32_t depStrideInUb, uint32_t rowStrideInub,
                                            uint16_t alignChannels, uint16_t repeatElms, float32_t divisor)
{
    MicroAPI::RegTensor<M> res;
    MicroAPI::RegTensor<M> vd0;
    MicroAPI::RegTensor<M> divRegs;
    uint32_t num = repeatElms;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::UnalignReg u0;
    __local_mem__ M* curSrcAddr = srcLocalAddr;

    MicroAPI::Duplicate(res, (float32_t)0);

    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                auto aReg =
                    MicroAPI::CreateAddrReg<U>(dIdx, depStrideInUb, hIdx, rowStrideInub, wIdx, alignChannels);
                MicroAPI::DataCopy(vd0, curSrcAddr, aReg);
                MicroAPI::Add(res, vd0, res, p0);
            }
        }
    }
    MicroAPI::Duplicate(divRegs, divisor);
    MicroAPI::Div(res, res, divRegs, p0);
    MicroAPI::DataCopy(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelB16(__local_mem__ M* dstLocalAddr, __local_mem__ M* srcLocalAddr, uint16_t kD,
                                            uint16_t kH, uint16_t kW, uint32_t depStrideInUb, uint32_t rowStrideInub,
                                            uint16_t alignChannels, uint16_t repeatElms, float32_t divisor)
{
    MicroAPI::RegTensor<M> res;
    MicroAPI::RegTensor<M> vd0;
    MicroAPI::RegTensor<M> zero;
    MicroAPI::RegTensor<float32_t> tmpRes1;
    MicroAPI::RegTensor<float32_t> tmpRes2;
    MicroAPI::RegTensor<float32_t> left;
    MicroAPI::RegTensor<float32_t> right;
    MicroAPI::RegTensor<float32_t> divisorReg;
    MicroAPI::RegTensor<M> tmpLeft;
    MicroAPI::RegTensor<M> tmpRight;

    MicroAPI::Duplicate(tmpRes1, (float32_t)0);
    MicroAPI::Duplicate(tmpRes2, (float32_t)0);
    MicroAPI::Duplicate((MicroAPI::RegTensor<float16_t>&)zero, (float16_t)0);
    uint32_t num = repeatElms;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::UnalignReg u0;
    __local_mem__ M* curSrcAddr = srcLocalAddr;
    MicroAPI::MaskReg defaultMask = MicroAPI::CreateMask<M, MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate(res, (float32_t)0);

    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                auto aReg =
                    MicroAPI::CreateAddrReg<U>(dIdx, depStrideInUb, hIdx, rowStrideInub, wIdx, alignChannels);
                MicroAPI::DataCopy(vd0, curSrcAddr, aReg);
                MicroAPI::Interleave(tmpLeft, tmpRight, vd0, zero);
                MicroAPI::Cast<float32_t, M, castTraitB16ToB32>(left, tmpLeft, defaultMask);
                MicroAPI::Cast<float32_t, M, castTraitB16ToB32>(right, tmpRight, defaultMask);
                MicroAPI::Add(tmpRes1, tmpRes1, left, defaultMask); 
                MicroAPI::Add(tmpRes2, tmpRes2, right, defaultMask); 
            }
        }
    }
    MicroAPI::Duplicate(divisorReg, divisor);
    MicroAPI::Div(tmpRes1, tmpRes1, divisorReg, defaultMask);
    MicroAPI::Div(tmpRes2, tmpRes2, divisorReg, defaultMask);
    MicroAPI::Cast<M, float32_t, castTraitB32ToB16>(tmpLeft, tmpRes1, defaultMask);
    MicroAPI::Cast<M, float32_t, castTraitB32ToB16>(tmpRight, tmpRes2, defaultMask);
    MicroAPI::DeInterleave(res, zero, tmpLeft, tmpRight);
    MicroAPI::DataCopy(dstLocalAddr, res, p0);
}

template <typename M, typename U>
__aicore__ inline void AvgPoolSingleChannelImpl(__local_mem__ M* dstLocalAddr, __local_mem__ M* srcLocalAddr, uint16_t kD,
                                            uint16_t kH, uint16_t kW, uint32_t depStrideInUb, uint32_t rowStrideInub,
                                            uint16_t alignChannels, uint16_t repeatElms, float32_t divisor) {
    if constexpr (sizeof(M) == TWO) {
        AvgPoolSingleChannelB16<M, U>(dstLocalAddr, srcLocalAddr, kD, kH, kW, depStrideInUb, rowStrideInub, alignChannels,
                                                       repeatElms, divisor);
    } else {
        AvgPoolSingleChannelB32<M, U>(dstLocalAddr, srcLocalAddr, kD, kH, kW, depStrideInUb, rowStrideInub, alignChannels,
                                                       repeatElms, divisor);
    }
}

}  // namespace Pool3D
#endif  // POOL_3D_COMMON_H_
