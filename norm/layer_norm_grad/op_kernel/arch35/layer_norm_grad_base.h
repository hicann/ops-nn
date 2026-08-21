/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_base.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_BASE
#define LAYER_NORM_GRAD_BASE

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "layer_norm_grad_api.h"

/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize() { return 32U; }

namespace LayerNormGrad {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;
using namespace LayerNormGrad::Arith;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::Move;
using AscendC::Reg::Reduce;
using AscendC::Reg::StoreAlign;

constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_FIVE = 5;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t CONST_SEVEN = 7;
constexpr static int64_t CONST_EIGHT = 8;
constexpr static int64_t CONST_SIXTY_THREE = 63;
constexpr static uint32_t VL_FP32 = static_cast<int64_t>(GetVRegSize()) / sizeof(float);
class LayerNormGradBase {
public:
    __aicore__ inline LayerNormGradBase() : pipe_(nullptr){};

protected:
    __aicore__ inline static int64_t FindNearestPower2(const int64_t value);
    __aicore__ inline static int64_t GetCacheID(const int64_t idx);

public:
    template <typename T>
    __aicore__ inline static void CastToFp32From(const LocalTensor<float>& dstTensor, const LocalTensor<T>& srcTensor,
                                                 const int64_t count);
    template <typename T>
    __aicore__ inline static void CastToFp32From(const LocalTensor<float>& dstTensor, const LocalTensor<T>& srcTensor,
                                                 const int64_t rowSize, const int64_t colSize, const int64_t stride);
    template <typename T>
    __aicore__ inline static void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor,
                                         const int64_t rowSize, const int64_t colSize, const int64_t dstStride,
                                         const int64_t srcStride);
    template <typename T>
    __aicore__ inline static void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor,
                                         const int64_t rowSize);
    template <typename T>
    __aicore__ inline static void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                          const int64_t rowSize);
    template <typename T>
    __aicore__ inline static void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                          const int64_t rowSize, const int64_t colSize, const int64_t dstStride,
                                          const int64_t srcStride);
    __aicore__ inline static void CopyUB2UB(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                            const int64_t count);
    template <typename T>
    __aicore__ inline static void CopyUB2UBWithCast(const LocalTensor<T>& dstTensor,
                                                    const LocalTensor<float>& srcTensor, const int64_t count);
    __aicore__ inline static void VectorAdd(const LocalTensor<float>& dstTensor, const LocalTensor<float>& src0Tensor,
                                            const LocalTensor<float>& src1Tensor, const int64_t count);
    __aicore__ inline static void VectorAdd(const LocalTensor<float>& dstTensor, const LocalTensor<float>& src0Tensor,
                                            const LocalTensor<float>& src1Tensor, const int64_t mSize,
                                            const int64_t nSize, const int64_t stride);
    __aicore__ inline static void VectorMul(const LocalTensor<float>& dstTensor, const LocalTensor<float>& src0Tensor,
                                            const LocalTensor<float>& src1Tensor, const int64_t count);
    __aicore__ inline static void NlastBroadcastMul(const LocalTensor<float>& dstTensor,
                                                    const LocalTensor<float>& src0Tensor,
                                                    const LocalTensor<float>& src1Tensor, const int64_t bSize,
                                                    const int64_t aSize);
    __aicore__ inline static void LastReduceSumSmallR(const LocalTensor<float>& dstTensor,
                                                      const LocalTensor<float>& srcTensor, const int64_t aSize,
                                                      const int64_t rSize, const int64_t stride);
    __aicore__ inline static void LastReduceSum(const LocalTensor<float>& dstTensor,
                                                const LocalTensor<float>& srcTensor,
                                                const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize,
                                                const int64_t rSize, const int64_t stride);
    __aicore__ inline static void NlastReduceSum(const LocalTensor<float>& dstTensor,
                                                 const LocalTensor<float>& srcTensor,
                                                 const LocalTensor<float>& reduceSumTempTensor, const int64_t rSize,
                                                 const int64_t aSize, const int64_t stride);
    __aicore__ inline static void UpdateCache(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                              const int64_t cacheID, const int64_t stride, const int64_t count);
    __aicore__ inline static void Normalize(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                            const LocalTensor<float>& meanTensor, const LocalTensor<float>& varTensor,
                                            const int64_t rowSize, const int64_t colSize, const float epsilon);
    template <typename T>
    __aicore__ inline static void StoreTensorForDtypeT(__ubuf__ T* dst, AscendC::MicroAPI::RegTensor<float>& src,
                                                       AscendC::MicroAPI::MaskReg& preg, uint32_t offset);

protected:
    TPipe* pipe_;
}; // class LayerNormGradBase

// IMPL
__aicore__ inline int64_t LayerNormGradBase::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - ScalarCountLeadingZero(num);
        return (CONST_ONE << pow);
    }
}

__aicore__ inline int64_t LayerNormGradBase::GetCacheID(const int64_t idx)
{
    return ScalarGetCountOfValue<1>(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

template <typename T>
__aicore__ inline void LayerNormGradBase::CastToFp32From(const LocalTensor<float>& dstTensor,
                                                         const LocalTensor<T>& srcTensor, const int64_t count)
{
    // CastToFp32From T
    CastToFp32From<T>(dstTensor, srcTensor, CONST_ONE, count, CONST_ZERO);
}

template <typename T>
__aicore__ inline void LayerNormGradBase::CastToFp32From(const LocalTensor<float>& dstTensor,
                                                         const LocalTensor<T>& srcTensor, const int64_t rowSize,
                                                         const int64_t colSize, const int64_t stride)
{
    // CastToFp32From T
    uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
    uint16_t innerLoopTimes = static_cast<uint16_t>(
        CeilDiv(static_cast<int64_t>(colSize * sizeof(float)), static_cast<int64_t>(GetVRegSize())));
    uint32_t outerLoopSrcStride = static_cast<uint32_t>(stride * CONST_TWO);
    uint32_t outerLoopDstStride = static_cast<uint32_t>(stride);
    uint32_t innerLoopStride = VL_FP32;
    if constexpr (!(IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value)) {
        return;
    }
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ T* src = (__ubuf__ T*)srcTensor.GetPhyAddr();
            uint32_t count;
            AscendC::MicroAPI::RegTensor<float> fp32Reg;
            AscendC::MicroAPI::RegTensor<T> b16Reg;
            AscendC::MicroAPI::MaskReg pMask;
            count = static_cast<uint32_t>(colSize);
            pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                AscendC::MicroAPI::AddrReg srcAddrReg = AscendC::MicroAPI::CreateAddrReg<T>(i, outerLoopSrcStride);
                LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(b16Reg, (__ubuf__ T*)src, srcAddrReg);
                Cast<float, T, castTraitB162B32>(fp32Reg, b16Reg, pMask);
                AscendC::MicroAPI::AddrReg dstAddrReg = AscendC::MicroAPI::CreateAddrReg<float>(i, outerLoopDstStride);
                StoreAlign((__ubuf__ float*)dst, fp32Reg, dstAddrReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ T* src = (__ubuf__ T*)srcTensor.GetPhyAddr();
            uint32_t count;
            AscendC::MicroAPI::RegTensor<float> fp32Reg;
            AscendC::MicroAPI::RegTensor<T> b16Reg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                count = static_cast<uint32_t>(colSize);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    AscendC::MicroAPI::AddrReg srcAddrReg = AscendC::MicroAPI::CreateAddrReg<T>(i, outerLoopSrcStride,
                                                                                                j, innerLoopStride);
                    LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(b16Reg, (__ubuf__ T*)src, srcAddrReg);
                    Cast<float, T, castTraitB162B32>(fp32Reg, b16Reg, pMask);
                    AscendC::MicroAPI::AddrReg dstAddrReg = AscendC::MicroAPI::CreateAddrReg<float>(
                        i, outerLoopDstStride, j, innerLoopStride);
                    StoreAlign((__ubuf__ float*)dst, fp32Reg, dstAddrReg, pMask);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void LayerNormGradBase::CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor,
                                                 const int64_t rowSize, const int64_t colSize, const int64_t dstStride,
                                                 const int64_t srcStride)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = rowSize;
    params.blockLen = colSize * sizeof(T);
    params.srcStride = srcStride * sizeof(T) - params.blockLen;
    params.dstStride = (dstStride * sizeof(T) -
                        Aligned(static_cast<int64_t>(params.blockLen), static_cast<int64_t>(GetUbBlockSize()))) /
                       GetUbBlockSize();
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void LayerNormGradBase::CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor,
                                                 const int64_t rowSize)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = rowSize * sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void LayerNormGradBase::CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                                  const int64_t rowSize)
{
    // CopyOut
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = rowSize * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void LayerNormGradBase::CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                                  const int64_t rowSize, const int64_t colSize, const int64_t dstStride,
                                                  const int64_t srcStride)
{
    // CopyOut
    DataCopyExtParams params;
    params.blockCount = rowSize;
    params.blockLen = colSize * sizeof(T);
    params.dstStride = dstStride * sizeof(T) - params.blockLen;
    params.srcStride = (srcStride * sizeof(T) -
                        Aligned(static_cast<int64_t>(params.blockLen), static_cast<int64_t>(GetUbBlockSize()))) /
                       GetUbBlockSize();
    DataCopyPad(dstTensor, srcTensor, params);
}

__aicore__ inline void LayerNormGradBase::CopyUB2UB(const LocalTensor<float>& dstTensor,
                                                    const LocalTensor<float>& srcTensor, const int64_t count)
{
    // CopyUB2UB
    DataCopy(dstTensor, srcTensor,
             Aligned(static_cast<int64_t>(count), static_cast<int64_t>(GetUbBlockSize() / sizeof(float))));
}

template <typename T>
__aicore__ inline void LayerNormGradBase::CopyUB2UBWithCast(const LocalTensor<T>& dstTensor,
                                                            const LocalTensor<float>& srcTensor, const int64_t count)
{
    if constexpr (IsSameType<T, float>::value) {
        CopyUB2UB(dstTensor, srcTensor, count);
    } else {
        __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __ubuf__ T* dst = (__ubuf__ T*)dstTensor.GetPhyAddr();

        uint32_t cnt = count;
        uint16_t loopNum = CeilDiv(cnt, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> srcReg;
            RegTensor<T> xFp16;
            MaskReg pregMask;
            uint32_t sreg = cnt;
            for (uint16_t k = 0; k < loopNum; k++) {
                pregMask = UpdateMask<float>(sreg);
                uint32_t offset = k * VL_FP32;
                LoadAlign<float, LoadDist::DIST_NORM>(srcReg, (__ubuf__ float*)src + offset);

                Cast<T, float, castTraitB322B16>(xFp16, srcReg, pregMask);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(((__ubuf__ T*)dst) + offset, xFp16, pregMask);
            }
        }
    }
}

__aicore__ inline void LayerNormGradBase::VectorAdd(const LocalTensor<float>& dstTensor,
                                                    const LocalTensor<float>& src0Tensor,
                                                    const LocalTensor<float>& src1Tensor, const int64_t count)
{
    // VectorAdd
    if (count <= 0) {
        return;
    }
    uint16_t loopTimes = CeilDiv(static_cast<int64_t>(count * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    __VEC_SCOPE__
    {
        __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
        __ubuf__ float* src0 = (__ubuf__ float*)src0Tensor.GetPhyAddr();
        __ubuf__ float* src1 = (__ubuf__ float*)src1Tensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(count);
        AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < loopTimes; ++i) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            LoadAlign(aReg, (__ubuf__ float*)src0 + i * VL_FP32);
            LoadAlign(bReg, (__ubuf__ float*)src1 + i * VL_FP32);
            Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
            Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
            StoreAlign((__ubuf__ float*)dst + i * VL_FP32, aReg, pMask);
        }
    }
}

__aicore__ inline void LayerNormGradBase::VectorAdd(const LocalTensor<float>& dstTensor,
                                                    const LocalTensor<float>& src0Tensor,
                                                    const LocalTensor<float>& src1Tensor, const int64_t mSize,
                                                    const int64_t nSize, const int64_t stride)
{
    // VectorAdd
    uint16_t outerLoopTimes = CeilDiv(static_cast<int64_t>(nSize * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    uint16_t innerLoopTimes = mSize;
    uint32_t outerLoopStride = VL_FP32;
    uint32_t innerLoopStride = stride;
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src0 = (__ubuf__ float*)src0Tensor.GetPhyAddr();
            __ubuf__ float* src1 = (__ubuf__ float*)src1Tensor.GetPhyAddr();
            uint32_t count = nSize;
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadAlign(aReg, (__ubuf__ float*)src0 + i * outerLoopStride + 0 * innerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)src1 + i * outerLoopStride + 0 * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + 0 * innerLoopStride, aReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src0 = (__ubuf__ float*)src0Tensor.GetPhyAddr();
            __ubuf__ float* src1 = (__ubuf__ float*)src1Tensor.GetPhyAddr();
            uint32_t count = nSize;
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    LoadAlign(aReg, (__ubuf__ float*)src0 + i * outerLoopStride + j * innerLoopStride);
                    LoadAlign(bReg, (__ubuf__ float*)src1 + i * outerLoopStride + j * innerLoopStride);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                    Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                    StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride, aReg, pMask);
                }
            }
        }
    }
}

__aicore__ inline void LayerNormGradBase::VectorMul(const LocalTensor<float>& dstTensor,
                                                    const LocalTensor<float>& src0Tensor,
                                                    const LocalTensor<float>& src1Tensor, const int64_t count)
{
    // VectorMul
    if (count <= 0) {
        return;
    }
    uint16_t loopTimes = CeilDiv(static_cast<int64_t>(count * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    __VEC_SCOPE__
    {
        __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
        __ubuf__ float* src0 = (__ubuf__ float*)src0Tensor.GetPhyAddr();
        __ubuf__ float* src1 = (__ubuf__ float*)src1Tensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(count);
        AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
        AscendC::MicroAPI::MaskReg pMask;

        for (uint16_t i = 0; i < loopTimes; ++i) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            LoadAlign(aReg, (__ubuf__ float*)src0 + i * VL_FP32);
            LoadAlign(bReg, (__ubuf__ float*)src1 + i * VL_FP32);
            Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
            StoreAlign((__ubuf__ float*)dst + i * VL_FP32, cReg, pMask);
        }
    }
}

__aicore__ inline void LayerNormGradBase::NlastBroadcastMul(const LocalTensor<float>& dstTensor,
                                                            const LocalTensor<float>& src0Tensor,
                                                            const LocalTensor<float>& src1Tensor, const int64_t bSize,
                                                            const int64_t aSize)
{
    // NlastBroadcastMul
    if (bSize <= 0) {
        return;
    }
    if (aSize <= 0) {
        return;
    }
    uint16_t outerLoopTimes = CeilDiv(static_cast<int64_t>(aSize * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    uint16_t innerLoopTimes = bSize;
    uint32_t outerLoopStride = VL_FP32;
    uint32_t innerLoopStride = aSize;
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src0 = (__ubuf__ float*)src0Tensor.GetPhyAddr();
            __ubuf__ float* src1 = (__ubuf__ float*)src1Tensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(aSize);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadAlign(bReg, (__ubuf__ float*)src1 + i * outerLoopStride);
                LoadAlign(aReg, (__ubuf__ float*)src0 + i * outerLoopStride + 0 * innerLoopStride);
                Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + 0 * innerLoopStride, cReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src0 = (__ubuf__ float*)src0Tensor.GetPhyAddr();
            __ubuf__ float* src1 = (__ubuf__ float*)src1Tensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(aSize);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadAlign(bReg, (__ubuf__ float*)src1 + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    LoadAlign(aReg, (__ubuf__ float*)src0 + i * outerLoopStride + j * innerLoopStride);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                    StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride, cReg, pMask);
                }
            }
        }
    }
}

__aicore__ inline void LayerNormGradBase::LastReduceSumSmallR(const LocalTensor<float>& dstTensor,
                                                              const LocalTensor<float>& srcTensor, const int64_t aSize,
                                                              const int64_t rSize, const int64_t stride)
{
    // LastReduceSumSmallR
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    if (rSize <= VL_FP32) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::UnalignRegForStore UReg;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(aReg, (__ubuf__ float*)src + i * stride);
                Reduce<AscendC::Reg::ReduceType::SUM>(bReg, aReg, pMask);
                AscendC::MicroAPI::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::MicroAPI::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src0 = (__ubuf__ float*)srcTensor.GetPhyAddr();
            __ubuf__ float* src1 = (__ubuf__ float*)srcTensor.GetPhyAddr() + VL_FP32;
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
            AscendC::MicroAPI::UnalignRegForStore UReg;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg
                pFull = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(aReg, (__ubuf__ float*)src0 + i * stride);
                LoadAlign(bReg, (__ubuf__ float*)src1 + i * stride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                Reduce<AscendC::Reg::ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::MicroAPI::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::MicroAPI::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    }
}

__aicore__ inline void LayerNormGradBase::LastReduceSum(const LocalTensor<float>& dstTensor,
                                                        const LocalTensor<float>& srcTensor,
                                                        const LocalTensor<float>& reduceSumTempTensor,
                                                        const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    // LastReduceSum
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize <= CONST_TWO * VL_FP32) {
        LastReduceSumSmallR(dstTensor, srcTensor, aSize, rSize, stride);
        return;
    }

    int64_t ceilVLCount = CeilDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    int64_t floorVLCount = FloorDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    int64_t foldPoint = FindNearestPower2(ceilVLCount);

    uint16_t outerLoopTimes = aSize;
    uint16_t tailFoldLoopTimes = ceilVLCount - floorVLCount;
    uint32_t tailFoldElemCount = static_cast<uint32_t>(rSize - floorVLCount * VL_FP32);
    uint16_t mainFoldLoopTimes = floorVLCount - foldPoint;
    uint16_t unFoldLoopTimes = foldPoint + foldPoint - ceilVLCount;
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL_FP32;
    uint32_t outerLoopDstStride = Aligned(static_cast<int64_t>(foldPoint),
                                          static_cast<int64_t>(GetUbBlockSize() / sizeof(float)));

    int64_t foldSrcBOffset = foldPoint * VL_FP32;
    int64_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
    int64_t tailSrcBOffset = floorVLCount * VL_FP32;
    int64_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

    __VEC_SCOPE__
    {
        __ubuf__ float* dst = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* foldSrcA = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __ubuf__ float* foldSrcB = (__ubuf__ float*)srcTensor.GetPhyAddr() + foldSrcBOffset;
        __ubuf__ float* tailSrcA = (__ubuf__ float*)srcTensor.GetPhyAddr() + tailSrcAOffset;
        __ubuf__ float* tailSrcB = (__ubuf__ float*)srcTensor.GetPhyAddr() + tailSrcBOffset;
        __ubuf__ float* unFoldSrc = (__ubuf__ float*)srcTensor.GetPhyAddr() + unFoldSrcOffset;
        AscendC::MicroAPI::MaskReg pFull = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignRegForStore UReg;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            dst = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr() + i * outerLoopDstStride;
            for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg, dReg;
                LoadAlign(aReg, (__ubuf__ float*)foldSrcA + i * outerLoopStride + j * innerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)foldSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pFull);
                Reduce<AscendC::Reg::ReduceType::SUM>(dReg, cReg, pFull);
                AscendC::MicroAPI::StoreUnAlign((__ubuf__ float*&)dst, dReg, UReg, 1);
            }
            for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
                AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadAlign(aReg, (__ubuf__ float*)tailSrcA + i * outerLoopStride + j * innerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)tailSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Move<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                Reduce<AscendC::Reg::ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::MicroAPI::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg;
                LoadAlign(aReg, (__ubuf__ float*)unFoldSrc + i * outerLoopStride + j * innerLoopStride);
                Reduce<AscendC::Reg::ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::MicroAPI::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::MicroAPI::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    }
    LastReduceSumSmallR(dstTensor, reduceSumTempTensor, aSize, foldPoint, outerLoopDstStride);
}

template <uint32_t RSize, int32_t TailCount = -1, int32_t Index = 0, int32_t Depth = 1>
struct NlastDichotomyAdd {
    __aicore__ static inline void LoadAndAccumulate(AscendC::MicroAPI::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::MicroAPI::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::MicroAPI::RegTensor<float> aReg, bReg;
        __ubuf__ float* srcAOffset = srcA + stride * CONST_TWO;
        __ubuf__ float* srcBOffset = srcB + stride * CONST_TWO;
        if constexpr (TailCount <= 0) {
            NlastDichotomyAdd<(RSize + 1) / CONST_TWO>::LoadAndAccumulate(aReg, srcA, srcAOffset, pMask,
                                                                          stride * CONST_TWO);
            NlastDichotomyAdd<RSize / CONST_TWO>::LoadAndAccumulate(bReg, srcB, srcBOffset, pMask, stride * CONST_TWO);
        }
        Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::MicroAPI::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::MicroAPI::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        AscendC::MicroAPI::RegTensor<float> aReg, bReg;
        __ubuf__ float* srcAOffset = srcA + stride * CONST_TWO;
        __ubuf__ float* srcBOffset = srcB + stride * CONST_TWO;
        if constexpr (TailCount <= 0) {
            NlastDichotomyAdd<(RSize + 1) / CONST_TWO>::LoadAndAccumulate(aReg, srcA, srcAOffset, pMask,
                                                                          stride * CONST_TWO, offset);
            NlastDichotomyAdd<RSize / CONST_TWO>::LoadAndAccumulate(bReg, srcB, srcBOffset, pMask, stride * CONST_TWO,
                                                                    offset);
        } else {
            NlastDichotomyAdd<(RSize + 1) / CONST_TWO, TailCount, Index, Depth * CONST_TWO>::LoadAndAccumulate(
                aReg, srcA, srcAOffset, pMask, stride * CONST_TWO, offset);
            NlastDichotomyAdd<RSize / CONST_TWO, TailCount, Index + Depth, Depth * CONST_TWO>::LoadAndAccumulate(
                bReg, srcB, srcBOffset, pMask, stride * CONST_TWO, offset);
        }
        Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
};

template <int32_t TailCount, int32_t Index, int32_t Depth>
struct NlastDichotomyAdd<CONST_TWO, TailCount, Index, Depth> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::MicroAPI::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::MicroAPI::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::MicroAPI::RegTensor<float> aReg, bReg;
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::MicroAPI::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::MicroAPI::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        if constexpr (TailCount <= 0) {
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
            LoadAlign(aReg, (__ubuf__ float*)srcA);
            LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
            Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            LoadAlign(bReg, (__ubuf__ float*)srcB);
            LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
            Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
            Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
        } else {
            if constexpr (Index + Depth < TailCount) {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
                LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            } else if constexpr (Index < TailCount) {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            } else {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            }
        }
    }
};

template <>
struct NlastDichotomyAdd<CONST_TWO> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::MicroAPI::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::MicroAPI::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::MicroAPI::RegTensor<float> aReg, bReg;
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::MicroAPI::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::MicroAPI::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
        Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
        Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
        Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
};

template <>
struct NlastDichotomyAdd<1> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::MicroAPI::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::MicroAPI::MaskReg& pMask,
                                                    uint32_t stride)
    {
        LoadAlign(acc, (__ubuf__ float*)srcA);
    }
};

__aicore__ inline void LayerNormGradBase::NlastReduceSum(const LocalTensor<float>& dstTensor,
                                                         const LocalTensor<float>& srcTensor,
                                                         const LocalTensor<float>& reduceSumTempTensor,
                                                         const int64_t rSize, const int64_t aSize, const int64_t stride)
{
    // AscendC API
    uint32_t srcShape[2] = {static_cast<uint32_t>(rSize), static_cast<uint32_t>(stride)};
    bool srcInnerPad = false;
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(dstTensor, srcTensor, srcShape, srcInnerPad);
}

__aicore__ inline void LayerNormGradBase::UpdateCache(const LocalTensor<float>& dstTensor,
                                                      const LocalTensor<float>& srcTensor, const int64_t cacheID,
                                                      const int64_t stride, const int64_t count)
{
    // UpdateCache
    uint16_t outerLoopTimes = CeilDiv(static_cast<int64_t>(count * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    uint16_t innerLoopTimes = cacheID;
    uint32_t outerLoopStride = VL_FP32;
    uint32_t innerLoopStride = stride;
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* cah = (__ubuf__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
            __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + 0 * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                StoreAlign((__ubuf__ float*)cah + i * outerLoopStride, aReg, pMask);
            }
        }
    } else if (innerLoopTimes == CONST_TWO) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* cah = (__ubuf__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
            __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + 0 * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + 1 * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                StoreAlign((__ubuf__ float*)cah + i * outerLoopStride, aReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* cah = (__ubuf__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
            __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                StoreAlign((__ubuf__ float*)cah + i * outerLoopStride, aReg, pMask);
            }
        }
    }
}

__aicore__ inline void LayerNormGradBase::Normalize(const LocalTensor<float>& dstTensor,
                                                    const LocalTensor<float>& srcTensor,
                                                    const LocalTensor<float>& meanTensor,
                                                    const LocalTensor<float>& varTensor, const int64_t rowSize,
                                                    const int64_t colSize, const float epsilon)
{
    // Normalize
    uint16_t outerLoopTimes = rowSize;
    uint16_t innerLoopTimes = CeilDiv(static_cast<int64_t>(colSize * sizeof(float)),
                                      static_cast<int64_t>(GetVRegSize()));
    uint32_t outerLoopStride = colSize;
    uint32_t innerLoopStride = VL_FP32;
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* var = (__ubuf__ float*)varTensor.GetPhyAddr();
            uint32_t count;
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
            AscendC::MicroAPI::RegTensor<float> meanReg, varReg, rstdReg;
            AscendC::MicroAPI::MaskReg pMask;
            count = static_cast<uint32_t>(colSize);
            pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, (__ubuf__ float*)mean + i);
                LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(varReg, (__ubuf__ float*)var + i);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride + 0 * innerLoopStride);
                AscendC::MicroAPI::MaskReg
                    pregRstdAll1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                NormCommon::ComputeRstdNewtonRaphsonReg(varReg, rstdReg, pregRstdAll1, epsilon);
                Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(bReg, aReg, meanReg, pMask);
                Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, bReg, rstdReg, pMask);
                StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + 0 * innerLoopStride, cReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* var = (__ubuf__ float*)varTensor.GetPhyAddr();
            uint32_t count;
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
            AscendC::MicroAPI::RegTensor<float> meanReg, varReg, rstdReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                count = static_cast<uint32_t>(colSize);
                LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, (__ubuf__ float*)mean + i);
                LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(varReg, (__ubuf__ float*)var + i);
                AscendC::MicroAPI::MaskReg
                    pregRstdAll2 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                NormCommon::ComputeRstdNewtonRaphsonReg(varReg, rstdReg, pregRstdAll2, epsilon);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride + j * innerLoopStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(bReg, aReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, bReg, rstdReg, pMask);
                    StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride, cReg, pMask);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void LayerNormGradBase::StoreTensorForDtypeT(__ubuf__ T* dst,
                                                               AscendC::MicroAPI::RegTensor<float>& src,
                                                               AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xFp16;
        Cast<T, float, castTraitB322B16>(xFp16, src, preg);
        StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

} // namespace LayerNormGrad
#endif
