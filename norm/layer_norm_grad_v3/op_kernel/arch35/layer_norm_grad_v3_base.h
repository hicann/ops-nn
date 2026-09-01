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
 * \file layer_norm_grad_v3_base.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_BASE
#define LAYER_NORM_GRAD_V3_BASE

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "layer_norm_grad_v3_api.h"

/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize() { return Ops::Base::GetUbBlockSize(); }

namespace LayerNormGradV3 {
using namespace AscendC;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;
using namespace LayerNormGradV3::Arith;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::Move;
using AscendC::Reg::Reduce;
using AscendC::Reg::StoreAlign;

constexpr static AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::Reg::CastTrait castTraitB322B16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
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
class LayerNormGradV3Base {
public:
    __aicore__ inline LayerNormGradV3Base() : pipe_(nullptr){};

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
                                            const LocalTensor<float>& meanTensor, const LocalTensor<float>& rstdTensor,
                                            const int64_t rowSize, const int64_t colSize);
    __aicore__ inline static void ComputeGammaCommon(const LocalTensor<float>& dstTensor,
                                                     const LocalTensor<float>& dyTensor,
                                                     const LocalTensor<float>& xTensor,
                                                     const LocalTensor<float>& rstdTensor,
                                                     const LocalTensor<float>& meanTensor, const int64_t rowSize,
                                                     const int64_t colSize, const int64_t outerStride);
    template <typename T, typename TilingData>
    __aicore__ inline static void ProcessGammaBetaMainBlockCommon(
        const TilingData* td, const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor,
        const int64_t nfactor, LocalTensor<float>& dyMain, LocalTensor<float>& xMain, LocalTensor<float>& rstd,
        LocalTensor<float>& mean, TQue<QuePosition::VECIN, 1>& inQueueDy, TQue<QuePosition::VECIN, 1>& inQueueX,
        TQue<QuePosition::VECIN, 1>& inQueueParam, GlobalTensor<T>& dyInTensorGM, GlobalTensor<T>& xInTensorGM,
        GlobalTensor<float>& rstdInTensorGM, GlobalTensor<float>& meanInTensorGM);
    template <typename T, typename TilingData>
    __aicore__ inline static void ProcessGammaBetaFoldBlockCommon(
        const TilingData* td, const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor,
        const int64_t nfactor, LocalTensor<float>& dyMain, LocalTensor<float>& xMain,
        TQue<QuePosition::VECIN, 1>& inQueueDy, TQue<QuePosition::VECIN, 1>& inQueueX,
        TQue<QuePosition::VECIN, 1>& inQueueParam, GlobalTensor<T>& dyInTensorGM, GlobalTensor<T>& xInTensorGM,
        GlobalTensor<float>& rstdInTensorGM, GlobalTensor<float>& meanInTensorGM);
    template <typename PD_GAMMA_TYPE, typename TilingData>
    __aicore__ inline static void GammaBetaPrologueCommon(const TilingData* td,
                                                          TQue<QuePosition::VECOUT, 1>& outQueueSum,
                                                          LocalTensor<PD_GAMMA_TYPE>& beta,
                                                          LocalTensor<PD_GAMMA_TYPE>& gamma);
    template <typename PD_GAMMA_TYPE, typename TilingData>
    __aicore__ inline static void GammaBetaEpilogueCommon(
        const TilingData* td, const int64_t offset, const int64_t extent, TQue<QuePosition::VECOUT, 1>& outQueueSum,
        LocalTensor<float>& cacheTensor0, LocalTensor<float>& cacheTensor1, LocalTensor<PD_GAMMA_TYPE>& beta,
        LocalTensor<PD_GAMMA_TYPE>& gamma, GlobalTensor<PD_GAMMA_TYPE>& pdBetaOutTensorGM,
        GlobalTensor<PD_GAMMA_TYPE>& pdGammaOutTensorGM);
    template <typename TilingData>
    __aicore__ inline static void GammaBetaProcessSummationCommon(
        const TilingData* td, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor,
        LocalTensor<float>& tempTensor, LocalTensor<float>& dyMain, LocalTensor<float>& xMain,
        LocalTensor<float>& cacheTensor0, LocalTensor<float>& cacheTensor1, TQue<QuePosition::VECIN, 1>& inQueueDy,
        TQue<QuePosition::VECIN, 1>& inQueueX);
    template <typename T>
    __aicore__ inline static void ComputeDxCommon(
        const LocalTensor<T>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
        const LocalTensor<float>& gammaTensor, const LocalTensor<float>& sum1Tensor,
        const LocalTensor<float>& sum2Tensor, const LocalTensor<float>& rstdTensor, const int64_t rowSize,
        const int64_t colSize, const int64_t stride, const int64_t fullColSize);
    template <typename T>
    __aicore__ inline static void StoreTensorForDtypeT(__ubuf__ T* dst, AscendC::Reg::RegTensor<float>& src,
                                                       AscendC::Reg::MaskReg& preg, uint32_t offset);

protected:
    TPipe* pipe_;
}; // class LayerNormGradV3Base

// IMPL
__aicore__ inline int64_t LayerNormGradV3Base::FindNearestPower2(const int64_t value)
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

__aicore__ inline int64_t LayerNormGradV3Base::GetCacheID(const int64_t idx)
{
    return ScalarGetCountOfValue<1>(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

template <typename T>
__aicore__ inline void LayerNormGradV3Base::CastToFp32From(const LocalTensor<float>& dstTensor,
                                                           const LocalTensor<T>& srcTensor, const int64_t count)
{
    // CastToFp32From T
    CastToFp32From<T>(dstTensor, srcTensor, CONST_ONE, count, CONST_ZERO);
}

template <typename T>
__aicore__ inline void LayerNormGradV3Base::CastToFp32From(const LocalTensor<float>& dstTensor,
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
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        if (innerLoopTimes == 1) {
            __VEC_SCOPE__
            {
                __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
                __ubuf__ T* src = (__ubuf__ T*)srcTensor.GetPhyAddr();
                uint32_t count;
                AscendC::Reg::RegTensor<float> fp32Reg;
                AscendC::Reg::RegTensor<T> b16Reg;
                AscendC::Reg::MaskReg pMask;
                count = static_cast<uint32_t>(colSize);
                pMask = AscendC::Reg::UpdateMask<float>(count);
                for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                    AscendC::Reg::AddrReg srcAddrReg = AscendC::Reg::CreateAddrReg<T>(i, outerLoopSrcStride);
                    LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, (__ubuf__ T*)src, srcAddrReg);
                    Cast<float, T, castTraitB162B32>(fp32Reg, b16Reg, pMask);
                    AscendC::Reg::AddrReg dstAddrReg = AscendC::Reg::CreateAddrReg<float>(i, outerLoopDstStride);
                    StoreAlign((__ubuf__ float*)dst, fp32Reg, dstAddrReg, pMask);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
                __ubuf__ T* src = (__ubuf__ T*)srcTensor.GetPhyAddr();
                uint32_t count;
                AscendC::Reg::RegTensor<float> fp32Reg;
                AscendC::Reg::RegTensor<T> b16Reg;
                AscendC::Reg::MaskReg pMask;
                for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                    count = static_cast<uint32_t>(colSize);
                    for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                        pMask = AscendC::Reg::UpdateMask<float>(count);
                        AscendC::Reg::AddrReg srcAddrReg = AscendC::Reg::CreateAddrReg<T>(i, outerLoopSrcStride, j,
                                                                                          innerLoopStride);
                        LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, (__ubuf__ T*)src, srcAddrReg);
                        Cast<float, T, castTraitB162B32>(fp32Reg, b16Reg, pMask);
                        AscendC::Reg::AddrReg dstAddrReg = AscendC::Reg::CreateAddrReg<float>(i, outerLoopDstStride, j,
                                                                                              innerLoopStride);
                        StoreAlign((__ubuf__ float*)dst, fp32Reg, dstAddrReg, pMask);
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void LayerNormGradV3Base::CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor,
                                                   const int64_t rowSize, const int64_t colSize,
                                                   const int64_t dstStride, const int64_t srcStride)
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
__aicore__ inline void LayerNormGradV3Base::CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor,
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
__aicore__ inline void LayerNormGradV3Base::CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                                    const int64_t rowSize)
{
    // CopyOut
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = rowSize * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void LayerNormGradV3Base::CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                                    const int64_t rowSize, const int64_t colSize,
                                                    const int64_t dstStride, const int64_t srcStride)
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

__aicore__ inline void LayerNormGradV3Base::CopyUB2UB(const LocalTensor<float>& dstTensor,
                                                      const LocalTensor<float>& srcTensor, const int64_t count)
{
    // CopyUB2UB
    DataCopy(dstTensor, srcTensor,
             Aligned(static_cast<int64_t>(count), static_cast<int64_t>(GetUbBlockSize() / sizeof(float))));
}

template <typename T>
__aicore__ inline void LayerNormGradV3Base::CopyUB2UBWithCast(const LocalTensor<T>& dstTensor,
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

__aicore__ inline void LayerNormGradV3Base::VectorAdd(const LocalTensor<float>& dstTensor,
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
        AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
        AscendC::Reg::MaskReg pMask;
        for (uint16_t i = 0; i < loopTimes; ++i) {
            pMask = AscendC::Reg::UpdateMask<float>(sreg);
            LoadAlign(aReg, (__ubuf__ float*)src0 + i * VL_FP32);
            LoadAlign(bReg, (__ubuf__ float*)src1 + i * VL_FP32);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
            Move<float, AscendC::Reg::MaskMergeMode::MERGING>(aReg, cReg, pMask);
            StoreAlign((__ubuf__ float*)dst + i * VL_FP32, aReg, pMask);
        }
    }
}

__aicore__ inline void LayerNormGradV3Base::VectorAdd(const LocalTensor<float>& dstTensor,
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
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::Reg::UpdateMask<float>(count);
                LoadAlign(aReg, (__ubuf__ float*)src0 + i * outerLoopStride + 0 * innerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)src1 + i * outerLoopStride + 0 * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Move<float, AscendC::Reg::MaskMergeMode::MERGING>(aReg, cReg, pMask);
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
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::Reg::UpdateMask<float>(count);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    LoadAlign(aReg, (__ubuf__ float*)src0 + i * outerLoopStride + j * innerLoopStride);
                    LoadAlign(bReg, (__ubuf__ float*)src1 + i * outerLoopStride + j * innerLoopStride);
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                    Move<float, AscendC::Reg::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                    StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride, aReg, pMask);
                }
            }
        }
    }
}

__aicore__ inline void LayerNormGradV3Base::VectorMul(const LocalTensor<float>& dstTensor,
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
        AscendC::Reg::RegTensor<float> aReg, bReg, mulReg;
        AscendC::Reg::MaskReg pMask;

        for (uint16_t i = 0; i < loopTimes; ++i) {
            pMask = AscendC::Reg::UpdateMask<float>(sreg);
            LoadAlign(aReg, (__ubuf__ float*)src0 + i * VL_FP32);
            LoadAlign(bReg, (__ubuf__ float*)src1 + i * VL_FP32);
            Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(mulReg, aReg, bReg, pMask);
            StoreAlign((__ubuf__ float*)dst + i * VL_FP32, mulReg, pMask);
        }
    }
}

__aicore__ inline void LayerNormGradV3Base::NlastBroadcastMul(const LocalTensor<float>& dstTensor,
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
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::Reg::UpdateMask<float>(count);
                LoadAlign(bReg, (__ubuf__ float*)src1 + i * outerLoopStride);
                LoadAlign(aReg, (__ubuf__ float*)src0 + i * outerLoopStride + 0 * innerLoopStride);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
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
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::Reg::UpdateMask<float>(count);
                LoadAlign(bReg, (__ubuf__ float*)src1 + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    LoadAlign(aReg, (__ubuf__ float*)src0 + i * outerLoopStride + j * innerLoopStride);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                    StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride, cReg, pMask);
                }
            }
        }
    }
}

__aicore__ inline void LayerNormGradV3Base::LastReduceSumSmallR(const LocalTensor<float>& dstTensor,
                                                                const LocalTensor<float>& srcTensor,
                                                                const int64_t aSize, const int64_t rSize,
                                                                const int64_t stride)
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
            AscendC::Reg::RegTensor<float> aReg, bReg;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::UnalignRegForStore UReg;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(aReg, (__ubuf__ float*)src + i * stride);
                Reduce<AscendC::Reg::ReduceType::SUM>(bReg, aReg, pMask);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::Reg::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src0 = (__ubuf__ float*)srcTensor.GetPhyAddr();
            __ubuf__ float* src1 = (__ubuf__ float*)srcTensor.GetPhyAddr() + VL_FP32;
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            AscendC::Reg::UnalignRegForStore UReg;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadAlign(aReg, (__ubuf__ float*)src0 + i * stride);
                LoadAlign(bReg, (__ubuf__ float*)src1 + i * stride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Move<float, AscendC::Reg::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                Reduce<AscendC::Reg::ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::Reg::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    }
}

__aicore__ inline void LayerNormGradV3Base::LastReduceSum(const LocalTensor<float>& dstTensor,
                                                          const LocalTensor<float>& srcTensor,
                                                          const LocalTensor<float>& reduceSumTempTensor,
                                                          const int64_t aSize, const int64_t rSize,
                                                          const int64_t stride)
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
        AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::UnalignRegForStore UReg;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            dst = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr() + i * outerLoopDstStride;
            for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
                AscendC::Reg::RegTensor<float> aReg, bReg, cReg, dReg;
                LoadAlign(aReg, (__ubuf__ float*)foldSrcA + i * outerLoopStride + j * innerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)foldSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pFull);
                Reduce<AscendC::Reg::ReduceType::SUM>(dReg, cReg, pFull);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, dReg, UReg, 1);
            }
            for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
                AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
                LoadAlign(aReg, (__ubuf__ float*)tailSrcA + i * outerLoopStride + j * innerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)tailSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Move<float, AscendC::Reg::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                Reduce<AscendC::Reg::ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
                AscendC::Reg::RegTensor<float> aReg, bReg;
                LoadAlign(aReg, (__ubuf__ float*)unFoldSrc + i * outerLoopStride + j * innerLoopStride);
                Reduce<AscendC::Reg::ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::Reg::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    }
    LastReduceSumSmallR(dstTensor, reduceSumTempTensor, aSize, foldPoint, outerLoopDstStride);
}

template <uint32_t RSize, int32_t TailCount = -1, int32_t Index = 0, int32_t Depth = 1>
struct NlastDichotomyAdd {
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        __ubuf__ float* srcAOffset = srcA + stride * CONST_TWO;
        __ubuf__ float* srcBOffset = srcB + stride * CONST_TWO;
        if constexpr (TailCount <= 0) {
            NlastDichotomyAdd<(RSize + 1) / CONST_TWO>::LoadAndAccumulate(aReg, srcA, srcAOffset, pMask,
                                                                          stride * CONST_TWO);
            NlastDichotomyAdd<RSize / CONST_TWO>::LoadAndAccumulate(bReg, srcB, srcBOffset, pMask, stride * CONST_TWO);
        }
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
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
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
};

template <int32_t TailCount, int32_t Index, int32_t Depth>
struct NlastDichotomyAdd<CONST_TWO, TailCount, Index, Depth> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        if constexpr (TailCount <= 0) {
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            LoadAlign(aReg, (__ubuf__ float*)srcA);
            LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            LoadAlign(bReg, (__ubuf__ float*)srcB);
            LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
        } else {
            if constexpr (Index + Depth < TailCount) {
                AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
                LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            } else if constexpr (Index < TailCount) {
                AscendC::Reg::RegTensor<float> aReg, bReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            } else {
                AscendC::Reg::RegTensor<float> aReg, bReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            }
        }
    }
};

template <>
struct NlastDichotomyAdd<CONST_TWO> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
};

template <>
struct NlastDichotomyAdd<1> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride)
    {
        LoadAlign(acc, (__ubuf__ float*)srcA);
    }
};

__aicore__ inline void LayerNormGradV3Base::NlastReduceSum(const LocalTensor<float>& dstTensor,
                                                           const LocalTensor<float>& srcTensor,
                                                           const LocalTensor<float>& reduceSumTempTensor,
                                                           const int64_t rSize, const int64_t aSize,
                                                           const int64_t stride)
{
    // AscendC API
    uint32_t srcShape[2] = {static_cast<uint32_t>(rSize), static_cast<uint32_t>(stride)};
    bool srcInnerPad = false;
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(dstTensor, srcTensor, srcShape, srcInnerPad);
}

__aicore__ inline void LayerNormGradV3Base::UpdateCache(const LocalTensor<float>& dstTensor,
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
            AscendC::Reg::RegTensor<float> aReg, bReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::Reg::UpdateMask<float>(sreg);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + 0 * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
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
            AscendC::Reg::RegTensor<float> aReg, bReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::Reg::UpdateMask<float>(sreg);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
                LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + 0 * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + 1 * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
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
            AscendC::Reg::RegTensor<float> aReg, bReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::Reg::UpdateMask<float>(sreg);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                StoreAlign((__ubuf__ float*)cah + i * outerLoopStride, aReg, pMask);
            }
        }
    }
}

__aicore__ inline void LayerNormGradV3Base::Normalize(const LocalTensor<float>& dstTensor,
                                                      const LocalTensor<float>& srcTensor,
                                                      const LocalTensor<float>& meanTensor,
                                                      const LocalTensor<float>& rstdTensor, const int64_t rowSize,
                                                      const int64_t colSize)
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
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            uint32_t count;
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            AscendC::Reg::RegTensor<float> meanReg, rstdReg;
            AscendC::Reg::MaskReg pMask;
            count = static_cast<uint32_t>(colSize);
            pMask = AscendC::Reg::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(meanReg, (__ubuf__ float*)mean + i);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstdReg, (__ubuf__ float*)rstd + i);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride + 0 * innerLoopStride);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, aReg, meanReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, bReg, rstdReg, pMask);
                StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + 0 * innerLoopStride, cReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            uint32_t count;
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            AscendC::Reg::RegTensor<float> meanReg, rstdReg;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                count = static_cast<uint32_t>(colSize);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(meanReg, (__ubuf__ float*)mean + i);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstdReg, (__ubuf__ float*)rstd + i);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride + j * innerLoopStride);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, aReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, bReg, rstdReg, pMask);
                    StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride, cReg, pMask);
                }
            }
        }
    }
}

__aicore__ inline void LayerNormGradV3Base::ComputeGammaCommon(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& rstdTensor, const LocalTensor<float>& meanTensor, const int64_t rowSize,
    const int64_t colSize, const int64_t outerStride)
{
    int64_t colLength = colSize * sizeof(float);
    uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
    uint16_t innerLoopTimes = CeilDiv(static_cast<int64_t>(colLength), static_cast<int64_t>(GetVRegSize()));
    uint32_t innerStride = static_cast<uint32_t>(GetVRegSize() / sizeof(float));
    uint32_t outerLoopStride = static_cast<uint32_t>(outerStride);
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* x = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ float* dy = (__ubuf__ float*)dyTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(colSize);
            AscendC::Reg::MaskReg pMask;
            pMask = AscendC::Reg::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                AscendC::Reg::RegTensor<float> meanReg;
                AscendC::Reg::RegTensor<float> rstdReg;
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(meanReg, (__ubuf__ float*)mean + i);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstdReg, (__ubuf__ float*)rstd + i);

                AscendC::Reg::RegTensor<float> xReg;
                AscendC::Reg::RegTensor<float> dyReg;
                LoadAlign(xReg, (__ubuf__ float*)x + i * outerLoopStride + 0 * innerStride);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                LoadAlign(dyReg, (__ubuf__ float*)dy + i * outerLoopStride + 0 * innerStride);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + 0 * innerStride, xReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* x = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ float* dy = (__ubuf__ float*)dyTensor.GetPhyAddr();
            __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                uint32_t count = static_cast<uint32_t>(colSize);
                AscendC::Reg::RegTensor<float> meanReg;
                AscendC::Reg::RegTensor<float> rstdReg;
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(meanReg, (__ubuf__ float*)mean + i);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstdReg, (__ubuf__ float*)rstd + i);

                AscendC::Reg::RegTensor<float> xReg;
                AscendC::Reg::RegTensor<float> dyReg;
                AscendC::Reg::MaskReg pMask;
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    LoadAlign(xReg, (__ubuf__ float*)x + i * outerLoopStride + j * innerStride);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    LoadAlign(dyReg, (__ubuf__ float*)dy + i * outerLoopStride + j * innerStride);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                    StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + j * innerStride, xReg, pMask);
                }
            }
        }
    }
}

template <typename T, typename TilingData>
__aicore__ inline void LayerNormGradV3Base::ProcessGammaBetaMainBlockCommon(
    const TilingData* td, const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor,
    LocalTensor<float>& dyMain, LocalTensor<float>& xMain, LocalTensor<float>& rstd, LocalTensor<float>& mean,
    TQue<QuePosition::VECIN, 1>& inQueueDy, TQue<QuePosition::VECIN, 1>& inQueueX,
    TQue<QuePosition::VECIN, 1>& inQueueParam, GlobalTensor<T>& dyInTensorGM, GlobalTensor<T>& xInTensorGM,
    GlobalTensor<float>& rstdInTensorGM, GlobalTensor<float>& meanInTensorGM)
{
    int64_t offset = ni * td->gammaBetaNfactor + basicBlockIdx * td->gammaBetaMfactor * td->col;
    dyMain = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyMain, dyInTensorGM[offset], mfactor, nfactor, td->gammaBetaNfactor, td->col);
        inQueueDy.EnQue(dyMain);
        dyMain = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyMain.ReinterpretCast<T>()[td->gammaBetaNfactor];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td->gammaBetaNfactor, td->col);
        inQueueDy.EnQue(dyMain);
        dyMain = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyMain, castTempTensor, mfactor, nfactor, td->gammaBetaNfactor);
    }

    if (td->pdgammaIsRequire) {
        xMain = inQueueX.template AllocTensor<float>();
        if constexpr (IsSameType<T, float>::value) {
            CopyIn(xMain.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td->gammaBetaNfactor, td->col);
            inQueueX.EnQue(xMain);
            xMain = inQueueX.template DeQue<float>();
        } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            LocalTensor<T> castTempTensor = xMain.ReinterpretCast<T>()[td->gammaBetaNfactor];
            CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td->gammaBetaNfactor, td->col);
            inQueueX.EnQue(xMain);
            xMain = inQueueX.template DeQue<float>();
            CastToFp32From<T>(xMain, castTempTensor, mfactor, nfactor, td->gammaBetaNfactor);
        }

        offset = basicBlockIdx * td->gammaBetaMfactor;
        rstd = inQueueParam.template AllocTensor<float>();
        CopyIn(rstd, rstdInTensorGM[offset], mfactor);
        inQueueParam.EnQue(rstd);
        rstd = inQueueParam.template DeQue<float>();

        mean = inQueueParam.template AllocTensor<float>();
        CopyIn(mean, meanInTensorGM[offset], mfactor);
        inQueueParam.EnQue(mean);
        mean = inQueueParam.template DeQue<float>();

        ComputeGammaCommon(xMain, dyMain, xMain, rstd, mean, mfactor, td->gammaBetaNfactor, td->gammaBetaNfactor);
        inQueueParam.FreeTensor(rstd);
        inQueueParam.FreeTensor(mean);
        if (!td->pdbetaIsRequire) {
            inQueueDy.FreeTensor(dyMain);
        }
    }
}

template <typename T, typename TilingData>
__aicore__ inline void LayerNormGradV3Base::ProcessGammaBetaFoldBlockCommon(
    const TilingData* td, const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor,
    LocalTensor<float>& dyMain, LocalTensor<float>& xMain, TQue<QuePosition::VECIN, 1>& inQueueDy,
    TQue<QuePosition::VECIN, 1>& inQueueX, TQue<QuePosition::VECIN, 1>& inQueueParam, GlobalTensor<T>& dyInTensorGM,
    GlobalTensor<T>& xInTensorGM, GlobalTensor<float>& rstdInTensorGM, GlobalTensor<float>& meanInTensorGM)
{
    int64_t offset = ni * td->gammaBetaNfactor +
                     (basicBlockIdx + td->gammaBetaBasicBlockLoop) * td->gammaBetaMfactor * td->col;
    LocalTensor<float> dyFold = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyFold, dyInTensorGM[offset], mfactor, nfactor, td->gammaBetaNfactor, td->col);
        inQueueDy.EnQue(dyFold);
        dyFold = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyFold.ReinterpretCast<T>()[td->gammaBetaNfactor];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td->gammaBetaNfactor, td->col);
        inQueueDy.EnQue(dyFold);
        dyFold = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyFold, castTempTensor, mfactor, nfactor, td->gammaBetaNfactor);
    }
    if (td->pdbetaIsRequire) {
        VectorAdd(dyMain, dyMain, dyFold, mfactor, nfactor, td->gammaBetaNfactor);
    }

    if (td->pdgammaIsRequire) {
        LocalTensor<float> xFold = inQueueX.template AllocTensor<float>();
        if constexpr (IsSameType<T, float>::value) {
            CopyIn(xFold.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td->gammaBetaNfactor, td->col);
            inQueueX.EnQue(xFold);
            xFold = inQueueX.template DeQue<float>();
        } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            LocalTensor<T> castTempTensor = xFold.ReinterpretCast<T>()[td->gammaBetaNfactor];
            CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td->gammaBetaNfactor, td->col);
            inQueueX.EnQue(xFold);
            xFold = inQueueX.template DeQue<float>();
            CastToFp32From<T>(xFold, castTempTensor, mfactor, nfactor, td->gammaBetaNfactor);
        }

        offset = (basicBlockIdx + td->gammaBetaBasicBlockLoop) * td->gammaBetaMfactor;
        LocalTensor<float> rstdFold = inQueueParam.template AllocTensor<float>();
        CopyIn(rstdFold, rstdInTensorGM[offset], mfactor);
        inQueueParam.EnQue(rstdFold);
        rstdFold = inQueueParam.template DeQue<float>();

        LocalTensor<float> meanFold = inQueueParam.template AllocTensor<float>();
        CopyIn(meanFold, meanInTensorGM[offset], mfactor);
        inQueueParam.EnQue(meanFold);
        meanFold = inQueueParam.template DeQue<float>();

        ComputeGammaCommon(xFold, dyFold, xFold, rstdFold, meanFold, mfactor, td->gammaBetaNfactor,
                           td->gammaBetaNfactor);
        inQueueParam.FreeTensor(rstdFold);
        inQueueParam.FreeTensor(meanFold);
        inQueueDy.FreeTensor(dyFold);
        VectorAdd(xMain, xMain, xFold, mfactor, nfactor, td->gammaBetaNfactor);
        inQueueX.FreeTensor(xFold);
    } else {
        inQueueDy.FreeTensor(dyFold);
    }
}

template <typename PD_GAMMA_TYPE, typename TilingData>
__aicore__ inline void LayerNormGradV3Base::GammaBetaPrologueCommon(const TilingData* td,
                                                                    TQue<QuePosition::VECOUT, 1>& outQueueSum,
                                                                    LocalTensor<PD_GAMMA_TYPE>& beta,
                                                                    LocalTensor<PD_GAMMA_TYPE>& gamma)
{
    if (td->pdbetaIsRequire) {
        beta = outQueueSum.template AllocTensor<PD_GAMMA_TYPE>();
    }

    if (td->pdgammaIsRequire) {
        gamma = outQueueSum.template AllocTensor<PD_GAMMA_TYPE>();
    }
}

template <typename PD_GAMMA_TYPE, typename TilingData>
__aicore__ inline void LayerNormGradV3Base::GammaBetaEpilogueCommon(
    const TilingData* td, const int64_t offset, const int64_t extent, TQue<QuePosition::VECOUT, 1>& outQueueSum,
    LocalTensor<float>& cacheTensor0, LocalTensor<float>& cacheTensor1, LocalTensor<PD_GAMMA_TYPE>& beta,
    LocalTensor<PD_GAMMA_TYPE>& gamma, GlobalTensor<PD_GAMMA_TYPE>& pdBetaOutTensorGM,
    GlobalTensor<PD_GAMMA_TYPE>& pdGammaOutTensorGM)
{
    if (td->pdbetaIsRequire) {
        CopyUB2UBWithCast<PD_GAMMA_TYPE>(beta, cacheTensor0[td->gammaBetaResultCacheID * td->gammaBetaNfactor], extent);
        outQueueSum.EnQue(beta);
        beta = outQueueSum.template DeQue<PD_GAMMA_TYPE>();
        CopyOut<PD_GAMMA_TYPE>(pdBetaOutTensorGM[offset], beta, extent);
        outQueueSum.FreeTensor(beta);
    }

    if (td->pdgammaIsRequire) {
        CopyUB2UBWithCast<PD_GAMMA_TYPE>(gamma, cacheTensor1[td->gammaBetaResultCacheID * td->gammaBetaNfactor],
                                         extent);
        outQueueSum.EnQue(gamma);
        gamma = outQueueSum.template DeQue<PD_GAMMA_TYPE>();
        CopyOut<PD_GAMMA_TYPE>(pdGammaOutTensorGM[offset], gamma, extent);
        outQueueSum.FreeTensor(gamma);
    }
}

template <typename TilingData>
__aicore__ inline void LayerNormGradV3Base::GammaBetaProcessSummationCommon(
    const TilingData* td, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor,
    LocalTensor<float>& tempTensor, LocalTensor<float>& dyMain, LocalTensor<float>& xMain,
    LocalTensor<float>& cacheTensor0, LocalTensor<float>& cacheTensor1, TQue<QuePosition::VECIN, 1>& inQueueDy,
    TQue<QuePosition::VECIN, 1>& inQueueX)
{
    int64_t cacheID = GetCacheID(basicBlockIdx);
    uint32_t srcShape[2] = {static_cast<uint32_t>(mfactor), static_cast<uint32_t>(td->gammaBetaNfactor)};

    if (td->pdbetaIsRequire) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tempTensor, dyMain, srcShape, false);
        inQueueDy.FreeTensor(dyMain);
        UpdateCache(cacheTensor0, tempTensor, cacheID, td->gammaBetaNfactor, nfactor);
    }

    if (td->pdgammaIsRequire) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tempTensor, xMain, srcShape, false);
        inQueueX.FreeTensor(xMain);
        UpdateCache(cacheTensor1, tempTensor, cacheID, td->gammaBetaNfactor, nfactor);
    }
}

template <typename T>
__aicore__ inline void LayerNormGradV3Base::ComputeDxCommon(
    const LocalTensor<T>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& gammaTensor, const LocalTensor<float>& sum1Tensor, const LocalTensor<float>& sum2Tensor,
    const LocalTensor<float>& rstdTensor, const int64_t rowSize, const int64_t colSize, const int64_t stride,
    const int64_t fullColSize)
{
    constexpr static uint32_t VL = GetVRegSize() / sizeof(float);
    uint16_t outerLoopTimes = rowSize;
    uint16_t innerLoopTimes = CeilDiv(static_cast<int64_t>(colSize * sizeof(float)),
                                      static_cast<int64_t>(GetVRegSize()));
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL;
    float floatN = static_cast<float>(fullColSize);
    float reciprocalN = (floatN != 0.0f) ? static_cast<float>(1) / floatN : 0.0f;

    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __ubuf__ T* dst = (__ubuf__ T*)dstTensor.GetPhyAddr();
            __ubuf__ float* dy = (__ubuf__ float*)dyTensor.GetPhyAddr();
            __ubuf__ float* x = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ float* gamma = (__ubuf__ float*)gammaTensor.GetPhyAddr();
            __ubuf__ float* sum1 = (__ubuf__ float*)sum1Tensor.GetPhyAddr();
            __ubuf__ float* sum2 = (__ubuf__ float*)sum2Tensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            uint32_t count;

            AscendC::Reg::RegTensor<float> xReg, dyReg, dxReg;
            AscendC::Reg::RegTensor<float> sum1Reg, sum2Reg, rstdReg;
            AscendC::Reg::RegTensor<float> gammaReg;
            AscendC::Reg::RegTensor<float> Reg0, Reg1, Reg2, Reg3, Reg4, Reg5;
            AscendC::Reg::MaskReg pMask;
            count = static_cast<uint32_t>(colSize);
            pMask = AscendC::Reg::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sum1Reg, (__ubuf__ float*)sum1 + i);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sum2Reg, (__ubuf__ float*)sum2 + i);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstdReg, (__ubuf__ float*)rstd + i);
                LoadAlign(dyReg, (__ubuf__ float*)dy + i * outerLoopStride + 0 * innerLoopStride);
                LoadAlign(xReg, (__ubuf__ float*)x + i * outerLoopStride + 0 * innerLoopStride);
                LoadAlign(gammaReg, (__ubuf__ float*)gamma + 0 * innerLoopStride);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg0, dyReg, gammaReg, pMask);
                Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg1, Reg0, floatN, pMask);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg2, Reg1, sum1Reg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg3, xReg, sum2Reg, pMask);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg4, Reg2, Reg3, pMask);
                Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg5, Reg4, reciprocalN, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dxReg, Reg5, rstdReg, pMask);
                StoreTensorForDtypeT<T>(dst, dxReg, pMask, i * outerLoopStride);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ T* dst = (__ubuf__ T*)dstTensor.GetPhyAddr();
            __ubuf__ float* dy = (__ubuf__ float*)dyTensor.GetPhyAddr();
            __ubuf__ float* x = (__ubuf__ float*)xTensor.GetPhyAddr();
            __ubuf__ float* gamma = (__ubuf__ float*)gammaTensor.GetPhyAddr();
            __ubuf__ float* sum1 = (__ubuf__ float*)sum1Tensor.GetPhyAddr();
            __ubuf__ float* sum2 = (__ubuf__ float*)sum2Tensor.GetPhyAddr();
            __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
            uint32_t count;

            AscendC::Reg::RegTensor<float> xReg, dyReg, dxReg;
            AscendC::Reg::RegTensor<float> sum1Reg, sum2Reg, rstdReg;
            AscendC::Reg::RegTensor<float> gammaReg;
            AscendC::Reg::RegTensor<float> Reg0, Reg1, Reg2, Reg3, Reg4, Reg5;
            AscendC::Reg::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                count = static_cast<uint32_t>(colSize);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sum1Reg, (__ubuf__ float*)sum1 + i);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sum2Reg, (__ubuf__ float*)sum2 + i);
                LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstdReg, (__ubuf__ float*)rstd + i);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    LoadAlign(dyReg, (__ubuf__ float*)dy + i * outerLoopStride + j * innerLoopStride);
                    LoadAlign(xReg, (__ubuf__ float*)x + i * outerLoopStride + j * innerLoopStride);
                    LoadAlign(gammaReg, (__ubuf__ float*)gamma + j * innerLoopStride);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg0, dyReg, gammaReg, pMask);
                    Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg1, Reg0, floatN, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg2, Reg1, sum1Reg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg3, xReg, sum2Reg, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg4, Reg2, Reg3, pMask);
                    Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(Reg5, Reg4, reciprocalN, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dxReg, Reg5, rstdReg, pMask);
                    StoreTensorForDtypeT<T>(dst, dxReg, pMask, i * outerLoopStride + j * innerLoopStride);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void LayerNormGradV3Base::StoreTensorForDtypeT(__ubuf__ T* dst, AscendC::Reg::RegTensor<float>& src,
                                                                 AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::Reg::RegTensor<T> xFp16;
        Cast<T, float, castTraitB322B16>(xFp16, src, preg);
        StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}
} // namespace LayerNormGradV3
#endif
