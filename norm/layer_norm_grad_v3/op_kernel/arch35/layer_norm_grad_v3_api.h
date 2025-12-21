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
 * \file layer_norm_grad_v3_api.h
 * \brief
 */
#ifndef LAYER_NORM_GRAD_V3_API_
#define LAYER_NORM_GRAD_V3_API_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormGradV3 {
using namespace AscendC;

constexpr static int64_t BLOCK_SIZE = 32;

namespace Arith {
/**
 * Computes the minimum of two 64-bit integers (aicore)
 *
 * @param a First integer value
 * @param b Second integer value
 * @return The smaller value between a and b
 *
 * Note: This is optimized for AI Core execution (low-latency).
 */
__aicore__ inline int64_t Min(int64_t a, int64_t b)
{
    return (a < b) ? a : b;
}

/**
 * Computes the maximum of two 64-bit integers (aicore)
 *
 * @param a First integer value
 * @param b Second integer value
 * @return The larger value between a and b
 *
 * Note: This is optimized for AI Core execution (low-latency).
 */
__aicore__ inline int64_t Max(int64_t a, int64_t b)
{
    return (a > b) ? a : b;
}

/**
 * Computes ceiling division of two 64-bit integers (aicore)
 *
 * @param dividend The number to be divided
 * @param divisor The number to divide by (must be != 0)
 * @return The smallest integer greater than or equal to the exact quotient
 *
 * Note: Implements (dividend + divisor - 1) / divisor without overflow risk
 *       when dividend >= 0 and divisor > 0.
 */
__aicore__ inline int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    // Assumes divisor != 0 (caller's responsibility)
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

/**
 * Computes floor division of two 64-bit integers (aicore)
 *
 * @param dividend The number to be divided
 * @param divisor The number to divide by (must be != 0)
 * @return The largest integer less than or equal to the exact quotient
 *
 * Note: Standard integer division already floors for positive numbers,
 *       but handles negative cases correctly.
 */
__aicore__ inline int64_t FloorDiv(int64_t dividend, int64_t divisor)
{
    // Standard integer division truncates toward zero
    if (divisor == 0) {
        return 0;
    }
    return dividend / divisor;
}

/**
 * Aligns a value up to the nearest multiple of alignment (aicore)
 *
 * @param x The value to be aligned
 * @param alignment The alignment boundary (must be > 0)
 * @return The smallest value >= x that's a multiple of alignment
 *
 * Note: Uses ceiling division to compute the alignment without overflow
 *       when x >= 0 and alignment > 0. Returns 0 for alignment <= 0.
 */
__aicore__ inline int64_t Align(int64_t x, int64_t alignment)
{
    if (alignment <= 0) {
        return 0;
    }
    return CeilDiv(x, alignment) * alignment;
}
} // namespace Arith

namespace DMA {

template <typename T>
__aicore__ inline void CopyIn1D(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, const int64_t n)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = n * sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void CopyIn2D(
    const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, const int64_t rowSize, const int64_t colSize,
    const int64_t dstStride, const int64_t srcStride)
{
    DataCopyExtParams params;
    params.blockCount = rowSize;
    params.blockLen = colSize * sizeof(T);
    params.srcStride = srcStride * sizeof(T) - params.blockLen;
    params.dstStride = (dstStride * sizeof(T) -
                        Arith::Align(static_cast<int64_t>(params.blockLen), static_cast<int64_t>(BLOCK_SIZE))) /
                       BLOCK_SIZE;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void CopyOut1D(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const int64_t n)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = n * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, params);
}

} // namespace DMA

namespace CalcOp {
__aicore__ inline void VectorAdd(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& src0Tensor, const LocalTensor<float>& src1Tensor,
    const int64_t count)
{
    constexpr static int64_t VREG_SIZE = 256;
    constexpr static int64_t VL_FP32 = VREG_SIZE / sizeof(float);
    uint16_t loopTimes = Arith::CeilDiv(static_cast<int64_t>(count * sizeof(float)), static_cast<int64_t>(VREG_SIZE));
    __VEC_SCOPE__
    {
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* src0 = (__local_mem__ float*)src0Tensor.GetPhyAddr();
        __local_mem__ float* src1 = (__local_mem__ float*)src1Tensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(count);
        AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < loopTimes; ++i) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            DataCopy(aReg, (__local_mem__ float*)src0 + i * VL_FP32);
            DataCopy(bReg, (__local_mem__ float*)src1 + i * VL_FP32);
            Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
            DataCopy((__local_mem__ float*)dst + i * VL_FP32, aReg, pMask);
        }
    }
}

__aicore__ inline int64_t FindNearestPower2(const int64_t value)
{
    constexpr static int64_t CONST_ZERO = 0;
    constexpr static int64_t CONST_ONE = 1;
    constexpr static int64_t CONST_TWO = 2;
    constexpr static int64_t CONST_FOUR = 4;
    constexpr static int64_t CONST_SIXTY_THREE = 63;
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

__aicore__ inline int64_t GetCacheID(const int64_t idx)
{
    constexpr static int64_t CONST_ONE = 1;
    return ScalarGetCountOfValue<1>(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

__aicore__ inline void UpdateCache(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t cacheID,
    const int64_t stride, const int64_t count)
{
    constexpr static int64_t VREG_SIZE = 256;
    constexpr static int64_t VL_FP32 = VREG_SIZE / sizeof(float);
    uint16_t outerLoopTimes = Arith::CeilDiv(count * sizeof(float), VREG_SIZE);
    uint16_t innerLoopTimes = cacheID;
    uint32_t outerLoopStride = VL_FP32;
    uint32_t innerLoopStride = stride;
    __VEC_SCOPE__
    {
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* cah = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(count);
        AscendC::MicroAPI::RegTensor<float> aReg, bReg;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            DataCopy(aReg, (__local_mem__ float*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                DataCopy(bReg, (__local_mem__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            DataCopy((__local_mem__ float*)cah + i * outerLoopStride, aReg, pMask);
        }
    }
}

} // namespace CalcOp

} // namespace LayerNormGradV3

#endif // LAYER_NORM_GRAD_V3_API_