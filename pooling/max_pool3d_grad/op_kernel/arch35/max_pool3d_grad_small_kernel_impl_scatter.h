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
 * \file max_pool3d_grad_small_kernel_impl_scatter.h
 * \brief
 */

#ifndef MAX_POOL3D_GRAD_SMALL_KERNEL_IMPL_SCATTER_H
#define MAX_POOL3D_GRAD_SMALL_KERNEL_IMPL_SCATTER_H

#include "max_pool3d_grad_small_kernel_scatter.h"
#include "max_pool3d_grad_small_kernel.h"

namespace MaxPool3DSmallKernelNameSpace
{
    
template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(TYPE_ORIG_X);
    LocalTensor<TYPE_ORIG_X> yLocal = outputQue_.AllocTensor<TYPE_ORIG_X>();
    Duplicate(yLocal, TYPE_ORIG_X(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::singleLineProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    int64_t wOutput = wOutput_;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curDIndex = dAxisIndex_ * dOutputInner_;
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    uint16_t dArgmaxActual = dArgmaxActual_;
    int64_t wArgmaxActual = wArgmaxActual_;
    int64_t wArgmaxAligned = wArgmaxAligned_;
    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t computeSizeT2 = V_REG_SIZE / sizeof(TYPE_ARGMAX);
    uint16_t repeatimes = wFullBatchCount / computeSizeT2;
    uint16_t wRemain = wArgmaxActual - repeatimes * wProBatchSize * computeSizeT2;
    uint32_t wRemainBatchCount = wRemain / wProBatchSize;
    uint16_t wRemainTail = wRemain % wProBatchSize;
    uint32_t one = 1;
    uint32_t all = computeSizeT2;
    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highArgmaxOffset = highIdx * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dArgmaxActual; dIdx++) {
            uint32_t dArgmaxOffset = dIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t dIndexOffset = dIdx * hArgmaxActual * wArgmaxActual;
            uint32_t dOutputOffset = dIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                {
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(hOutput_ * wOutput_));
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegGrad;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);
                        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGrad, wProBatchSize);
                        for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                                uint32_t offset = (wBatchIdx + wRepeatIdx * computeSizeT2 * wProBatchSize +
                                                hIdx * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                                uint32_t indexOffset = (wBatchIdx + wRepeatIdx * computeSizeT2 * wProBatchSize +
                                                hIdx * wArgmaxActual + dIndexOffset + highIndexOffset);
                                AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                                AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                                DoSingleNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, all, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex,
                                    wOutputAligned, highOutputOffset, hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
                            }
                        }
                    }
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(hOutput_ * wOutput_));
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegGrad;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);
                        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGrad, wProBatchSize);
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        uint32_t offset = (wBatchIdx + repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned + dArgmaxOffset +
                                    highArgmaxOffset);
                        uint32_t indexOffset = (wBatchIdx + repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxActual + dIndexOffset +
                                    highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, wRemainBatchCount, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex,
                            wOutputAligned, highOutputOffset, hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            uint32_t offset = (wBatchIdx + wRemainBatchCount * wProBatchSize +
                                        repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                            uint32_t indexOffset = (wBatchIdx + wRemainBatchCount * wProBatchSize +
                                        repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxActual + dIndexOffset + highIndexOffset);
                            AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                            DoSingleNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, one, hwOutputConstReg, wOutputConstReg, curDIndex,
                                                                curHIndex, curWIndex, wOutputAligned, highOutputOffset, hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
                        }
                    }
                }
            }
        }
    }
}
template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::multipleLineHwProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    int64_t wOutput = wOutput_;
    int64_t hOutput = hOutput_;

    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;

    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);

    int64_t curDIndex = dAxisIndex_ * dOutputInner_;
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;

    int64_t wArgmaxAligned = wArgmaxAligned_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t dArgmaxActual = dArgmaxActual_;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;
    uint16_t hConcurrentCount = V_REG_SIZE / (wFullBatchCount * sizeof(TYPE_ARGMAX));

    uint16_t blockConcurrentCount = hFullBatchCount / hConcurrentCount;
    uint16_t hRemain = hArgmaxActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;

    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;

    uint32_t blockOne = 1 * hConcurrentCount;
    uint32_t remainBatchOne = 1 * hRemainBatchCount;
    uint32_t remainTailOne = 1;
    uint32_t maskBlock = wFullBatchCount * hConcurrentCount;
    uint32_t maskRemainBatch = wFullBatchCount * hRemainBatchCount;
    uint32_t maskRemainTail = wFullBatchCount;
    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highArgmaxOffset = highIdx * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dArgmaxActual; dIdx++) {
            uint32_t dArgmaxOffset = dIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t dIndexOffset = dIdx * hArgmaxActual * wArgmaxActual;
            uint32_t dOutputOffset = dIdx * hOutputActual * wOutputAligned;
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                    AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                    AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                }

                AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));

                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegGrad;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegGradOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;

                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxActual, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxActual);
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGrad, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGradOne, hProBatchSize, wArgmaxAligned);

                for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            TYPE_ARGMAX offset = (dArgmaxOffset + highArgmaxOffset + wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                        hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount);
                            TYPE_ARGMAX indexOffset = (dIndexOffset + highIndexOffset + wBatchIdx + hProBatchIdx * wArgmaxActual +
                                        hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount);
                            AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                            DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskBlock, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                        hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + dArgmaxOffset + highArgmaxOffset);
                            TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                        hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount + dIndexOffset + highIndexOffset);
                            AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                            DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, blockOne, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }
                    }
                }
            }

            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                    AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                    AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                }

                AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));

                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegGrad;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegGradOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;

                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxActual, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxActual);
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGrad, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGradOne, hProBatchSize, wArgmaxAligned);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        TYPE_ARGMAX offset =
                            (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + dArgmaxOffset+ highArgmaxOffset);
                        TYPE_ARGMAX indexOffset =
                            (wBatchIdx + hProBatchIdx * wArgmaxActual +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual + dIndexOffset+ highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskRemainBatch, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                            curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        TYPE_ARGMAX offset =
                            (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset =
                            (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual + dIndexOffset + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, remainBatchOne, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                            curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                }
            }
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                    AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                    AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                }
                AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));

                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegGrad;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegGradOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;

                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxActual, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxActual);
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGrad, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegGradOne, hProBatchSize, wArgmaxAligned);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        TYPE_ARGMAX offset =
                            (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                            hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset =
                            (wBatchIdx + hProBatchIdx * wArgmaxActual +
                            hRemainBatchCount * hProBatchSize * wArgmaxActual +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual + dIndexOffset + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskRemainTail, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                            curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        TYPE_ARGMAX offset =
                            (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                            hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset =
                            (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                            hRemainBatchCount * hProBatchSize * wArgmaxActual +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual + dIndexOffset + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, remainTailOne, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                            curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                }
            }
        }
    }    
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::multipleLineDhwProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr)
{
    int64_t wOutput = wOutput_;
    int64_t hOutput = hOutput_;
    int64_t dOutput = dOutput_;

    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;

    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);

    int64_t curDIndex = dAxisIndex_ * dOutputInner_;
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;

    int64_t wArgmaxAligned = wArgmaxAligned_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t dArgmaxActual = dArgmaxActual_;

    uint16_t dProBatchSize = curDProBatchSize_;
    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;
    uint32_t hwFullBatchCount = wFullBatchCount * hFullBatchCount;

    uint16_t hwConcurrentCount = V_REG_SIZE / (hwFullBatchCount * sizeof(TYPE_ARGMAX));

    uint16_t dFullBatchCount = dArgmaxActual / dProBatchSize;
    uint16_t dBlockConcurrentCount = dFullBatchCount / hwConcurrentCount;

    uint16_t dRemain = dArgmaxActual - dBlockConcurrentCount * hwConcurrentCount * dProBatchSize;

    uint16_t dRemainBatchCount = dRemain / dProBatchSize;
    uint16_t dRemainTail = dRemain - dRemainBatchCount * dProBatchSize;

    uint16_t hRemainTail = hArgmaxActual - hFullBatchCount * hProBatchSize;

    uint32_t mask0 = hwConcurrentCount * hwFullBatchCount;
    uint32_t mask1 = hwConcurrentCount * hFullBatchCount * 1;
    uint32_t mask2 = hwConcurrentCount * 1 * wFullBatchCount;
    uint32_t mask3 = hwConcurrentCount * 1 * 1;

    uint32_t mask4 = dRemainBatchCount * hwFullBatchCount;
    uint32_t mask5 = dRemainBatchCount * hFullBatchCount * 1;
    uint32_t mask6 = dRemainBatchCount * 1 * wFullBatchCount;
    uint32_t mask7 = dRemainBatchCount * 1 * 1;

    uint32_t mask8 = 1 * hwFullBatchCount;
    uint32_t mask9 = 1 * hFullBatchCount * 1;
    uint32_t mask10 = 1 * 1 * wFullBatchCount;
    uint32_t mask11 = 1 * 1 * 1;

    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highArgmaxOffset = highIdx * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highIdx * dArgmaxActual * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dBlockConcurrentCount; dIdx++) {
            for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradDw;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOneDw;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexDw;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOneDw;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;


                    AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGrad, dProBatchSize, hProBatchSize, wProBatchSize,
                                        hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOne, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                        hFullBatchCount, hArgmaxActual);
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                                1, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOneDw, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                        1, hArgmaxActual);
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize, wProBatchSize,
                                        hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxActual);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize, wArgmaxActual,
                                        hFullBatchCount, hArgmaxActual);
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                                1, hArgmaxActual, wFullBatchCount, wArgmaxActual);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize, wArgmaxActual,
                                        1, hArgmaxActual);
                    
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            TYPE_ARGMAX gradOffset = (wBatchIdx
                                        + hProBatchIdx * wArgmaxAligned
                                        + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount
                                        + highArgmaxOffset);

                                AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGrad, gradOffset, allMaskU32);
                            TYPE_ARGMAX indexOffset = (wBatchIdx
                                        + hProBatchIdx * wArgmaxActual
                                        + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual * hwConcurrentCount
                                        + highIndexOffset);

                                AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, indexOffset, allMaskU32);
                                DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask0, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }

                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            TYPE_ARGMAX gradOffset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                        + hProBatchIdx * wArgmaxAligned
                                        + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount
                                        + highArgmaxOffset);

                            AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradOne, gradOffset, allMaskU32);
                            TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                        + hProBatchIdx * wArgmaxActual
                                        + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual * hwConcurrentCount
                                        + highIndexOffset);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, indexOffset, allMaskU32);
                            DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask1, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }   
                    }
                }

                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradDw;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOneDw;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexDw;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOneDw;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;

                    AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGrad, dProBatchSize, hProBatchSize, wProBatchSize,
                                        hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOne, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                        hFullBatchCount, hArgmaxActual);
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                                1, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOneDw, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                        1, hArgmaxActual);
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize, wProBatchSize,
                                        hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxActual);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize, wArgmaxActual,
                                        hFullBatchCount, hArgmaxActual);
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                                1, hArgmaxActual, wFullBatchCount, wArgmaxActual);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize, wArgmaxActual,
                                        1, hArgmaxActual);

                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            TYPE_ARGMAX gradOffset = (wBatchIdx
                                        + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                        + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount
                                        + highArgmaxOffset);
                            AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradDw, gradOffset, allMaskU32);
                            TYPE_ARGMAX indexOffset = (wBatchIdx
                                        + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual
                                        + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual * hwConcurrentCount
                                        + highIndexOffset);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexDw, indexOffset, allMaskU32);
                            DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask2, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            TYPE_ARGMAX gradOffset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                        + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                        + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount
                                        + highArgmaxOffset);
                            AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradOneDw, gradOffset, allMaskU32);
                            TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                        + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual
                                        + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxActual * hwConcurrentCount
                                        + highIndexOffset);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOneDw, indexOffset, allMaskU32);
                            DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask3, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }
                    }
                }
            }
        }

        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                    AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                    AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                }
                AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexDw;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOneDw;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradDw;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOneDw;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize, wProBatchSize,
                                    hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxActual);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize, wArgmaxActual,
                            hFullBatchCount, hArgmaxActual);
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                    1, hArgmaxActual, wFullBatchCount, wArgmaxActual);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize, wArgmaxActual,
                                   1, hArgmaxActual);
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGrad, dProBatchSize, hProBatchSize, wProBatchSize,
                                    hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOne, dProBatchSize, hProBatchSize, wArgmaxAligned,
                            hFullBatchCount, hArgmaxActual);
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                    1, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOneDw, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                   1, hArgmaxActual);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        TYPE_ARGMAX offset = (wBatchIdx
                                    + hProBatchIdx * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset = (wBatchIdx
                                    + hProBatchIdx * wArgmaxActual
                                    + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxActual
                                    + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask4, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + hProBatchIdx * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + hProBatchIdx * wArgmaxActual
                                    + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxActual
                                    + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask5, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                }
            }
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                       AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                    AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                    AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                }
                AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexDw;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOneDw;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradDw;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOneDw;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize, wProBatchSize,
                                    hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxActual);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize, wArgmaxActual,
                            hFullBatchCount, hArgmaxActual);
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                    1, hArgmaxActual, wFullBatchCount, wArgmaxActual);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize, wArgmaxActual,
                                    1, hArgmaxActual);
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGrad, dProBatchSize, hProBatchSize, wProBatchSize,
                                    hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOne, dProBatchSize, hProBatchSize, wArgmaxAligned,
                            hFullBatchCount, hArgmaxActual);
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                    1, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOneDw, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                    1, hArgmaxActual);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        TYPE_ARGMAX offset = (wBatchIdx
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset = (wBatchIdx
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual
                                    + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxActual
                                    + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradDw, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexDw, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask6, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual
                                    + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxActual
                                    + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradOneDw, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOneDw, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask7, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                }
            }
        }
        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dRemainTail; dProBatchIdx++) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                    AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                    AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                }
                AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxActual, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hProBatchSize, wArgmaxActual);
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegGrad, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegGradOne, hProBatchSize, wArgmaxAligned);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        TYPE_ARGMAX offset = (wBatchIdx
                                    + hProBatchIdx * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset = (wBatchIdx
                                    + hProBatchIdx * wArgmaxActual
                                    + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxActual
                                    + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask8, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + hProBatchIdx * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + hProBatchIdx * wArgmaxActual
                                    + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxActual
                                    + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask9, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                }
            }
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                    AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                    AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                }
                AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput * hOutput));
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxActual, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hProBatchSize, wArgmaxActual);
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegGrad, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegGradOne, hProBatchSize, wArgmaxAligned);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        TYPE_ARGMAX offset = (wBatchIdx
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset = (wBatchIdx
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual
                                    + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxActual
                                    + highIndexOffset);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask10, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxActual
                                    + dProBatchIdx * hArgmaxActual * wArgmaxActual
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxActual
                                    + highIndexOffset);

                        AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                        DoSingleNCNchw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask11, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                }
            }
         }
    }
}

template <typename TYPE_ORIG_X, typename TYPE_ARGMAX, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void Pool3DGradSmallKernel<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>::multipleLineProcessVF2(
    __local_mem__ computeType* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr, __local_mem__ TYPE_ARGMAX* argmaxAddr, __local_mem__ uint32_t* helpAddr)
{
    int64_t wOutput = wOutput_;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    int32_t highOutputPlaneActual = wOutputAligned * hOutputActual * dOutputActual;
    int64_t highAxisActual = highAxisActual_;
    int64_t curDIndex = dAxisIndex_ * dOutputInner_;
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    int64_t wArgmaxAligned = wArgmaxAligned_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t dArgmaxActual = dArgmaxActual_;
    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint16_t dProBatchSize = curDProBatchSize_;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;
    uint32_t dFullBatchCount = dArgmaxActual / dProBatchSize;
    uint16_t dRemainTail = dArgmaxActual % dProBatchSize;
    uint32_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t hRemainTail = hArgmaxActual % hProBatchSize;
    uint32_t dhwFullBatchCount = wFullBatchCount * hFullBatchCount * dFullBatchCount;
    uint16_t highConcurrentCount = V_REG_SIZE / (dhwFullBatchCount * sizeof(TYPE_ARGMAX));
    uint16_t highBlockConcurrentCount = highAxisActual / highConcurrentCount;
    uint16_t highBlockRemainTail = highAxisActual - highBlockConcurrentCount * highConcurrentCount;
    int64_t depthStride = hArgmaxActual * wArgmaxAligned * dProBatchSize;
    int64_t highStride = dArgmaxActual * hArgmaxActual * wArgmaxAligned;
    int64_t depthStrideNonAlign = hArgmaxActual * wArgmaxActual * dProBatchSize;
    int64_t highStrideNonAlign = dArgmaxActual * hArgmaxActual * wArgmaxActual;
    uint32_t mask0 = highConcurrentCount * dFullBatchCount * hFullBatchCount * wFullBatchCount;
    uint32_t mask1 = highConcurrentCount * dFullBatchCount * hFullBatchCount;
    uint32_t mask2 = highConcurrentCount * dFullBatchCount * wFullBatchCount;
    uint32_t mask3 = highConcurrentCount * dFullBatchCount;
    uint32_t mask4 = highConcurrentCount * hFullBatchCount * wFullBatchCount;
    uint32_t mask5 = highConcurrentCount * hFullBatchCount;
    uint32_t mask6 = highConcurrentCount * wFullBatchCount;
    uint32_t mask7 = highConcurrentCount;
    uint32_t mask8 = highBlockRemainTail * dFullBatchCount * hFullBatchCount * wFullBatchCount;
    uint32_t mask9 = highBlockRemainTail * dFullBatchCount * hFullBatchCount;
    uint32_t mask10 = highBlockRemainTail * dFullBatchCount * wFullBatchCount;
    uint32_t mask11 = highBlockRemainTail * dFullBatchCount;
    uint32_t mask12 = highBlockRemainTail * hFullBatchCount * wFullBatchCount;
    uint32_t mask13 = highBlockRemainTail * hFullBatchCount;
    uint32_t mask14 = highBlockRemainTail * wFullBatchCount;
    uint32_t mask15 = highBlockRemainTail;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradDW;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradOneHD;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
        GenInitial4DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegGrad, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, hFullBatchCount, dFullBatchCount, depthStride, highStride);
        Gen4DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegGradOne, hProBatchSize, wArgmaxAligned,
                      hFullBatchCount, dFullBatchCount, depthStride, highStride);
        GenInitial4DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegGradDW, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, 1, dFullBatchCount, depthStride, highStride);
        Gen4DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegGradOneHD, hProBatchSize, wArgmaxAligned,
                      1, dFullBatchCount, depthStride, highStride);
        GenInitial3DHighIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGrad, highStride, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, hFullBatchCount);
        Gen3DHighIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegGradOne, highStride, hProBatchSize, wArgmaxAligned,
                      hFullBatchCount);
        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegGrad, wProBatchSize, dArgmaxActual * hArgmaxActual,
                            wArgmaxAligned, wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegGradOne, dArgmaxActual * hArgmaxActual, wArgmaxAligned);
        AscendC::MicroAPI::MaskReg allMask =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(helpAddr, initial4DRegGrad, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initial4DRegGradOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t), initial3DRegGrad, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t), initial3DRegGradOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_FOUR * V_REG_SIZE / sizeof(uint32_t), initial2DRegGrad, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_FIVE * V_REG_SIZE / sizeof(uint32_t), initial2DRegGradOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_SIX * V_REG_SIZE / sizeof(uint32_t), initial4DRegGradDW, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_SEVEN * V_REG_SIZE / sizeof(uint32_t), initial4DRegGradOneHD, allMask);
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg allMask =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexDW;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOneHD;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        GenInitial4DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndex, wProBatchSize, hProBatchSize,
                            wArgmaxActual, wFullBatchCount, hFullBatchCount, dFullBatchCount, depthStrideNonAlign, highStrideNonAlign);
        Gen4DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexOne, hProBatchSize, wArgmaxActual,
                      hFullBatchCount, dFullBatchCount, depthStrideNonAlign, highStrideNonAlign);
        GenInitial4DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexDW, wProBatchSize, hProBatchSize,
                            wArgmaxActual, wFullBatchCount, 1, dFullBatchCount, depthStrideNonAlign, highStrideNonAlign);
        Gen4DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexOneHD, hProBatchSize, wArgmaxActual,
                      1, dFullBatchCount, depthStrideNonAlign, highStrideNonAlign);
        GenInitial3DHighIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, highStrideNonAlign, wProBatchSize, hProBatchSize,
                            wArgmaxActual, wFullBatchCount, hFullBatchCount);
        Gen3DHighIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, highStrideNonAlign, hProBatchSize, wArgmaxActual,
                      hFullBatchCount);
        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, dArgmaxActual * hArgmaxActual,
                            wArgmaxActual, wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, dArgmaxActual * hArgmaxActual, wArgmaxActual);
        AscendC::MicroAPI::DataCopy(helpAddr + 8 * V_REG_SIZE / sizeof(uint32_t), initial4DRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + 9 * V_REG_SIZE / sizeof(uint32_t), initial4DRegIndexOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + 10 * V_REG_SIZE / sizeof(uint32_t), initial3DRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + 11 * V_REG_SIZE / sizeof(uint32_t), initial3DRegIndexOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + 12 * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + 13 * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndexOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + 14 * V_REG_SIZE / sizeof(uint32_t), initial4DRegIndexDW, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + 15 * V_REG_SIZE / sizeof(uint32_t), initial4DRegIndexOneHD, allMask);
    }
    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    TYPE_ARGMAX offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    TYPE_ARGMAX indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual + dProBatchIdx * hArgmaxActual * wArgmaxActual + highIndexOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGrad;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial4DRegGrad, helpAddr);
                        AscendC::MicroAPI::DataCopy(initial4DRegIndex, helpAddr + 8 * V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndex, indexOffset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial4DRegGrad, offset, allMaskU32);
                        DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask0, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * hFullBatchCount * wFullBatchCount, helpAddr);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual + dProBatchIdx * hArgmaxActual * wArgmaxActual + highIndexOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGrad;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial4DRegGradOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initial4DRegIndexOne, helpAddr + 9 * V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial4DRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOne, indexOffset, allMaskU32);
                        DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask1, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * hFullBatchCount, helpAddr);
                    }
                }
            }
        }
    }
    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
            for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    TYPE_ARGMAX offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    TYPE_ARGMAX indexOffset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxActual + dProBatchIdx * hArgmaxActual * wArgmaxActual + highIndexOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexDW;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradDW;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial4DRegGradDW, helpAddr + INDEX_SIX * V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initial4DRegIndexDW, helpAddr + 14 * V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial4DRegGradDW, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexDW, indexOffset, allMaskU32);
                        DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask2, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * wFullBatchCount, helpAddr);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    TYPE_ARGMAX offset = wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset;
                    TYPE_ARGMAX indexOffset = wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxActual + dProBatchIdx * hArgmaxActual * wArgmaxActual + highIndexOffset;
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOneHD;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradOneHD;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial4DRegGradOneHD, helpAddr + INDEX_SEVEN * V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initial4DRegIndexOneHD, helpAddr + 15 * V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial4DRegGradOneHD, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOneHD, indexOffset, allMaskU32);
                        DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask3, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount, helpAddr);
                    }
                }
            }
        }
    }

    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    TYPE_ARGMAX offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    TYPE_ARGMAX indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxActual + highIndexOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial3DRegGrad, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr + 10 * V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, indexOffset, allMaskU32);
                        DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask4, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, hFullBatchCount * wFullBatchCount, helpAddr);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxActual + highIndexOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial3DRegGradOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + 11 * V_REG_SIZE / sizeof(uint32_t));
                        
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, indexOffset, allMaskU32);
                        DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask5, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, hFullBatchCount, helpAddr);
                    }
                }
            }
        }
    }
    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
            for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    TYPE_ARGMAX offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    TYPE_ARGMAX indexOffset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxActual + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxActual + highIndexOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial2DRegGrad, helpAddr + INDEX_FOUR * V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + 12 * V_REG_SIZE / sizeof(uint32_t));

                        AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGrad, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, indexOffset, allMaskU32);
                        DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask6, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                            hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, wFullBatchCount, helpAddr);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + (dProBatchSize * dFullBatchCount + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxActual + (dProBatchSize * dFullBatchCount + dTailIdx) * hArgmaxActual * wArgmaxActual + highIndexOffset);
                   __VEC_SCOPE__
                   {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial2DRegGradOne, helpAddr + INDEX_FIVE * V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + 13 * V_REG_SIZE / sizeof(uint32_t));
                            
                        AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                        DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask7, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                        hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, 1, helpAddr);
                    }
                }
            }
        }
    }

    uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
    uint32_t highIndexOffset = highBlockConcurrentCount * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxActual;
    uint32_t highOutputOffset= highBlockConcurrentCount * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
    for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                TYPE_ARGMAX offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                TYPE_ARGMAX indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual + dProBatchIdx * hArgmaxActual * wArgmaxActual + highIndexOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }

                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial4DRegGrad, helpAddr);
                    AscendC::MicroAPI::DataCopy(initial4DRegIndex, helpAddr + 8 * V_REG_SIZE / sizeof(uint32_t));

                    AscendC::MicroAPI::Adds(parallelRegGrad, initial4DRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndex, indexOffset, allMaskU32);
                    DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask8, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                            hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * hFullBatchCount * wFullBatchCount, helpAddr);
                }
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual + dProBatchIdx * hArgmaxActual * wArgmaxActual + highIndexOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial4DRegGradOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial4DRegIndexOne, helpAddr + 9 * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial4DRegGradOne, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOne, indexOffset, allMaskU32);
                    DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask9, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                    hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * hFullBatchCount, helpAddr);
                }
            }
        }
    }
    for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                TYPE_ARGMAX offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                TYPE_ARGMAX indexOffset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxActual + dProBatchIdx * hArgmaxActual * wArgmaxActual + highIndexOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexDW;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradDW;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial4DRegGradDW, helpAddr + INDEX_SIX * V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial4DRegIndexDW, helpAddr + 14 * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial4DRegGradDW, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexDW, indexOffset, allMaskU32);
                    DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask10, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                        hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * wFullBatchCount, helpAddr);
                }
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxActual + dProBatchIdx * hArgmaxActual * wArgmaxActual + highIndexOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOneHD;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegGradOneHD;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial4DRegGradOneHD, helpAddr + INDEX_SEVEN * V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial4DRegIndexOneHD, helpAddr + 15 * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial4DRegGradOneHD, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOneHD, indexOffset, allMaskU32);
                    DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask11, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                            hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount, helpAddr);
                }
            }
        }
    }

    for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                TYPE_ARGMAX offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                TYPE_ARGMAX indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxActual + highIndexOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial3DRegGrad, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr + 10 * V_REG_SIZE / sizeof(uint32_t));

                    AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, indexOffset, allMaskU32);
                    DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask12, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                        hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, hFullBatchCount * wFullBatchCount, helpAddr);
                }
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxActual + highIndexOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegGradOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial3DRegGradOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + 11 * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial3DRegGradOne, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, indexOffset, allMaskU32);
                    DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask13, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                           hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, hFullBatchCount, helpAddr);
                }
            }
        }
    }

    for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
        for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                TYPE_ARGMAX offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                TYPE_ARGMAX indexOffset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxActual + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxActual + highIndexOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial2DRegGrad, helpAddr + INDEX_FOUR * V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + 12 * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGrad, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, indexOffset, allMaskU32);
                    DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask14, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                        hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, wFullBatchCount, helpAddr);
                }
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                TYPE_ARGMAX offset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + (dProBatchSize * dFullBatchCount + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                TYPE_ARGMAX indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxActual + (dProBatchSize * dFullBatchCount + dTailIdx) * hArgmaxActual * wArgmaxActual + highIndexOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, TYPE_ARGMAX(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGrad;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegGradOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegGrad;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial2DRegGradOne, helpAddr + INDEX_FIVE * V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + 13 * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                    DoMulNCNcdhw<TYPE_ORIG_X, TYPE_ARGMAX, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask15, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                            hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, 1, helpAddr);
                }
            }
        }
    }
}
}  // namespace MaxPool3DGradWithArgmaxNCDHWNameSpace
#endif  // MAX_POOL_GRAD_WITH_ARGMAX_SIMD_IMPL_H_
