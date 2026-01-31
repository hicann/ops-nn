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
 * \file max_pool_grad_with_argmax_simd_impl.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_SIMD_IMPL_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_SIMD_IMPL_H_

#include "max_pool3d_grad_with_argmax_simd.h"
namespace MaxPool3DGradWithArgmaxNCDHWNameSpace
{

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, T3, IS_CHECK_RANGE>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}
template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, T3, IS_CHECK_RANGE>::Compute()
{
    uint32_t calCount = outputBufferSize_ / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);
    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();
    LocalTensor<T2> argmaxLocal = argmaxQue_.DeQue<T2>();
    //UB
    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
    __local_mem__ T2* argmaxAddr = (__local_mem__ T2*)argmaxLocal.GetPhyAddr();
    uint32_t wConcurrentCount = wArgmaxActual_ / curWProBatchSize_;
    uint32_t hConcurrentCount = hArgmaxActual_ / curHProBatchSize_;
    uint32_t dConcurrentCount = dArgmaxActual_ / curDProBatchSize_;
    if (wConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        singleLineProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        multipleLineHwProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * dConcurrentCount * DOUBLE * sizeof(T2) > V_REG_SIZE) {
        multipleLineDhwProcessVF(yAddr, gradAddr, argmaxAddr);
    } else {
        LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
        __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();
        multipleLineProcessVF2(yAddr, gradAddr, argmaxAddr, helpAddr);
    }
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }

    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
    argmaxQue_.FreeTensor(argmaxLocal);
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, T3, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, T3, IS_CHECK_RANGE>::ProcessPerLoop()
{
    if (hArgmaxActual_ <= 0 || wArgmaxActual_ <= 0 || dArgmaxActual_ <= 0) {
        ProcessNoArgmaxBlock();
        return;
    }
    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, T3, IS_CHECK_RANGE>::singleLineProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
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
    uint16_t computeSizeT2 = V_REG_SIZE / sizeof(T2);
    uint16_t repeatimes = wFullBatchCount / computeSizeT2;
    uint16_t wRemain = wArgmaxActual - repeatimes * wProBatchSize * computeSizeT2;
    uint32_t wRemainBatchCount = wRemain / wProBatchSize;
    uint16_t wRemainTail = wRemain % wProBatchSize;
    uint32_t one = 1;
    uint32_t all = computeSizeT2;
    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highArgmaxOffset = highIdx * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dArgmaxActual; dIdx++) {
            uint32_t dArgmaxOffset = dIdx * hArgmaxActual * wArgmaxAligned;
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
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);
                        for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                                uint32_t offset = (wBatchIdx + wRepeatIdx * computeSizeT2 * wProBatchSize +
                                                hIdx * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                                AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                                DoSingleNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, all, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex,
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
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        GenInitial1DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        uint32_t offset = (wBatchIdx + repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned + dArgmaxOffset +
                                    highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, wRemainBatchCount, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex,
                            wOutputAligned, highOutputOffset, hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            uint32_t offset = (wBatchIdx + wRemainBatchCount * wProBatchSize +
                                        repeatimes * computeSizeT2 * wProBatchSize + hIdx * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                            DoSingleNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, one, hwOutputConstReg, wOutputConstReg, curDIndex,
                                                                curHIndex, curWIndex, wOutputAligned, highOutputOffset, hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg);
                        }
                    }
                }
            }
        }
    }
}
template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, T3, IS_CHECK_RANGE>::multipleLineHwProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
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
    uint16_t hConcurrentCount = V_REG_SIZE / (wFullBatchCount * sizeof(T2));

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
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dArgmaxActual; dIdx++) {
            uint32_t dArgmaxOffset = dIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t dOutputOffset = dIdx * hOutputActual * wOutputAligned;
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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

                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                    DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxAligned);

                for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T2 offset = (dArgmaxOffset + highArgmaxOffset + wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                        hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                            DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskBlock, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                        hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + dArgmaxOffset + highArgmaxOffset);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                            DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                yAddr, gradAddr, argmaxAddr, parallelRegIndex, blockOne, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
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
                    AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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

                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxAligned);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset =
                            (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + dArgmaxOffset+ highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskRemainBatch, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                            curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset =
                            (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, remainBatchOne, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
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
                    AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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

                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wArgmaxAligned);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset =
                            (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                            hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, maskRemainTail, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                            curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset =
                            (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                            hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                            blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned + dArgmaxOffset + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, remainTailOne, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                            curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                }
            }
        }
    }    
}

template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, T3, IS_CHECK_RANGE>::multipleLineDhwProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr)
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

    uint16_t hwConcurrentCount = V_REG_SIZE / (hwFullBatchCount * sizeof(T2));

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
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize, wProBatchSize,
                                        hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                        hFullBatchCount, hArgmaxActual);
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                                1, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                        1, hArgmaxActual);
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T2 offset = (wBatchIdx
                                        + hProBatchIdx * wArgmaxAligned
                                        + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount
                                        + highArgmaxOffset);

                                AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                                DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }

                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                        + hProBatchIdx * wArgmaxAligned
                                        + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount
                                        + highArgmaxOffset);

                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                            DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
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
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize, wProBatchSize,
                                        hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                hFullBatchCount, hArgmaxActual);
                    GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                        1, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                    Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                1, hArgmaxActual);
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T2 offset = (wBatchIdx
                                        + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                        + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount
                                        + highArgmaxOffset);

                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexDw, offset, allMaskU32);
                            DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                        + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                        + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                        + dIdx * dProBatchSize * hArgmaxActual * wArgmaxAligned * hwConcurrentCount
                                        + highArgmaxOffset);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOneDw, offset, allMaskU32);
                            DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask3, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
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
                    AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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
                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize, wProBatchSize,
                                    hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize, wArgmaxAligned,
                            hFullBatchCount, hArgmaxActual);
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                    1, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                   1, hArgmaxActual);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx
                                    + hProBatchIdx * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);

                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask4, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + hProBatchIdx * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask5, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
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
                       AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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
                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, dProBatchSize, hProBatchSize, wProBatchSize,
                                    hFullBatchCount, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, dProBatchSize, hProBatchSize, wArgmaxAligned,
                            hFullBatchCount, hArgmaxActual);
                GenInitial3DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexDw, dProBatchSize, hProBatchSize, wProBatchSize,
                                    1, hArgmaxActual, wFullBatchCount, wArgmaxAligned);
                Gen3DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOneDw, dProBatchSize, hProBatchSize, wArgmaxAligned,
                                    1, hArgmaxActual);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);

                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexDw, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask6, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);

                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOneDw, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask7, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
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
                    AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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
                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hProBatchSize, wArgmaxAligned);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx
                                    + hProBatchIdx * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask8, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + hProBatchIdx * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);

                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask9, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
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
                    AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
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
                AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                DhwGenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hProBatchSize,
                                    wArgmaxAligned, wFullBatchCount);
                DhwGen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, hProBatchSize, wArgmaxAligned);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T2 offset = (wBatchIdx
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask10, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                        }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount
                                    + (hProBatchIdx + hFullBatchCount * hProBatchSize) * wArgmaxAligned
                                    + dProBatchIdx * hArgmaxActual * wArgmaxAligned
                                    + (dRemainBatchCount + dBlockConcurrentCount * hwConcurrentCount) * dProBatchSize * hArgmaxActual * wArgmaxAligned
                                    + highArgmaxOffset);

                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                        DoSingleNCNchw<T1, T2, T3, IS_CHECK_RANGE>(
                                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask11, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex,
                                    curWIndex, hOutputActual, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, dMaxReg);
                    }
                }
            }
         }
    }
}
template <typename T1, typename T2, typename T3, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradWithArgmaxNCDHWKernel<T1, T2, T3, IS_CHECK_RANGE>::multipleLineProcessVF2(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr, __local_mem__ uint32_t* helpAddr)
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
    uint16_t highConcurrentCount = V_REG_SIZE / (dhwFullBatchCount * sizeof(T2));
    uint16_t highBlockConcurrentCount = highAxisActual / highConcurrentCount;
    uint16_t highBlockRemainTail = highAxisActual - highBlockConcurrentCount * highConcurrentCount;
    int64_t depthStride = hArgmaxActual * wArgmaxAligned * dProBatchSize;
    int64_t highStride = dArgmaxActual * hArgmaxActual * wArgmaxAligned;
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
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexDW;
        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOneHD;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
        GenInitial4DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndex, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, hFullBatchCount, dFullBatchCount, depthStride, highStride);
        Gen4DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexOne, hProBatchSize, wArgmaxAligned,
                      hFullBatchCount, dFullBatchCount, depthStride, highStride);
        GenInitial4DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexDW, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, 1, dFullBatchCount, depthStride, highStride);
        Gen4DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial4DRegIndexOneHD, hProBatchSize, wArgmaxAligned,
                      1, dFullBatchCount, depthStride, highStride);
        GenInitial3DHighIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndex, highStride, wProBatchSize, hProBatchSize,
                            wArgmaxAligned, wFullBatchCount, hFullBatchCount);
        Gen3DHighIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial3DRegIndexOne, highStride, hProBatchSize, wArgmaxAligned,
                      hFullBatchCount);
        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, dArgmaxActual * hArgmaxActual,
                            wArgmaxAligned, wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initial2DRegIndexOne, dArgmaxActual * hArgmaxActual, wArgmaxAligned);
        AscendC::MicroAPI::MaskReg allMask =
        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(helpAddr, initial4DRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initial4DRegIndexOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t), initial3DRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t), initial3DRegIndexOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_FOUR * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_FIVE * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndexOne, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_SIX * V_REG_SIZE / sizeof(uint32_t), initial4DRegIndexDW, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_SEVEN * V_REG_SIZE / sizeof(uint32_t), initial4DRegIndexOneHD, allMask);
    }
    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial4DRegIndex, helpAddr);
                    
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndex, offset, allMaskU32);
                        DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask0, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * hFullBatchCount * wFullBatchCount, helpAddr);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial4DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOne, offset, allMaskU32);
                        DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask1, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * hFullBatchCount, helpAddr);
                    }
                }
            }
        }
    }
    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
            for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexDW;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial4DRegIndexDW, helpAddr + INDEX_SIX * V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexDW, offset, allMaskU32);
                        DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask2, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * wFullBatchCount, helpAddr);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset;
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOneHD;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial4DRegIndexOneHD, helpAddr + INDEX_SEVEN * V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOneHD, offset, allMaskU32);
                        DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask3, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount, helpAddr);
                    }
                }
            }
        }
    }

    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
                    
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                        DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask4, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, hFullBatchCount * wFullBatchCount, helpAddr);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));
                        
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                        DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask5, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, hFullBatchCount, helpAddr);
                    }
                }
            }
        }
    }
    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
            for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T2 offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + INDEX_FOUR * V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                        DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask6, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                            hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, wFullBatchCount, helpAddr);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + (dProBatchSize * dFullBatchCount + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                   __VEC_SCOPE__
                   {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                            AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                        AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                        AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                        AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                        AscendC::MicroAPI::MaskReg allMaskU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_FIVE * V_REG_SIZE / sizeof(uint32_t));
                            
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                        DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask7, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                                        hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, 1, helpAddr);
                    }
                }
            }
        }
    }

    uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * dArgmaxActual * hArgmaxActual * wArgmaxAligned;
    uint32_t highOutputOffset= highBlockConcurrentCount * highConcurrentCount * dOutputActual * hOutputActual * wOutputAligned;
    for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }

                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial4DRegIndex, helpAddr);
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndex, offset, allMaskU32);
                    DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask8, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                            hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * hFullBatchCount * wFullBatchCount, helpAddr);
                }
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial4DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOne, offset, allMaskU32);
                    DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask9, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                    hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * hFullBatchCount, helpAddr);
                }
            }
        }
    }
    for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; dProBatchIdx++) {
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexDW;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial4DRegIndexDW, helpAddr + INDEX_SIX * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexDW, offset, allMaskU32);
                    DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask10, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                        hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount * wFullBatchCount, helpAddr);
                }
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned + dProBatchIdx * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial4DRegIndexOneHD;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial4DRegIndexOneHD, helpAddr + INDEX_SEVEN * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial4DRegIndexOneHD, offset, allMaskU32);
                    DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask11, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                            hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, dFullBatchCount, helpAddr);
                }
            }
        }
    }

    for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                    DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask12, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                        hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, hFullBatchCount * wFullBatchCount, helpAddr);
                }
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                    DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask13, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                           hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, hFullBatchCount, helpAddr);
                }
            }
        }
    }

    for (uint16_t dTailIdx = 0; dTailIdx < dRemainTail; dTailIdx++) {
        for (uint16_t hTailIdx = 0; hTailIdx < hRemainTail; hTailIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T2 offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + (dFullBatchCount * dProBatchSize + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + INDEX_FOUR * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                    DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask14, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                        hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, wFullBatchCount, helpAddr);
                }
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T2 offset = (wBatchIdx + wProBatchSize * wFullBatchCount + (hProBatchSize * hFullBatchCount + hTailIdx) * wArgmaxAligned + (dProBatchSize * dFullBatchCount + dTailIdx) * hArgmaxActual * wArgmaxAligned + highArgmaxOffset);
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, T2(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, int32_t(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, int32_t(hOutputActual));
                        AscendC::MicroAPI::Duplicate(dMaxReg, int32_t(dOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<T3> wOutputConstReg;
                    AscendC::MicroAPI::Duplicate(wOutputConstReg, T3(wOutput));
                    AscendC::MicroAPI::RegTensor<T3> hwOutputConstReg;
                    AscendC::MicroAPI::Duplicate(hwOutputConstReg, T3(wOutput_ * hOutput_));
                    AscendC::MicroAPI::MaskReg allMaskU32 =
                    AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_FIVE * V_REG_SIZE / sizeof(uint32_t));
                
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                    DoMulNCNcdhw<T1, T2, T3, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, parallelRegIndex, mask15, hwOutputConstReg, wOutputConstReg, curDIndex, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                            hOutputActual, zeroConstReg, dMaxReg, hMaxReg, wMaxReg, highOutputPlaneActual, 1, helpAddr);
                }
            }
        }
    }
}
}  // namespace MaxPool3DGradWithArgmaxNCDHWNameSpace
#endif  // MAX_POOL_GRAD_WITH_ARGMAX_SIMD_IMPL_H_