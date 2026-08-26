/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_v2_grad_nchw_kernel.h
 * \brief
 */

#ifndef AVG_POOL_V2_GRAD_NCHW_KERNEL_H_
#define AVG_POOL_V2_GRAD_NCHW_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "avg_pool_v2_grad_base.h"
#include "avg_pool_v2_grad_tiling_data.h"

namespace AvgPoolV2GradNCHWNameSpace {
using namespace AscendC;
using namespace AvgPoolV2Grad;

constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchw(__ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr,
                                      Reg::RegTensor<uint32_t>& parallelRegIndex, uint32_t gradMaskCount,
                                      int32_t wOutputAligned, int32_t highOutputOffset,
                                      Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& wMaxReg,
                                      Reg::RegTensor<int32_t>& hMaxReg, uint16_t kH, uint16_t kW,
                                      Reg::RegTensor<int32_t>& divisorReg, Reg::RegTensor<int32_t>& wIndexReg,
                                      Reg::RegTensor<int32_t>& hIndexReg, Reg::RegTensor<int32_t>& highIdxReg)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> scatterStartIdxReg;
    AscendC::Reg::RegTensor<int32_t> scatterIndexReg;

    uint32_t maskT1 = gradMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    uint32_t maskI32 = gradMaskCount;
    AscendC::Reg::MaskReg pregI32 = AscendC::Reg::UpdateMask<int32_t>(maskI32);
    GetConCurrentInput<T1>(gradReg, gradAddr, parallelRegIndex, pregT1);

    AscendC::Reg::Muls(scatterStartIdxReg, hIndexReg, wOutputAligned, pregI32);
    AscendC::Reg::Add(scatterStartIdxReg, scatterStartIdxReg, wIndexReg, pregI32);
    AscendC::Reg::Add(scatterStartIdxReg, scatterStartIdxReg, highIdxReg, pregI32);
    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        int32_t hKernelOffset = hIdx * wOutputAligned;

        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            uint32_t gradMask = gradMaskCount;
            AscendC::Reg::MaskReg pregRes = AscendC::Reg::UpdateMask<int32_t>(gradMask);

            int32_t scatterIndexOffsetTotal = highOutputOffset + hKernelOffset + wIdx;
            AscendC::Reg::Adds(scatterIndexReg, scatterStartIdxReg, scatterIndexOffsetTotal, pregRes);

            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::RegTensor<int32_t> wCurIndexReg;
                AscendC::Reg::RegTensor<int32_t> hCurIndexReg;
                AscendC::Reg::Adds(wCurIndexReg, wIndexReg, static_cast<int32_t>(wIdx), pregRes);
                AscendC::Reg::Adds(hCurIndexReg, hIndexReg, static_cast<int32_t>(hIdx), pregRes);
                FilterMask(pregRes, hCurIndexReg, wCurIndexReg, zeroConstReg, wMaxReg, hMaxReg);
            }

            GradientAcc(yAddr, gradReg, scatterIndexReg, divisorReg, pregRes);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoSingleNCNchwForMergeW(__ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr,
                                               Reg::RegTensor<uint32_t>& parallelRegIndex, uint32_t gradMaskCount,
                                               int32_t wOutputAligned, int32_t highOutputOffset,
                                               Reg::RegTensor<int32_t>& zeroConstReg, Reg::RegTensor<int32_t>& wMaxReg,
                                               uint16_t kW, Reg::RegTensor<int32_t>& divisorReg,
                                               Reg::RegTensor<int32_t>& wIndexReg, int32_t hIndex, int32_t kHStart,
                                               int32_t kHEnd)
{
    AscendC::Reg::RegTensor<computeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> scatterStartIdxReg;
    AscendC::Reg::RegTensor<int32_t> scatterIndexReg;

    uint32_t maskT1 = gradMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    uint32_t maskI32 = gradMaskCount;
    AscendC::Reg::MaskReg pregI32 = AscendC::Reg::UpdateMask<int32_t>(maskI32);
    GetConCurrentInput<T1>(gradReg, gradAddr, parallelRegIndex, pregT1);

    int32_t scatterStartIdx = hIndex * wOutputAligned;
    AscendC::Reg::Adds(scatterStartIdxReg, wIndexReg, scatterStartIdx, pregI32);
    for (uint16_t hIdx = kHStart; hIdx < kHEnd; hIdx++) {
        int32_t hKernelOffset = hIdx * wOutputAligned;

        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            uint32_t gradMask = gradMaskCount;
            AscendC::Reg::MaskReg pregRes = AscendC::Reg::UpdateMask<int32_t>(gradMask);

            int32_t scatterIndexOffsetTotal = highOutputOffset + hKernelOffset + wIdx;
            AscendC::Reg::Adds(scatterIndexReg, scatterStartIdxReg, scatterIndexOffsetTotal, pregRes);

            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::RegTensor<int32_t> wCurIndexReg;
                AscendC::Reg::Adds(wCurIndexReg, wIndexReg, static_cast<int32_t>(wIdx), pregRes);
                FilterMaskForMergeW(pregRes, wCurIndexReg, zeroConstReg, wMaxReg);
            }

            GradientAcc(yAddr, gradReg, scatterIndexReg, divisorReg, pregRes);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T, const Reg::RegTrait& Trait = Reg::RegTraitNumOne>
__aicore__ inline void ComputeOutRegStart(Reg::RegTensor<T, Trait>& outRegStart,
                                          Reg::RegTensor<T, Trait>& initialRegIndex, T wGradOffset, T strideW)
{
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL, Trait>();
    // grad绝对索引
    AscendC::Reg::Adds(outRegStart, initialRegIndex, wGradOffset, allMask);
    // 输出绝对索引
    AscendC::Reg::Muls(outRegStart, outRegStart, strideW, allMask);
}

template <typename T, const Reg::RegTrait& Trait = Reg::RegTraitNumOne>
__aicore__ inline void ComputeOutWHIndex(Reg::RegTensor<int32_t>& wIndexReg, Reg::RegTensor<int32_t>& hIndexReg,
                                         Reg::RegTensor<T, Trait>& outWStart, Reg::RegTensor<T, Trait>& outHStart,
                                         T curWIndex, T curHIndex, uint16_t padH, uint16_t padW, uint32_t count)
{
    AscendC::Reg::RegTensor<T, Trait> wIndexRegTwo;
    AscendC::Reg::RegTensor<T, Trait> hIndexRegTwo;
    uint32_t numT = count;
    AscendC::Reg::MaskReg maskT = AscendC::Reg::UpdateMask<T, Trait>(numT);
    AscendC::Reg::Adds(wIndexRegTwo, outWStart, static_cast<T>(-curWIndex - padW), maskT);
    AscendC::Reg::Adds(hIndexRegTwo, outHStart, static_cast<T>(-curHIndex - padH), maskT);
    wIndexReg = (AscendC::Reg::RegTensor<int32_t>&)wIndexRegTwo.reg[0];
    hIndexReg = (AscendC::Reg::RegTensor<int32_t>&)hIndexRegTwo.reg[0];
}

template <typename T, const Reg::RegTrait& Trait = Reg::RegTraitNumOne>
__aicore__ inline void ComputeOutWIndex(Reg::RegTensor<int32_t>& wIndexReg, Reg::RegTensor<T, Trait>& outWStart,
                                        T curWIndex, uint16_t padW, uint32_t count)
{
    AscendC::Reg::RegTensor<T, Trait> wIndexRegTwo;
    uint32_t numT = count;
    AscendC::Reg::MaskReg maskT = AscendC::Reg::UpdateMask<T, Trait>(numT);
    AscendC::Reg::Adds(wIndexRegTwo, outWStart, static_cast<T>(-curWIndex - padW), maskT);
    wIndexReg = (AscendC::Reg::RegTensor<int32_t>&)wIndexRegTwo.reg[0];
}

template <typename T, const Reg::RegTrait& Trait = Reg::RegTraitNumOne>
__aicore__ inline void GenInitial1DIndices(Reg::RegTensor<T, Trait>& indexReg, int64_t colGenRate)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL, Trait>();
    AscendC::Reg::Muls(indexReg, indexReg, static_cast<T>(colGenRate), preg);
}

template <typename T>
__aicore__ inline void GenInitial2DIndices(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::Duplicate(constReg, static_cast<T>(fullBatchColNum));
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);

    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, static_cast<T>(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, static_cast<T>(colGenRate), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rowGenRate * colNumAligned), preg);
    AscendC::Reg::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void Gen2DIndexOne(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Muls(indexReg, indexReg, static_cast<T>(rowGenRate * colNumAligned), preg);
}

template <typename T>
__aicore__ inline void GenInitial3DIndices(Reg::RegTensor<T>& indexReg, int64_t colGenRate, int64_t rowGenRate,
                                           int64_t colNumAligned, int64_t fullBatchColNum, int64_t fullBatchRowNum,
                                           int64_t rowNumCount)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(constReg, static_cast<T>(fullBatchColNum * fullBatchRowNum));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, static_cast<T>(fullBatchColNum * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rowNumCount * colNumAligned), preg);

    AscendC::Reg::Duplicate(constReg, static_cast<T>(fullBatchColNum));
    AscendC::Reg::Div(segmentScalarReg2, segmentIncReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentScalarReg2, static_cast<T>(fullBatchColNum), preg);
    AscendC::Reg::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::Reg::Muls(segmentIncReg2, segmentIncReg2, colGenRate, preg);

    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, static_cast<T>(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOne(Reg::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNumAligned,
                                     int64_t fullBatchRowNum, int64_t rowNumCount)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentIncReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentIncReg2;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(constReg, static_cast<T>(1 * fullBatchRowNum));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(segmentIncReg, segmentScalarReg, static_cast<T>(1 * fullBatchRowNum), preg);
    AscendC::Reg::Sub(segmentIncReg, indexReg, segmentIncReg, preg);

    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rowNumCount * colNumAligned), preg);

    AscendC::Reg::Muls(segmentIncReg, segmentIncReg, static_cast<T>(rowGenRate * colNumAligned), preg);

    AscendC::Reg::Add(indexReg, segmentIncReg, segmentScalarReg, preg);
}

__aicore__ inline void GenIndicesToUb(__ubuf__ uint32_t* helpAddr, int64_t wProBatchSize, int64_t hProBatchSize,
                                      int64_t wGradAligned, int64_t wFullBatchCount, int64_t hFullBatchCount,
                                      int64_t hGradActual)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;

        GenInitial3DIndices((AscendC::Reg::RegTensor<int32_t>&)initial3DRegIndex, wProBatchSize, hProBatchSize,
                            wGradAligned, wFullBatchCount, hFullBatchCount, hGradActual);
        Gen3DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initial3DRegIndexOne, hProBatchSize, wGradAligned,
                      hFullBatchCount, hGradActual);

        GenInitial2DIndices((AscendC::Reg::RegTensor<int32_t>&)initial2DRegIndex, wProBatchSize, hGradActual,
                            wGradAligned, wFullBatchCount);
        Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initial2DRegIndexOne, hGradActual, wGradAligned);

        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::StoreAlign(helpAddr, initial3DRegIndex, allMask);
        AscendC::Reg::StoreAlign(helpAddr + V_REG_SIZE / sizeof(uint32_t), initial3DRegIndexOne, allMask);
        AscendC::Reg::StoreAlign(helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndex, allMask);
        AscendC::Reg::StoreAlign(helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndexOne, allMask);
    }
}

template <typename T3, const Reg::RegTrait& Trait = Reg::RegTraitNumOne>
__aicore__ inline void GenIndicesToUbForT3(__ubuf__ T3* helpAddrT3, T3 whFullBatchCount, T3 wFullBatchCount,
                                           T3 wProBatchSize, T3 hProBatchSize, T3 hFullBatchCount)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdx;
        AscendC::Reg::RegTensor<T3, Trait> initial3DRegWIdx;
        AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdxOne;
        AscendC::Reg::RegTensor<T3, Trait> initial2DRegWIdx;

        GenGatterIndex3D<T3, Trait>(initial3DRegWIdx, 0, whFullBatchCount, 0, wFullBatchCount, wProBatchSize);
        GenGatterIndex3D<T3, Trait>(initial3DRegHIdx, 0, whFullBatchCount, hProBatchSize, wFullBatchCount, 0);
        GenGatterIndex2D<T3, Trait>(initial3DRegHIdxOne, 0, hFullBatchCount, hProBatchSize);
        GenGatterIndex2D<T3, Trait>(initial2DRegWIdx, 0, wFullBatchCount, wProBatchSize);

        AscendC::Reg::MaskReg allMaskT3 = AscendC::Reg::CreateMask<T3, AscendC::Reg::MaskPattern::ALL, Trait>();
        AscendC::Reg::StoreAlign(helpAddrT3, initial3DRegWIdx, allMaskT3);
        AscendC::Reg::StoreAlign(helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3), initial3DRegHIdx, allMaskT3);
        AscendC::Reg::StoreAlign(helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3), initial3DRegHIdxOne,
                                 allMaskT3);
        AscendC::Reg::StoreAlign(helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3), initial2DRegWIdx,
                                 allMaskT3);
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
class AvgPoolV2GradNCHWKernel {
public:
    __aicore__ inline AvgPoolV2GradNCHWKernel(TPipe* pipe, const AvgPoolV2GradNCHWTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    template <const Reg::RegTrait& Trait>
    __aicore__ inline void singleLineProcessVF(__ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr);
    template <const Reg::RegTrait& Trait>
    __aicore__ inline void multipleLineProcessVF1(__ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr,
                                                  __ubuf__ uint32_t* helpAddr, __ubuf__ T3* helpAddrT3);
    template <const Reg::RegTrait& Trait>
    __aicore__ inline void multipleLineProcessVF2(__ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr,
                                                  __ubuf__ uint32_t* helpAddr, __ubuf__ T3* helpAddrT3);
    template <const Reg::RegTrait& Trait>
    __aicore__ inline void multipleLineProcessVF2Int64(__ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr,
                                                       __ubuf__ uint32_t* helpAddr, __ubuf__ T3* helpAddrT3);
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void CopyOut();

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> helpBuf_;
    TBuf<QuePosition::VECCALC> helpBufT3_;

    GlobalTensor<T1> gradGm_;
    GlobalTensor<T1> yGm_;
    const AvgPoolV2GradNCHWTilingData* tilingData_;

    uint32_t blockIdx_ = 0;
    int64_t highAxisActual_ = 1;
    int64_t hOutputActual_ = 1;

    int64_t wOutputActual_ = 1;
    int64_t wOutputAligned_ = 1;

    int64_t curCoreProcessNum_ = 1;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t hGradActual_ = 0;
    int64_t wGradActual_ = 0;
    int64_t wGradAligned_ = 0;
    int64_t hGradActualStart_ = 0;
    int64_t wGradActualStart_ = 0;

    int64_t highAxisGradOffset_ = 0;
    int64_t hAxisGradOffset_ = 0;
    int64_t wAxisGradOffset_ = 0;

    int64_t gradPlaneSize_ = 1;

    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;

    constexpr static int64_t DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T1);
};

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Init(GM_ADDR grad,
                                                                                                     GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    gradPlaneSize_ = tilingData_->hGrad * tilingData_->wGrad;
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    curCoreProcessNum_ = (blockIdx_ + 1 == tilingData_->usedCoreNum) ? tilingData_->tailCoreProcessNum :
                                                                       tilingData_->normalCoreProcessNum;
    gradGm_.SetGlobalBuffer((__gm__ T1*)grad);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);

    pipe_->InitBuffer(outputQue_, BUFFER_NUM, tilingData_->outputBufferSize);
    pipe_->InitBuffer(gradQue_, BUFFER_NUM, tilingData_->gradBufferSize);
    pipe_->InitBuffer(helpBuf_, HELP_BUFFER);
    pipe_->InitBuffer(helpBufT3_, HELP_BUFFER_T3);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ScalarCompute(
    int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tilingData_->normalCoreProcessNum + loopNum;
    highAxisIndex_ = baseBlockIdx / (tilingData_->hOutputOuter * tilingData_->wOutputOuter);
    highAxisActual_ = highAxisIndex_ == (tilingData_->highAxisOuter - 1) ? tilingData_->highAxisTail :
                                                                           tilingData_->highAxisInner;

    int64_t tempTail = baseBlockIdx % (tilingData_->hOutputOuter * tilingData_->wOutputOuter);
    hAxisIndex_ = tempTail / tilingData_->wOutputOuter;
    hOutputActual_ = hAxisIndex_ == (tilingData_->hOutputOuter - 1) ? tilingData_->hOutputTail :
                                                                      tilingData_->hOutputInner;

    wAxisIndex_ = tempTail % tilingData_->wOutputOuter;
    wOutputActual_ = wAxisIndex_ == (tilingData_->wOutputOuter - 1) ? tilingData_->wOutputTail :
                                                                      tilingData_->wOutputInner;
    wOutputAligned_ = (wOutputActual_ + DATA_NUM_IN_ONE_BLOCK - 1) / DATA_NUM_IN_ONE_BLOCK * DATA_NUM_IN_ONE_BLOCK;

    hGradActualStart_ = PStart(hAxisIndex_ * tilingData_->hOutputInner, tilingData_->padTopH, tilingData_->hKernel,
                               tilingData_->hStride);
    int64_t hGradActualEnd = PEnd(hAxisIndex_ * tilingData_->hOutputInner + hOutputActual_ - 1, tilingData_->padTopH,
                                  tilingData_->hStride, tilingData_->hGrad);
    wGradActualStart_ = PStart(wAxisIndex_ * tilingData_->wOutputInner, tilingData_->padLeftW, tilingData_->wKernel,
                               tilingData_->wStride);
    int64_t wGradActualEnd = PEnd(wAxisIndex_ * tilingData_->wOutputInner + wOutputActual_ - 1, tilingData_->padLeftW,
                                  tilingData_->wStride, tilingData_->wGrad);
    wGradActual_ = wGradActualEnd - wGradActualStart_;
    wGradAligned_ = (wGradActual_ + DATA_NUM_IN_ONE_BLOCK - 1) / DATA_NUM_IN_ONE_BLOCK * DATA_NUM_IN_ONE_BLOCK;
    hGradActual_ = hGradActualEnd - hGradActualStart_;

    curHProBatchSize_ = tilingData_->hProBatchSize > hGradActual_ ? hGradActual_ : tilingData_->hProBatchSize;
    curWProBatchSize_ = tilingData_->wProBatchSize > wGradActual_ ? wGradActual_ : tilingData_->wProBatchSize;

    highAxisGradOffset_ = highAxisIndex_ * tilingData_->highAxisInner * gradPlaneSize_;
    hAxisGradOffset_ = hGradActualStart_ * tilingData_->wGrad;
    wAxisGradOffset_ = wGradActualStart_;
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Compute()
{
    uint32_t calCount = tilingData_->outputBufferSize / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);

    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();

    __ubuf__ computeType* yAddr = (__ubuf__ computeType*)yLocal.GetPhyAddr();
    __ubuf__ T1* gradAddr = (__ubuf__ T1*)gradLocal.GetPhyAddr();
    LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
    __ubuf__ uint32_t* helpAddr = (__ubuf__ uint32_t*)helpTensor.GetPhyAddr();
    LocalTensor<T3> helpTensorT3 = helpBufT3_.Get<T3>();
    __ubuf__ T3* helpAddrT3 = (__ubuf__ T3*)helpTensorT3.GetPhyAddr();

    uint32_t wConcurrentCount = wGradActual_ / curWProBatchSize_;
    uint32_t hConcurrentCount = hGradActual_ / curHProBatchSize_;
    if (wConcurrentCount * DOUBLE * sizeof(float) > V_REG_SIZE) {
        if constexpr (std::is_same<T3, int64_t>::value) {
            singleLineProcessVF<AscendC::Reg::RegTraitNumTwo>(yAddr, gradAddr);
        } else {
            singleLineProcessVF<AscendC::Reg::RegTraitNumOne>(yAddr, gradAddr);
        }
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(float) > V_REG_SIZE) {
        // HW 并发处理
        if constexpr (std::is_same<T3, int64_t>::value) {
            multipleLineProcessVF1<AscendC::Reg::RegTraitNumTwo>(yAddr, gradAddr, helpAddr, helpAddrT3);
        } else {
            multipleLineProcessVF1<AscendC::Reg::RegTraitNumOne>(yAddr, gradAddr, helpAddr, helpAddrT3);
        }
    } else {
        // NCHW 并发处理
        if constexpr (std::is_same<T3, int64_t>::value) {
            multipleLineProcessVF2Int64<AscendC::Reg::RegTraitNumTwo>(yAddr, gradAddr, helpAddr, helpAddrT3);
        } else {
            multipleLineProcessVF2<AscendC::Reg::RegTraitNumOne>(yAddr, gradAddr, helpAddr, helpAddrT3);
        }
    }

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }

    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(tilingData_->outputBufferSize) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, static_cast<T1>(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
    return;
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ProcessPerLoop()
{
    if (hGradActual_ <= 0 || wGradActual_ <= 0) {
        ProcessNoArgmaxBlock(); // ceilMode为false时，最后的尾块可能是这种情况
        return;
    }

    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();

    int64_t gradGmOffset = highAxisGradOffset_ + hAxisGradOffset_ + wAxisGradOffset_;
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop1SrcStride = gradPlaneSize_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop1DstStride = hGradActual_ * wGradAligned_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = 0;

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(hGradActual_),
                                        static_cast<uint32_t>(wGradActual_ * sizeof(T1)),
                                        static_cast<uint32_t>((tilingData_->wGrad - wGradActual_) * sizeof(T1)),
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(gradLocal, gradGm_[gradGmOffset], copyOutParamT1, paramsT1);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const Reg::RegTrait& Trait>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::singleLineProcessVF(
    __ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr)
{
    int64_t wOutput = tilingData_->wOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradActual = wGradActual_;
    int64_t wGradAligned = wGradAligned_;
    uint16_t hGradActual = hGradActual_;
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);

    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint32_t wFullBatchCount = wGradActual / wProBatchSize;

    uint16_t computeSizeFp32 = V_REG_SIZE / sizeof(float);

    uint16_t repeatimes = wFullBatchCount / computeSizeFp32;
    uint16_t wRemain = wGradActual - repeatimes * wProBatchSize * computeSizeFp32;

    uint32_t wRemainBatchCount = wRemain / wProBatchSize;
    uint16_t wRemainTail = wRemain % wProBatchSize;

    uint32_t one = 1;
    uint32_t all = computeSizeFp32;

    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highGradOffset = highIdx * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
        for (uint16_t hIdx = 0; hIdx < hGradActual; hIdx++) {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<int32_t> zeroConstReg;
                AscendC::Reg::RegTensor<int32_t> wMaxReg;
                AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                }

                AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
                AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
                AscendC::Reg::RegTensor<int32_t> wIndexReg;
                AscendC::Reg::RegTensor<int32_t> divisorReg;

                AscendC::Reg::RegTensor<T3, Trait> initialWRegIdx;
                AscendC::Reg::RegTensor<T3, Trait> outWStart;
                AscendC::Reg::RegTensor<T3, Trait> outHStart;
                AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }

                AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
                GenInitial1DIndices<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);
                GenInitial1DIndices<T3, Trait>(initialWRegIdx, wProBatchSize);

                T3 hGradOffset = hIdx + hGradActualStart;
                AscendC::Reg::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                int32_t hIndex = hGradOffset * strideH - curHIndex - padH;
                int32_t hkStart = hIndex >= 0 ? 0 : (-hIndex);
                int32_t hkEnd = (hOutputActual - hIndex) > kH ? kH : (hOutputActual - hIndex);
                for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wRepeatIdx * computeSizeFp32 * wProBatchSize + wGradActualStart;
                        uint32_t offset = wBatchIdx + wRepeatIdx * computeSizeFp32 * wProBatchSize +
                                          hIdx * wGradAligned + highGradOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);

                        ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                        ComputeOutWIndex<T3, Trait>(wIndexReg, outWStart, curWIndex, padW, all);
                        GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                            padRightW, kH, kW, divisorOverride, all);
                        DoSingleNCNchwForMergeW<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, all, wOutputAligned, highOutputOffset, zeroConstReg,
                            wMaxReg, kW, divisorReg, wIndexReg, hIndex, hkStart, hkEnd);
                    }
                }
                // 尾段整batch  用不满mask
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + repeatimes * computeSizeFp32 * wProBatchSize + wGradActualStart;
                    uint32_t offset = wBatchIdx + repeatimes * computeSizeFp32 * wProBatchSize + hIdx * wGradAligned +
                                      highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);

                    ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                    ComputeOutWIndex<T3, Trait>(wIndexReg, outWStart, curWIndex, padW, all);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, wRemainBatchCount);
                    DoSingleNCNchwForMergeW<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, wRemainBatchCount,
                                                                wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg,
                                                                kW, divisorReg, wIndexReg, hIndex, hkStart, hkEnd);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    uint32_t wGradOffset = wBatchIdx + wRemainBatchCount * wProBatchSize +
                                           repeatimes * computeSizeFp32 * wProBatchSize + wGradActualStart;
                    uint32_t offset = wBatchIdx + wRemainBatchCount * wProBatchSize +
                                      repeatimes * computeSizeFp32 * wProBatchSize + hIdx * wGradAligned +
                                      highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);

                    ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                    ComputeOutWIndex<T3, Trait>(wIndexReg, outWStart, curWIndex, padW, all);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, one);
                    DoSingleNCNchwForMergeW<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, one, wOutputAligned,
                                                                highOutputOffset, zeroConstReg, wMaxReg, kW, divisorReg,
                                                                wIndexReg, hIndex, hkStart, hkEnd);
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const Reg::RegTrait& Trait>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::multipleLineProcessVF1(
    __ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr, __ubuf__ uint32_t* helpAddr, __ubuf__ T3* helpAddrT3)
{
    int64_t wOutput = tilingData_->wOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradAligned = wGradAligned_;
    int64_t wGradActual = wGradActual_;
    uint16_t hGradActual = hGradActual_;
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);

    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint32_t wFullBatchCount = wGradActual / wProBatchSize;
    uint16_t hFullBatchCount = hGradActual / hProBatchSize;
    uint16_t wRemainTail = wGradActual % wProBatchSize;

    uint16_t hConcurrentCount = V_REG_SIZE / (wFullBatchCount * sizeof(float));

    uint16_t blockConcurrentCount = hFullBatchCount / hConcurrentCount;
    uint16_t hRemain = hGradActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;

    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;

    uint32_t blockOne = 1 * hConcurrentCount;
    uint32_t remainBatchOne = 1 * hRemainBatchCount;
    uint32_t remainTailOne = 1;
    uint32_t maskBlock = wFullBatchCount * hConcurrentCount;
    uint32_t maskRemainBatch = wFullBatchCount * hRemainBatchCount;
    uint32_t maskRemainTail = wFullBatchCount;

    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highGradOffset = highIdx * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
            AscendC::Reg::RegTensor<uint32_t> initialRegIndexOne;
            GenInitial2DIndices((AscendC::Reg::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                                wGradAligned, wFullBatchCount);
            Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wGradAligned);

            AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::StoreAlign(helpAddr, initialRegIndex, allMask);
            AscendC::Reg::StoreAlign(helpAddr + V_REG_SIZE / sizeof(uint32_t), initialRegIndexOne, allMask);
        }

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T3, Trait> initialWRegIdx;
            AscendC::Reg::RegTensor<T3, Trait> initialHRegIdx;
            AscendC::Reg::RegTensor<T3, Trait> initialHRegIdxOne;
            GenGatterIndex2D<T3, Trait>(initialWRegIdx, 0, wFullBatchCount, wProBatchSize);
            GenGatterIndex2D<T3, Trait>(initialHRegIdx, hProBatchSize, wFullBatchCount, 0);
            GenInitial1DIndices<T3, Trait>(initialHRegIdxOne, hProBatchSize);

            AscendC::Reg::MaskReg allMaskT3 = AscendC::Reg::CreateMask<T3, AscendC::Reg::MaskPattern::ALL, Trait>();
            AscendC::Reg::StoreAlign(helpAddrT3, initialWRegIdx, allMaskT3);
            AscendC::Reg::StoreAlign(helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3), initialHRegIdx, allMaskT3);
            AscendC::Reg::StoreAlign(helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3), initialHRegIdxOne,
                                     allMaskT3);
        }

        for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::Reg::RegTensor<int32_t> zeroConstReg;
                    AscendC::Reg::RegTensor<int32_t> wMaxReg;
                    AscendC::Reg::RegTensor<int32_t> hMaxReg;
                    AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                        AscendC::Reg::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    }

                    AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
                    AscendC::Reg::RegTensor<uint32_t> initialRegIndexOne;
                    AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::Reg::RegTensor<int32_t> wIndexReg;
                    AscendC::Reg::RegTensor<int32_t> hIndexReg;
                    AscendC::Reg::RegTensor<int32_t> divisorReg;

                    AscendC::Reg::RegTensor<T3, Trait> initialWRegIdx;
                    AscendC::Reg::RegTensor<T3, Trait> initialHRegIdx;
                    AscendC::Reg::RegTensor<T3, Trait> initialHRegIdxOne;
                    AscendC::Reg::RegTensor<T3, Trait> outWStart;
                    AscendC::Reg::RegTensor<T3, Trait> outHStart;
                    AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }

                    AscendC::Reg::MaskReg
                        allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::LoadAlign(initialRegIndex, helpAddr);
                    AscendC::Reg::LoadAlign(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));

                    AscendC::Reg::LoadAlign(initialWRegIdx, helpAddrT3);
                    AscendC::Reg::LoadAlign(initialHRegIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
                    AscendC::Reg::LoadAlign(initialHRegIdxOne,
                                            helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));

                    // 整batch
                    T3 hGradOffset = hProBatchIdx + hIdx * hProBatchSize * hConcurrentCount + hGradActualStart;
                    ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdx, hGradOffset, strideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                          hIdx * wGradAligned * hProBatchSize * hConcurrentCount + highGradOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);

                        ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, maskBlock);
                        GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                            padRightW, kH, kW, divisorOverride, maskBlock);
                        DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, maskBlock, wOutputAligned,
                                                           highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                           divisorReg, wIndexReg, hIndexReg, zeroConstReg);
                    }

                    // 尾段零散点
                    ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdxOne, hGradOffset, strideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                          hIdx * wGradAligned * hProBatchSize * hConcurrentCount + highGradOffset;
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);

                        AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, blockOne);
                        GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                            padRightW, kH, kW, divisorOverride, blockOne);
                        DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, blockOne, wOutputAligned,
                                                           highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                           divisorReg, wIndexReg, hIndexReg, zeroConstReg);
                    }
                }
            }
        }

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<int32_t> zeroConstReg;
            AscendC::Reg::RegTensor<int32_t> wMaxReg;
            AscendC::Reg::RegTensor<int32_t> hMaxReg;
            AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                AscendC::Reg::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
            }

            AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
            AscendC::Reg::RegTensor<uint32_t> initialRegIndexOne;
            AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
            AscendC::Reg::RegTensor<int32_t> wIndexReg;
            AscendC::Reg::RegTensor<int32_t> hIndexReg;
            AscendC::Reg::RegTensor<int32_t> divisorReg;

            AscendC::Reg::RegTensor<T3, Trait> initialWRegIdx;
            AscendC::Reg::RegTensor<T3, Trait> initialHRegIdx;
            AscendC::Reg::RegTensor<T3, Trait> initialHRegIdxOne;
            AscendC::Reg::RegTensor<T3, Trait> outWStart;
            AscendC::Reg::RegTensor<T3, Trait> outHStart;
            AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
            if constexpr (COUNT_PAD == 0) {
                AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
            }

            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::LoadAlign(initialRegIndex, helpAddr);
            AscendC::Reg::LoadAlign(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));

            AscendC::Reg::LoadAlign(initialWRegIdx, helpAddrT3);
            AscendC::Reg::LoadAlign(initialHRegIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
            AscendC::Reg::LoadAlign(initialHRegIdxOne, helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
            // 尾行  完整hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                T3 hGradOffset = hProBatchIdx + blockConcurrentCount * hProBatchSize * hConcurrentCount +
                                 hGradActualStart;
                ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdx, hGradOffset, strideH);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                      blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                      highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);

                    ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, maskRemainBatch);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, maskRemainBatch);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, maskRemainBatch,
                                                       wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                       kH, kW, divisorReg, wIndexReg, hIndexReg, zeroConstReg);
                }

                // 尾段零散点
                ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdxOne, hGradOffset, strideH);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                      blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                      highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);

                    AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, remainBatchOne);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, remainBatchOne);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, remainBatchOne,
                                                       wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                       kH, kW, divisorReg, wIndexReg, hIndexReg, zeroConstReg);
                }
            }
            // 尾行  零散hProBatch
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                T3 hGradOffset = hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                 blockConcurrentCount * hProBatchSize * hConcurrentCount + hGradActualStart;
                AscendC::Reg::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                      hRemainBatchCount * hProBatchSize * wGradAligned +
                                      blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                      highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);

                    ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, maskRemainTail);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, maskRemainTail);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, maskRemainTail,
                                                       wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                                                       kH, kW, divisorReg, wIndexReg, hIndexReg, zeroConstReg);
                }

                // 尾段零散点
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                      hRemainBatchCount * hProBatchSize * wGradAligned +
                                      blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                      highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);

                    AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, remainTailOne);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, remainTailOne);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, remainTailOne, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, zeroConstReg);
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const Reg::RegTrait& Trait>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::multipleLineProcessVF2(
    __ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr, __ubuf__ uint32_t* helpAddr, __ubuf__ T3* helpAddrT3)
{
    int64_t wOutput = tilingData_->wOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t highAxisActual = highAxisActual_;
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradAligned = wGradAligned_;
    int64_t wGradActual = wGradActual_;
    uint16_t hGradActual = hGradActual_;
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);
    int64_t highOutStride = wOutputAligned * hOutputActual;

    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint32_t wFullBatchCount = wGradActual / wProBatchSize;
    uint16_t hFullBatchCount = hGradActual / hProBatchSize;
    uint16_t wRemainTail = wGradActual % wProBatchSize;
    uint32_t whFullBatchCount = wFullBatchCount * hFullBatchCount;

    uint16_t highConcurrentCount = V_REG_SIZE / (whFullBatchCount * sizeof(float));

    uint16_t highBlockConcurrentCount = highAxisActual / highConcurrentCount;
    uint16_t highBlockRemainTail = highAxisActual - highBlockConcurrentCount * highConcurrentCount;

    uint16_t hRemainTail = hGradActual - hFullBatchCount * hProBatchSize;

    uint32_t mask0 = highConcurrentCount * whFullBatchCount;
    uint32_t mask1 = highConcurrentCount * hFullBatchCount * 1;
    uint32_t mask2 = highConcurrentCount * 1 * wFullBatchCount;
    uint32_t mask3 = highConcurrentCount * 1 * 1;
    uint32_t mask4 = highBlockRemainTail * whFullBatchCount;
    uint32_t mask5 = highBlockRemainTail * hFullBatchCount * 1;
    uint32_t mask6 = highBlockRemainTail * 1 * wFullBatchCount;
    uint32_t mask7 = highBlockRemainTail * 1 * 1;

    GenIndicesToUb(helpAddr, wProBatchSize, hProBatchSize, wGradAligned, wFullBatchCount, hFullBatchCount, hGradActual);
    GenIndicesToUbForT3<T3, Trait>(helpAddrT3, whFullBatchCount, wFullBatchCount, wProBatchSize, hProBatchSize,
                                   hFullBatchCount);

    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highGradOffset = highBlockIdx * highConcurrentCount * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<int32_t> zeroConstReg;
            AscendC::Reg::RegTensor<int32_t> wMaxReg;
            AscendC::Reg::RegTensor<int32_t> hMaxReg;
            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                AscendC::Reg::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
            }

            AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
            AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
            AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
            AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
            AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
            AscendC::Reg::RegTensor<int32_t> wIndexReg;
            AscendC::Reg::RegTensor<int32_t> hIndexReg;
            AscendC::Reg::RegTensor<int32_t> highIdxReg;
            AscendC::Reg::RegTensor<int32_t> divisorReg;

            AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdx;
            AscendC::Reg::RegTensor<T3, Trait> initial3DRegWIdx;
            AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdxOne;
            AscendC::Reg::RegTensor<T3, Trait> initial2DRegWIdx;
            AscendC::Reg::RegTensor<T3, Trait> outWStart;
            AscendC::Reg::RegTensor<T3, Trait> outHStart;
            AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
            if constexpr (COUNT_PAD == 0) {
                AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
            }

            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

            AscendC::Reg::LoadAlign(initial3DRegIndex, helpAddr);
            AscendC::Reg::LoadAlign(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
            AscendC::Reg::LoadAlign(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
            AscendC::Reg::LoadAlign(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

            AscendC::Reg::LoadAlign(initial3DRegWIdx, helpAddrT3);
            AscendC::Reg::LoadAlign(initial3DRegHIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
            AscendC::Reg::LoadAlign(initial3DRegHIdxOne, helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
            AscendC::Reg::LoadAlign(initial2DRegWIdx, helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3));

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                // 整batch
                T3 hGradOffset = hProBatchIdx + hGradActualStart;
                ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdx, hGradOffset, strideH);
                GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, whFullBatchCount, 0);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned + highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);

                    ComputeOutRegStart<T3, Trait>(outWStart, initial3DRegWIdx, wGradOffset, strideW);
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, mask0);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask0);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask0, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, highIdxReg);
                }

                // 尾段零散点
                ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdxOne, hGradOffset, strideH);
                GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, hFullBatchCount, 0);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                      highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);

                    AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, mask1);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask1);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask1, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, highIdxReg);
                }
            }

            // hRemainTail
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
                AscendC::Reg::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, wFullBatchCount, 0);
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned +
                                      highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);

                    ComputeOutRegStart<T3, Trait>(outWStart, initial2DRegWIdx, wGradOffset, strideW);
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, mask2);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask2);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask2, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, highIdxReg);
                }

                // 尾段零散点
                GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, 1, 0);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount +
                                      (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned + highGradOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);

                    AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, mask3);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask3);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask3, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, highIdxReg);
                }
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
            AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
        }

        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
        AscendC::Reg::RegTensor<int32_t> wIndexReg;
        AscendC::Reg::RegTensor<int32_t> hIndexReg;
        AscendC::Reg::RegTensor<int32_t> highIdxReg;
        AscendC::Reg::RegTensor<int32_t> divisorReg;

        AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdx;
        AscendC::Reg::RegTensor<T3, Trait> initial3DRegWIdx;
        AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdxOne;
        AscendC::Reg::RegTensor<T3, Trait> initial2DRegWIdx;
        AscendC::Reg::RegTensor<T3, Trait> outWStart;
        AscendC::Reg::RegTensor<T3, Trait> outHStart;
        AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
        if constexpr (COUNT_PAD == 0) {
            AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
        }

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::LoadAlign(initial3DRegIndex, helpAddr);
        AscendC::Reg::LoadAlign(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::LoadAlign(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::LoadAlign(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

        AscendC::Reg::LoadAlign(initial3DRegWIdx, helpAddrT3);
        AscendC::Reg::LoadAlign(initial3DRegHIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
        AscendC::Reg::LoadAlign(initial3DRegHIdxOne, helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
        AscendC::Reg::LoadAlign(initial2DRegWIdx, helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3));

        // highBlockRemainTail
        uint32_t highGradOffset = highBlockConcurrentCount * highConcurrentCount * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        // 整H batch
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            // 整batch
            T3 hGradOffset = hProBatchIdx + hGradActualStart;
            ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdx, hGradOffset, strideH);
            GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, whFullBatchCount, 0);
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T3 wGradOffset = wBatchIdx + wGradActualStart;
                uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned + highGradOffset;
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);

                ComputeOutRegStart<T3, Trait>(outWStart, initial3DRegWIdx, wGradOffset, strideW);
                ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                             padW, mask4);
                GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                    divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH, padRightW,
                    kH, kW, divisorOverride, mask4);
                DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask4, wOutputAligned,
                                                   highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg,
                                                   wIndexReg, hIndexReg, highIdxReg);
            }

            // 尾段零散点
            ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdxOne, hGradOffset, strideH);
            GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, hFullBatchCount, 0);
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                  highGradOffset;
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);

                AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                             padW, mask5);
                GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                    divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH, padRightW,
                    kH, kW, divisorOverride, mask5);
                DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask5, wOutputAligned,
                                                   highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg,
                                                   wIndexReg, hIndexReg, highIdxReg);
            }
        }

        // hRemainTail
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
            AscendC::Reg::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
            GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, wFullBatchCount, 0);
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T3 wGradOffset = wBatchIdx + wGradActualStart;
                uint32_t offset = wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wGradAligned +
                                  highGradOffset;
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);

                ComputeOutRegStart<T3, Trait>(outWStart, initial2DRegWIdx, wGradOffset, strideW);
                ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                             padW, mask6);
                GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                    divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH, padRightW,
                    kH, kW, divisorOverride, mask6);
                DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask6, wOutputAligned,
                                                   highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg,
                                                   wIndexReg, hIndexReg, highIdxReg);
            }

            // 尾段零散点
            GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, 1, 0);
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount +
                                  (hFullBatchCount * hProBatchSize + hProBatchIdx) * wGradAligned + highGradOffset;
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);

                AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                             padW, mask7);
                GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                    divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH, padRightW,
                    kH, kW, divisorOverride, mask7);
                DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask7, wOutputAligned,
                                                   highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg,
                                                   wIndexReg, hIndexReg, highIdxReg);
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const Reg::RegTrait& Trait>
__aicore__ inline void
AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::multipleLineProcessVF2Int64(
    __ubuf__ computeType* yAddr, __ubuf__ T1* gradAddr, __ubuf__ uint32_t* helpAddr, __ubuf__ T3* helpAddrT3)
{
    int64_t wOutput = tilingData_->wOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t highAxisActual = highAxisActual_;
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradAligned = wGradAligned_;
    int64_t wGradActual = wGradActual_;
    uint16_t hGradActual = hGradActual_;
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);
    int64_t highOutStride = wOutputAligned * hOutputActual;

    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint32_t wFullBatchCount = wGradActual / wProBatchSize;
    uint16_t hFullBatchCount = hGradActual / hProBatchSize;
    uint16_t wRemainTail = wGradActual % wProBatchSize;
    uint32_t whFullBatchCount = wFullBatchCount * hFullBatchCount;

    uint16_t highConcurrentCount = V_REG_SIZE / (whFullBatchCount * sizeof(float));

    uint16_t highBlockConcurrentCount = highAxisActual / highConcurrentCount;
    uint16_t highBlockRemainTail = highAxisActual - highBlockConcurrentCount * highConcurrentCount;

    uint16_t hRemainTail = hGradActual - hFullBatchCount * hProBatchSize;

    uint32_t mask0 = highConcurrentCount * whFullBatchCount;
    uint32_t mask1 = highConcurrentCount * hFullBatchCount * 1;
    uint32_t mask2 = highConcurrentCount * 1 * wFullBatchCount;
    uint32_t mask3 = highConcurrentCount * 1 * 1;
    uint32_t mask4 = highBlockRemainTail * whFullBatchCount;
    uint32_t mask5 = highBlockRemainTail * hFullBatchCount * 1;
    uint32_t mask6 = highBlockRemainTail * 1 * wFullBatchCount;
    uint32_t mask7 = highBlockRemainTail * 1 * 1;

    GenIndicesToUb(helpAddr, wProBatchSize, hProBatchSize, wGradAligned, wFullBatchCount, hFullBatchCount, hGradActual);
    GenIndicesToUbForT3<T3, Trait>(helpAddrT3, whFullBatchCount, wFullBatchCount, wProBatchSize, hProBatchSize,
                                   hFullBatchCount);

    for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
        uint32_t highGradOffset = highBlockIdx * highConcurrentCount * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<int32_t> zeroConstReg;
                AscendC::Reg::RegTensor<int32_t> wMaxReg;
                AscendC::Reg::RegTensor<int32_t> hMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                    AscendC::Reg::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                }

                AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
                AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
                AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
                AscendC::Reg::RegTensor<int32_t> wIndexReg;
                AscendC::Reg::RegTensor<int32_t> hIndexReg;
                AscendC::Reg::RegTensor<int32_t> highIdxReg;
                AscendC::Reg::RegTensor<int32_t> divisorReg;

                AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdx;
                AscendC::Reg::RegTensor<T3, Trait> initial3DRegWIdx;
                AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdxOne;
                AscendC::Reg::RegTensor<T3, Trait> outWStart;
                AscendC::Reg::RegTensor<T3, Trait> outHStart;
                AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }

                AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                AscendC::Reg::LoadAlign(initial3DRegIndex, helpAddr);
                AscendC::Reg::LoadAlign(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));

                AscendC::Reg::LoadAlign(initial3DRegWIdx, helpAddrT3);
                AscendC::Reg::LoadAlign(initial3DRegHIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
                AscendC::Reg::LoadAlign(initial3DRegHIdxOne,
                                        helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));

                // 整batch
                T3 hGradOffset = hProBatchIdx + hGradActualStart;
                ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdx, hGradOffset, strideH);
                GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, whFullBatchCount, 0);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = (wBatchIdx + hProBatchIdx * wGradAligned + highGradOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);

                    ComputeOutRegStart<T3, Trait>(outWStart, initial3DRegWIdx, wGradOffset, strideW);
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, mask0);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask0);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask0, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, highIdxReg);
                }

                // 尾段零散点
                ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdxOne, hGradOffset, strideH);
                GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, hFullBatchCount, 0);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                       highGradOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);

                    AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, mask1);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask1);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask1, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, highIdxReg);
                }
            }
        }

        // hRemainTail
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<int32_t> zeroConstReg;
                AscendC::Reg::RegTensor<int32_t> wMaxReg;
                AscendC::Reg::RegTensor<int32_t> hMaxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                    AscendC::Reg::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                }

                AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
                AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
                AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
                AscendC::Reg::RegTensor<int32_t> wIndexReg;
                AscendC::Reg::RegTensor<int32_t> hIndexReg;
                AscendC::Reg::RegTensor<int32_t> highIdxReg;
                AscendC::Reg::RegTensor<int32_t> divisorReg;

                AscendC::Reg::RegTensor<T3, Trait> initial2DRegWIdx;
                AscendC::Reg::RegTensor<T3, Trait> outWStart;
                AscendC::Reg::RegTensor<T3, Trait> outHStart;
                AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }

                AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                AscendC::Reg::LoadAlign(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
                AscendC::Reg::LoadAlign(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

                AscendC::Reg::LoadAlign(initial2DRegWIdx,
                                        helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3));

                T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
                AscendC::Reg::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, wFullBatchCount, 0);
                // 整batch
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned +
                                       highGradOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);

                    ComputeOutRegStart<T3, Trait>(outWStart, initial2DRegWIdx, wGradOffset, strideW);
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, mask2);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask2);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask2, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, highIdxReg);
                }

                // 尾段零散点
                GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, 1, 0);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                       (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned +
                                       highGradOffset);
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);

                    AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                                 padW, mask3);
                    GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH,
                        padRightW, kH, kW, divisorOverride, mask3);
                    DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask3, wOutputAligned,
                                                       highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW,
                                                       divisorReg, wIndexReg, hIndexReg, highIdxReg);
                }
            }
        }
    }

    // highBlockRemainTail
    uint32_t highGradOffset = highBlockConcurrentCount * highConcurrentCount * hGradActual * wGradAligned;
    uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
    // 整H batch
    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<int32_t> zeroConstReg;
            AscendC::Reg::RegTensor<int32_t> wMaxReg;
            AscendC::Reg::RegTensor<int32_t> hMaxReg;
            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                AscendC::Reg::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
            }

            AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
            AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
            AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
            AscendC::Reg::RegTensor<int32_t> wIndexReg;
            AscendC::Reg::RegTensor<int32_t> hIndexReg;
            AscendC::Reg::RegTensor<int32_t> highIdxReg;
            AscendC::Reg::RegTensor<int32_t> divisorReg;

            AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdx;
            AscendC::Reg::RegTensor<T3, Trait> initial3DRegWIdx;
            AscendC::Reg::RegTensor<T3, Trait> initial3DRegHIdxOne;
            AscendC::Reg::RegTensor<T3, Trait> outWStart;
            AscendC::Reg::RegTensor<T3, Trait> outHStart;
            AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
            if constexpr (COUNT_PAD == 0) {
                AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
            }

            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

            AscendC::Reg::LoadAlign(initial3DRegIndex, helpAddr);
            AscendC::Reg::LoadAlign(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));

            AscendC::Reg::LoadAlign(initial3DRegWIdx, helpAddrT3);
            AscendC::Reg::LoadAlign(initial3DRegHIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
            AscendC::Reg::LoadAlign(initial3DRegHIdxOne, helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));

            // 整batch
            T3 hGradOffset = hProBatchIdx + hGradActualStart;
            ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdx, hGradOffset, strideH);
            GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, whFullBatchCount, 0);
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T3 wGradOffset = wBatchIdx + wGradActualStart;
                uint32_t offset = (wBatchIdx + hProBatchIdx * wGradAligned + highGradOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);

                ComputeOutRegStart<T3, Trait>(outWStart, initial3DRegWIdx, wGradOffset, strideW);
                ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                             padW, mask4);
                GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                    divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH, padRightW,
                    kH, kW, divisorOverride, mask4);
                DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask4, wOutputAligned,
                                                   highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg,
                                                   wIndexReg, hIndexReg, highIdxReg);
            }

            // 尾段零散点
            ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdxOne, hGradOffset, strideH);
            GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, hFullBatchCount, 0);
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                   highGradOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);

                AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                             padW, mask5);
                GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                    divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH, padRightW,
                    kH, kW, divisorOverride, mask5);
                DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask5, wOutputAligned,
                                                   highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg,
                                                   wIndexReg, hIndexReg, highIdxReg);
            }
        }
    }

    // hRemainTail
    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<int32_t> zeroConstReg;
            AscendC::Reg::RegTensor<int32_t> wMaxReg;
            AscendC::Reg::RegTensor<int32_t> hMaxReg;
            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::Reg::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                AscendC::Reg::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                AscendC::Reg::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
            }

            AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
            AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
            AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
            AscendC::Reg::RegTensor<int32_t> wIndexReg;
            AscendC::Reg::RegTensor<int32_t> hIndexReg;
            AscendC::Reg::RegTensor<int32_t> highIdxReg;
            AscendC::Reg::RegTensor<int32_t> divisorReg;

            AscendC::Reg::RegTensor<T3, Trait> initial2DRegWIdx;
            AscendC::Reg::RegTensor<T3, Trait> outWStart;
            AscendC::Reg::RegTensor<T3, Trait> outHStart;
            AscendC::Reg::RegTensor<T3, Trait> zeroConstRegT;
            if constexpr (COUNT_PAD == 0) {
                AscendC::Reg::Duplicate(zeroConstRegT, static_cast<T3>(0));
            }

            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

            AscendC::Reg::LoadAlign(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
            AscendC::Reg::LoadAlign(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

            AscendC::Reg::LoadAlign(initial2DRegWIdx, helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3));

            T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
            AscendC::Reg::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
            GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, wFullBatchCount, 0);
            // 整batch
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                T3 wGradOffset = wBatchIdx + wGradActualStart;
                uint32_t offset = (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wGradAligned +
                                   highGradOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);

                ComputeOutRegStart<T3, Trait>(outWStart, initial2DRegWIdx, wGradOffset, strideW);
                ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                             padW, mask6);
                GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                    divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH, padRightW,
                    kH, kW, divisorOverride, mask6);
                DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask6, wOutputAligned,
                                                   highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg,
                                                   wIndexReg, hIndexReg, highIdxReg);
            }

            // 尾段零散点
            GenGatterIndex2D<int32_t>(highIdxReg, highOutStride, 1, 0);
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                   (hFullBatchCount * hProBatchSize + hProBatchIdx) * wGradAligned + highGradOffset);
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);

                AscendC::Reg::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex, padH,
                                             padW, mask7);
                GenDivisor<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                    divisorReg, outWStart, outHStart, zeroConstRegT, hOutput, wOutput, padH, padW, padDownH, padRightW,
                    kH, kW, divisorOverride, mask7);
                DoSingleNCNchw<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask7, wOutputAligned,
                                                   highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, kH, kW, divisorReg,
                                                   wIndexReg, hIndexReg, highIdxReg);
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPoolV2GradNCHWKernel<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();

    int64_t outputPlaneSize = tilingData_->hOutput * tilingData_->wOutput;
    int64_t highOutputAxisOffset = highAxisIndex_ * tilingData_->highAxisInner * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * tilingData_->hOutputInner * tilingData_->wOutput;
    int64_t wOutputAxisOffset = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t outputGmOffset = highOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop1DstStride = tilingData_->hOutput * tilingData_->wOutput * sizeof(T1);
    loopModeParamsT1.loop2DstStride = 0;

    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(hOutputActual_),
                                        static_cast<uint32_t>(wOutputActual_ * sizeof(T1)), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((tilingData_->wOutput - wOutputActual_) * sizeof(T1)),
                                        static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}
} // namespace AvgPoolV2GradNCHWNameSpace
#endif // AVG_POOL_V2_GRAD_NCHW_KERNEL_H_
