/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_ndhwc_kernel.h
 * \brief NDHWC kernel for 3D average pooling backward (arch35).
 *        Extends avg_pool_v2_grad_nhwc_kernel.h from H*W to D*H*W concurrency.
 */

#ifndef AVG_POOL3_D_GRAD_NDHWC_KERNEL_H_
#define AVG_POOL3_D_GRAD_NDHWC_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "avg_pool3_d_grad_base.h"
#include "avg_pool3_d_grad_tiling_data.h"

namespace AvgPool3DGradNDHWCNameSpace {
using namespace AscendC;
using namespace AvgPool3DGrad;
using computeType = float;
constexpr static int64_t V_REG_SIZE = platform::GetVRegSize();

template <typename T1>
__aicore__ inline void GetContinuousInput(MicroAPI::RegTensor<computeType>& gradReg, __local_mem__ T1* gradAddr,
                                          uint32_t gradOffset)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::MicroAPI::RegTensor<T1> gradRegT1;
        AscendC::MicroAPI::MaskReg
            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(gradRegT1, gradAddr + gradOffset);
        AscendC::MicroAPI::UnPack((AscendC::MicroAPI::RegTensor<uint32_t>&)gradRegT1,
                                  (AscendC::MicroAPI::RegTensor<uint16_t>&)gradRegT1);
        AscendC::MicroAPI::Cast<computeType, T1, castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::MicroAPI::DataCopy(gradReg, gradAddr + gradOffset);
    }
}

template <typename T>
__aicore__ inline void GradientAccBigC(__local_mem__ computeType* yAddr, MicroAPI::RegTensor<computeType>& gradReg,
                                       T scatterIndex, MicroAPI::RegTensor<int32_t> divisorReg,
                                       MicroAPI::MaskReg& pregRes)
{
    AscendC::MicroAPI::RegTensor<computeType> scatterAccResReg;
    AscendC::MicroAPI::RegTensor<computeType> divisorCastReg;
    AscendC::MicroAPI::RegTensor<computeType> divisorResReg;
    AscendC::MicroAPI::DataCopy(scatterAccResReg, yAddr + scatterIndex);
    AscendC::MicroAPI::Cast<computeType, int32_t, castTraitI32F32>(divisorCastReg, divisorReg, pregRes);
    AscendC::MicroAPI::Div(divisorResReg, gradReg, divisorCastReg, pregRes);
    AscendC::MicroAPI::Add(scatterAccResReg, scatterAccResReg, divisorResReg, pregRes);
    AscendC::MicroAPI::DataCopy(yAddr + scatterIndex, scatterAccResReg, pregRes);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void DoSingleCNdhwc(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, uint32_t gradOffset,
                                      uint32_t gradMaskCount, int32_t dOutputActual, int32_t hOutputActual,
                                      int32_t wOutputActual, int32_t cOutputAligned, int32_t cOffset, int32_t nOffset,
                                      int32_t dkStart, int32_t dkEnd, int32_t hkStart, int32_t hkEnd, int32_t wkStart,
                                      int32_t wkEnd, MicroAPI::RegTensor<int32_t>& divisorReg, int32_t dIndex,
                                      int32_t hIndex, int32_t wIndex)
{
    int32_t scatterIndex = 0;
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    uint32_t gradMask = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregRes = AscendC::MicroAPI::UpdateMask<int32_t>(gradMask);
    GetContinuousInput(gradReg, gradAddr, gradOffset);
    int32_t dhwPlaneSize = hOutputActual * wOutputActual * cOutputAligned;
    int32_t hwPlaneSize = wOutputActual * cOutputAligned;
    int32_t scatterStartIndex = nOffset + dIndex * dhwPlaneSize + hIndex * hwPlaneSize + wIndex * cOutputAligned +
                                cOffset;

    for (uint16_t dIdx = dkStart; dIdx < dkEnd; dIdx++) {
        int32_t dKernelOffset = dIdx * dhwPlaneSize;
        for (uint16_t hIdx = hkStart; hIdx < hkEnd; hIdx++) {
            int32_t hKernelOffset = hIdx * hwPlaneSize;
            for (uint16_t wIdx = wkStart; wIdx < wkEnd; wIdx++) {
                int32_t scatterIndexOffsetTotal = dKernelOffset + hKernelOffset + wIdx * cOutputAligned;
                scatterIndex = scatterIndexOffsetTotal + scatterStartIndex;
                GradientAccBigC(yAddr, gradReg, scatterIndex, divisorReg, pregRes);
            }
        }
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoMulCNdhwc(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                   MicroAPI::RegTensor<uint32_t>& parallelRegIndex, uint32_t gradMaskCount,
                                   int32_t nOffset, int32_t wOutputActual, int32_t hOutputActual, int32_t dOutputActual,
                                   int32_t cOutputAligned, MicroAPI::RegTensor<int32_t>& zeroConstReg,
                                   MicroAPI::RegTensor<int32_t>& wMaxReg, MicroAPI::RegTensor<int32_t>& hMaxReg,
                                   MicroAPI::RegTensor<int32_t>& dMaxReg, uint16_t kD, uint16_t kH, uint16_t kW,
                                   MicroAPI::RegTensor<int32_t>& divisorReg, MicroAPI::RegTensor<int32_t>& wIndexReg,
                                   MicroAPI::RegTensor<int32_t>& hIndexReg, MicroAPI::RegTensor<int32_t>& dIndexReg,
                                   AscendC::MicroAPI::RegTensor<int32_t>& tmplWRegIdx)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterStartIdxReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> tmpReg;

    uint32_t maskT1 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    uint32_t maskI32 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregI32 = AscendC::MicroAPI::UpdateMask<int32_t>(maskI32);
    GetConCurrentInput<T1>(gradReg, gradAddr, parallelRegIndex, pregT1);

    AscendC::MicroAPI::Muls(scatterStartIdxReg, dIndexReg, hOutputActual * wOutputActual * cOutputAligned, pregI32);
    AscendC::MicroAPI::Muls(tmpReg, hIndexReg, wOutputActual * cOutputAligned, pregI32);
    AscendC::MicroAPI::Add(scatterStartIdxReg, scatterStartIdxReg, tmpReg, pregI32);
    AscendC::MicroAPI::Add(scatterStartIdxReg, scatterStartIdxReg, wIndexReg, pregI32);

    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                uint32_t gradMask = gradMaskCount;
                AscendC::MicroAPI::MaskReg pregRes = AscendC::MicroAPI::UpdateMask<int32_t>(gradMask);

                int32_t scatterIndexOffsetTotal = nOffset + dIdx * hOutputActual * wOutputActual * cOutputAligned +
                                                  hIdx * wOutputActual * cOutputAligned + wIdx * cOutputAligned;
                AscendC::MicroAPI::Adds(scatterIndexReg, scatterStartIdxReg, scatterIndexOffsetTotal, pregRes);

                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::RegTensor<int32_t> wCurIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hCurIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dCurIndexReg;
                    AscendC::MicroAPI::Adds(wCurIndexReg, tmplWRegIdx, int32_t(wIdx), pregRes);
                    AscendC::MicroAPI::Adds(hCurIndexReg, hIndexReg, int32_t(hIdx), pregRes);
                    AscendC::MicroAPI::Adds(dCurIndexReg, dIndexReg, int32_t(dIdx), pregRes);
                    FilterMask3D(pregRes, dCurIndexReg, hCurIndexReg, wCurIndexReg, zeroConstReg, dMaxReg, hMaxReg,
                                 wMaxReg);
                }

                GradientAcc<int32_t>(yAddr, gradReg, scatterIndexReg, divisorReg, pregRes);
            }
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void GenRepeatIndices(MicroAPI::RegTensor<T, Trait>& indexReg, uint16_t repeatCount,
                                        uint16_t repeatSize)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T, Trait> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(repeatSize));
    AscendC::MicroAPI::Div(indexReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, static_cast<T>(repeatCount), preg);
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void GenRepeatIndicesWithLoop(MicroAPI::RegTensor<T, Trait>& indexReg, uint16_t repeatCount,
                                                uint16_t repeatSize, uint16_t wProBatchSize)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T, Trait> constReg;
    AscendC::MicroAPI::RegTensor<T, Trait> modReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(repeatSize));
    AscendC::MicroAPI::Div(indexReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(repeatCount));
    AscendC::MicroAPI::Div(modReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(modReg, modReg, static_cast<T>(repeatCount), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, modReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, static_cast<T>(wProBatchSize), preg);
}

template <typename T, const AscendC::MicroAPI::RegTrait& Trait = AscendC::MicroAPI::RegTraitNumOne>
__aicore__ inline void ComputeStridedIndices(MicroAPI::RegTensor<T, Trait>& outReg,
                                             MicroAPI::RegTensor<T, Trait>& inputGradRepeatReg, uint16_t cOutputActual,
                                             uint16_t cOutputAligned, int64_t curIndex, uint16_t padW)
{
    AscendC::MicroAPI::RegTensor<T, Trait> tmpReg;
    AscendC::MicroAPI::RegTensor<T, Trait> modReg;
    AscendC::MicroAPI::RegTensor<T, Trait> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
    AscendC::MicroAPI::Arange(tmpReg, 0);
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(cOutputActual));
    AscendC::MicroAPI::Div(modReg, tmpReg, constReg, preg);
    AscendC::MicroAPI::Muls(modReg, modReg, static_cast<T>(cOutputActual), preg);
    AscendC::MicroAPI::Sub(tmpReg, tmpReg, modReg, preg);
    AscendC::MicroAPI::Muls(outReg, inputGradRepeatReg, static_cast<T>(cOutputAligned), preg);
    AscendC::MicroAPI::Add(outReg, outReg, tmpReg, preg);
    AscendC::MicroAPI::Adds(inputGradRepeatReg, inputGradRepeatReg, static_cast<T>(-curIndex - padW), preg);
}

template <typename T, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void ComputeOutDHWIndex(MicroAPI::RegTensor<int32_t>& wIndexReg,
                                          MicroAPI::RegTensor<int32_t>& hIndexReg,
                                          MicroAPI::RegTensor<int32_t>& dIndexReg,
                                          MicroAPI::RegTensor<T, Trait>& outWStart,
                                          MicroAPI::RegTensor<T, Trait>& outHStart,
                                          MicroAPI::RegTensor<T, Trait>& outDStart, int64_t curWIndex,
                                          int64_t curHIndex, int64_t curDIndex, uint16_t cOutputAligned, uint16_t padD,
                                          uint16_t padH, uint16_t padW, uint32_t count)
{
    AscendC::MicroAPI::RegTensor<T, Trait> wIndexRegT;
    AscendC::MicroAPI::RegTensor<T, Trait> hIndexRegT;
    AscendC::MicroAPI::RegTensor<T, Trait> dIndexRegT;
    AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<T, Trait>(count);
    AscendC::MicroAPI::Adds(wIndexRegT, outWStart, static_cast<T>(-(curWIndex + padW) * cOutputAligned), maskT);
    AscendC::MicroAPI::Adds(hIndexRegT, outHStart, static_cast<T>(-curHIndex - padH), maskT);
    AscendC::MicroAPI::Adds(dIndexRegT, outDStart, static_cast<T>(-curDIndex - padD), maskT);
    wIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)wIndexRegT;
    hIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)hIndexRegT;
    dIndexReg = (AscendC::MicroAPI::RegTensor<int32_t>&)dIndexRegT;
}

template <typename T>
__aicore__ inline void GenInitial3DIndicesForNDHWC(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate,
                                                   int64_t rowGenRate, int64_t colNum, int64_t fullBatchColNum,
                                                   int64_t cOutputActual, int64_t cOutputAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg2;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(fullBatchColNum * cOutputActual));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, static_cast<T>(fullBatchColNum * cOutputActual), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, static_cast<T>(rowGenRate * colNum * cOutputAligned),
                            preg);
    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(cOutputActual));
    AscendC::MicroAPI::Div(segmentScalarReg2, segmentIncReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg2, segmentScalarReg2, static_cast<T>(cOutputActual), preg);
    AscendC::MicroAPI::Sub(segmentIncReg2, segmentIncReg, segmentIncReg2, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, static_cast<T>(colGenRate * cOutputAligned), preg);
    AscendC::MicroAPI::Add(indexReg, segmentIncReg2, segmentScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void Gen3DIndexOneForNDHWC(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t colNum,
                                             int64_t cOutputActual, int64_t cOutputAligned)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentIncReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(cOutputActual));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(segmentIncReg, segmentScalarReg, T(cOutputActual), preg);
    AscendC::MicroAPI::Sub(segmentIncReg, indexReg, segmentIncReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rowGenRate * colNum * cOutputAligned), preg);
    AscendC::MicroAPI::Add(indexReg, segmentScalarReg, segmentIncReg, preg);
}

template <typename T>
__aicore__ inline void GenInitial4DIndicesForNDHWC(MicroAPI::RegTensor<T>& indexReg, int64_t colGenRate,
                                                   int64_t rowGenRate, int64_t depthGenRate, int64_t colNum,
                                                   int64_t fullBatchColNum, int64_t fullBatchRowNum,
                                                   int64_t cOutputActual, int64_t cOutputAligned, int64_t rowNum)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segD;
    AscendC::MicroAPI::RegTensor<T> remD;
    AscendC::MicroAPI::RegTensor<T> segH;
    AscendC::MicroAPI::RegTensor<T> remH;
    AscendC::MicroAPI::RegTensor<T> segW;
    AscendC::MicroAPI::RegTensor<T> remW;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(fullBatchRowNum * fullBatchColNum * cOutputActual));
    AscendC::MicroAPI::Div(segD, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(remD, segD, static_cast<T>(fullBatchRowNum * fullBatchColNum * cOutputActual), preg);
    AscendC::MicroAPI::Sub(remD, indexReg, remD, preg);
    AscendC::MicroAPI::Muls(segD, segD, static_cast<T>(depthGenRate * rowNum * colNum * cOutputAligned), preg);

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(fullBatchColNum * cOutputActual));
    AscendC::MicroAPI::Div(segH, remD, constReg, preg);
    AscendC::MicroAPI::Muls(remH, segH, static_cast<T>(fullBatchColNum * cOutputActual), preg);
    AscendC::MicroAPI::Sub(remH, remD, remH, preg);
    AscendC::MicroAPI::Muls(segH, segH, static_cast<T>(rowGenRate * colNum * cOutputAligned), preg);

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(cOutputActual));
    AscendC::MicroAPI::Div(segW, remH, constReg, preg);
    AscendC::MicroAPI::Muls(remW, segW, static_cast<T>(cOutputActual), preg);
    AscendC::MicroAPI::Sub(remW, remH, remW, preg);
    AscendC::MicroAPI::Muls(segW, segW, static_cast<T>(colGenRate * cOutputAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segD, segH, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segW, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, remW, preg);
}

template <typename T>
__aicore__ inline void Gen4DIndexOneForNDHWC(MicroAPI::RegTensor<T>& indexReg, int64_t rowGenRate, int64_t depthGenRate,
                                             int64_t colNum, int64_t fullBatchRowNum, int64_t cOutputActual,
                                             int64_t cOutputAligned, int64_t rowNum)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segD;
    AscendC::MicroAPI::RegTensor<T> remD;
    AscendC::MicroAPI::RegTensor<T> segH;
    AscendC::MicroAPI::RegTensor<T> remH;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(fullBatchRowNum * cOutputActual));
    AscendC::MicroAPI::Div(segD, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(remD, segD, static_cast<T>(fullBatchRowNum * cOutputActual), preg);
    AscendC::MicroAPI::Sub(remD, indexReg, remD, preg);
    AscendC::MicroAPI::Muls(segD, segD, static_cast<T>(depthGenRate * rowNum * colNum * cOutputAligned), preg);

    AscendC::MicroAPI::Duplicate(constReg, static_cast<T>(cOutputActual));
    AscendC::MicroAPI::Div(segH, remD, constReg, preg);
    AscendC::MicroAPI::Muls(remH, segH, static_cast<T>(cOutputActual), preg);
    AscendC::MicroAPI::Sub(remH, remD, remH, preg);
    AscendC::MicroAPI::Muls(segH, segH, static_cast<T>(rowGenRate * colNum * cOutputAligned), preg);

    AscendC::MicroAPI::Add(indexReg, segD, segH, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, remH, preg);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
class AvgPool3DGradNDHWC {
public:
    __aicore__ inline AvgPool3DGradNDHWC(TPipe* pipe, const AvgPool3DGradNDHWCTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void ConCProcVF3D(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr);
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void ConCMergeWProcVF3D(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                              __local_mem__ uint32_t* helpAddr, __local_mem__ T3* helpAddrT3);
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void ConCMergeHWProcVF3D(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                               __local_mem__ uint32_t* helpAddr, __local_mem__ T3* helpAddrT3);
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void ConCMergeDHWCProcVF3D(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                 __local_mem__ uint32_t* helpAddr, __local_mem__ T3* helpAddrT3);
    __aicore__ inline void ProcessNoGradBlock();
    __aicore__ inline void CopyOut();

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> helpBuf_;
    TBuf<QuePosition::VECCALC> helpBufT3_;

    GlobalTensor<T1> yGm_;
    GlobalTensor<T1> gradGm_;
    const AvgPool3DGradNDHWCTilingData* tilingData_;

    uint32_t blockIdx_ = 0;

    int64_t nOutputActual_ = 1;
    int64_t dOutputActual_ = 1;
    int64_t hOutputActual_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t cOutputActual_ = 1;
    int64_t cOutputAligned_ = 1;

    int64_t nAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t cAxisIndex_ = 0;

    int64_t dGradActual_ = 0;
    int64_t hGradActual_ = 0;
    int64_t wGradActual_ = 0;

    int64_t nOutputGradOffset_ = 0;
    int64_t dAxisGradOffset_ = 0;
    int64_t hAxisGradOffset_ = 0;
    int64_t wAxisGradOffset_ = 0;
    int64_t cAxisGradOffset_ = 0;
    int64_t dGradActualStart_ = 0;
    int64_t hGradActualStart_ = 0;
    int64_t wGradActualStart_ = 0;

    int64_t gradPlaneSize_ = 1;
    int64_t outputPlaneSize_ = 1;
    int64_t curDProBatchSize_ = 1;
    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;
    int64_t curCoreProcessNum_ = 1;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T1);
};

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Init(GM_ADDR grad, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    gradPlaneSize_ = tilingData_->dGrad * tilingData_->hGrad * tilingData_->wGrad;
    outputPlaneSize_ = tilingData_->dOutput * tilingData_->hOutput * tilingData_->wOutput;
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    curCoreProcessNum_ = (blockIdx_ + 1 == tilingData_->usedCoreNum) ? tilingData_->tailCoreProcessNum :
                                                                       tilingData_->normalCoreProcessNum;
    gradGm_.SetGlobalBuffer((__gm__ T1*)grad);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);
    pipe_->InitBuffer(outputQue_, BUFFER_NUM, tilingData_->outputBufferSize);
    pipe_->InitBuffer(gradQue_, BUFFER_NUM, tilingData_->inputGradBufferSize);
    pipe_->InitBuffer(helpBuf_, HELP_BUFFER);
    pipe_->InitBuffer(helpBufT3_, HELP_BUFFER_T3_NDHWC);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Process()
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
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ScalarCompute(
    int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tilingData_->normalCoreProcessNum + loopNum;
    int64_t dhwcOuter = tilingData_->dOutputOuter * tilingData_->hOutputOuter * tilingData_->wOutputOuter *
                        tilingData_->cOutputOuter;
    nAxisIndex_ = baseBlockIdx / dhwcOuter;
    nOutputActual_ = nAxisIndex_ == (tilingData_->nOutputOuter - 1) ? tilingData_->nOutputTail :
                                                                      tilingData_->nOutputInner;

    int64_t tempNTail = baseBlockIdx % dhwcOuter;
    cAxisIndex_ = tempNTail / (tilingData_->dOutputOuter * tilingData_->hOutputOuter * tilingData_->wOutputOuter);
    cOutputActual_ = cAxisIndex_ == (tilingData_->cOutputOuter - 1) ? tilingData_->cOutputTail :
                                                                      tilingData_->cOutputInner;
    cOutputAligned_ = (cOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK *
                      MAX_DATA_NUM_IN_ONE_BLOCK;

    int64_t tempCTail = tempNTail % (tilingData_->dOutputOuter * tilingData_->hOutputOuter * tilingData_->wOutputOuter);
    dAxisIndex_ = tempCTail / (tilingData_->hOutputOuter * tilingData_->wOutputOuter);
    dOutputActual_ = dAxisIndex_ == (tilingData_->dOutputOuter - 1) ? tilingData_->dOutputTail :
                                                                      tilingData_->dOutputInner;

    int64_t tempDTail = tempCTail % (tilingData_->hOutputOuter * tilingData_->wOutputOuter);
    hAxisIndex_ = tempDTail / tilingData_->wOutputOuter;
    hOutputActual_ = hAxisIndex_ == (tilingData_->hOutputOuter - 1) ? tilingData_->hOutputTail :
                                                                      tilingData_->hOutputInner;

    wAxisIndex_ = tempDTail % tilingData_->wOutputOuter;
    wOutputActual_ = wAxisIndex_ == (tilingData_->wOutputOuter - 1) ? tilingData_->wOutputTail :
                                                                      tilingData_->wOutputInner;

    dGradActualStart_ = PStart(dAxisIndex_ * tilingData_->dOutputInner, tilingData_->padFront, tilingData_->dKernel,
                               tilingData_->dStride);
    int64_t dGradActualEnd = PEnd(dAxisIndex_ * tilingData_->dOutputInner + dOutputActual_ - 1, tilingData_->padFront,
                                  tilingData_->dStride, tilingData_->dGrad);
    hGradActualStart_ = PStart(hAxisIndex_ * tilingData_->hOutputInner, tilingData_->padTop, tilingData_->hKernel,
                               tilingData_->hStride);
    int64_t hGradActualEnd = PEnd(hAxisIndex_ * tilingData_->hOutputInner + hOutputActual_ - 1, tilingData_->padTop,
                                  tilingData_->hStride, tilingData_->hGrad);
    wGradActualStart_ = PStart(wAxisIndex_ * tilingData_->wOutputInner, tilingData_->padLeft, tilingData_->wKernel,
                               tilingData_->wStride);
    int64_t wGradActualEnd = PEnd(wAxisIndex_ * tilingData_->wOutputInner + wOutputActual_ - 1, tilingData_->padLeft,
                                  tilingData_->wStride, tilingData_->wGrad);
    dGradActual_ = dGradActualEnd - dGradActualStart_;
    hGradActual_ = hGradActualEnd - hGradActualStart_;
    wGradActual_ = wGradActualEnd - wGradActualStart_;

    curDProBatchSize_ = tilingData_->dProBatchSize > dGradActual_ ? dGradActual_ : tilingData_->dProBatchSize;
    curHProBatchSize_ = tilingData_->hProBatchSize > hGradActual_ ? hGradActual_ : tilingData_->hProBatchSize;
    curWProBatchSize_ = tilingData_->wProBatchSize > wGradActual_ ? wGradActual_ : tilingData_->wProBatchSize;

    nOutputGradOffset_ = nAxisIndex_ * tilingData_->nOutputInner * gradPlaneSize_ * tilingData_->cOutput;
    dAxisGradOffset_ = dGradActualStart_ * tilingData_->hGrad * tilingData_->wGrad * tilingData_->cOutput;
    hAxisGradOffset_ = hGradActualStart_ * tilingData_->wGrad * tilingData_->cOutput;
    wAxisGradOffset_ = wGradActualStart_ * tilingData_->cOutput;
    cAxisGradOffset_ = cAxisIndex_ * tilingData_->cOutputInner;
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ProcessPerLoop()
{
    if (dGradActual_ <= 0 || hGradActual_ <= 0 || wGradActual_ <= 0) {
        ProcessNoGradBlock();
        return;
    }
    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    int64_t gradGmOffset = nOutputGradOffset_ + dAxisGradOffset_ + hAxisGradOffset_ + wAxisGradOffset_ +
                           cAxisGradOffset_;
    MultiCopyLoopInfo<DIGIT_FIVE> loopInfo;
    loopInfo.loopSize[ZERO] = cOutputActual_;
    loopInfo.loopSize[DIGIT_ONE] = wGradActual_;
    loopInfo.loopSize[DIGIT_TWO] = hGradActual_;
    loopInfo.loopSize[DIGIT_THREE] = dGradActual_;
    loopInfo.loopSize[DIGIT_FOUR] = nOutputActual_;
    loopInfo.loopSrcStride[ZERO] = 1;
    loopInfo.loopSrcStride[DIGIT_ONE] = tilingData_->cOutput;
    loopInfo.loopSrcStride[DIGIT_TWO] = tilingData_->wGrad * tilingData_->cOutput;
    loopInfo.loopSrcStride[DIGIT_THREE] = tilingData_->hGrad * tilingData_->wGrad * tilingData_->cOutput;
    loopInfo.loopSrcStride[DIGIT_FOUR] = gradPlaneSize_ * tilingData_->cOutput;
    loopInfo.loopDstStride[ZERO] = 1;
    loopInfo.loopDstStride[DIGIT_ONE] = cOutputAligned_;
    loopInfo.loopDstStride[DIGIT_TWO] = wGradActual_ * cOutputAligned_;
    loopInfo.loopDstStride[DIGIT_THREE] = hGradActual_ * wGradActual_ * cOutputAligned_;
    loopInfo.loopDstStride[DIGIT_FOUR] = dGradActual_ * hGradActual_ * wGradActual_ * cOutputAligned_;
    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<T1, DIGIT_FIVE> paramsMain = {loopInfo};
    DataCopy<T1, DIGIT_FIVE, config>(gradLocal, gradGm_[gradGmOffset], paramsMain);
    gradQue_.EnQue(gradLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Compute()
{
    uint32_t calCount = tilingData_->outputBufferSize / sizeof(computeType);
    LocalTensor<computeType> yLocal = outputQue_.AllocTensor<computeType>();
    Duplicate(yLocal, computeType(0), calCount);
    LocalTensor<T1> gradLocal = gradQue_.DeQue<T1>();

    __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
    __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
    LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();
    LocalTensor<T3> helpTensorT3 = helpBufT3_.Get<T3>();
    __local_mem__ T3* helpAddrT3 = (__local_mem__ T3*)helpTensorT3.GetPhyAddr();

    uint16_t computeSize = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSize / cOutputActual_;
    if (concurrencyCount < 2) {
        ConCProcVF3D<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr);
    } else {
        uint32_t wFullBatchCount = wGradActual_ / curWProBatchSize_;
        uint16_t hConcurrentCount = concurrencyCount / wFullBatchCount;
        if (hConcurrentCount < 2) {
            if constexpr (std::is_same<T3, int64_t>::value) {
                ConCMergeWProcVF3D<AscendC::MicroAPI::RegTraitNumTwo>(yAddr, gradAddr, helpAddr, helpAddrT3);
            } else {
                ConCMergeWProcVF3D<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr, helpAddr, helpAddrT3);
            }
        } else {
            uint16_t hFullBatchCount = static_cast<uint16_t>(hGradActual_ / curHProBatchSize_);
            uint16_t dConcurrentCount = hConcurrentCount / hFullBatchCount;
            if (dConcurrentCount >= 2) {
                if constexpr (std::is_same<T3, int64_t>::value) {
                    ConCMergeDHWCProcVF3D<AscendC::MicroAPI::RegTraitNumTwo>(yAddr, gradAddr, helpAddr, helpAddrT3);
                } else {
                    ConCMergeDHWCProcVF3D<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr, helpAddr, helpAddrT3);
                }
            } else {
                if constexpr (std::is_same<T3, int64_t>::value) {
                    ConCMergeHWProcVF3D<AscendC::MicroAPI::RegTraitNumTwo>(yAddr, gradAddr, helpAddr, helpAddrT3);
                } else {
                    ConCMergeHWProcVF3D<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr, helpAddr, helpAddrT3);
                }
            }
        }
    }

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }
    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = hOutputActual_;
    loopModeParamsT1.loop2Size = dOutputActual_;
    loopModeParamsT1.loop1SrcStride = wOutputActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = hOutputActual_ * wOutputActual_ * cOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = tilingData_->wOutput * tilingData_->cOutput * sizeof(T1);
    loopModeParamsT1.loop2DstStride = tilingData_->hOutput * tilingData_->wOutput * tilingData_->cOutput * sizeof(T1);

    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(wOutputActual_),
                                        static_cast<uint32_t>(cOutputActual_ * sizeof(T1)), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((tilingData_->cOutput - cOutputActual_) * sizeof(T1)),
                                        static_cast<uint32_t>(0)};

    for (int64_t nIdx = 0; nIdx < nOutputActual_; nIdx++) {
        int64_t nOutputAxisOffset = (nAxisIndex_ * tilingData_->nOutputInner + nIdx) * outputPlaneSize_ *
                                    tilingData_->cOutput;
        int64_t dOutputAxisOffset = dAxisIndex_ * tilingData_->dOutputInner * tilingData_->hOutput *
                                    tilingData_->wOutput * tilingData_->cOutput;
        int64_t hOutputAxisOffset = hAxisIndex_ * tilingData_->hOutputInner * tilingData_->wOutput *
                                    tilingData_->cOutput;
        int64_t wOutputAxisOffset = wAxisIndex_ * tilingData_->wOutputInner * tilingData_->cOutput;
        int64_t cOutputAxisOffset = cAxisIndex_ * tilingData_->cOutputInner;
        int64_t outputGmOffset = nOutputAxisOffset + dOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset +
                                 cOutputAxisOffset;
        int64_t yLocalOffset = nIdx * dOutputActual_ * hOutputActual_ * wOutputActual_ * cOutputAligned_;
        SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
        DataCopyPad(yGm_[outputGmOffset], yLocal[yLocalOffset], copyOutParamT1);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
    outputQue_.FreeTensor(yLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ProcessNoGradBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(tilingData_->outputBufferSize) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ConCProcVF3D(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr)
{
    int32_t dOutput = static_cast<int32_t>(tilingData_->dOutput);
    int32_t hOutput = static_cast<int32_t>(tilingData_->hOutput);
    int32_t wOutput = static_cast<int32_t>(tilingData_->wOutput);
    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int32_t dOutputActual = static_cast<int32_t>(dOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t curDIndex = static_cast<int32_t>(dAxisIndex_ * tilingData_->dOutputInner);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_->hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_->wOutputInner);
    uint16_t dGradActual = static_cast<uint16_t>(dGradActual_);
    uint16_t hGradActual = static_cast<uint16_t>(hGradActual_);
    uint16_t wGradActual = static_cast<uint16_t>(wGradActual_);
    uint16_t cOutputAligned = cOutputAligned_;
    uint16_t cOutputActual = cOutputActual_;
    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFront);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBack);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTop);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padBottom);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeft);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRight);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t computeSizeFP32 = V_REG_SIZE / sizeof(float);
    uint16_t cRepeatimes = cOutputActual / computeSizeFP32;
    uint16_t cRemain = cOutputActual - cRepeatimes * computeSizeFP32;
    uint16_t cRemainLoopTimes = cRemain == 0 ? 0 : 1;

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nOffset = nIdx * dOutputActual * hOutputActual * wOutputActual * cOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dGradActual; ++dIdx) {
            T3 dGradOffset = dIdx + dGradActualStart;
            T3 dGradOffsetMulStrideD = dGradOffset * strideD;
            for (uint16_t hIdx = 0; hIdx < hGradActual; ++hIdx) {
                T3 hGradOffset = hIdx + hGradActualStart;
                T3 hGradOffsetMulStrideH = hGradOffset * strideH;
                for (uint16_t wIdx = 0; wIdx < wGradActual; ++wIdx) {
                    T3 wGradOffset = wIdx + wGradActualStart;
                    T3 wGradOffsetMulStrideW = wGradOffset * strideW;
                    T3 dIndex = dGradOffsetMulStrideD - curDIndex - padD;
                    T3 hIndex = hGradOffsetMulStrideH - curHIndex - padH;
                    T3 wIndex = wGradOffsetMulStrideW - curWIndex - padW;
                    int32_t dkStart = dIndex > 0 ? 0 : (-dIndex);
                    int32_t dkEnd = (dOutputActual - dIndex) > kD ? kD : (dOutputActual - dIndex);
                    int32_t hkStart = hIndex > 0 ? 0 : (-hIndex);
                    int32_t hkEnd = (hOutputActual - hIndex) > kH ? kH : (hOutputActual - hIndex);
                    int32_t wkStart = wIndex > 0 ? 0 : (-wIndex);
                    int32_t wkEnd = (wOutputActual - wIndex) > kW ? kW : (wOutputActual - wIndex);

                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                        AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                        AscendC::MicroAPI::Duplicate(outDStart, dGradOffsetMulStrideD);
                        AscendC::MicroAPI::Duplicate(outHStart, hGradOffsetMulStrideH);
                        AscendC::MicroAPI::Duplicate(outWStart, wGradOffsetMulStrideW);
                        AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride,
                            uint32_t(computeSizeFP32));

                        for (uint16_t cIdx = 0; cIdx < cRepeatimes; ++cIdx) {
                            uint32_t cOffset = cIdx * computeSizeFP32;
                            uint32_t gradOffset = (((nIdx * dGradActual + dIdx) * hGradActual + hIdx) * wGradActual +
                                                   wIdx) *
                                                      cOutputAligned +
                                                  cOffset;
                            DoSingleCNdhwc<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                                yAddr, gradAddr, gradOffset, computeSizeFP32, dOutputActual, hOutputActual,
                                wOutputActual, cOutputAligned, cOffset, nOffset, dkStart, dkEnd, hkStart, hkEnd,
                                wkStart, wkEnd, divisorReg, dIndex, hIndex, wIndex);
                        }
                        for (uint16_t cIdx = 0; cIdx < cRemainLoopTimes; ++cIdx) {
                            uint32_t cOffset = cRepeatimes * computeSizeFP32;
                            uint32_t gradOffset = (((nIdx * dGradActual + dIdx) * hGradActual + hIdx) * wGradActual +
                                                   wIdx) *
                                                      cOutputAligned +
                                                  cOffset;
                            DoSingleCNdhwc<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                                yAddr, gradAddr, gradOffset, cRemain, dOutputActual, hOutputActual, wOutputActual,
                                cOutputAligned, cOffset, nOffset, dkStart, dkEnd, hkStart, hkEnd, wkStart, wkEnd,
                                divisorReg, dIndex, hIndex, wIndex);
                        }
                    }
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ConCMergeWProcVF3D(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3)
{
    int32_t dOutput = static_cast<int32_t>(tilingData_->dOutput);
    int32_t hOutput = static_cast<int32_t>(tilingData_->hOutput);
    int32_t wOutput = static_cast<int32_t>(tilingData_->wOutput);
    uint16_t cOutputActual = cOutputActual_;
    uint16_t cOutputAligned = cOutputAligned_;
    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t dOutputActual = static_cast<int32_t>(dOutputActual_);
    int32_t curDIndex = static_cast<int32_t>(dAxisIndex_ * tilingData_->dOutputInner);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_->hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_->wOutputInner);
    int32_t wGradActual = static_cast<int32_t>(wGradActual_);
    uint16_t hGradActual = static_cast<uint16_t>(hGradActual_);
    uint16_t dGradActual = static_cast<uint16_t>(dGradActual_);
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);
    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFront);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBack);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTop);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeft);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padBottom);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRight);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wGradActual / wProBatchSize;

    uint16_t computeSizeFp32 = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSizeFp32 / cOutputActual;
    uint16_t repeatimes = wFullBatchCount / concurrencyCount;
    uint16_t wRemain = wGradActual - repeatimes * wProBatchSize * concurrencyCount;
    uint32_t wRemainBatch = wRemain / wProBatchSize;
    uint16_t wRemainTail = wRemain % wProBatchSize;
    uint32_t mask0 = concurrencyCount * cOutputActual;
    uint32_t mask1 = wRemainBatch * cOutputActual;
    uint32_t mask2 = 1 * cOutputActual;

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nOffset = nIdx * dOutputActual * hOutputActual * wOutputActual * cOutputAligned;
        uint32_t nGradOffset = nIdx * dGradActual * hGradActual * wGradActual * cOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dGradActual; dIdx++) {
            for (uint16_t hIdx = 0; hIdx < hGradActual; hIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                    AscendC::MicroAPI::MaskReg
                        allMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::MaskReg
                        allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                    GenInitial3DIndicesForNDHWC<int32_t>((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex,
                                                         wProBatchSize, 1, wGradActual, wFullBatchCount, cOutputActual,
                                                         cOutputAligned);
                    GenRepeatIndices<T3, Trait>(initialWRegIdx, wProBatchSize, cOutputActual);
                    AscendC::MicroAPI::DataCopy(helpAddr, initialRegIndex, allMask);
                    AscendC::MicroAPI::DataCopy(helpAddrT3, initialWRegIdx, allMaskT3);
                }

                T3 dGradOffset = dIdx + dGradActualStart;
                T3 hGradOffset = hIdx + hGradActualStart;
                T3 dGradOffsetMulStrideD = dGradOffset * strideD;
                T3 hGradOffsetMulStrideH = hGradOffset * strideH;
                uint32_t dhwGradBase = dIdx * hGradActual * wGradActual + hIdx * wGradActual;

                for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize + wGradActualStart;
                        uint32_t offset = (wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize + dhwGradBase) *
                                              cOutputAligned +
                                          nGradOffset;
                        __VEC_SCOPE__
                        {
                            AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                            AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                            AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                            AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                            if constexpr (IS_CHECK_RANGE == 1) {
                                AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));
                                AscendC::MicroAPI::Duplicate(wMaxReg, wOutputActual);
                                AscendC::MicroAPI::Duplicate(hMaxReg, hOutputActual);
                                AscendC::MicroAPI::Duplicate(dMaxReg, dOutputActual);
                            }
                            AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                            AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                            AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                            AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                            AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                            AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                            AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                            AscendC::MicroAPI::RegTensor<T3, Trait> tmplWRegIdx;
                            AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                            AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                            AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                            AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                            if constexpr (COUNT_PAD == 0) {
                                AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
                            }
                            AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<
                                uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                            AscendC::MicroAPI::MaskReg allMaskT3 = AscendC::MicroAPI::CreateMask<
                                T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                            AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                            AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMask);
                            AscendC::MicroAPI::Adds(tmplWRegIdx, initialWRegIdx, wGradOffset, allMaskT3);
                            AscendC::MicroAPI::Muls(tmplWRegIdx, tmplWRegIdx, strideW, allMaskT3);
                            AscendC::MicroAPI::Duplicate(outDStart, dGradOffsetMulStrideD);
                            AscendC::MicroAPI::Duplicate(outHStart, hGradOffsetMulStrideH);
                            GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                                divisorReg, outDStart, outHStart, tmplWRegIdx, zeroConstRegT, dOutput, hOutput, wOutput,
                                padD, padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask0);
                            ComputeStridedIndices<T3, Trait>(outWStart, tmplWRegIdx, cOutputActual, cOutputAligned,
                                                             curWIndex, padW);
                            ComputeOutDHWIndex<T3, Trait>(wIndexReg, hIndexReg, dIndexReg, outWStart, outHStart,
                                                          outDStart, curWIndex, curHIndex, curDIndex, cOutputAligned,
                                                          padD, padH, padW, mask0);
                            DoMulCNdhwc<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask0, nOffset,
                                                            wOutputActual, hOutputActual, dOutputActual, cOutputAligned,
                                                            zeroConstReg, wMaxReg, hMaxReg, dMaxReg, kD, kH, kW,
                                                            divisorReg, wIndexReg, hIndexReg, dIndexReg,
                                                            (AscendC::MicroAPI::RegTensor<int32_t>&)tmplWRegIdx);
                        }
                    }
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + repeatimes * concurrencyCount * wProBatchSize + wGradActualStart;
                    uint32_t offset = (wBatchIdx + repeatimes * concurrencyCount * wProBatchSize + dhwGradBase) *
                                          cOutputAligned +
                                      nGradOffset;
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));
                            AscendC::MicroAPI::Duplicate(wMaxReg, wOutputActual);
                            AscendC::MicroAPI::Duplicate(hMaxReg, hOutputActual);
                            AscendC::MicroAPI::Duplicate(dMaxReg, dOutputActual);
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                        AscendC::MicroAPI::RegTensor<T3, Trait> tmplWRegIdx;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                        if constexpr (COUNT_PAD == 0) {
                            AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
                        }
                        AscendC::MicroAPI::MaskReg
                            allMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::MaskReg
                            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                        AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                        AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMask);
                        AscendC::MicroAPI::Adds(tmplWRegIdx, initialWRegIdx, wGradOffset, allMaskT3);
                        AscendC::MicroAPI::Muls(tmplWRegIdx, tmplWRegIdx, strideW, allMaskT3);
                        AscendC::MicroAPI::Duplicate(outDStart, dGradOffsetMulStrideD);
                        AscendC::MicroAPI::Duplicate(outHStart, hGradOffsetMulStrideH);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, tmplWRegIdx, zeroConstRegT, dOutput, hOutput, wOutput,
                            padD, padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask1);
                        ComputeStridedIndices<T3, Trait>(outWStart, tmplWRegIdx, cOutputActual, cOutputAligned,
                                                         curWIndex, padW);
                        ComputeOutDHWIndex<T3, Trait>(wIndexReg, hIndexReg, dIndexReg, outWStart, outHStart, outDStart,
                                                      curWIndex, curHIndex, curDIndex, cOutputAligned, padD, padH, padW,
                                                      mask1);
                        DoMulCNdhwc<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask1, nOffset,
                                                        wOutputActual, hOutputActual, dOutputActual, cOutputAligned,
                                                        zeroConstReg, wMaxReg, hMaxReg, dMaxReg, kD, kH, kW, divisorReg,
                                                        wIndexReg, hIndexReg, dIndexReg,
                                                        (AscendC::MicroAPI::RegTensor<int32_t>&)tmplWRegIdx);
                    }
                }

                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));
                        AscendC::MicroAPI::Duplicate(wMaxReg, wOutputActual);
                        AscendC::MicroAPI::Duplicate(hMaxReg, hOutputActual);
                        AscendC::MicroAPI::Duplicate(dMaxReg, dOutputActual);
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> tmplWRegIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::MaskReg
                        allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                    AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                    AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);
                    AscendC::MicroAPI::Duplicate(outDStart, dGradOffsetMulStrideD);
                    AscendC::MicroAPI::Duplicate(outHStart, hGradOffsetMulStrideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        uint32_t wGradOffset = wBatchIdx + wRemainBatch * wProBatchSize +
                                               repeatimes * concurrencyCount * wProBatchSize + wGradActualStart;
                        uint32_t offset = (wBatchIdx + wRemainBatch * wProBatchSize +
                                           repeatimes * concurrencyCount * wProBatchSize + dhwGradBase) *
                                              cOutputAligned +
                                          nGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMask);
                        AscendC::MicroAPI::Adds(tmplWRegIdx, initialWRegIdx, wGradOffset, allMaskT3);
                        AscendC::MicroAPI::Muls(tmplWRegIdx, tmplWRegIdx, strideW, allMaskT3);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, tmplWRegIdx, zeroConstRegT, dOutput, hOutput, wOutput,
                            padD, padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask2);
                        ComputeStridedIndices<T3, Trait>(outWStart, tmplWRegIdx, cOutputActual, cOutputAligned,
                                                         curWIndex, padW);
                        ComputeOutDHWIndex<T3, Trait>(wIndexReg, hIndexReg, dIndexReg, outWStart, outHStart, outDStart,
                                                      curWIndex, curHIndex, curDIndex, cOutputAligned, padD, padH, padW,
                                                      mask2);
                        DoMulCNdhwc<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask2, nOffset,
                                                        wOutputActual, hOutputActual, dOutputActual, cOutputAligned,
                                                        zeroConstReg, wMaxReg, hMaxReg, dMaxReg, kD, kH, kW, divisorReg,
                                                        wIndexReg, hIndexReg, dIndexReg,
                                                        (AscendC::MicroAPI::RegTensor<int32_t>&)tmplWRegIdx);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD,
          const MicroAPI::RegTrait& Trait, bool UseOneIndex>
__aicore__ inline void DoMergeHWBlock3D(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3, uint32_t mask, uint32_t nOffset, uint32_t nGradOffset, uint32_t dhwGradBase,
    uint32_t wGradActualStart, T3 dGradOffsetMulStrideD, T3 hGradOffset, T3 wGradOffset, int32_t wOutputActual,
    int32_t hOutputActual, int32_t dOutputActual, uint16_t cOutputAligned, uint16_t cOutputActual, int32_t dOutput,
    int32_t hOutput, int32_t wOutput, uint16_t kD, uint16_t kH, uint16_t kW, uint16_t padD, uint16_t padH,
    uint16_t padW, uint16_t padBackD, uint16_t padDownH, uint16_t padRightW, int32_t divisorOverride, uint32_t strideD,
    uint32_t strideH, uint32_t strideW, int32_t curDIndex, int32_t curHIndex, int32_t curWIndex)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, wOutputActual);
            AscendC::MicroAPI::Duplicate(hMaxReg, hOutputActual);
            AscendC::MicroAPI::Duplicate(dMaxReg, dOutputActual);
        }
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
        AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
        AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
        AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
        AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> tmplWRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
        AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
        AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
        AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
        if constexpr (COUNT_PAD == 0) {
            AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
        }
        AscendC::MicroAPI::MaskReg
            allMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        if constexpr (UseOneIndex) {
            AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr + (V_REG_SIZE / sizeof(uint32_t)));
            AscendC::MicroAPI::DataCopy(initialWRegIdx,
                                        helpAddrT3 + INDEX_THREE * INDEX_TWO * (V_REG_SIZE / sizeof(T3)));
        } else {
            AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
            AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3 + INDEX_TWO * (V_REG_SIZE / sizeof(T3)));
        }
        uint32_t wGradLocalOffset = static_cast<uint32_t>(wGradOffset) - wGradActualStart;
        uint32_t offset = (wGradLocalOffset + dhwGradBase) * cOutputAligned + nGradOffset;
        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMask);
        AscendC::MicroAPI::Adds(tmplWRegIdx, initialWRegIdx, wGradOffset, allMaskT3);
        AscendC::MicroAPI::Muls(tmplWRegIdx, tmplWRegIdx, strideW, allMaskT3);
        AscendC::MicroAPI::Duplicate(outDStart, dGradOffsetMulStrideD);
        if constexpr (UseOneIndex) {
            AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
            AscendC::MicroAPI::DataCopy(initialHRegIdxOne,
                                        helpAddrT3 + INDEX_TWO * INDEX_TWO * (V_REG_SIZE / sizeof(T3)));
            AscendC::MicroAPI::Adds(outHStart, initialHRegIdxOne, hGradOffset, allMaskT3);
        } else {
            AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
            AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3);
            AscendC::MicroAPI::Adds(outHStart, initialHRegIdx, hGradOffset, allMaskT3);
        }
        AscendC::MicroAPI::Muls(outHStart, outHStart, strideH, allMaskT3);
        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
            divisorReg, outDStart, outHStart, tmplWRegIdx, zeroConstRegT, dOutput, hOutput, wOutput, padD, padH, padW,
            padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask);
        ComputeStridedIndices<T3, Trait>(outWStart, tmplWRegIdx, cOutputActual, cOutputAligned, curWIndex, padW);
        ComputeOutDHWIndex<T3, Trait>(wIndexReg, hIndexReg, dIndexReg, outWStart, outHStart, outDStart, curWIndex,
                                      curHIndex, curDIndex, cOutputAligned, padD, padH, padW, mask);
        DoMulCNdhwc<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask, nOffset, wOutputActual, hOutputActual,
                                        dOutputActual, cOutputAligned, zeroConstReg, wMaxReg, hMaxReg, dMaxReg, kD, kH,
                                        kW, divisorReg, wIndexReg, hIndexReg, dIndexReg,
                                        (AscendC::MicroAPI::RegTensor<int32_t>&)tmplWRegIdx);
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD,
          const MicroAPI::RegTrait& Trait, uint8_t IndexMode>
__aicore__ inline void DoMergeDHWCBlock3D(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3, uint32_t mask, uint32_t nOffset, uint32_t nGradOffset, uint32_t dhwGradBase,
    uint32_t wGradActualStart, T3 dGradOffset, T3 hGradOffset, T3 wGradOffset, int32_t wOutputActual,
    int32_t hOutputActual, int32_t dOutputActual, uint16_t cOutputAligned, uint16_t cOutputActual, int32_t dOutput,
    int32_t hOutput, int32_t wOutput, uint16_t kD, uint16_t kH, uint16_t kW, uint16_t padD, uint16_t padH,
    uint16_t padW, uint16_t padBackD, uint16_t padDownH, uint16_t padRightW, int32_t divisorOverride, uint32_t strideD,
    uint32_t strideH, uint32_t strideW, int32_t curDIndex, int32_t curHIndex, int32_t curWIndex)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
        AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::MicroAPI::Duplicate(zeroConstReg, int32_t(0));
            AscendC::MicroAPI::Duplicate(wMaxReg, wOutputActual);
            AscendC::MicroAPI::Duplicate(hMaxReg, hOutputActual);
            AscendC::MicroAPI::Duplicate(dMaxReg, dOutputActual);
        }
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
        AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
        AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
        AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
        AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialDRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> tmplWRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
        AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
        AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
        AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
        if constexpr (COUNT_PAD == 0) {
            AscendC::MicroAPI::Duplicate(zeroConstRegT, T3(0));
        }
        AscendC::MicroAPI::MaskReg
            allMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        uint16_t vecElemCountU32 = V_REG_SIZE / sizeof(uint32_t);
        uint16_t vecElemCountT3 = V_REG_SIZE / sizeof(T3);
        if constexpr (IndexMode == 1) {
            AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr + vecElemCountU32);
            AscendC::MicroAPI::DataCopy(initialDRegIdx, helpAddrT3 + INDEX_THREE * INDEX_TWO * vecElemCountT3);
            AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3 + INDEX_FOUR * INDEX_TWO * vecElemCountT3);
            AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3 + INDEX_FIVE * INDEX_TWO * vecElemCountT3);
        } else if constexpr (IndexMode == 2) {
            AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr + INDEX_TWO * vecElemCountU32);
            AscendC::MicroAPI::DataCopy(initialDRegIdx, helpAddrT3 + INDEX_SIX * INDEX_TWO * vecElemCountT3);
            AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3 + INDEX_SEVEN * INDEX_TWO * vecElemCountT3);
            AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3 + INDEX_EIGHT * INDEX_TWO * vecElemCountT3);
        } else if constexpr (IndexMode == 3) {
            AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr + INDEX_THREE * vecElemCountU32);
            AscendC::MicroAPI::DataCopy(initialDRegIdx, helpAddrT3 + INDEX_NINE * INDEX_TWO * vecElemCountT3);
            AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3 + INDEX_TEN * INDEX_TWO * vecElemCountT3);
            AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3 + INDEX_ELEVEN * INDEX_TWO * vecElemCountT3);
        } else {
            AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
            AscendC::MicroAPI::DataCopy(initialDRegIdx, helpAddrT3);
            AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3 + INDEX_TWO * vecElemCountT3);
            AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3 + INDEX_TWO * INDEX_TWO * vecElemCountT3);
        }
        uint32_t wGradLocalOffset = static_cast<uint32_t>(wGradOffset) - wGradActualStart;
        uint32_t offset = (wGradLocalOffset + dhwGradBase) * cOutputAligned + nGradOffset;
        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMask);
        AscendC::MicroAPI::Adds(tmplWRegIdx, initialWRegIdx, wGradOffset, allMaskT3);
        AscendC::MicroAPI::Muls(tmplWRegIdx, tmplWRegIdx, strideW, allMaskT3);
        AscendC::MicroAPI::Adds(outDStart, initialDRegIdx, dGradOffset, allMaskT3);
        AscendC::MicroAPI::Muls(outDStart, outDStart, strideD, allMaskT3);
        AscendC::MicroAPI::Adds(outHStart, initialHRegIdx, hGradOffset, allMaskT3);
        AscendC::MicroAPI::Muls(outHStart, outHStart, strideH, allMaskT3);
        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
            divisorReg, outDStart, outHStart, tmplWRegIdx, zeroConstRegT, dOutput, hOutput, wOutput, padD, padH, padW,
            padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask);
        ComputeStridedIndices<T3, Trait>(outWStart, tmplWRegIdx, cOutputActual, cOutputAligned, curWIndex, padW);
        ComputeOutDHWIndex<T3, Trait>(wIndexReg, hIndexReg, dIndexReg, outWStart, outHStart, outDStart, curWIndex,
                                      curHIndex, curDIndex, cOutputAligned, padD, padH, padW, mask);
        DoMulCNdhwc<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, mask, nOffset, wOutputActual, hOutputActual,
                                        dOutputActual, cOutputAligned, zeroConstReg, wMaxReg, hMaxReg, dMaxReg, kD, kH,
                                        kW, divisorReg, wIndexReg, hIndexReg, dIndexReg,
                                        (AscendC::MicroAPI::RegTensor<int32_t>&)tmplWRegIdx);
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ConCMergeHWProcVF3D(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3)
{
    int32_t dOutput = static_cast<int32_t>(tilingData_->dOutput);
    int32_t hOutput = static_cast<int32_t>(tilingData_->hOutput);
    int32_t wOutput = static_cast<int32_t>(tilingData_->wOutput);
    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t dOutputActual = static_cast<int32_t>(dOutputActual_);
    int32_t curDIndex = static_cast<int32_t>(dAxisIndex_ * tilingData_->dOutputInner);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_->hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_->wOutputInner);
    uint16_t cOutputActual = cOutputActual_;
    uint16_t cOutputAligned = cOutputAligned_;
    uint16_t dGradActual = static_cast<uint16_t>(dGradActual_);
    uint16_t hGradActual = static_cast<uint16_t>(hGradActual_);
    uint16_t wGradActual = static_cast<uint16_t>(wGradActual_);
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);
    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFront);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBack);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTop);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padBottom);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeft);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRight);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);
    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wGradActual / wProBatchSize;
    uint16_t computeSize = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSize / cOutputActual;
    uint16_t hConcurrentCount = concurrencyCount / wFullBatchCount;
    uint16_t blockConcurrentCount = (hGradActual / hProBatchSize) / hConcurrentCount;
    uint16_t hRemain = hGradActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;
    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;
    uint16_t wRemainTail = wGradActual - wFullBatchCount * wProBatchSize;
    uint16_t vecElemCountU32 = V_REG_SIZE / sizeof(uint32_t);
    uint16_t vecElemCountT3 = V_REG_SIZE / sizeof(T3);
    uint32_t mask0 = wFullBatchCount * hConcurrentCount * cOutputActual;
    uint32_t mask1 = hConcurrentCount * cOutputActual;
    uint32_t mask2 = wFullBatchCount * hRemainBatchCount * cOutputActual;
    uint32_t mask3 = hRemainBatchCount * cOutputActual;
    uint32_t mask4 = wFullBatchCount * cOutputActual;
    uint32_t mask5 = cOutputActual;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdxOne;
        AscendC::MicroAPI::MaskReg
            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        GenInitial3DIndicesForNDHWC<int32_t>((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize,
                                             hProBatchSize, wGradActual, wFullBatchCount, cOutputActual,
                                             cOutputAligned);
        Gen3DIndexOneForNDHWC<int32_t>((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize,
                                       wGradActual, cOutputActual, cOutputAligned);
        GenRepeatIndicesWithLoop<T3, Trait>(initialWRegIdx, wFullBatchCount, cOutputActual, wProBatchSize);
        GenRepeatIndices<T3, Trait>(initialHRegIdx, hProBatchSize, wFullBatchCount * cOutputActual);
        GenRepeatIndicesWithLoop<T3, Trait>(initialWRegIdxOne, 1, cOutputActual, wProBatchSize);
        GenRepeatIndices<T3, Trait>(initialHRegIdxOne, hProBatchSize, cOutputActual);
        AscendC::MicroAPI::DataCopy(helpAddr, initialRegIndex, allMaskU32);
        AscendC::MicroAPI::DataCopy(helpAddr + vecElemCountU32, initialRegIndexOne, allMaskU32);
        AscendC::MicroAPI::DataCopy(helpAddrT3, initialHRegIdx, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * vecElemCountT3, initialWRegIdx, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * INDEX_TWO * vecElemCountT3, initialHRegIdxOne, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_THREE * INDEX_TWO * vecElemCountT3, initialWRegIdxOne,
                                    allMaskT3);
    }

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nOffset = nIdx * dOutputActual * hOutputActual * wOutputActual * cOutputAligned;
        uint32_t nGradOffset = nIdx * dGradActual * hGradActual * wGradActual * cOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dGradActual; ++dIdx) {
            T3 dGradOffset = dIdx + dGradActualStart;
            T3 dGradOffsetMulStrideD = dGradOffset * strideD;
            uint32_t dGradBase = dIdx * hGradActual * wGradActual;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    T3 hGradOffset = hIdx * hConcurrentCount * hProBatchSize + hProBatchIdx + hGradActualStart;
                    uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                           hIdx * wGradActual * hProBatchSize * hConcurrentCount;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        DoMergeHWBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, false>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask0, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffsetMulStrideD, hGradOffset, wGradOffset, wOutputActual,
                            hOutputActual, dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD,
                            kH, kW, padD, padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH,
                            strideW, curDIndex, curHIndex, curWIndex);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                        DoMergeHWBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, true>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask1, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffsetMulStrideD, hGradOffset, wGradOffset, wOutputActual,
                            hOutputActual, dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD,
                            kH, kW, padD, padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH,
                            strideW, curDIndex, curHIndex, curWIndex);
                    }
                }
            }
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                T3 hGradOffset = hProBatchIdx + blockConcurrentCount * hProBatchSize * hConcurrentCount +
                                 hGradActualStart;
                uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                       blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradActual;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    DoMergeHWBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, false>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask2, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffsetMulStrideD, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                    DoMergeHWBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, true>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask3, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffsetMulStrideD, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
            }
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                T3 hGradOffset = hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                 blockConcurrentCount * hProBatchSize * hConcurrentCount + hGradActualStart;
                uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                       hRemainBatchCount * hProBatchSize * wGradActual +
                                       blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradActual;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    DoMergeHWBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, false>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask4, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffsetMulStrideD, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                    DoMergeHWBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, true>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask5, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffsetMulStrideD, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNDHWC<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ConCMergeDHWCProcVF3D(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3)
{
    int32_t dOutput = static_cast<int32_t>(tilingData_->dOutput);
    int32_t hOutput = static_cast<int32_t>(tilingData_->hOutput);
    int32_t wOutput = static_cast<int32_t>(tilingData_->wOutput);
    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t dOutputActual = static_cast<int32_t>(dOutputActual_);
    int32_t curDIndex = static_cast<int32_t>(dAxisIndex_ * tilingData_->dOutputInner);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_->hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_->wOutputInner);
    uint16_t cOutputActual = cOutputActual_;
    uint16_t cOutputAligned = cOutputAligned_;
    uint16_t dGradActual = static_cast<uint16_t>(dGradActual_);
    uint16_t hGradActual = static_cast<uint16_t>(hGradActual_);
    uint16_t wGradActual = static_cast<uint16_t>(wGradActual_);
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);
    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFront);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBack);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTop);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeft);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padBottom);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRight);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);
    uint16_t dProBatchSize = curDProBatchSize_;
    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wGradActual / wProBatchSize;
    uint16_t hFullBatchCount = hGradActual / hProBatchSize;
    uint32_t dFullBatchCount = dGradActual / dProBatchSize;
    uint16_t computeSize = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSize / cOutputActual;
    uint16_t hConcurrentCount = concurrencyCount / wFullBatchCount;
    uint16_t dConcurrentCount = hConcurrentCount / hFullBatchCount;
    uint16_t dBlockConcurrentCount = dFullBatchCount / dConcurrentCount;
    uint16_t dRemain = dGradActual - dBlockConcurrentCount * dConcurrentCount * dProBatchSize;
    uint16_t dRemainBatchCount = dRemain / dProBatchSize;
    uint16_t dRemainTail = dRemain - dRemainBatchCount * dProBatchSize;
    uint16_t blockConcurrentCount = 1;
    uint16_t hRemain = hGradActual - hFullBatchCount * hProBatchSize;
    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;
    uint16_t wRemainTail = wGradActual - wFullBatchCount * wProBatchSize;
    uint16_t vecElemCountU32 = V_REG_SIZE / sizeof(uint32_t);
    uint16_t vecElemCountT3 = V_REG_SIZE / sizeof(T3);

    uint32_t mask0 = wFullBatchCount * hFullBatchCount * dConcurrentCount * cOutputActual;
    uint32_t mask1 = hFullBatchCount * dConcurrentCount * cOutputActual;
    uint32_t mask2 = wFullBatchCount * hRemainBatchCount * dConcurrentCount * cOutputActual;
    uint32_t mask3 = hRemainBatchCount * dConcurrentCount * cOutputActual;
    uint32_t mask4 = wFullBatchCount * dConcurrentCount * cOutputActual;
    uint32_t mask5 = dConcurrentCount * cOutputActual;
    uint32_t mask6 = wFullBatchCount * hFullBatchCount * dRemainBatchCount * cOutputActual;
    uint32_t mask7 = hFullBatchCount * dRemainBatchCount * cOutputActual;
    uint32_t mask8 = wFullBatchCount * hRemainBatchCount * dRemainBatchCount * cOutputActual;
    uint32_t mask9 = hRemainBatchCount * dRemainBatchCount * cOutputActual;
    uint32_t mask10 = wFullBatchCount * dRemainBatchCount * cOutputActual;
    uint32_t mask11 = dRemainBatchCount * cOutputActual;
    uint32_t mask12 = wFullBatchCount * hFullBatchCount * cOutputActual;
    uint32_t mask13 = hFullBatchCount * cOutputActual;
    uint32_t mask14 = wFullBatchCount * hRemainBatchCount * cOutputActual;
    uint32_t mask15 = hRemainBatchCount * cOutputActual;
    uint32_t mask16 = wFullBatchCount * cOutputActual;
    uint32_t mask17 = cOutputActual;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexHRemain;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexHRemainOne;
        AscendC::MicroAPI::MaskReg
            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        GenInitial4DIndicesForNDHWC<int32_t>((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize,
                                             hProBatchSize, dProBatchSize, wGradActual, wFullBatchCount,
                                             hFullBatchCount, cOutputActual, cOutputAligned, hGradActual);
        AscendC::MicroAPI::DataCopy(helpAddr, initialRegIndex, allMaskU32);
        Gen4DIndexOneForNDHWC<int32_t>((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize,
                                       dProBatchSize, wGradActual, hFullBatchCount, cOutputActual, cOutputAligned,
                                       hGradActual);
        AscendC::MicroAPI::DataCopy(helpAddr + vecElemCountU32, initialRegIndexOne, allMaskU32);
        GenInitial4DIndicesForNDHWC<int32_t>((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexHRemain,
                                             wProBatchSize, hProBatchSize, dProBatchSize, wGradActual, wFullBatchCount,
                                             1, cOutputActual, cOutputAligned, hGradActual);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_TWO * vecElemCountU32, initialRegIndexHRemain, allMaskU32);
        Gen4DIndexOneForNDHWC<int32_t>((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexHRemainOne, hProBatchSize,
                                       dProBatchSize, wGradActual, 1, cOutputActual, cOutputAligned, hGradActual);
        AscendC::MicroAPI::DataCopy(helpAddr + INDEX_THREE * vecElemCountU32, initialRegIndexHRemainOne, allMaskU32);
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T3, Trait> initialDRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialDRegIdxOne;
        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        GenRepeatIndices<T3, Trait>(initialDRegIdx, dProBatchSize, hFullBatchCount * wFullBatchCount * cOutputActual);
        AscendC::MicroAPI::DataCopy(helpAddrT3, initialDRegIdx, allMaskT3);
        GenRepeatIndicesWithLoop<T3, Trait>(initialHRegIdx, hFullBatchCount, wFullBatchCount * cOutputActual,
                                            hProBatchSize);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * vecElemCountT3, initialHRegIdx, allMaskT3);
        GenRepeatIndicesWithLoop<T3, Trait>(initialWRegIdx, wFullBatchCount, cOutputActual, wProBatchSize);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * INDEX_TWO * vecElemCountT3, initialWRegIdx, allMaskT3);
        GenRepeatIndices<T3, Trait>(initialDRegIdxOne, dProBatchSize, hFullBatchCount * cOutputActual);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_THREE * INDEX_TWO * vecElemCountT3, initialDRegIdxOne,
                                    allMaskT3);
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdxOne;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialDRegIdxHRemain;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxHRemain;
        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        GenRepeatIndicesWithLoop<T3, Trait>(initialHRegIdxOne, hFullBatchCount, cOutputActual, hProBatchSize);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_FOUR * INDEX_TWO * vecElemCountT3, initialHRegIdxOne, allMaskT3);
        GenRepeatIndicesWithLoop<T3, Trait>(initialWRegIdxOne, 1, cOutputActual, wProBatchSize);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_FIVE * INDEX_TWO * vecElemCountT3, initialWRegIdxOne, allMaskT3);
        GenRepeatIndices<T3, Trait>(initialDRegIdxHRemain, dProBatchSize, wFullBatchCount * cOutputActual);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_SIX * INDEX_TWO * vecElemCountT3, initialDRegIdxHRemain,
                                    allMaskT3);
        GenRepeatIndicesWithLoop<T3, Trait>(initialHRegIdxHRemain, 1, wFullBatchCount * cOutputActual, hProBatchSize);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_SEVEN * INDEX_TWO * vecElemCountT3, initialHRegIdxHRemain,
                                    allMaskT3);
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdxHRemain;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialDRegIdxHRemainOne;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxHRemainOne;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdxHRemainOne;
        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        GenRepeatIndicesWithLoop<T3, Trait>(initialWRegIdxHRemain, wFullBatchCount, cOutputActual, wProBatchSize);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_EIGHT * INDEX_TWO * vecElemCountT3, initialWRegIdxHRemain,
                                    allMaskT3);
        GenRepeatIndices<T3, Trait>(initialDRegIdxHRemainOne, dProBatchSize, cOutputActual);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_NINE * INDEX_TWO * vecElemCountT3, initialDRegIdxHRemainOne,
                                    allMaskT3);
        GenRepeatIndicesWithLoop<T3, Trait>(initialHRegIdxHRemainOne, 1, cOutputActual, hProBatchSize);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TEN * INDEX_TWO * vecElemCountT3, initialHRegIdxHRemainOne,
                                    allMaskT3);
        GenRepeatIndicesWithLoop<T3, Trait>(initialWRegIdxHRemainOne, 1, cOutputActual, wProBatchSize);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_ELEVEN * INDEX_TWO * vecElemCountT3, initialWRegIdxHRemainOne,
                                    allMaskT3);
    }

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nOffset = nIdx * dOutputActual * hOutputActual * wOutputActual * cOutputAligned;
        uint32_t nGradOffset = nIdx * dGradActual * hGradActual * wGradActual * cOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dBlockConcurrentCount; ++dIdx) {
            for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; ++dProBatchIdx) {
                T3 dGradOffset = dIdx * dConcurrentCount * dProBatchSize + dProBatchIdx + dGradActualStart;
                uint32_t dGradBase = (dIdx * dConcurrentCount * dProBatchSize + dProBatchIdx) * hGradActual *
                                     wGradActual;
                for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; ++hIdx) {
                    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                        T3 hGradOffset = hIdx * hFullBatchCount * hProBatchSize + hProBatchIdx + hGradActualStart;
                        uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                               hIdx * wGradActual * hProBatchSize * hFullBatchCount;
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                            T3 wGradOffset = wBatchIdx + wGradActualStart;
                            DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 0>(
                                yAddr, gradAddr, helpAddr, helpAddrT3, mask0, nOffset, nGradOffset, dhwGradBase,
                                wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                                dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW,
                                padD, padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH,
                                strideW, curDIndex, curHIndex, curWIndex);
                        }
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                            T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                            DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 1>(
                                yAddr, gradAddr, helpAddr, helpAddrT3, mask1, nOffset, nGradOffset, dhwGradBase,
                                wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                                dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW,
                                padD, padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH,
                                strideW, curDIndex, curHIndex, curWIndex);
                        }
                    }
                }
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                    T3 hGradOffset = blockConcurrentCount * hFullBatchCount * hProBatchSize + hProBatchIdx +
                                     hGradActualStart;
                    uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                           blockConcurrentCount * hFullBatchCount * hProBatchSize * wGradActual;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 0>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask2, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                            dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD,
                            padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW,
                            curDIndex, curHIndex, curWIndex);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                        DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 1>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask3, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                            dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD,
                            padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW,
                            curDIndex, curHIndex, curWIndex);
                    }
                }
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; ++hProBatchIdx) {
                    T3 hGradOffset = hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                     blockConcurrentCount * hFullBatchCount * hProBatchSize + hGradActualStart;
                    uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                           hRemainBatchCount * hProBatchSize * wGradActual +
                                           blockConcurrentCount * hFullBatchCount * hProBatchSize * wGradActual;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 2>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask4, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                            dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD,
                            padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW,
                            curDIndex, curHIndex, curWIndex);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                        DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 3>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask5, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                            dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD,
                            padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW,
                            curDIndex, curHIndex, curWIndex);
                    }
                }
            }
        }
        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dProBatchSize; ++dProBatchIdx) {
            T3 dGradOffset = dBlockConcurrentCount * dConcurrentCount * dProBatchSize + dProBatchIdx + dGradActualStart;
            uint32_t dGradBase = (dBlockConcurrentCount * dConcurrentCount * dProBatchSize + dProBatchIdx) *
                                 hGradActual * wGradActual;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; ++hIdx) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                    T3 hGradOffset = hIdx * hFullBatchCount * hProBatchSize + hProBatchIdx + hGradActualStart;
                    uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                           hIdx * wGradActual * hProBatchSize * hFullBatchCount;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 0>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask6, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                            dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD,
                            padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW,
                            curDIndex, curHIndex, curWIndex);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                        DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 1>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask7, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                            dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD,
                            padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW,
                            curDIndex, curHIndex, curWIndex);
                    }
                }
            }
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                T3 hGradOffset = blockConcurrentCount * hFullBatchCount * hProBatchSize + hProBatchIdx +
                                 hGradActualStart;
                uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                       blockConcurrentCount * hFullBatchCount * hProBatchSize * wGradActual;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 0>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask8, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                    T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                    DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 1>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask9, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
            }
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; ++hProBatchIdx) {
                T3 hGradOffset = hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                 blockConcurrentCount * hFullBatchCount * hProBatchSize + hGradActualStart;
                uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                       hRemainBatchCount * hProBatchSize * wGradActual +
                                       blockConcurrentCount * hFullBatchCount * hProBatchSize * wGradActual;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 2>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask10, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                    T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                    DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 3>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask11, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
            }
        }
        for (uint16_t dProBatchIdx = 0; dProBatchIdx < dRemainTail; ++dProBatchIdx) {
            T3 dGradOffset = dProBatchIdx + dRemainBatchCount * dProBatchSize +
                             dBlockConcurrentCount * dConcurrentCount * dProBatchSize + dGradActualStart;
            uint32_t dGradBase = (dProBatchIdx + dRemainBatchCount * dProBatchSize +
                                  dBlockConcurrentCount * dConcurrentCount * dProBatchSize) *
                                 hGradActual * wGradActual;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; ++hIdx) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                    T3 hGradOffset = hIdx * hFullBatchCount * hProBatchSize + hProBatchIdx + hGradActualStart;
                    uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                           hIdx * wGradActual * hProBatchSize * hFullBatchCount;
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 0>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask12, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                            dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD,
                            padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW,
                            curDIndex, curHIndex, curWIndex);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                        T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                        DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 1>(
                            yAddr, gradAddr, helpAddr, helpAddrT3, mask13, nOffset, nGradOffset, dhwGradBase,
                            wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                            dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD,
                            padH, padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW,
                            curDIndex, curHIndex, curWIndex);
                    }
                }
            }
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
                T3 hGradOffset = blockConcurrentCount * hFullBatchCount * hProBatchSize + hProBatchIdx +
                                 hGradActualStart;
                uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                       blockConcurrentCount * hFullBatchCount * hProBatchSize * wGradActual;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 0>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask14, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                    T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                    DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 1>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask15, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
            }
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; ++hProBatchIdx) {
                T3 hGradOffset = hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                 blockConcurrentCount * hFullBatchCount * hProBatchSize + hGradActualStart;
                uint32_t dhwGradBase = dGradBase + hProBatchIdx * wGradActual +
                                       hRemainBatchCount * hProBatchSize * wGradActual +
                                       blockConcurrentCount * hFullBatchCount * hProBatchSize * wGradActual;
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 2>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask16, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                    T3 wGradOffset = wBatchIdx + wFullBatchCount * wProBatchSize + wGradActualStart;
                    DoMergeDHWCBlock3D<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD, Trait, 3>(
                        yAddr, gradAddr, helpAddr, helpAddrT3, mask17, nOffset, nGradOffset, dhwGradBase,
                        wGradActualStart, dGradOffset, hGradOffset, wGradOffset, wOutputActual, hOutputActual,
                        dOutputActual, cOutputAligned, cOutputActual, dOutput, hOutput, wOutput, kD, kH, kW, padD, padH,
                        padW, padBackD, padDownH, padRightW, divisorOverride, strideD, strideH, strideW, curDIndex,
                        curHIndex, curWIndex);
                }
            }
        }
    }
}

} // namespace AvgPool3DGradNDHWCNameSpace
#endif // AVG_POOL3_D_GRAD_NDHWC_KERNEL_H_
