/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AVG_POOL3_D_GRAD_NCDHW_KERNEL_H_
#define AVG_POOL3_D_GRAD_NCDHW_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "avg_pool3_d_grad_base.h"
#include "avg_pool3_d_grad_tiling_data.h"

namespace AvgPool3DGradNCDHWNameSpace {
using namespace AscendC;
using namespace AvgPool3DGrad;

constexpr static int64_t V_REG_SIZE = platform::GetVRegSize();

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoScatterForDhwParallel(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex,
    uint32_t gradMaskCount, int32_t wOutputAligned, int32_t hOutputActual, int32_t highOutputOffset,
    MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& dMaxReg,
    MicroAPI::RegTensor<int32_t>& hMaxReg, MicroAPI::RegTensor<int32_t>& wMaxReg, uint16_t kD, uint16_t kH, uint16_t kW,
    MicroAPI::RegTensor<int32_t>& divisorReg, MicroAPI::RegTensor<int32_t>& wIndexReg,
    MicroAPI::RegTensor<int32_t>& hIndexReg, MicroAPI::RegTensor<int32_t>& dIndexReg,
    MicroAPI::RegTensor<int32_t>& highIdxReg)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterStartIdxReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterIndexReg;
    AscendC::MicroAPI::RegTensor<int32_t> dOffsetReg;
    AscendC::MicroAPI::RegTensor<int32_t> hOffsetReg;

    uint32_t maskT1 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    uint32_t maskI32 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregI32 = AscendC::MicroAPI::UpdateMask<int32_t>(maskI32);
    GetConCurrentInput<T1>(gradReg, gradAddr, parallelRegIndex, pregT1);

    AscendC::MicroAPI::Muls(dOffsetReg, dIndexReg, hOutputActual * wOutputAligned, pregI32);
    AscendC::MicroAPI::Muls(hOffsetReg, hIndexReg, wOutputAligned, pregI32);
    AscendC::MicroAPI::Add(scatterStartIdxReg, dOffsetReg, hOffsetReg, pregI32);
    AscendC::MicroAPI::Add(scatterStartIdxReg, scatterStartIdxReg, wIndexReg, pregI32);
    AscendC::MicroAPI::Add(scatterStartIdxReg, scatterStartIdxReg, highIdxReg, pregI32);
    for (uint16_t dKIdx = 0; dKIdx < kD; dKIdx++) {
        int32_t dKernelOffset = dKIdx * hOutputActual * wOutputAligned;
        for (uint16_t hKIdx = 0; hKIdx < kH; hKIdx++) {
            int32_t hKernelOffset = hKIdx * wOutputAligned;
            for (uint16_t wKIdx = 0; wKIdx < kW; wKIdx++) {
                uint32_t gradMask = gradMaskCount;
                AscendC::MicroAPI::MaskReg pregRes = AscendC::MicroAPI::UpdateMask<int32_t>(gradMask);
                int32_t scatterIndexOffsetTotal = highOutputOffset + dKernelOffset + hKernelOffset + wKIdx;
                AscendC::MicroAPI::Adds(scatterIndexReg, scatterStartIdxReg, scatterIndexOffsetTotal, pregRes);
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::RegTensor<int32_t> wCurIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hCurIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dCurIndexReg;
                    AscendC::MicroAPI::Adds(wCurIndexReg, wIndexReg, static_cast<int32_t>(wKIdx), pregRes);
                    AscendC::MicroAPI::Adds(hCurIndexReg, hIndexReg, static_cast<int32_t>(hKIdx), pregRes);
                    AscendC::MicroAPI::Adds(dCurIndexReg, dIndexReg, static_cast<int32_t>(dKIdx), pregRes);
                    FilterMask3D(pregRes, dCurIndexReg, hCurIndexReg, wCurIndexReg, zeroConstReg, dMaxReg, hMaxReg,
                                 wMaxReg);
                }
                GradientAcc(yAddr, gradReg, scatterIndexReg, divisorReg, pregRes);
            }
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoScatterForHwParallel(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, MicroAPI::RegTensor<uint32_t>& parallelRegIndex,
    uint32_t gradMaskCount, int32_t wOutputAligned, int32_t hOutputActual, int32_t highOutputOffset,
    MicroAPI::RegTensor<int32_t>& zeroConstReg, MicroAPI::RegTensor<int32_t>& hMaxReg,
    MicroAPI::RegTensor<int32_t>& wMaxReg, uint16_t kH, uint16_t kW, MicroAPI::RegTensor<int32_t>& divisorReg,
    MicroAPI::RegTensor<int32_t>& wIndexReg, MicroAPI::RegTensor<int32_t>& hIndexReg, int32_t dIndex, int32_t dkStart,
    int32_t dkEnd)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterStartIdxReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterIndexReg;

    uint32_t maskT1 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    uint32_t maskI32 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregI32 = AscendC::MicroAPI::UpdateMask<int32_t>(maskI32);
    GetConCurrentInput<T1>(gradReg, gradAddr, parallelRegIndex, pregT1);

    AscendC::MicroAPI::Muls(scatterStartIdxReg, hIndexReg, wOutputAligned, pregI32);
    AscendC::MicroAPI::Add(scatterStartIdxReg, scatterStartIdxReg, wIndexReg, pregI32);
    int64_t dBaseOffset = static_cast<int64_t>(dIndex) * hOutputActual * wOutputAligned;
    for (uint16_t dKIdx = dkStart; dKIdx < dkEnd; dKIdx++) {
        int32_t dKernelOffset = static_cast<int32_t>(dKIdx * hOutputActual * wOutputAligned + dBaseOffset);
        for (uint16_t hKIdx = 0; hKIdx < kH; hKIdx++) {
            int32_t hKernelOffset = hKIdx * wOutputAligned;
            for (uint16_t wKIdx = 0; wKIdx < kW; wKIdx++) {
                uint32_t gradMask = gradMaskCount;
                AscendC::MicroAPI::MaskReg pregRes = AscendC::MicroAPI::UpdateMask<int32_t>(gradMask);
                int32_t scatterIndexOffsetTotal = highOutputOffset + dKernelOffset + hKernelOffset + wKIdx;
                AscendC::MicroAPI::Adds(scatterIndexReg, scatterStartIdxReg, scatterIndexOffsetTotal, pregRes);
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::RegTensor<int32_t> wCurIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hCurIndexReg;
                    AscendC::MicroAPI::Adds(wCurIndexReg, wIndexReg, static_cast<int32_t>(wKIdx), pregRes);
                    AscendC::MicroAPI::Adds(hCurIndexReg, hIndexReg, static_cast<int32_t>(hKIdx), pregRes);
                    FilterMaskForHwParallel(pregRes, hCurIndexReg, wCurIndexReg, zeroConstReg, wMaxReg, hMaxReg);
                }
                GradientAcc(yAddr, gradReg, scatterIndexReg, divisorReg, pregRes);
            }
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void DoScatterForWParallel(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                             MicroAPI::RegTensor<uint32_t>& parallelRegIndex, uint32_t gradMaskCount,
                                             int32_t wOutputAligned, int32_t hOutputActual, int32_t highOutputOffset,
                                             MicroAPI::RegTensor<int32_t>& zeroConstReg,
                                             MicroAPI::RegTensor<int32_t>& wMaxReg, uint16_t kW,
                                             MicroAPI::RegTensor<int32_t>& divisorReg,
                                             MicroAPI::RegTensor<int32_t>& wIndexReg, int32_t dIndex, int32_t dkStart,
                                             int32_t dkEnd, int32_t hIndex, int32_t hkStart, int32_t hkEnd)
{
    AscendC::MicroAPI::RegTensor<computeType> gradReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterStartIdxReg;
    AscendC::MicroAPI::RegTensor<int32_t> scatterIndexReg;

    uint32_t maskT1 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregT1 = AscendC::MicroAPI::UpdateMask<T1>(maskT1);
    uint32_t maskI32 = gradMaskCount;
    AscendC::MicroAPI::MaskReg pregI32 = AscendC::MicroAPI::UpdateMask<int32_t>(maskI32);
    GetConCurrentInput<T1>(gradReg, gradAddr, parallelRegIndex, pregT1);

    int32_t scatterStartIdx = dIndex * hOutputActual * wOutputAligned + hIndex * wOutputAligned;
    AscendC::MicroAPI::Adds(scatterStartIdxReg, wIndexReg, scatterStartIdx, pregI32);
    for (uint16_t dKIdx = dkStart; dKIdx < dkEnd; dKIdx++) {
        int32_t dKernelOffset = dKIdx * hOutputActual * wOutputAligned;
        for (uint16_t hKIdx = hkStart; hKIdx < hkEnd; hKIdx++) {
            int32_t hKernelOffset = hKIdx * wOutputAligned;
            for (uint16_t wKIdx = 0; wKIdx < kW; wKIdx++) {
                uint32_t gradMask = gradMaskCount;
                AscendC::MicroAPI::MaskReg pregRes = AscendC::MicroAPI::UpdateMask<int32_t>(gradMask);
                int32_t scatterIndexOffsetTotal = highOutputOffset + dKernelOffset + hKernelOffset + wKIdx;
                AscendC::MicroAPI::Adds(scatterIndexReg, scatterStartIdxReg, scatterIndexOffsetTotal, pregRes);
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::RegTensor<int32_t> wCurIndexReg;
                    AscendC::MicroAPI::Adds(wCurIndexReg, wIndexReg, static_cast<int32_t>(wKIdx), pregRes);
                    FilterMaskForWParallel(pregRes, wCurIndexReg, zeroConstReg, wMaxReg);
                }
                GradientAcc(yAddr, gradReg, scatterIndexReg, divisorReg, pregRes);
            }
        }
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
class AvgPool3DGradNCDHW {
public:
    __aicore__ inline AvgPool3DGradNCDHW(TPipe* pipe, const AvgPool3DGradNCDHWTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}
    ~AvgPool3DGradNCDHW() {}
    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR output);
    __aicore__ inline void Process();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ProcessNoArgmaxBlock();
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void SingleLineProcessVF(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr);
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void MultipleLineProcessVF1(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                  __local_mem__ uint32_t* helpAddr, __local_mem__ T3* helpAddrT3);
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void MultipleLineProcessVF1Int64(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                       __local_mem__ uint32_t* helpAddr, __local_mem__ T3* helpAddrT3);
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void MultipleLineProcessVF2(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                  __local_mem__ uint32_t* helpAddr, __local_mem__ T3* helpAddrT3);
    template <const MicroAPI::RegTrait& Trait>
    __aicore__ inline void MultipleLineProcessVF2Int64(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                       __local_mem__ uint32_t* helpAddr, __local_mem__ T3* helpAddrT3);

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> helpBuf_;
    TBuf<QuePosition::VECCALC> helpBufT3_;
    GlobalTensor<T1> gradGm_;
    GlobalTensor<T1> yGm_;
    const AvgPool3DGradNCDHWTilingData* tilingData_;

    uint32_t blockIdx_ = 0;
    int64_t curCoreProcessNum_ = 1;
    int64_t gradPlaneSize_ = 1;
    int64_t outputPlaneSize_ = 1;

    int64_t highAxisActual_ = 1;
    int64_t dOutputActual_ = 1;
    int64_t hOutputActual_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t wOutputAligned_ = 1;

    int64_t highAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t dGradActual_ = 0;
    int64_t hGradActual_ = 0;
    int64_t wGradActual_ = 0;
    int64_t wGradAligned_ = 0;
    int64_t dGradActualStart_ = 0;
    int64_t hGradActualStart_ = 0;
    int64_t wGradActualStart_ = 0;

    int64_t highAxisGradOffset_ = 0;
    int64_t dAxisGradOffset_ = 0;
    int64_t hAxisGradOffset_ = 0;
    int64_t wAxisGradOffset_ = 0;

    int64_t curDProBatchSize_ = 1;
    int64_t curHProBatchSize_ = 1;
    int64_t curWProBatchSize_ = 1;

    constexpr static int64_t DATA_NUM_IN_ONE_BLOCK = platform::GetUbBlockSize() / sizeof(T1);
};

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Init(GM_ADDR grads,
                                                                                                GM_ADDR output)
{
    blockIdx_ = GetBlockIdx();
    gradPlaneSize_ = tilingData_->dGrad * tilingData_->hGrad * tilingData_->wGrad;
    outputPlaneSize_ = tilingData_->dOutput * tilingData_->hOutput * tilingData_->wOutput;
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    curCoreProcessNum_ = (blockIdx_ + 1 == tilingData_->usedCoreNum) ? tilingData_->tailCoreProcessNum :
                                                                       tilingData_->normalCoreProcessNum;
    gradGm_.SetGlobalBuffer((__gm__ T1*)grads);
    yGm_.SetGlobalBuffer((__gm__ T1*)output);
    pipe_->InitBuffer(gradQue_, BUFFER_NUM, tilingData_->gradBufferSize);
    pipe_->InitBuffer(outputQue_, BUFFER_NUM, tilingData_->outputBufferSize);
    pipe_->InitBuffer(helpBuf_, HELP_BUFFER);
    pipe_->InitBuffer(helpBufT3_, HELP_BUFFER_T3);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ScalarCompute(
    int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tilingData_->normalCoreProcessNum + loopNum;
    int64_t dhwOut = tilingData_->dOutputOuter * tilingData_->hOutputOuter * tilingData_->wOutputOuter;
    highAxisIndex_ = baseBlockIdx / dhwOut;
    highAxisActual_ = highAxisIndex_ == (tilingData_->highAxisOuter - 1) ? tilingData_->highAxisTail :
                                                                           tilingData_->highAxisInner;

    int64_t hwOut = tilingData_->hOutputOuter * tilingData_->wOutputOuter;
    int64_t tempDhw = baseBlockIdx % dhwOut;
    dAxisIndex_ = tempDhw / hwOut;
    dOutputActual_ = dAxisIndex_ == (tilingData_->dOutputOuter - 1) ? tilingData_->dOutputTail :
                                                                      tilingData_->dOutputInner;

    int64_t tempHw = tempDhw % hwOut;
    hAxisIndex_ = tempHw / tilingData_->wOutputOuter;
    hOutputActual_ = hAxisIndex_ == (tilingData_->hOutputOuter - 1) ? tilingData_->hOutputTail :
                                                                      tilingData_->hOutputInner;

    wAxisIndex_ = tempHw % tilingData_->wOutputOuter;
    wOutputActual_ = wAxisIndex_ == (tilingData_->wOutputOuter - 1) ? tilingData_->wOutputTail :
                                                                      tilingData_->wOutputInner;
    wOutputAligned_ = (wOutputActual_ + DATA_NUM_IN_ONE_BLOCK - 1) / DATA_NUM_IN_ONE_BLOCK * DATA_NUM_IN_ONE_BLOCK;

    dGradActualStart_ = PStart(dAxisIndex_ * tilingData_->dOutputInner, tilingData_->padFrontD, tilingData_->dKernel,
                               tilingData_->dStride);
    int64_t dGradActualEnd = PEnd(dAxisIndex_ * tilingData_->dOutputInner + dOutputActual_ - 1, tilingData_->padFrontD,
                                  tilingData_->dStride, tilingData_->dGrad);
    dGradActual_ = dGradActualEnd - dGradActualStart_;

    hGradActualStart_ = PStart(hAxisIndex_ * tilingData_->hOutputInner, tilingData_->padTopH, tilingData_->hKernel,
                               tilingData_->hStride);
    int64_t hGradActualEnd = PEnd(hAxisIndex_ * tilingData_->hOutputInner + hOutputActual_ - 1, tilingData_->padTopH,
                                  tilingData_->hStride, tilingData_->hGrad);
    hGradActual_ = hGradActualEnd - hGradActualStart_;

    wGradActualStart_ = PStart(wAxisIndex_ * tilingData_->wOutputInner, tilingData_->padLeftW, tilingData_->wKernel,
                               tilingData_->wStride);
    int64_t wGradActualEnd = PEnd(wAxisIndex_ * tilingData_->wOutputInner + wOutputActual_ - 1, tilingData_->padLeftW,
                                  tilingData_->wStride, tilingData_->wGrad);
    wGradActual_ = wGradActualEnd - wGradActualStart_;
    wGradAligned_ = (wGradActual_ + DATA_NUM_IN_ONE_BLOCK - 1) / DATA_NUM_IN_ONE_BLOCK * DATA_NUM_IN_ONE_BLOCK;

    curDProBatchSize_ = tilingData_->dProBatchSize > dGradActual_ ? dGradActual_ : tilingData_->dProBatchSize;
    curHProBatchSize_ = tilingData_->hProBatchSize > hGradActual_ ? hGradActual_ : tilingData_->hProBatchSize;
    curWProBatchSize_ = tilingData_->wProBatchSize > wGradActual_ ? wGradActual_ : tilingData_->wProBatchSize;

    highAxisGradOffset_ = highAxisIndex_ * tilingData_->highAxisInner * gradPlaneSize_;
    dAxisGradOffset_ = dGradActualStart_ * tilingData_->hGrad * tilingData_->wGrad;
    hAxisGradOffset_ = hGradActualStart_ * tilingData_->wGrad;
    wAxisGradOffset_ = wGradActualStart_;
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Process()
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
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ProcessPerLoop()
{
    if (dGradActual_ <= 0 || hGradActual_ <= 0 || wGradActual_ <= 0) {
        ProcessNoArgmaxBlock();
        return;
    }
    CopyIn();
    Compute();
    CopyOut();
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(tilingData_->outputBufferSize) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, static_cast<T1>(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::CopyIn()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    int64_t gradGmOffset = highAxisGradOffset_ + dAxisGradOffset_ + hAxisGradOffset_ + wAxisGradOffset_;
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop2Size = dGradActual_;
    loopModeParamsT1.loop1SrcStride = gradPlaneSize_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = tilingData_->hGrad * tilingData_->wGrad * sizeof(T1);
    loopModeParamsT1.loop1DstStride = dGradActual_ * hGradActual_ * wGradAligned_ * sizeof(T1);
    loopModeParamsT1.loop2DstStride = hGradActual_ * wGradAligned_ * sizeof(T1);
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
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::Compute()
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

    uint32_t wConcurrentCount = wGradActual_ / curWProBatchSize_;
    uint32_t hConcurrentCount = hGradActual_ / curHProBatchSize_;
    uint32_t dConcurrentCount = dGradActual_ / curDProBatchSize_;
    if (wConcurrentCount * DOUBLE * sizeof(float) > V_REG_SIZE) {
        if constexpr (std::is_same<T3, int64_t>::value) {
            SingleLineProcessVF<AscendC::MicroAPI::RegTraitNumTwo>(yAddr, gradAddr);
        } else {
            SingleLineProcessVF<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr);
        }
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(float) > V_REG_SIZE) {
        if constexpr (std::is_same<T3, int64_t>::value) {
            MultipleLineProcessVF1Int64<AscendC::MicroAPI::RegTraitNumTwo>(yAddr, gradAddr, helpAddr, helpAddrT3);
        } else {
            MultipleLineProcessVF1<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr, helpAddr, helpAddrT3);
        }
    } else {
        if constexpr (std::is_same<T3, int64_t>::value) {
            MultipleLineProcessVF2Int64<AscendC::MicroAPI::RegTraitNumTwo>(yAddr, gradAddr, helpAddr, helpAddrT3);
        } else {
            MultipleLineProcessVF2<AscendC::MicroAPI::RegTraitNumOne>(yAddr, gradAddr, helpAddr, helpAddrT3);
        }
    }

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
    }
    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();
    int64_t highOutputAxisOffset = highAxisIndex_ * tilingData_->highAxisInner * outputPlaneSize_;
    int64_t dOutputAxisOffset = dAxisIndex_ * tilingData_->dOutputInner * tilingData_->hOutput * tilingData_->wOutput;
    int64_t hOutputAxisOffset = hAxisIndex_ * tilingData_->hOutputInner * tilingData_->wOutput;
    int64_t wOutputAxisOffset = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t outputGmOffset = highOutputAxisOffset + dOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop2Size = dOutputActual_;
    loopModeParamsT1.loop1SrcStride = dOutputActual_ * hOutputActual_ * wOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop2SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = tilingData_->dOutput * tilingData_->hOutput * tilingData_->wOutput * sizeof(T1);
    loopModeParamsT1.loop2DstStride = tilingData_->hOutput * tilingData_->wOutput * sizeof(T1);
    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(hOutputActual_),
                                        static_cast<uint32_t>(wOutputActual_ * sizeof(T1)), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((tilingData_->wOutput - wOutputActual_) * sizeof(T1)),
                                        static_cast<uint32_t>(0)};
    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::SingleLineProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr)
{
    int64_t dOutput = tilingData_->dOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutput = tilingData_->wOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curDIndex = dAxisIndex_ * tilingData_->dOutputInner;
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradActual = wGradActual_;
    int64_t wGradAligned = wGradAligned_;
    uint16_t hGradActual = hGradActual_;
    uint16_t dGradActual = dGradActual_;
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);

    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFrontD);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBackD);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

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
        uint32_t highGradOffset = highIdx * dGradActual * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;
        for (uint16_t dIdx = 0; dIdx < dGradActual; dIdx++) {
            T3 dGradOffset = dIdx + dGradActualStart;
            int32_t dIndex = dGradOffset * strideD - curDIndex - padD;
            int32_t dkStart = dIndex >= 0 ? 0 : (-dIndex);
            int32_t dkEnd = (dOutputActual - dIndex) > kD ? kD : (dOutputActual - dIndex);

            for (uint16_t hIdx = 0; hIdx < hGradActual; hIdx++) {
                T3 hGradOffset = hIdx + hGradActualStart;
                int32_t hIndex = hGradOffset * strideH - curHIndex - padH;
                int32_t hkStart = hIndex >= 0 ? 0 : (-hIndex);
                int32_t hkEnd = (hOutputActual - hIndex) > kH ? kH : (hOutputActual - hIndex);

                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    GenInitial1DIndices<int32_t>((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex,
                                                 wProBatchSize);
                    GenInitial1DIndices<T3, Trait>(initialWRegIdx, wProBatchSize);
                    AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));
                    AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));

                    for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T3 wGradOffset = wBatchIdx + wRepeatIdx * computeSizeFp32 * wProBatchSize +
                                             wGradActualStart;
                            uint32_t offset = wBatchIdx + wRepeatIdx * computeSizeFp32 * wProBatchSize +
                                              hIdx * wGradAligned + dIdx * hGradActual * wGradAligned + highGradOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                            ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                            ComputeOutWIndex<T3, Trait>(wIndexReg, outWStart, curWIndex, padW, all);
                            GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                                divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput,
                                padD, padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, all);
                            DoScatterForWParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, all,
                                                                      wOutputAligned, hOutputActual, highOutputOffset,
                                                                      zeroConstReg, wMaxReg, kW, divisorReg, wIndexReg,
                                                                      dIndex, dkStart, dkEnd, hIndex, hkStart, hkEnd);
                        }
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + repeatimes * computeSizeFp32 * wProBatchSize + wGradActualStart;
                        uint32_t offset = wBatchIdx + repeatimes * computeSizeFp32 * wProBatchSize +
                                          hIdx * wGradAligned + dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                        ComputeOutWIndex<T3, Trait>(wIndexReg, outWStart, curWIndex, padW, all);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, wRemainBatchCount);
                        DoScatterForWParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, wRemainBatchCount,
                                                                  wOutputAligned, hOutputActual, highOutputOffset,
                                                                  zeroConstReg, wMaxReg, kW, divisorReg, wIndexReg,
                                                                  dIndex, dkStart, dkEnd, hIndex, hkStart, hkEnd);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        uint32_t wGradOffset = wBatchIdx + wRemainBatchCount * wProBatchSize +
                                               repeatimes * computeSizeFp32 * wProBatchSize + wGradActualStart;
                        uint32_t offset = wBatchIdx + wRemainBatchCount * wProBatchSize +
                                          repeatimes * computeSizeFp32 * wProBatchSize + hIdx * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                        ComputeOutWIndex<T3, Trait>(wIndexReg, outWStart, curWIndex, padW, all);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, one);
                        DoScatterForWParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, one,
                                                                  wOutputAligned, hOutputActual, highOutputOffset,
                                                                  zeroConstReg, wMaxReg, kW, divisorReg, wIndexReg,
                                                                  dIndex, dkStart, dkEnd, hIndex, hkStart, hkEnd);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::MultipleLineProcessVF1(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3)
{
    int64_t dOutput = tilingData_->dOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutput = tilingData_->wOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curDIndex = dAxisIndex_ * tilingData_->dOutputInner;
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradAligned = wGradAligned_;
    int64_t wGradActual = wGradActual_;
    uint16_t hGradActual = hGradActual_;
    uint16_t dGradActual = dGradActual_;
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);

    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFrontD);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBackD);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
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

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                            wGradAligned, wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wGradAligned);
        AscendC::MicroAPI::MaskReg
            allMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(helpAddr, initialRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initialRegIndexOne, allMask);
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
        GenGatterIndex2D<T3, Trait>(initialWRegIdx, 0, wFullBatchCount, wProBatchSize);
        GenGatterIndex2D<T3, Trait>(initialHRegIdx, hProBatchSize, wFullBatchCount, 0);
        GenInitial1DIndices<T3, Trait>(initialHRegIdxOne, hProBatchSize);
        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        AscendC::MicroAPI::DataCopy(helpAddrT3, initialWRegIdx, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3), initialHRegIdx, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3), initialHRegIdxOne,
                                    allMaskT3);
    }

    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highGradOffset = highIdx * dGradActual * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;

        for (uint16_t dIdx = 0; dIdx < dGradActual; dIdx++) {
            T3 dGradOffset = dIdx + dGradActualStart;
            int32_t dIndex = dGradOffset * strideD - curDIndex - padD;
            int32_t dkStart = dIndex >= 0 ? 0 : (-dIndex);
            int32_t dkEnd = (dOutputActual - dIndex) > kD ? kD : (dOutputActual - dIndex);

            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
                        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                        if constexpr (COUNT_PAD == 0) {
                            AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                        }
                        AscendC::MicroAPI::MaskReg
                            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                        AscendC::MicroAPI::DataCopy(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);
                        AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
                        AscendC::MicroAPI::DataCopy(initialHRegIdxOne,
                                                    helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
                        AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));

                        T3 hGradOffset = hProBatchIdx + hIdx * hProBatchSize * hConcurrentCount + hGradActualStart;
                        ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdx, hGradOffset, strideH);
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T3 wGradOffset = wBatchIdx + wGradActualStart;
                            uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                              hIdx * wGradAligned * hProBatchSize * hConcurrentCount +
                                              dIdx * hGradActual * wGradAligned + highGradOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                            ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                            ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex,
                                                         curHIndex, padH, padW, maskBlock);
                            GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                                divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput,
                                padD, padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride,
                                maskBlock);
                            DoScatterForHwParallel<T1, IS_CHECK_RANGE>(
                                yAddr, gradAddr, parallelRegIndex, maskBlock, wOutputAligned, hOutputActual,
                                highOutputOffset, zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg, wIndexReg,
                                hIndexReg, dIndex, dkStart, dkEnd);
                        }
                        ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdxOne, hGradOffset, strideH);
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                            uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount +
                                              hProBatchIdx * wGradAligned +
                                              hIdx * wGradAligned * hProBatchSize * hConcurrentCount +
                                              dIdx * hGradActual * wGradAligned + highGradOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                            AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                            ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex,
                                                         curHIndex, padH, padW, blockOne);
                            GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                                divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput,
                                padD, padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, blockOne);
                            DoScatterForHwParallel<T1, IS_CHECK_RANGE>(
                                yAddr, gradAddr, parallelRegIndex, blockOne, wOutputAligned, hOutputActual,
                                highOutputOffset, zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg, wIndexReg,
                                hIndexReg, dIndex, dkStart, dkEnd);
                        }
                    }
                }
            }
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                }
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
                AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }
                AscendC::MicroAPI::MaskReg
                    allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                AscendC::MicroAPI::DataCopy(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);
                AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
                AscendC::MicroAPI::DataCopy(initialHRegIdxOne,
                                            helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
                AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));

                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    T3 hGradOffset = hProBatchIdx + blockConcurrentCount * hProBatchSize * hConcurrentCount +
                                     hGradActualStart;
                    ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdx, hGradOffset, strideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                          blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, maskRemainBatch);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, maskRemainBatch);
                        DoScatterForHwParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, maskRemainBatch,
                                                                   wOutputAligned, hOutputActual, highOutputOffset,
                                                                   zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg,
                                                                   wIndexReg, hIndexReg, dIndex, dkStart, dkEnd);
                    }
                    ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdxOne, hGradOffset, strideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                          blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, remainBatchOne);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, remainBatchOne);
                        DoScatterForHwParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, remainBatchOne,
                                                                   wOutputAligned, hOutputActual, highOutputOffset,
                                                                   zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg,
                                                                   wIndexReg, hIndexReg, dIndex, dkStart, dkEnd);
                    }
                }
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    T3 hGradOffset = hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                     blockConcurrentCount * hProBatchSize * hConcurrentCount + hGradActualStart;
                    AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                          hRemainBatchCount * hProBatchSize * wGradAligned +
                                          blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, maskRemainTail);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, maskRemainTail);
                        DoScatterForHwParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, maskRemainTail,
                                                                   wOutputAligned, hOutputActual, highOutputOffset,
                                                                   zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg,
                                                                   wIndexReg, hIndexReg, dIndex, dkStart, dkEnd);
                    }
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                          hRemainBatchCount * hProBatchSize * wGradAligned +
                                          blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, remainTailOne);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, remainTailOne);
                        DoScatterForHwParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, remainTailOne,
                                                                   wOutputAligned, hOutputActual, highOutputOffset,
                                                                   zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg,
                                                                   wIndexReg, hIndexReg, dIndex, dkStart, dkEnd);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::MultipleLineProcessVF1Int64(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3)
{
    int64_t dOutput = tilingData_->dOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutput = tilingData_->wOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curDIndex = dAxisIndex_ * tilingData_->dOutputInner;
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradAligned = wGradAligned_;
    int64_t wGradActual = wGradActual_;
    uint16_t hGradActual = hGradActual_;
    uint16_t dGradActual = dGradActual_;
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);

    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFrontD);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBackD);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
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

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
        GenInitial2DIndices((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndex, wProBatchSize, hProBatchSize,
                            wGradAligned, wFullBatchCount);
        Gen2DIndexOne((AscendC::MicroAPI::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize, wGradAligned);
        AscendC::MicroAPI::MaskReg
            allMask = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::DataCopy(helpAddr, initialRegIndex, allMask);
        AscendC::MicroAPI::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initialRegIndexOne, allMask);
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
        GenGatterIndex2D<T3, Trait>(initialWRegIdx, 0, wFullBatchCount, wProBatchSize);
        GenGatterIndex2D<T3, Trait>(initialHRegIdx, hProBatchSize, wFullBatchCount, 0);
        GenInitial1DIndices<T3, Trait>(initialHRegIdxOne, hProBatchSize);
        AscendC::MicroAPI::MaskReg
            allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
        AscendC::MicroAPI::DataCopy(helpAddrT3, initialWRegIdx, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3), initialHRegIdx, allMaskT3);
        AscendC::MicroAPI::DataCopy(helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3), initialHRegIdxOne,
                                    allMaskT3);
    }

    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highGradOffset = highIdx * dGradActual * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;

        for (uint16_t dIdx = 0; dIdx < dGradActual; dIdx++) {
            T3 dGradOffset = dIdx + dGradActualStart;
            int32_t dIndex = dGradOffset * strideD - curDIndex - padD;
            int32_t dkStart = dIndex >= 0 ? 0 : (-dIndex);
            int32_t dkEnd = (dOutputActual - dIndex) > kD ? kD : (dOutputActual - dIndex);

            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                        AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                        if constexpr (COUNT_PAD == 0) {
                            AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                        }
                        AscendC::MicroAPI::MaskReg
                            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                        AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);
                        AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
                        AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));

                        T3 hGradOffset = hProBatchIdx + hIdx * hProBatchSize * hConcurrentCount + hGradActualStart;
                        ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdx, hGradOffset, strideH);
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                            T3 wGradOffset = wBatchIdx + wGradActualStart;
                            uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                              hIdx * wGradAligned * hProBatchSize * hConcurrentCount +
                                              dIdx * hGradActual * wGradAligned + highGradOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                            ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                            ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex,
                                                         curHIndex, padH, padW, maskBlock);
                            GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                                divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput,
                                padD, padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride,
                                maskBlock);
                            DoScatterForHwParallel<T1, IS_CHECK_RANGE>(
                                yAddr, gradAddr, parallelRegIndex, maskBlock, wOutputAligned, hOutputActual,
                                highOutputOffset, zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg, wIndexReg,
                                hIndexReg, dIndex, dkStart, dkEnd);
                        }
                    }
                    __VEC_SCOPE__
                    {
                        AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                        AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                        AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                        if constexpr (IS_CHECK_RANGE == 1) {
                            AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                            AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                        }
                        AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
                        AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                        AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                        AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                        AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                        AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                        if constexpr (COUNT_PAD == 0) {
                            AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                        }
                        AscendC::MicroAPI::MaskReg
                            allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                        AscendC::MicroAPI::DataCopy(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                        AscendC::MicroAPI::DataCopy(initialHRegIdxOne,
                                                    helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
                        AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));

                        T3 hGradOffset = hProBatchIdx + hIdx * hProBatchSize * hConcurrentCount + hGradActualStart;
                        ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdxOne, hGradOffset, strideH);
                        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                            T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                            uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount +
                                              hProBatchIdx * wGradAligned +
                                              hIdx * wGradAligned * hProBatchSize * hConcurrentCount +
                                              dIdx * hGradActual * wGradAligned + highGradOffset;
                            AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                            AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                            ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex,
                                                         curHIndex, padH, padW, blockOne);
                            GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                                divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput,
                                padD, padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, blockOne);
                            DoScatterForHwParallel<T1, IS_CHECK_RANGE>(
                                yAddr, gradAddr, parallelRegIndex, blockOne, wOutputAligned, hOutputActual,
                                highOutputOffset, zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg, wIndexReg,
                                hIndexReg, dIndex, dkStart, dkEnd);
                        }
                    }
                }
            }

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                    AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);
                    AscendC::MicroAPI::DataCopy(initialHRegIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
                    AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));

                    T3 hGradOffset = hProBatchIdx + blockConcurrentCount * hProBatchSize * hConcurrentCount +
                                     hGradActualStart;
                    ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdx, hGradOffset, strideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                          blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, maskRemainBatch);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, maskRemainBatch);
                        DoScatterForHwParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, maskRemainBatch,
                                                                   wOutputAligned, hOutputActual, highOutputOffset,
                                                                   zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg,
                                                                   wIndexReg, hIndexReg, dIndex, dkStart, dkEnd);
                    }
                }
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initialHRegIdxOne;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initialHRegIdxOne,
                                                helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
                    AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));

                    T3 hGradOffset = hProBatchIdx + blockConcurrentCount * hProBatchSize * hConcurrentCount +
                                     hGradActualStart;
                    ComputeOutRegStart<T3, Trait>(outHStart, initialHRegIdxOne, hGradOffset, strideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                          blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, remainBatchOne);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, remainBatchOne);
                        DoScatterForHwParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, remainBatchOne,
                                                                   wOutputAligned, hOutputActual, highOutputOffset,
                                                                   zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg,
                                                                   wIndexReg, hIndexReg, dIndex, dkStart, dkEnd);
                    }
                }
            }

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initialWRegIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initialRegIndex, helpAddr);
                    AscendC::MicroAPI::DataCopy(initialWRegIdx, helpAddrT3);
                    AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));

                    T3 hGradOffset = hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                     blockConcurrentCount * hProBatchSize * hConcurrentCount + hGradActualStart;
                    AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned +
                                          hRemainBatchCount * hProBatchSize * wGradAligned +
                                          blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initialWRegIdx, wGradOffset, strideW);
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, maskRemainTail);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, maskRemainTail);
                        DoScatterForHwParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, maskRemainTail,
                                                                   wOutputAligned, hOutputActual, highOutputOffset,
                                                                   zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg,
                                                                   wIndexReg, hIndexReg, dIndex, dkStart, dkEnd);
                    }
                }
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    }
                    AscendC::MicroAPI::RegTensor<uint32_t> initialRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::DataCopy(initialRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::Duplicate(outDStart, static_cast<T3>(dGradOffset * strideD));

                    T3 hGradOffset = hProBatchIdx + hRemainBatchCount * hProBatchSize +
                                     blockConcurrentCount * hProBatchSize * hConcurrentCount + hGradActualStart;
                    AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                          hRemainBatchCount * hProBatchSize * wGradAligned +
                                          blockConcurrentCount * hConcurrentCount * hProBatchSize * wGradAligned +
                                          dIdx * hGradActual * wGradAligned + highGradOffset;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initialRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutWHIndex<T3, Trait>(wIndexReg, hIndexReg, outWStart, outHStart, curWIndex, curHIndex,
                                                     padH, padW, remainTailOne);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, remainTailOne);
                        DoScatterForHwParallel<T1, IS_CHECK_RANGE>(yAddr, gradAddr, parallelRegIndex, remainTailOne,
                                                                   wOutputAligned, hOutputActual, highOutputOffset,
                                                                   zeroConstReg, hMaxReg, wMaxReg, kH, kW, divisorReg,
                                                                   wIndexReg, hIndexReg, dIndex, dkStart, dkEnd);
                    }
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::MultipleLineProcessVF2(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3)
{
    int64_t dOutput = tilingData_->dOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutput = tilingData_->wOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curDIndex = dAxisIndex_ * tilingData_->dOutputInner;
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradAligned = wGradAligned_;
    int64_t wGradActual = wGradActual_;
    uint16_t hGradActual = hGradActual_;
    uint16_t dGradActual = dGradActual_;
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);

    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFrontD);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBackD);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wGradActual / wProBatchSize;
    uint16_t hFullBatchCount = hGradActual / hProBatchSize;
    uint16_t wRemainTail = wGradActual % wProBatchSize;
    uint32_t whFullBatchCount = wFullBatchCount * hFullBatchCount;

    uint16_t dConcurrentCount = V_REG_SIZE / (whFullBatchCount * sizeof(float));
    uint16_t dBlockConcurrentCount = dGradActual / dConcurrentCount;
    uint16_t dBlockRemainTail = dGradActual - dBlockConcurrentCount * dConcurrentCount;
    uint16_t hRemainTail = hGradActual - hFullBatchCount * hProBatchSize;

    uint32_t mask0 = dConcurrentCount * whFullBatchCount;
    uint32_t mask1 = dConcurrentCount * hFullBatchCount * 1;
    uint32_t mask2 = dConcurrentCount * 1 * wFullBatchCount;
    uint32_t mask3 = dConcurrentCount * 1 * 1;
    uint32_t mask4 = dBlockRemainTail * whFullBatchCount;
    uint32_t mask5 = dBlockRemainTail * hFullBatchCount * 1;
    uint32_t mask6 = dBlockRemainTail * 1 * wFullBatchCount;
    uint32_t mask7 = dBlockRemainTail * 1 * 1;

    GenIndicesToUb(helpAddr, wProBatchSize, hProBatchSize, wGradAligned, wFullBatchCount, hFullBatchCount, hGradActual);
    GenIndicesToUbForT3<T3, Trait>(helpAddrT3, whFullBatchCount, wFullBatchCount, wProBatchSize, hProBatchSize,
                                   hFullBatchCount);

    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highGradOffset = highIdx * dGradActual * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;

        for (uint16_t dBlockIdx = 0; dBlockIdx < dBlockConcurrentCount; ++dBlockIdx) {
            uint32_t dGradBase = dBlockIdx * dConcurrentCount * hGradActual * wGradAligned + highGradOffset;
            uint32_t dBase = dBlockIdx * dConcurrentCount + dGradActualStart;
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                }
                AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));

                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> divisorReg;

                AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegWIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdxOne;
                AscendC::MicroAPI::RegTensor<T3, Trait> initial2DRegWIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }

                AscendC::MicroAPI::MaskReg
                    allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::MaskReg
                    allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr);
                AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
                AscendC::MicroAPI::DataCopy(initial2DRegIndexOne,
                                            helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

                AscendC::MicroAPI::DataCopy(initial3DRegWIdx, helpAddrT3);
                AscendC::MicroAPI::DataCopy(initial3DRegHIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
                AscendC::MicroAPI::DataCopy(initial3DRegHIdxOne,
                                            helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
                AscendC::MicroAPI::DataCopy(initial2DRegWIdx,
                                            helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3));

                GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(whFullBatchCount),
                                            static_cast<T3>(0));
                AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                        allMaskT3);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    T3 hGradOffset = hProBatchIdx + hGradActualStart;
                    ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdx, hGradOffset, strideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned + dGradBase;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initial3DRegWIdx, wGradOffset, strideW);
                        ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                      curDIndex, curHIndex, curWIndex, padD, padH, padW, mask0);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask0);
                        DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask0, wOutputAligned, hOutputActual, highOutputOffset,
                            zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            dIndexReg, highIdxReg);
                    }
                    GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(hFullBatchCount),
                                                static_cast<T3>(0));
                    AscendC::MicroAPI::Adds(outDStart, outDStart,
                                            static_cast<T3>(static_cast<int64_t>(dBase) * strideD), allMaskT3);
                    ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdxOne, hGradOffset, strideH);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                          dGradBase;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                      curDIndex, curHIndex, curWIndex, padD, padH, padW, mask1);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask1);
                        DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask1, wOutputAligned, hOutputActual, highOutputOffset,
                            zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            dIndexReg, highIdxReg);
                    }
                }
                GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(wFullBatchCount),
                                            static_cast<T3>(0));
                AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                        allMaskT3);
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                    T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
                    AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned +
                                          dGradBase;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initial2DRegWIdx, wGradOffset, strideW);
                        ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                      curDIndex, curHIndex, curWIndex, padD, padH, padW, mask2);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask2);
                        DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask2, wOutputAligned, hOutputActual, highOutputOffset,
                            zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            dIndexReg, highIdxReg);
                    }
                    GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(1),
                                                static_cast<T3>(0));
                    AscendC::MicroAPI::Adds(outDStart, outDStart,
                                            static_cast<T3>(static_cast<int64_t>(dBase) * strideD), allMaskT3);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount +
                                          (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned + dGradBase;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                      curDIndex, curHIndex, curWIndex, padD, padH, padW, mask3);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask3);
                        DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask3, wOutputAligned, hOutputActual, highOutputOffset,
                            zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            dIndexReg, highIdxReg);
                    }
                }
            }
        }

        uint32_t dGradBase = dBlockConcurrentCount * dConcurrentCount * hGradActual * wGradAligned + highGradOffset;
        uint32_t dBase = dBlockConcurrentCount * dConcurrentCount + dGradActualStart;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
            AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
            AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
            AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
            AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
            if constexpr (IS_CHECK_RANGE == 1) {
                AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
            }
            AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));

            AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
            AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
            AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
            AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
            AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
            AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
            AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
            AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
            AscendC::MicroAPI::RegTensor<int32_t> divisorReg;

            AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdx;
            AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegWIdx;
            AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdxOne;
            AscendC::MicroAPI::RegTensor<T3, Trait> initial2DRegWIdx;
            AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
            AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
            AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
            AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
            if constexpr (COUNT_PAD == 0) {
                AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
            }

            AscendC::MicroAPI::MaskReg
                allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg
                allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
            AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr);
            AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
            AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
            AscendC::MicroAPI::DataCopy(initial2DRegIndexOne, helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

            AscendC::MicroAPI::DataCopy(initial3DRegWIdx, helpAddrT3);
            AscendC::MicroAPI::DataCopy(initial3DRegHIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));
            AscendC::MicroAPI::DataCopy(initial3DRegHIdxOne,
                                        helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));
            AscendC::MicroAPI::DataCopy(initial2DRegWIdx,
                                        helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3));

            GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(whFullBatchCount),
                                        static_cast<T3>(0));
            AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                    allMaskT3);
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                T3 hGradOffset = hProBatchIdx + hGradActualStart;
                ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdx, hGradOffset, strideH);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned + dGradBase;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                    ComputeOutRegStart<T3, Trait>(outWStart, initial3DRegWIdx, wGradOffset, strideW);
                    ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                  curDIndex, curHIndex, curWIndex, padD, padH, padW, mask4);
                    GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                        padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask4);
                    DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask4, wOutputAligned, hOutputActual, highOutputOffset,
                        zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        dIndexReg, highIdxReg);
                }
                ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdxOne, hGradOffset, strideH);
                GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(hFullBatchCount),
                                            static_cast<T3>(0));
                AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                        allMaskT3);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                      dGradBase;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                  curDIndex, curHIndex, curWIndex, padD, padH, padW, mask5);
                    GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                        padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask5);
                    DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask5, wOutputAligned, hOutputActual, highOutputOffset,
                        zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        dIndexReg, highIdxReg);
                }
            }
            GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(wFullBatchCount),
                                        static_cast<T3>(0));
            AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                    allMaskT3);
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
                AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned +
                                      dGradBase;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                    ComputeOutRegStart<T3, Trait>(outWStart, initial2DRegWIdx, wGradOffset, strideW);
                    ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                  curDIndex, curHIndex, curWIndex, padD, padH, padW, mask6);
                    GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                        padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask6);
                    DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask6, wOutputAligned, hOutputActual, highOutputOffset,
                        zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        dIndexReg, highIdxReg);
                }
                GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(1),
                                            static_cast<T3>(0));
                AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                        allMaskT3);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount +
                                      (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned + dGradBase;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                  curDIndex, curHIndex, curWIndex, padD, padH, padW, mask7);
                    GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                        padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask7);
                    DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask7, wOutputAligned, hOutputActual, highOutputOffset,
                        zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        dIndexReg, highIdxReg);
                }
            }
        }
    }
}

template <typename T1, typename T3, const uint32_t HAS_DIVISOR, const uint32_t IS_CHECK_RANGE, const uint32_t COUNT_PAD>
template <const MicroAPI::RegTrait& Trait>
__aicore__ inline void AvgPool3DGradNCDHW<T1, T3, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>::MultipleLineProcessVF2Int64(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ uint32_t* helpAddr,
    __local_mem__ T3* helpAddrT3)
{
    int64_t dOutput = tilingData_->dOutput;
    int64_t hOutput = tilingData_->hOutput;
    int64_t wOutput = tilingData_->wOutput;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int64_t dOutputActual = dOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curDIndex = dAxisIndex_ * tilingData_->dOutputInner;
    int64_t curHIndex = hAxisIndex_ * tilingData_->hOutputInner;
    int64_t curWIndex = wAxisIndex_ * tilingData_->wOutputInner;
    int64_t wGradAligned = wGradAligned_;
    int64_t wGradActual = wGradActual_;
    uint16_t hGradActual = hGradActual_;
    uint16_t dGradActual = dGradActual_;
    uint32_t dGradActualStart = static_cast<uint32_t>(dGradActualStart_);
    uint32_t hGradActualStart = static_cast<uint32_t>(hGradActualStart_);
    uint32_t wGradActualStart = static_cast<uint32_t>(wGradActualStart_);
    int32_t divisorOverride = static_cast<int32_t>(tilingData_->divisorOverride);

    uint16_t kD = static_cast<uint16_t>(tilingData_->dKernel);
    uint16_t kH = static_cast<uint16_t>(tilingData_->hKernel);
    uint16_t kW = static_cast<uint16_t>(tilingData_->wKernel);
    uint16_t padD = static_cast<uint16_t>(tilingData_->padFrontD);
    uint16_t padH = static_cast<uint16_t>(tilingData_->padTopH);
    uint16_t padW = static_cast<uint16_t>(tilingData_->padLeftW);
    uint16_t padBackD = static_cast<uint16_t>(tilingData_->padBackD);
    uint16_t padDownH = static_cast<uint16_t>(tilingData_->padDownH);
    uint16_t padRightW = static_cast<uint16_t>(tilingData_->padRightW);
    uint32_t strideD = static_cast<uint32_t>(tilingData_->dStride);
    uint32_t strideH = static_cast<uint32_t>(tilingData_->hStride);
    uint32_t strideW = static_cast<uint32_t>(tilingData_->wStride);

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wGradActual / wProBatchSize;
    uint16_t hFullBatchCount = hGradActual / hProBatchSize;
    uint16_t wRemainTail = wGradActual % wProBatchSize;
    uint32_t whFullBatchCount = wFullBatchCount * hFullBatchCount;

    uint16_t dConcurrentCount = V_REG_SIZE / (whFullBatchCount * sizeof(float));
    uint16_t dBlockConcurrentCount = dGradActual / dConcurrentCount;
    uint16_t dBlockRemainTail = dGradActual - dBlockConcurrentCount * dConcurrentCount;
    uint16_t hRemainTail = hGradActual - hFullBatchCount * hProBatchSize;

    uint32_t mask0 = dConcurrentCount * whFullBatchCount;
    uint32_t mask1 = dConcurrentCount * hFullBatchCount * 1;
    uint32_t mask2 = dConcurrentCount * 1 * wFullBatchCount;
    uint32_t mask3 = dConcurrentCount * 1 * 1;
    uint32_t mask4 = dBlockRemainTail * whFullBatchCount;
    uint32_t mask5 = dBlockRemainTail * hFullBatchCount * 1;
    uint32_t mask6 = dBlockRemainTail * 1 * wFullBatchCount;
    uint32_t mask7 = dBlockRemainTail * 1 * 1;

    GenIndicesToUb(helpAddr, wProBatchSize, hProBatchSize, wGradAligned, wFullBatchCount, hFullBatchCount, hGradActual);
    GenIndicesToUbForT3<T3, Trait>(helpAddrT3, whFullBatchCount, wFullBatchCount, wProBatchSize, hProBatchSize,
                                   hFullBatchCount);

    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        uint32_t highGradOffset = highIdx * dGradActual * hGradActual * wGradAligned;
        uint32_t highOutputOffset = highIdx * dOutputActual * hOutputActual * wOutputAligned;

        for (uint16_t dBlockIdx = 0; dBlockIdx < dBlockConcurrentCount; ++dBlockIdx) {
            uint32_t dGradBase = dBlockIdx * dConcurrentCount * hGradActual * wGradAligned + highGradOffset;
            uint32_t dBase = dBlockIdx * dConcurrentCount + dGradActualStart;

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                        AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                    }
                    AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegWIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::MaskReg
                        allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                    AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr);
                    AscendC::MicroAPI::DataCopy(initial3DRegWIdx, helpAddrT3);
                    AscendC::MicroAPI::DataCopy(initial3DRegHIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));

                    T3 hGradOffset = hProBatchIdx + hGradActualStart;
                    ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdx, hGradOffset, strideH);
                    GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(whFullBatchCount),
                                                static_cast<T3>(0));
                    AscendC::MicroAPI::Adds(outDStart, outDStart,
                                            static_cast<T3>(static_cast<int64_t>(dBase) * strideD), allMaskT3);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned + dGradBase;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initial3DRegWIdx, wGradOffset, strideW);
                        ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                      curDIndex, curHIndex, curWIndex, padD, padH, padW, mask0);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask0);
                        DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask0, wOutputAligned, hOutputActual, highOutputOffset,
                            zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            dIndexReg, highIdxReg);
                    }
                }
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                        AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                    }
                    AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdxOne;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::MaskReg
                        allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                    AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial3DRegHIdxOne,
                                                helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));

                    T3 hGradOffset = hProBatchIdx + hGradActualStart;
                    ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdxOne, hGradOffset, strideH);
                    GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(hFullBatchCount),
                                                static_cast<T3>(0));
                    AscendC::MicroAPI::Adds(outDStart, outDStart,
                                            static_cast<T3>(static_cast<int64_t>(dBase) * strideD), allMaskT3);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                          dGradBase;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                      curDIndex, curHIndex, curWIndex, padD, padH, padW, mask1);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask1);
                        DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask1, wOutputAligned, hOutputActual, highOutputOffset,
                            zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            dIndexReg, highIdxReg);
                    }
                }
            }

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                        AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                    }
                    AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> initial2DRegWIdx;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::MaskReg
                        allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                    AscendC::MicroAPI::DataCopy(initial2DRegIndex,
                                                helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
                    AscendC::MicroAPI::DataCopy(initial2DRegWIdx,
                                                helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3));

                    T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
                    AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                    GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(wFullBatchCount),
                                                static_cast<T3>(0));
                    AscendC::MicroAPI::Adds(outDStart, outDStart,
                                            static_cast<T3>(static_cast<int64_t>(dBase) * strideD), allMaskT3);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wGradActualStart;
                        uint32_t offset = wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned +
                                          dGradBase;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                        ComputeOutRegStart<T3, Trait>(outWStart, initial2DRegWIdx, wGradOffset, strideW);
                        ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                      curDIndex, curHIndex, curWIndex, padD, padH, padW, mask2);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask2);
                        DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask2, wOutputAligned, hOutputActual, highOutputOffset,
                            zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            dIndexReg, highIdxReg);
                    }
                }
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                    AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                    if constexpr (IS_CHECK_RANGE == 1) {
                        AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                        AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                        AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                        AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                    }
                    AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                    AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                    AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                    AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                    AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                    if constexpr (COUNT_PAD == 0) {
                        AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                    }
                    AscendC::MicroAPI::MaskReg
                        allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::MaskReg
                        allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                    AscendC::MicroAPI::DataCopy(initial2DRegIndexOne,
                                                helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

                    T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
                    AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                    GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(1),
                                                static_cast<T3>(0));
                    AscendC::MicroAPI::Adds(outDStart, outDStart,
                                            static_cast<T3>(static_cast<int64_t>(dBase) * strideD), allMaskT3);
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                        uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount +
                                          (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned + dGradBase;
                        AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                        AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                        ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                      curDIndex, curHIndex, curWIndex, padD, padH, padW, mask3);
                        GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                            divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                            padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask3);
                        DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                            yAddr, gradAddr, parallelRegIndex, mask3, wOutputAligned, hOutputActual, highOutputOffset,
                            zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                            dIndexReg, highIdxReg);
                    }
                }
            }
        }

        uint32_t dGradBase = dBlockConcurrentCount * dConcurrentCount * hGradActual * wGradAligned + highGradOffset;
        uint32_t dBase = dBlockConcurrentCount * dConcurrentCount + dGradActualStart;

        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                }
                AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegWIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }
                AscendC::MicroAPI::MaskReg
                    allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::MaskReg
                    allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                AscendC::MicroAPI::DataCopy(initial3DRegIndex, helpAddr);
                AscendC::MicroAPI::DataCopy(initial3DRegWIdx, helpAddrT3);
                AscendC::MicroAPI::DataCopy(initial3DRegHIdx, helpAddrT3 + INDEX_TWO * V_REG_SIZE / sizeof(T3));

                T3 hGradOffset = hProBatchIdx + hGradActualStart;
                ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdx, hGradOffset, strideH);
                GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(whFullBatchCount),
                                            static_cast<T3>(0));
                AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                        allMaskT3);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = wBatchIdx + hProBatchIdx * wGradAligned + dGradBase;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndex, offset, allMaskU32);
                    ComputeOutRegStart<T3, Trait>(outWStart, initial3DRegWIdx, wGradOffset, strideW);
                    ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                  curDIndex, curHIndex, curWIndex, padD, padH, padW, mask4);
                    GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                        padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask4);
                    DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask4, wOutputAligned, hOutputActual, highOutputOffset,
                        zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        dIndexReg, highIdxReg);
                }
            }
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                }
                AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));
                AscendC::MicroAPI::RegTensor<uint32_t> initial3DRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                AscendC::MicroAPI::RegTensor<T3, Trait> initial3DRegHIdxOne;
                AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }
                AscendC::MicroAPI::MaskReg
                    allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::MaskReg
                    allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                AscendC::MicroAPI::DataCopy(initial3DRegIndexOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
                AscendC::MicroAPI::DataCopy(initial3DRegHIdxOne,
                                            helpAddrT3 + INDEX_TWO * INDEX_TWO * V_REG_SIZE / sizeof(T3));

                T3 hGradOffset = hProBatchIdx + hGradActualStart;
                ComputeOutRegStart<T3, Trait>(outHStart, initial3DRegHIdxOne, hGradOffset, strideH);
                GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(hFullBatchCount),
                                            static_cast<T3>(0));
                AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                        allMaskT3);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wGradAligned +
                                      dGradBase;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial3DRegIndexOne, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                  curDIndex, curHIndex, curWIndex, padD, padH, padW, mask5);
                    GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                        padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask5);
                    DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask5, wOutputAligned, hOutputActual, highOutputOffset,
                        zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        dIndexReg, highIdxReg);
                }
            }
        }

        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                }
                AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndex;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                AscendC::MicroAPI::RegTensor<T3, Trait> initial2DRegWIdx;
                AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }
                AscendC::MicroAPI::MaskReg
                    allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::MaskReg
                    allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                AscendC::MicroAPI::DataCopy(initial2DRegIndex, helpAddr + INDEX_TWO * V_REG_SIZE / sizeof(uint32_t));
                AscendC::MicroAPI::DataCopy(initial2DRegWIdx,
                                            helpAddrT3 + INDEX_THREE * INDEX_TWO * V_REG_SIZE / sizeof(T3));

                T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
                AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(wFullBatchCount),
                                            static_cast<T3>(0));
                AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                        allMaskT3);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wGradActualStart;
                    uint32_t offset = wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned +
                                      dGradBase;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndex, offset, allMaskU32);
                    ComputeOutRegStart<T3, Trait>(outWStart, initial2DRegWIdx, wGradOffset, strideW);
                    ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                  curDIndex, curHIndex, curWIndex, padD, padH, padW, mask6);
                    GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                        padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask6);
                    DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask6, wOutputAligned, hOutputActual, highOutputOffset,
                        zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        dIndexReg, highIdxReg);
                }
            }
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<int32_t> zeroConstReg;
                AscendC::MicroAPI::RegTensor<int32_t> dMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> hMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> wMaxReg;
                AscendC::MicroAPI::RegTensor<int32_t> highIdxReg;
                if constexpr (IS_CHECK_RANGE == 1) {
                    AscendC::MicroAPI::Duplicate(zeroConstReg, static_cast<int32_t>(0));
                    AscendC::MicroAPI::Duplicate(dMaxReg, static_cast<int32_t>(dOutputActual));
                    AscendC::MicroAPI::Duplicate(hMaxReg, static_cast<int32_t>(hOutputActual));
                    AscendC::MicroAPI::Duplicate(wMaxReg, static_cast<int32_t>(wOutputActual));
                }
                AscendC::MicroAPI::Duplicate(highIdxReg, static_cast<int32_t>(0));
                AscendC::MicroAPI::RegTensor<uint32_t> initial2DRegIndexOne;
                AscendC::MicroAPI::RegTensor<uint32_t> parallelRegIndex;
                AscendC::MicroAPI::RegTensor<int32_t> wIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> hIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> dIndexReg;
                AscendC::MicroAPI::RegTensor<int32_t> divisorReg;
                AscendC::MicroAPI::RegTensor<T3, Trait> outWStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outHStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> outDStart;
                AscendC::MicroAPI::RegTensor<T3, Trait> zeroConstRegT;
                if constexpr (COUNT_PAD == 0) {
                    AscendC::MicroAPI::Duplicate(zeroConstRegT, static_cast<T3>(0));
                }
                AscendC::MicroAPI::MaskReg
                    allMaskU32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::MaskReg
                    allMaskT3 = AscendC::MicroAPI::CreateMask<T3, AscendC::MicroAPI::MaskPattern::ALL, Trait>();
                AscendC::MicroAPI::DataCopy(initial2DRegIndexOne,
                                            helpAddr + INDEX_THREE * V_REG_SIZE / sizeof(uint32_t));

                T3 hGradOffset = hProBatchIdx + hProBatchSize * hFullBatchCount + hGradActualStart;
                AscendC::MicroAPI::Duplicate(outHStart, static_cast<T3>(hGradOffset * strideH));
                GenGatterIndex2D<T3, Trait>(outDStart, static_cast<T3>(strideD), static_cast<T3>(1),
                                            static_cast<T3>(0));
                AscendC::MicroAPI::Adds(outDStart, outDStart, static_cast<T3>(static_cast<int64_t>(dBase) * strideD),
                                        allMaskT3);
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    T3 wGradOffset = wBatchIdx + wProBatchSize * wFullBatchCount + wGradActualStart;
                    uint32_t offset = wBatchIdx + wProBatchSize * wFullBatchCount +
                                      (hProBatchSize * hFullBatchCount + hProBatchIdx) * wGradAligned + dGradBase;
                    AscendC::MicroAPI::Adds(parallelRegIndex, initial2DRegIndexOne, offset, allMaskU32);
                    AscendC::MicroAPI::Duplicate(outWStart, static_cast<T3>(wGradOffset * strideW));
                    ComputeOutDHWIndex<T3, Trait>(dIndexReg, hIndexReg, wIndexReg, outDStart, outHStart, outWStart,
                                                  curDIndex, curHIndex, curWIndex, padD, padH, padW, mask7);
                    GenDivisor3D<T3, Trait, HAS_DIVISOR, IS_CHECK_RANGE, COUNT_PAD>(
                        divisorReg, outDStart, outHStart, outWStart, zeroConstRegT, dOutput, hOutput, wOutput, padD,
                        padH, padW, padBackD, padDownH, padRightW, kD, kH, kW, divisorOverride, mask7);
                    DoScatterForDhwParallel<T1, IS_CHECK_RANGE>(
                        yAddr, gradAddr, parallelRegIndex, mask7, wOutputAligned, hOutputActual, highOutputOffset,
                        zeroConstReg, dMaxReg, hMaxReg, wMaxReg, kD, kH, kW, divisorReg, wIndexReg, hIndexReg,
                        dIndexReg, highIdxReg);
                }
            }
        }
    }
}

} // namespace AvgPool3DGradNCDHWNameSpace
#endif // AVG_POOL3_D_GRAD_NCDHW_KERNEL_H_
