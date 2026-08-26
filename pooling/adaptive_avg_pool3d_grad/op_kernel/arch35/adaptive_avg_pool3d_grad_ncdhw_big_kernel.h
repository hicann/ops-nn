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
 * \file adaptive_avg_pool3d_grad_ncdhw_big_kernel_impl.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_BIG_KERNEL_IMPL_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_BIG_KERNEL_IMPL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "adaptive_avg_pool3d_grad_struct.h"

namespace AdaptiveAvgPool3dGradOp {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DOUBLE = 2;

using COMPUTE_TYPE = float;

template <typename T>
__aicore__ inline T GetStartFromOutputInputSize(T idx, T sizeA, T sizeB)
{
    return (idx * sizeA) / sizeB;
}

template <typename T>
__aicore__ inline T GetEndFromOutputInputSize(T idx, T sizeA, T sizeB)
{
    return ((idx + 1) * sizeA + sizeB - 1) / sizeB;
}

__aicore__ inline void PIPE_S_V()
{
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

__aicore__ inline void PIPE_V_S()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

constexpr AscendC::Reg::CastTrait castTraitI32TF32 = {
    AscendC::Reg::RegLayout::UNKNOWN,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait castTraitI64TF32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait castTraitTF32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitI64I32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr AscendC::Reg::CastTrait castTraitU32U16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait castTraitInt32Int64 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T, typename INDEX>
class AdaptiveAvgPool3dGradNCDHWBigKernel {
public:
    __aicore__ inline AdaptiveAvgPool3dGradNCDHWBigKernel(void){};
    __aicore__ inline void Init(GM_ADDR gradInput, GM_ADDR y, TPipe& pipeIn,
                                const AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35& tilingData);
    __aicore__ inline void ParseTilingData(const AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();
    __aicore__ inline void CalGradInputGMDHW(
        AscendC::Reg::RegTensor<INDEX>& gradInputUbIdx, AscendC::Reg::RegTensor<INDEX>& gradInputHigh,
        AscendC::Reg::RegTensor<INDEX>& gradInputD, AscendC::Reg::RegTensor<INDEX>& gradInputH,
        AscendC::Reg::RegTensor<INDEX>& gradInputW, AscendC::Reg::RegTensor<INDEX>& gradInputDHWValueConstReg,
        AscendC::Reg::RegTensor<INDEX>& gradInputHWValueConstReg,
        AscendC::Reg::RegTensor<INDEX>& gradInputWValueConstReg, AscendC::Reg::MaskReg& allMaskIndex);
    __aicore__ inline void CalAndCopyGradInputStAndEdSingle(
        AscendC::Reg::RegTensor<INDEX>& gradKernelRegConstBig, AscendC::Reg::RegTensor<INDEX>& gradKernelRegConstSmall,
        AscendC::Reg::RegTensor<INDEX>& gradKernelKernelStIdx, AscendC::Reg::RegTensor<INDEX>& gradKernelKernelEdIdx,
        AscendC::Reg::RegTensor<INDEX>& gradInputIdxValue, AscendC::Reg::RegTensor<INDEX>& gradKernelKernelMulTemp,
        AscendC::Reg::RegTensor<INDEX>& gradKernelSize, __ubuf__ INDEX* gmStAddr, __ubuf__ INDEX* gmEdAddr,
        INDEX idxOutput_, INDEX idxGradInput_, INDEX idxAxisIndex_, INDEX idxInner_, INDEX idxOutputActual_,
        AscendC::Reg::MaskReg& allMaskIndex);
    __aicore__ inline void CalAndCopyGradInputStAndEdDHW(
        AscendC::Reg::RegTensor<INDEX>& gradKernelRegConstBig, AscendC::Reg::RegTensor<INDEX>& gradKernelRegConstSmall,
        AscendC::Reg::RegTensor<INDEX>& gradKernelKernelStIdx, AscendC::Reg::RegTensor<INDEX>& gradKernelKernelEdIdx,
        AscendC::Reg::RegTensor<INDEX>& gradInputD, AscendC::Reg::RegTensor<INDEX>& gradInputH,
        AscendC::Reg::RegTensor<INDEX>& gradInputW, AscendC::Reg::RegTensor<INDEX>& gradKernelKernelMulTemp,
        AscendC::Reg::RegTensor<INDEX>& gradKernelSize, __ubuf__ INDEX* gmStDAddr, __ubuf__ INDEX* gmEdDAddr,
        __ubuf__ INDEX* gmStHAddr, __ubuf__ INDEX* gmEdHAddr, __ubuf__ INDEX* gmStWAddr, __ubuf__ INDEX* gmEdWAddr,
        __ubuf__ INDEX* gmKernelSizeAddr, AscendC::Reg::MaskReg& allMaskIndex);
    __aicore__ inline void CalAndCopyGradInputInfo(uint32_t gradInputUbIdxValue, __ubuf__ INDEX* gmStDAddr,
                                                   __ubuf__ INDEX* gmEdDAddr, __ubuf__ INDEX* gmStHAddr,
                                                   __ubuf__ INDEX* gmEdHAddr, __ubuf__ INDEX* gmStWAddr,
                                                   __ubuf__ INDEX* gmEdWAddr, __ubuf__ INDEX* gmHighIdxAddr,
                                                   __ubuf__ INDEX* gmKernelSizeAddr);
    __aicore__ inline void DoGradInputAccUb(uint32_t gradInputLoopCount, uint32_t gradInputUbIdxValue,
                                            __ubuf__ T* gradAddr, __ubuf__ COMPUTE_TYPE* yAddr,
                                            __ubuf__ INDEX* gmStDAddr, __ubuf__ INDEX* gmEdDAddr,
                                            __ubuf__ INDEX* gmStHAddr, __ubuf__ INDEX* gmEdHAddr,
                                            __ubuf__ INDEX* gmStWAddr, __ubuf__ INDEX* gmEdWAddr,
                                            __ubuf__ INDEX* gmHighIdxAddr, __ubuf__ INDEX* gmKernelSizeAddr,
                                            __ubuf__ COMPUTE_TYPE* gmGradInputF32Addr);
    __aicore__ inline void DoAllGradInputProcess(uint32_t gradInputLoopCount, uint32_t gradInputUbIdxValue,
                                                 __ubuf__ T* gradAddr, __ubuf__ COMPUTE_TYPE* yAddr,
                                                 __ubuf__ INDEX* gmStDAddr, __ubuf__ INDEX* gmEdDAddr,
                                                 __ubuf__ INDEX* gmStHAddr, __ubuf__ INDEX* gmEdHAddr,
                                                 __ubuf__ INDEX* gmStWAddr, __ubuf__ INDEX* gmEdWAddr,
                                                 __ubuf__ INDEX* gmHighIdxAddr, __ubuf__ INDEX* gmKernelSizeAddr,
                                                 __ubuf__ COMPUTE_TYPE* gmGradInputF32Addr);
    __aicore__ inline void GatherCopyGradUb2Reg(AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx,
                                                AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
                                                __ubuf__ COMPUTE_TYPE* yAddr, uint32_t& maskCount);
    __aicore__ inline void ScatterCopyGradReg2Ub(AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx,
                                                 AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
                                                 __ubuf__ COMPUTE_TYPE* yAddr, uint32_t& maskCount);
    __aicore__ inline void DoGradRegAdds(AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
                                         COMPUTE_TYPE& gradInputValue, __ubuf__ COMPUTE_TYPE* yAddr,
                                         uint32_t& maskCount);
    __aicore__ inline void ComputeGradIndexHW(AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx, INDEX gradKernelW,
                                              INDEX gradStHIdxOffset, INDEX gradStWIdx, INDEX highDHOffset,
                                              AscendC::Reg::MaskReg& pregU32MaskAll);
    __aicore__ inline void ComputeGradIndexDHW(AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx, INDEX gradKernelHW,
                                               INDEX gradKernelW, INDEX gradStDIdxOffset, INDEX outputHWAlign,
                                               INDEX gradStHIdx, INDEX gradStWIdx, INDEX highOffset,
                                               AscendC::Reg::MaskReg& pregU32MaskAll);
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradInputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;

    TBuf<QuePosition::VECCALC> gmKernelSize;
    TBuf<QuePosition::VECCALC> gmhighIdx;
    TBuf<QuePosition::VECCALC> gmStD;
    TBuf<QuePosition::VECCALC> gmEdD;
    TBuf<QuePosition::VECCALC> gmStH;
    TBuf<QuePosition::VECCALC> gmEdH;
    TBuf<QuePosition::VECCALC> gmStW;
    TBuf<QuePosition::VECCALC> gmEdW;
    TBuf<QuePosition::VECCALC> gmGradInputF32;

    GlobalTensor<T> gradInputGm_;
    GlobalTensor<T> yGm_;

    uint32_t blockIdx_ = 0;

    int64_t dOutput_ = 1;
    int64_t hOutput_ = 1;
    int64_t wOutput_ = 1;

    int64_t dGradInput_ = 1;
    int64_t hGradInput_ = 1;
    int64_t wGradInput_ = 1;

    int64_t highAxisInner_ = 1;
    int64_t highAxisTail_ = 1;
    int64_t highAxisOuter_ = 1;
    int64_t highAxisActual_ = 1;

    int64_t dOutputInner_ = 1;
    int64_t dOutputTail_ = 1;
    int64_t dOutputOuter_ = 1;
    int64_t dOutputActual_ = 1;

    int64_t hOutputInner_ = 1;
    int64_t hOutputTail_ = 1;
    int64_t hOutputOuter_ = 1;
    int64_t hOutputActual_ = 1;

    int64_t wOutputInner_ = 1;
    int64_t wOutputTail_ = 1;
    int64_t wOutputOuter_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t wOutputAligned_ = 1;

    int64_t normalCoreProcessNum_ = 1;
    int64_t tailCoreProcessNum_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t usedCoreNum_ = 1;

    int64_t outputBufferSize_ = 1;
    int64_t gradInputBufferSize_ = 1;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;

    int64_t hGradInputActual_ = 0;
    int64_t dGradInputActual_ = 0;
    int64_t wGradInputActual_ = 0;
    int64_t wGradInputAligned_ = 0;

    int64_t gradInputPlaneSize_ = 0;

    int64_t highAxisGradInputOffset_ = 0;
    int64_t hAxisGradInputOffset_ = 0;
    int64_t dAxisGradInputOffset_ = 0;
    int64_t wAxisGradInputOffset_ = 0;

    int64_t dStLeftCornerIdx = 0;
    int64_t hStLeftCornerIdx = 0;
    int64_t wStLeftCornerIdx = 0;
    int64_t dEndRightCornerIdx = 0;
    int64_t hEndRightCornerIdx = 0;
    int64_t wEndRightCornerIdx = 0;

    int64_t gradInputLoopCount = 0;
    int64_t gradInputLoopTail = 0;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T);

    constexpr static int64_t GRADINPUT_ONE_VL = platform::GetVRegSize() / sizeof(INDEX);

    using INDEX_U = std::conditional_t<std::is_same_v<INDEX, int32_t>, uint32_t, uint64_t>;
};

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::ParseTilingData(
    const AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35& tilingData)
{
    dGradInput_ = tilingData.dInput;
    hGradInput_ = tilingData.hInput;
    wGradInput_ = tilingData.wInput;

    dOutput_ = tilingData.dOutput;
    hOutput_ = tilingData.hOutput;
    wOutput_ = tilingData.wOutput;

    highAxisInner_ = tilingData.highAxisInner;
    highAxisTail_ = tilingData.highAxisTail;
    highAxisOuter_ = tilingData.highAxisOuter;

    dOutputInner_ = tilingData.dOutputInner;
    dOutputTail_ = tilingData.dOutputTail;
    dOutputOuter_ = tilingData.dOutputOuter;

    hOutputInner_ = tilingData.hOutputInner;
    hOutputTail_ = tilingData.hOutputTail;
    hOutputOuter_ = tilingData.hOutputOuter;

    wOutputInner_ = tilingData.wOutputInner;
    wOutputTail_ = tilingData.wOutputTail;
    wOutputOuter_ = tilingData.wOutputOuter;

    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    usedCoreNum_ = tilingData.usedCoreNum;

    outputBufferSize_ = tilingData.outputBufferSize;
    gradInputBufferSize_ = tilingData.gradInputBufferSize;
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::Init(
    GM_ADDR gradInput, GM_ADDR y, TPipe& pipeIn, const AdaptiveAvgPool3dNCDHWGradBigKernelTilingDataV35& tilingData)
{
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    gradInputPlaneSize_ = dGradInput_ * hGradInput_ * wGradInput_;
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    curCoreProcessNum_ = (blockIdx_ + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;
    gradInputGm_.SetGlobalBuffer((__gm__ T*)gradInput);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_ = pipeIn;
    pipe_.InitBuffer(outputQue_, BUFFER_NUM, outputBufferSize_);
    pipe_.InitBuffer(gradInputQue_, BUFFER_NUM, gradInputBufferSize_);

    pipe_.InitBuffer(gmhighIdx, V_REG_SIZE);
    pipe_.InitBuffer(gmKernelSize, V_REG_SIZE);
    pipe_.InitBuffer(gmStD, V_REG_SIZE);
    pipe_.InitBuffer(gmEdD, V_REG_SIZE);
    pipe_.InitBuffer(gmStH, V_REG_SIZE);
    pipe_.InitBuffer(gmEdH, V_REG_SIZE);
    pipe_.InitBuffer(gmStW, V_REG_SIZE);
    pipe_.InitBuffer(gmEdW, V_REG_SIZE);
    pipe_.InitBuffer(gmGradInputF32, V_REG_SIZE);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;
    highAxisIndex_ = baseBlockIdx / (dOutputOuter_ * hOutputOuter_ * wOutputOuter_);
    highAxisActual_ = highAxisIndex_ == (highAxisOuter_ - 1) ? highAxisTail_ : highAxisInner_;
    int64_t tempTail = baseBlockIdx % (dOutputOuter_ * hOutputOuter_ * wOutputOuter_);
    dAxisIndex_ = tempTail / (hOutputOuter_ * wOutputOuter_);
    dOutputActual_ = dAxisIndex_ == (dOutputOuter_ - 1) ? dOutputTail_ : dOutputInner_;
    int64_t tempTail2 = tempTail % (hOutputOuter_ * wOutputOuter_);
    hAxisIndex_ = tempTail2 / wOutputOuter_;
    hOutputActual_ = hAxisIndex_ == (hOutputOuter_ - 1) ? hOutputTail_ : hOutputInner_;
    wAxisIndex_ = tempTail2 % wOutputOuter_;
    wOutputActual_ = wAxisIndex_ == (wOutputOuter_ - 1) ? wOutputTail_ : wOutputInner_;

    dStLeftCornerIdx = GetStartFromOutputInputSize(dAxisIndex_ * dOutputInner_, dGradInput_, dOutput_);
    hStLeftCornerIdx = GetStartFromOutputInputSize(hAxisIndex_ * hOutputInner_, hGradInput_, hOutput_);
    wStLeftCornerIdx = GetStartFromOutputInputSize(wAxisIndex_ * wOutputInner_, wGradInput_, wOutput_);
    dEndRightCornerIdx = GetEndFromOutputInputSize(dAxisIndex_ * dOutputInner_ + dOutputActual_ - 1, dGradInput_,
                                                   dOutput_);
    hEndRightCornerIdx = GetEndFromOutputInputSize(hAxisIndex_ * hOutputInner_ + hOutputActual_ - 1, hGradInput_,
                                                   hOutput_);
    wEndRightCornerIdx = GetEndFromOutputInputSize(wAxisIndex_ * wOutputInner_ + wOutputActual_ - 1, wGradInput_,
                                                   wOutput_);

    wOutputAligned_ = (wOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK *
                      MAX_DATA_NUM_IN_ONE_BLOCK;

    wGradInputActual_ = wEndRightCornerIdx - wStLeftCornerIdx;

    hGradInputActual_ = hEndRightCornerIdx - hStLeftCornerIdx;
    dGradInputActual_ = dEndRightCornerIdx - dStLeftCornerIdx;

    highAxisGradInputOffset_ = highAxisIndex_ * highAxisInner_ * gradInputPlaneSize_;
    dAxisGradInputOffset_ = dStLeftCornerIdx * hGradInput_ * wGradInput_;
    hAxisGradInputOffset_ = hStLeftCornerIdx * wGradInput_;
    wAxisGradInputOffset_ = wStLeftCornerIdx;

    int64_t gradInputTotalCount = dGradInputActual_ * hGradInputActual_ * wGradInputActual_ * highAxisActual_;
    gradInputLoopCount = gradInputTotalCount / GRADINPUT_ONE_VL;
    gradInputLoopTail = gradInputTotalCount - gradInputLoopCount * GRADINPUT_ONE_VL;
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::CopyIn()
{
    LocalTensor<T> gradInputLocal = gradInputQue_.AllocTensor<T>();
    int64_t gradInputGmOffset = highAxisGradInputOffset_ + dAxisGradInputOffset_ + hAxisGradInputOffset_ +
                                wAxisGradInputOffset_;
    NdDmaLoopInfo<4> loopInfo;
    loopInfo.loopSize[0] = wGradInputActual_;
    loopInfo.loopSize[1] = hGradInputActual_;
    loopInfo.loopSize[2] = dGradInputActual_;
    loopInfo.loopSize[3] = highAxisActual_;

    loopInfo.loopSrcStride[0] = 1;
    loopInfo.loopSrcStride[1] = wGradInput_;
    loopInfo.loopSrcStride[2] = wGradInput_ * hGradInput_;
    loopInfo.loopSrcStride[3] = wGradInput_ * hGradInput_ * dGradInput_;

    loopInfo.loopDstStride[0] = 1;
    loopInfo.loopDstStride[1] = wGradInputActual_;
    loopInfo.loopDstStride[2] = wGradInputActual_ * hGradInputActual_;
    loopInfo.loopDstStride[3] = wGradInputActual_ * hGradInputActual_ * dGradInputActual_;

    static constexpr NdDmaConfig config = {false};
    NdDmaParams<T, 4> paramsMain = {loopInfo};
    DataCopy<T, 4, config>(gradInputLocal, gradInputGm_[gradInputGmOffset], paramsMain);
    gradInputQue_.EnQue(gradInputLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::CopyOut()
{
    LocalTensor<T> yLocal = outputQue_.DeQue<T>();
    int64_t outputPlaneSize = hOutput_ * wOutput_;
    int64_t outputPlaneDHW = dOutput_ * outputPlaneSize;
    int64_t ncBase = highAxisIndex_ * highAxisInner_ * outputPlaneDHW;
    int64_t dBase = dAxisIndex_ * dOutputInner_ * outputPlaneSize;
    int64_t hBase = hAxisIndex_ * hOutputInner_ * wOutput_;
    int64_t wBase = wAxisIndex_ * wOutputInner_;
    int64_t outputGmOffset = ncBase + dBase + hBase + wBase;

    LoopModeParams loopModeParamsT;
    loopModeParamsT.loop1Size = dOutputActual_;
    loopModeParamsT.loop2Size = highAxisActual_;
    loopModeParamsT.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T);
    loopModeParamsT.loop2SrcStride = dOutputActual_ * hOutputActual_ * wOutputAligned_ * sizeof(T);
    loopModeParamsT.loop1DstStride = outputPlaneSize * sizeof(T);
    loopModeParamsT.loop2DstStride = outputPlaneDHW * sizeof(T);

    SetLoopModePara(loopModeParamsT, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT = {static_cast<uint16_t>(hOutputActual_),
                                       static_cast<uint32_t>(wOutputActual_ * sizeof(T)), static_cast<uint32_t>(0),
                                       static_cast<uint32_t>((wOutput_ - wOutputActual_) * sizeof(T)),
                                       static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::ProcessPerLoop()
{
    CopyIn();
    Compute();
    CopyOut();
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::CalGradInputGMDHW(
    AscendC::Reg::RegTensor<INDEX>& gradInputUbIdx, AscendC::Reg::RegTensor<INDEX>& gradInputHigh,
    AscendC::Reg::RegTensor<INDEX>& gradInputD, AscendC::Reg::RegTensor<INDEX>& gradInputH,
    AscendC::Reg::RegTensor<INDEX>& gradInputW, AscendC::Reg::RegTensor<INDEX>& gradInputDHWValueConstReg,
    AscendC::Reg::RegTensor<INDEX>& gradInputHWValueConstReg, AscendC::Reg::RegTensor<INDEX>& gradInputWValueConstReg,
    AscendC::Reg::MaskReg& allMaskIndex)
{
    AscendC::Reg::RegTensor<INDEX> gradInputUbCalTemp;

    AscendC::Reg::Div(gradInputHigh, gradInputUbIdx, gradInputDHWValueConstReg, allMaskIndex);
    AscendC::Reg::Mul(gradInputUbCalTemp, gradInputHigh, gradInputDHWValueConstReg, allMaskIndex);
    AscendC::Reg::Sub(gradInputUbIdx, gradInputUbIdx, gradInputUbCalTemp, allMaskIndex);
    AscendC::Reg::Div(gradInputD, gradInputUbIdx, gradInputHWValueConstReg, allMaskIndex);
    AscendC::Reg::Mul(gradInputUbCalTemp, gradInputD, gradInputHWValueConstReg, allMaskIndex);
    AscendC::Reg::Sub(gradInputUbIdx, gradInputUbIdx, gradInputUbCalTemp, allMaskIndex);
    AscendC::Reg::Div(gradInputH, gradInputUbIdx, gradInputWValueConstReg, allMaskIndex);
    AscendC::Reg::Mul(gradInputUbCalTemp, gradInputH, gradInputWValueConstReg, allMaskIndex);
    AscendC::Reg::Sub(gradInputW, gradInputUbIdx, gradInputUbCalTemp, allMaskIndex);

    AscendC::Reg::Adds(gradInputD, gradInputD, INDEX(dStLeftCornerIdx), allMaskIndex);
    AscendC::Reg::Adds(gradInputH, gradInputH, INDEX(hStLeftCornerIdx), allMaskIndex);
    AscendC::Reg::Adds(gradInputW, gradInputW, INDEX(wStLeftCornerIdx), allMaskIndex);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::CalAndCopyGradInputStAndEdSingle(
    AscendC::Reg::RegTensor<INDEX>& gradKernelRegConstBig, AscendC::Reg::RegTensor<INDEX>& gradKernelRegConstSmall,
    AscendC::Reg::RegTensor<INDEX>& gradKernelKernelStIdx, AscendC::Reg::RegTensor<INDEX>& gradKernelKernelEdIdx,
    AscendC::Reg::RegTensor<INDEX>& gradInputIdxValue, AscendC::Reg::RegTensor<INDEX>& gradKernelKernelMulTemp,
    AscendC::Reg::RegTensor<INDEX>& gradKernelSize, __ubuf__ INDEX* gmStAddr, __ubuf__ INDEX* gmEdAddr,
    INDEX idxOutput_, INDEX idxGradInput_, INDEX idxAxisIndex_, INDEX idxInner_, INDEX idxOutputActual_,
    AscendC::Reg::MaskReg& allMaskIndex)
{
    AscendC::Reg::Duplicate(gradKernelRegConstBig, idxOutput_);
    AscendC::Reg::Duplicate(gradKernelRegConstSmall, idxGradInput_);

    AscendC::Reg::Mul(gradKernelKernelStIdx, gradInputIdxValue, gradKernelRegConstBig, allMaskIndex);
    AscendC::Reg::Div(gradKernelKernelStIdx, gradKernelKernelStIdx, gradKernelRegConstSmall, allMaskIndex);
    AscendC::Reg::Adds(gradKernelKernelEdIdx, gradInputIdxValue, INDEX(1), allMaskIndex);
    AscendC::Reg::Mul(gradKernelKernelEdIdx, gradKernelKernelEdIdx, gradKernelRegConstBig, allMaskIndex);
    AscendC::Reg::Add(gradKernelKernelEdIdx, gradKernelKernelEdIdx, gradKernelRegConstSmall, allMaskIndex);
    AscendC::Reg::Adds(gradKernelKernelEdIdx, gradKernelKernelEdIdx, INDEX(-1), allMaskIndex);
    AscendC::Reg::Div(gradKernelKernelEdIdx, gradKernelKernelEdIdx, gradKernelRegConstSmall, allMaskIndex);

    AscendC::Reg::Sub(gradKernelKernelMulTemp, gradKernelKernelEdIdx, gradKernelKernelStIdx, allMaskIndex);
    AscendC::Reg::Mul(gradKernelSize, gradKernelSize, gradKernelKernelMulTemp, allMaskIndex);

    AscendC::Reg::Maxs(gradKernelKernelStIdx, gradKernelKernelStIdx, idxAxisIndex_ * idxInner_, allMaskIndex);
    AscendC::Reg::Adds(gradKernelKernelStIdx, gradKernelKernelStIdx, -idxAxisIndex_ * idxInner_, allMaskIndex);
    AscendC::Reg::Mins(gradKernelKernelEdIdx, gradKernelKernelEdIdx, idxAxisIndex_ * idxInner_ + idxOutputActual_,
                       allMaskIndex);
    AscendC::Reg::Adds(gradKernelKernelEdIdx, gradKernelKernelEdIdx, -idxAxisIndex_ * idxInner_, allMaskIndex);
    AscendC::Reg::StoreAlign(gmStAddr, gradKernelKernelStIdx, allMaskIndex);
    AscendC::Reg::StoreAlign(gmEdAddr, gradKernelKernelEdIdx, allMaskIndex);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::CalAndCopyGradInputInfo(
    uint32_t gradInputUbIdxValue, __ubuf__ INDEX* gmStDAddr, __ubuf__ INDEX* gmEdDAddr, __ubuf__ INDEX* gmStHAddr,
    __ubuf__ INDEX* gmEdHAddr, __ubuf__ INDEX* gmStWAddr, __ubuf__ INDEX* gmEdWAddr, __ubuf__ INDEX* gmHighIdxAddr,
    __ubuf__ INDEX* gmKernelSizeAddr)
{
    INDEX dOutput = dOutput_;
    INDEX dGradInput = dGradInput_;
    INDEX dAxisIndex = dAxisIndex_;
    INDEX dOutputInner = dOutputInner_;
    INDEX dOutputActual = dOutputActual_;

    INDEX hOutput = hOutput_;
    INDEX hGradInput = hGradInput_;
    INDEX hAxisIndex = hAxisIndex_;
    INDEX hOutputInner = hOutputInner_;
    INDEX hOutputActual = hOutputActual_;

    INDEX wOutput = wOutput_;
    INDEX wGradInput = wGradInput_;
    INDEX wAxisIndex = wAxisIndex_;
    INDEX wOutputInner = wOutputInner_;
    INDEX wOutputActual = wOutputActual_;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::MaskReg allMaskIndex = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<INDEX> gradInputUbIdx;
        AscendC::Reg::Arange(gradInputUbIdx, gradInputUbIdxValue);
        AscendC::Reg::RegTensor<INDEX> gradInputHigh;
        AscendC::Reg::RegTensor<INDEX> gradInputD;
        AscendC::Reg::RegTensor<INDEX> gradInputH;
        AscendC::Reg::RegTensor<INDEX> gradInputW;

        AscendC::Reg::RegTensor<INDEX> gradInputDHWValueConstReg;
        AscendC::Reg::Duplicate(gradInputDHWValueConstReg,
                                INDEX(dGradInputActual_ * hGradInputActual_ * wGradInputActual_));
        AscendC::Reg::RegTensor<INDEX> gradInputHWValueConstReg;
        AscendC::Reg::Duplicate(gradInputHWValueConstReg, INDEX(hGradInputActual_ * wGradInputActual_));
        AscendC::Reg::RegTensor<INDEX> gradInputWValueConstReg;
        AscendC::Reg::Duplicate(gradInputWValueConstReg, INDEX(wGradInputActual_));
        CalGradInputGMDHW(gradInputUbIdx, gradInputHigh, gradInputD, gradInputH, gradInputW, gradInputDHWValueConstReg,
                          gradInputHWValueConstReg, gradInputWValueConstReg, allMaskIndex);
        AscendC::Reg::StoreAlign(gmHighIdxAddr, gradInputHigh, allMaskIndex);

        AscendC::Reg::StoreAlign(gmStDAddr, gradInputD, allMaskIndex);
        AscendC::Reg::StoreAlign(gmStHAddr, gradInputH, allMaskIndex);
        AscendC::Reg::StoreAlign(gmStWAddr, gradInputW, allMaskIndex);
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg allMaskIndex = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::RegTensor<INDEX> gradInputD;
        AscendC::Reg::LoadAlign(gradInputD, gmStDAddr);

        AscendC::Reg::RegTensor<INDEX> gradKernelSize;
        AscendC::Reg::Duplicate(gradKernelSize, INDEX(1));
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelStIdx;
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelEdIdx;
        AscendC::Reg::RegTensor<INDEX> gradKernelRegConstBig;
        AscendC::Reg::RegTensor<INDEX> gradKernelRegConstSmall;
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelMulTemp;

        CalAndCopyGradInputStAndEdSingle(gradKernelRegConstBig, gradKernelRegConstSmall, gradKernelKernelStIdx,
                                         gradKernelKernelEdIdx, gradInputD, gradKernelKernelMulTemp, gradKernelSize,
                                         gmStDAddr, gmEdDAddr, dOutput, dGradInput, dAxisIndex, dOutputInner,
                                         dOutputActual, allMaskIndex);
        AscendC::Reg::StoreAlign(gmKernelSizeAddr, gradKernelSize, allMaskIndex);
    }
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg allMaskIndex = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::RegTensor<INDEX> gradInputH;
        AscendC::Reg::LoadAlign(gradInputH, gmStHAddr);

        AscendC::Reg::RegTensor<INDEX> gradKernelSize;
        AscendC::Reg::LoadAlign(gradKernelSize, gmKernelSizeAddr);
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelStIdx;
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelEdIdx;
        AscendC::Reg::RegTensor<INDEX> gradKernelRegConstBig;
        AscendC::Reg::RegTensor<INDEX> gradKernelRegConstSmall;
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelMulTemp;

        CalAndCopyGradInputStAndEdSingle(gradKernelRegConstBig, gradKernelRegConstSmall, gradKernelKernelStIdx,
                                         gradKernelKernelEdIdx, gradInputH, gradKernelKernelMulTemp, gradKernelSize,
                                         gmStHAddr, gmEdHAddr, hOutput, hGradInput, hAxisIndex, hOutputInner,
                                         hOutputActual, allMaskIndex);
        AscendC::Reg::StoreAlign(gmKernelSizeAddr, gradKernelSize, allMaskIndex);
    }
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg allMaskIndex = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::RegTensor<INDEX> gradInputW;
        AscendC::Reg::LoadAlign(gradInputW, gmStWAddr);

        AscendC::Reg::RegTensor<INDEX> gradKernelSize;
        AscendC::Reg::LoadAlign(gradKernelSize, gmKernelSizeAddr);
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelStIdx;
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelEdIdx;
        AscendC::Reg::RegTensor<INDEX> gradKernelRegConstBig;
        AscendC::Reg::RegTensor<INDEX> gradKernelRegConstSmall;
        AscendC::Reg::RegTensor<INDEX> gradKernelKernelMulTemp;

        CalAndCopyGradInputStAndEdSingle(gradKernelRegConstBig, gradKernelRegConstSmall, gradKernelKernelStIdx,
                                         gradKernelKernelEdIdx, gradInputW, gradKernelKernelMulTemp, gradKernelSize,
                                         gmStWAddr, gmEdWAddr, wOutput, wGradInput, wAxisIndex, wOutputInner,
                                         wOutputActual, allMaskIndex);
        AscendC::Reg::StoreAlign(gmKernelSizeAddr, gradKernelSize, allMaskIndex);
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::GatherCopyGradUb2Reg(
    AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx, AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
    __ubuf__ COMPUTE_TYPE* yAddr, uint32_t& maskCount)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::Reg::MaskReg pregU32 = AscendC::Reg::UpdateMask<uint32_t>(maskCountTemp);
    if constexpr (std::is_same<INDEX, int64_t>::value) {
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<int32_t> gradOutputUBIdxI32;
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(gradOutputUBIdxI32, gradOutputUBIdx, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdxI32,
                           (AscendC::Reg::RegTensor<int64_t>&)gradOutputUBIdxI32);
        AscendC::Reg::Gather(gradOutputUbValue, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdxI32, pregU32);
    } else {
        AscendC::Reg::Gather(gradOutputUbValue, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdx, pregU32);
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::ScatterCopyGradReg2Ub(
    AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx, AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
    __ubuf__ COMPUTE_TYPE* yAddr, uint32_t& maskCount)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::Reg::MaskReg pregU32 = AscendC::Reg::UpdateMask<uint32_t>(maskCountTemp);
    if constexpr (std::is_same<INDEX, int64_t>::value) {
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<int32_t> gradOutputUBIdxI32;
        AscendC::Reg::Cast<int32_t, int64_t, castTraitI64I32>(gradOutputUBIdxI32, gradOutputUBIdx, allMask);
        AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdxI32,
                           (AscendC::Reg::RegTensor<int64_t>&)gradOutputUBIdxI32);
        AscendC::Reg::Scatter(yAddr, gradOutputUbValue, (AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdxI32,
                              pregU32);
    } else {
        AscendC::Reg::Scatter(yAddr, gradOutputUbValue, (AscendC::Reg::RegTensor<uint32_t>&)gradOutputUBIdx, pregU32);
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::DoGradRegAdds(
    AscendC::Reg::RegTensor<COMPUTE_TYPE>& gradOutputUbValue, COMPUTE_TYPE& gradInputValue,
    __ubuf__ COMPUTE_TYPE* yAddr, uint32_t& maskCount)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::Reg::MaskReg pregU32 = AscendC::Reg::UpdateMask<uint32_t>(maskCountTemp);
    AscendC::Reg::Adds(gradOutputUbValue, gradOutputUbValue, COMPUTE_TYPE(gradInputValue), pregU32);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::ComputeGradIndexHW(
    AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx, INDEX gradKernelW, INDEX gradStHIdxOffset, INDEX gradStWIdx,
    INDEX highDHOffset, AscendC::Reg::MaskReg& pregU32MaskAll)
{
    AscendC::Reg::Arange(gradOutputUBIdx, INDEX(0));

    AscendC::Reg::RegTensor<INDEX> regConst;
    AscendC::Reg::Duplicate(regConst, INDEX(gradKernelW));
    AscendC::Reg::RegTensor<INDEX> gradOutputUBDivKW;

    AscendC::Reg::Div(gradOutputUBDivKW, gradOutputUBIdx, regConst, pregU32MaskAll);
    AscendC::Reg::RegTensor<INDEX> gradOutputUBCellAlignKW;
    AscendC::Reg::Mul(gradOutputUBCellAlignKW, gradOutputUBDivKW, regConst, pregU32MaskAll);
    AscendC::Reg::RegTensor<INDEX> gradOutputUBLineInKW;
    AscendC::Reg::Sub(gradOutputUBLineInKW, gradOutputUBIdx, gradOutputUBCellAlignKW, pregU32MaskAll);

    AscendC::Reg::Duplicate(regConst, INDEX(gradStHIdxOffset));
    AscendC::Reg::Add(gradOutputUBDivKW, gradOutputUBDivKW, regConst, pregU32MaskAll);
    AscendC::Reg::Duplicate(regConst, INDEX(wOutputAligned_));
    AscendC::Reg::Mul(gradOutputUBDivKW, gradOutputUBDivKW, regConst, pregU32MaskAll);

    AscendC::Reg::Add(gradOutputUBLineInKW, gradOutputUBLineInKW, gradOutputUBDivKW, pregU32MaskAll);
    AscendC::Reg::Duplicate(regConst, INDEX(gradStWIdx));
    AscendC::Reg::Add(gradOutputUBIdx, gradOutputUBLineInKW, regConst, pregU32MaskAll);
    AscendC::Reg::Duplicate(regConst, INDEX(highDHOffset));
    AscendC::Reg::Add(gradOutputUBIdx, gradOutputUBIdx, regConst, pregU32MaskAll);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::ComputeGradIndexDHW(
    AscendC::Reg::RegTensor<INDEX>& gradOutputUBIdx, INDEX gradKernelHW, INDEX gradKernelW, INDEX gradStDIdxOffset,
    INDEX outputHWAlign, INDEX gradStHIdx, INDEX gradStWIdx, INDEX highOffset, AscendC::Reg::MaskReg& pregU32MaskAll)
{
    AscendC::Reg::Arange(gradOutputUBIdx, INDEX(0));

    AscendC::Reg::RegTensor<INDEX> regConst;
    AscendC::Reg::Duplicate(regConst, INDEX(gradKernelHW));
    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdxDivKHW;
    AscendC::Reg::Div(gradOutputUBIdxDivKHW, gradOutputUBIdx, regConst, pregU32MaskAll);
    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdxCeilAlignKHW;
    AscendC::Reg::Mul(gradOutputUBIdxCeilAlignKHW, gradOutputUBIdxDivKHW, regConst, pregU32MaskAll);

    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdxLineInHW;
    AscendC::Reg::Sub(gradOutputUBIdxLineInHW, gradOutputUBIdx, gradOutputUBIdxCeilAlignKHW, pregU32MaskAll);
    AscendC::Reg::Duplicate(regConst, INDEX(gradKernelW));
    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdxDivKW;
    AscendC::Reg::Div(gradOutputUBIdxDivKW, gradOutputUBIdxLineInHW, regConst, pregU32MaskAll);
    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdxCeilAlignKW;
    AscendC::Reg::Mul(gradOutputUBIdxCeilAlignKW, gradOutputUBIdxDivKW, regConst, pregU32MaskAll);

    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdxLineInW;
    AscendC::Reg::Sub(gradOutputUBIdxLineInW, gradOutputUBIdxLineInHW, gradOutputUBIdxCeilAlignKW, pregU32MaskAll);

    AscendC::Reg::Duplicate(regConst, INDEX(gradStDIdxOffset));
    AscendC::Reg::Add(gradOutputUBIdxDivKHW, gradOutputUBIdxDivKHW, regConst, pregU32MaskAll);
    AscendC::Reg::Duplicate(regConst, INDEX(outputHWAlign));
    AscendC::Reg::Mul(gradOutputUBIdxDivKHW, gradOutputUBIdxDivKHW, regConst, pregU32MaskAll);
    AscendC::Reg::Add(gradOutputUBIdxLineInW, gradOutputUBIdxLineInW, gradOutputUBIdxDivKHW, pregU32MaskAll);

    AscendC::Reg::Duplicate(regConst, INDEX(gradStHIdx));
    AscendC::Reg::Add(gradOutputUBIdxDivKW, gradOutputUBIdxDivKW, regConst, pregU32MaskAll);
    AscendC::Reg::Duplicate(regConst, INDEX(wOutputAligned_));
    AscendC::Reg::Mul(gradOutputUBIdxDivKW, gradOutputUBIdxDivKW, regConst, pregU32MaskAll);
    AscendC::Reg::Add(gradOutputUBIdxLineInW, gradOutputUBIdxLineInW, gradOutputUBIdxDivKW, pregU32MaskAll);

    AscendC::Reg::Duplicate(regConst, INDEX(gradStWIdx));
    AscendC::Reg::Add(gradOutputUBIdx, gradOutputUBIdxLineInW, regConst, pregU32MaskAll);

    AscendC::Reg::Duplicate(regConst, INDEX(highOffset));
    AscendC::Reg::Add(gradOutputUBIdx, gradOutputUBIdx, regConst, pregU32MaskAll);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::DoGradInputAccUb(
    uint32_t gradInputLoopCount, uint32_t gradInputUbIdxValue, __ubuf__ T* gradAddr, __ubuf__ COMPUTE_TYPE* yAddr,
    __ubuf__ INDEX* gmStDAddr, __ubuf__ INDEX* gmEdDAddr, __ubuf__ INDEX* gmStHAddr, __ubuf__ INDEX* gmEdHAddr,
    __ubuf__ INDEX* gmStWAddr, __ubuf__ INDEX* gmEdWAddr, __ubuf__ INDEX* gmHighIdxAddr,
    __ubuf__ INDEX* gmKernelSizeAddr, __ubuf__ COMPUTE_TYPE* gmGradInputF32Addr)
{
    for (uint32_t loopGradInputIdx = 0; loopGradInputIdx < gradInputLoopCount; ++loopGradInputIdx) {
        uint32_t loopGradInputUbRealIdx = loopGradInputIdx + gradInputUbIdxValue;

        COMPUTE_TYPE gradInputValue = static_cast<COMPUTE_TYPE>(gmGradInputF32Addr[loopGradInputIdx]);
        INDEX gradHightIdx = gmHighIdxAddr[loopGradInputIdx];
        INDEX gradStDIdx = gmStDAddr[loopGradInputIdx];
        INDEX gradEdDIdx = gmEdDAddr[loopGradInputIdx];
        INDEX gradStHIdx = gmStHAddr[loopGradInputIdx];
        INDEX gradEdHIdx = gmEdHAddr[loopGradInputIdx];
        INDEX gradStWIdx = gmStWAddr[loopGradInputIdx];
        INDEX gradEdWIdx = gmEdWAddr[loopGradInputIdx];
        INDEX gradKernelD = gradEdDIdx - gradStDIdx;
        INDEX gradKernelH = gradEdHIdx - gradStHIdx;
        INDEX gradKernelW = gradEdWIdx - gradStWIdx;
        if (gradKernelW * DOUBLE * sizeof(INDEX) > V_REG_SIZE) {
            uint32_t vlFullSize = GRADINPUT_ONE_VL;
            uint16_t wloopCount = gradKernelW / vlFullSize;
            uint32_t wloopCountTail = gradKernelW % vlFullSize;

            for (uint32_t dloopIdx = gradStDIdx; dloopIdx < gradEdDIdx; ++dloopIdx) {
                for (uint32_t hloopIdx = gradStHIdx; hloopIdx < gradEdHIdx; ++hloopIdx) {
                    INDEX highDHWOffset = gradHightIdx * (dOutputActual_ * hOutputActual_ * wOutputAligned_) +
                                          dloopIdx * (hOutputActual_ * wOutputAligned_) + hloopIdx * (wOutputAligned_) +
                                          gradStWIdx;
                    __VEC_SCOPE__
                    {
                        AscendC::Reg::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                        AscendC::Reg::MaskReg
                            pregU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
                        AscendC::Reg::UnalignRegForLoad u0;
                        AscendC::Reg::UnalignRegForStore u1;
                        auto yAddrSrcUnalign = yAddr + highDHWOffset;
                        auto yAddrDstUnalign = yAddr + highDHWOffset;
                        AscendC::Reg::LoadUnAlignPre(u0, yAddrSrcUnalign);
                        for (uint16_t wloopIdx = 0; wloopIdx < wloopCount; ++wloopIdx) {
                            AscendC::Reg::LoadUnAlign(gradOutputUbValue, u0, yAddrSrcUnalign, vlFullSize);
                            AscendC::Reg::Adds(gradOutputUbValue, gradOutputUbValue,
                                               static_cast<COMPUTE_TYPE>(gradInputValue), pregU32);
                            AscendC::Reg::StoreUnAlign(yAddrDstUnalign, gradOutputUbValue, u1, vlFullSize);
                        }
                        AscendC::Reg::StoreUnAlignPost(yAddrDstUnalign, u1, 0);

                        highDHWOffset += wloopCount * vlFullSize;
                        auto yAddrUnalign = yAddr + highDHWOffset;
                        AscendC::Reg::LoadUnAlignPre(u0, yAddrUnalign);
                        AscendC::Reg::LoadUnAlign(gradOutputUbValue, u0, yAddrUnalign);
                        DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, wloopCountTail);
                        AscendC::Reg::StoreUnAlign(yAddrUnalign, gradOutputUbValue, u1, wloopCountTail);
                        AscendC::Reg::StoreUnAlignPost(yAddrUnalign, u1, wloopCountTail);
                    }
                }
            }
        } else if (gradKernelH * gradKernelW * DOUBLE * sizeof(INDEX) > V_REG_SIZE) {
            uint32_t vlFullSize = GRADINPUT_ONE_VL;
            uint32_t maxWParaNum = vlFullSize / gradKernelW;
            uint16_t hwloopCount = gradKernelH / maxWParaNum;
            uint32_t hwloopCountTail = gradKernelH % maxWParaNum;
            uint32_t maskCountFull = gradKernelW * maxWParaNum;
            uint32_t maskCountTail = gradKernelW * hwloopCountTail;

            for (uint32_t dloopIdx = gradStDIdx; dloopIdx < gradEdDIdx; ++dloopIdx) {
                INDEX highDHOffset = gradHightIdx * (dOutputActual_ * hOutputActual_ * wOutputAligned_) +
                                     dloopIdx * (hOutputActual_ * wOutputAligned_);
                if constexpr (std::is_same<INDEX, int32_t>::value) {
                    __VEC_SCOPE__
                    {
                        AscendC::Reg::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                        AscendC::Reg::MaskReg
                            pregU32MaskAll = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                        AscendC::Reg::RegTensor<INDEX> gradOutputUBIdx;
                        for (uint16_t hwloopIdx = 0; hwloopIdx < hwloopCount; ++hwloopIdx) {
                            INDEX gradStHIdxOffset = gradStHIdx + hwloopIdx * maxWParaNum;
                            ComputeGradIndexHW(gradOutputUBIdx, gradKernelW, gradStHIdxOffset, gradStWIdx, highDHOffset,
                                               pregU32MaskAll);
                            GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                            DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountFull);
                            ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                        }

                        INDEX gradStHIdxOffset = gradStHIdx + hwloopCount * maxWParaNum;
                        ComputeGradIndexHW(gradOutputUBIdx, gradKernelW, gradStHIdxOffset, gradStWIdx, highDHOffset,
                                           pregU32MaskAll);
                        GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                        DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountTail);
                        ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                    }
                } else {
                    __VEC_SCOPE__
                    {
                        AscendC::Reg::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                        AscendC::Reg::MaskReg
                            pregU32MaskAll = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                        AscendC::Reg::RegTensor<INDEX> gradOutputUBIdx;
                        for (uint16_t hwloopIdx = 0; hwloopIdx < hwloopCount; ++hwloopIdx) {
                            INDEX gradStHIdxOffset = gradStHIdx + hwloopIdx * maxWParaNum;
                            ComputeGradIndexHW(gradOutputUBIdx, gradKernelW, gradStHIdxOffset, gradStWIdx, highDHOffset,
                                               pregU32MaskAll);
                            GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                            DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountFull);
                            ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                        }
                    }

                    __VEC_SCOPE__
                    {
                        AscendC::Reg::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                        AscendC::Reg::MaskReg
                            pregU32MaskAll = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                        AscendC::Reg::RegTensor<INDEX> gradOutputUBIdx;
                        INDEX gradStHIdxOffset = gradStHIdx + hwloopCount * maxWParaNum;
                        ComputeGradIndexHW(gradOutputUBIdx, gradKernelW, gradStHIdxOffset, gradStWIdx, highDHOffset,
                                           pregU32MaskAll);
                        GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                        DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountTail);
                        ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                    }
                }
            }
        } else if (gradKernelD * gradKernelH * gradKernelW * DOUBLE * sizeof(INDEX) > V_REG_SIZE) {
            uint32_t vlFullSize = GRADINPUT_ONE_VL;
            uint32_t gradKernelHW = gradKernelH * gradKernelW;
            uint32_t maxHWParaNum = vlFullSize / gradKernelHW;
            uint16_t dhwloopCount = gradKernelD / maxHWParaNum;
            uint32_t dhwloopCountTail = gradKernelD % maxHWParaNum;
            uint32_t maskCountFull = gradKernelHW * maxHWParaNum;
            uint32_t maskCountTail = gradKernelHW * dhwloopCountTail;
            uint32_t hOutputActualU32 = hOutputActual_;
            uint32_t wOutputAlignedU32 = wOutputAligned_;
            uint32_t outputHWAlign = hOutputActualU32 * wOutputAlignedU32;

            INDEX highOffset = gradHightIdx * (dOutputActual_ * hOutputActual_ * wOutputAligned_);

            if constexpr (std::is_same<INDEX, int32_t>::value) {
                __VEC_SCOPE__
                {
                    AscendC::Reg::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                    AscendC::Reg::MaskReg
                        pregU32MaskAll = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdx;
                    for (uint16_t dhwloopIdx = 0; dhwloopIdx < dhwloopCount; ++dhwloopIdx) {
                        INDEX gradStDIdxOffset = gradStDIdx + dhwloopIdx * maxHWParaNum;
                        ComputeGradIndexDHW(gradOutputUBIdx, gradKernelHW, gradKernelW, gradStDIdxOffset, outputHWAlign,
                                            gradStHIdx, gradStWIdx, highOffset, pregU32MaskAll);
                        GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                        DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountFull);
                        ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                    }

                    INDEX gradStDIdxOffset = gradStDIdx + dhwloopCount * maxHWParaNum;
                    ComputeGradIndexDHW(gradOutputUBIdx, gradKernelHW, gradKernelW, gradStDIdxOffset, outputHWAlign,
                                        gradStHIdx, gradStWIdx, highOffset, pregU32MaskAll);
                    GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                    DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountTail);
                    ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                }
            } else {
                __VEC_SCOPE__
                {
                    AscendC::Reg::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                    AscendC::Reg::MaskReg
                        pregU32MaskAll = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
                    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdx;
                    for (uint16_t dhwloopIdx = 0; dhwloopIdx < dhwloopCount; ++dhwloopIdx) {
                        INDEX gradStDIdxOffset = gradStDIdx + dhwloopIdx * maxHWParaNum;
                        ComputeGradIndexDHW(gradOutputUBIdx, gradKernelHW, gradKernelW, gradStDIdxOffset, outputHWAlign,
                                            gradStHIdx, gradStWIdx, highOffset, pregU32MaskAll);
                        GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                        DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountFull);
                        ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                    }
                }

                __VEC_SCOPE__
                {
                    AscendC::Reg::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                    AscendC::Reg::MaskReg
                        pregU32MaskAll = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                    AscendC::Reg::RegTensor<INDEX> gradOutputUBIdx;
                    INDEX gradStDIdxOffset = gradStDIdx + dhwloopCount * maxHWParaNum;
                    ComputeGradIndexDHW(gradOutputUBIdx, gradKernelHW, gradKernelW, gradStDIdxOffset, outputHWAlign,
                                        gradStHIdx, gradStWIdx, highOffset, pregU32MaskAll);
                    GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                    DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountTail);
                    ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                }
            }
        } else {
            INDEX highOffset = gradHightIdx * (dOutputActual_ * hOutputActual_ * wOutputAligned_);
            uint32_t maskCountFull = gradKernelD * gradKernelH * gradKernelW;
            uint32_t gradKernelHW = gradKernelH * gradKernelW;
            uint32_t hOutputActualU32 = hOutputActual_;
            uint32_t wOutputAlignedU32 = wOutputAligned_;
            uint32_t outputHWAlign = hOutputActualU32 * wOutputAlignedU32;

            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                AscendC::Reg::MaskReg
                    pregU32MaskAll = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

                AscendC::Reg::RegTensor<INDEX> gradOutputUBIdx;
                ComputeGradIndexDHW(gradOutputUBIdx, gradKernelHW, gradKernelW, gradStDIdx, outputHWAlign, gradStHIdx,
                                    gradStWIdx, highOffset, pregU32MaskAll);
                GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountFull);
                ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
            }
        }
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::DoAllGradInputProcess(
    uint32_t gradInputLoopCount, uint32_t gradInputUbIdxValue, __ubuf__ T* gradAddr, __ubuf__ COMPUTE_TYPE* yAddr,
    __ubuf__ INDEX* gmStDAddr, __ubuf__ INDEX* gmEdDAddr, __ubuf__ INDEX* gmStHAddr, __ubuf__ INDEX* gmEdHAddr,
    __ubuf__ INDEX* gmStWAddr, __ubuf__ INDEX* gmEdWAddr, __ubuf__ INDEX* gmHighIdxAddr,
    __ubuf__ INDEX* gmKernelSizeAddr, __ubuf__ COMPUTE_TYPE* gmGradInputF32Addr)
{
    CalAndCopyGradInputInfo(gradInputUbIdxValue, gmStDAddr, gmEdDAddr, gmStHAddr, gmEdHAddr, gmStWAddr, gmEdWAddr,
                            gmHighIdxAddr, gmKernelSizeAddr);

    if constexpr (std::is_same<INDEX, int64_t>::value) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<COMPUTE_TYPE> gradInputUbValue;
            AscendC::Reg::MaskReg pregT = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
            ops::LoadOneTensorForDtypeT<T>(gradAddr, gradInputUbValue, pregT, gradInputUbIdxValue);

            AscendC::Reg::RegTensor<INDEX> kernelSize;
            AscendC::Reg::RegTensor<COMPUTE_TYPE> kernelSizeF32;
            AscendC::Reg::LoadAlign(kernelSize, gmKernelSizeAddr);
            AscendC::Reg::MaskReg allMaskU = AscendC::Reg::CreateMask<INDEX, AscendC::Reg::MaskPattern::ALL>();

            AscendC::Reg::Cast<COMPUTE_TYPE, INDEX, castTraitI64TF32>(kernelSizeF32, kernelSize, allMaskU);
            AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint32_t>&)kernelSizeF32,
                               (AscendC::Reg::RegTensor<uint64_t>&)kernelSizeF32);
            AscendC::Reg::Div(gradInputUbValue, gradInputUbValue, kernelSizeF32, allMaskU32);
            AscendC::Reg::StoreAlign(gmGradInputF32Addr, gradInputUbValue, allMaskU32);
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<COMPUTE_TYPE> gradInputUbValue;
            AscendC::Reg::MaskReg pregT = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
            ops::LoadOneTensorForDtypeT<T>(gradAddr, gradInputUbValue, pregT, gradInputUbIdxValue);

            AscendC::Reg::RegTensor<INDEX> kernelSize;
            AscendC::Reg::RegTensor<COMPUTE_TYPE> kernelSizeF32;
            AscendC::Reg::LoadAlign(kernelSize, gmKernelSizeAddr);

            AscendC::Reg::Cast<COMPUTE_TYPE, INDEX, castTraitI32TF32>(kernelSizeF32, kernelSize, allMaskU32);
            AscendC::Reg::Div(gradInputUbValue, gradInputUbValue, kernelSizeF32, allMaskU32);
            AscendC::Reg::StoreAlign(gmGradInputF32Addr, gradInputUbValue, allMaskU32);
        }
    }
    PIPE_V_S();
    DoGradInputAccUb(gradInputLoopCount, gradInputUbIdxValue, gradAddr, yAddr, gmStDAddr, gmEdDAddr, gmStHAddr,
                     gmEdHAddr, gmStWAddr, gmEdWAddr, gmHighIdxAddr, gmKernelSizeAddr, gmGradInputF32Addr);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWBigKernel<T, INDEX>::Compute()
{
    LocalTensor<COMPUTE_TYPE> yLocal = outputQue_.AllocTensor<COMPUTE_TYPE>();
    uint32_t calCount = outputBufferSize_ / sizeof(COMPUTE_TYPE);
    Duplicate(yLocal, COMPUTE_TYPE(0), calCount);
    LocalTensor<T> gradLocal = gradInputQue_.DeQue<T>();

    LocalTensor<INDEX> gmhighIdxLocal = gmhighIdx.DeQue<INDEX>();
    LocalTensor<INDEX> gmKernelSizeLocal = gmKernelSize.DeQue<INDEX>();
    LocalTensor<INDEX> gmStDLocal = gmStD.DeQue<INDEX>();
    LocalTensor<INDEX> gmEdDLocal = gmEdD.DeQue<INDEX>();
    LocalTensor<INDEX> gmStHLocal = gmStH.DeQue<INDEX>();
    LocalTensor<INDEX> gmEdHLocal = gmEdH.DeQue<INDEX>();
    LocalTensor<INDEX> gmStWLocal = gmStW.DeQue<INDEX>();
    LocalTensor<INDEX> gmEdWLocal = gmEdW.DeQue<INDEX>();

    __ubuf__ COMPUTE_TYPE* yAddr = (__ubuf__ COMPUTE_TYPE*)yLocal.GetPhyAddr();
    __ubuf__ T* gradAddr = (__ubuf__ T*)gradLocal.GetPhyAddr();
    __ubuf__ INDEX* gmHighIdxAddr = (__ubuf__ INDEX*)gmhighIdxLocal.GetPhyAddr();
    __ubuf__ INDEX* gmKernelSizeAddr = (__ubuf__ INDEX*)gmKernelSizeLocal.GetPhyAddr();
    __ubuf__ INDEX* gmStDAddr = (__ubuf__ INDEX*)gmStDLocal.GetPhyAddr();
    __ubuf__ INDEX* gmEdDAddr = (__ubuf__ INDEX*)gmEdDLocal.GetPhyAddr();
    __ubuf__ INDEX* gmStHAddr = (__ubuf__ INDEX*)gmStHLocal.GetPhyAddr();
    __ubuf__ INDEX* gmEdHAddr = (__ubuf__ INDEX*)gmEdHLocal.GetPhyAddr();
    __ubuf__ INDEX* gmStWAddr = (__ubuf__ INDEX*)gmStWLocal.GetPhyAddr();
    __ubuf__ INDEX* gmEdWAddr = (__ubuf__ INDEX*)gmEdWLocal.GetPhyAddr();

    LocalTensor<COMPUTE_TYPE> gmGradInputF32Local;
    __ubuf__ COMPUTE_TYPE* gmGradInputF32Addr;
    gmGradInputF32Local = gmGradInputF32.DeQue<COMPUTE_TYPE>();
    gmGradInputF32Addr = (__ubuf__ COMPUTE_TYPE*)gmGradInputF32Local.GetPhyAddr();

    uint32_t gradInputUbIdx = 0;
    for (int32_t i = 0; i < gradInputLoopCount; ++i) {
        DoAllGradInputProcess(static_cast<uint32_t>(GRADINPUT_ONE_VL), gradInputUbIdx, gradAddr, yAddr, gmStDAddr,
                              gmEdDAddr, gmStHAddr, gmEdHAddr, gmStWAddr, gmEdWAddr, gmHighIdxAddr, gmKernelSizeAddr,
                              gmGradInputF32Addr);
        gradInputUbIdx += GRADINPUT_ONE_VL;
    }

    if (gradInputLoopTail > 0) {
        DoAllGradInputProcess(gradInputLoopTail, gradInputUbIdx, gradAddr, yAddr, gmStDAddr, gmEdDAddr, gmStHAddr,
                              gmEdHAddr, gmStWAddr, gmEdWAddr, gmHighIdxAddr, gmKernelSizeAddr, gmGradInputF32Addr);
    }

    if constexpr (std::negation<std::is_same<T, COMPUTE_TYPE>>::value) {
        Cast(yLocal.ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, calCount);
    }
    outputQue_.EnQue(yLocal);
    gradInputQue_.FreeTensor(gradLocal);
}
} // namespace AdaptiveAvgPool3dGradOp

#endif
