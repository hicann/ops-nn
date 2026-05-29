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
 * \file adaptive_avg_pool2d_grad_big_kernel.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL2D_GRAD_BIG_KERNEL
#define ADAPTIVE_AVG_POOL2D_GRAD_BIG_KERNEL

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "adaptive_avg_pool2d_grad_struct.h"

namespace AdaptiveAvgPool2dGradOp
{
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t NDDMA_DIMS = 3;

using COMPUTE_TYPE = float;

template<typename T>
__aicore__ inline T GetStartFromOutputInputSize(T idx, T sizeA, T sizeB)
{
    return (idx * sizeA) / sizeB;
}

template<typename T>
__aicore__ inline T GetEndFromOutputInputSize(T idx, T sizeA, T sizeB)
{
    return ((idx + 1) * sizeA + sizeB - 1) / sizeB;
}

__aicore__ inline void PIPE_S_V() {
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

__aicore__ inline void PIPE_V_S() {
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

constexpr AscendC::MicroAPI::CastTrait castTraitI32TF32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitI64TF32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitI64I32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

template <typename T, typename INDEX>
class AdaptiveAvgPool2dGradBigKernel
{
public:
    __aicore__ inline AdaptiveAvgPool2dGradBigKernel(TPipe& pipe, 
        const AdaptiveAvgPool2dGradBigKernelTiling& __restrict__ tilingData_): pipe_(pipe), tilingData_(tilingData_){};
    __aicore__ inline void Init(GM_ADDR gradInput, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyIn();
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut();
    __aicore__ inline void CalGradInputGMHW(AscendC::MicroAPI::RegTensor<INDEX>& gradInputUbIdx,
            AscendC::MicroAPI::RegTensor<INDEX>& gradInputHigh,
            AscendC::MicroAPI::RegTensor<INDEX>& gradInputH, AscendC::MicroAPI::RegTensor<INDEX>& gradInputW,
            AscendC::MicroAPI::RegTensor<INDEX>& gradInputHWValueConstReg,
            AscendC::MicroAPI::RegTensor<INDEX>& gradInputWValueConstReg, AscendC::MicroAPI::MaskReg& allMaskIndex);
    __aicore__ inline void CalAndCopyGradInputStAndEdSingle(AscendC::MicroAPI::RegTensor<INDEX>& gradKernelRegConstBig,
        AscendC::MicroAPI::RegTensor<INDEX>& gradKernelRegConstSmall, AscendC::MicroAPI::RegTensor<INDEX>& gradKernelKernelStIdx,
        AscendC::MicroAPI::RegTensor<INDEX>& gradKernelKernelEdIdx, AscendC::MicroAPI::RegTensor<INDEX>& gradInputIdxValue,
        AscendC::MicroAPI::RegTensor<INDEX>& gradKernelKernelMulTemp, AscendC::MicroAPI::RegTensor<INDEX>& gradKernelSize,
        __local_mem__ INDEX* gmStAddr, __local_mem__ INDEX* gmEdAddr, INDEX idxOutput,
        INDEX idxGradInput, INDEX idxAxisIndex, INDEX idxInner, INDEX idxOutputActual,
        AscendC::MicroAPI::MaskReg& allMaskIndex
    );
    __aicore__ inline void CalAndCopyGradInputInfo(uint32_t gradInputUbIdxValue,
        __local_mem__ INDEX* gmStHAddr,
        __local_mem__ INDEX* gmEdHAddr, __local_mem__ INDEX* gmStWAddr, __local_mem__ INDEX* gmEdWAddr,
        __local_mem__ INDEX* gmHighIdxAddr, __local_mem__ INDEX* gmKernelSizeAddr
    );
    __aicore__ inline void DoGradInputAccUb(
        uint32_t gradInputLoopCount, uint32_t gradInputUbIdxValue, __local_mem__ T* gradAddr, __local_mem__ COMPUTE_TYPE* yAddr,
        __local_mem__ INDEX* gmStHAddr, __local_mem__ INDEX* gmEdHAddr, __local_mem__ INDEX* gmStWAddr, __local_mem__ INDEX* gmEdWAddr,
        __local_mem__ INDEX* gmHighIdxAddr, __local_mem__ INDEX* gmKernelSizeAddr, __local_mem__ COMPUTE_TYPE* gmGradInputF32Addr
    );
    __aicore__ inline void DoAllGradInputProcess(
        uint32_t gradInputLoopCount, uint32_t gradInputUbIdxValue, __local_mem__ T* gradAddr, __local_mem__ COMPUTE_TYPE* yAddr,
        __local_mem__ INDEX* gmStHAddr, __local_mem__ INDEX* gmEdHAddr, __local_mem__ INDEX* gmStWAddr, __local_mem__ INDEX* gmEdWAddr,
        __local_mem__ INDEX* gmHighIdxAddr, __local_mem__ INDEX* gmKernelSizeAddr, __local_mem__ COMPUTE_TYPE* gmGradInputF32Addr
    );
    __aicore__ inline void GatherCopyGradUb2Reg(
        AscendC::MicroAPI::RegTensor<INDEX>& gradOutputUBIdx, AscendC::MicroAPI::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
        __local_mem__ COMPUTE_TYPE* yAddr, uint32_t& maskCount
    );
    __aicore__ inline void ScatterCopyGradReg2Ub(
        AscendC::MicroAPI::RegTensor<INDEX>& gradOutputUBIdx, AscendC::MicroAPI::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
        __local_mem__ COMPUTE_TYPE* yAddr, uint32_t& maskCount
    );
    __aicore__ inline void DoGradRegAdds(
        AscendC::MicroAPI::RegTensor<COMPUTE_TYPE>& gradOutputUbValue, COMPUTE_TYPE& gradInputValue,
        __local_mem__ COMPUTE_TYPE* yAddr, uint32_t& maskCount
    );
    __aicore__ inline void ComputeGradIndexHW(
        AscendC::MicroAPI::RegTensor<INDEX>& gradOutputUBIdx, INDEX gradKernelW, INDEX gradStHIdxOffset,
        INDEX gradStWIdx, INDEX highHOffset, AscendC::MicroAPI::MaskReg& pregU32MaskAll
    );
    __aicore__ inline void ComputeGradIndexEntireHW(
        AscendC::MicroAPI::RegTensor<INDEX>& gradOutputUBIdx, INDEX gradKernelW,
        INDEX outputHWAlign, INDEX gradStHIdx, INDEX gradStWIdx, INDEX highOffset, AscendC::MicroAPI::MaskReg& pregU32MaskAll
    );
    TPipe& pipe_;
    const AdaptiveAvgPool2dGradBigKernelTiling& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradInputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    
    TBuf<QuePosition::VECCALC> gmKernelSize_;
    TBuf<QuePosition::VECCALC> gmHighIdx_;
    TBuf<QuePosition::VECCALC> gmStH_;
    TBuf<QuePosition::VECCALC> gmEdH_;
    TBuf<QuePosition::VECCALC> gmStW_;
    TBuf<QuePosition::VECCALC> gmEdW_;
    TBuf<QuePosition::VECCALC> gmGradInputF32_;

    GlobalTensor<T> gradInputGm_;
    GlobalTensor<T> yGm_;

    uint32_t blockIdx_ = 0;
    int64_t highAxisActual_ = 1;
    int64_t hOutputActual_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t wOutputAligned_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t hGradInputActual_ = 0;
    int64_t wGradInputActual_ = 0;
    int64_t wGradInputAligned_ = 0;

    int64_t gradInputPlaneSize_ = 0;

    int64_t highAxisGradInputOffset_ = 0;
    int64_t hAxisGradInputOffset_ = 0;
    int64_t wAxisGradInputOffset_ = 0;


    int64_t hStLeftCornerIdx_ = 0;
    int64_t wStLeftCornerIdx_ = 0;
    int64_t hEndRightCornerIdx_ = 0;
    int64_t wEndRightCornerIdx_ = 0;

    int64_t gradInputLoopCount_ = 0;
    int64_t gradInputLoopTail_ = 0;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T);

    constexpr static int64_t GRADINPUT_ONE_VL = platform::GetVRegSize() / sizeof(INDEX);
    constexpr static int64_t CALC_ONE_VL = platform::GetVRegSize() / sizeof(COMPUTE_TYPE);

    using INDEX_U = std::conditional_t<std::is_same_v<INDEX, int32_t>, uint32_t, uint64_t>;
};

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::Init(
    GM_ADDR gradInput, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    gradInputPlaneSize_ = tilingData_.hInput * tilingData_.wInput;
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    curCoreProcessNum_ = (blockIdx_ + 1 == tilingData_.usedCoreNum) ? tilingData_.tailCoreProcessNum : tilingData_.normalCoreProcessNum;
    gradInputGm_.SetGlobalBuffer((__gm__ T*)gradInput);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_.InitBuffer(outputQue_, BUFFER_NUM, tilingData_.outputBufferSize);
    pipe_.InitBuffer(gradInputQue_, BUFFER_NUM, tilingData_.gradInputBufferSize);

    pipe_.InitBuffer(gmHighIdx_, V_REG_SIZE);
    pipe_.InitBuffer(gmKernelSize_, V_REG_SIZE);
    pipe_.InitBuffer(gmStH_, V_REG_SIZE);
    pipe_.InitBuffer(gmEdH_, V_REG_SIZE);
    pipe_.InitBuffer(gmStW_, V_REG_SIZE);
    pipe_.InitBuffer(gmEdW_, V_REG_SIZE);
    pipe_.InitBuffer(gmGradInputF32_, V_REG_SIZE);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::ScalarCompute(int64_t loopNum)
{   
    int64_t baseBlockIdx = blockIdx_ * tilingData_.normalCoreProcessNum + loopNum;  
    highAxisIndex_ = baseBlockIdx / (tilingData_.hOutputOuter * tilingData_.wOutputOuter); 
    highAxisActual_ = highAxisIndex_ == (tilingData_.highAxisOuter - 1) ? tilingData_.highAxisTail : tilingData_.highAxisInner; 

    int64_t tempTail2 = baseBlockIdx % (tilingData_.hOutputOuter * tilingData_.wOutputOuter);

    hAxisIndex_ = tempTail2 / tilingData_.wOutputOuter;  
    hOutputActual_ = hAxisIndex_ == (tilingData_.hOutputOuter - 1) ? tilingData_.hOutputTail : tilingData_.hOutputInner; 
    wAxisIndex_ = tempTail2 % tilingData_.wOutputOuter;
    wOutputActual_ = wAxisIndex_ == (tilingData_.wOutputOuter - 1) ? tilingData_.wOutputTail : tilingData_.wOutputInner;

    hStLeftCornerIdx_ = GetStartFromOutputInputSize(hAxisIndex_ * tilingData_.hOutputInner, tilingData_.hInput, tilingData_.hOutput);
    wStLeftCornerIdx_ = GetStartFromOutputInputSize(wAxisIndex_ * tilingData_.wOutputInner, tilingData_.wInput, tilingData_.wOutput);
    hEndRightCornerIdx_ = GetEndFromOutputInputSize(hAxisIndex_ * tilingData_.hOutputInner+ hOutputActual_ - 1, tilingData_.hInput, tilingData_.hOutput);
    wEndRightCornerIdx_ = GetEndFromOutputInputSize(wAxisIndex_ * tilingData_.wOutputInner + wOutputActual_ - 1, tilingData_.wInput, tilingData_.wOutput);

    wOutputAligned_ =
        (wOutputActual_ + MAX_DATA_NUM_IN_ONE_BLOCK - 1) / MAX_DATA_NUM_IN_ONE_BLOCK * MAX_DATA_NUM_IN_ONE_BLOCK;

    wGradInputActual_ = wEndRightCornerIdx_ - wStLeftCornerIdx_;
    hGradInputActual_ = hEndRightCornerIdx_ - hStLeftCornerIdx_;

    highAxisGradInputOffset_ = highAxisIndex_ * tilingData_.highAxisInner * gradInputPlaneSize_;
    hAxisGradInputOffset_ = hStLeftCornerIdx_ * tilingData_.wInput;
    wAxisGradInputOffset_ = wStLeftCornerIdx_;

    int64_t gradInputTotalCount =  hGradInputActual_ * wGradInputActual_ * highAxisActual_;
    gradInputLoopCount_ = gradInputTotalCount / GRADINPUT_ONE_VL;
    gradInputLoopTail_ = gradInputTotalCount - gradInputLoopCount_ * GRADINPUT_ONE_VL;
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::CopyIn()
{
    LocalTensor<T> gradInputLocal = gradInputQue_.AllocTensor<T>();
    int64_t gradInputGmOffset = highAxisGradInputOffset_ + hAxisGradInputOffset_ + wAxisGradInputOffset_;
    MultiCopyLoopInfo<NDDMA_DIMS> loopInfo;
    loopInfo.loopSize[0] = wGradInputActual_;
    loopInfo.loopSize[1] = hGradInputActual_;
    loopInfo.loopSize[2] = highAxisActual_;

    loopInfo.loopSrcStride[0] = 1;
    loopInfo.loopSrcStride[1] = tilingData_.wInput;
    loopInfo.loopSrcStride[2] = tilingData_.wInput * tilingData_.hInput;

    loopInfo.loopDstStride[0] = 1;
    loopInfo.loopDstStride[1] = wGradInputActual_;
    loopInfo.loopDstStride[2] = wGradInputActual_ * hGradInputActual_;

    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<T, NDDMA_DIMS> paramsMain = {loopInfo};
    DataCopy<T, NDDMA_DIMS, config>(gradInputLocal, gradInputGm_[gradInputGmOffset], paramsMain);
    gradInputQue_.EnQue(gradInputLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::CopyOut()
{
    LocalTensor<T> yLocal = outputQue_.DeQue<T>();
    int64_t outputPlaneSize = tilingData_.hOutput * tilingData_.wOutput;
    int64_t ncBase = highAxisIndex_ * tilingData_.highAxisInner * outputPlaneSize; 
    int64_t hBase = hAxisIndex_ * tilingData_.hOutputInner* tilingData_.wOutput;
    int64_t wBase = wAxisIndex_ * tilingData_.wOutputInner;
    int64_t outputGmOffset = ncBase + hBase + wBase;

    LoopModeParams loopModeParamsT;
    loopModeParamsT.loop1Size = highAxisActual_;
    loopModeParamsT.loop2Size = 1;
    loopModeParamsT.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T);
    loopModeParamsT.loop2SrcStride = 0;
    loopModeParamsT.loop1DstStride = outputPlaneSize * sizeof(T);
    loopModeParamsT.loop2DstStride = 0; 

    SetLoopModePara(loopModeParamsT, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT = {static_cast<uint16_t>(hOutputActual_),
                                        static_cast<uint32_t>(wOutputActual_ * sizeof(T)), static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((tilingData_.wOutput - wOutputActual_) * sizeof(T)),
                                        static_cast<uint32_t>(0)};
    
    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::ProcessPerLoop()
{
    CopyIn();
    Compute();
    CopyOut();
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::CalGradInputGMHW(
    AscendC::MicroAPI::RegTensor<INDEX>& gradInputUbIdx, AscendC::MicroAPI::RegTensor<INDEX>& gradInputHigh,
    AscendC::MicroAPI::RegTensor<INDEX>& gradInputH, AscendC::MicroAPI::RegTensor<INDEX>& gradInputW,
    AscendC::MicroAPI::RegTensor<INDEX>& gradInputHWValueConstReg,
    AscendC::MicroAPI::RegTensor<INDEX>& gradInputWValueConstReg, AscendC::MicroAPI::MaskReg& allMaskIndex)
{
    AscendC::MicroAPI::RegTensor<INDEX> gradInputUbCalTemp;
    AscendC::MicroAPI::Div(gradInputHigh, gradInputUbIdx, gradInputHWValueConstReg, allMaskIndex);
    AscendC::MicroAPI::Mul(gradInputUbCalTemp, gradInputHigh, gradInputHWValueConstReg, allMaskIndex);
    AscendC::MicroAPI::Sub(gradInputUbIdx, gradInputUbIdx, gradInputUbCalTemp, allMaskIndex);
    AscendC::MicroAPI::Div(gradInputH, gradInputUbIdx, gradInputWValueConstReg, allMaskIndex);
    AscendC::MicroAPI::Mul(gradInputUbCalTemp, gradInputH, gradInputWValueConstReg, allMaskIndex);
    AscendC::MicroAPI::Sub(gradInputW, gradInputUbIdx, gradInputUbCalTemp, allMaskIndex);
    AscendC::MicroAPI::Adds(gradInputH, gradInputH, INDEX(hStLeftCornerIdx_), allMaskIndex);
    AscendC::MicroAPI::Adds(gradInputW, gradInputW, INDEX(wStLeftCornerIdx_), allMaskIndex);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::CalAndCopyGradInputStAndEdSingle(
    AscendC::MicroAPI::RegTensor<INDEX>& gradKernelRegConstBig, AscendC::MicroAPI::RegTensor<INDEX>& gradKernelRegConstSmall,
    AscendC::MicroAPI::RegTensor<INDEX>& gradKernelKernelStIdx, AscendC::MicroAPI::RegTensor<INDEX>& gradKernelKernelEdIdx,
    AscendC::MicroAPI::RegTensor<INDEX>& gradInputIdxValue, AscendC::MicroAPI::RegTensor<INDEX>& gradKernelKernelMulTemp,
    AscendC::MicroAPI::RegTensor<INDEX>& gradKernelSize, __local_mem__ INDEX* gmStAddr,
    __local_mem__ INDEX* gmEdAddr, INDEX idxOutput, INDEX idxGradInput,
    INDEX idxAxisIndex, INDEX idxInner, INDEX idxOutputActual, AscendC::MicroAPI::MaskReg& allMaskIndex
)
{
    AscendC::MicroAPI::Duplicate(gradKernelRegConstBig, idxOutput);
    AscendC::MicroAPI::Duplicate(gradKernelRegConstSmall, idxGradInput);

    AscendC::MicroAPI::Mul(gradKernelKernelStIdx, gradInputIdxValue, gradKernelRegConstBig, allMaskIndex);
    AscendC::MicroAPI::Div(gradKernelKernelStIdx, gradKernelKernelStIdx, gradKernelRegConstSmall, allMaskIndex);
    AscendC::MicroAPI::Adds(gradKernelKernelEdIdx, gradInputIdxValue, INDEX(1), allMaskIndex);
    AscendC::MicroAPI::Mul(gradKernelKernelEdIdx, gradKernelKernelEdIdx, gradKernelRegConstBig, allMaskIndex);
    AscendC::MicroAPI::Add(gradKernelKernelEdIdx, gradKernelKernelEdIdx, gradKernelRegConstSmall, allMaskIndex);
    AscendC::MicroAPI::Adds(gradKernelKernelEdIdx, gradKernelKernelEdIdx, INDEX(-1), allMaskIndex);
    AscendC::MicroAPI::Div(gradKernelKernelEdIdx, gradKernelKernelEdIdx, gradKernelRegConstSmall, allMaskIndex);

    AscendC::MicroAPI::Sub(gradKernelKernelMulTemp, gradKernelKernelEdIdx, gradKernelKernelStIdx, allMaskIndex);
    AscendC::MicroAPI::Mul(gradKernelSize, gradKernelSize, gradKernelKernelMulTemp, allMaskIndex);

    AscendC::MicroAPI::Maxs(gradKernelKernelStIdx, gradKernelKernelStIdx, idxAxisIndex * idxInner, allMaskIndex);
    AscendC::MicroAPI::Adds(gradKernelKernelStIdx, gradKernelKernelStIdx, -idxAxisIndex * idxInner, allMaskIndex);
    AscendC::MicroAPI::Mins(gradKernelKernelEdIdx, gradKernelKernelEdIdx, idxAxisIndex * idxInner + idxOutputActual, allMaskIndex);
    AscendC::MicroAPI::Adds(gradKernelKernelEdIdx, gradKernelKernelEdIdx, -idxAxisIndex * idxInner, allMaskIndex);
    AscendC::MicroAPI::DataCopy(gmStAddr, gradKernelKernelStIdx, allMaskIndex);
    AscendC::MicroAPI::DataCopy(gmEdAddr, gradKernelKernelEdIdx, allMaskIndex);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::CalAndCopyGradInputInfo(
    uint32_t gradInputUbIdxValue, __local_mem__ INDEX* gmStHAddr, __local_mem__ INDEX* gmEdHAddr,
    __local_mem__ INDEX* gmStWAddr, __local_mem__ INDEX* gmEdWAddr, __local_mem__ INDEX* gmHighIdxAddr,
    __local_mem__ INDEX* gmKernelSizeAddr)
{
    INDEX hOutput = tilingData_.hOutput;
    INDEX hGradInput = tilingData_.hInput;
    INDEX hAxisIndex = hAxisIndex_;
    INDEX hOutputInner = tilingData_.hOutputInner;
    INDEX hOutputActual = hOutputActual_;

    INDEX wOutput = tilingData_.wOutput;
    INDEX wGradInput = tilingData_.wInput;
    INDEX wAxisIndex = wAxisIndex_;
    INDEX wOutputInner = tilingData_.wOutputInner;
    INDEX wOutputActual = wOutputActual_;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg allMaskU32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::MaskReg allMaskIndex =
            AscendC::MicroAPI::CreateMask<INDEX, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<INDEX> gradInputUbIdx;
        AscendC::MicroAPI::Arange(gradInputUbIdx, gradInputUbIdxValue);
        AscendC::MicroAPI::RegTensor<INDEX> gradInputHigh;
        AscendC::MicroAPI::RegTensor<INDEX> gradInputH;
        AscendC::MicroAPI::RegTensor<INDEX> gradInputW;

        AscendC::MicroAPI::RegTensor<INDEX> gradInputHWValueConstReg;
        AscendC::MicroAPI::Duplicate(gradInputHWValueConstReg, INDEX(hGradInputActual_ * wGradInputActual_));
        AscendC::MicroAPI::RegTensor<INDEX> gradInputWValueConstReg;
        AscendC::MicroAPI::Duplicate(gradInputWValueConstReg, INDEX(wGradInputActual_));
        CalGradInputGMHW(
            gradInputUbIdx, gradInputHigh, gradInputH, gradInputW, gradInputHWValueConstReg, gradInputWValueConstReg,
            allMaskIndex);
        AscendC::MicroAPI::DataCopy(gmHighIdxAddr, gradInputHigh, allMaskIndex);
        AscendC::MicroAPI::DataCopy(gmStHAddr, gradInputH, allMaskIndex);
        AscendC::MicroAPI::DataCopy(gmStWAddr, gradInputW, allMaskIndex);
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg allMaskIndex =
                    AscendC::MicroAPI::CreateMask<INDEX, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::RegTensor<INDEX> gradInputH;
        AscendC::MicroAPI::DataCopy(gradInputH, gmStHAddr);

        AscendC::MicroAPI::RegTensor<INDEX> gradKernelSize;
        AscendC::MicroAPI::Duplicate(gradKernelSize, INDEX(1));
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelKernelStIdx;
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelKernelEdIdx;
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelRegConstBig;
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelRegConstSmall;
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelKernelMulTemp;

        CalAndCopyGradInputStAndEdSingle(
            gradKernelRegConstBig, gradKernelRegConstSmall, gradKernelKernelStIdx,
            gradKernelKernelEdIdx, gradInputH, gradKernelKernelMulTemp, gradKernelSize,
            gmStHAddr, gmEdHAddr, hOutput, hGradInput,
            hAxisIndex, hOutputInner, hOutputActual,
            allMaskIndex
        );
        AscendC::MicroAPI::DataCopy(gmKernelSizeAddr, gradKernelSize, allMaskIndex);
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg allMaskIndex =
                    AscendC::MicroAPI::CreateMask<INDEX, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::RegTensor<INDEX> gradInputW;
        AscendC::MicroAPI::DataCopy(gradInputW, gmStWAddr);

        AscendC::MicroAPI::RegTensor<INDEX> gradKernelSize;
        AscendC::MicroAPI::DataCopy(gradKernelSize, gmKernelSizeAddr);
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelKernelStIdx;
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelKernelEdIdx;
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelRegConstBig;
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelRegConstSmall;
        AscendC::MicroAPI::RegTensor<INDEX> gradKernelKernelMulTemp;

        CalAndCopyGradInputStAndEdSingle(
            gradKernelRegConstBig, gradKernelRegConstSmall, gradKernelKernelStIdx,
            gradKernelKernelEdIdx, gradInputW, gradKernelKernelMulTemp, gradKernelSize,
            gmStWAddr, gmEdWAddr, wOutput, wGradInput,
            wAxisIndex, wOutputInner, wOutputActual,
            allMaskIndex
        );
        AscendC::MicroAPI::DataCopy(gmKernelSizeAddr, gradKernelSize, allMaskIndex);
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::GatherCopyGradUb2Reg(
    AscendC::MicroAPI::RegTensor<INDEX>& gradOutputUBIdx, AscendC::MicroAPI::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
    __local_mem__ COMPUTE_TYPE* yAddr, uint32_t& maskCount
)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::MicroAPI::MaskReg pregU32 = AscendC::MicroAPI::UpdateMask<uint32_t>(maskCountTemp);
    if constexpr (std::is_same<INDEX, int64_t>::value) {
        AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<INDEX, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<int32_t> gradOutputUBIdxI32;
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(gradOutputUBIdxI32, gradOutputUBIdx, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)gradOutputUBIdxI32,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)gradOutputUBIdxI32);
        AscendC::MicroAPI::DataCopyGather(gradOutputUbValue, yAddr, (AscendC::MicroAPI::RegTensor<uint32_t>&)gradOutputUBIdxI32, pregU32);
    } else {
        AscendC::MicroAPI::DataCopyGather(gradOutputUbValue, yAddr, (AscendC::MicroAPI::RegTensor<uint32_t>&)gradOutputUBIdx, pregU32);
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::ScatterCopyGradReg2Ub(
    AscendC::MicroAPI::RegTensor<INDEX>& gradOutputUBIdx, AscendC::MicroAPI::RegTensor<COMPUTE_TYPE>& gradOutputUbValue,
    __local_mem__ COMPUTE_TYPE* yAddr, uint32_t& maskCount
)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::MicroAPI::MaskReg pregU32 = AscendC::MicroAPI::UpdateMask<uint32_t>(maskCountTemp);
    if constexpr (std::is_same<INDEX, int64_t>::value) {
        AscendC::MicroAPI::MaskReg allMask = AscendC::MicroAPI::CreateMask<INDEX, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<int32_t> gradOutputUBIdxI32;
        AscendC::MicroAPI::Cast<int32_t, int64_t, castTraitI64I32>(gradOutputUBIdxI32, gradOutputUBIdx, allMask);
        AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)gradOutputUBIdxI32,
                                (AscendC::MicroAPI::RegTensor<int64_t>&)gradOutputUBIdxI32);
        AscendC::MicroAPI::DataCopyScatter(yAddr, gradOutputUbValue, (AscendC::MicroAPI::RegTensor<uint32_t>&)gradOutputUBIdxI32, pregU32);
    } else {
        AscendC::MicroAPI::DataCopyScatter(yAddr, gradOutputUbValue, (AscendC::MicroAPI::RegTensor<uint32_t>&)gradOutputUBIdx, pregU32);
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::DoGradRegAdds(
    AscendC::MicroAPI::RegTensor<COMPUTE_TYPE>& gradOutputUbValue, COMPUTE_TYPE& gradInputValue,
    __local_mem__ COMPUTE_TYPE* yAddr, uint32_t& maskCount
)
{
    uint32_t maskCountTemp = maskCount;
    AscendC::MicroAPI::MaskReg pregU32 = AscendC::MicroAPI::UpdateMask<uint32_t>(maskCountTemp);
    AscendC::MicroAPI::Adds(gradOutputUbValue, gradOutputUbValue, COMPUTE_TYPE(gradInputValue), pregU32);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::ComputeGradIndexHW(
    AscendC::MicroAPI::RegTensor<INDEX>& gradOutputUBIdx, INDEX gradKernelW, INDEX gradStHIdxOffset,
    INDEX gradStWIdx, INDEX highHOffset, AscendC::MicroAPI::MaskReg& pregU32MaskAll
)
{
    AscendC::MicroAPI::Arange(gradOutputUBIdx, INDEX(0));

    AscendC::MicroAPI::RegTensor<INDEX> regConst;
    AscendC::MicroAPI::Duplicate(regConst, INDEX(gradKernelW));
    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBDivKW;

    AscendC::MicroAPI::Div(gradOutputUBDivKW, gradOutputUBIdx, regConst, pregU32MaskAll);
    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBCellAlignKW;
    AscendC::MicroAPI::Mul(gradOutputUBCellAlignKW, gradOutputUBDivKW, regConst, pregU32MaskAll);
    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBLineInKW;
    AscendC::MicroAPI::Sub(gradOutputUBLineInKW, gradOutputUBIdx, gradOutputUBCellAlignKW, pregU32MaskAll);

    AscendC::MicroAPI::Duplicate(regConst, INDEX(gradStHIdxOffset));
    AscendC::MicroAPI::Add(gradOutputUBDivKW, gradOutputUBDivKW, regConst, pregU32MaskAll);
    AscendC::MicroAPI::Duplicate(regConst, INDEX(wOutputAligned_));
    AscendC::MicroAPI::Mul(gradOutputUBDivKW, gradOutputUBDivKW, regConst, pregU32MaskAll);

    AscendC::MicroAPI::Add(gradOutputUBLineInKW, gradOutputUBLineInKW, gradOutputUBDivKW, pregU32MaskAll);
    AscendC::MicroAPI::Duplicate(regConst, INDEX(gradStWIdx));
    AscendC::MicroAPI::Add(gradOutputUBIdx, gradOutputUBLineInKW, regConst, pregU32MaskAll);
    AscendC::MicroAPI::Duplicate(regConst, INDEX(highHOffset));
    AscendC::MicroAPI::Add(gradOutputUBIdx, gradOutputUBIdx, regConst, pregU32MaskAll);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::ComputeGradIndexEntireHW(
    AscendC::MicroAPI::RegTensor<INDEX>& gradOutputUBIdx, INDEX gradKernelW,
    INDEX outputHWAlign, INDEX gradStHIdx, INDEX gradStWIdx, INDEX highOffset, AscendC::MicroAPI::MaskReg& pregU32MaskAll
)
{
    AscendC::MicroAPI::Arange(gradOutputUBIdx, INDEX(0));

    AscendC::MicroAPI::RegTensor<INDEX> regConst;
    AscendC::MicroAPI::Duplicate(regConst, INDEX(gradKernelW));
    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBIdxDivKW;
    AscendC::MicroAPI::Div(gradOutputUBIdxDivKW, gradOutputUBIdx, regConst, pregU32MaskAll);
    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBIdxCeilAlignKW;
    AscendC::MicroAPI::Mul(gradOutputUBIdxCeilAlignKW, gradOutputUBIdxDivKW, regConst, pregU32MaskAll);

    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBIdxLineInW;
    AscendC::MicroAPI::Sub(gradOutputUBIdxLineInW, gradOutputUBIdx, gradOutputUBIdxCeilAlignKW, pregU32MaskAll);

    AscendC::MicroAPI::Duplicate(regConst, INDEX(gradStHIdx));
    AscendC::MicroAPI::Add(gradOutputUBIdxDivKW, gradOutputUBIdxDivKW, regConst, pregU32MaskAll);
    AscendC::MicroAPI::Duplicate(regConst, INDEX(wOutputAligned_));
    AscendC::MicroAPI::Mul(gradOutputUBIdxDivKW, gradOutputUBIdxDivKW, regConst, pregU32MaskAll);
    AscendC::MicroAPI::Add(gradOutputUBIdxLineInW, gradOutputUBIdxLineInW, gradOutputUBIdxDivKW, pregU32MaskAll);

    AscendC::MicroAPI::Duplicate(regConst, INDEX(gradStWIdx));
    AscendC::MicroAPI::Add(gradOutputUBIdx, gradOutputUBIdxLineInW, regConst, pregU32MaskAll);

    AscendC::MicroAPI::Duplicate(regConst, INDEX(highOffset));
    AscendC::MicroAPI::Add(gradOutputUBIdx, gradOutputUBIdx, regConst, pregU32MaskAll);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::DoGradInputAccUb(
    uint32_t gradInputLoopCount, uint32_t gradInputUbIdxValue, __local_mem__ T* gradAddr, __local_mem__ COMPUTE_TYPE* yAddr,
    __local_mem__ INDEX* gmStHAddr, __local_mem__ INDEX* gmEdHAddr, __local_mem__ INDEX* gmStWAddr, __local_mem__ INDEX* gmEdWAddr,
    __local_mem__ INDEX* gmHighIdxAddr, __local_mem__ INDEX* gmKernelSizeAddr, __local_mem__ COMPUTE_TYPE* gmGradInputF32Addr
)
{
    for (uint32_t loopGradInputIdx = 0; loopGradInputIdx < gradInputLoopCount; ++loopGradInputIdx) {

        uint32_t loopGradInputUbRealIdx = loopGradInputIdx + gradInputUbIdxValue;

        COMPUTE_TYPE gradInputValue = static_cast<COMPUTE_TYPE>(gmGradInputF32Addr[loopGradInputIdx]);
        INDEX gradHightIdx = gmHighIdxAddr[loopGradInputIdx];
        INDEX gradStHIdx = gmStHAddr[loopGradInputIdx];
        INDEX gradEdHIdx = gmEdHAddr[loopGradInputIdx];
        INDEX gradStWIdx = gmStWAddr[loopGradInputIdx];
        INDEX gradEdWIdx = gmEdWAddr[loopGradInputIdx];
        INDEX gradKernelH = gradEdHIdx - gradStHIdx;
        INDEX gradKernelW = gradEdWIdx - gradStWIdx;
        if (gradKernelW * DOUBLE * sizeof(COMPUTE_TYPE) > V_REG_SIZE) {
            uint32_t vlFullSize = CALC_ONE_VL;
            uint16_t wloopCount = gradKernelW / vlFullSize;
            uint32_t wloopCountTail = gradKernelW % vlFullSize;

            for (uint32_t hloopIdx = gradStHIdx; hloopIdx < gradEdHIdx; ++hloopIdx) {
                INDEX highHWOffset = gradHightIdx  * hOutputActual_ * wOutputAligned_ +
                                      hloopIdx * (wOutputAligned_) + gradStWIdx;
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                    AscendC::MicroAPI::MaskReg pregU32 =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
                    AscendC::MicroAPI::UnalignReg u0;
                    AscendC::MicroAPI::UnalignReg u1;
                    auto yAddrSrcUnalign = yAddr + highHWOffset;
                    auto yAddrDstUnalign = yAddr + highHWOffset;
                    AscendC::MicroAPI::DataCopyUnAlignPre(u0, yAddrSrcUnalign);
                    for (uint16_t wloopIdx = 0; wloopIdx < wloopCount; ++wloopIdx) {
                        AscendC::MicroAPI::DataCopyUnAlign(gradOutputUbValue, u0, yAddrSrcUnalign, vlFullSize);
                        AscendC::MicroAPI::Adds(
                            gradOutputUbValue, gradOutputUbValue, static_cast<COMPUTE_TYPE>(gradInputValue), pregU32);
                        AscendC::MicroAPI::DataCopyUnAlign(yAddrDstUnalign, gradOutputUbValue, u1, vlFullSize);
                    }
                    AscendC::MicroAPI::DataCopyUnAlignPost(yAddrDstUnalign, u1, 0);

                    highHWOffset += wloopCount * vlFullSize;
                    auto yAddrUnalign = yAddr + highHWOffset;
                    AscendC::MicroAPI::DataCopyUnAlignPre(u0, yAddrUnalign);
                    AscendC::MicroAPI::DataCopyUnAlign(gradOutputUbValue, u0, yAddrUnalign);
                    DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, wloopCountTail);
                    AscendC::MicroAPI::DataCopyUnAlign(yAddrUnalign, gradOutputUbValue, u0, wloopCountTail);
                    AscendC::MicroAPI::DataCopyUnAlignPost(yAddrUnalign, u0, wloopCountTail);
                }
            }
        } else if (gradKernelH * gradKernelW * DOUBLE * sizeof(COMPUTE_TYPE) > V_REG_SIZE) {
            uint32_t vlFullSize = CALC_ONE_VL;
            uint32_t maxWParaNum = vlFullSize / gradKernelW;
            uint16_t hwloopCount = gradKernelH / maxWParaNum;
            uint32_t hwloopCountTail = gradKernelH % maxWParaNum;
            uint32_t maskCountFull = gradKernelW * maxWParaNum;
            uint32_t maskCountTail = gradKernelW * hwloopCountTail;
            INDEX highDHOffset = gradHightIdx * hOutputActual_ * wOutputAligned_;
            if constexpr (std::is_same<INDEX, int32_t>::value) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                    AscendC::MicroAPI::MaskReg pregU32MaskAll =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

                    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBIdx;
                    for (uint16_t hwloopIdx = 0; hwloopIdx < hwloopCount; ++hwloopIdx) {
                        INDEX gradStHIdxOffset = gradStHIdx + hwloopIdx * maxWParaNum;
                        ComputeGradIndexHW(
                            gradOutputUBIdx, gradKernelW, gradStHIdxOffset, gradStWIdx, highDHOffset, pregU32MaskAll);
                        GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                        DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountFull);
                        ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                    }

                    INDEX gradStHIdxOffset = gradStHIdx + hwloopCount * maxWParaNum;
                    ComputeGradIndexHW(
                        gradOutputUBIdx, gradKernelW, gradStHIdxOffset, gradStWIdx, highDHOffset, pregU32MaskAll);
                    GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                    DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountTail);
                    ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                }
            } else {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                    AscendC::MicroAPI::MaskReg pregU32MaskAll =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

                    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBIdx;
                    for (uint16_t hwloopIdx = 0; hwloopIdx < hwloopCount; ++hwloopIdx) {
                        INDEX gradStHIdxOffset = gradStHIdx + hwloopIdx * maxWParaNum;
                        ComputeGradIndexHW(
                            gradOutputUBIdx, gradKernelW, gradStHIdxOffset, gradStWIdx, highDHOffset, pregU32MaskAll);
                        GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                        DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountFull);
                        ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                    }
                }

                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                    AscendC::MicroAPI::MaskReg pregU32MaskAll =
                        AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

                    AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBIdx;
                    INDEX gradStHIdxOffset = gradStHIdx + hwloopCount * maxWParaNum;
                    ComputeGradIndexHW(
                        gradOutputUBIdx, gradKernelW, gradStHIdxOffset, gradStWIdx, highDHOffset, pregU32MaskAll);
                    GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                    DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountTail);
                    ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountTail);
                }
            }

        } else {
            INDEX highOffset = gradHightIdx * hOutputActual_ * wOutputAligned_;
            uint32_t maskCountFull = gradKernelH * gradKernelW;
            uint32_t hOutputActualU32 = hOutputActual_;
            uint32_t wOutputAlignedU32 = wOutputAligned_;
            uint32_t outputHWAlign = hOutputActualU32 * wOutputAlignedU32;

            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> gradOutputUbValue;
                AscendC::MicroAPI::MaskReg pregU32MaskAll = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

                AscendC::MicroAPI::RegTensor<INDEX> gradOutputUBIdx;
                ComputeGradIndexEntireHW(
                    gradOutputUBIdx, gradKernelW,
                    outputHWAlign, gradStHIdx, gradStWIdx, highOffset, pregU32MaskAll
                );
                GatherCopyGradUb2Reg(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
                DoGradRegAdds(gradOutputUbValue, gradInputValue, yAddr, maskCountFull);
                ScatterCopyGradReg2Ub(gradOutputUBIdx, gradOutputUbValue, yAddr, maskCountFull);
            }
        }
    } 
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::DoAllGradInputProcess(
    uint32_t gradInputLoopCount, uint32_t gradInputUbIdxValue, __local_mem__ T* gradAddr, __local_mem__ COMPUTE_TYPE* yAddr,
    __local_mem__ INDEX* gmStHAddr, __local_mem__ INDEX* gmEdHAddr, __local_mem__ INDEX* gmStWAddr, __local_mem__ INDEX* gmEdWAddr,
    __local_mem__ INDEX* gmHighIdxAddr, __local_mem__ INDEX* gmKernelSizeAddr, __local_mem__ COMPUTE_TYPE* gmGradInputF32Addr
)
{
    CalAndCopyGradInputInfo(gradInputUbIdxValue, gmStHAddr, gmEdHAddr,
        gmStWAddr, gmEdWAddr, gmHighIdxAddr, gmKernelSizeAddr);

    if constexpr (std::is_same<INDEX, int64_t>::value) {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> gradInputUbValue;
            AscendC::MicroAPI::MaskReg pregT = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg allMaskU32 =
                AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            ops::LoadOneTensorForDtypeT<T>(gradAddr, gradInputUbValue, pregT, gradInputUbIdxValue);

            AscendC::MicroAPI::RegTensor<INDEX> kernelSize;
            AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> kernelSizeF32;
            AscendC::MicroAPI::DataCopy(kernelSize, gmKernelSizeAddr);
            AscendC::MicroAPI::MaskReg allMaskU =
                AscendC::MicroAPI::CreateMask<INDEX, AscendC::MicroAPI::MaskPattern::ALL>();

            AscendC::MicroAPI::Cast<COMPUTE_TYPE, INDEX, castTraitI64TF32>(
                kernelSizeF32, kernelSize, allMaskU);
            AscendC::MicroAPI::Pack((AscendC::MicroAPI::RegTensor<uint32_t>&)kernelSizeF32,
                                    (AscendC::MicroAPI::RegTensor<uint64_t>&)kernelSizeF32);
            AscendC::MicroAPI::Div(gradInputUbValue, gradInputUbValue, kernelSizeF32, allMaskU32);
            AscendC::MicroAPI::DataCopy(gmGradInputF32Addr, gradInputUbValue, allMaskU32);
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> gradInputUbValue;
            AscendC::MicroAPI::MaskReg pregT = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg allMaskU32 =
                AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            ops::LoadOneTensorForDtypeT<T>(gradAddr, gradInputUbValue, pregT, gradInputUbIdxValue);

            AscendC::MicroAPI::RegTensor<INDEX> kernelSize;
            AscendC::MicroAPI::RegTensor<COMPUTE_TYPE> kernelSizeF32;
            AscendC::MicroAPI::DataCopy(kernelSize, gmKernelSizeAddr);

            AscendC::MicroAPI::Cast<COMPUTE_TYPE, INDEX, castTraitI32TF32>(
                kernelSizeF32, kernelSize, allMaskU32);
            AscendC::MicroAPI::Div(gradInputUbValue, gradInputUbValue, kernelSizeF32, allMaskU32);
            AscendC::MicroAPI::DataCopy(gmGradInputF32Addr, gradInputUbValue, allMaskU32);
        }
    }
    PIPE_V_S();
    DoGradInputAccUb(gradInputLoopCount, gradInputUbIdxValue, gradAddr, yAddr, gmStHAddr, gmEdHAddr,
        gmStWAddr, gmEdWAddr, gmHighIdxAddr, gmKernelSizeAddr, gmGradInputF32Addr);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradBigKernel<T, INDEX>::Compute()
{
    LocalTensor<COMPUTE_TYPE> yLocal = outputQue_.AllocTensor<COMPUTE_TYPE>();
    uint32_t calCount = tilingData_.outputBufferSize / sizeof(COMPUTE_TYPE);
    Duplicate(yLocal, COMPUTE_TYPE(0), calCount);
    LocalTensor<T> gradLocal = gradInputQue_.DeQue<T>();

    LocalTensor<INDEX> gmhighIdxLocal = gmHighIdx_.DeQue<INDEX>();
    LocalTensor<INDEX> gmKernelSizeLocal = gmKernelSize_.DeQue<INDEX>();
    LocalTensor<INDEX> gmStHLocal = gmStH_.DeQue<INDEX>();
    LocalTensor<INDEX> gmEdHLocal = gmEdH_.DeQue<INDEX>();
    LocalTensor<INDEX> gmStWLocal = gmStW_.DeQue<INDEX>();
    LocalTensor<INDEX> gmEdWLocal = gmEdW_.DeQue<INDEX>();

    __local_mem__ COMPUTE_TYPE* yAddr = (__local_mem__ COMPUTE_TYPE*)yLocal.GetPhyAddr();
    __local_mem__ T* gradAddr = (__local_mem__ T*)gradLocal.GetPhyAddr();
    __local_mem__ INDEX* gmHighIdxAddr = (__local_mem__ INDEX*)gmhighIdxLocal.GetPhyAddr();
    __local_mem__ INDEX* gmKernelSizeAddr = (__local_mem__ INDEX*)gmKernelSizeLocal.GetPhyAddr();
    __local_mem__ INDEX* gmStHAddr = (__local_mem__ INDEX*)gmStHLocal.GetPhyAddr();
    __local_mem__ INDEX* gmEdHAddr = (__local_mem__ INDEX*)gmEdHLocal.GetPhyAddr();
    __local_mem__ INDEX* gmStWAddr = (__local_mem__ INDEX*)gmStWLocal.GetPhyAddr();
    __local_mem__ INDEX* gmEdWAddr = (__local_mem__ INDEX*)gmEdWLocal.GetPhyAddr();

    LocalTensor<COMPUTE_TYPE> gmGradInputF32Local;
    __local_mem__ COMPUTE_TYPE* gmGradInputF32Addr;
    gmGradInputF32Local = gmGradInputF32_.DeQue<COMPUTE_TYPE>();
    gmGradInputF32Addr = (__local_mem__ COMPUTE_TYPE*)gmGradInputF32Local.GetPhyAddr();

    uint32_t gradInputUbIdx = 0;
    for (int32_t i = 0; i < gradInputLoopCount_; ++i) {
        DoAllGradInputProcess(static_cast<uint32_t>(GRADINPUT_ONE_VL), gradInputUbIdx, gradAddr, yAddr, gmStHAddr, gmEdHAddr,
            gmStWAddr, gmEdWAddr, gmHighIdxAddr, gmKernelSizeAddr, gmGradInputF32Addr);
        gradInputUbIdx += GRADINPUT_ONE_VL;
    }

    if (gradInputLoopTail_ > 0)
    {
        DoAllGradInputProcess(gradInputLoopTail_, gradInputUbIdx, gradAddr, yAddr, gmStHAddr, gmEdHAddr,
            gmStWAddr, gmEdWAddr, gmHighIdxAddr, gmKernelSizeAddr, gmGradInputF32Addr);
    }

    if constexpr (std::negation<std::is_same<T, COMPUTE_TYPE>>::value) {
        Cast(yLocal.ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, calCount);
    }
    outputQue_.EnQue(yLocal);
    gradInputQue_.FreeTensor(gradLocal);
}
}

#endif