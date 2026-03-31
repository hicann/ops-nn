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
 * \file adaptive_avg_pool3d_grad_ncdhw_small_kernel.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_SMALL_KERNEL_IMPL_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_SMALL_KERNEL_IMPL_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adaptive_avg_pool3d_grad_struct.h"
#include "adaptive_avg_pool3d_grad_ncdhw_big_kernel.h"

namespace AdaptiveAvgPool3dGradOp {
using namespace AscendC;
using namespace ops;

constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t TRANS_LEN_B32 = 8;

template <typename T, typename INDEX>
class AdaptiveAvgPool3dGradNCDHWSmallKernel {
public:
    __aicore__ inline AdaptiveAvgPool3dGradNCDHWSmallKernel() {}

    __aicore__ inline void Init(
        GM_ADDR gradInput,
        GM_ADDR y,
        TPipe& pipeIn,
        const AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35& tilingData);

    __aicore__ inline void ParseTilingData(
        const AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35& tilingData);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyIn();
    __aicore__ inline void TransInput(uint32_t rowNum, uint32_t colNum);
    __aicore__ inline void Compute();
    __aicore__ inline void ComputeWithFp32Acc();
    __aicore__ inline void TransOut();
    __aicore__ inline void CopyOut();

    __aicore__ inline void TransposeB16(
        LocalTensor<T> dst,
        LocalTensor<T> src,
        uint32_t rowNum,
        uint32_t colNum);

    template <typename I>
    __aicore__ inline void TransposeB32(
        LocalTensor<I> dst,
        LocalTensor<I> src,
        uint32_t rowNum,
        uint32_t colNum);

private:
    __aicore__ inline void CalcOutputRangeFromInputIndex(
        int64_t inputIdxGlobal,
        int64_t outputSize,
        int64_t inputSize,
        int64_t axisTileIndex,
        int64_t axisInner,
        int64_t axisOutputActual,
        int64_t& stLocal,
        int64_t& edLocal,
        int64_t& coverCount) const;

    __aicore__ inline void AccumulateOutputRowsForInputPointReg(
        LocalTensor<T> srcLocal,
        LocalTensor<T> dstLocal,
        int64_t inBase,
        T scale,
        int64_t stD,
        int64_t edD,
        int64_t stH,
        int64_t edH,
        int64_t stW,
        int64_t edW);

    __aicore__ inline void AccumulateOutputRowsForInputPointRegFp32(
        LocalTensor<COMPUTE_TYPE> srcLocal,
        LocalTensor<COMPUTE_TYPE> dstLocal,
        int64_t inBase,
        COMPUTE_TYPE scale,
        int64_t stD,
        int64_t edD,
        int64_t stH,
        int64_t edH,
        int64_t stW,
        int64_t edW);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> transQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> transOutQue_;

    // 仅供 half/bfloat16 升精度使用
    TBuf<QuePosition::VECCALC> computeSrcBuf_;
    TBuf<QuePosition::VECCALC> computeAccumBuf_;

    GlobalTensor<T> gradInputGm_;
    GlobalTensor<T> yGm_;

    uint32_t blockIdx_ = 0;
    int64_t curLoopNum_ = 0;

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

    int64_t normalCoreProcessNum_ = 1;
    int64_t tailCoreProcessNum_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t usedCoreNum_ = 1;

    // 逐 queue buffer size
    int64_t inputQueBufferSize_ = 1;
    int64_t transQueBufferSize_ = 1;
    int64_t transOutQueBufferSize_ = 1;

    // float scratch buffer size（仅 half/bfloat16 有效）
    int64_t computeSrcBufferSize_ = 0;
    int64_t computeAccumBufferSize_ = 0;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;

    int64_t hGradInputActual_ = 0;
    int64_t dGradInputActual_ = 0;
    int64_t wGradInputActual_ = 0;

    int64_t gradInputPlaneSize_ = 0;
    int64_t outputPlaneSize_ = 0;

    int64_t highAxisGradInputOffset_ = 0;
    int64_t hAxisGradInputOffset_ = 0;
    int64_t dAxisGradInputOffset_ = 0;
    int64_t wAxisGradInputOffset_ = 0;

    int64_t dStLeftCornerIdx_ = 0;
    int64_t hStLeftCornerIdx_ = 0;
    int64_t wStLeftCornerIdx_ = 0;
    int64_t dEndRightCornerIdx_ = 0;
    int64_t hEndRightCornerIdx_ = 0;
    int64_t wEndRightCornerIdx_ = 0;

    int64_t highAxisAligned_ = 1;
    int64_t wGradInputAligned_ = 1;
    int64_t inputColNum_ = 1;
    int64_t wOutputAligned_ = 1;
    int64_t outputRowNum_ = 1;
    int64_t outputRowNumAligned_ = 1;

    uint32_t vfLen_ = 1;
    uint32_t ubBlockSize_ = 1;
    uint32_t ubAlignNum_ = 1;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T);
    constexpr static int64_t COMPUTE_VF_LEN = platform::GetVRegSize() / sizeof(COMPUTE_TYPE);
};

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::ParseTilingData(
    const AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35& tilingData)
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

    inputQueBufferSize_ = tilingData.inputQueBufferSize;
    transQueBufferSize_ = tilingData.transQueBufferSize;
    transOutQueBufferSize_ = tilingData.transOutQueBufferSize;

    if constexpr (std::is_same_v<T, float>) {
        computeSrcBufferSize_ = 0;
        computeAccumBufferSize_ = 0;
    } else {
        computeSrcBufferSize_ =
            (transQueBufferSize_ / static_cast<int64_t>(sizeof(T))) *
            static_cast<int64_t>(sizeof(COMPUTE_TYPE));
        computeAccumBufferSize_ =
            (transOutQueBufferSize_ / static_cast<int64_t>(sizeof(T))) *
            static_cast<int64_t>(sizeof(COMPUTE_TYPE));
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::Init(
    GM_ADDR gradInput,
    GM_ADDR y,
    TPipe& pipeIn,
    const AdaptiveAvgPool3dNCDHWGradSmallKernelTilingDataV35& tilingData)
{
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    gradInputPlaneSize_ = dGradInput_ * hGradInput_ * wGradInput_;
    outputPlaneSize_ = dOutput_ * hOutput_ * wOutput_;
    vfLen_ = platform::GetVRegSize() / sizeof(T);
    ubBlockSize_ = platform::GetUbBlockSize();
    ubAlignNum_ = ubBlockSize_ / sizeof(T);

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    curCoreProcessNum_ =
        (blockIdx_ + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;

    gradInputGm_.SetGlobalBuffer((__gm__ T*)gradInput);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_ = pipeIn;
    pipe_.InitBuffer(inputQue_, BUFFER_NUM, inputQueBufferSize_);
    pipe_.InitBuffer(transQue_, BUFFER_NUM, transQueBufferSize_);
    pipe_.InitBuffer(transOutQue_, BUFFER_NUM, transOutQueBufferSize_);

    if constexpr (!std::is_same_v<T, float>) {
        pipe_.InitBuffer(computeSrcBuf_, computeSrcBufferSize_);
        pipe_.InitBuffer(computeAccumBuf_, computeAccumBufferSize_);
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::ScalarCompute(int64_t loopNum)
{
    curLoopNum_ = loopNum;
    int64_t baseBlockIdx = blockIdx_ * normalCoreProcessNum_ + loopNum;

    highAxisIndex_ = baseBlockIdx / (dOutputOuter_ * hOutputOuter_ * wOutputOuter_);
    highAxisActual_ = (highAxisIndex_ == (highAxisOuter_ - 1)) ? highAxisTail_ : highAxisInner_;

    int64_t tempTail = baseBlockIdx % (dOutputOuter_ * hOutputOuter_ * wOutputOuter_);
    dAxisIndex_ = tempTail / (hOutputOuter_ * wOutputOuter_);
    dOutputActual_ = (dAxisIndex_ == (dOutputOuter_ - 1)) ? dOutputTail_ : dOutputInner_;

    int64_t tempTail2 = tempTail % (hOutputOuter_ * wOutputOuter_);
    hAxisIndex_ = tempTail2 / wOutputOuter_;
    hOutputActual_ = (hAxisIndex_ == (hOutputOuter_ - 1)) ? hOutputTail_ : hOutputInner_;

    wAxisIndex_ = tempTail2 % wOutputOuter_;
    wOutputActual_ = (wAxisIndex_ == (wOutputOuter_ - 1)) ? wOutputTail_ : wOutputInner_;

    dStLeftCornerIdx_ =
        GetStartFromOutputInputSize(dAxisIndex_ * dOutputInner_, dGradInput_, dOutput_);
    hStLeftCornerIdx_ =
        GetStartFromOutputInputSize(hAxisIndex_ * hOutputInner_, hGradInput_, hOutput_);
    wStLeftCornerIdx_ =
        GetStartFromOutputInputSize(wAxisIndex_ * wOutputInner_, wGradInput_, wOutput_);

    dEndRightCornerIdx_ =
        GetEndFromOutputInputSize(
            dAxisIndex_ * dOutputInner_ + dOutputActual_ - 1, dGradInput_, dOutput_);
    hEndRightCornerIdx_ =
        GetEndFromOutputInputSize(
            hAxisIndex_ * hOutputInner_ + hOutputActual_ - 1, hGradInput_, hOutput_);
    wEndRightCornerIdx_ =
        GetEndFromOutputInputSize(
            wAxisIndex_ * wOutputInner_ + wOutputActual_ - 1, wGradInput_, wOutput_);

    // 搬运 y_grad 的实际长度
    dGradInputActual_ = dEndRightCornerIdx_ - dStLeftCornerIdx_;
    hGradInputActual_ = hEndRightCornerIdx_ - hStLeftCornerIdx_;
    wGradInputActual_ = wEndRightCornerIdx_ - wStLeftCornerIdx_;

    // 按转置接口对齐
    highAxisAligned_ = highAxisInner_;
    wGradInputAligned_ = CeilAlign(wGradInputActual_, static_cast<int64_t>(MAX_DATA_NUM_IN_ONE_BLOCK));

    // 输入转置列数
    inputColNum_ = dGradInputActual_ * hGradInputActual_ * wGradInputAligned_;

    wOutputAligned_ = CeilAlign(wOutputActual_, static_cast<int64_t>(MAX_DATA_NUM_IN_ONE_BLOCK));

    // 输出转置列数
    outputRowNum_ = dOutputActual_ * hOutputActual_ * wOutputAligned_;
    outputRowNumAligned_ = CeilAlign(outputRowNum_, static_cast<int64_t>(TRANS_ADDR_LEN));

    // GM 偏移
    highAxisGradInputOffset_ = highAxisIndex_ * highAxisInner_ * gradInputPlaneSize_;
    dAxisGradInputOffset_ = dStLeftCornerIdx_ * hGradInput_ * wGradInput_;
    hAxisGradInputOffset_ = hStLeftCornerIdx_ * wGradInput_;
    wAxisGradInputOffset_ = wStLeftCornerIdx_;
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::CopyIn()
{
    LocalTensor<T> gradInputLocal = inputQue_.AllocTensor<T>();
    Duplicate(gradInputLocal, (T)0, inputQueBufferSize_ / sizeof(T));

    // 当前 tile 的起始偏移
    const int64_t gradInputGmOffset =
        highAxisGradInputOffset_ + dAxisGradInputOffset_ +
        hAxisGradInputOffset_ + wAxisGradInputOffset_;

    DataCopyExtParams paramsIn = {
        static_cast<uint16_t>(hGradInputActual_),
        static_cast<uint32_t>(wGradInputActual_ * sizeof(T)),
        static_cast<uint32_t>((wGradInput_ - wGradInputActual_) * sizeof(T)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };

    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = dGradInputActual_;
    loopModeParams.loop2Size = highAxisActual_;
    loopModeParams.loop1SrcStride = hGradInput_ * wGradInput_ * sizeof(T);
    loopModeParams.loop2SrcStride = dGradInput_ * hGradInput_ * wGradInput_ * sizeof(T);
    loopModeParams.loop1DstStride = hGradInputActual_ * wGradInputAligned_ * sizeof(T);
    loopModeParams.loop2DstStride = dGradInputActual_ * hGradInputActual_ * wGradInputAligned_ * sizeof(T);

    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(gradInputLocal, gradInputGm_[gradInputGmOffset], paramsIn, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    inputQue_.EnQue(gradInputLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::TransposeB16(
    LocalTensor<T> dst,
    LocalTensor<T> src,
    uint32_t rowNum,
    uint32_t colNum)
{
    uint64_t dstList[TRANS_ADDR_LEN];
    uint64_t srcList[TRANS_ADDR_LEN];
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;

    const uint32_t transPoseAlign = BLOCK_SIZE / sizeof(T);
    if (colNum == transPoseAlign) {
        transDataParams.repeatTimes = rowNum / TRANS_ADDR_LEN;
        transDataParams.dstRepStride = TRANS_ADDR_LEN * sizeof(T) / BLOCK_SIZE;
        transDataParams.srcRepStride = TRANS_ADDR_LEN;

        for (int32_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcList[i] = static_cast<uint64_t>(src[i * transPoseAlign].GetPhyAddr());
            dstList[i] = static_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
        }

        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }

        TransDataTo5HD<T>(dstList, srcList, transDataParams);
    } else {
        transDataParams.repeatTimes = colNum / transPoseAlign;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;

        for (int32_t rowLoopIdx = 0; rowLoopIdx < static_cast<int32_t>(rowNum / TRANS_ADDR_LEN); rowLoopIdx++) {
            for (int32_t i = 0; i < TRANS_ADDR_LEN; i++) {
                srcList[i] = static_cast<uint64_t>(
                    src[rowLoopIdx * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
                dstList[i] = static_cast<uint64_t>(
                    dst[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transDataParams);
        }
    }
}

template <typename T, typename INDEX>
template <typename I>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::TransposeB32(
    LocalTensor<I> dst,
    LocalTensor<I> src,
    uint32_t rowNum,
    uint32_t colNum)
{
    uint64_t dstList[TRANS_ADDR_LEN];
    uint64_t srcList[TRANS_ADDR_LEN];
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;

    const uint32_t transPoseAlign = BLOCK_SIZE / sizeof(I);
    if (colNum == transPoseAlign) {
        transDataParams.repeatTimes = rowNum / TRANS_ADDR_LEN;
        transDataParams.dstRepStride = TRANS_ADDR_LEN * sizeof(I) / BLOCK_SIZE;
        transDataParams.srcRepStride = TRANS_ADDR_LEN;

        for (int32_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcList[i] = static_cast<uint64_t>(src[i * transPoseAlign].GetPhyAddr());
        }
        for (int32_t i = 0; i < TRANS_LEN_B32; i++) {
            dstList[i * 2] = static_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
            dstList[i * 2 + 1] = static_cast<uint64_t>(dst[i * rowNum + transPoseAlign].GetPhyAddr());
        }

        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }

        TransDataTo5HD<I>(dstList, srcList, transDataParams);
    } else {
        transDataParams.repeatTimes = colNum / transPoseAlign;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;

        for (int32_t rowLoopIdx = 0; rowLoopIdx < static_cast<int32_t>(rowNum / TRANS_ADDR_LEN); rowLoopIdx++) {
            for (int32_t i = 0; i < TRANS_ADDR_LEN; i++) {
                srcList[i] = static_cast<uint64_t>(
                    src[rowLoopIdx * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            }
            for (int32_t i = 0; i < TRANS_LEN_B32; i++) {
                dstList[i * 2] = static_cast<uint64_t>(
                    dst[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
                dstList[i * 2 + 1] = static_cast<uint64_t>(
                    dst[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum + transPoseAlign].GetPhyAddr());
            }
            TransDataTo5HD<I>(dstList, srcList, transDataParams);
        }
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::TransInput(
    uint32_t rowNum,
    uint32_t colNum)
{
    LocalTensor<T> srcLocal = inputQue_.DeQue<T>();
    LocalTensor<T> dstLocal = transQue_.AllocTensor<T>();
    Duplicate(dstLocal, (T)0, transQueBufferSize_ / sizeof(T));

    if constexpr (IsSameType<T, float>::value) {
        this->template TransposeB32<T>(dstLocal, srcLocal, rowNum, colNum);
    } else {
        this->TransposeB16(dstLocal, srcLocal, rowNum, colNum);
    }

    PIPE_V_S();
    inputQue_.FreeTensor(srcLocal);
    transQue_.EnQue(dstLocal);
}

// 求一个输入点覆盖当前 tile 的输出区间
template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::CalcOutputRangeFromInputIndex(
    int64_t inputIdxGlobal,
    int64_t outputSize,
    int64_t inputSize,
    int64_t axisTileIndex,
    int64_t axisInner,
    int64_t axisOutputActual,
    int64_t& stLocal,
    int64_t& edLocal,
    int64_t& coverCount) const
{
    const int64_t stGlobal = GetStartFromOutputInputSize(inputIdxGlobal, outputSize, inputSize);
    const int64_t edGlobal = GetEndFromOutputInputSize(inputIdxGlobal, outputSize, inputSize);

    const int64_t tileStart = axisTileIndex * axisInner;
    const int64_t tileEnd = tileStart + axisOutputActual;

    const int64_t stClamped = stGlobal > tileStart ? stGlobal : tileStart;
    const int64_t edClamped = edGlobal < tileEnd ? edGlobal : tileEnd;

    // 转换为局部索引
    stLocal = stClamped - tileStart;
    edLocal = edClamped - tileStart;
    coverCount = edGlobal - stGlobal;
}

// 批量 scatter 回 x_grad
template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::AccumulateOutputRowsForInputPointReg(
    LocalTensor<T> srcLocal,
    LocalTensor<T> dstLocal,
    int64_t inBase,
    T scale,
    int64_t stD,
    int64_t edD,
    int64_t stH,
    int64_t edH,
    int64_t stW,
    int64_t edW)
{
    const uint16_t dLoopCount = static_cast<uint16_t>(edD - stD);
    const uint16_t hLoopCount = static_cast<uint16_t>(edH - stH);
    const uint16_t wLoopCount = static_cast<uint16_t>(edW - stW);
    const int64_t dRowBase0 = stD * hOutputActual_ * wOutputAligned_;

    uint32_t processed = 0;
    while (processed < static_cast<uint32_t>(highAxisActual_)) {
        uint32_t remain = static_cast<uint32_t>(highAxisActual_) - processed;
        uint32_t curCount = remain > vfLen_ ? vfLen_ : remain;

        __local_mem__ T* srcAddr = (__local_mem__ T*)srcLocal[inBase + processed].GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::RegTensor<T> srcReg;
            MicroAPI::RegTensor<T> scaledReg;
            MicroAPI::RegTensor<T> dstReg;
            MicroAPI::MaskReg computeMask = MicroAPI::UpdateMask<uint32_t>(curCount);

            MicroAPI::DataCopy(srcReg, srcAddr);
            MicroAPI::Muls(scaledReg, srcReg, scale, computeMask);

            for (uint16_t od = 0; od < dLoopCount; ++od) {
                const int64_t dRowBase =
                    dRowBase0 + static_cast<int64_t>(od) * hOutputActual_ * wOutputAligned_;
                for (uint16_t oh = 0; oh < hLoopCount; ++oh) {
                    const int64_t hRowBase =
                        dRowBase + static_cast<int64_t>(stH + oh) * wOutputAligned_;
                    for (uint16_t ow = 0; ow < wLoopCount; ++ow) {
                        const int64_t outRow = hRowBase + static_cast<int64_t>(stW + ow);
                        const int64_t outBase = outRow * highAxisAligned_;

                        // gather 同一（d,h,w）位置所有 NC 上的点
                        __local_mem__ T* dstAddr =
                            (__local_mem__ T*)dstLocal[outBase + processed].GetPhyAddr();
                        MicroAPI::DataCopy(dstReg, dstAddr);
                        MicroAPI::Add(dstReg, dstReg, scaledReg, computeMask);
                        MicroAPI::DataCopy(dstAddr, dstReg, computeMask);
                    }
                }
            }

            MicroAPI::LocalMemBar<
                MicroAPI::MemType::VEC_STORE,
                MicroAPI::MemType::VEC_LOAD>();
        }

        processed += curCount;
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::AccumulateOutputRowsForInputPointRegFp32(
    LocalTensor<COMPUTE_TYPE> srcLocal,
    LocalTensor<COMPUTE_TYPE> dstLocal,
    int64_t inBase,
    COMPUTE_TYPE scale,
    int64_t stD,
    int64_t edD,
    int64_t stH,
    int64_t edH,
    int64_t stW,
    int64_t edW)
{
    const uint16_t dLoopCount = static_cast<uint16_t>(edD - stD);
    const uint16_t hLoopCount = static_cast<uint16_t>(edH - stH);
    const uint16_t wLoopCount = static_cast<uint16_t>(edW - stW);
    const int64_t dRowBase0 = stD * hOutputActual_ * wOutputAligned_;

    uint32_t processed = 0;
    while (processed < static_cast<uint32_t>(highAxisActual_)) {
        uint32_t remain = static_cast<uint32_t>(highAxisActual_) - processed;
        uint32_t curCount =
            remain > static_cast<uint32_t>(COMPUTE_VF_LEN) ?
            static_cast<uint32_t>(COMPUTE_VF_LEN) : remain;

        __local_mem__ COMPUTE_TYPE* srcAddr =
            (__local_mem__ COMPUTE_TYPE*)srcLocal[inBase + processed].GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::RegTensor<COMPUTE_TYPE> srcReg;
            MicroAPI::RegTensor<COMPUTE_TYPE> scaledReg;
            MicroAPI::RegTensor<COMPUTE_TYPE> dstReg;
            MicroAPI::MaskReg computeMask = MicroAPI::UpdateMask<uint32_t>(curCount);

            MicroAPI::DataCopy(srcReg, srcAddr);
            MicroAPI::Muls(scaledReg, srcReg, scale, computeMask);

            for (uint16_t od = 0; od < dLoopCount; ++od) {
                const int64_t dRowBase =
                    dRowBase0 + static_cast<int64_t>(od) * hOutputActual_ * wOutputAligned_;
                for (uint16_t oh = 0; oh < hLoopCount; ++oh) {
                    const int64_t hRowBase =
                        dRowBase + static_cast<int64_t>(stH + oh) * wOutputAligned_;
                    for (uint16_t ow = 0; ow < wLoopCount; ++ow) {
                        const int64_t outRow = hRowBase + static_cast<int64_t>(stW + ow);
                        const int64_t outBase = outRow * highAxisAligned_;
                        __local_mem__ COMPUTE_TYPE* dstAddr =
                            (__local_mem__ COMPUTE_TYPE*)dstLocal[outBase + processed].GetPhyAddr();
                        MicroAPI::DataCopy(dstReg, dstAddr);
                        MicroAPI::Add(dstReg, dstReg, scaledReg, computeMask);
                        MicroAPI::DataCopy(dstAddr, dstReg, computeMask);
                    }
                }
            }

            MicroAPI::LocalMemBar<
                MicroAPI::MemType::VEC_STORE,
                MicroAPI::MemType::VEC_LOAD>();
        }

        processed += curCount;
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::Compute()
{
    LocalTensor<T> srcLocal = transQue_.DeQue<T>();
    LocalTensor<T> dstLocal = transOutQue_.AllocTensor<T>();
    Duplicate(dstLocal, (T)0, transOutQueBufferSize_ / sizeof(T));

    for (int64_t sdLocal = 0; sdLocal < dGradInputActual_; ++sdLocal) {
        int64_t stD = 0;
        int64_t edD = 0;
        int64_t coverD = 0;
        CalcOutputRangeFromInputIndex(
            dStLeftCornerIdx_ + sdLocal,
            dOutput_,
            dGradInput_,
            dAxisIndex_,
            dOutputInner_,
            dOutputActual_,
            stD,
            edD,
            coverD);

        // 快速判断当前 y_grad 点在 x_grad 内有没有作出贡献
        if (edD <= stD || coverD <= 0) {
            continue;
        }

        const int64_t dBase = sdLocal * hGradInputActual_ * wGradInputAligned_;

        // 对于每一个 y_grad，计算它会回传到当前 x_grad 中的哪些位置
        for (int64_t shLocal = 0; shLocal < hGradInputActual_; ++shLocal) {
            int64_t stH = 0;
            int64_t edH = 0;
            int64_t coverH = 0;
            CalcOutputRangeFromInputIndex(
                hStLeftCornerIdx_ + shLocal,
                hOutput_,
                hGradInput_,
                hAxisIndex_,
                hOutputInner_,
                hOutputActual_,
                stH,
                edH,
                coverH);

            if (edH <= stH || coverH <= 0) {
                continue;
            }

            const int64_t hBase = dBase + shLocal * wGradInputAligned_;
            const int64_t dhKernel = coverD * coverH;

            for (int64_t swLocal = 0; swLocal < wGradInputActual_; ++swLocal) {
                int64_t stW = 0;
                int64_t edW = 0;
                int64_t coverW = 0;
                CalcOutputRangeFromInputIndex(
                    wStLeftCornerIdx_ + swLocal,
                    wOutput_,
                    wGradInput_,
                    wAxisIndex_,
                    wOutputInner_,
                    wOutputActual_,
                    stW,
                    edW,
                    coverW);

                if (edW <= stW || coverW <= 0) {
                    continue;
                }

                const int64_t kernelSize = dhKernel * coverW;
                if (kernelSize <= 0) {
                    continue;
                }

                const int64_t inBase = (hBase + swLocal) * highAxisAligned_;
                const T scale = static_cast<T>(1.0f / static_cast<float>(kernelSize));
                AccumulateOutputRowsForInputPointReg(
                    srcLocal,
                    dstLocal,
                    inBase,
                    scale,
                    stD,
                    edD,
                    stH,
                    edH,
                    stW,
                    edW);
            }
        }
    }

    PIPE_V_S();
    transQue_.FreeTensor(srcLocal);
    transOutQue_.EnQue(dstLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::ComputeWithFp32Acc()
{
    LocalTensor<T> srcLocalT = transQue_.DeQue<T>();
    LocalTensor<COMPUTE_TYPE> srcLocal = computeSrcBuf_.Get<COMPUTE_TYPE>();
    LocalTensor<COMPUTE_TYPE> dstLocal = computeAccumBuf_.Get<COMPUTE_TYPE>();

    const uint32_t srcElemCount = static_cast<uint32_t>(highAxisAligned_ * inputColNum_);
    const uint32_t dstElemCount = static_cast<uint32_t>(outputRowNumAligned_ * highAxisAligned_);

    Cast(srcLocal, srcLocalT, RoundMode::CAST_NONE, srcElemCount);
    PIPE_V_S();

    Duplicate(dstLocal, static_cast<COMPUTE_TYPE>(0), dstElemCount);

    for (int64_t sdLocal = 0; sdLocal < dGradInputActual_; ++sdLocal) {
        int64_t stD = 0;
        int64_t edD = 0;
        int64_t coverD = 0;
        CalcOutputRangeFromInputIndex(
            dStLeftCornerIdx_ + sdLocal,
            dOutput_,
            dGradInput_,
            dAxisIndex_,
            dOutputInner_,
            dOutputActual_,
            stD,
            edD,
            coverD);

        if (edD <= stD || coverD <= 0) {
            continue;
        }

        const int64_t dBase = sdLocal * hGradInputActual_ * wGradInputAligned_;

        for (int64_t shLocal = 0; shLocal < hGradInputActual_; ++shLocal) {
            int64_t stH = 0;
            int64_t edH = 0;
            int64_t coverH = 0;
            CalcOutputRangeFromInputIndex(
                hStLeftCornerIdx_ + shLocal,
                hOutput_,
                hGradInput_,
                hAxisIndex_,
                hOutputInner_,
                hOutputActual_,
                stH,
                edH,
                coverH);

            if (edH <= stH || coverH <= 0) {
                continue;
            }

            const int64_t hBase = dBase + shLocal * wGradInputAligned_;
            const int64_t dhKernel = coverD * coverH;

            for (int64_t swLocal = 0; swLocal < wGradInputActual_; ++swLocal) {
                int64_t stW = 0;
                int64_t edW = 0;
                int64_t coverW = 0;
                CalcOutputRangeFromInputIndex(
                    wStLeftCornerIdx_ + swLocal,
                    wOutput_,
                    wGradInput_,
                    wAxisIndex_,
                    wOutputInner_,
                    wOutputActual_,
                    stW,
                    edW,
                    coverW);

                if (edW <= stW || coverW <= 0) {
                    continue;
                }

                const int64_t kernelSize = dhKernel * coverW;
                if (kernelSize <= 0) {
                    continue;
                }

                const int64_t inBase = (hBase + swLocal) * highAxisAligned_;
                const COMPUTE_TYPE scale =
                    static_cast<COMPUTE_TYPE>(1.0f / static_cast<float>(kernelSize));
                AccumulateOutputRowsForInputPointRegFp32(
                    srcLocal,
                    dstLocal,
                    inBase,
                    scale,
                    stD,
                    edD,
                    stH,
                    edH,
                    stW,
                    edW);
            }
        }
    }

    PIPE_V_S();

    LocalTensor<T> dstLocalT = transOutQue_.AllocTensor<T>();
    Cast(dstLocalT, dstLocal, RoundMode::CAST_RINT, dstElemCount);
    PIPE_V_S();

    transQue_.FreeTensor(srcLocalT);
    transOutQue_.EnQue(dstLocalT);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::TransOut()
{
    const uint32_t rowNum = static_cast<uint32_t>(outputRowNumAligned_);
    const uint32_t colNum = static_cast<uint32_t>(highAxisAligned_);

    LocalTensor<T> srcLocal = transOutQue_.DeQue<T>();
    LocalTensor<T> dstLocal = transQue_.AllocTensor<T>();
    Duplicate(dstLocal, (T)0, transQueBufferSize_ / sizeof(T));

    if constexpr (IsSameType<T, float>::value) {
        this->template TransposeB32<T>(dstLocal, srcLocal, rowNum, colNum);
    } else {
        this->TransposeB16(dstLocal, srcLocal, rowNum, colNum);
    }

    PIPE_V_S();
    transOutQue_.FreeTensor(srcLocal);
    transQue_.EnQue(dstLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::CopyOut()
{
    LocalTensor<T> yLocal = transQue_.DeQue<T>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = static_cast<uint16_t>(hOutputActual_);
    valueParams.blockLen = static_cast<uint32_t>(wOutputActual_ * sizeof(T));
    valueParams.srcStride = 0;
    valueParams.dstStride = static_cast<uint32_t>((wOutput_ - wOutputActual_) * sizeof(T));
    valueParams.rsv = 0;

    const int64_t highAxisOutputOffset = highAxisIndex_ * highAxisInner_ * outputPlaneSize_;
    const int64_t dAxisOutputOffset = dAxisIndex_ * dOutputInner_ * hOutput_ * wOutput_;
    const int64_t hAxisOutputOffset = hAxisIndex_ * hOutputInner_ * wOutput_;
    const int64_t wAxisOutputOffset = wAxisIndex_ * wOutputInner_;
    const int64_t outputGmOffset =
        highAxisOutputOffset + dAxisOutputOffset + hAxisOutputOffset + wAxisOutputOffset;

    const int64_t dhwInStride = outputRowNumAligned_;

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = dOutputActual_;
    loopModeParams.loop2Size = highAxisActual_;
    loopModeParams.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T);
    loopModeParams.loop2SrcStride = dhwInStride * sizeof(T);
    loopModeParams.loop1DstStride = hOutput_ * wOutput_ * sizeof(T);
    loopModeParams.loop2DstStride = outputPlaneSize_ * sizeof(T);

    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(yGm_[outputGmOffset], yLocal, valueParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    transQue_.FreeTensor(yLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::ProcessPerLoop()
{
    CopyIn();
    TransInput(static_cast<uint32_t>(highAxisAligned_), static_cast<uint32_t>(inputColNum_));

    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        ComputeWithFp32Acc();
    } else {
        Compute();
    }

    TransOut();
    CopyOut();
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool3dGradNCDHWSmallKernel<T, INDEX>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; ++loopNum) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}

}  // namespace AdaptiveAvgPool3dGradOp

#endif  // ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_SMALL_KERNEL_IMPL_H_