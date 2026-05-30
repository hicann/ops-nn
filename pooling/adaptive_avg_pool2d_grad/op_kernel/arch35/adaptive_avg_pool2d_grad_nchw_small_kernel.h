/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS", BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool2d_grad_nchw_small_kernel.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL2D_GRAD_NCHW_SMALL_KERNEL_IMPL_H_
#define ADAPTIVE_AVG_POOL2D_GRAD_NCHW_SMALL_KERNEL_IMPL_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adaptive_avg_pool2d_grad_struct.h"
#include <type_traits>

namespace AdaptiveAvgPool2dGradOp {
using namespace AscendC;
using namespace ops;

constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t TRANS_LEN_B32 = 8;

using COMPUTE_TYPE = float;

template <typename T, typename INDEX>
class AdaptiveAvgPool2dGradNCHWSmallKernel {
public:
    __aicore__ inline AdaptiveAvgPool2dGradNCHWSmallKernel() {}

    __aicore__ inline void Init(
        GM_ADDR gradInput, GM_ADDR y, TPipe& pipeIn,
        const AdaptiveAvgPool2dNCHWGradSmallKernelTilingDataV35& tilingData);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessPerLoop();
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyIn();
    __aicore__ inline void TransInput(uint32_t rowNum, uint32_t colNum);
    __aicore__ inline void Compute();
    __aicore__ inline void TransOut();
    __aicore__ inline void CopyOut();

    __aicore__ inline void TransposeB16(LocalTensor<T> dst, LocalTensor<T> src, uint32_t rowNum, uint32_t colNum);

    template <typename I>
    __aicore__ inline void TransposeB32(LocalTensor<I> dst, LocalTensor<I> src, uint32_t rowNum, uint32_t colNum);

private:
    __aicore__ inline void CalcOutputRangeFromInputIndex(
        int64_t inputIdxGlobal, int64_t outputSize, int64_t inputSize, int64_t axisTileIndex, int64_t axisInner,
        int64_t axisOutputActual, int64_t& stLocal, int64_t& edLocal, int64_t& coverCount) const;

    __aicore__ inline void AccumulateOutputRowsForInputPointRegFp32(
        LocalTensor<COMPUTE_TYPE> srcLocal, LocalTensor<COMPUTE_TYPE> dstLocal, int64_t inBase, COMPUTE_TYPE scale,
        int64_t stH, int64_t edH, int64_t stW, int64_t edW);

    __aicore__ inline void ComputeFp32Core(
        LocalTensor<COMPUTE_TYPE> srcLocal, LocalTensor<COMPUTE_TYPE> dstLocal, uint32_t dstElemCount);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> transQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> transOutQue_;

    TBuf<QuePosition::VECCALC> stWRegBuf_;
    TBuf<QuePosition::VECCALC> edWRegBuf_;
    TBuf<QuePosition::VECCALC> coverWRegBuf_;

    GlobalTensor<T> gradInputGm_;
    GlobalTensor<T> yGm_;

    const AdaptiveAvgPool2dNCHWGradSmallKernelTilingDataV35* tiling_ = nullptr;

    uint32_t blockIdx_ = 0;

    int64_t highAxisActual_ = 1;
    int64_t hOutputActual_ = 1;
    int64_t wOutputActual_ = 1;
    int64_t curCoreProcessNum_ = 1;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t hGradInputActual_ = 0;
    int64_t wGradInputActual_ = 0;

    int64_t gradInputPlaneSize_ = 0;
    int64_t outputPlaneSize_ = 0;

    int64_t highAxisGradInputOffset_ = 0;
    int64_t hAxisGradInputOffset_ = 0;
    int64_t wAxisGradInputOffset_ = 0;

    int64_t hStLeftCornerIdx_ = 0;
    int64_t wStLeftCornerIdx_ = 0;
    int64_t hEndRightCornerIdx_ = 0;
    int64_t wEndRightCornerIdx_ = 0;

    int64_t wGradInputAligned_ = 1;
    int64_t inputColNum_ = 1;
    int64_t wOutputAligned_ = 1;
    int64_t outputRowNum_ = 1;
    int64_t outputRowNumAligned_ = 1;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK = BLOCK_SIZE / sizeof(T);
    constexpr static int64_t COMPUTE_VF_LEN = platform::GetVRegSize() / sizeof(COMPUTE_TYPE);
    constexpr static int64_t INDEX_VF_LEN = platform::GetVRegSize() / sizeof(INDEX);
};

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::Init(
    GM_ADDR gradInput, GM_ADDR y, TPipe& pipeIn, const AdaptiveAvgPool2dNCHWGradSmallKernelTilingDataV35& tilingData)
{
    tiling_ = &tilingData;

    blockIdx_ = GetBlockIdx();
    gradInputPlaneSize_ = tiling_->hInput * tiling_->wInput;
    outputPlaneSize_ = tiling_->hOutput * tiling_->wOutput;

    if (blockIdx_ >= tiling_->usedCoreNum) {
        return;
    }

    curCoreProcessNum_ =
        (blockIdx_ + 1 == tiling_->usedCoreNum) ? tiling_->tailCoreProcessNum : tiling_->normalCoreProcessNum;

    gradInputGm_.SetGlobalBuffer((__gm__ T*)gradInput);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_ = pipeIn;
    pipe_.InitBuffer(inputQue_, BUFFER_NUM, tiling_->inputQueBufferSize);
    pipe_.InitBuffer(transQue_, BUFFER_NUM, tiling_->transQueBufferSize);
    pipe_.InitBuffer(transOutQue_, BUFFER_NUM, tiling_->transOutQueBufferSize);

    pipe_.InitBuffer(stWRegBuf_, platform::GetVRegSize());
    pipe_.InitBuffer(edWRegBuf_, platform::GetVRegSize());
    pipe_.InitBuffer(coverWRegBuf_, platform::GetVRegSize());
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tiling_->normalCoreProcessNum + loopNum;

    highAxisIndex_ = baseBlockIdx / (tiling_->hOutputOuter * tiling_->wOutputOuter);
    highAxisActual_ = (highAxisIndex_ == (tiling_->highAxisOuter - 1)) ? tiling_->highAxisTail : tiling_->highAxisInner;

    int64_t tempTail = baseBlockIdx % (tiling_->hOutputOuter * tiling_->wOutputOuter);
    hAxisIndex_ = tempTail / tiling_->wOutputOuter;
    hOutputActual_ = (hAxisIndex_ == (tiling_->hOutputOuter - 1)) ? tiling_->hOutputTail : tiling_->hOutputInner;

    wAxisIndex_ = tempTail % tiling_->wOutputOuter;
    wOutputActual_ = (wAxisIndex_ == (tiling_->wOutputOuter - 1)) ? tiling_->wOutputTail : tiling_->wOutputInner;

    hStLeftCornerIdx_ =
        GetStartFromOutputInputSize(hAxisIndex_ * tiling_->hOutputInner, tiling_->hInput, tiling_->hOutput);
    wStLeftCornerIdx_ =
        GetStartFromOutputInputSize(wAxisIndex_ * tiling_->wOutputInner, tiling_->wInput, tiling_->wOutput);

    hEndRightCornerIdx_ = GetEndFromOutputInputSize(
        hAxisIndex_ * tiling_->hOutputInner + hOutputActual_ - 1, tiling_->hInput, tiling_->hOutput);
    wEndRightCornerIdx_ = GetEndFromOutputInputSize(
        wAxisIndex_ * tiling_->wOutputInner + wOutputActual_ - 1, tiling_->wInput, tiling_->wOutput);

    hGradInputActual_ = hEndRightCornerIdx_ - hStLeftCornerIdx_;
    wGradInputActual_ = wEndRightCornerIdx_ - wStLeftCornerIdx_;

    wGradInputAligned_ = CeilAlign(wGradInputActual_, static_cast<int64_t>(MAX_DATA_NUM_IN_ONE_BLOCK));

    inputColNum_ = hGradInputActual_ * wGradInputAligned_;

    wOutputAligned_ = CeilAlign(wOutputActual_, static_cast<int64_t>(MAX_DATA_NUM_IN_ONE_BLOCK));

    outputRowNum_ = hOutputActual_ * wOutputAligned_;
    outputRowNumAligned_ = CeilAlign(outputRowNum_, static_cast<int64_t>(TRANS_ADDR_LEN));

    highAxisGradInputOffset_ = highAxisIndex_ * tiling_->highAxisInner * gradInputPlaneSize_;
    hAxisGradInputOffset_ = hStLeftCornerIdx_ * tiling_->wInput;
    wAxisGradInputOffset_ = wStLeftCornerIdx_;
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::CopyIn()
{
    LocalTensor<T> gradInputLocal = inputQue_.AllocTensor<T>();

    const int64_t gradInputGmOffset = highAxisGradInputOffset_ + hAxisGradInputOffset_ + wAxisGradInputOffset_;

    DataCopyExtParams paramsIn = {
        static_cast<uint16_t>(hGradInputActual_), static_cast<uint32_t>(wGradInputActual_ * sizeof(T)),
        static_cast<uint32_t>((tiling_->wInput - wGradInputActual_) * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {true, 0, 0, 0};

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = highAxisActual_;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = gradInputPlaneSize_ * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = hGradInputActual_ * wGradInputAligned_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(gradInputLocal, gradInputGm_[gradInputGmOffset], paramsIn, padParams);
    event_t eventIDMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

    inputQue_.EnQue(gradInputLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::TransposeB16(
    LocalTensor<T> dst, LocalTensor<T> src, uint32_t rowNum, uint32_t colNum)
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

        for (int32_t i = 0; i < static_cast<int32_t>(TRANS_ADDR_LEN); i++) {
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
            for (int32_t i = 0; i < static_cast<int32_t>(TRANS_ADDR_LEN); i++) {
                srcList[i] = static_cast<uint64_t>(src[rowLoopIdx * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
                dstList[i] = static_cast<uint64_t>(dst[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transDataParams);
        }
    }
}

template <typename T, typename INDEX>
template <typename I>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::TransposeB32(
    LocalTensor<I> dst, LocalTensor<I> src, uint32_t rowNum, uint32_t colNum)
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

        for (int32_t i = 0; i < static_cast<int32_t>(TRANS_ADDR_LEN); i++) {
            srcList[i] = reinterpret_cast<uint64_t>(src[i * transPoseAlign].GetPhyAddr());
        }
        for (int32_t i = 0; i < static_cast<int32_t>(TRANS_LEN_B32); i++) {
            dstList[i * 2] = reinterpret_cast<uint64_t>(dst[i * rowNum].GetPhyAddr());
            dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(dst[i * rowNum + transPoseAlign].GetPhyAddr());
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
            for (int32_t i = 0; i < static_cast<int32_t>(TRANS_ADDR_LEN); i++) {
                srcList[i] =
                    reinterpret_cast<uint64_t>(src[rowLoopIdx * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            }
            for (int32_t i = 0; i < static_cast<int32_t>(TRANS_LEN_B32); i++) {
                dstList[i * 2] = reinterpret_cast<uint64_t>(dst[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(
                    dst[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum + transPoseAlign].GetPhyAddr());
            }
            TransDataTo5HD<I>(dstList, srcList, transDataParams);
        }
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::TransInput(uint32_t rowNum, uint32_t colNum)
{
    LocalTensor<T> srcLocal = inputQue_.DeQue<T>();
    LocalTensor<T> dstLocal = transQue_.AllocTensor<T>();

    if constexpr (IsSameType<T, float>::value) {
        this->template TransposeB32<T>(dstLocal, srcLocal, rowNum, colNum);
    } else {
        this->TransposeB16(dstLocal, srcLocal, rowNum, colNum);
    }

    PIPE_V_S();

    inputQue_.FreeTensor(srcLocal);
    transQue_.EnQue(dstLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::CalcOutputRangeFromInputIndex(
    int64_t inputIdxGlobal, int64_t outputSize, int64_t inputSize, int64_t axisTileIndex, int64_t axisInner,
    int64_t axisOutputActual, int64_t& stLocal, int64_t& edLocal, int64_t& coverCount) const
{
    const int64_t stGlobal = GetStartFromOutputInputSize(inputIdxGlobal, outputSize, inputSize);
    const int64_t edGlobal = GetEndFromOutputInputSize(inputIdxGlobal, outputSize, inputSize);

    const int64_t tileStart = axisTileIndex * axisInner;
    const int64_t tileEnd = tileStart + axisOutputActual;

    const int64_t stClamped = stGlobal > tileStart ? stGlobal : tileStart;
    const int64_t edClamped = edGlobal < tileEnd ? edGlobal : tileEnd;

    stLocal = stClamped - tileStart;
    edLocal = edClamped - tileStart;
    coverCount = edGlobal - stGlobal;
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::AccumulateOutputRowsForInputPointRegFp32(
    LocalTensor<COMPUTE_TYPE> srcLocal, LocalTensor<COMPUTE_TYPE> dstLocal, int64_t inBase, COMPUTE_TYPE scale,
    int64_t stH, int64_t edH, int64_t stW, int64_t edW)
{
    const uint16_t hLoopCount = static_cast<uint16_t>(edH - stH);
    const uint16_t wLoopCount = static_cast<uint16_t>(edW - stW);
    const int64_t hRowBase0 = stH * wOutputAligned_;

    uint32_t total = static_cast<uint32_t>(highAxisActual_);
    uint32_t fullLoops = total / static_cast<uint32_t>(COMPUTE_VF_LEN);
    uint32_t tail = total % static_cast<uint32_t>(COMPUTE_VF_LEN);
    uint32_t processed = 0;
    uint32_t fullMaskCount = static_cast<uint32_t>(COMPUTE_VF_LEN);

    for (uint32_t loop = 0; loop < fullLoops; ++loop) {
        __local_mem__ COMPUTE_TYPE* srcAddr = (__local_mem__ COMPUTE_TYPE*)srcLocal[inBase + processed].GetPhyAddr();

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<COMPUTE_TYPE> srcReg;
            MicroAPI::RegTensor<COMPUTE_TYPE> scaledReg;
            MicroAPI::RegTensor<COMPUTE_TYPE> dstReg;
            MicroAPI::MaskReg computeMask = MicroAPI::UpdateMask<COMPUTE_TYPE>(fullMaskCount);

            MicroAPI::DataCopy(srcReg, srcAddr);
            MicroAPI::Muls(scaledReg, srcReg, scale, computeMask);

            for (uint16_t oh = 0; oh < hLoopCount; ++oh) {
                const int64_t hRowBase = hRowBase0 + static_cast<int64_t>(oh) * wOutputAligned_;
                for (uint16_t ow = 0; ow < wLoopCount; ++ow) {
                    const int64_t outRow = hRowBase + static_cast<int64_t>(stW + ow);
                    const int64_t outBase = outRow * tiling_->highAxisInner;
                    __local_mem__ COMPUTE_TYPE* dstAddr =
                        (__local_mem__ COMPUTE_TYPE*)dstLocal[outBase + processed].GetPhyAddr();

                    MicroAPI::DataCopy(dstReg, dstAddr);
                    MicroAPI::Add(dstReg, dstReg, scaledReg, computeMask);
                    MicroAPI::DataCopy(dstAddr, dstReg, computeMask);
                }
            }
        }

        processed += static_cast<uint32_t>(COMPUTE_VF_LEN);
    }

    if (tail > 0) {
        __local_mem__ COMPUTE_TYPE* srcAddr = (__local_mem__ COMPUTE_TYPE*)srcLocal[inBase + processed].GetPhyAddr();

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<COMPUTE_TYPE> srcReg;
            MicroAPI::RegTensor<COMPUTE_TYPE> scaledReg;
            MicroAPI::RegTensor<COMPUTE_TYPE> dstReg;
            MicroAPI::MaskReg computeMask = MicroAPI::UpdateMask<COMPUTE_TYPE>(tail);

            MicroAPI::DataCopy(srcReg, srcAddr);
            MicroAPI::Muls(scaledReg, srcReg, scale, computeMask);

            for (uint16_t oh = 0; oh < hLoopCount; ++oh) {
                const int64_t hRowBase = hRowBase0 + static_cast<int64_t>(oh) * wOutputAligned_;
                for (uint16_t ow = 0; ow < wLoopCount; ++ow) {
                    const int64_t outRow = hRowBase + static_cast<int64_t>(stW + ow);
                    const int64_t outBase = outRow * tiling_->highAxisInner;
                    __local_mem__ COMPUTE_TYPE* dstAddr =
                        (__local_mem__ COMPUTE_TYPE*)dstLocal[outBase + processed].GetPhyAddr();

                    MicroAPI::DataCopy(dstReg, dstAddr);
                    MicroAPI::Add(dstReg, dstReg, scaledReg, computeMask);
                    MicroAPI::DataCopy(dstAddr, dstReg, computeMask);
                }
            }
        }
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::ComputeFp32Core(
    LocalTensor<COMPUTE_TYPE> srcLocal, LocalTensor<COMPUTE_TYPE> dstLocal, uint32_t dstElemCount)
{
    Duplicate(dstLocal, static_cast<COMPUTE_TYPE>(0), dstElemCount);
    __VEC_SCOPE__ { MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>(); }

    LocalTensor<INDEX> stWLocal = stWRegBuf_.Get<INDEX>();
    LocalTensor<INDEX> edWLocal = edWRegBuf_.Get<INDEX>();
    LocalTensor<INDEX> coverWLocal = coverWRegBuf_.Get<INDEX>();

    __local_mem__ INDEX* stWAddr = reinterpret_cast<__local_mem__ INDEX*>(stWLocal.GetPhyAddr());
    __local_mem__ INDEX* edWAddr = reinterpret_cast<__local_mem__ INDEX*>(edWLocal.GetPhyAddr());
    __local_mem__ INDEX* coverWAddr = reinterpret_cast<__local_mem__ INDEX*>(coverWLocal.GetPhyAddr());

    const INDEX wTileStart = static_cast<INDEX>(wAxisIndex_ * tiling_->wOutputInner);
    const INDEX wTileEnd = static_cast<INDEX>(wTileStart + wOutputActual_);
    const INDEX wOutput = static_cast<INDEX>(tiling_->wOutput);
    const INDEX wGradInput = static_cast<INDEX>(tiling_->wInput);

    for (int64_t swLocalBatch = 0; swLocalBatch < wGradInputActual_; swLocalBatch += INDEX_VF_LEN) {
        int64_t curBatchCount = wGradInputActual_ - swLocalBatch;
        curBatchCount = curBatchCount > INDEX_VF_LEN ? INDEX_VF_LEN : curBatchCount;

        const INDEX wBaseIdx = static_cast<INDEX>(wStLeftCornerIdx_ + swLocalBatch);
        uint32_t batchCountMask = static_cast<uint32_t>(curBatchCount);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<INDEX> idx;
            MicroAPI::Arange(idx, wBaseIdx);

            MicroAPI::MaskReg allMask = MicroAPI::CreateMask<INDEX, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg batchMask = MicroAPI::UpdateMask<INDEX>(batchCountMask);

            MicroAPI::RegTensor<INDEX> regConstOutput;
            MicroAPI::Duplicate(regConstOutput, wOutput);

            MicroAPI::RegTensor<INDEX> regConstInput;
            MicroAPI::Duplicate(regConstInput, wGradInput);

            MicroAPI::RegTensor<INDEX> stGlobal;
            MicroAPI::Mul(stGlobal, idx, regConstOutput, allMask);
            MicroAPI::Div(stGlobal, stGlobal, regConstInput, allMask);

            MicroAPI::RegTensor<INDEX> edGlobal;
            MicroAPI::Adds(edGlobal, idx, INDEX(1), allMask);
            MicroAPI::Mul(edGlobal, edGlobal, regConstOutput, allMask);
            MicroAPI::Add(edGlobal, edGlobal, regConstInput, allMask);
            MicroAPI::Adds(edGlobal, edGlobal, INDEX(-1), allMask);
            MicroAPI::Div(edGlobal, edGlobal, regConstInput, allMask);

            MicroAPI::RegTensor<INDEX> cover;
            MicroAPI::Sub(cover, edGlobal, stGlobal, allMask);

            MicroAPI::Maxs(stGlobal, stGlobal, wTileStart, allMask);
            MicroAPI::Adds(stGlobal, stGlobal, INDEX(-wTileStart), allMask);

            MicroAPI::Mins(edGlobal, edGlobal, wTileEnd, allMask);
            MicroAPI::Adds(edGlobal, edGlobal, INDEX(-wTileStart), allMask);

            MicroAPI::DataCopy(stWAddr, stGlobal, batchMask);
            MicroAPI::DataCopy(edWAddr, edGlobal, batchMask);
            MicroAPI::DataCopy(coverWAddr, cover, batchMask);
        }

        PIPE_V_S();

        for (int64_t shLocal = 0; shLocal < hGradInputActual_; ++shLocal) {
            int64_t stH = 0;
            int64_t edH = 0;
            int64_t coverH = 0;
            CalcOutputRangeFromInputIndex(
                hStLeftCornerIdx_ + shLocal, tiling_->hOutput, tiling_->hInput, hAxisIndex_, tiling_->hOutputInner,
                hOutputActual_, stH, edH, coverH);

            if (edH <= stH || coverH <= 0) {
                continue;
            }

            const int64_t hBase = shLocal * wGradInputAligned_;

            for (int64_t wInBatch = 0; wInBatch < curBatchCount; ++wInBatch) {
                const int64_t swLocal = swLocalBatch + wInBatch;
                const int64_t stW = static_cast<int64_t>(stWAddr[wInBatch]);
                const int64_t edW = static_cast<int64_t>(edWAddr[wInBatch]);
                const int64_t coverW = static_cast<int64_t>(coverWAddr[wInBatch]);

                if (edW <= stW || coverW <= 0) {
                    continue;
                }

                const int64_t kernelSize = coverH * coverW;
                if (kernelSize <= 0) {
                    continue;
                }

                const int64_t inBase = (hBase + swLocal) * tiling_->highAxisInner;
                const COMPUTE_TYPE scale = static_cast<COMPUTE_TYPE>(1.0f / static_cast<float>(kernelSize));

                AccumulateOutputRowsForInputPointRegFp32(srcLocal, dstLocal, inBase, scale, stH, edH, stW, edW);
            }
        }
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::Compute()
{
    LocalTensor<T> srcLocalT = transQue_.DeQue<T>();

    const uint32_t srcElemCount = static_cast<uint32_t>(tiling_->highAxisInner * inputColNum_);
    const uint32_t dstElemCount = static_cast<uint32_t>(outputRowNumAligned_ * tiling_->highAxisInner);

    if constexpr (std::is_same_v<T, float>) {
        LocalTensor<T> dstLocalT = transOutQue_.AllocTensor<T>();
        LocalTensor<COMPUTE_TYPE> srcLocal = srcLocalT;
        LocalTensor<COMPUTE_TYPE> dstLocal = dstLocalT;

        ComputeFp32Core(srcLocal, dstLocal, dstElemCount);

        PIPE_V_S();

        transQue_.FreeTensor(srcLocalT);
        transOutQue_.EnQue(dstLocalT);
    } else {
        LocalTensor<COMPUTE_TYPE> srcLocal = srcLocalT.template ReinterpretCast<COMPUTE_TYPE>();
        LocalTensor<COMPUTE_TYPE> dstLocal = transOutQue_.AllocTensor<COMPUTE_TYPE>();

        const uint32_t computeVfLen = static_cast<uint32_t>(COMPUTE_VF_LEN);
        const uint32_t castLoopNum = (srcElemCount + computeVfLen - 1) / computeVfLen;
        for (uint32_t loop = castLoopNum; loop > 0; --loop) {
            const uint32_t castOffset = (loop - 1) * computeVfLen;
            const uint32_t remain = srcElemCount - castOffset;
            const uint32_t curCount = remain > computeVfLen ? computeVfLen : remain;
            Cast(srcLocal[castOffset], srcLocalT[castOffset], RoundMode::CAST_NONE, curCount);
        }
        PIPE_V_S();

        ComputeFp32Core(srcLocal, dstLocal, dstElemCount);

        PIPE_V_S();

        Cast(dstLocal.template ReinterpretCast<T>(), dstLocal, RoundMode::CAST_RINT, dstElemCount);
        PIPE_V_S();

        transQue_.FreeTensor(srcLocalT);
        transOutQue_.EnQue(dstLocal.template ReinterpretCast<T>());
    }
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::TransOut()
{
    const uint32_t rowNum = static_cast<uint32_t>(outputRowNumAligned_);
    const uint32_t colNum = static_cast<uint32_t>(tiling_->highAxisInner);

    LocalTensor<T> srcLocal = transOutQue_.DeQue<T>();
    LocalTensor<T> dstLocal = transQue_.AllocTensor<T>();

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
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::CopyOut()
{
    LocalTensor<T> yLocal = transQue_.DeQue<T>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = static_cast<uint16_t>(hOutputActual_);
    valueParams.blockLen = static_cast<uint32_t>(wOutputActual_ * sizeof(T));
    valueParams.srcStride = 0;
    valueParams.dstStride = static_cast<uint32_t>((tiling_->wOutput - wOutputActual_) * sizeof(T));
    valueParams.rsv = 0;

    const int64_t highAxisOutputOffset = highAxisIndex_ * tiling_->highAxisInner * outputPlaneSize_;
    const int64_t hAxisOutputOffset = hAxisIndex_ * tiling_->hOutputInner * tiling_->wOutput;
    const int64_t wAxisOutputOffset = wAxisIndex_ * tiling_->wOutputInner;
    const int64_t outputGmOffset = highAxisOutputOffset + hAxisOutputOffset + wAxisOutputOffset;

    const int64_t hwInStride = outputRowNumAligned_;

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = highAxisActual_;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = hwInStride * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = outputPlaneSize_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(yGm_[outputGmOffset], yLocal, valueParams);
    event_t eventIDMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMte3ToS);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    transQue_.FreeTensor(yLocal);
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::ProcessPerLoop()
{
    CopyIn();
    TransInput(static_cast<uint32_t>(tiling_->highAxisInner), static_cast<uint32_t>(inputColNum_));
    Compute();
    TransOut();
    CopyOut();
}

template <typename T, typename INDEX>
__aicore__ inline void AdaptiveAvgPool2dGradNCHWSmallKernel<T, INDEX>::Process()
{
    if (blockIdx_ >= tiling_->usedCoreNum) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; ++loopNum) {
        ScalarCompute(loopNum);
        ProcessPerLoop();
    }
}

} // namespace AdaptiveAvgPool2dGradOp

#endif // ADAPTIVE_AVG_POOL2D_GRAD_NCHW_SMALL_KERNEL_IMPL_H_