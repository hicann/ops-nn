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
 * \file adaptive_avg_pool2d_small_kernel.h
 * \brief
 */

#ifndef ADAPTIVE_AVG_POOL2D_SMALL_KERNEL_H_
#define ADAPTIVE_AVG_POOL2D_SMALL_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "../inc/load_store_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adaptive_avg_pool2d_struct.h"

namespace AdaptivePool2dSmallKernelNamespace {
using namespace AscendC;
using namespace ops;
using namespace AdaptiveAvgPool2dOp;

constexpr static uint64_t TRANS_ADDR_LEN = 16;
constexpr static uint64_t TRANS_LEN_B32 = 8;
constexpr static uint32_t UB_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static uint32_t V_REG_SIZE = platform::GetVRegSize();

struct BlockSplitParam {
    int64_t ncIdx;
    int64_t hoIdx;
    int64_t woIdx;
    int64_t ncNum;
    int64_t hoNum;
    int64_t woNum;

    int64_t hiDataLen;
    int64_t wiDataLen;
    int64_t xOffset;
};

constexpr AscendC::MicroAPI::CastTrait castTraitB642B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitI32F32 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};

template <typename T, typename ID_T>
class AdaptiveAvgPool2dSmallKernel {
public:
    __aicore__ inline AdaptiveAvgPool2dSmallKernel(const AdaptivePool2dSmallKernelTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void CalInputBlockPara(int64_t curBlockIdx, BlockSplitParam& blockPara);
    __aicore__ inline void CopyInput(uint32_t ncNum, uint32_t hiDataLen, uint32_t wiDataLen, int64_t xOffset);
    __aicore__ inline void TransposeB16(
        LocalTensor<T> xLocalTrans, LocalTensor<T> xLocal, uint32_t rowNum, uint32_t colNum);
    template <typename U>
    __aicore__ inline void TransposeB32(
        LocalTensor<U> xLocalTrans, LocalTensor<U> xLocal, uint32_t rowNum, uint32_t colNum);
    __aicore__ inline void TransInput(uint32_t hiDataLen, uint32_t wiDataLen);
    __aicore__ inline void CalKernelSize(
        int64_t kernelIdx, int32_t kernelNum, int64_t dimIn, int64_t dimOut, LocalTensor<int32_t> startIdxLocal,
        LocalTensor<int32_t> kernelSizeLocal);
    template <typename U>
    __aicore__ inline void CustomSum(
        LocalTensor<U> inputLocal, LocalTensor<float> outLocal, uint32_t repeatTimes, uint32_t kernelSize,
        uint32_t srcMidStride);
    __aicore__ inline void PoolOnW(uint32_t hiDataLen, uint32_t wiDataLen, int64_t woIdx, int32_t woNum);
    __aicore__ inline void PoolOnH(uint32_t hiDataLen, uint32_t woNum, int64_t hoIdx, int32_t hoNum);
    __aicore__ inline void CalAvg(int64_t hoNum, int64_t woNum);
    __aicore__ inline void TransOut(int64_t hoNum, int64_t woNum);
    __aicore__ inline void CopyOut(int64_t ncNum, int64_t hoNum, int64_t woNum, int64_t yGmOffset);
    __aicore__ inline void Process();

protected:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inputQue_;
    TQue<QuePosition::VECOUT, 1> resQue1_;
    TQue<QuePosition::VECOUT, 1> resQue2_;
    TBuf<QuePosition::VECCALC> startIdxBuf;
    TBuf<QuePosition::VECCALC> hKerSizeBuf;
    TBuf<QuePosition::VECCALC> wKerSizeBuf;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t inHW_ = 1;
    int64_t outHW_ = 1;
    int64_t startBlockIdx_ = 0;
    int64_t endBlockIdx_ = 0;

    uint32_t vlNum_;
    uint32_t ubAlignNum_;
    BlockSplitParam blockPara_;
    const AdaptivePool2dSmallKernelTilingData* tilingData_;
    using IndexRegType = typename std::conditional<IsSameType<ID_T, int64_t>::value,
                         typename AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo>,
                         typename AscendC::MicroAPI::RegTensor<int32_t>>::type;  
};

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::Init(GM_ADDR x, GM_ADDR y)
{
    if (GetBlockIdx() >= tilingData_->useCoreNum) {
        return;
    }
    inHW_ = tilingData_->hIn * tilingData_->wIn;
    outHW_ = tilingData_->hOut * tilingData_->wOut;

    vlNum_ = V_REG_SIZE / sizeof(T);
    ubAlignNum_ = UB_BLOCK_SIZE / sizeof(T);

    int64_t curHandleBlockNum = tilingData_->blockFactor;
    if (GetBlockIdx() == tilingData_->useCoreNum - 1) {
        curHandleBlockNum = tilingData_->blockTail;
    }
    startBlockIdx_ = GetBlockIdx() * tilingData_->blockFactor;
    endBlockIdx_ = startBlockIdx_ + curHandleBlockNum;

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    uint64_t startBufSize =
        ops::CeilAlign(tilingData_->maxDimOut * sizeof(int32_t), static_cast<uint64_t>(UB_BLOCK_SIZE));
    uint64_t hBufSize =
        ops::CeilAlign(tilingData_->hoFactor * sizeof(int32_t), static_cast<uint64_t>(UB_BLOCK_SIZE));
    uint64_t wBufSize =
        ops::CeilAlign(tilingData_->woFactor * sizeof(int32_t), static_cast<uint64_t>(UB_BLOCK_SIZE));

    pipe_->InitBuffer(startIdxBuf, startBufSize);
    pipe_->InitBuffer(hKerSizeBuf, hBufSize);
    pipe_->InitBuffer(wKerSizeBuf, wBufSize);
    pipe_->InitBuffer(inputQue_, 1, tilingData_->inputQueSize);
    pipe_->InitBuffer(resQue1_, 1, tilingData_->resQue1Size);
    pipe_->InitBuffer(resQue2_, 1, tilingData_->resQue2Size);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::CalInputBlockPara(
    int64_t curBlockIdx, BlockSplitParam& blockPara)
{
    int64_t hwOuter = tilingData_->hoOuter * tilingData_->woOuter;

    blockPara.ncIdx = curBlockIdx / hwOuter;
    blockPara.ncNum = (blockPara.ncIdx == (tilingData_->ncOuter - 1)) ? tilingData_->ncTail : tilingData_->ncFactor;

    int64_t tempTail = curBlockIdx % hwOuter;
    blockPara.hoIdx = tempTail / tilingData_->woOuter;
    blockPara.hoNum = (blockPara.hoIdx == (tilingData_->hoOuter - 1)) ? tilingData_->hoTail : tilingData_->hoFactor;

    blockPara.woIdx = tempTail % tilingData_->woOuter;
    blockPara.woNum = (blockPara.woIdx == (tilingData_->woOuter - 1)) ? tilingData_->woTail : tilingData_->woFactor;

    // out->in
    int64_t kerHStartIdx = ((blockPara.hoIdx * tilingData_->hoFactor) * tilingData_->hIn) / tilingData_->hOut;
    int64_t kerWStartIdx = ((blockPara.woIdx * tilingData_->woFactor) * tilingData_->wIn) / tilingData_->wOut;

    int64_t kerHEndIdx =
        ops::Ceil((blockPara.hoIdx * tilingData_->hoFactor + blockPara.hoNum) * tilingData_->hIn, tilingData_->hOut);
    int64_t kerWEndIdx =
        ops::Ceil((blockPara.woIdx * tilingData_->woFactor + blockPara.woNum) * tilingData_->wIn, tilingData_->wOut);

    blockPara.hiDataLen = kerHEndIdx - kerHStartIdx;
    blockPara.wiDataLen = kerWEndIdx - kerWStartIdx;
    blockPara.xOffset =
        blockPara.ncIdx * tilingData_->ncFactor * inHW_ + kerHStartIdx * tilingData_->wIn + kerWStartIdx;
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::CopyInput(
    uint32_t ncNum, uint32_t hiDataLen, uint32_t wiDataLen, int64_t xOffset)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();

    uint32_t wiDataAlign = ops::CeilAlign(wiDataLen, ubAlignNum_);
    DataCopyExtParams paramsIn = {
        static_cast<uint16_t>(hiDataLen), static_cast<uint32_t>(wiDataLen * sizeof(T)),
        static_cast<uint32_t>((tilingData_->wIn - wiDataLen) * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = inHW_ * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = hiDataLen * wiDataAlign * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(xLocal, xGm_[xOffset], paramsIn, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue_.EnQue(xLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::TransposeB16(
    LocalTensor<T> xLocalTrans, LocalTensor<T> xLocal, uint32_t rowNum, uint32_t colNum)
{
    uint64_t dstList[TRANS_ADDR_LEN];
    uint64_t srcList[TRANS_ADDR_LEN];

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;

    uint64_t transPoseAlign = ubAlignNum_;
    if (colNum == transPoseAlign) {
        /* repeat在行方向，一次处理16*8个b32或者16*16个B16 */
        transDataParams.repeatTimes = rowNum / TRANS_ADDR_LEN;
        /* dstSride16*sizeof(T), srcSride16dataBlock */
        transDataParams.dstRepStride = TRANS_ADDR_LEN * sizeof(T) / UB_BLOCK_SIZE;
        transDataParams.srcRepStride = TRANS_ADDR_LEN;
        for (int32_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcList[i] = static_cast<uint64_t>(xLocal[i * transPoseAlign].GetPhyAddr());
            dstList[i] = static_cast<uint64_t>(xLocalTrans[i * rowNum].GetPhyAddr());
        }
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HD<T>(dstList, srcList, transDataParams);
    } else {
        /* repeatTimes不会为1, colNum为vl或者hw，hw在外面已经对齐ubAlignNum_ */
        transDataParams.repeatTimes = colNum / transPoseAlign;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;
        /* repeat在列方向,一次处理16*8个b32或者16*16个B16， 行方向一次处理16行 */
        for (int32_t rowLoopIdx = 0; rowLoopIdx < rowNum / TRANS_ADDR_LEN; rowLoopIdx++) {
            for (int32_t i = 0; i < TRANS_ADDR_LEN; i++) {
                srcList[i] =
                    static_cast<uint64_t>(xLocal[rowLoopIdx * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
                dstList[i] = static_cast<uint64_t>(xLocalTrans[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
            }
            TransDataTo5HD<T>(dstList, srcList, transDataParams);
        }
    }
}

template <typename T, typename ID_T>
template <typename U>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::TransposeB32(
    LocalTensor<U> xLocalTrans, LocalTensor<U> xLocal, uint32_t rowNum, uint32_t colNum)
{
    uint64_t dstList[TRANS_ADDR_LEN];
    uint64_t srcList[TRANS_ADDR_LEN];

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    uint64_t transPoseAlign = UB_BLOCK_SIZE / sizeof(U);
    if (colNum == transPoseAlign) {
        /* repeat在行方向，一次处理16*8个b32或者16*16个B16 */
        transDataParams.repeatTimes = rowNum / TRANS_ADDR_LEN;
        /* dstSride大小为16*sizeof(U), srcSride大小为16个dataBlock */
        transDataParams.dstRepStride = TRANS_ADDR_LEN * sizeof(U) / UB_BLOCK_SIZE;
        transDataParams.srcRepStride = TRANS_ADDR_LEN;

        for (int32_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcList[i] = static_cast<uint64_t>(xLocal[i * transPoseAlign].GetPhyAddr());
        }
        for (int32_t i = 0; i < TRANS_LEN_B32; i++) {
            dstList[i * 2] = static_cast<uint64_t>(xLocalTrans[i * rowNum].GetPhyAddr());
            dstList[i * 2 + 1] = static_cast<uint64_t>(xLocalTrans[i * rowNum + transPoseAlign].GetPhyAddr());
        }
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        }
        TransDataTo5HD<U>(dstList, srcList, transDataParams);
    } else {
        /* repeatTimes不会为1, colNum为vl或者hw，hw在外面已经对齐ubAlignNum_ */
        transDataParams.repeatTimes = colNum / transPoseAlign;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;
        /* repeat在列方向, 一次处理16*8个b32或者16*16个B16, 行方向一次处理16行 */
        for (int32_t rowLoopIdx = 0; rowLoopIdx < rowNum / TRANS_ADDR_LEN; rowLoopIdx++) {
            for (int32_t i = 0; i < TRANS_ADDR_LEN; i++) {
                srcList[i] =
                    static_cast<uint64_t>(xLocal[rowLoopIdx * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            }
            for (int32_t i = 0; i < TRANS_LEN_B32; i++) {
                dstList[i * 2] =
                    static_cast<uint64_t>(xLocalTrans[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
                dstList[i * 2 + 1] = static_cast<uint64_t>(
                    xLocalTrans[rowLoopIdx * TRANS_ADDR_LEN + i * rowNum + transPoseAlign].GetPhyAddr());
            }
            TransDataTo5HD<U>(dstList, srcList, transDataParams);
        }
    }
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::TransInput(
    uint32_t hiDataLen, uint32_t wiDataLen)
{
    uint32_t wiDataAlign = ops::CeilAlign(wiDataLen, ubAlignNum_);
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    LocalTensor<T> xLocalTransVL = resQue1_.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        TransposeB32<T>(xLocalTransVL, xLocal, vlNum_, hiDataLen * wiDataAlign);
    } else {
        TransposeB16(xLocalTransVL, xLocal, vlNum_, hiDataLen * wiDataAlign);
    }
    inputQue_.FreeTensor(xLocal);
    resQue1_.EnQue(xLocalTransVL);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::CalKernelSize(
    int64_t kernelIdx, int32_t kernelNum, int64_t dimIn, int64_t dimOut, LocalTensor<int32_t> startIdxLocal,
    LocalTensor<int32_t> kernelSizeLocal)
{
    __ubuf__ int32_t* startIdxAddr = (__ubuf__ int32_t*)startIdxLocal.GetPhyAddr();
    __ubuf__ int32_t* kernelSizeAddr = (__ubuf__ int32_t*)kernelSizeLocal.GetPhyAddr();

    uint32_t dataLen = kernelNum;
    uint16_t vfLen = V_REG_SIZE / sizeof(int32_t);
    uint16_t loopSize = ops::CeilDiv(static_cast<uint16_t>(kernelNum), vfLen);
    __VEC_SCOPE__
    {
        IndexRegType startIdxReg;
        IndexRegType endIdxReg;
        IndexRegType kerSizeReg;
        IndexRegType dupReg;
        IndexRegType gblStIdxReg;
        MicroAPI::RegTensor<int32_t> startDstReg;
        MicroAPI::RegTensor<int32_t> kSizeDstReg;
        MicroAPI::MaskReg calMask;

        ID_T globalStartIdx = kernelIdx * dimIn / dimOut;
        MicroAPI::Duplicate(dupReg, static_cast<ID_T>(dimOut));
        MicroAPI::Duplicate(gblStIdxReg, globalStartIdx);
        for (uint16_t i = 0; i < loopSize; i++) {
            if constexpr (IsSameType<ID_T, int64_t>::value) {
                calMask = MicroAPI::UpdateMask<ID_T, MicroAPI::RegTraitNumTwo>(dataLen);
            } else {
                calMask = MicroAPI::UpdateMask<ID_T>(dataLen);
            }
            ID_T startIdx = kernelIdx + i * vfLen;
            MicroAPI::Arange(startIdxReg, startIdx);
            MicroAPI::Adds(endIdxReg, startIdxReg, static_cast<ID_T>(1), calMask);
            MicroAPI::Muls(startIdxReg, startIdxReg, static_cast<ID_T>(dimIn), calMask);
            MicroAPI::Muls(endIdxReg, endIdxReg, static_cast<ID_T>(dimIn), calMask);
            MicroAPI::Adds(endIdxReg, endIdxReg, static_cast<ID_T>(dimOut - 1), calMask);

            MicroAPI::Div(startIdxReg, startIdxReg, dupReg, calMask);
            MicroAPI::Div(endIdxReg, endIdxReg, dupReg, calMask);
            MicroAPI::Sub(kerSizeReg, endIdxReg, startIdxReg, calMask);
            MicroAPI::Sub(startIdxReg, startIdxReg, gblStIdxReg, calMask);

            if constexpr (IsSameType<ID_T, int64_t>::value) {
                startDstReg = (AscendC::MicroAPI::RegTensor<int32_t>&)startIdxReg.reg[0];
                kSizeDstReg = (AscendC::MicroAPI::RegTensor<int32_t>&)kerSizeReg.reg[0];
                MicroAPI::DataCopy(startIdxAddr + i * vfLen, startDstReg, calMask);
                MicroAPI::DataCopy(kernelSizeAddr + i * vfLen, kSizeDstReg, calMask);
            } else {
                MicroAPI::DataCopy(startIdxAddr + i * vfLen, startIdxReg, calMask);
                MicroAPI::DataCopy(kernelSizeAddr + i * vfLen, kerSizeReg, calMask);
            }
            
        }
    }
}

template <typename T, typename ID_T>
template <typename U>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::CustomSum(
    LocalTensor<U> inputLocal, LocalTensor<float> outLocal, uint32_t repeatTimes, uint32_t kernelSize,
    uint32_t srcMidStride)
{
    __ubuf__ U* inputAddr = (__ubuf__ U*)inputLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();

    uint32_t vfLenFp32 = V_REG_SIZE / sizeof(float);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;

        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); i++) {
            uint32_t srcOffset = i * srcMidStride * vlNum_;
            uint32_t dstOffset = i * vlNum_;

            ops::LoadOneTensorForDtypeT<U>(inputAddr, sumReg, preg, srcOffset);
            for (uint16_t k = 1; k < static_cast<uint16_t>(kernelSize); k++) {
                ops::LoadOneTensorForDtypeT<U>(inputAddr, inputReg, preg, srcOffset + k * vlNum_);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            MicroAPI::DataCopy(outAddr + dstOffset, sumReg, preg);

            // fp16/bf16 vlNum_是2倍的vfLenFp32，再来一次累加
            if constexpr (!IsSameType<T, float>::value) {
                ops::LoadOneTensorForDtypeT<U>(inputAddr, sumReg, preg, srcOffset + vfLenFp32);
                for (uint16_t k = 1; k < static_cast<uint16_t>(kernelSize); k++) {
                    ops::LoadOneTensorForDtypeT<U>(inputAddr, inputReg, preg, srcOffset + k * vlNum_ + vfLenFp32);
                    MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                }
                MicroAPI::DataCopy(outAddr + dstOffset + vfLenFp32, sumReg, preg);
            }
        }
    }
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::PoolOnW(
    uint32_t hiDataLen, uint32_t wiDataLen, int64_t woIdx, int32_t woNum)
{
    LocalTensor<T> xTransLocal = resQue1_.DeQue<T>();
    LocalTensor<float> resOutLocal = resQue2_.AllocTensor<float>();
    LocalTensor<int32_t> startIdxLocal = startIdxBuf.Get<int32_t>();
    LocalTensor<int32_t> kernelSizeLocal = wKerSizeBuf.Get<int32_t>();

    uint32_t wiDataAlign = ops::CeilAlign(wiDataLen, ubAlignNum_);
    int64_t kerWStartIdxGlobal = woIdx * tilingData_->woFactor;
    CalKernelSize(kerWStartIdxGlobal, woNum, tilingData_->wIn, tilingData_->wOut, startIdxLocal, kernelSizeLocal);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    // w方向多个窗口
    for (int32_t kernelIdx = 0; kernelIdx < woNum; kernelIdx++) {
        uint32_t kernelSize = kernelSizeLocal(kernelIdx);
        int32_t kerWOffset = startIdxLocal(kernelIdx);
        // [hiDataLen, wiDataLenAlign, vl] w方向偏移
        uint32_t inputOffset = kerWOffset * vlNum_;
        // [woNumAlign, hiDataLen, vl] w第i个窗口偏移
        uint32_t maxBufferOffset = kernelIdx * hiDataLen * vlNum_;
        CustomSum<T>(xTransLocal[inputOffset], resOutLocal[maxBufferOffset], hiDataLen, kernelSize, wiDataAlign);
    }
    resQue1_.EnQue(xTransLocal);
    resQue2_.EnQue(resOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::PoolOnH(
    uint32_t hiDataLen, uint32_t woNum, int64_t hoIdx, int32_t hoNum)
{
    LocalTensor<float> wTransLocal = resQue2_.DeQue<float>();
    LocalTensor<float> resOutLocal = resQue1_.DeQue<float>();
    LocalTensor<int32_t> startIdxLocal = startIdxBuf.Get<int32_t>();
    LocalTensor<int32_t> kernelSizeLocal = hKerSizeBuf.Get<int32_t>();

    uint32_t woNumAlign = ops::CeilAlign(woNum, ubAlignNum_);
    int64_t kerHStartIdxGlobal = hoIdx * tilingData_->hoFactor;
    CalKernelSize(kerHStartIdxGlobal, hoNum, tilingData_->hIn, tilingData_->hOut, startIdxLocal, kernelSizeLocal);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    // h方向多个窗口
    for (int32_t kernelIdx = 0; kernelIdx < hoNum; kernelIdx++) {
        uint32_t kernelSize = kernelSizeLocal(kernelIdx);
        int32_t kerHOffset = startIdxLocal(kernelIdx);
        // [woNumAlign, hiDataLen, vl] h方向偏移
        uint32_t inputOffset = kerHOffset * vlNum_;
        // [hoNum, woNumAlign, vl] h第i个窗口偏移
        uint32_t maxBufferOffset = kernelIdx * woNumAlign * vlNum_;
        CustomSum<float>(wTransLocal[inputOffset], resOutLocal[maxBufferOffset], woNum, kernelSize, hiDataLen);
    }
    resQue2_.EnQue(wTransLocal);
    resQue1_.EnQue(resOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::CalAvg(int64_t hoNum, int64_t woNum)
{
    LocalTensor<float> sumLocal = resQue1_.DeQue<float>();
    LocalTensor<int32_t> hKerSizeLocal = hKerSizeBuf.Get<int32_t>();
    LocalTensor<int32_t> wKerSizeLocal = wKerSizeBuf.Get<int32_t>();

    __ubuf__ float* sumAddr = (__ubuf__ float*)sumLocal.GetPhyAddr();
    __ubuf__ int32_t* hKerSizeAddr = (__ubuf__ int32_t*)hKerSizeLocal.GetPhyAddr();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t woNumAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubAlignNum_));
    uint16_t vfLenFp32 = V_REG_SIZE / sizeof(float);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<float> avgReg;
        MicroAPI::RegTensor<int32_t> divisorReg;
        MicroAPI::RegTensor<float> divisorCastReg;
        MicroAPI::MaskReg calMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        for (uint16_t hIdx = 0; hIdx < static_cast<uint16_t>(hoNum); hIdx++) {
            int32_t hSize = hKerSizeAddr[hIdx];
            int64_t hOffset = hIdx * woNumAlign;
            for (uint16_t wIdx = 0; wIdx < static_cast<uint16_t>(woNum); wIdx++) {
                int32_t wSize = wKerSizeAddr[wIdx];
                int64_t srcOffset = (hOffset + wIdx) * vlNum_;
                int32_t kerSize = hSize * wSize;
                MicroAPI::Duplicate(divisorReg, kerSize);
                MicroAPI::Cast<float, int32_t, castTraitI32F32>(divisorCastReg, divisorReg, calMask);
                MicroAPI::DataCopy(sumReg, sumAddr + srcOffset);
                MicroAPI::Div(avgReg, sumReg, divisorCastReg, calMask);
                MicroAPI::DataCopy(sumAddr + srcOffset, avgReg, calMask);
                // fp16/bf16 vlNum_是2倍的vfLenFp32，再来一次
                if constexpr (!IsSameType<T, float>::value) {
                    srcOffset += vfLenFp32;
                    MicroAPI::DataCopy(sumReg, sumAddr + srcOffset);
                    MicroAPI::Div(avgReg, sumReg, divisorCastReg, calMask);
                    MicroAPI::DataCopy(sumAddr + srcOffset, avgReg, calMask);
                }
            }
        }
    }
    resQue1_.EnQue(sumLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::TransOut(int64_t hoNum, int64_t woNum)
{
    int64_t woNumAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubAlignNum_));
    int64_t rowNum = hoNum * woNumAlign;

    LocalTensor<float> dTransLocal = resQue1_.DeQue<float>();
    LocalTensor<float> resOutLocal = resQue2_.DeQue<float>();

    /* 5HD transpose时rowNum按16对齐 */
    uint64_t rowNumAlign = ops::CeilAlign(static_cast<uint64_t>(rowNum), TRANS_ADDR_LEN);
    TransposeB32<float>(resOutLocal, dTransLocal, rowNumAlign, vlNum_);
    resQue1_.FreeTensor(dTransLocal);
    resQue2_.EnQue(resOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::CopyOut(
    int64_t ncNum, int64_t hoNum, int64_t woNum, int64_t yGmOffset)
{
    uint32_t woNumAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubAlignNum_));
    LocalTensor<float> resOutLocal = resQue2_.DeQue<float>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = hoNum;
    valueParams.blockLen = woNum * sizeof(T);
    valueParams.srcStride = 0;
    valueParams.dstStride = (tilingData_->wOut - woNum) * sizeof(T);

    uint64_t hwInStride = ops::CeilAlign(static_cast<uint64_t>(hoNum * woNumAlign), TRANS_ADDR_LEN);
    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = hwInStride * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = outHW_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(yGm_[yGmOffset], resOutLocal, valueParams);
    } else {
        LocalTensor<T> castOutLocal = resQue1_.AllocTensor<T>();
        if constexpr (IsSameType<T, half>::value) {
            Cast(castOutLocal, resOutLocal, RoundMode::CAST_NONE, ncNum * hwInStride);
        } else {
            Cast(castOutLocal, resOutLocal, RoundMode::CAST_RINT, ncNum * hwInStride);
        }
        resQue1_.EnQue(castOutLocal);
        castOutLocal = resQue1_.DeQue<T>();
        DataCopyPad(yGm_[yGmOffset], castOutLocal, valueParams);
        resQue1_.FreeTensor(castOutLocal);
    }
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    resQue2_.FreeTensor(resOutLocal);
}

template <typename T, typename ID_T>
__aicore__ inline void AdaptiveAvgPool2dSmallKernel<T, ID_T>::Process()
{
    if (GetBlockIdx() >= tilingData_->useCoreNum) {
        return;
    }

    if (startBlockIdx_ >= endBlockIdx_) {
        return;
    }
    // 第一个块先搬运
    CalInputBlockPara(startBlockIdx_, blockPara_);
    /* outshape: [vl, hiDataLen, wiDataLenAlign] */
    CopyInput(blockPara_.ncNum, blockPara_.hiDataLen, blockPara_.wiDataLen, blockPara_.xOffset);
    for (int64_t curIdx = startBlockIdx_; curIdx < endBlockIdx_ - 1; curIdx++) {
        int64_t ncIdx = blockPara_.ncIdx;
        int64_t hoIdx = blockPara_.hoIdx;
        int64_t woIdx = blockPara_.woIdx;
        int64_t ncNum = blockPara_.ncNum;
        int64_t hoNum = blockPara_.hoNum;
        int64_t woNum = blockPara_.woNum;

        int64_t hiDataLen = blockPara_.hiDataLen;
        int64_t wiDataLen = blockPara_.wiDataLen;
        int64_t xOffset = blockPara_.xOffset;

        /* outshape: [hiDataLen, wiDataLenAlign, vl] */
        TransInput(hiDataLen, wiDataLen);

        // 提前搬运下一轮数据
        CalInputBlockPara(curIdx + 1, blockPara_);
        CopyInput(blockPara_.ncNum, blockPara_.hiDataLen, blockPara_.wiDataLen, blockPara_.xOffset);

        /* outshape: [woNumAlign, hiDataLen, vl] */
        PoolOnW(hiDataLen, wiDataLen, woIdx, woNum);

        /* outshape: [hoNum, woNumAlign, vl] */
        PoolOnH(hiDataLen, woNum, hoIdx, hoNum);

        int64_t yOffset = ncIdx * tilingData_->ncFactor * outHW_ + hoIdx * tilingData_->hoFactor * tilingData_->wOut +
                          woIdx * tilingData_->woFactor;
        CalAvg(hoNum, woNum);
        /* outshape: [vl, hoNum, woNumAlign] */
        TransOut(hoNum, woNum);
        CopyOut(ncNum, hoNum, woNum, yOffset);
    }

    // 最后一个块直接处理
    TransInput(blockPara_.hiDataLen, blockPara_.wiDataLen);
    PoolOnW(blockPara_.hiDataLen, blockPara_.wiDataLen, blockPara_.woIdx, blockPara_.woNum);
    PoolOnH(blockPara_.hiDataLen, blockPara_.woNum, blockPara_.hoIdx, blockPara_.hoNum);
    int64_t yOffset = blockPara_.ncIdx * tilingData_->ncFactor * outHW_ +
                      blockPara_.hoIdx * tilingData_->hoFactor * tilingData_->wOut +
                      blockPara_.woIdx * tilingData_->woFactor;
    CalAvg(blockPara_.hoNum, blockPara_.woNum);
    TransOut(blockPara_.hoNum, blockPara_.woNum);
    CopyOut(blockPara_.ncNum, blockPara_.hoNum, blockPara_.woNum, yOffset);
}
} // namespace AdaptivePool2dSmallKernelNamespace
#endif // ADAPTIVE_AVG_POOL2D_SMALL_KERNEL_H_
