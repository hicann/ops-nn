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
 * \file adaptive_max_pool3d_parall_pool.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_PARALL_POOL_H_
#define ADAPTIVE_MAX_POOL3D_PARALL_POOL_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adaptive_pool3d_tiling_struct.h"

namespace AdaptivePool3d {
using namespace AscendC;
using namespace ops;

constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t TRANS_LEN_B32 = 8;

struct BlockSplitParam {
    int64_t ncIdx;
    int64_t doIdx;
    int64_t hoIdx;
    int64_t woIdx;
    int64_t ncNum;
    int64_t doNum;
    int64_t hoNum;
    int64_t woNum;

    int64_t kerDStartIdx;
    int64_t kerHStartIdx;
    int64_t kerWStartIdx;

    int64_t diDataLen;
    int64_t hiDataLen;
    int64_t wiDataLen;
    int64_t xOffset;
};

template <typename T, typename U>
class AdaptiveMaxPool3dParaPool {
public:
    __aicore__ inline AdaptiveMaxPool3dParaPool(
        const AdaptivePool3DTiling::AdaptivePool3dParaKernelTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices);
    __aicore__ inline void CalInputBlockPara(int64_t curBlockIdx, BlockSplitParam& blockPara);
    __aicore__ inline void CopyInput(
        uint32_t ncNum, uint32_t diDataLen, uint32_t hiDataLen, uint32_t wiDataLen, int64_t xOffset);
    __aicore__ inline void TransposeB16(
        LocalTensor<T> xLocalTrans, LocalTensor<T> xLocal, uint32_t rowNum, uint32_t colNum);
    template <typename I>
    __aicore__ inline void TransposeB32(
        LocalTensor<I> xLocalTrans, LocalTensor<I> xLocal, uint32_t rowNum, uint32_t colNum);
    __aicore__ inline void TransInput(uint32_t ncNum, uint32_t diDataLen, uint32_t hiDataLen, uint32_t wiDataLen);
    __aicore__ inline void CustomSelect(
        LocalTensor<T> inputLocal, LocalTensor<int32_t> inputIndex, LocalTensor<T> outLocal,
        LocalTensor<int32_t> outIndex, uint16_t repeatTimes, uint16_t kernelSize, uint16_t srcStride);
    __aicore__ inline void CustomSelectOnW(
        LocalTensor<T> inputLocal, LocalTensor<T> outLocal, LocalTensor<int32_t> outIndex, uint16_t diDataLen,
        uint16_t hiDataLen, uint16_t kernelSize, uint16_t srcStride, uint32_t indexOfset);
    __aicore__ inline void MaxPoolOnW(
        uint32_t diDataLen, uint32_t hiDataLen, uint32_t wiDataLen, int64_t woIdx, uint32_t woNum);
    __aicore__ inline void MaxPoolOnH(
        uint32_t diDataLen, uint32_t hiDataLen, uint32_t woNum, int64_t hoIdx, uint32_t hoNum);
    __aicore__ inline void MaxPoolOnD(
        uint32_t diDataLen, uint32_t hoNum, uint32_t woNum, int64_t doIdx, uint32_t doNum);
    __aicore__ inline void TransOut(int64_t doNum, int64_t hoNum, int64_t woNum, int64_t indexOffset);
    __aicore__ inline void CopyOut(int64_t ncNum, int64_t doNum, int64_t hoNum, int64_t woNum, int64_t yGmOffset);
    __aicore__ inline void Process();

protected:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQue;
    TQue<QuePosition::VECOUT, 1> maxQue;
    TQue<QuePosition::VECOUT, 1> maxTransQue;
    TQue<QuePosition::VECOUT, 1> maxIndexQue;
    TQue<QuePosition::VECOUT, 1> maxIndexTransQue;

    GlobalTensor<T> xGm, maxGm;
    GlobalTensor<U> indicesGm;

    int64_t inDHW_ = 1;
    int64_t inHW_ = 1;
    int64_t outDHW_ = 1;
    int64_t outHW_ = 1;
    int64_t startBlockIdx_ = 0;
    int64_t endBlockIdx_ = 0;

    uint32_t vlNum_;
    uint32_t ubBlockSize_;
    uint32_t ubAlignNum_;
    BlockSplitParam blockPara_;
    const AdaptivePool3DTiling::AdaptivePool3dParaKernelTilingData tilingData_;
};

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices)
{
    if (GetBlockIdx() >= tilingData_.useCoreNum) {
        return;
    }
    inDHW_ = tilingData_.dIn * tilingData_.hIn * tilingData_.wIn;
    inHW_ = tilingData_.hIn * tilingData_.wIn;
    outDHW_ = tilingData_.dOut * tilingData_.hOut * tilingData_.wOut;
    outHW_ = tilingData_.hOut * tilingData_.wOut;

    vlNum_ = platform::GetVRegSize() / sizeof(T);
    ubBlockSize_ = platform::GetUbBlockSize();
    ubAlignNum_ = ubBlockSize_ / sizeof(T);

    int64_t curHandleBlockNum = tilingData_.blockFactor;
    if (GetBlockIdx() == tilingData_.useCoreNum - 1) {
        curHandleBlockNum = tilingData_.blockTail;
    }
    startBlockIdx_ = GetBlockIdx() * tilingData_.blockFactor;
    endBlockIdx_ = startBlockIdx_ + curHandleBlockNum;

    xGm.SetGlobalBuffer((__gm__ T*)x);
    maxGm.SetGlobalBuffer((__gm__ T*)y);
    indicesGm.SetGlobalBuffer((__gm__ U*)indices);

    pipe_.InitBuffer(inputQue, 1, tilingData_.maxInputSize * sizeof(T));
    pipe_.InitBuffer(maxQue, 1, tilingData_.maxInputSize * sizeof(T));
    pipe_.InitBuffer(maxTransQue, 1, tilingData_.maxInputSize * sizeof(T));
    pipe_.InitBuffer(maxIndexQue, 1, tilingData_.maxInputSize * sizeof(int32_t));
    pipe_.InitBuffer(maxIndexTransQue, 1, tilingData_.maxInputSize * sizeof(U));
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::CalInputBlockPara(
    int64_t curBlockIdx, BlockSplitParam& blockPara)
{
    int64_t dhwOuter = tilingData_.doOuter * tilingData_.hoOuter * tilingData_.woOuter;
    int64_t hwOuter = tilingData_.hoOuter * tilingData_.woOuter;

    blockPara.ncIdx = curBlockIdx / dhwOuter;
    blockPara.ncNum = (blockPara.ncIdx == (tilingData_.ncOuter - 1)) ? tilingData_.ncTail : tilingData_.ncFactor;
    /* ncdhw */
    int64_t blockIdxOnNc = curBlockIdx % dhwOuter;
    blockPara.doIdx = blockIdxOnNc / hwOuter;
    blockPara.doNum = (blockPara.doIdx == (tilingData_.doOuter - 1)) ? tilingData_.doTail : tilingData_.doFactor;

    int64_t blockIdxOnD = blockIdxOnNc % hwOuter;
    blockPara.hoIdx = blockIdxOnD / tilingData_.woOuter;
    blockPara.hoNum = (blockPara.hoIdx == (tilingData_.hoOuter - 1)) ? tilingData_.hoTail : tilingData_.hoFactor;

    int64_t blockIdxOnDH = blockIdxOnD % tilingData_.woOuter;
    blockPara.woIdx = blockIdxOnDH % tilingData_.woOuter;
    blockPara.woNum = (blockPara.woIdx == (tilingData_.woOuter - 1)) ? tilingData_.woTail : tilingData_.woFactor;

    blockPara.kerDStartIdx = ((blockPara.doIdx * tilingData_.doFactor) * tilingData_.dIn) / tilingData_.dOut;
    blockPara.kerHStartIdx = ((blockPara.hoIdx * tilingData_.hoFactor) * tilingData_.hIn) / tilingData_.hOut;
    blockPara.kerWStartIdx = ((blockPara.woIdx * tilingData_.woFactor) * tilingData_.wIn) / tilingData_.wOut;
    int32_t kerDEndIdx =
        Ceil((blockPara.doIdx * tilingData_.doFactor + blockPara.doNum) * tilingData_.dIn, tilingData_.dOut);
    int32_t kerHEndIdx =
        Ceil((blockPara.hoIdx * tilingData_.hoFactor + blockPara.hoNum) * tilingData_.hIn, tilingData_.hOut);
    int32_t kerWEndIdx =
        Ceil((blockPara.woIdx * tilingData_.woFactor + blockPara.woNum) * tilingData_.wIn, tilingData_.wOut);

    blockPara.diDataLen = kerDEndIdx - blockPara.kerDStartIdx;
    blockPara.hiDataLen = kerHEndIdx - blockPara.kerHStartIdx;
    blockPara.wiDataLen = kerWEndIdx - blockPara.kerWStartIdx;
    blockPara.xOffset = blockPara.ncIdx * tilingData_.ncFactor * inDHW_ + blockPara.kerDStartIdx * inHW_ +
                        blockPara.kerHStartIdx * tilingData_.wIn + blockPara.kerWStartIdx;
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::CopyInput(
    uint32_t ncNum, uint32_t diDataLen, uint32_t hiDataLen, uint32_t wiDataLen, int64_t xOffset)
{
    LocalTensor<T> xLocal = inputQue.AllocTensor<T>();

    uint32_t wiDataAlign = ops::CeilAlign(wiDataLen, ubAlignNum_);
    DataCopyExtParams paramsIn = {
        static_cast<uint16_t>(hiDataLen), static_cast<uint32_t>(wiDataLen * sizeof(T)),
        static_cast<uint32_t>((tilingData_.wIn - wiDataLen) * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = diDataLen;
    loopModeParams.loop2Size = ncNum;
    loopModeParams.loop1SrcStride = inHW_ * sizeof(T);
    loopModeParams.loop2SrcStride = inDHW_ * sizeof(T);
    loopModeParams.loop1DstStride = hiDataLen * wiDataAlign * sizeof(T);
    loopModeParams.loop2DstStride = diDataLen * hiDataLen * wiDataAlign * sizeof(T);

    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(xLocal, xGm[xOffset], paramsIn, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue.EnQue(xLocal);
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::TransposeB16(
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
        transDataParams.dstRepStride = TRANS_ADDR_LEN * sizeof(T) / ubBlockSize_;
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
        /* repeatTimes不会为1 */
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

template <typename T, typename U>
template <typename I>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::TransposeB32(
    LocalTensor<I> xLocalTrans, LocalTensor<I> xLocal, uint32_t rowNum, uint32_t colNum)
{
    uint64_t dstList[TRANS_ADDR_LEN];
    uint64_t srcList[TRANS_ADDR_LEN];

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    uint64_t transPoseAlign = ubBlockSize_ / sizeof(I);
    if (colNum == transPoseAlign) {
        /* repeat在行方向，一次处理16*8个b32或者16*16个B16 */
        transDataParams.repeatTimes = rowNum / TRANS_ADDR_LEN;
        /* dstSride大小为16*sizeof(T), srcSride大小为16个dataBlock */
        transDataParams.dstRepStride = TRANS_ADDR_LEN * sizeof(int32_t) / ubBlockSize_;
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
        TransDataTo5HD<I>(dstList, srcList, transDataParams);
    } else {
        /* repeatTimes不会为1 */
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
            TransDataTo5HD<I>(dstList, srcList, transDataParams);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::TransInput(
    uint32_t ncNum, uint32_t diDataLen, uint32_t hiDataLen, uint32_t wiDataLen)
{
    uint32_t wiDataAlign = ops::CeilAlign(wiDataLen, ubAlignNum_);
    LocalTensor<T> xLocal = inputQue.DeQue<T>();
    LocalTensor<T> xLocalTransVL = maxQue.AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        TransposeB32<T>(xLocalTransVL, xLocal, vlNum_, diDataLen * hiDataLen * wiDataAlign);
    } else {
        TransposeB16(xLocalTransVL, xLocal, vlNum_, diDataLen * hiDataLen * wiDataAlign);
    }
    inputQue.FreeTensor(xLocal);
    maxQue.EnQue(xLocalTransVL);
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::CustomSelect(
    LocalTensor<T> inputLocal, LocalTensor<int32_t> inputIndex, LocalTensor<T> outLocal, LocalTensor<int32_t> outIndex,
    uint16_t repeatTimes, uint16_t kernelSize, uint16_t srcStride)
{
    __ubuf__ T* inputAddr = (__ubuf__ T*)inputLocal.GetPhyAddr();
    __ubuf__ int32_t* inputIndexAddr = (__ubuf__ int32_t*)inputIndex.GetPhyAddr();

    __ubuf__ T* outAddr = (__ubuf__ T*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* outIndexAddr = (__ubuf__ int32_t*)outIndex.GetPhyAddr();

    int64_t vfLen = platform::GetVRegSize() / sizeof(T);
    int64_t vfLenInt = platform::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> inputReg;
        MicroAPI::RegTensor<T> maxReg;
        MicroAPI::RegTensor<int32_t> inputIndexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg1;

        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg indexMask = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg cmpMask = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg cmpMaskNan = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg cmpIndexMask = MicroAPI::CreateMask<int32_t>();

        for (uint16_t i = 0; i < repeatTimes; i++) {
            auto srcStrideOfset = i * srcStride * vfLen;
            auto dstStrideOfset = i * vfLen;
            auto srcOfset = inputAddr + srcStrideOfset;
            auto srcIdxOfset = inputIndexAddr + srcStrideOfset;
            auto dstOfset = outAddr + dstStrideOfset;
            auto dstIdxOfset = outIndexAddr + dstStrideOfset;

            MicroAPI::DataCopy(maxReg, srcOfset);
            MicroAPI::DataCopy(maxIndexReg, srcIdxOfset);
            if constexpr (!IsSameType<T, float>::value) {
                MicroAPI::DataCopy(maxIndexReg1, srcIdxOfset + vfLenInt);
            }
            for (uint16_t k = 1; k < kernelSize; k++) {
                MicroAPI::DataCopy(inputReg, srcOfset + k * vfLen);
                MicroAPI::Compare<T, CMPMODE::GT>(cmpMask, inputReg, maxReg, fullMask);
                MicroAPI::Compare<T, CMPMODE::NE>(cmpMaskNan, inputReg, inputReg, fullMask);
                MicroAPI::Or(cmpMask, cmpMask, cmpMaskNan, fullMask);
                AscendC::MicroAPI::Select(maxReg, inputReg, maxReg, cmpMask);

                MicroAPI::DataCopy(inputIndexReg, srcIdxOfset + k * vfLen);
                if constexpr (IsSameType<T, float>::value) {
                    AscendC::MicroAPI::Select(maxIndexReg, inputIndexReg, maxIndexReg, cmpMask);
                } else {
                    AscendC::MicroAPI::UnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(cmpIndexMask, cmpMask);
                    AscendC::MicroAPI::Select(maxIndexReg, inputIndexReg, maxIndexReg, cmpIndexMask);

                    MicroAPI::DataCopy(inputIndexReg, srcIdxOfset + k * vfLen + vfLenInt);
                    AscendC::MicroAPI::UnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(cmpIndexMask, cmpMask);
                    AscendC::MicroAPI::Select(maxIndexReg1, inputIndexReg, maxIndexReg1, cmpIndexMask);
                }
            }

            MicroAPI::DataCopy(dstOfset, maxReg, fullMask);
            MicroAPI::DataCopy(dstIdxOfset, maxIndexReg, indexMask);
            if constexpr (!IsSameType<T, float>::value) {
                MicroAPI::DataCopy(dstIdxOfset + vfLenInt, maxIndexReg1, indexMask);
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::CustomSelectOnW(
    LocalTensor<T> inputLocal, LocalTensor<T> outLocal, LocalTensor<int32_t> outIndex, uint16_t diDataLen,
    uint16_t hiDataLen, uint16_t kernelSize, uint16_t srcStride, uint32_t indexOfset)
{
    __ubuf__ T* inputAddr = (__ubuf__ T*)inputLocal.GetPhyAddr();

    __ubuf__ T* outAddr = (__ubuf__ T*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* outIndexAddr = (__ubuf__ int32_t*)outIndex.GetPhyAddr();

    int64_t vfLen = platform::GetVRegSize() / sizeof(T);
    /* T位宽为B16、index位宽为B32时有差异 */
    int64_t vfLenInt = platform::GetVRegSize() / sizeof(int32_t);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> inputReg;
        MicroAPI::RegTensor<int32_t> inputIndexReg;
        MicroAPI::RegTensor<T> maxReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg;
        MicroAPI::RegTensor<int32_t> maxIndexReg1;

        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg cmpMask = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg cmpMaskNan = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg indexMask = MicroAPI::CreateMask<int32_t>();
        MicroAPI::MaskReg cmpIndexMask = MicroAPI::CreateMask<int32_t>();

        for (uint16_t dIdx = 0; dIdx < diDataLen; dIdx++) {
            auto srcOfsetOnD = inputAddr + dIdx * hiDataLen * srcStride * vfLen;
            auto dstOfsetOnD = outAddr + dIdx * hiDataLen * vfLen;
            auto dstIdxOfsetOnD = outIndexAddr + dIdx * hiDataLen * vfLen;
            auto assistIdxOnD = dIdx * inHW_;

            for (uint16_t hIdx = 0; hIdx < hiDataLen; hIdx++) {
                auto srcOfset = srcOfsetOnD + hIdx * srcStride * vfLen;
                auto dstOfset = dstOfsetOnD + hIdx * vfLen;
                auto dstIdxOfset = dstIdxOfsetOnD + hIdx * vfLen;
                auto assistIdx = assistIdxOnD + hIdx * tilingData_.wIn + indexOfset;

                MicroAPI::DataCopy(maxReg, srcOfset);
                MicroAPI::Duplicate(inputIndexReg, assistIdx);
                MicroAPI::Duplicate(maxIndexReg, assistIdx);
                if constexpr (!IsSameType<T, float>::value) {
                    MicroAPI::Duplicate(maxIndexReg1, assistIdx);
                }
                for (uint16_t k = 1; k < kernelSize; k++) {
                    MicroAPI::DataCopy(inputReg, srcOfset + k * vfLen);
                    MicroAPI::Compare<T, CMPMODE::GT>(cmpMask, inputReg, maxReg, fullMask);
                    MicroAPI::Compare<T, CMPMODE::NE>(cmpMaskNan, inputReg, inputReg, fullMask);
                    MicroAPI::Or(cmpMask, cmpMask, cmpMaskNan, fullMask);
                    AscendC::MicroAPI::Select(maxReg, inputReg, maxReg, cmpMask);

                    MicroAPI::Adds(inputIndexReg, inputIndexReg, static_cast<int32_t>(1), indexMask);
                    if constexpr (IsSameType<T, float>::value) {
                        AscendC::MicroAPI::Select(maxIndexReg, inputIndexReg, maxIndexReg, cmpMask);
                    } else {
                        AscendC::MicroAPI::UnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(cmpIndexMask, cmpMask);
                        AscendC::MicroAPI::Select(maxIndexReg, inputIndexReg, maxIndexReg, cmpIndexMask);

                        AscendC::MicroAPI::UnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(cmpIndexMask, cmpMask);
                        AscendC::MicroAPI::Select(maxIndexReg1, inputIndexReg, maxIndexReg1, cmpIndexMask);
                    }
                }
                MicroAPI::DataCopy(dstIdxOfset, maxIndexReg, indexMask);
                MicroAPI::DataCopy(dstOfset, maxReg, fullMask);
                if constexpr (!IsSameType<T, float>::value) {
                    MicroAPI::DataCopy(dstIdxOfset + vfLenInt, maxIndexReg1, indexMask);
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::MaxPoolOnW(
    uint32_t diDataLen, uint32_t hiDataLen, uint32_t wiDataLen, int64_t woIdx, uint32_t woNum)
{
    LocalTensor<T> xTransLocal = maxQue.DeQue<T>();

    LocalTensor<T> maxOutLocal = maxTransQue.AllocTensor<T>();
    LocalTensor<int32_t> maxIndexOutLocal = maxIndexTransQue.AllocTensor<int32_t>();

    uint32_t wiDataAlign = ops::CeilAlign(wiDataLen, ubAlignNum_);
    int32_t kerWStartIdxGlobal = ((woIdx * tilingData_.woFactor) * tilingData_.wIn) / tilingData_.wOut;
    for (int32_t kernelIdx = 0; kernelIdx < woNum; kernelIdx++) {
        int32_t kerWStartIdx = ((woIdx * tilingData_.woFactor + kernelIdx) * tilingData_.wIn) / tilingData_.wOut;
        int32_t kerWEndIdx = Ceil((woIdx * tilingData_.woFactor + kernelIdx + 1) * tilingData_.wIn, tilingData_.wOut);
        auto kernelSize = kerWEndIdx - kerWStartIdx;

        auto kerWOffset = kerWStartIdx - kerWStartIdxGlobal;
        auto inputOffset = kerWOffset * vlNum_;
        auto maxBufferOffset = kernelIdx * diDataLen * hiDataLen * vlNum_;
        CustomSelectOnW(
            xTransLocal[inputOffset], maxOutLocal[maxBufferOffset], maxIndexOutLocal[maxBufferOffset], diDataLen,
            hiDataLen, kernelSize, wiDataAlign, kerWOffset);
    }
    maxQue.EnQue(xTransLocal);
    maxTransQue.EnQue(maxOutLocal);
    maxIndexTransQue.EnQue(maxIndexOutLocal);
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::MaxPoolOnH(
    uint32_t diDataLen, uint32_t hiDataLen, uint32_t woNum, int64_t hoIdx, uint32_t hoNum)
{
    LocalTensor<T> wTransLocal = maxTransQue.DeQue<T>();
    LocalTensor<int32_t> maxIndexInLocal = maxIndexTransQue.DeQue<int32_t>();

    LocalTensor<T> maxOutLocal = maxQue.DeQue<T>();
    LocalTensor<int32_t> maxIndexOutLocal = maxIndexQue.AllocTensor<int32_t>();

    uint32_t woNumAlign = ops::CeilAlign(woNum, ubAlignNum_);
    auto repeat = woNum * diDataLen;
    int32_t kerHStartIdxGlobal = ((hoIdx * tilingData_.hoFactor) * tilingData_.hIn) / tilingData_.hOut;
    for (int kernelIdx = 0; kernelIdx < hoNum; kernelIdx++) {
        int32_t kerHStartIdx = ((hoIdx * tilingData_.hoFactor + kernelIdx) * tilingData_.hIn) / tilingData_.hOut;
        int32_t kerHEndIdx = Ceil((hoIdx * tilingData_.hoFactor + kernelIdx + 1) * tilingData_.hIn, tilingData_.hOut);
        auto kernelSize = kerHEndIdx - kerHStartIdx;

        auto kerHOffset = kerHStartIdx - kerHStartIdxGlobal;
        auto inputOffset = kerHOffset * vlNum_;
        auto maxBufferOffset = kernelIdx * woNumAlign * diDataLen * vlNum_;
        CustomSelect(
            wTransLocal[inputOffset], maxIndexInLocal[inputOffset], maxOutLocal[maxBufferOffset],
            maxIndexOutLocal[maxBufferOffset], repeat, kernelSize, hiDataLen);
    }
    maxTransQue.EnQue(wTransLocal);
    maxIndexTransQue.EnQue(maxIndexInLocal);
    maxQue.EnQue(maxOutLocal);
    maxIndexQue.EnQue(maxIndexOutLocal);
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::MaxPoolOnD(
    uint32_t diDataLen, uint32_t hoNum, uint32_t woNum, int64_t doIdx, uint32_t doNum)
{
    LocalTensor<T> hTransLocal = maxQue.DeQue<T>();
    LocalTensor<int32_t> maxIndexInLocal = maxIndexQue.DeQue<int32_t>();

    LocalTensor<T> maxOutLocal = maxTransQue.DeQue<T>();
    LocalTensor<int32_t> maxIndexOutLocal = maxIndexTransQue.DeQue<int32_t>();

    uint32_t woNumAlign = ops::CeilAlign(woNum, ubAlignNum_);
    auto repeat = hoNum * woNumAlign;
    int32_t kerDStartIdxGlobal = ((doIdx * tilingData_.doFactor) * tilingData_.dIn) / tilingData_.dOut;
    for (int kernelIdx = 0; kernelIdx < doNum; kernelIdx++) {
        int32_t kerDStartIdx = ((kernelIdx + doIdx * tilingData_.doFactor) * tilingData_.dIn) / tilingData_.dOut;
        int32_t kerDEndIdx = Ceil((kernelIdx + doIdx * tilingData_.doFactor + 1) * tilingData_.dIn, tilingData_.dOut);
        auto kernelSize = kerDEndIdx - kerDStartIdx;

        auto kerDOffset = kerDStartIdx - kerDStartIdxGlobal;
        auto inputOffset = kerDOffset * vlNum_;
        auto maxBufferOffset = kernelIdx * repeat * vlNum_;
        CustomSelect(
            hTransLocal[inputOffset], maxIndexInLocal[inputOffset], maxOutLocal[maxBufferOffset],
            maxIndexOutLocal[maxBufferOffset], repeat, kernelSize, diDataLen);
    }
    maxQue.EnQue(hTransLocal);
    maxIndexQue.EnQue(maxIndexInLocal);

    maxTransQue.EnQue(maxOutLocal);
    maxIndexTransQue.EnQue(maxIndexOutLocal);
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::TransOut(
    int64_t doNum, int64_t hoNum, int64_t woNum, int64_t indexOffset)
{
    int64_t woNumAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubAlignNum_));
    int64_t rowNum = doNum * hoNum * woNumAlign;

    LocalTensor<T> dTransLocal = maxTransQue.DeQue<T>();
    LocalTensor<int32_t> maxIndexInLocal = maxIndexTransQue.DeQue<int32_t>();
    Adds(maxIndexInLocal, maxIndexInLocal, (int32_t)indexOffset, rowNum * vlNum_);

    LocalTensor<T> maxOutLocal = maxQue.DeQue<T>();
    LocalTensor<int32_t> maxIndexOutLocal = maxIndexQue.DeQue<int32_t>();

    /* 5HD transpose时rowNum按16对齐 */
    auto rowNumAlign = ops::CeilAlign(static_cast<uint64_t>(rowNum), TRANS_ADDR_LEN);
    if constexpr (IsSameType<T, float>::value) {
        TransposeB32<T>(maxOutLocal, dTransLocal, rowNumAlign, vlNum_);
    } else {
        TransposeB16(maxOutLocal, dTransLocal, rowNumAlign, vlNum_);
    }
    TransposeB32<int32_t>(maxIndexOutLocal, maxIndexInLocal, rowNumAlign, vlNum_);
    maxTransQue.FreeTensor(dTransLocal);
    maxIndexTransQue.FreeTensor(maxIndexInLocal);
    maxQue.EnQue(maxOutLocal);
    maxIndexQue.EnQue(maxIndexOutLocal);
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::CopyOut(
    int64_t ncNum, int64_t doNum, int64_t hoNum, int64_t woNum, int64_t yGmOffset)
{
    uint32_t woNumAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubAlignNum_));
    uint32_t woNumIntAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubBlockSize_ / sizeof(int32_t)));
    uint32_t woNumTypeUAlign = ops::CeilAlign(woNum, static_cast<int64_t>(ubBlockSize_ / sizeof(U)));

    LocalTensor<T> maxOutLocal = maxQue.DeQue<T>();
    LocalTensor<int32_t> maxIndexOutLocal = maxIndexQue.DeQue<int32_t>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = hoNum;
    valueParams.blockLen = woNum * sizeof(T);
    valueParams.srcStride = 0;
    valueParams.dstStride = (tilingData_.wOut - woNum) * sizeof(T);

    auto dhwInStride = ops::CeilAlign(static_cast<uint64_t>(doNum * hoNum * woNumAlign), TRANS_ADDR_LEN);
    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = doNum;
    loopModeParams.loop2Size = ncNum;
    loopModeParams.loop1SrcStride = hoNum * woNumAlign * sizeof(T);
    loopModeParams.loop2SrcStride = dhwInStride * sizeof(T);
    loopModeParams.loop1DstStride = outHW_ * sizeof(T);
    loopModeParams.loop2DstStride = outDHW_ * sizeof(T);

    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(maxGm[yGmOffset], maxOutLocal, valueParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    loopModeParams.loop1SrcStride = hoNum * woNumAlign * sizeof(U);
    loopModeParams.loop2SrcStride = dhwInStride * sizeof(U);
    loopModeParams.loop1DstStride = outHW_ * sizeof(U);
    loopModeParams.loop2DstStride = outDHW_ * sizeof(U);
    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams indexParams;
    indexParams.blockCount = hoNum;
    indexParams.blockLen = woNum * sizeof(U);
    indexParams.dstStride = (tilingData_.wOut - woNum) * sizeof(U);

    if constexpr (IsSameType<U, int64_t>::value) {
        indexParams.srcStride = (woNumAlign - woNumTypeUAlign) * sizeof(U) / ubBlockSize_;
        LocalTensor<U> indexOutLocal = maxIndexTransQue.AllocTensor<U>();
        Cast(indexOutLocal, maxIndexOutLocal, RoundMode::CAST_NONE, ncNum * dhwInStride);
        maxIndexTransQue.EnQue(indexOutLocal);
        indexOutLocal = maxIndexTransQue.DeQue<U>();
        DataCopyPad(indicesGm[yGmOffset], indexOutLocal, indexParams);
        maxIndexTransQue.FreeTensor(indexOutLocal);
    } else {
        indexParams.srcStride = (woNumAlign - woNumIntAlign) * sizeof(int32_t) / ubBlockSize_;
        DataCopyPad(indicesGm[yGmOffset], maxIndexOutLocal, indexParams);
    }
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    maxQue.FreeTensor(maxOutLocal);
    maxIndexQue.FreeTensor(maxIndexOutLocal);
}

template <typename T, typename U>
__aicore__ inline void AdaptiveMaxPool3dParaPool<T, U>::Process()
{
    if (GetBlockIdx() >= tilingData_.useCoreNum) {
        return;
    }

    for (auto curIdx = startBlockIdx_; curIdx < endBlockIdx_; curIdx++) {
        if (curIdx == startBlockIdx_) {
            CalInputBlockPara(curIdx, blockPara_);
        }
        auto ncIdx = blockPara_.ncIdx;
        auto doIdx = blockPara_.doIdx;
        auto hoIdx = blockPara_.hoIdx;
        auto woIdx = blockPara_.woIdx;
        auto ncNum = blockPara_.ncNum;
        auto doNum = blockPara_.doNum;
        auto hoNum = blockPara_.hoNum;
        auto woNum = blockPara_.woNum;

        auto kerDStartIdx = blockPara_.kerDStartIdx;
        auto kerHStartIdx = blockPara_.kerHStartIdx;
        auto kerWStartIdx = blockPara_.kerWStartIdx;

        auto diDataLen = blockPara_.diDataLen;
        auto hiDataLen = blockPara_.hiDataLen;
        auto wiDataLen = blockPara_.wiDataLen;
        auto xOffset = blockPara_.xOffset;

        if (curIdx == startBlockIdx_) {
            CopyInput(ncNum, diDataLen, hiDataLen, wiDataLen, xOffset);
        }
        TransInput(ncNum, diDataLen, hiDataLen, wiDataLen);

        if (curIdx != endBlockIdx_ - 1) {
            CalInputBlockPara(curIdx + 1, blockPara_);
            CopyInput(
                blockPara_.ncNum, blockPara_.diDataLen, blockPara_.hiDataLen, blockPara_.wiDataLen, blockPara_.xOffset);
        }

        /* outshape: [woNum, hiDataLen, diDataLen, vl] */
        MaxPoolOnW(diDataLen, hiDataLen, wiDataLen, woIdx, woNum);

        /* outshape: [hoNum, woNumAlign, diDataLen, vl] */
        MaxPoolOnH(diDataLen, hiDataLen, woNum, hoIdx, hoNum);

        /* outshape: [doNum, hoNum, woNumAlign, vl] */
        MaxPoolOnD(diDataLen, hoNum, woNum, doIdx, doNum);

        int64_t yOffset = ncIdx * tilingData_.ncFactor * outDHW_ + doIdx * tilingData_.doFactor * outHW_ +
                          hoIdx * tilingData_.hoFactor * tilingData_.wOut + woIdx * tilingData_.woFactor;
        int64_t indexOffset = kerDStartIdx * inHW_ + kerHStartIdx * tilingData_.wIn + kerWStartIdx;

        TransOut(doNum, hoNum, woNum, indexOffset);
        CopyOut(ncNum, doNum, hoNum, woNum, yOffset);
    }
}
} // namespace AdaptivePool3d
#endif // ADAPTIVE_MAX_POOL3D_PARALL_POOL_H_
