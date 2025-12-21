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
 * \file avg_pool3_d_grad_utils.h
 * \brief
 */

#ifndef AVG_POOL3D_GRAD_UTILS_H
#define AVG_POOL3D_GRAD_UTILS_H

namespace AvgPool3DGradUtils {
using namespace AscendC;
constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t BLOCK_NUM_16 = BLOCK_SIZE / sizeof(half);
constexpr uint64_t BLOCK_NUM_32 = BLOCK_SIZE / sizeof(float);

__aicore__ inline int64_t min(int64_t a, int64_t b)
{
    return a <= b ? a : b;
}

__aicore__ inline int64_t max(int64_t a, int64_t b)
{
    return a >= b ? a : b;
}

__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

__aicore__ inline uint64_t FloorDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (uint64_t)(x / y);
}

// only support float/int32_t
// [row, col] -> [col, row]: row:align16, col:align8
template <typename T>
__aicore__ inline void TransposeBase16M8(LocalTensor<T>& dstUb, LocalTensor<T>& srcUb, uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i / 2 * rowNum + i % 2 * BLOCK_NUM_32].GetPhyAddr());
        }
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = 1;
            transDataParams.dstRepStride = rowNum;
        }
        TransDataTo5HD<float>(dstAddrList, srcAddrList, transDataParams);
    }
}

// only support float/int32_t
// [row, col] -> [col, row]: row:align8, col:align16
template <typename T>
__aicore__ inline void TransposeBase8M16(LocalTensor<T>& dstUb, LocalTensor<T>& srcUb, uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = rowNum / BLOCK_NUM_32;
    for (uint64_t r = 0; r < colNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] =
                (uint64_t)(srcUb[r * TRANS_ADDR_LEN + i % BLOCK_NUM_32 * colNum + i / BLOCK_NUM_32 * BLOCK_NUM_32]
                               .GetPhyAddr());
            dstAddrList[i] =
                (uint64_t)(dstUb[r * TRANS_ADDR_LEN * rowNum + (i % 2 * BLOCK_NUM_32 + i / 2) * rowNum].GetPhyAddr());
        }
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = colNum;
            transDataParams.dstRepStride = 1;
        }
        TransDataTo5HD<float>(dstAddrList, srcAddrList, transDataParams);
    }
}

// only support float16/bfloat16
// [row, col] -> [col, row]: row:align16, col:align16
template <typename T>
__aicore__ inline void TransposeBase16M16(
    LocalTensor<T>& dstUb, LocalTensor<T>& srcUb, uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = colNum / BLOCK_NUM_16;
    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
        }
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = 1;
            transDataParams.dstRepStride = rowNum;
        }
        TransDataTo5HD<half>(dstAddrList, srcAddrList, transDataParams);
    }
}

template <typename T>
class AvgPool3DGradBase {
public:
    TPipe pipe;
    uint64_t coreNum, ubSize;
    uint64_t isDetermine;
    uint64_t thisCoreIdx;
    uint64_t ncTotal;
    uint64_t outD, outH, outW;
    uint64_t singleCoreNc, singleCoreDo, singleCoreHo, singleCoreWo;
    uint64_t ncCnt, doCnt, hoCnt, woCnt, totalCnt;
    uint64_t ncTail, doTail, hoTail, woTail;
    uint64_t baseDo, baseHo, baseWo;
    uint64_t baseDi, baseHi, baseWi;
    uint64_t baseCalcDo, baseCalcHo, baseCalcWo;
    int64_t inD, inH, inW;
    int64_t dD, dH, dW;
    int64_t kD, kH, kW;
    int64_t padD, padH, padW;
    int64_t dStart, hStart, wStart;
    int64_t dStartPad, hStartPad, wStartPad;
    int64_t dEnd, hEnd, wEnd;
    int64_t dEndPad, hEndPad, wEndPad;
    int64_t blockDStart, blockHStart, blockWStart;
    int64_t blockDEnd, blockHEnd, blockWEnd;
    int64_t indexD, indexH, indexW, indexNC;
    int64_t kernelSize;
    int64_t poolSize;
    int64_t divisorOverride;
    int64_t ncCntIndex, doCntIndex, hoCntIndex, woCntIndex;
    int64_t ncShape, doShape, hoShape, woShape;
    int64_t ncCoreIdx, doCoreIdx, hoCoreIdx, woCoreIdx;
    bool IsOverlap;
    bool isScatter;
    bool countIncludePad;
    float mulsFactor;
    int64_t outputUbNum, castUbNum, ubW, delW;

    TQue<QuePosition::VECIN, 1> wsQue; // f16 + overlap
    TQue<QuePosition::VECOUT, 1> yQue; // f16 + overlap
    TBuf<TPosition::VECCALC> xInBuf;
    TBuf<QuePosition::VECCALC> yOutBuf;
    TBuf<TPosition::VECCALC> TransposeBuf, CastTransposeBuf;
    TBuf<TPosition::VECCALC> CastBuf, CastTmpBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> workspaceGm;
    LocalTensor<T> inputGradUb, outputGradUb;
    LocalTensor<float> inputGradUbFp32, outputGradUbFp32;
    LocalTensor<T> TranUb;
    LocalTensor<float> TranUbFp32;
    LocalTensor<float> castUb;

    __aicore__ inline AvgPool3DGradBase()
    {}

    __aicore__ inline int64_t min(int64_t a, int64_t b);

    __aicore__ inline int64_t max(int64_t a, int64_t b);

    __aicore__ inline void InitParams(const AvgPool3dGradTilingParam& tilingData);

    __aicore__ inline void InitOutputs(GM_ADDR output, GM_ADDR workspace);

    __aicore__ inline void SToMTE2Sync();

    __aicore__ inline void MTE2ToSSync();

    __aicore__ inline void SToMTE3Sync();

    __aicore__ inline void MTE3ToSSync();

    __aicore__ inline void SToVSync();

    __aicore__ inline void VToSSync();

    __aicore__ inline void MTE3ToVSync();

    __aicore__ inline void VToMTE3Sync();

    __aicore__ inline void MTE3ToMTE2Sync();

    __aicore__ inline void MTE2ToVSync();

    __aicore__ inline void VToMTE2Sync();

    __aicore__ inline void MTE2ToMTE3Sync();

    __aicore__ inline void CalcIndexD(int64_t indexD);

    __aicore__ inline void CalcIndexH(int64_t indexH);

    __aicore__ inline void CalcIndexW(int64_t indexW);

    __aicore__ inline void CopyInGrad(uint64_t gradOffset, uint64_t ncShape, uint64_t ubLength);

    __aicore__ inline void InitCastUbBuffer();

    __aicore__ inline void ProcessCast();

    __aicore__ inline void CopyInWorkspace(uint64_t gmOffset, uint64_t calcNum);

    __aicore__ inline void ComputeCast(uint64_t calcNum);

    __aicore__ inline void CopyOutCast(uint64_t gmOffset, uint64_t calcNum);
};

template <typename T>
__aicore__ inline int64_t AvgPool3DGradBase<T>::min(int64_t a, int64_t b)
{
    return a <= b ? a : b;
}

template <typename T>
__aicore__ inline int64_t AvgPool3DGradBase<T>::max(int64_t a, int64_t b)
{
    return a >= b ? a : b;
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::InitParams(const AvgPool3dGradTilingParam& tilingData)
{
    this->ncTotal = tilingData.cParam.cTotal;
    this->outD = tilingData.attrParam.outD;
    this->outH = tilingData.attrParam.outH;
    this->outW = tilingData.attrParam.outW;
    this->inD = tilingData.attrParam.inD;
    this->inH = tilingData.attrParam.inH;
    this->inW = tilingData.attrParam.inW;
    this->kD = tilingData.attrParam.kD;
    this->kH = tilingData.attrParam.kH;
    this->kW = tilingData.attrParam.kW;
    this->dD = tilingData.attrParam.dD;
    this->dH = tilingData.attrParam.dH;
    this->dW = tilingData.attrParam.dW;
    this->padD = tilingData.attrParam.padD;
    this->padH = tilingData.attrParam.padH;
    this->padW = tilingData.attrParam.padW;
    this->countIncludePad = tilingData.attrParam.countIncludePad;
    this->divisorOverride = tilingData.attrParam.divisorOverride;
    this->IsOverlap = tilingData.attrParam.isOverLap; // automatic add 没加
    this->isDetermine = tilingData.attrParam.isDetermine;
    this->ubSize = tilingData.blockParam.ubSize;
    this->singleCoreNc = tilingData.blockParam.singleCoreNc;
    this->singleCoreDo = tilingData.blockParam.singleCoreDo;
    this->singleCoreHo = tilingData.blockParam.singleCoreHo;
    this->singleCoreWo = tilingData.blockParam.singleCoreWo;
    this->singleCoreWo = tilingData.blockParam.singleCoreWo;
    this->ncCnt = tilingData.blockParam.ncCnt;
    this->doCnt = tilingData.blockParam.doCnt;
    this->hoCnt = tilingData.blockParam.hoCnt;
    this->woCnt = tilingData.blockParam.woCnt;
    this->totalCnt = tilingData.blockParam.totalCnt;
    this->ncTail = tilingData.blockParam.ncTailData;
    this->doTail = tilingData.blockParam.doTailData;
    this->hoTail = tilingData.blockParam.hoTailData;
    this->woTail = tilingData.blockParam.woTailData;
    this->baseDo = tilingData.blockParam.baseDo;
    this->baseHo = tilingData.blockParam.baseHo;
    this->baseWo = tilingData.blockParam.baseWo;
    this->baseDi = tilingData.blockParam.baseDi;
    this->baseHi = tilingData.blockParam.baseHi;
    this->baseWi = tilingData.blockParam.baseWi;
    this->isScatter = tilingData.blockParam.isScatter;
    this->thisCoreIdx = GetBlockIdx();
    this->coreNum = GetBlockNum();
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::InitOutputs(GM_ADDR output, GM_ADDR workspace)
{
    auto yGmCount = this->inD * this->inH * this->inW * this->ncTotal;
    auto formerNum = yGmCount % this->coreNum;
    auto tailNum = this->coreNum - formerNum;
    auto formerCount = (yGmCount + this->coreNum - 1) / this->coreNum;
    auto tailCount = yGmCount / this->coreNum;
    if (this->thisCoreIdx < formerNum) {
        auto offset = this->thisCoreIdx * formerCount;
        if (!std::is_same<T, float>::value && this->IsOverlap) {
            InitOutput(workspaceGm[offset], formerCount, static_cast<float>(0));
        } else {
            InitOutput(this->yGm[offset], formerCount, static_cast<T>(0));
        }
    } else {
        auto offset = formerNum * formerCount + (this->thisCoreIdx - formerNum) * tailCount;
        if (!std::is_same<T, float>::value && this->IsOverlap) {
            InitOutput(workspaceGm[offset], tailCount, static_cast<float>(0));
        } else {
            InitOutput(this->yGm[offset], tailCount, static_cast<T>(0));
        }
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::SToMTE2Sync()
{
    event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::MTE2ToSSync()
{
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::SToMTE3Sync()
{
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::MTE3ToSSync()
{
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::SToVSync()
{
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::VToSSync()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::MTE3ToVSync()
{
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::VToMTE3Sync()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::MTE3ToMTE2Sync()
{
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::MTE2ToVSync()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::VToMTE2Sync()
{
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::MTE2ToMTE3Sync()
{
    event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::CalcIndexD(int64_t indexD)
{
    dStartPad = indexD * dD - padD;
    dEndPad = min(dStartPad + kD, inD + padD);
    dStart = max(dStartPad, 0);
    dEnd = min(dEndPad, inD);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::CalcIndexH(int64_t indexH)
{
    hStartPad = indexH * dH - padH;
    hEndPad = min(hStartPad + kH, inH + padH);
    hStart = max(hStartPad, 0);
    hEnd = min(hEndPad, inH);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::CalcIndexW(int64_t indexW)
{
    wStartPad = indexW * dW - padW;
    wEndPad = min(wStartPad + kW, inW + padW);
    wStart = max(wStartPad, 0);
    wEnd = min(wEndPad, inW);
    delW = wEnd - wStart;

    if (divisorOverride) {
        mulsFactor = (float)1.0 / static_cast<float>(divisorOverride);
    } else if (countIncludePad) {
        poolSize = (dEndPad - dStartPad) * (hEndPad - hStartPad) * (wEndPad - wStartPad);
        mulsFactor = (float)1.0 / static_cast<float>(poolSize);
    } else {
        kernelSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
        mulsFactor = (float)1.0 / static_cast<float>(kernelSize);
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::CopyInGrad(uint64_t gradOffset, uint64_t ncShape, uint64_t ubLength)
{
    DataCopyExtParams copyParamsGrad;
    copyParamsGrad.blockCount = ncShape; //(ncShape, ubLength)
    copyParamsGrad.blockLen = ubLength * sizeof(T);
    copyParamsGrad.srcStride = (this->outD * this->outH * this->outW - ubLength) * sizeof(T);
    copyParamsGrad.dstStride = 0;
    DataCopyPadExtParams<T> padGrad{false, 0, 0, 0};
    DataCopyPad(this->outputGradUb, xGm[gradOffset], copyParamsGrad, padGrad);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::InitCastUbBuffer()
{
    this->pipe.Reset();
    uint64_t maxCalcNum = this->ubSize / (sizeof(half) + sizeof(float));
    this->pipe.InitBuffer(wsQue, 1, maxCalcNum * sizeof(float));
    this->pipe.InitBuffer(yQue, 1, maxCalcNum * sizeof(half));
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::ProcessCast()
{
    uint64_t maxCalcNum = this->ubSize / (sizeof(half) + sizeof(float));
    uint64_t totalLoops = CeilDiv(this->inD * this->inH * this->inW * this->ncTotal, maxCalcNum);
    uint64_t calcTail = (this->inD * this->inH * this->inW * this->ncTotal) - (totalLoops - 1) * maxCalcNum;
    for (uint64_t loopIndex = 0; loopIndex < totalLoops; loopIndex++) {
        if (GetBlockIdx() == loopIndex % GetBlockNum()) {
            uint64_t calcNum = (loopIndex == totalLoops - 1) ? calcTail : maxCalcNum;
            CopyInWorkspace(loopIndex * maxCalcNum, calcNum);
            ComputeCast(calcNum);
            CopyOutCast(loopIndex * maxCalcNum, calcNum);
        }
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::CopyInWorkspace(uint64_t gmOffset, uint64_t calcNum)
{
    LocalTensor<float> fp32Ub = wsQue.AllocTensor<float>();

    DataCopyExtParams copyParamsWs;
    copyParamsWs.blockCount = 1;
    copyParamsWs.blockLen = calcNum * sizeof(float);
    copyParamsWs.srcStride = 0;
    copyParamsWs.dstStride = 0;
    DataCopyPadExtParams<float> padWs{false, 0, 0, 0};
    DataCopyPad(fp32Ub, workspaceGm[gmOffset], copyParamsWs, padWs);

    wsQue.EnQue(fp32Ub);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::ComputeCast(uint64_t calcNum)
{
    LocalTensor<float> fp32Ub = wsQue.DeQue<float>();
    LocalTensor<T> b16Ub = yQue.AllocTensor<T>();
    if constexpr (std::is_same<T, half>::value) {
        Cast(b16Ub, fp32Ub, RoundMode::CAST_NONE, calcNum);
    } else if (std::is_same<T, bfloat16_t>::value) {
        Cast(b16Ub, fp32Ub, RoundMode::CAST_RINT, calcNum);
    }
    wsQue.FreeTensor(fp32Ub);
    yQue.EnQue(b16Ub);
}

template <typename T>
__aicore__ inline void AvgPool3DGradBase<T>::CopyOutCast(uint64_t gmOffset, uint64_t calcNum)
{
    LocalTensor<T> yUb = yQue.DeQue<T>();
    DataCopyExtParams copyParamsY;
    copyParamsY.blockCount = 1;
    copyParamsY.blockLen = calcNum * sizeof(T);
    copyParamsY.srcStride = 0;
    copyParamsY.dstStride = 0;
    DataCopyPad(yGm[gmOffset], yUb, copyParamsY);
    yQue.FreeTensor(yUb);
}
} // namespace AvgPool3DGradUtils
#endif // AVG_POOL3D_GRAD_UTILS_H