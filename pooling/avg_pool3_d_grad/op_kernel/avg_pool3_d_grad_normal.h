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
 * \file avg_pool3_d_grad_normal.h
 * \brief
 */

#ifndef AVG_POOL3D_GRAD_NORMAL_H
#define AVG_POOL3D_GRAD_NORMAL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "avg_pool3_d_grad_utils.h"

namespace AvgPool3DGrad {
using namespace AscendC;
using namespace AvgPool3DGradUtils;

template <typename T>
class AvgPool3DGradNormal : public AvgPool3DGradBase<T> {
public:
    __aicore__ inline AvgPool3DGradNormal()
    {}

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR output, const AvgPool3dGradTilingParam& tilingData, GM_ADDR workspace);

    __aicore__ inline void InitUbBuffer(GM_ADDR grads, GM_ADDR output, GM_ADDR workspace);

    __aicore__ inline void CalcIndexW(int64_t indexW);

    __aicore__ inline void Process();

    __aicore__ inline void SubProcess(uint64_t totalIndex);

    __aicore__ inline void CalcBlock(uint64_t doBlockIdx, uint64_t hoBlockIdx, uint64_t woBlockIdx);

    __aicore__ inline void Compute(int64_t ubReadIndex, int64_t ncShape);

    __aicore__ inline void CopyOut(int64_t d, int64_t h, int64_t ncShape);
};

template <typename T>
__aicore__ inline void AvgPool3DGradNormal<T>::Init(
    GM_ADDR grads, GM_ADDR output, const AvgPool3dGradTilingParam& tilingData, GM_ADDR workspace)
{
    this->InitParams(tilingData);
    this->InitUbBuffer(grads, output, workspace);
    this->InitOutputs(output, workspace);
}

template <typename T>
__aicore__ inline void AvgPool3DGradNormal<T>::InitUbBuffer(GM_ADDR grads, GM_ADDR output, GM_ADDR workspace)
{
    this->xGm.SetGlobalBuffer((__gm__ T*)grads);
    this->yGm.SetGlobalBuffer((__gm__ T*)output);
    this->workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);

    auto outputGradUbSize = 0;
    auto inputGradUbSize = 0;
    auto TranUbSize = 0;
    auto castUbSize = 0;
    auto alignNum = 32UL / sizeof(T);
    auto baseDoHoWoAlign = CeilDiv(this->baseDo * this->baseHo * this->baseWo, alignNum) * alignNum;
    auto baseDiHiWiAlign8 = CeilDiv(this->baseDi * this->baseHi * this->baseWi, 8) * 8;
    auto baseDiHiWiAlign16 = CeilDiv(this->baseDi * this->baseHi * this->baseWi, 16) * 16;

    if (std::is_same<T, float>::value) {
        outputGradUbSize = max(baseDoHoWoAlign, baseDiHiWiAlign8) * this->singleCoreNc * sizeof(T);
        inputGradUbSize = baseDiHiWiAlign8 * this->singleCoreNc * sizeof(T);
        TranUbSize = baseDoHoWoAlign * this->singleCoreNc * sizeof(T);
        this->outputUbNum = outputGradUbSize / sizeof(T);
    } else if (!std::is_same<T, float>::value && this->IsOverlap) {
        outputGradUbSize = baseDoHoWoAlign * this->singleCoreNc * sizeof(T);
        TranUbSize = baseDoHoWoAlign * this->singleCoreNc * sizeof(T);
        castUbSize = 2 * this->singleCoreNc * sizeof(float);
        outputGradUbSize = this->max(outputGradUbSize, baseDiHiWiAlign8 * this->singleCoreNc * sizeof(float));
        inputGradUbSize = baseDiHiWiAlign8 * this->singleCoreNc * sizeof(float);
        this->outputUbNum = outputGradUbSize / sizeof(float);
    } else {
        outputGradUbSize = max(baseDoHoWoAlign, baseDiHiWiAlign16) * this->singleCoreNc * sizeof(T);
        TranUbSize = baseDoHoWoAlign * this->singleCoreNc * sizeof(T);
        castUbSize = 2 * this->singleCoreNc * sizeof(float);
        inputGradUbSize = baseDiHiWiAlign16 * this->singleCoreNc * sizeof(T);
        this->outputUbNum = outputGradUbSize / sizeof(T);
    }

    this->pipe.InitBuffer(this->xInBuf, outputGradUbSize);
    this->pipe.InitBuffer(this->yOutBuf, inputGradUbSize);
    this->pipe.InitBuffer(this->TransposeBuf, TranUbSize);
    this->pipe.InitBuffer(this->CastBuf, castUbSize);
    this->outputGradUb = this->xInBuf.template Get<T>();
    this->outputGradUbFp32 = this->outputGradUb.template ReinterpretCast<float>();
    this->TranUb = this->TransposeBuf.template Get<T>();
    this->TranUbFp32 = this->TranUb.template ReinterpretCast<float>();
    this->inputGradUb = this->yOutBuf.template Get<T>();
    this->inputGradUbFp32 = this->inputGradUb.template ReinterpretCast<float>();
    this->castUb = this->CastBuf.template Get<float>();
}

template <typename T>
__aicore__ inline void AvgPool3DGradNormal<T>::CalcIndexW(int64_t indexW)
{
    this->wStartPad = indexW * this->dW - this->padW;
    this->wEndPad = this->min(this->wStartPad + this->kW, this->inW + this->padW);
    this->wStart = this->max(this->wStartPad, 0);
    this->wEnd = this->min(this->wEndPad, this->inW);

    if (this->divisorOverride) {
        this->mulsFactor = 1.0f / static_cast<float>(this->divisorOverride);
    } else if (this->countIncludePad) {
        this->poolSize =
            (this->dEndPad - this->dStartPad) * (this->hEndPad - this->hStartPad) * (this->wEndPad - this->wStartPad);
        this->mulsFactor = 1.0f / static_cast<float>(this->poolSize);
    } else {
        this->kernelSize = (this->dEnd - this->dStart) * (this->hEnd - this->hStart) * (this->wEnd - this->wStart);
        this->mulsFactor = 1.0f / static_cast<float>(this->kernelSize);
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradNormal<T>::Process()
{
    for (uint64_t totalIndex = 0; totalIndex < this->totalCnt; totalIndex++) {
        if (GetBlockIdx() == totalIndex % GetBlockNum()) {
            this->SubProcess(totalIndex);
        }
    }

    if (!std::is_same<T, float>::value && this->IsOverlap) {
        SyncAll();
        this->InitCastUbBuffer();
        this->ProcessCast();
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradNormal<T>::SubProcess(uint64_t totalIndex)
{
    this->ncCntIndex = totalIndex / (this->doCnt * this->hoCnt * this->woCnt);
    this->doCntIndex = totalIndex / (this->hoCnt * this->woCnt) % this->doCnt;
    this->hoCntIndex = totalIndex / (this->woCnt) % this->hoCnt;
    this->woCntIndex = totalIndex % this->woCnt;

    this->ncShape = this->ncCntIndex == this->ncCnt - 1 ? this->ncTail : this->singleCoreNc;
    this->doShape = this->doCntIndex == this->doCnt - 1 ? this->doTail : this->singleCoreDo;
    this->hoShape = this->hoCntIndex == this->hoCnt - 1 ? this->hoTail : this->singleCoreHo;
    this->woShape = this->woCntIndex == this->woCnt - 1 ? this->woTail : this->singleCoreWo;

    this->ncCoreIdx = this->singleCoreNc * this->ncCntIndex;
    this->doCoreIdx = this->doCntIndex * this->singleCoreDo;
    this->hoCoreIdx = this->hoCntIndex * this->singleCoreHo;
    this->woCoreIdx = this->woCntIndex * this->singleCoreWo;

    uint64_t incoreDCnt = CeilDiv(this->doShape, this->baseDo);
    uint64_t incoreHCnt = CeilDiv(this->hoShape, this->baseHo);
    uint64_t incoreWCnt = CeilDiv(this->woShape, this->baseWo);

    auto baseWoTail = this->woShape - this->baseWo * (incoreWCnt - 1);

    for (uint64_t doLoop = 0; doLoop < incoreDCnt; doLoop++) {
        uint64_t doBlockIdx = this->doCoreIdx + doLoop;
        this->CalcIndexD(doBlockIdx);
        for (uint64_t hoLoop = 0; hoLoop < incoreHCnt; hoLoop++) {
            uint64_t hoBlockIdx = this->hoCoreIdx + hoLoop;
            this->CalcIndexH(hoBlockIdx);
            for (uint64_t woLoop = 0; woLoop < incoreWCnt; woLoop++) {
                uint64_t woBlockIdx = this->woCoreIdx + woLoop * this->baseWo;
                this->baseCalcWo = (woLoop == incoreWCnt - 1) ? baseWoTail : this->baseWo;
                CalcBlock(doBlockIdx, hoBlockIdx, woBlockIdx);
            }
        }
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradNormal<T>::CalcBlock(uint64_t doBlockIdx, uint64_t hoBlockIdx, uint64_t woBlockIdx)
{
    auto ubLength = this->baseCalcWo;

    uint64_t gradOffset = this->ncCoreIdx * this->outD * this->outH * this->outW +
                          doBlockIdx * this->outH * this->outW + hoBlockIdx * this->outW + woBlockIdx;

    this->CopyInGrad(gradOffset, this->ncShape, ubLength);
    this->MTE2ToVSync();

    if constexpr (std::is_same<T, float>::value) {
        TransposeBase16M8(this->TranUb, this->outputGradUb, this->singleCoreNc, (ubLength + 8 - 1) / 8 * 8);
    } else {
        TransposeBase16M16(this->TranUb, this->outputGradUb, this->singleCoreNc, (ubLength + 16 - 1) / 16 * 16);
    }
    // 清除outputGradUb f32 f16 + IsOverlap == false
    if (std::is_same<T, float>::value || (!std::is_same<T, float>::value && this->IsOverlap == false)) {
        AscendC::Duplicate<T>(this->outputGradUb, T(0.0), this->outputUbNum);
    } else { // f16 & overlap
        AscendC::Duplicate<float>(this->outputGradUbFp32, float(0.0), this->outputUbNum);
    }

    this->VToSSync();

    // computer

    for (int64_t w = 0; w < this->baseCalcWo; w++) {
        // calcIndexW
        CalcIndexW(woBlockIdx + w);
        if (w == 0) {
            this->blockWStart = this->wStart;
        }
        Compute(w, this->ncShape); // TransUb(ubReadIndex) = TransUb(ubReadIndex) * mulsFactor
    }

    this->VToSSync();

    if (std::is_same<T, float>::value) {
        TransposeBase8M16<T>(
            this->inputGradUb, this->outputGradUb, CeilDiv(this->wEnd - this->blockWStart, 8) * 8, this->singleCoreNc);
    } else if (!std::is_same<T, float>::value && this->IsOverlap == false) {
        TransposeBase16M16<T>(
            this->inputGradUb, this->outputGradUb, CeilDiv(this->wEnd - this->blockWStart, 16) * 16,
            this->singleCoreNc);
    } else {
        TransposeBase8M16<float>(
            this->inputGradUbFp32, this->outputGradUbFp32, CeilDiv(this->wEnd - this->blockWStart, 8) * 8,
            this->singleCoreNc);
    }

    this->VToSSync();

    for (auto iD = this->dStart; iD < this->dEnd; ++iD) {
        for (auto iH = this->hStart; iH < this->hEnd; ++iH) {
            this->CopyOut(iD, iH, this->ncShape);
        }
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradNormal<T>::Compute(int64_t ubReadIndex, int64_t ncShape)
{
    auto offset = ubReadIndex * this->singleCoreNc;

    if constexpr (std::is_same<T, float>::value) {
        Muls(this->TranUb[offset], this->TranUb[offset], T(this->mulsFactor), ncShape);
    } else {
        Cast(this->castUb, this->TranUb[offset], RoundMode::CAST_NONE, ncShape);
        Muls(this->castUb, this->castUb, this->mulsFactor, ncShape);
    }
    // dStart dEnd hStart hEnd Wstart Wend , baseDi, baseHi, baseWi
    // ubWrite
    PipeBarrier<PIPE_ALL>();

    for (int64_t k = this->wStart - this->blockWStart; k < this->wEnd - this->blockWStart; k++) {
        auto ubWriteIndex = k;
        if constexpr (std::is_same<T, float>::value) {
            Add(this->outputGradUb[ubWriteIndex * this->singleCoreNc],
                this->outputGradUb[ubWriteIndex * this->singleCoreNc], this->TranUb[offset], ncShape);
        } else if (!std::is_same<T, float>::value && this->IsOverlap == false) {
            Cast(this->outputGradUb[ubWriteIndex * this->singleCoreNc], this->castUb, RoundMode::CAST_RINT, ncShape);
        } else {
            Add(this->outputGradUbFp32[ubWriteIndex * this->singleCoreNc],
                this->outputGradUbFp32[ubWriteIndex * this->singleCoreNc], this->castUb, ncShape);
        }
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradNormal<T>::CopyOut(int64_t d, int64_t h, int64_t ncShape)
{
    // ubIndex --> 第几列
    auto dstGmOffset = this->ncCoreIdx * (this->inD * this->inH * this->inW) + d * (this->inH * this->inW) +
                       h * this->inW + this->blockWStart;
    auto ubInGradLenght = this->wEnd - this->blockWStart;
    auto ubH = this->singleCoreNc;
    auto ubW = CeilDiv(ubInGradLenght, 8) * 8;
    if (!std::is_same<T, float>::value && this->IsOverlap == false) {
        ubW = CeilDiv(ubInGradLenght, 16) * 16;
    }
    if (this->IsOverlap) {
        SetAtomicAdd<float>();
    }
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = ncShape;
    if (!std::is_same<T, float>::value && this->IsOverlap == true) {
        copyExtParams.blockLen = (ubInGradLenght) * sizeof(float);
        copyExtParams.srcStride = ((ubW - (ubInGradLenght)) * sizeof(float)) / 32;
        copyExtParams.dstStride = (this->inD * this->inH * this->inW - (ubInGradLenght)) * sizeof(float);
        DataCopyPad(this->workspaceGm[dstGmOffset], this->inputGradUbFp32, copyExtParams);
    } else {
        copyExtParams.blockLen = (ubInGradLenght) * sizeof(T);
        copyExtParams.srcStride = ((ubW - (ubInGradLenght)) * sizeof(T)) / 32;
        copyExtParams.dstStride = (this->inD * this->inH * this->inW - (ubInGradLenght)) * sizeof(T);
        DataCopyPad(this->yGm[dstGmOffset], this->inputGradUb, copyExtParams);
    }
    if (this->IsOverlap) {
        SetAtomicNone();
    }
}
} // namespace AvgPool3DGrad
#endif // AVG_POOL3D_GRAD_NORMAL_H