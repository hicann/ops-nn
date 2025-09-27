/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_scatter.h
 * \brief
 */

#ifndef AVG_POOL3D_GRAD_SCATTER_H
#define AVG_POOL3D_GRAD_SCATTER_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "avg_pool3_d_grad_utils.h"

namespace AvgPool3DGrad {
using namespace AscendC;
using namespace AvgPool3DGradUtils;

template <typename T>
class AvgPool3DGradScatter : public AvgPool3DGradBase<T> {
public:
    __aicore__ inline AvgPool3DGradScatter()
    {}

    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR output, const AvgPool3dGradTilingParam& tilingData, GM_ADDR workspace);

    __aicore__ inline void InitUbBuffer(GM_ADDR grads, GM_ADDR output, GM_ADDR workspace);

    __aicore__ inline void Process();

    __aicore__ inline void SubProcess(uint64_t totalIndex);

    __aicore__ inline void CalcBlock(uint64_t doBlockIdx, uint64_t hoBlockIdx, uint64_t woBlockIdx);

    __aicore__ inline void Compute(int64_t ncShape);

    __aicore__ inline void CopyOut(int64_t d, int64_t h, int64_t w, int64_t copyOutLen, int64_t ncShape);
};

template <typename T>
__aicore__ inline void AvgPool3DGradScatter<T>::Init(
    GM_ADDR grads, GM_ADDR output, const AvgPool3dGradTilingParam& tilingData, GM_ADDR workspace)
{
    this->InitParams(tilingData);
    this->InitUbBuffer(grads, output, workspace);
    this->InitOutputs(output, workspace);
}

template <typename T>
__aicore__ inline void AvgPool3DGradScatter<T>::InitUbBuffer(GM_ADDR grads, GM_ADDR output, GM_ADDR workspace)
{
    this->xGm.SetGlobalBuffer((__gm__ T*)grads);
    this->yGm.SetGlobalBuffer((__gm__ T*)output);
    this->workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);

    auto outputGradUbSize = 0;
    auto inputGradUbSize = 0;
    auto TranUbSize = 0;
    auto castUbSize = 0;

    if (std::is_same<T, float>::value) {
        outputGradUbSize = 8 * this->singleCoreNc * sizeof(T);
        inputGradUbSize = 8 * this->singleCoreNc * sizeof(T);
    } else if (!std::is_same<T, float>::value && this->IsOverlap) {
        outputGradUbSize = 16 * this->singleCoreNc * sizeof(T);
        castUbSize = 16 * this->singleCoreNc * sizeof(float);
        inputGradUbSize = 16 * this->singleCoreNc * sizeof(float);
    } else {
        outputGradUbSize = 16 * this->singleCoreNc * sizeof(T);
        castUbSize = 16 * this->singleCoreNc * sizeof(float);
        inputGradUbSize = 16 * this->singleCoreNc * sizeof(T);
    }

    this->pipe.InitBuffer(this->xInBuf, outputGradUbSize);
    this->pipe.InitBuffer(this->yOutBuf, inputGradUbSize);
    this->pipe.InitBuffer(this->CastBuf, castUbSize);

    this->outputGradUb = this->xInBuf.template Get<T>();
    this->outputGradUbFp32 = this->outputGradUb.template ReinterpretCast<float>();

    this->inputGradUb = this->yOutBuf.template Get<T>();
    this->inputGradUbFp32 = this->inputGradUb.template ReinterpretCast<float>();

    this->castUb = this->CastBuf.template Get<float>();
}

template <typename T>
__aicore__ inline void AvgPool3DGradScatter<T>::Process()
{
    for (uint64_t totalIndex = 0; totalIndex < this->totalCnt; totalIndex++) {
        if (GetBlockIdx() == totalIndex % GetBlockNum()) {
            SubProcess(totalIndex);
        }
    }

    if (!std::is_same<T, float>::value && this->IsOverlap) {
        this->InitCastUbBuffer();
        SyncAll();
        this->ProcessCast();
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradScatter<T>::SubProcess(uint64_t totalIndex)
{
    this->ncCntIndex = totalIndex / (this->doCnt * this->hoCnt * this->woCnt);
    this->doCntIndex = totalIndex / (this->hoCnt * this->woCnt) % this->doCnt;
    this->hoCntIndex = totalIndex / (this->woCnt) % this->hoCnt;
    this->woCntIndex = totalIndex % this->woCnt;

    this->ncShape = this->ncCntIndex == this->ncCnt - 1 ? this->ncTail : this->singleCoreNc;
    this->doShape = this->doCntIndex == this->doCnt - 1 ? this->doTail : this->singleCoreDo;
    this->hoShape = this->hoCntIndex == this->hoCnt - 1 ? this->hoTail : this->singleCoreHo;
    this->woShape = this->woCntIndex == this->woCnt - 1 ? this->woTail : this->singleCoreWo;

    this->ncCoreIdx = this->ncCntIndex * this->singleCoreNc;
    this->doCoreIdx = this->doCntIndex * this->singleCoreDo;
    this->hoCoreIdx = this->hoCntIndex * this->singleCoreHo;
    this->woCoreIdx = this->woCntIndex * this->singleCoreWo;

    uint64_t incoreDCnt = CeilDiv(this->doShape, this->baseDo);
    uint64_t incoreHCnt = CeilDiv(this->hoShape, this->baseHo);
    uint64_t incoreWCnt = CeilDiv(this->woShape, this->baseWo);

    auto baseWoTail = this->woShape - this->baseWo * (incoreWCnt - 1);

    for (uint64_t doLoop = 0; doLoop < this->doShape; doLoop++) {
        uint64_t doBlockIdx = this->doCoreIdx + doLoop;
        this->CalcIndexD(doBlockIdx);
        for (uint64_t hoLoop = 0; hoLoop < this->hoShape; hoLoop++) {
            uint64_t hoBlockIdx = this->hoCoreIdx + hoLoop;
            this->CalcIndexH(hoBlockIdx);
            for (uint64_t woLoop = 0; woLoop < this->woShape; woLoop++) {
                uint64_t woBlockIdx = this->woCoreIdx + woLoop;
                this->CalcIndexW(woBlockIdx);
                CalcBlock(doBlockIdx, hoBlockIdx, woBlockIdx);
            }
        }
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradScatter<T>::CalcBlock(uint64_t doBlockIdx, uint64_t hoBlockIdx, uint64_t woBlockIdx)
{
    auto ubLength = 1;

    uint64_t gradOffset = this->ncCoreIdx * this->outD * this->outH * this->outW +
                          doBlockIdx * this->outH * this->outW + hoBlockIdx * this->outW + woBlockIdx;

    this->CopyInGrad(gradOffset, this->ncShape, ubLength);
    PipeBarrier<PIPE_ALL>();
    this->Compute(this->ncShape);
    if (!std::is_same<T, float>::value) {
        this->ubW = 16;
    } else {
        this->ubW = 8;
    }
    uint64_t ubWCnt = CeilDiv(this->delW, this->ubW);
    this->VToMTE3Sync();
    for (auto iD = this->dStart; iD < this->dEnd; ++iD) {
        for (auto iH = this->hStart; iH < this->hEnd; ++iH) {
            for (auto iW = 0; iW < ubWCnt; iW++) {
                uint64_t copyOutLen = iW == ubWCnt - 1 ? (this->delW - iW * this->ubW) : this->ubW;
                this->CopyOut(iD, iH, this->wStart + iW * this->ubW, copyOutLen, this->ncShape);
            }
        }
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradScatter<T>::Compute(int64_t ncShape)
{
    if constexpr (std::is_same_v<T, float>) {
        Muls(this->inputGradUb, this->outputGradUb, T(this->mulsFactor), ncShape * 8); // f32 * 8
    } else if (!std::is_same<T, float>::value && this->IsOverlap == true) {
        Cast(this->inputGradUbFp32, this->outputGradUb, RoundMode::CAST_NONE, ncShape * 16);
        Muls(this->inputGradUbFp32, this->inputGradUbFp32, this->mulsFactor, ncShape * 16);
    } else {
        Cast(this->castUb, this->outputGradUb, RoundMode::CAST_NONE, ncShape * 16);
        Muls(this->castUb, this->castUb, this->mulsFactor, ncShape * 16);
        Cast(this->inputGradUb, this->castUb, RoundMode::CAST_RINT, ncShape * 16);
    }
}

template <typename T>
__aicore__ inline void AvgPool3DGradScatter<T>::CopyOut(
    int64_t d, int64_t h, int64_t w, int64_t copyOutLen, int64_t ncShape)
{
    // ubIndex --> 第几列
    auto dstGmOffset =
        this->ncCoreIdx * (this->inD * this->inH * this->inW) + d * (this->inH * this->inW) + h * this->inW + w;
    if (this->IsOverlap) {
        SetAtomicAdd<float>();
    }
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = ncShape;
    if (!std::is_same<T, float>::value && this->IsOverlap) {
        copyExtParams.blockLen = (copyOutLen) * sizeof(float);
        copyExtParams.srcStride = ((this->ubW - (copyOutLen)) * sizeof(float)) / 32;
        copyExtParams.dstStride = (this->inD * this->inH * this->inW - (copyOutLen)) * sizeof(float);
        DataCopyPad(this->workspaceGm[dstGmOffset], this->inputGradUbFp32, copyExtParams);
    } else {
        copyExtParams.blockLen = (copyOutLen) * sizeof(T);
        copyExtParams.srcStride = ((this->ubW - (copyOutLen)) * sizeof(T)) / 32;
        copyExtParams.dstStride = (this->inD * this->inH * this->inW - (copyOutLen)) * sizeof(T);
        DataCopyPad(this->yGm[dstGmOffset], this->inputGradUb, copyExtParams);
    }
    if (this->IsOverlap) {
        SetAtomicNone();
    }
}
} // namespace AvgPool3DGrad
#endif // AVG_POOL3D_GRAD_SCATTER_H