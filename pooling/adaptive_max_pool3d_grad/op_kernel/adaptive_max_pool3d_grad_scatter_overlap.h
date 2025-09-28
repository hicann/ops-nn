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
 * \file adaptive_max_pool3d_grad_scatter_overlap.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_OVERLAP_H
#define ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_OVERLAP_H
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "adaptive_max_pool3d_grad_common.h"
#include "adaptive_max_pool3d_grad_scatter_base.h"

namespace AdaptiveMaxPool3DGrad {
using namespace AscendC;
using namespace AdaptiveMaxPool3DGradComm;

template <typename TX, typename TGrad, typename TArgmax, typename TY>
class AdaptiveMaxPool3DGradScatterOverlap : public AdaptiveMaxPool3DGradScatterBase<TX, TGrad, TArgmax, TY>
{
public:
    __aicore__ inline AdaptiveMaxPool3DGradScatterOverlap(TPipe* pipe)
        : AdaptiveMaxPool3DGradScatterBase<TX, TGrad, TArgmax, TY>(pipe)
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace,
        const AdaptiveMaxPool3DGradTilingData* tiling)
    {
        // load tiling data
        this->InitParams(tiling);
        // set global buffer
        this->InitInputsOutputs(x, grad, argmax, y, usrWorkspace);
        // init global memory
        InitOutGlobalMemory(x, grad, argmax, y, usrWorkspace);
        // ub buffer init
        this->InitUbBuffer();
    }

    __aicore__ inline void InitOutGlobalMemory(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace)
    {
        if constexpr (!is_same<TY, float>::value) {
            InitGlobalMemory(this->workspaceGm, this->params_.initLen, 0.0f);
        } else {
            InitGlobalMemory(this->yGm, this->params_.initLen, static_cast<TY>(0));
        }
        event_t eventMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3ToS);
    }

    __aicore__ inline void CalcOutOffset()
    {
        LocalTensor<TArgmax> argmaxUb = this->argmaxQue.template DeQue<TArgmax>(); //  need free in the end
        LocalTensor<TGrad> gradUb = this->gradQue.template DeQue<TGrad>();         //  need free in the end

        event_t eventMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventMTE2ToS);

        uint64_t basedhwLen = this->block_.doShape * this->block_.hoShape * this->block_.woShape;
        for (uint64_t ncIdx = 0; ncIdx < this->block_.ncShape; ncIdx++) {
            uint64_t ncOffset = this->block_.baseNcOffset * this->params_.diHiWiLen;
            for (uint64_t dhwIdx = 0; dhwIdx < basedhwLen; dhwIdx++) {
                uint64_t ubOffset = ncIdx * basedhwLen + dhwIdx;
                uint64_t gmOffset = ncOffset + (uint64_t)argmaxUb.GetValue(ubOffset);
                float gradValueFloat;
                DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(this->workspaceGm);
                if constexpr (is_same<TY, float>::value) {
                    gradValueFloat = (this->yGm.GetValue(gmOffset) + gradUb.GetValue(ubOffset));
                    this->yGm.SetValue(gmOffset, gradValueFloat);
                } else {
                    if constexpr (IsSameType<TGrad, bfloat16_t>::value) {
                        float ubValueFloat32 = ToFloat(gradUb.GetValue(ubOffset));
                        float gmValueFloat32 = this->workspaceGm.GetValue(gmOffset);
                        gradValueFloat = gmValueFloat32 + ubValueFloat32;
                    } else {
                        gradValueFloat = (this->workspaceGm.GetValue(gmOffset) + (float)gradUb.GetValue(ubOffset));
                    }
                    this->workspaceGm.SetValue(gmOffset, gradValueFloat);
                }
                DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(this->workspaceGm);
            }
            this->block_.ShapeSum += basedhwLen;
            if (this->block_.ShapeSum == this->params_.doDim * this->params_.hoDim * this->params_.woDim) {
                this->block_.baseNcOffset += 1;
                this->block_.ShapeSum = 0;
            }
        }

        this->gradQue.FreeTensor(gradUb);
        this->argmaxQue.FreeTensor(argmaxUb);
    }

    __aicore__ inline void CalcBlock()
    {
        this->CopyInGrad();
        this->CopyInArgmax();
        CalcOutOffset();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitCastUbBuffer()
    {
        this->pipe_->Reset();
        uint64_t maxCalcNum = this->params_.ubSize / (sizeof(half) + sizeof(float));
        this->pipe_->InitBuffer(this->wsQue, 1, maxCalcNum * sizeof(float));
        this->pipe_->InitBuffer(this->yQue, 1, maxCalcNum * sizeof(half));
    }

    __aicore__ inline void ProcessCast()
    {
        uint64_t maxCalcNum = this->params_.ubSize / (sizeof(half) + sizeof(float));
        uint64_t totalLoops = CeilDiv(this->params_.initLen, maxCalcNum);
        uint64_t calcTail = this->params_.initLen - (totalLoops - 1) * maxCalcNum;
        for (uint64_t loopIndex = 0; loopIndex < totalLoops; loopIndex++) {
            uint64_t calcNum = (loopIndex == totalLoops - 1) ? calcTail : maxCalcNum;
            CopyInWorkspace(loopIndex * maxCalcNum, calcNum);
            ComputeCast(calcNum);
            CopyOutCast(loopIndex * maxCalcNum, calcNum);
        }
    }

    __aicore__ inline void CopyInWorkspace(uint64_t gmOffset, uint64_t calcNum)
    {
        LocalTensor<float> fp32Ub = this->wsQue.template AllocTensor<float>();

        DataCopyExtParams copyParamsWs;
        copyParamsWs.blockCount = 1;
        copyParamsWs.blockLen = calcNum * sizeof(float);
        copyParamsWs.srcStride = 0;
        copyParamsWs.dstStride = 0;
        DataCopyPadExtParams<float> padWs{false, 0, 0, 0};

        DataCopyPad(fp32Ub, this->workspaceGm[gmOffset], copyParamsWs, padWs);
        this->wsQue.EnQue(fp32Ub);
    }

    __aicore__ inline void ComputeCast(uint64_t calcNum)
    {
        LocalTensor<float> fp32Ub = this->wsQue.template DeQue<float>();
        LocalTensor<TY> b16Ub = this->yQue.template AllocTensor<TY>();
        if constexpr (is_same<TY, half>::value) {
            Cast(b16Ub, fp32Ub, RoundMode::CAST_NONE, calcNum); // 也可以只cast valid
        } else if constexpr (is_same<TY, bfloat16_t>::value) {
            Cast(b16Ub, fp32Ub, RoundMode::CAST_RINT, calcNum);
        }
        this->wsQue.template FreeTensor(fp32Ub);
        this->yQue.template EnQue(b16Ub);
    }

    __aicore__ inline void CopyOutCast(uint64_t gmOffset, uint64_t calcNum)
    {
        LocalTensor<TY> yUb = this->yQue.template DeQue<TY>();
        DataCopyExtParams copyParamsY;
        copyParamsY.blockCount = 1;
        copyParamsY.blockLen = calcNum * sizeof(TY);
        copyParamsY.srcStride = 0;
        copyParamsY.dstStride = 0;
        DataCopyPad(this->yGm[gmOffset], yUb, copyParamsY);
        this->yQue.template FreeTensor(yUb);
    }

    __aicore__ inline void Process()
    {
        uint64_t ncIndex = this->params_.ncIndex;
        for (uint64_t i = 0; i < this->params_.ncRealRound; i++) {
            if (ncIndex < this->params_.ncCnt) {
                this->block_.ncCntIndex = ncIndex;
                this->block_.ncShape =
                    this->block_.ncCntIndex >= (this->params_.ncCnt - 1) ? this->params_.ncTail : this->params_.baseNc;
                for (uint64_t j = 0; j < this->params_.dCnt; j++) {
                    this->block_.doCntIndex = j;
                    this->block_.doShape = this->block_.doCntIndex >= (this->params_.dCnt - 1) ? this->params_.doTail :
                                                                                                 this->params_.baseDo;
                    for (uint64_t k = 0; k < this->params_.hCnt; k++) {
                        this->block_.hoCntIndex = k;
                        this->block_.hoShape = this->block_.hoCntIndex >= (this->params_.hCnt - 1) ?
                                                   this->params_.hoTail :
                                                   this->params_.baseHo;
                        for (uint64_t w = 0; w < this->params_.wCnt; w++) {
                            this->block_.woCntIndex = w;
                            this->block_.woShape = this->block_.woCntIndex >= (this->params_.wCnt - 1) ?
                                                       this->params_.woTail :
                                                       this->params_.baseWo;
                            this->block_.offsetGrad =
                                this->block_.ncCntIndex * this->params_.baseNc * this->params_.doDim *
                                    this->params_.hoDim * this->params_.woDim +
                                this->block_.doCntIndex * this->params_.baseDo * this->params_.hoDim *
                                    this->params_.woDim +
                                this->block_.hoCntIndex * this->params_.baseHo * this->params_.woDim +
                                this->block_.woCntIndex * this->params_.baseWo;
                            this->block_.offsetArgmax = this->block_.offsetGrad;
                            CalcBlock();
                        }
                    }
                }
                ncIndex += 1; // 当前ncCntIndex
            }
        }
        if constexpr (!is_same<TY, float>::value) {
            PipeBarrier<PIPE_ALL>();
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(this->workspaceGm);
            InitCastUbBuffer();
            ProcessCast();
        }
    }
};
} // namespace AdaptiveMaxPool3DGrad
#endif // ADAPTIVE_MAX_POOL3D_GRAD_SCATTER_OVERLAP_H