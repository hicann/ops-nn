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
 * \file max_pool3d_grad_scatter_overlap_unified.h
 * \brief
 */

#ifndef MAX_POOL3D_GRAD_SCATTER_OVERLAP_UNIFIED_H
#define MAX_POOL3D_GRAD_SCATTER_OVERLAP_UNIFIED_H

#include "max_pool3d_grad_scatter_base_template.h"
#include "max_pool3d_grad_common.h"

namespace MaxPool3DGradCommon {
using namespace AscendC;

// ScatterOverlap的统一实现类
template <typename TX, typename TGrad, typename TArgmax, typename TY,
          typename TilingData, typename ParamsType, typename BlockType,
          template<typename, typename, typename, typename, 
                   typename, typename, typename> class BaseClass>
class MaxPool3DGradScatterOverlapUnified : 
    public BaseClass<TX, TGrad, TArgmax, TY, TilingData, ParamsType, BlockType>
{
private:
    using Base = BaseClass<TX, TGrad, TArgmax, TY, TilingData, ParamsType, BlockType>;
    
    // 类型检查辅助
    template <typename T>
    using IsFloat = std::is_same<T, float>;
    
    template <typename T>
    using IsHalf = std::is_same<T, half>;
    
    template <typename T>
    using IsBFloat16 = std::is_same<T, bfloat16_t>;
    
public:
    __aicore__ inline MaxPool3DGradScatterOverlapUnified(TPipe* pipe) : Base(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace,
        const TilingData* __restrict__ tiling)
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

    __aicore__ inline void InitOutGlobalMemory(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, 
                                               GM_ADDR y, GM_ADDR usrWorkspace)
    {
        if constexpr (!IsFloat<TY>::value) {
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
        LocalTensor<TArgmax> argmaxUb = this->argmaxQue.template DeQue<TArgmax>();
        LocalTensor<TGrad> gradUb = this->gradQue.template DeQue<TGrad>();

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
                
                if constexpr (IsFloat<TY>::value) {
                    gradValueFloat = (this->yGm.GetValue(gmOffset) + gradUb.GetValue(ubOffset));
                    this->yGm.SetValue(gmOffset, gradValueFloat);
                } else {
                    if constexpr (IsBFloat16<TGrad>::value) {
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
        PipeBarrier<PIPE_ALL>();
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
        
        if constexpr (IsHalf<TY>::value) {
            Cast(b16Ub, fp32Ub, RoundMode::CAST_NONE, calcNum);
        } else if constexpr (IsBFloat16<TY>::value) {
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
        ProcessCommon<ParamsType, BlockType>(this, this->params_.ncIndex);

        // 类型转换逻辑（仅当 TY 不是 float 时执行）
        if constexpr (!IsFloat<TY>::value) {
            PipeBarrier<PIPE_ALL>();
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(this->workspaceGm);
            InitCastUbBuffer();
            ProcessCast();
        }
    }
};

} // namespace MaxPool3DGradCommon

#endif // MAX_POOL3D_GRAD_SCATTER_OVERLAP_UNIFIED_H