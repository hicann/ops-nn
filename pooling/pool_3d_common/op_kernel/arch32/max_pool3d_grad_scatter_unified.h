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
 * \file max_pool3d_grad_scatter_unified.h
 * \brief
 */

#ifndef MAX_POOL3D_GRAD_SCATTER_UNIFIED_H
#define MAX_POOL3D_GRAD_SCATTER_UNIFIED_H

#include "max_pool3d_grad_scatter_base_template.h"
#include "max_pool3d_grad_common.h"

namespace MaxPool3DGradCommon {
using namespace AscendC;

// Scatter的统一实现类
template <typename TX, typename TGrad, typename TArgmax, typename TY,
          typename TilingData, typename ParamsType, typename BlockType,
          template<typename, typename, typename, typename, 
                   typename, typename, typename> class BaseClass>
class MaxPool3DGradScatterUnified : 
    public BaseClass<TX, TGrad, TArgmax, TY, TilingData, ParamsType, BlockType>
{
private:
    using Base = BaseClass<TX, TGrad, TArgmax, TY, TilingData, ParamsType, BlockType>;
    
public:
    __aicore__ inline MaxPool3DGradScatterUnified(TPipe* pipe) : Base(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace,
        const TilingData* __restrict__ tiling)
    {
        // load tiling data
        this->InitParams(tiling);
        // set global buffer
        this->InitInputsOutputs(x, grad, argmax, y, usrWorkspace);
        // init global memory
        InitYGMGlobalMemory(x, grad, argmax, y, usrWorkspace);
        // ub buffer init
        this->InitUbBuffer();
        this->pipe_->InitBuffer(this->yQue, 1, BLOCK_SIZE);
    }

    __aicore__ inline void InitYGMGlobalMemory(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, 
                                               GM_ADDR y, GM_ADDR usrWorkspace)
    {
        InitGlobalMemory(this->yGm, this->params_.initLen, static_cast<TY>(0));
        event_t eventMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3ToS);
    }

    __aicore__ inline void CalcOutOffset()
    {
        LocalTensor<TArgmax> argmaxUb = this->argmaxQue.template DeQue<TArgmax>();
        LocalTensor<TGrad> gradUb = this->gradQue.template DeQue<TGrad>();
        LocalTensor<TGrad> yUb = this->yQue.template AllocTensor<TGrad>();

        event_t eventMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventMTE2ToS);

        uint64_t basedhwLen = this->block_.doShape * this->block_.hoShape * this->block_.woShape;
        for (uint64_t ncIdx = 0; ncIdx < this->block_.ncShape; ncIdx++) {
            uint64_t ncOffset = this->block_.baseNcOffset * this->params_.diHiWiLen;
            for (uint64_t dhwIdx = 0; dhwIdx < basedhwLen; dhwIdx++) {
                uint64_t ubOffset = ncIdx * basedhwLen + dhwIdx;
                uint64_t outOffset = ncOffset + (uint64_t)argmaxUb.GetValue(ubOffset);
                TGrad gradValue = gradUb.GetValue(ubOffset);
                yUb.SetValue(0, gradValue);
                
                event_t eventSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventSToMTE3);
                WaitFlag<HardEvent::S_MTE3>(eventSToMTE3);
                
                DataCopyPad(this->yGm[outOffset], yUb, {1, static_cast<uint16_t>(sizeof(TGrad)), 0, 0});
                
                event_t eventMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                SetFlag<HardEvent::MTE3_S>(eventMTE3ToS);
                WaitFlag<HardEvent::MTE3_S>(eventMTE3ToS);
            }
            
            this->block_.ShapeSum += basedhwLen;
            if (this->block_.ShapeSum == this->params_.doDim * this->params_.hoDim * this->params_.woDim) {
                this->block_.baseNcOffset += 1;
                this->block_.ShapeSum = 0;
            }
        }

        this->gradQue.FreeTensor(gradUb);
        this->argmaxQue.FreeTensor(argmaxUb);
        this->yQue.FreeTensor(yUb);
    }

    __aicore__ inline void CalcBlock()
    {
        this->CopyInGrad();
        this->CopyInArgmax();
        CalcOutOffset();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        ProcessCommon<ParamsType, BlockType>(this, this->params_.ncIndex);
    }
};

} // namespace MaxPool3DGradCommon

#endif // MAX_POOL3D_GRAD_SCATTER_UNIFIED_H