/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file max_pool3d_grad_with_argmax_scatter.h
 * \brief
 */
#ifndef MAX_POOL_GRAD3D_WITH_ARGMAX_SCATTER_H
#define MAX_POOL_GRAD3D_WITH_ARGMAX_SCATTER_H
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "max_pool3d_grad_with_argmax_common.h"
#include "max_pool3d_grad_with_argmax_scatter_base.h"

namespace MaxPool3DGradWithArgmax {
using namespace AscendC;
using namespace MaxPool3DGradWithArgmaxComm;

template <typename TX, typename TGrad, typename TArgmax, typename TY>
class MaxPoolGradWithArgScatter : public MaxPoolGradWithArgScatterBase<TX, TGrad, TArgmax, TY> {
public:
    __aicore__ inline MaxPoolGradWithArgScatter(TPipe* pipe)
        : MaxPoolGradWithArgScatterBase<TX, TGrad, TArgmax, TY>(pipe)
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace,
        const MaxPool3DGradWithArgmaxTilingData* tiling)
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

    __aicore__ inline void InitYGMGlobalMemory(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR usrWorkspace)
    {
        InitGlobalMemory(this->yGm, this->params_.initLen, static_cast<TY>(0));
        event_t eventMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3ToS);
    }

    __aicore__ inline void CalcOutOffset()
    {
        LocalTensor<TArgmax> argmaxUb = this->argmaxQue.template DeQue<TArgmax>(); //  need free in the end
        LocalTensor<TGrad> gradUb = this->gradQue.template DeQue<TGrad>();         //  need free in the end
        LocalTensor<TGrad> yUb = this->yQue.template AllocTensor<TGrad>();         //  need free in the end

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
        uint64_t totalCnt = this->params_.totalCnt;     // 需要处理base矩阵个数
        uint64_t round = this->params_.round;           // 每个核需要处理多少个base矩阵，向上取整
        uint64_t preCoreNum = this->params_.preCoreNum; // 每个核均分完后剩余base由前preCoreNum个核进行填充
        uint64_t index = this->params_.ncIndex;         // 当前核填充数的起始index
        uint64_t realRound = this->params_.realRound;   // 填充轮数

        uint64_t ncIndex = this->params_.ncIndex;
        for (uint64_t i = 0; i < this->params_.ncRealRound; i++) {
            if (ncIndex < this->params_.ncCnt) {
                this->block_.ncCntIndex = ncIndex;
                this->block_.ncShape = this->block_.ncCntIndex >= (this->params_.ncCnt - 1UL) ? this->params_.ncTail :
                                                                                                this->params_.baseNc;
                for (uint64_t j = 0; j < this->params_.doCnt; j++) {
                    this->block_.doCntIndex = j;
                    this->block_.doShape = this->block_.doCntIndex >= (this->params_.doCnt - 1) ? this->params_.doTail :
                                                                                                  this->params_.baseDo;
                    for (uint64_t k = 0; k < this->params_.hoCnt; k++) {
                        this->block_.hoCntIndex = k;
                        this->block_.hoShape = this->block_.hoCntIndex >= (this->params_.hoCnt - 1) ?
                                                   this->params_.hoTail :
                                                   this->params_.baseHo;
                        for (uint64_t l = 0; l < this->params_.woCnt; l++) {
                            this->block_.woCntIndex = l;
                            this->block_.woShape = this->block_.woCntIndex >= (this->params_.woCnt - 1) ?
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
    }
};
} // namespace MaxPool3DGradWithArgmax
#endif // MAX_POOL_GRAD3D_WITH_ARGMAX_SCATTER_H