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
 * \file max_pool3d_grad_scatter_base_template.h
 * \brief
 */

#ifndef MAX_POOL3D_GRAD_SCATTER_BASE_TEMPLATE_H
#define MAX_POOL3D_GRAD_SCATTER_BASE_TEMPLATE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "max_pool3d_grad_common.h"

namespace MaxPool3DGradScatterInternal {
using namespace AscendC;
using namespace MaxPool3DGradCommon;

// 通用的 Base 模板类
template <typename TX, typename TGrad, typename TArgmax, typename TY,
          typename TilingData, typename ParamsType, typename BlockType>
class MaxPool3DGradScatterBaseTemplate {
public:
    __aicore__ inline MaxPool3DGradScatterBaseTemplate(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void InitParams(const TilingData* __restrict__ tiling)
    {
        params_.ncDim = tiling->ncDim; // 池化正向输出结果NC维度
        params_.doDim = tiling->doDim; // 池化正向输出结果d维度
        params_.hoDim = tiling->hoDim; // 池化正向输出结果h维度
        params_.woDim = tiling->woDim; // 池化正向输出结果w维度
        params_.diDim = tiling->diDim; // 池化正向输入d维度
        params_.hiDim = tiling->hiDim; // 池化正向输入h维度
        params_.wiDim = tiling->wiDim; // 池化正向输入w维度

        params_.baseNc = tiling->baseNc; // 每次CalcOutOffset矩阵的NC维度
        params_.baseDo = tiling->baseDo; // 每次CalcOutOffset矩阵的d维度
        params_.baseHo = tiling->baseHo; // 每次CalcOutOffset矩阵的h维度
        params_.baseWo = tiling->baseWo; // 每次CalcOutOffset矩阵的w维度
        params_.ncTail = tiling->ncTail;
        params_.doTail = tiling->doTail;
        params_.hoTail = tiling->hoTail;
        params_.woTail = tiling->woTail;
        params_.ncCnt = tiling->ncCnt; // nc方向base矩阵个数
        params_.doCnt = tiling->doCnt; // d方向base矩阵个数
        params_.hoCnt = tiling->hoCnt; // h方向base矩阵个数
        params_.woCnt = tiling->woCnt; // w方向base矩阵个数
        params_.usedCoreNum = tiling->usedCoreNum;

        params_.totalCnt = tiling->totalCnt;     // 需要处理base矩阵个数
        params_.ncCntRound = tiling->ncRound;    // 多核切nc，先分nc,向上取整,
        params_.preCoreNum = tiling->preCoreNum; // 每个核均分完后剩余nc由前preCoreNum个核进行填充
        params_.diHiWiLen = params_.diDim * params_.hiDim * params_.wiDim;
        params_.ncRealRound = 0;
        params_.ubSize = tiling->totalUBSize;
        uint64_t blockId = GetBlockIdx();
        if (params_.preCoreNum == 0 || blockId < params_.preCoreNum) { // 前preCoreNum个核
            params_.ncIndex =
                blockId * params_.ncCntRound; // 由于轮数为向上取整，所以当前核nc数的起始位置为 每个核平均nc数*核数
            params_.ncRealRound = params_.ncCntRound;
        } else {
            params_.ncIndex =
                params_.preCoreNum * (params_.ncCntRound) + (blockId - params_.preCoreNum) * tiling->ncRoundTail;
            params_.ncRealRound = tiling->ncRoundTail;
        }
        params_.realRound = params_.ncRealRound * params_.doCnt * params_.hoCnt *
                            params_.woCnt; // 处理的base矩阵个数，包括base矩阵和tail矩阵
    }

    __aicore__ inline void InitInputsOutputs(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, 
                                             GM_ADDR y, GM_ADDR usrWorkspace)
    {
        gradGm.SetGlobalBuffer((__gm__ TGrad*)grad);
        argmaxGm.SetGlobalBuffer((__gm__ TArgmax*)argmax);

        // 起始地址
        uint64_t initOffset = params_.ncIndex * params_.baseNc * params_.diHiWiLen;
        uint64_t initLen = 0;
        uint64_t ncIndex = params_.ncIndex;
        for (uint64_t j = 0; j < params_.ncRealRound; j++) {
            block_.ncShape = ncIndex >= (params_.ncCnt - 1) ? params_.ncTail : params_.baseNc;
            initLen += block_.ncShape * params_.diHiWiLen;
            ncIndex += 1; // 当前ncCntIndex
        }
        params_.initLen = initLen;
        params_.initOffset = initOffset;

        yGm.SetGlobalBuffer((__gm__ TY*)y + initOffset, initLen);
        workspaceGm.SetGlobalBuffer((__gm__ float*)usrWorkspace + initOffset, initLen);
    }

    __aicore__ inline void InitUbBuffer()
    {
        pipe_->InitBuffer(
            gradQue, 1, params_.baseNc * params_.baseDo * params_.baseHo * params_.baseWo * sizeof(TGrad));
        pipe_->InitBuffer(
            argmaxQue, 1, params_.baseNc * params_.baseDo * params_.baseHo * params_.baseWo * sizeof(TArgmax));
    }

    template <typename T>
    __aicore__ inline void CopyInDataGeneric(
        GlobalTensor<T>& sourceGm, TQue<QuePosition::VECIN, 1>& targetQue, uint64_t dataOffset)
    {
        LocalTensor<T> dataUb = targetQue.AllocTensor<T>();
        const uint32_t elementSize = sizeof(T);
        
        // 计算基础块长度
        const uint32_t baseblockLen = block_.doShape * block_.hoShape * block_.woShape * elementSize;
        
        if (params_.baseWo == params_.woDim) {
            // 情况1: 宽度维度相同，可以使用连续拷贝
            if (baseblockLen > UNIT_BLOCK_LEN && baseblockLen % UNIT_BLOCK_LEN == 0) {
                // 块长度是 UNIT_BLOCK_LEN 的整数倍
                DataCopyExtParams copyParams;
                copyParams.blockCount = block_.ncShape;
                copyParams.blockLen = baseblockLen;
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                
                DataCopyPad(dataUb, sourceGm[dataOffset], copyParams, padParams);
            } else {
                // 块长度不是 UNIT_BLOCK_LEN 的整数倍，需要分块拷贝
                DataCopyExtParams copyParams;
                const uint32_t totalBlockLen = block_.ncShape * baseblockLen;
                const uint16_t baseBlockCount = totalBlockLen / UNIT_BLOCK_LEN;
                
                // 拷贝基础块
                copyParams.blockCount = baseBlockCount;
                copyParams.blockLen = UNIT_BLOCK_LEN;
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(dataUb, sourceGm[dataOffset], copyParams, padParams);
                
                // 拷贝尾部剩余数据
                const uint32_t tailBlockLen = totalBlockLen % UNIT_BLOCK_LEN;
                if (tailBlockLen != 0) {
                    copyParams.blockCount = 1;
                    copyParams.blockLen = tailBlockLen;
                    padParams.isPad = true;
                    DataCopyPad(
                        dataUb[baseBlockCount * (UNIT_BLOCK_LEN / elementSize)],
                        sourceGm[dataOffset + baseBlockCount * (UNIT_BLOCK_LEN / elementSize)],
                        copyParams, padParams);
                }
            }
        } else {
            // 情况2: 宽度维度不同，需要循环拷贝
            for (uint64_t loopidx = 0; loopidx < block_.ncShape; loopidx++) {
                DataCopyExtParams copyParams;
                copyParams.blockCount = params_.baseHo;
                copyParams.blockLen = params_.baseWo * elementSize;
                copyParams.srcStride = (params_.woDim - block_.woShape) * elementSize;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                
                DataCopyPad(
                    dataUb[loopidx * block_.hoShape * block_.woShape],
                    sourceGm[dataOffset + loopidx * params_.hoDim * params_.woDim], 
                    copyParams, padParams);
            }
        }
        
        targetQue.EnQue(dataUb);
    }

    __aicore__ inline void CopyInGrad()
    {
        CopyInDataGeneric<TGrad>(gradGm, gradQue, block_.offsetGrad);
    }

    __aicore__ inline void CopyInArgmax()
    {
        CopyInDataGeneric<TArgmax>(argmaxGm, argmaxQue, block_.offsetArgmax);
    }

public:
    ParamsType params_;
    BlockType block_;
    TPipe* pipe_ = nullptr;

    GlobalTensor<TGrad> gradGm;
    GlobalTensor<TArgmax> argmaxGm;

    GlobalTensor<TY> yGm;
    GlobalTensor<float> workspaceGm;

    TQue<QuePosition::VECIN, 1> gradQue;
    TQue<QuePosition::VECIN, 1> argmaxQue;
    TQue<QuePosition::VECIN, 1> wsQue;
    TQue<QuePosition::VECOUT, 1> yQue;
};

} // namespace MaxPool3DGradScatterInternal

#endif // MAX_POOL3D_GRAD_SCATTER_BASE_TEMPLATE_H