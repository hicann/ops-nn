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
 * \file ada_layer_norm_grad_common_init.inl
 * \brief
 */

#ifndef ADA_LAYER_NORM_GRAD_COMMON_INIT_INL
#define ADA_LAYER_NORM_GRAD_COMMON_INIT_INL

namespace AdaLayerNormGrad {

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR pdX, GM_ADDR pdScale, GM_ADDR pdShift, GM_ADDR pdGamma, GM_ADDR pdBeta,
    GM_ADDR workspace, const AdaLayerNormGradTilingDataCommon* tilingData, TPipe& pipeIn)
{
    pipe = pipeIn;
    int64_t curRowsNum =
        (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail;
    int64_t formerBlockLength = tilingData->blockFormer * tilingData->col;
    int64_t curBlockLength = curRowsNum * tilingData->col;
    int64_t blockBatchOffset = tilingData->blockFormer / tilingData->seq;
    int64_t curBlockBatchNum = curRowsNum / tilingData->seq;
    int64_t wsLenPerBlock = tilingData->colAlignV * (COMMON_CONSTANT_TWO + blockBatchOffset * 2);

    pdGammaOutTensorGM.SetGlobalBuffer((__gm__ U*)pdGamma, tilingData->col);
    pdBetaOutTensorGM.SetGlobalBuffer((__gm__ U*)pdBeta, tilingData->col);
    workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace, wsLenPerBlock * (tilingData->blockNum - 1) + (tilingData->blockTail / tilingData->seq) * tilingData->colAlignV);

    if (GetBlockIdx() < tilingData->blockNum) {
        int64_t blockStartBatchIdx = (GetBlockIdx() * tilingData->blockFormer) / tilingData->seq;

        // int64_t scaleBufferBytes =
        //     (GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormerScaleBufferBytes
        //                                                 : tilingData->blockTailScaleBufferBytes;

        dyInTensorGM_.SetGlobalBuffer((__gm__ T*)dy + formerBlockLength * GetBlockIdx(), curBlockLength);
        xInTensorGM_.SetGlobalBuffer((__gm__ T*)x + formerBlockLength * GetBlockIdx(), curBlockLength);
        rstdInTensorGM_.SetGlobalBuffer((__gm__ float*)rstd + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
        meanInTensorGM_.SetGlobalBuffer((__gm__ float*)mean + tilingData->blockFormer * GetBlockIdx(), curRowsNum);
        gammaInTensorGM_.SetGlobalBuffer((__gm__ U*)gamma, tilingData->col);
        betaInTensorGM_.SetGlobalBuffer((__gm__ U*)beta, tilingData->col);
        scaleInTensorGM_.SetGlobalBuffer((__gm__ T*)scale + blockStartBatchIdx * tilingData->col, (curRowsNum / tilingData->seq) * tilingData->col);
        
        pdScaleOutTensorGM.SetGlobalBuffer((__gm__ T*)pdScale + blockStartBatchIdx * tilingData->col, (curRowsNum / tilingData->seq) * tilingData->col);
        pdShiftOutTensorGM.SetGlobalBuffer((__gm__ T*)pdShift + blockStartBatchIdx * tilingData->col, (curRowsNum / tilingData->seq) * tilingData->col);
        pdXOutTensorGM_.SetGlobalBuffer((__gm__ T*)pdX + formerBlockLength * GetBlockIdx(), curBlockLength);

        workspaceGM.SetGlobalBuffer(
            (__gm__ float*)workspace + wsLenPerBlock * GetBlockIdx(), (2 + 2 * curBlockBatchNum) * tilingData->colAlignV);
        dScaleWorkspaceGM.SetGlobalBuffer(
            (__gm__ float*)workspace + wsLenPerBlock * GetBlockIdx() + 2 * tilingData->colAlignV, curBlockBatchNum * tilingData->colAlignV);
        dShiftWorkspaceGM.SetGlobalBuffer(
            (__gm__ float*)workspace + wsLenPerBlock * GetBlockIdx() + (2 + curBlockBatchNum) * tilingData->colAlignV, curBlockBatchNum * tilingData->colAlignV);

        InitOutput<float>(workspaceGM, ( 2 + 2 * curBlockBatchNum) * tilingData->colAlignV, 0.0);

        if constexpr (!isDeterministic) {
            if (GetBlockIdx() == 0) {
                InitOutput<U>(pdGammaOutTensorGM, tilingData->col, 0.0);
                InitOutput<U>(pdBetaOutTensorGM, tilingData->col, 0.0);
            }
            CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
        }
        PipeBarrier<PIPE_ALL>();

        pipe.InitBuffer(queue0_, 1, tilingData->wholeBufferBytes);
        pipe.InitBuffer(queue1_, 1, tilingData->wholeBufferBytes);
        pipe.InitBuffer(queue2_, 1, tilingData->lastRBufferBytes);
        pipe.InitBuffer(queue3_, 1, tilingData->nlastRBufferBytes);
        pipe.InitBuffer(queue4_, 1, tilingData->nlastRBufferBytes);
        pipe.InitBuffer(queue5_, 1, tilingData->blockFormerScaleBufferBytes);
        pipe.InitBuffer(queue6_, 1, tilingData->wholeBufferBytes);
        pipe.InitBuffer(queue7_, 1, tilingData->blockFormerScaleBufferBytes);
        pipe.InitBuffer(queue8_, 1, tilingData->blockFormerScaleBufferBytes);
        pipe.InitBuffer(queue9_, 1, tilingData->nlastRBufferBytes);
        pipe.InitBuffer(queue10_, 1, tilingData->nlastRBufferBytes);
        pipe.InitBuffer(tmpTensor0_, tilingData->wholeBufferBytes);
        pipe.InitBuffer(tmpTensor1_, tilingData->lastBrcbBufferBytes);

        coff_ = static_cast<float>(1.0) / static_cast<float>(tilingData->col);
    } else if (!isDeterministic) {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL);
    }
}

} // namespace AdaLayerNormGrad

#endif