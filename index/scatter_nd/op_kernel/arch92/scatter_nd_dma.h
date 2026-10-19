/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_dma.h
 * \brief
 */

#ifndef SCATTER_ND_DMA_SMID_H
#define SCATTER_ND_DMA_SMID_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "scatter_offset_utils.h"

namespace ScatterNdDmaSimd {
using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint16_t MAX_RANK_COUNT = 7;
constexpr uint16_t MAX_SHAPE_RANK = 8;
constexpr uint64_t UB_AGLIN_VALUE = 32;

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
class ScatterNdDma {
public:
    __aicore__ inline ScatterNdDma(){};
    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR x, GM_ADDR shape, GM_ADDR y,
                                const ScatterNdTilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process(const ScatterNdTilingData* __restrict tilingData);

private:
    TPipe* pipe;
    GlobalTensor<INDICES_T> idxGm;
    GlobalTensor<PARAMS_T> xGm;
    GlobalTensor<PARAMS_T> clrGm, outputGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueIdx;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> inQueXOffset;
    TBuf<TPosition::VECCALC> strideListBuf;
    TBuf<TPosition::VECCALC> outputShapeBuf;

    TYPE_T currClrBlockTilingSize = 0;
    uint32_t blockIdx = 0;
    uint64_t indicesOffset = 0;
    uint64_t indicesLoop = 0;
    uint32_t indicesTailNum = 0;
    uint32_t updateTailLoopNum = 0;
    uint32_t updateTailTailNum = 0;
};

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__aicore__ inline void ScatterNdDma<INDICES_T, PARAMS_T, TYPE_T>::Init(GM_ADDR indices, GM_ADDR x, GM_ADDR shape,
                                                                       GM_ADDR y,
                                                                       const ScatterNdTilingData* __restrict tilingData,
                                                                       TPipe* pipeIn)
{
    blockIdx = GetBlockIdx();

    // calculate indice offset size by x offset size

    if (blockIdx == tilingData->clrBlockNum - 1) {
        this->currClrBlockTilingSize = static_cast<TYPE_T>(tilingData->clrTailBlockTilingSize);
    } else {
        this->currClrBlockTilingSize = static_cast<TYPE_T>(tilingData->clrBlockTilingSize);
    }
    if (blockIdx < tilingData->tailDmaBlockFactor) {
        indicesOffset = (tilingData->dmaBlockFactor + 1) * blockIdx;
        indicesLoop = tilingData->normBlockIndicesLoop;
        indicesTailNum = tilingData->normBlockIndicesTailNum;
        updateTailLoopNum = tilingData->normBlockTailLoop;
        updateTailTailNum = tilingData->normBlockTailTailNum;
    } else {
        indicesOffset = tilingData->dmaBlockFactor * blockIdx + tilingData->tailDmaBlockFactor;
        indicesLoop = tilingData->tailBlockIndicesLoop;
        indicesTailNum = tilingData->tailBlockIndicesTailNum;
        updateTailLoopNum = tilingData->tailBlockTailLoop;
        updateTailTailNum = tilingData->tailBlockTailTailNum;
    }

    idxGm.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    xGm.SetGlobalBuffer((__gm__ PARAMS_T*)x);
    clrGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);
    outputGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);
    if (blockIdx < tilingData->clrBlockNum) {
        // fill编译有问题
        // __gm__ PARAMS_T* clrGmPtr = (__gm__ PARAMS_T*)(clrGm.GetPhyAddr(tilingData->clrBlockTilingSize * blockIdx));
        // AscendC::Fill<PARAMS_T>(clrGmPtr, this->currClrBlockTilingSize, static_cast<PARAMS_T>(0));
        InitOutput<PARAMS_T>(clrGm[tilingData->clrBlockTilingSize * blockIdx], this->currClrBlockTilingSize, 0);
    }
    SyncAll();
    pipe = pipeIn;
    pipe->InitBuffer(inQueIdx, BUFFER_NUM, tilingData->ubIndicesSize);
    pipe->InitBuffer(inQueXOffset, BUFFER_NUM,
                     ops::CeilAlign(tilingData->ubIndicesFactor * sizeof(TYPE_T), UB_AGLIN_VALUE));
    pipe->InitBuffer(inQueX, BUFFER_NUM, tilingData->ubUpdateSize);
    pipe->InitBuffer(strideListBuf, MAX_RANK_COUNT * sizeof(TYPE_T));
    pipe->InitBuffer(outputShapeBuf, MAX_SHAPE_RANK * sizeof(TYPE_T));
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__aicore__ inline void ScatterNdDma<INDICES_T, PARAMS_T, TYPE_T>::Process(
    const ScatterNdTilingData* __restrict tilingData)
{
    if (blockIdx < tilingData->blockNum) {
        LocalTensor<TYPE_T> strideList = strideListBuf.Get<TYPE_T>();
        LocalTensor<TYPE_T> outputShape = outputShapeBuf.Get<TYPE_T>();
        for (uint32_t i = 0; i < MAX_RANK_COUNT; i++) {
            strideList.SetValue(i, static_cast<TYPE_T>(tilingData->strideList[i]));
        }
        for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
            outputShape.SetValue(i, static_cast<TYPE_T>(tilingData->outPutShape[i]));
        }
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);

        for (uint64_t loop = 0; loop < indicesLoop; loop++) {
            uint32_t curIndicesNum = 0;
            uint32_t updateLoopNum = 0;
            uint32_t updateTailNum = 0;
            if (loop != indicesLoop - 1) {
                curIndicesNum = tilingData->ubIndicesFactor;
                updateLoopNum = tilingData->normBlockUbLoop;
                updateTailNum = tilingData->normBlockUbTailNum;
            } else {
                curIndicesNum = indicesTailNum;
                updateLoopNum = updateTailLoopNum;
                updateTailNum = updateTailTailNum;
            }
            LocalTensor<INDICES_T> idxLocal = inQueIdx.AllocTensor<INDICES_T>();
            DataCopyExtParams idxCopyParams{
                1, static_cast<uint32_t>(curIndicesNum * tilingData->rankSize * sizeof(INDICES_T)), 0, 0, 0};
            DataCopyPadExtParams<INDICES_T> idxPadParams{false, 0, 0, 0};
            DataCopyPad(idxLocal, idxGm[(indicesOffset + loop * tilingData->ubIndicesFactor) * tilingData->rankSize],
                        idxCopyParams, idxPadParams);
            inQueIdx.EnQue(idxLocal);
            idxLocal = inQueIdx.DeQue<INDICES_T>();
            LocalTensor<TYPE_T> xOffsetLocal = inQueXOffset.AllocTensor<TYPE_T>();
            ScatterOffsetUtils::ComputeOutOfset<INDICES_T, TYPE_T>(
                idxLocal, xOffsetLocal, strideList, outputShape, curIndicesNum, tilingData->rankSize,
                static_cast<uint32_t>(tilingData->sliceSize * sizeof(PARAMS_T)));
            inQueXOffset.EnQue(xOffsetLocal);
            xOffsetLocal = inQueXOffset.DeQue<TYPE_T>();

            for (uint32_t updateloop = 0; updateloop < updateLoopNum; updateloop++) {
                uint64_t updateOffset = (indicesOffset + loop * tilingData->ubIndicesFactor +
                                         updateloop * tilingData->ubSliceFactor) *
                                        tilingData->sliceSize;
                uint32_t curUpdateNum = updateloop == updateLoopNum - 1 ? updateTailNum : tilingData->ubSliceFactor;
                LocalTensor<PARAMS_T> xLocal = inQueX.AllocTensor<PARAMS_T>();
                DataCopyExtParams xCopyParams{static_cast<uint16_t>(curUpdateNum),
                                              static_cast<uint32_t>(tilingData->sliceSize * sizeof(PARAMS_T)), 0, 0, 0};
                DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
                DataCopyPad(xLocal, xGm[updateOffset], xCopyParams, xPadParams);
                inQueX.EnQue(xLocal);
                xLocal = inQueX.DeQue<PARAMS_T>();

                SetAtomicAdd<PARAMS_T>();
                ScatterParams scatterDataCopyParams;
                scatterDataCopyParams.offsetNum = curUpdateNum;
                scatterDataCopyParams.burstLen = tilingData->sliceSize * sizeof(PARAMS_T);
                scatterDataCopyParams.srcStride = tilingData->oneSliceAlign * sizeof(PARAMS_T);
                ScatterDataCopy(outputGm, xLocal, xOffsetLocal[updateloop * tilingData->ubSliceFactor],
                                scatterDataCopyParams);
                SetAtomicNone();

                inQueX.FreeTensor(xLocal);
            }
            inQueXOffset.FreeTensor(xOffsetLocal);
            inQueIdx.FreeTensor(idxLocal);
        }
    }
}

} // namespace ScatterNdDmaSimd

#endif // SCATTER_ND_SMIT_H
