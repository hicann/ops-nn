/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AVERAGE_POOLING_GRAD_TILED_BASE_H
#define AVERAGE_POOLING_GRAD_TILED_BASE_H

#include "kernel_operator.h"
#include "average_pooling_grad_tiling_data.h"
#include "average_pooling_grad_utils.h"

namespace NsAveragePoolingGrad {

using namespace AscendC;

class AveragePoolingGradTiledBase {
public:
    __aicore__ inline AveragePoolingGradTiledBase() {}

    __aicore__ inline void Init(GM_ADDR origInputShape, GM_ADDR gradOutput, GM_ADDR gradInput,
        const AveragePoolingGradTilingData& tiling)
    {
        (void)origInputShape;
        gradOutputGm.SetGlobalBuffer((__gm__ float*)gradOutput, tiling.n * tiling.c * tiling.hOut * tiling.wOut);
        gradInputGm.SetGlobalBuffer((__gm__ float*)gradInput, tiling.totalElements);
        n = tiling.n;
        c = tiling.c;
        hIn = tiling.hIn;
        wIn = tiling.wIn;
        hOut = tiling.hOut;
        wOut = tiling.wOut;
        kernelH = tiling.kernelH;
        kernelW = tiling.kernelW;
        strideH = tiling.strideH;
        strideW = tiling.strideW;
        padTop = tiling.padTop;
        padLeft = tiling.padLeft;
        countIncludePad = tiling.countIncludePad;
        divisorOverride = tiling.divisorOverride;
        invDivisor = tiling.invDivisor;
        usedCoreNum = tiling.usedCoreNum;
        tileH = tiling.tileH;
        tileW = tiling.tileW;
        outTileStride = AlignUpElements(tileW);
        tileTaskNum = tiling.tileTaskNum;
        gradOutCacheDataNum = tiling.gradOutCacheDataNum;

        const uint32_t outputBytes = tileH * outTileStride * sizeof(float);
        const uint32_t outputAllocBytes = (outputBytes + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        const uint32_t cacheBytes = gradOutCacheDataNum * sizeof(float);
        const uint32_t cacheAllocBytes = (cacheBytes + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        pipe.InitBuffer(outQueue, BUFFER_NUM, outputAllocBytes);
        pipe.InitBuffer(gradOutQueue, BUFFER_NUM, cacheAllocBytes);
    }

protected:
    __aicore__ inline TileTask DecodeTileTask(uint32_t taskId) const
    {
        const uint32_t hTileNum = CeilDivU32(hIn, tileH);
        const uint32_t wTileNum = CeilDivU32(wIn, tileW);
        const uint32_t tasksPerNc = hTileNum * wTileNum;
        if (wTileNum == 0 || tasksPerNc == 0) {
            return TileTask{0, 0, 0, 0, 0, 0, 0};
        }
        const uint32_t ncIndex = taskId / tasksPerNc;
        const uint32_t tileIdInNc = taskId - ncIndex * tasksPerNc;
        const uint32_t hTileId = tileIdInNc / wTileNum;
        const uint32_t wTileId = tileIdInNc - hTileId * wTileNum;

        TileTask task;
        task.nc = ncIndex;
        task.batch = ncIndex / c;
        task.channel = ncIndex - task.batch * c;
        task.hStart = hTileId * tileH;
        task.wStart = wTileId * tileW;
        task.hEnd = static_cast<uint32_t>(ScalarMin(static_cast<int32_t>(task.hStart + tileH), static_cast<int32_t>(hIn)));
        task.wEnd = static_cast<uint32_t>(ScalarMin(static_cast<int32_t>(task.wStart + tileW), static_cast<int32_t>(wIn)));
        return task;
    }

    __aicore__ inline OutputCacheRange CalcOutputCacheRange(const TileTask& task) const
    {
        OutputCacheRange range;
        const int32_t hFirst = static_cast<int32_t>(task.hStart);
        const int32_t hLast = static_cast<int32_t>(task.hEnd) - 1;
        const int32_t wFirst = static_cast<int32_t>(task.wStart);
        const int32_t wLast = static_cast<int32_t>(task.wEnd) - 1;
        range.ohStart = ClampStart(PStart(hFirst, padTop, kernelH, strideH));
        range.ohEnd = ClampEnd(PEnd(hLast, padTop, strideH), static_cast<int32_t>(hOut));
        range.owStart = ClampStart(PStart(wFirst, padLeft, kernelW, strideW));
        range.owEnd = ClampEnd(PEnd(wLast, padLeft, strideW), static_cast<int32_t>(wOut));
        range.cacheH = range.ohEnd > range.ohStart ? static_cast<uint32_t>(range.ohEnd - range.ohStart) : 0U;
        range.cacheW = range.owEnd > range.owStart ? static_cast<uint32_t>(range.owEnd - range.owStart) : 0U;
        range.cacheStride = AlignUpElements(range.cacheW);
        return range;
    }

    __aicore__ inline uint32_t GradInputOffset(const TileTask& task, uint32_t ih, uint32_t iw) const
    {
        return ((task.batch * c + task.channel) * hIn + ih) * wIn + iw;
    }

    __aicore__ inline uint32_t GradOutputOffset(uint32_t batch, uint32_t channel, int32_t oh, int32_t ow) const
    {
        return ((batch * c + channel) * hOut + static_cast<uint32_t>(oh)) * wOut + static_cast<uint32_t>(ow);
    }

    __aicore__ inline int32_t CalcDivisor(int32_t oh, int32_t ow) const
    {
        if (divisorOverride != 0) {
            return divisorOverride;
        }
        const int32_t hStart = oh * static_cast<int32_t>(strideH) - static_cast<int32_t>(padTop);
        const int32_t wStart = ow * static_cast<int32_t>(strideW) - static_cast<int32_t>(padLeft);
        const int32_t validH = ScalarMin(hStart + static_cast<int32_t>(kernelH), static_cast<int32_t>(hIn)) - ScalarMax(hStart, 0);
        const int32_t validW = ScalarMin(wStart + static_cast<int32_t>(kernelW), static_cast<int32_t>(wIn)) - ScalarMax(wStart, 0);
        if (countIncludePad != 0) {
            return static_cast<int32_t>(kernelH * kernelW);
        }
        return validH * validW;
    }

    __aicore__ inline void FillZeroGradInputTile(const TileTask& task)
    {
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        const uint32_t tileRows = task.hEnd - task.hStart;
        Duplicate<float>(outLocal, 0.0f, tileRows * outTileStride);
        outQueue.EnQue(outLocal);
        CopyGradInputTileOut(task);
    }

    __aicore__ inline void CopyGradOutputCache(const TileTask& task, const OutputCacheRange& range)
    {
        LocalTensor<float> gradOutLocal = gradOutQueue.AllocTensor<float>();
        Duplicate<float>(gradOutLocal, 0.0f, gradOutCacheDataNum);
        const uint32_t firstGmOffset = GradOutputOffset(task.batch, task.channel, range.ohStart, range.owStart);
        const uint32_t blockLenBytes = range.cacheW * static_cast<uint32_t>(sizeof(float));
        const uint32_t srcStrideBytes = (wOut - range.cacheW) * static_cast<uint32_t>(sizeof(float));
        const uint32_t blockLenAligned = ((blockLenBytes + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
        const uint32_t rowStrideBytes = range.cacheStride * static_cast<uint32_t>(sizeof(float));
        const uint32_t dstStrideBlocks = (rowStrideBytes - blockLenAligned) / ALIGN_BYTES;
        DataCopyExtParams copyParams{static_cast<uint16_t>(range.cacheH), blockLenBytes, srcStrideBytes, dstStrideBlocks, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0.0f};
        DataCopyPad(gradOutLocal[0], gradOutputGm[firstGmOffset], copyParams, padParams);
        gradOutQueue.EnQue(gradOutLocal);
    }

    __aicore__ inline void CopyGradInputTileOut(const TileTask& task)
    {
        LocalTensor<float> outLocal = outQueue.DeQue<float>();
        const uint32_t tileRows = task.hEnd - task.hStart;
        const uint32_t rowLen = task.wEnd - task.wStart;
        for (uint32_t row = 0; row < tileRows; ++row) {
            const uint32_t ih = task.hStart + row;
            const uint32_t gmOffset = GradInputOffset(task, ih, task.wStart);
            const uint32_t ubOffset = row * outTileStride;
            DataCopyExtParams copyParams{1U, rowLen * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(gradInputGm[gmOffset], outLocal[ubOffset], copyParams);
        }
        outQueue.FreeTensor(outLocal);
    }

protected:
    TPipe pipe;
    TQue<TPosition::VECCALC, BUFFER_NUM> outQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> gradOutQueue;
    GlobalTensor<float> gradOutputGm;
    GlobalTensor<float> gradInputGm;

    uint32_t n = 0;
    uint32_t c = 0;
    uint32_t hIn = 0;
    uint32_t wIn = 0;
    uint32_t hOut = 0;
    uint32_t wOut = 0;
    uint32_t kernelH = 0;
    uint32_t kernelW = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t padTop = 0;
    uint32_t padLeft = 0;
    uint32_t countIncludePad = 1;
    int32_t divisorOverride = 0;
    float invDivisor = 1.0f;
    uint32_t usedCoreNum = 1;
    uint32_t tileH = 1;
    uint32_t tileW = 1;
    uint32_t outTileStride = 1;
    uint32_t tileTaskNum = 0;
    uint32_t gradOutCacheDataNum = 0;
};

} // namespace NsAveragePoolingGrad

#endif // AVERAGE_POOLING_GRAD_TILED_BASE_H
