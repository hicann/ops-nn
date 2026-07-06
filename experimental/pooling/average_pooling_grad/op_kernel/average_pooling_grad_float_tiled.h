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

#ifndef AVERAGE_POOLING_GRAD_FLOAT_TILED_H
#define AVERAGE_POOLING_GRAD_FLOAT_TILED_H

#include "average_pooling_grad_tiled_base.h"

namespace NsAveragePoolingGrad {

class AveragePoolingGradFloatTiled : public AveragePoolingGradTiledBase {
public:
    __aicore__ inline AveragePoolingGradFloatTiled() {}
    using AveragePoolingGradTiledBase::Init;

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        for (uint32_t taskId = blockIdx; taskId < tileTaskNum; taskId += usedCoreNum) {
            const TileTask task = DecodeTileTask(taskId);
            const OutputCacheRange range = CalcOutputCacheRange(task);
            if (range.cacheH == 0 || range.cacheW == 0) {
                FillZeroGradInputTile(task);
                continue;
            }
            CopyGradOutputCache(task, range);
            ComputeGradInput(task, range);
            CopyGradInputTileOut(task);
        }
    }

private:
    __aicore__ inline void CalcCoverRange(uint32_t ih, uint32_t iw, int32_t& ohStart, int32_t& ohEnd,
        int32_t& owStart, int32_t& owEnd) const
    {
        ohStart = PStart(static_cast<int32_t>(ih), padTop, kernelH, strideH);
        ohEnd = PEnd(static_cast<int32_t>(ih), padTop, strideH);
        owStart = PStart(static_cast<int32_t>(iw), padLeft, kernelW, strideW);
        owEnd = PEnd(static_cast<int32_t>(iw), padLeft, strideW);
        ohStart = ClampStart(ohStart);
        owStart = ClampStart(owStart);
        ohEnd = ClampEnd(ohEnd, static_cast<int32_t>(hOut));
        owEnd = ClampEnd(owEnd, static_cast<int32_t>(wOut));
    }

    __aicore__ inline void PreDivideInPlace(LocalTensor<float>& gradOutLocal, const OutputCacheRange& range) const
    {
        for (uint32_t ohLocal = 0; ohLocal < range.cacheH; ++ohLocal) {
            for (uint32_t owLocal = 0; owLocal < range.cacheW; ++owLocal) {
                const int32_t oh = range.ohStart + static_cast<int32_t>(ohLocal);
                const int32_t ow = range.owStart + static_cast<int32_t>(owLocal);
                const int32_t divisor = CalcDivisor(oh, ow);
                if (divisor <= 0) {
                    continue;
                }
                const uint32_t offset = ohLocal * range.cacheStride + owLocal;
                const float val = gradOutLocal.GetValue(offset) / static_cast<float>(divisor);
                gradOutLocal.SetValue(offset, val);
            }
        }
    }

    __aicore__ inline float Accumulate(LocalTensor<float>& gradOutLocal, const OutputCacheRange& range,
        uint32_t ih, uint32_t iw) const
    {
        int32_t ohStart = 0;
        int32_t ohEnd = 0;
        int32_t owStart = 0;
        int32_t owEnd = 0;
        CalcCoverRange(ih, iw, ohStart, ohEnd, owStart, owEnd);
        float acc = 0.0f;
        for (int32_t oh = ohStart; oh < ohEnd; ++oh) {
            if (oh < range.ohStart || oh >= range.ohEnd) {
                continue;
            }
            const uint32_t rowBase = static_cast<uint32_t>(oh - range.ohStart) * range.cacheStride;
            for (int32_t ow = owStart; ow < owEnd; ++ow) {
                if (ow < range.owStart || ow >= range.owEnd) {
                    continue;
                }
                acc += gradOutLocal.GetValue(rowBase + static_cast<uint32_t>(ow - range.owStart));
            }
        }
        return acc;
    }

    __aicore__ inline void ComputeGradInput(const TileTask& task, const OutputCacheRange& range)
    {
        LocalTensor<float> gradOutLocal = gradOutQueue.DeQue<float>();
        PreDivideInPlace(gradOutLocal, range);

        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        const uint32_t tileRows = task.hEnd - task.hStart;
        Duplicate<float>(outLocal, 0.0f, tileRows * outTileStride);
        for (uint32_t ih = task.hStart; ih < task.hEnd; ++ih) {
            const uint32_t outRow = (ih - task.hStart) * outTileStride;
            for (uint32_t iw = task.wStart; iw < task.wEnd; ++iw) {
                outLocal.SetValue(outRow + iw - task.wStart, Accumulate(gradOutLocal, range, ih, iw));
            }
        }
        outQueue.EnQue(outLocal);
        gradOutQueue.FreeTensor(gradOutLocal);
    }
};

} // namespace NsAveragePoolingGrad

#endif // AVERAGE_POOLING_GRAD_FLOAT_TILED_H
