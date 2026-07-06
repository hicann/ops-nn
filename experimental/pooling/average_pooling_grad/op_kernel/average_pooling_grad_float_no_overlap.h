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

#ifndef AVERAGE_POOLING_GRAD_FLOAT_NO_OVERLAP_H
#define AVERAGE_POOLING_GRAD_FLOAT_NO_OVERLAP_H

#include "average_pooling_grad_tiled_base.h"

namespace NsAveragePoolingGrad {

class AveragePoolingGradFloatNoOverlap : public AveragePoolingGradTiledBase {
public:
    __aicore__ inline AveragePoolingGradFloatNoOverlap() {}
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
    __aicore__ inline void ComputeGradInput(const TileTask& task, const OutputCacheRange& range)
    {
        LocalTensor<float> gradOutLocal = gradOutQueue.DeQue<float>();
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

        const uint32_t tileRows = task.hEnd - task.hStart;
        const uint32_t tileCols = task.wEnd - task.wStart;
        Duplicate<float>(outLocal, 0.0f, tileRows * outTileStride);

        const float invDiv = invDivisor;
        const int32_t hStartS = static_cast<int32_t>(task.hStart);
        const int32_t wStartS = static_cast<int32_t>(task.wStart);
        const int32_t strideHS = static_cast<int32_t>(strideH);
        const int32_t strideWS = static_cast<int32_t>(strideW);
        const int32_t ohStartS = range.ohStart;
        const int32_t owStartS = range.owStart;

        for (uint32_t ohLocal = 0; ohLocal < range.cacheH; ++ohLocal) {
            const int32_t oh = ohStartS + static_cast<int32_t>(ohLocal);
            const int32_t ihBase = oh * strideHS - hStartS;
            const uint32_t cacheRow = ohLocal * range.cacheStride;
            Muls<float>(gradOutLocal[cacheRow], gradOutLocal[cacheRow], invDiv, range.cacheW);

            for (uint32_t owLocal = 0; owLocal < range.cacheW; ++owLocal) {
                const float val = gradOutLocal.GetValue(cacheRow + owLocal);
                const int32_t iwBase = (owStartS + static_cast<int32_t>(owLocal)) * strideWS - wStartS;
                for (uint32_t kh = 0; kh < kernelH; ++kh) {
                    const int32_t ihLocal = ihBase + static_cast<int32_t>(kh);
                    if (ihLocal < 0 || ihLocal >= static_cast<int32_t>(tileRows)) {
                        continue;
                    }
                    const uint32_t outRow = static_cast<uint32_t>(ihLocal) * outTileStride;
                    for (uint32_t kw = 0; kw < kernelW; ++kw) {
                        const int32_t iwLocal = iwBase + static_cast<int32_t>(kw);
                        if (iwLocal < 0 || iwLocal >= static_cast<int32_t>(tileCols)) {
                            continue;
                        }
                        outLocal.SetValue(outRow + static_cast<uint32_t>(iwLocal), val);
                    }
                }
            }
        }

        outQueue.EnQue(outLocal);
        gradOutQueue.FreeTensor(gradOutLocal);
    }
};

} // namespace NsAveragePoolingGrad

#endif // AVERAGE_POOLING_GRAD_FLOAT_NO_OVERLAP_H
