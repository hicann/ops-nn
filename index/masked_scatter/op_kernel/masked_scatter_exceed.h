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
 * \file masked_scatter_exceed.h
 * \brief
 */
#ifndef __MASKED_SCATTER_EXCEED_H__
#define __MASKED_SCATTER_EXCEED_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "masked_scatter_base.h"
#include "masked_scatter_tiling_data.h"
#include "masked_scatter_tiling_key.h"

namespace MaskedScatterNS {

class MaskedScatterExceed : public MaskedScatterBase {
public:
    __aicore__ inline MaskedScatterExceed(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, const MaskedScatterV1TilingData* tilingData)
    {
        BaseMemberDataInit(tilingData);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(x) + xBlockOffset, xBlockLength);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(mask) + maskBlockOffset, maskBlockLength);
        updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(updates), totalUpdatesNum);
        preMaskGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(mask), totalPreMaskLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(y) + xBlockOffset, xBlockLength);

        pipe.InitBuffer(outQueueY, BUFFER_NUM, MAX_UPDATES_CYCLE_LEN);
        pipe.InitBuffer(inQueueMask, BUFFER_NUM, alignedMaskLengthB * sizeof(bool));
        pipe.InitBuffer(inQueuePreMask, BUFFER_NUM, alignedPreMaskTileLength * sizeof(bool));
        pipe.InitBuffer(maskFp32Buf, BUFFER_NUM * alignedMaskLengthFp32 * sizeof(float));
        pipe.InitBuffer(reduceWorkBuf, BUFFER_NUM * alignedMaskLengthFp32 * sizeof(float));
        pipe.InitBuffer(calInt32Buf, BUFFER_NUM * alignedMaskLengthFp32 * sizeof(int32_t));
        pipe.InitBuffer(maskLocalHfBuf, BUFFER_NUM * alignedMaskLengthHf * sizeof(half));
        pipe.InitBuffer(maskLocalFp32Buf, BUFFER_NUM * alignedMaskLengthFp32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        maskLocal = inQueueMask.AllocTensor<bool>();
        preMaskLocal = inQueuePreMask.AllocTensor<bool>();
        updatesLocal = outQueueY.AllocTensor<DTYPE_X>();
        PreparePreMaskIndex();
        uint32_t tempMaskOffset = 0;
        uint32_t updateOffset = updatesIndex * updatesLineNum;
        uint32_t tempMaskTileLength = MAX_MASK_CYCLE_NUM;

        for (uint32_t i = 0; i < circleNum; i++) {
            if (i == circleNum - 1 && tailMaskTileLength != 0) {
                tempMaskTileLength = tailMaskTileLength;
            }
            Compute(updateOffset, tempMaskOffset, tempMaskTileLength);
            PipeBarrier<PIPE_ALL>();
        }
        inQueueMask.FreeTensor(maskLocal);
        inQueuePreMask.FreeTensor(preMaskLocal);
        outQueueY.FreeTensor(updatesLocal);
    }

private:
    __aicore__ inline void BaseMemberDataInit(const MaskedScatterV1TilingData* tilingData)
    {
        loopNum = tilingData->loopNum;
        remainNum = tilingData->remainNum;
        updatesLineNum = tilingData->updatesLineNum;
        totalUpdatesNum = tilingData->totalUpdatesNum;
        updatesNum = tilingData->updatesNum;

        PrepareBaseData();

        // prepare mask compute
        circleNum = CeilDiv(maskBlockLength, MAX_MASK_CYCLE_NUM);
        tailMaskTileLength = maskBlockLength % MAX_MASK_CYCLE_NUM;

        // prepare updates compute
        if (updatesLineNum * sizeof(DTYPE_X) <= MAX_UPDATES_CYCLE_LEN) {
            updatesLoopNum = 1;
            updatesComputeNum = updatesLineNum;
            tailUpdatesComputeNum = 0;
        } else {
            updatesComputeNum = MAX_UPDATES_CYCLE_LEN / sizeof(DTYPE_X);
            updatesLoopNum = CeilDiv(updatesLineNum, updatesComputeNum);
            tailUpdatesComputeNum = updatesLineNum % updatesComputeNum;
        }

        uint32_t maxMaskComputeNum = MAX_MASK_CYCLE_NUM >= preMaskComputeNum ? MAX_MASK_CYCLE_NUM : preMaskComputeNum;
        alignedPreMaskTileLength = CeilAlign(preMaskComputeNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(bool)));
        alignedUpdatesLineNum = CeilAlign(updatesLineNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(DTYPE_X)));
        alignedMaskLengthB = CeilAlign(MAX_MASK_CYCLE_NUM, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(bool)));
        alignedMaskLengthFp32 = CeilAlign(maxMaskComputeNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(float)));
        alignedMaskLengthHf = CeilAlign(maxMaskComputeNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(half)));
    }

    __aicore__ inline void Compute(uint32_t& updateOffset, uint32_t& maskOffset, uint32_t maskTileNum)
    {
        if (updatesIndex >= updatesNum) {
            return;
        }
        remainUpdates = updatesNum - updatesIndex;
        uint32_t index = 0;
        CommonCopyIn<bool>(maskLocal, maskGm, maskOffset, 1, maskTileNum);
        for (uint32_t i = 0; i < maskTileNum; i++) {
            if (index >= remainUpdates) {
                break;
            }
            if (maskLocal.GetValue(i)) {
                uint32_t tempUpdatesComputeNum = updatesComputeNum;
                uint32_t tempUpdatesOffset = 0;
                for (uint32_t j = 0; j < updatesLoopNum; j++) {
                    if (j == updatesLoopNum - 1 && tailUpdatesComputeNum != 0) {
                        tempUpdatesComputeNum = tailUpdatesComputeNum;
                    }
                    CommonCopyIn<DTYPE_X>(updatesLocal, updatesGm, updateOffset + tempUpdatesOffset, 1, tempUpdatesComputeNum);
                    PipeBarrier<PIPE_ALL>();
                    CopyOut(maskOffset, maskTileNum, tempUpdatesComputeNum, tempUpdatesOffset);
                    PipeBarrier<PIPE_ALL>();
                    tempUpdatesOffset += tempUpdatesComputeNum;
                }
                updateOffset += updatesLineNum;
                index++;
                updatesIndex++;
            }
            maskOffset++;
        }
    }

    __aicore__ inline void CopyOut(uint32_t maskOffset, uint32_t maskCount, uint32_t updatesCopyNum, uint32_t updatesOffset)
    {
        DataCopyExtParams outCopyOutParams;
        outCopyOutParams.blockCount = 1;
        outCopyOutParams.blockLen = updatesCopyNum * sizeof(DTYPE_X);
        outCopyOutParams.dstStride = 0;
        outCopyOutParams.srcStride = 0;
        outCopyOutParams.rsv = 0;
        DataCopyPad(yGm[maskOffset * updatesLineNum + updatesOffset], updatesLocal, outCopyOutParams);
    }

private:
    uint32_t updatesLoopNum, updatesComputeNum, tailUpdatesComputeNum;
};

} // namespace MaskedScatterNS
#endif // MASKED_SCATTER_EXCEED_H