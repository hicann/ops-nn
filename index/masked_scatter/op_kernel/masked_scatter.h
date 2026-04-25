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
 * \file masked_scatter.h
 * \brief
 */
#ifndef __MASKED_SCATTER_H__
#define __MASKED_SCATTER_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "masked_scatter_base.h"
#include "masked_scatter_tiling_data.h"
#include "masked_scatter_tiling_key.h"

namespace MaskedScatterNS {

class MaskedScatter : public MaskedScatterBase {
public:
    __aicore__ inline MaskedScatter(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, const MaskedScatterV1TilingData* tilingData)
    {
        BaseMemberDataInit(tilingData);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(x) + xBlockOffset, xBlockLength);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(mask) + maskBlockOffset, maskBlockLength);
        updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(updates), totalUpdatesNum);
        preMaskGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(mask), totalPreMaskLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(y) + xBlockOffset, xBlockLength);
 
        pipe.InitBuffer(inQueuePreMask, BUFFER_NUM, alignedPreMaskTileLength * sizeof(bool));
        pipe.InitBuffer(inQueueMask, BUFFER_NUM, alignedMaskLengthB * sizeof(bool));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, alignedTotalUpdatesNum * sizeof(DTYPE_X));
        pipe.InitBuffer(maskFp32Buf, BUFFER_NUM * alignedMaskLengthFp32 * sizeof(float));
        pipe.InitBuffer(reduceWorkBuf, BUFFER_NUM * alignedMaskLengthFp32 * sizeof(float));
        pipe.InitBuffer(calInt32Buf, BUFFER_NUM * alignedMaskLengthFp32 * sizeof(int32_t));
        pipe.InitBuffer(maskLocalHfBuf, BUFFER_NUM * alignedMaskLengthHf * sizeof(half));
        pipe.InitBuffer(maskLocalFp32Buf, BUFFER_NUM * alignedMaskLengthFp32 * sizeof(float));
        if (updatesLineNum == 1) {
            pipe.InitBuffer(outQueueX, BUFFER_NUM, alignedTotalUpdatesNum * sizeof(DTYPE_X));
        }
    }

    __aicore__ inline void Process()
    {
        maskLocal = inQueueMask.AllocTensor<bool>();
        preMaskLocal = inQueuePreMask.AllocTensor<bool>();
        updatesLocal = outQueueY.AllocTensor<DTYPE_X>();
        if (updatesLineNum == 1) {
            xLocal = outQueueX.AllocTensor<DTYPE_X>();
        }
        PreparePreMaskIndex();
        uint32_t tempMaskOffset = 0;
        uint32_t updateOffset = updatesIndex * updatesLineNum;
        uint32_t tempMaskTileLength = maskTileLength;

        for (uint32_t i = 0; i < circleNum; i++) {
            if (i == circleNum - 1 && tailMaskTileLength != 0) {
                tempMaskTileLength = tailMaskTileLength;
            }
            CommonCopyIn<bool>(maskLocal, maskGm, tempMaskOffset, 1, tempMaskTileLength);
            if (updatesLineNum == 1) { 
                PipeBarrier<PIPE_ALL>();
                CommonCopyIn<DTYPE_X>(xLocal, xGm, tempMaskOffset, 1, tempMaskTileLength);
            }
            PipeBarrier<PIPE_ALL>();
            Compute(updateOffset, tempMaskTileLength);
            PipeBarrier<PIPE_ALL>();
            CopyOut(tempMaskOffset, tempMaskTileLength);
            PipeBarrier<PIPE_ALL>();
            tempMaskOffset += tempMaskTileLength;
        }
        inQueueMask.FreeTensor(maskLocal);
        inQueuePreMask.FreeTensor(preMaskLocal);
        outQueueY.FreeTensor(updatesLocal);
        if (updatesLineNum == 1) {
            outQueueX.FreeTensor(xLocal);
        }
    }

private:
    __aicore__ inline void BaseMemberDataInit(const MaskedScatterV1TilingData* tilingData)
    {
        loopNum = tilingData->loopNum;
        remainNum = tilingData->remainNum;
        updatesLineNum = tilingData->updatesLineNum;
        maskTileLength = tilingData->maskTileLength;
        totalUpdatesNum = tilingData->totalUpdatesNum;

        if (updatesLineNum == 1) {
            maskTileLength = 1024;
        }

        PrepareBaseData();

        // prepare mask compute
        circleNum = CeilDiv(maskBlockLength, maskTileLength);
        tailMaskTileLength = maskBlockLength % maskTileLength;

        uint32_t maxMaskComputeNum = maskTileLength >= preMaskComputeNum ? maskTileLength : preMaskComputeNum;
        alignedPreMaskTileLength = CeilAlign(preMaskComputeNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(bool)));
        alignedUpdatesLineNum = CeilAlign(updatesLineNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(DTYPE_X)));
        alignedMaskLengthB = CeilAlign(maskTileLength, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(bool)));
        alignedMaskLengthFp32 = CeilAlign(maxMaskComputeNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(float)));
        alignedMaskLengthHf = CeilAlign(maxMaskComputeNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(half)));
        alignedTotalUpdatesNum = CeilAlign(maskTileLength * updatesLineNum, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(DTYPE_X)));
    }

    __aicore__ inline void Compute(uint32_t& updateOffset, uint32_t maskCount)
    {
        uint32_t maskTrueCount = ComputeMaskTrueCount(maskLocal, maskCount);
        PipeBarrier<PIPE_ALL>();
        uint32_t tempUpdateLength = updatesLineNum * maskTrueCount;
        if (updatesLineNum == 1) {
            CommonCopyIn<DTYPE_X>(updatesLocal, updatesGm, updateOffset, 1, maskTrueCount * updatesLineNum);
        } else {
            CommonCopyIn<DTYPE_X>(updatesLocal, updatesGm, updateOffset, maskTrueCount, updatesLineNum);
        }
        
        PipeBarrier<PIPE_ALL>();
        updateOffset = updateOffset + tempUpdateLength;
    }

    __aicore__ inline void CopyOut(uint32_t maskOffset, uint32_t maskCount)
    {
        if (updatesLineNum == 1) {
            const uint32_t LOOP_SIZE = 8;
            DTYPE_X updatesCache[LOOP_SIZE];
            uint64_t loopNum = CeilDiv(maskCount, LOOP_SIZE);
            auto maskUbAddress = reinterpret_cast<__ubuf__ bool *>(maskLocal.GetPhyAddr(0));
            auto xUbAddress = reinterpret_cast<__ubuf__ DTYPE_X *>(xLocal.GetPhyAddr(0));
            auto updatesUbAddress = reinterpret_cast<__ubuf__ DTYPE_X *>(updatesLocal.GetPhyAddr(0));
            uint32_t index = 0;
            uint32_t xOffset = 0;
            for(uint32_t i = 0; i < loopNum; i++) {
                for (uint32_t j = 0; j < LOOP_SIZE; j++) {
                    bool isUpdate = *(maskUbAddress + (i * LOOP_SIZE + j));
                    if (isUpdate) {
                        updatesCache[j] = *(updatesUbAddress + index);
                        index++;
                    } else {
                        updatesCache[j] = *(xUbAddress + (i * LOOP_SIZE + j));
                    }
                }
                for (uint32_t j = 0; j < LOOP_SIZE; j++) {
                    *(xUbAddress + xOffset) = updatesCache[j];
                    xOffset++;
                }
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams outCopyOutParams;
            outCopyOutParams.blockCount = 1;
            outCopyOutParams.blockLen = maskCount * sizeof(DTYPE_X);
            outCopyOutParams.dstStride = 0;
            outCopyOutParams.srcStride = 0;
            outCopyOutParams.rsv = 0;
            DataCopyPad(yGm[maskOffset], xLocal, outCopyOutParams);
        } else {
            DataCopyExtParams outCopyOutParams;
            outCopyOutParams.blockCount = 1;
            outCopyOutParams.blockLen = updatesLineNum * sizeof(DTYPE_X);
            outCopyOutParams.dstStride = 0;
            outCopyOutParams.srcStride = 0;
            outCopyOutParams.rsv = 0;

            uint32_t index = 0;
            for (uint32_t i = 0; i < maskCount; i++) {
                if (maskLocal.GetValue(i)) {
                    DataCopyPad(yGm[(maskOffset + i) * updatesLineNum], updatesLocal[index * alignedUpdatesLineNum], outCopyOutParams);
                    index++;
                }
            }
        }
    }

private:
    uint32_t maskTileLength, alignedTotalUpdatesNum;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueX;
    LocalTensor<DTYPE_X> xLocal;
};

} // namespace MaskedScatterNS
#endif // MASKED_SCATTER_H