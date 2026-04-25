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
 * \file masked_scatter_base.h
 * \brief tiling data struct
 */

#ifndef __MASKED_SCATTER_BASE_H__
#define __MASKED_SCATTER_BASE_H__

#include "kernel_operator.h"
namespace MaskedScatterNS {

using namespace AscendC;

constexpr uint32_t ONE_BLK_SIZE = 32;

constexpr uint32_t BUFFER_NUM = 2;

constexpr uint32_t MAX_MASK_CYCLE_NUM = 1600;

constexpr uint32_t MAX_UPDATES_CYCLE_LEN = 60000;

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

class MaskedScatterBase {
public:
    __aicore__ inline MaskedScatterBase(){};

protected:
    __aicore__ inline void PrepareBaseData()
    {
        uint32_t blockIdx = GetBlockIdx();
        if (remainNum == 0) {
            totalPreMaskLength = loopNum * blockIdx;
            maskBlockLength = loopNum;
            maskBlockOffset = loopNum * blockIdx;
            xBlockLength = loopNum * updatesLineNum;
            xBlockOffset = loopNum * blockIdx * updatesLineNum;
        } else {
            totalPreMaskLength = blockIdx <= remainNum ? (loopNum + 1) * blockIdx : loopNum * blockIdx + remainNum;
            maskBlockLength = blockIdx < remainNum ? loopNum + 1 : loopNum;
            maskBlockOffset = blockIdx <= remainNum ? (loopNum + 1) * blockIdx : loopNum * blockIdx + remainNum;
            xBlockLength = blockIdx < remainNum ? (loopNum + 1) * updatesLineNum : loopNum * updatesLineNum;
            xBlockOffset = blockIdx <= remainNum ? (loopNum + 1) * blockIdx * updatesLineNum : (loopNum * blockIdx + remainNum) * updatesLineNum;
        }

        // prepare preMask compute
        if (totalPreMaskLength < MAX_MASK_CYCLE_NUM) {
            preMaskComputeNum = totalPreMaskLength == 0 ? 1 : totalPreMaskLength;
            preMaskLoopNum = 1;
            tailPreMaskNum = 0;
        } else {
            preMaskComputeNum = MAX_MASK_CYCLE_NUM;
            preMaskLoopNum = CeilDiv(totalPreMaskLength, MAX_MASK_CYCLE_NUM);
            tailPreMaskNum = totalPreMaskLength % MAX_MASK_CYCLE_NUM;
        }
    }

    template <typename T>
    __aicore__ inline void CommonCopyIn(LocalTensor<T>& selfLocal, GlobalTensor<T>& selfGm, uint32_t offset, uint16_t blockCount, uint32_t blockLen)
    {
        DataCopyExtParams copyParams{static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(selfLocal, selfGm[offset], copyParams, padParams);
    }

    __aicore__ inline uint32_t ComputeMaskTrueCount(LocalTensor<bool>& maskLocalTensor, uint32_t calCount)
    {
        LocalTensor<float> maskFp32Temp = maskFp32Buf.Get<float>();
        LocalTensor<float> reduceLocalBuf = reduceWorkBuf.Get<float>();
        LocalTensor<half> maskLocalHfTensor = maskLocalHfBuf.Get<half>();
        LocalTensor<float> maskLocalFp32Tensor = maskLocalFp32Buf.Get<float>();
        LocalTensor<int32_t> calInt32Temp = calInt32Buf.Get<int32_t>();

        LocalTensor<uint8_t> maskLocalUint8Tensor = maskLocalTensor.ReinterpretCast<uint8_t>();
        Cast(maskLocalHfTensor, maskLocalUint8Tensor, RoundMode::CAST_NONE, calCount);
        Cast(maskLocalFp32Tensor, maskLocalHfTensor, RoundMode::CAST_NONE, calCount);
        PipeBarrier<PIPE_ALL>();
        ReduceSum<float>(maskFp32Temp, maskLocalFp32Tensor, reduceLocalBuf, calCount);
        PipeBarrier<PIPE_ALL>();
        Cast(calInt32Temp, maskFp32Temp, RoundMode::CAST_RINT, calCount);
        PipeBarrier<PIPE_ALL>();

        return static_cast<uint32_t>(calInt32Temp.GetValue(0));
    }

    __aicore__ inline void PreparePreMaskIndex()
    {
        if (totalPreMaskLength == 0) {
            return;
        }

        uint32_t offset = 0;
        uint32_t tempMaskTileLength = preMaskComputeNum;
        for (uint32_t i = 0; i < preMaskLoopNum; i++) {
            if (i == preMaskLoopNum - 1 && tailPreMaskNum != 0) {
                tempMaskTileLength = tailPreMaskNum;
            }
            CommonCopyIn<bool>(preMaskLocal, preMaskGm, offset, 1, tempMaskTileLength);
            PipeBarrier<PIPE_ALL>();
            uint32_t preMaskTrueCount = ComputeMaskTrueCount(preMaskLocal, tempMaskTileLength);
            offset += tempMaskTileLength;
            updatesIndex += preMaskTrueCount;
        }
    }

protected:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePreMask;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueMask;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> maskFp32Buf, reduceWorkBuf, calInt32Buf, maskLocalHfBuf, maskLocalFp32Buf;

    GlobalTensor<DTYPE_X> xGm, updatesGm, yGm;
    GlobalTensor<bool> maskGm, preMaskGm;

    LocalTensor<bool> maskLocal;
    LocalTensor<bool> preMaskLocal;
    LocalTensor<DTYPE_X> updatesLocal;

    uint32_t xBlockOffset, xBlockLength, maskBlockOffset, maskBlockLength, alignedUpdatesLineNum, alignedMaskLengthB,
        alignedMaskLengthFp32, alignedMaskLengthHf, circleNum, loopNum, remainNum, alignedPreMaskTileLength, totalUpdatesNum,
        preMaskLoopNum, preMaskComputeNum, tailPreMaskNum, tailMaskTileLength, updatesLineNum, totalPreMaskLength;
    uint32_t updatesIndex = 0;
};

} // namespace MaskedScatterNS
#endif