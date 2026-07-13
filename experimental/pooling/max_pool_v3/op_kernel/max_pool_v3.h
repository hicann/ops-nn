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
 * \file max_pool_v3.h
 * \brief max_pool_v3 kernel implementation
 */
#ifndef __MAX_POOL_V3_H__
#define __MAX_POOL_V3_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool_v3_tiling_data.h"
#include "max_pool_v3_tiling_key.h"

namespace NsMaxPoolV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = ::MAX_POOL_V3_BUFFER_NUM;

template <typename T>
class KernelMaxPoolV3 {
public:
    __aicore__ inline KernelMaxPoolV3() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum, uint32_t tailBlockNum, uint32_t n,
                                uint32_t c, uint32_t hIn, uint32_t wIn, uint32_t hOut, uint32_t wOut, uint32_t kH,
                                uint32_t kW, uint32_t sH, uint32_t sW, uint32_t padT, uint32_t padL,
                                uint32_t windowSize)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreIdx = GetBlockIdx();

        this->nVal = n;
        this->cVal = c;
        this->hIn = hIn;
        this->wIn = wIn;
        this->hOut = hOut;
        this->wOut = wOut;
        this->kH = kH;
        this->kW = kW;
        this->sH = sH;
        this->sW = sW;
        this->padT = padT;
        this->padL = padL;
        this->inHW = hIn * wIn;
        this->outHW = hOut * wOut;
        this->windowSize = windowSize;

        uint64_t globalBufferIndex = static_cast<uint64_t>(bigCoreDataNum) * coreIdx;
        if (coreIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= static_cast<uint64_t>(bigCoreDataNum - smallCoreDataNum) * (coreIdx - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

        this->globalOutOffset = globalBufferIndex;

        this->alignedWindowSize = this->windowSize;
        uint32_t bufBytes = this->windowSize * sizeof(T);
        if (bufBytes < 32) {
            this->alignedWindowSize = 32 / sizeof(T);
            bufBytes = 32;
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufBytes);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, sizeof(T));

        pipe.InitBuffer(calcBufIn, this->alignedWindowSize * sizeof(float));
        pipe.InitBuffer(calcBufOut, sizeof(float));

        this->tileDataNum = tileDataNum;
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        if (loopCount == 0) {
            return;
        }
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline T GetNegInf()
    {
        if constexpr (std::is_same_v<T, half>) {
            constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
            return *reinterpret_cast<const half*>(&FLOAT16_NEG_INF);
        } else if constexpr (std::is_same_v<T, float>) {
            constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
            return *reinterpret_cast<const float*>(&FLOAT32_NEG_INF);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
            return *reinterpret_cast<const bfloat16_t*>(&BFLOAT16_NEG_INF);
        } else {
            static_assert(sizeof(T) != sizeof(T),
                          "GetNegInf: unsupported dtype, only half/float/bfloat16_t are supported");
        }
    }

    __aicore__ inline void CalcWindow(int32_t outIdx, int32_t& curkH, int32_t& curkW, int32_t& curInOffset)
    {
        int32_t nc = outIdx / static_cast<int32_t>(this->outHW);
        int32_t spatialIdx = outIdx % static_cast<int32_t>(this->outHW);
        int32_t ho = spatialIdx / static_cast<int32_t>(this->wOut);
        int32_t wo = spatialIdx % static_cast<int32_t>(this->wOut);

        int32_t hStart = ho * static_cast<int32_t>(this->sH) - static_cast<int32_t>(this->padT);
        int32_t wStart = wo * static_cast<int32_t>(this->sW) - static_cast<int32_t>(this->padL);

        int32_t hStartClamped = (hStart < 0) ? 0 : hStart;
        int32_t wStartClamped = (wStart < 0) ? 0 : wStart;

        if (hStart < 0) {
            curkH = static_cast<int32_t>(this->kH) + hStart;
        } else {
            curkH = static_cast<int32_t>(this->kH);
        }
        if (hStartClamped + curkH > static_cast<int32_t>(this->hIn)) {
            curkH = static_cast<int32_t>(this->hIn) - hStartClamped;
        }
        if (curkH < 0)
            curkH = 0;

        if (wStart < 0) {
            curkW = static_cast<int32_t>(this->kW) + wStart;
        } else {
            curkW = static_cast<int32_t>(this->kW);
        }
        if (wStartClamped + curkW > static_cast<int32_t>(this->wIn)) {
            curkW = static_cast<int32_t>(this->wIn) - wStartClamped;
        }
        if (curkW < 0)
            curkW = 0;

        curInOffset = nc * static_cast<int32_t>(this->inHW) + hStartClamped * static_cast<int32_t>(this->wIn) +
                      wStartClamped;
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        int32_t outIdx = static_cast<int32_t>(this->globalOutOffset) + progress;
        int32_t curkH = 0, curkW = 0, curInOffset = 0;
        CalcWindow(outIdx, curkH, curkW, curInOffset);

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        T negInf = GetNegInf();
        Duplicate(xLocal, negInf, this->alignedWindowSize);

        for (int32_t h = 0; h < curkH; h++) {
            int32_t rowOffset = curInOffset + h * static_cast<int32_t>(this->wIn);
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curkW * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal[h * curkW], xGm[rowOffset], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        (void)progress;
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> xFp32 = calcBufIn.Get<float>();
        LocalTensor<float> yFp32 = calcBufOut.Get<float>();

        if constexpr (std::is_same_v<T, float>) {
            DataCopy(xFp32, xLocal, this->windowSize);
            WholeReduceMax<float, false>(yFp32, xFp32, this->windowSize, 1, 1, 1, 1);
            DataCopy(yLocal, yFp32, 1);
        } else {
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, this->windowSize);
            WholeReduceMax<float, false>(yFp32, xFp32, this->windowSize, 1, 1, 1, 1);
            Cast(yLocal, yFp32, RoundMode::CAST_ROUND, 1);
        }

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyExtParams copyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPad(yGm[progress], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBufIn;
    TBuf<QuePosition::VECCALC> calcBufOut;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t coreDataNum, tileNum, tileDataNum, tailDataNum;
    uint64_t globalOutOffset;
    uint32_t nVal, cVal, hIn, wIn, hOut, wOut;
    uint32_t kH, kW, sH, sW;
    uint32_t padT, padL;
    uint32_t inHW, outHW;
    uint32_t windowSize;
    uint32_t alignedWindowSize;
};

} // namespace NsMaxPoolV3
#endif // __MAX_POOL_V3_H__
