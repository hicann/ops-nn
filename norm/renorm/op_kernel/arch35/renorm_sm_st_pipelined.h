/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_SM_ST_PIPELINED_H_
#define _RENORM_SM_ST_PIPELINED_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"

namespace NsRenormSmStPipelined {

using namespace AscendC;

constexpr int64_t CMP_ALIGN = 8;

// This isolated FP32 path handles the long-slice/small-R envelope.  Unlike
// Template D, it launches the next GM load before processing the current tile.
class RenormSmStPipelined {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> dataBuf0;
    TBuf<QuePosition::VECCALC> dataBuf1;
    TBuf<QuePosition::VECCALC> normBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> maxNormBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    GlobalTensor<float> inputGM;
    GlobalTensor<float> outputGM;
    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t numBlocks_ = 0;
    int64_t sliceTileLength_ = 0;
    int64_t slicesPerCore_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
};

__aicore__ inline void RenormSmStPipelined::Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData)
{
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    numBlocks_ = tilingData->numBlocks;
    sliceTileLength_ = tilingData->sliceTileLength;
    slicesPerCore_ = tilingData->slicesPerCore;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    if (totalElements_ == 0 || sliceCount_ == 0 || numBlocks_ == 0 || sliceTileLength_ == 0) {
        return;
    }

    inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), totalElements_);
    outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), totalElements_);
    int64_t alignedTile = (sliceTileLength_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
    pipe.InitBuffer(dataBuf0, alignedTile * sizeof(float));
    pipe.InitBuffer(dataBuf1, alignedTile * sizeof(float));
    pipe.InitBuffer(normBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(scaleBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(maskBuf, alignedTile);
    pipe.InitBuffer(zerosBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(onesBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(maxNormBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(tmpBuf, alignedTile * sizeof(float));
}

__aicore__ inline void RenormSmStPipelined::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || numBlocks_ == 0 || sliceTileLength_ == 0) {
        return;
    }
    int64_t startSlice = GetBlockIdx() * slicesPerCore_;
    int64_t endSlice = startSlice + slicesPerCore_;
    if (endSlice > sliceCount_) {
        endSlice = sliceCount_;
    }

    LocalTensor<float> dataLocal0 = dataBuf0.Get<float>();
    LocalTensor<float> dataLocal1 = dataBuf1.Get<float>();
    LocalTensor<float> normLocal = normBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<float> maxNormLocal = maxNormBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    for (int64_t sliceTile = startSlice; sliceTile < endSlice; sliceTile += sliceTileLength_) {
        int64_t currentTile = (sliceTileLength_ < endSlice - sliceTile) ? sliceTileLength_ : endSlice - sliceTile;
        int64_t alignedLen = (currentTile + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
        int64_t padLen = alignedLen - currentTile;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentTile * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams = {true, 0, static_cast<uint8_t>(padLen), 0.0f};

        Duplicate(normLocal, 0.0f, static_cast<int32_t>(alignedLen));
        Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(alignedLen));
        Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedLen));
        Duplicate(maxNormLocal, maxNorm_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        TEventID loadReady[2] = {GetTPipePtr()->FetchEventID(HardEvent::MTE2_V),
                                 GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)};
        TEventID dataFree[2] = {GetTPipePtr()->FetchEventID(HardEvent::V_MTE2),
                                GetTPipePtr()->FetchEventID(HardEvent::V_MTE2)};
        DataCopyPad(dataLocal0, inputGM[sliceTile], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(loadReady[0]);

        for (int64_t b = 0; b < numBlocks_; ++b) {
            int64_t currentBuffer = b & 1;
            int64_t nextBuffer = 1 - currentBuffer;
            LocalTensor<float>& currentData = currentBuffer == 0 ? dataLocal0 : dataLocal1;
            LocalTensor<float>& nextData = nextBuffer == 0 ? dataLocal0 : dataLocal1;
            WaitFlag<HardEvent::MTE2_V>(loadReady[currentBuffer]);

            if (b + 1 < numBlocks_) {
                if (b >= 1) {
                    WaitFlag<HardEvent::V_MTE2>(dataFree[nextBuffer]);
                }
                DataCopyPad(nextData, inputGM[(b + 1) * sliceCount_ + sliceTile], copyParams, padParams);
                SetFlag<HardEvent::MTE2_V>(loadReady[nextBuffer]);
            }

            Abs(currentData, currentData, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            if (p_ == 2.0f) {
                Mul(currentData, currentData, currentData, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                Maxs(currentData, currentData, eps_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Log(currentData, currentData, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Muls(currentData, currentData, p_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Exp(currentData, currentData, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            }
            Add(normLocal, normLocal, currentData, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(dataFree[currentBuffer]);
        }
        WaitFlag<HardEvent::V_MTE2>(dataFree[(numBlocks_ - 1) & 1]);
        if (numBlocks_ > 1) {
            WaitFlag<HardEvent::V_MTE2>(dataFree[(numBlocks_ - 2) & 1]);
        }

        if (p_ == 2.0f) {
            Sqrt(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        } else if (p_ != 1.0f) {
            Maxs(normLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(normLocal, normLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }
        Maxs(tmpLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Reciprocal(tmpLocal, tmpLocal, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, tmpLocal, maxNorm_, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, normLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();

        TEventID outputReady[2] = {GetTPipePtr()->FetchEventID(HardEvent::V_MTE3),
                                   GetTPipePtr()->FetchEventID(HardEvent::V_MTE3)};
        TEventID storeDone[2] = {GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2),
                                 GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2)};
        DataCopyPad(dataLocal0, inputGM[sliceTile], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(loadReady[0]);

        for (int64_t b = 0; b < numBlocks_; ++b) {
            int64_t currentBuffer = b & 1;
            int64_t nextBuffer = 1 - currentBuffer;
            LocalTensor<float>& currentData = currentBuffer == 0 ? dataLocal0 : dataLocal1;
            LocalTensor<float>& nextData = nextBuffer == 0 ? dataLocal0 : dataLocal1;
            WaitFlag<HardEvent::MTE2_V>(loadReady[currentBuffer]);

            if (b + 1 < numBlocks_) {
                if (b >= 1) {
                    WaitFlag<HardEvent::MTE3_MTE2>(storeDone[nextBuffer]);
                }
                DataCopyPad(nextData, inputGM[(b + 1) * sliceCount_ + sliceTile], copyParams, padParams);
                SetFlag<HardEvent::MTE2_V>(loadReady[nextBuffer]);
            }

            Mul(currentData, currentData, scaleLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(outputReady[currentBuffer]);
            WaitFlag<HardEvent::V_MTE3>(outputReady[currentBuffer]);
            DataCopyPad(outputGM[b * sliceCount_ + sliceTile], currentData, copyParams);
            SetFlag<HardEvent::MTE3_MTE2>(storeDone[currentBuffer]);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(storeDone[(numBlocks_ - 1) & 1]);
        if (numBlocks_ > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(storeDone[(numBlocks_ - 2) & 1]);
        }
    }
}

} // namespace NsRenormSmStPipelined

#endif // _RENORM_SM_ST_PIPELINED_H_
