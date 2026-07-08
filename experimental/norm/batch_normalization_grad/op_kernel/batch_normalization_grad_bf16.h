/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_normalization_grad_bf16.h
 * \brief BatchNormalizationGrad bfloat16全特化实现
 */

#ifndef BATCH_NORMALIZATION_GRAD_BF16_H_
#define BATCH_NORMALIZATION_GRAD_BF16_H_

#include "kernel_operator.h"

namespace NsBatchNormalizationGrad {

__aicore__ inline void Bf16WaitVectorScalarSync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
}

__aicore__ inline void Bf16WaitVectorToMte3Sync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
}

template <>
class KernelBatchNormalizationGrad<bfloat16_t> {
public:
    __aicore__ inline KernelBatchNormalizationGrad() {}

    __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR input, GM_ADDR weight,
                                GM_ADDR bias, GM_ADDR save_mean, GM_ADDR save_invstd,
                                GM_ADDR grad_input, GM_ADDR grad_weight, GM_ADDR grad_bias,
                                GM_ADDR workspace, const BatchNormalizationGradTilingData* tilingData)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        using namespace AscendC;
        this->numChannels       = tilingData->numChannels;
        this->numBatches        = tilingData->numBatches;
        this->numSpatial        = tilingData->numSpatial;
        this->mFloat            = tilingData->mFloat;
        this->coreNum           = tilingData->coreNum;
        this->channelsPerCore   = tilingData->channelsPerCore;
        this->tailChannels      = tilingData->tailChannels;
        this->typeLength        = tilingData->typeLength;
        this->elementsPerBlock  = tilingData->elementsPerBlock;
        this->spatialSplitMode  = tilingData->spatialSplitMode;
        this->spatialTileSize   = tilingData->spatialTileSize;
        this->spatialTileNum    = tilingData->spatialTileNum;
        this->spatialTailSize   = tilingData->spatialTailSize;
        this->batchesPerTile    = tilingData->batchesPerTile;
        this->batchGroups       = tilingData->batchGroups;
        this->tailBatches       = tilingData->tailBatches;
        this->paddedSpatial     = tilingData->paddedSpatial;
        this->needPerBatchLoad  = (tilingData->needPerBatchLoad != 0);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        if (coreIdx < this->tailChannels) {
            this->channelStart = coreIdx * (this->channelsPerCore + 1);
            this->channelEnd = this->channelStart + this->channelsPerCore + 1;
        } else if (coreIdx < this->coreNum) {
            this->channelStart = this->tailChannels * (this->channelsPerCore + 1) +
                                 (coreIdx - this->tailChannels) * this->channelsPerCore;
            this->channelEnd = this->channelStart + this->channelsPerCore;
        } else {
            this->channelStart = 0; this->channelEnd = 0;
        }
        this->numChannelsPerCore = this->channelEnd - this->channelStart;
        this->alignedCPC = ((this->numChannelsPerCore + 7) / 8) * 8;
        if (this->alignedCPC == 0) this->alignedCPC = 8;

        uint32_t totalElements = this->numBatches * this->numChannels * this->numSpatial;
        gradOutputGm.SetGlobalBuffer((__gm__ bfloat16_t*)grad_output, totalElements);
        inputGm.SetGlobalBuffer((__gm__ bfloat16_t*)input, totalElements);
        weightGm.SetGlobalBuffer((__gm__ bfloat16_t*)weight, this->numChannels);
        saveMeanGm.SetGlobalBuffer((__gm__ bfloat16_t*)save_mean, this->numChannels);
        saveInvstdGm.SetGlobalBuffer((__gm__ bfloat16_t*)save_invstd, this->numChannels);

        gradInputGm.SetGlobalBuffer((__gm__ bfloat16_t*)grad_input, totalElements);
        gradWeightGm.SetGlobalBuffer((__gm__ bfloat16_t*)grad_weight, this->numChannels);
        gradBiasGm.SetGlobalBuffer((__gm__ bfloat16_t*)grad_bias, this->numChannels);

        if (this->spatialSplitMode) {
            uint32_t alignElems = 32u / sizeof(bfloat16_t);
            uint32_t tileAligned = ((this->spatialTileSize + alignElems - 1) / alignElems) * alignElems;
            if (tileAligned < alignElems) tileAligned = alignElems;

            pipe.InitBuffer(inQueueGradOut, BUFFER_NUM, tileAligned * sizeof(bfloat16_t));
            pipe.InitBuffer(inQueueInput,   BUFFER_NUM, tileAligned * sizeof(bfloat16_t));
            pipe.InitBuffer(outQueueGradInput, BUFFER_NUM, tileAligned * sizeof(bfloat16_t));
            pipe.InitBuffer(gradOutFloatBuf, tileAligned * sizeof(float));
            pipe.InitBuffer(inputFloatBuf,   tileAligned * sizeof(float));
            pipe.InitBuffer(xhatBuf,         tileAligned * sizeof(float));
            uint32_t tBufSz1 = tileAligned * sizeof(float);
            uint32_t outAl1  = 32 / sizeof(bfloat16_t);
            uint32_t outNd1  = ((this->numChannelsPerCore + outAl1 - 1) / outAl1) * outAl1;
            uint32_t outBt1  = 2 * outNd1 * sizeof(bfloat16_t);
            pipe.InitBuffer(tmpBuf, (tBufSz1 > outBt1) ? tBufSz1 : outBt1);
            pipe.InitBuffer(reduceBuf,       tileAligned * sizeof(float));
            pipe.InitBuffer(scalarBuf,       8 * sizeof(float));
            pipe.InitBuffer(batchGBuf,    this->batchesPerTile * sizeof(float));
            pipe.InitBuffer(batchGXhatBuf, this->batchesPerTile * sizeof(float));

            uint64_t epR = 256u / sizeof(float), epB = 32u / sizeof(float);
            uint64_t fr = (tileAligned + epR - 1u) / epR;
            uint64_t ar = ((fr + epB - 1u) / epB) * epB;
            pipe.InitBuffer(reduceWsBuf, static_cast<uint32_t>(ar * epB * sizeof(float)));

            if (this->coreNum > 1) pipe.InitBuffer(accumBuf, this->alignedCPC * 2 * sizeof(float));
        } else {
            pipe.InitBuffer(inQueueGradOut, BUFFER_NUM, this->paddedSpatial * sizeof(bfloat16_t));
            pipe.InitBuffer(inQueueInput,   BUFFER_NUM, this->paddedSpatial * sizeof(bfloat16_t));
            pipe.InitBuffer(outQueueGradInput, BUFFER_NUM, this->paddedSpatial * sizeof(bfloat16_t));

            uint32_t bufBS = this->paddedSpatial * sizeof(float);
            uint32_t outAlign = 32 / sizeof(bfloat16_t);
            uint32_t outNeed = ((this->numChannelsPerCore + outAlign - 1) / outAlign) * outAlign;
            uint32_t outBytes = 2 * outNeed * sizeof(bfloat16_t);
            uint32_t tmpBufSz = (bufBS > outBytes) ? bufBS : outBytes;
            pipe.InitBuffer(gradOutFloatBuf, bufBS);
            pipe.InitBuffer(inputFloatBuf,   bufBS);
            pipe.InitBuffer(xhatBuf,         bufBS);
            pipe.InitBuffer(tmpBuf,          tmpBufSz);
            pipe.InitBuffer(reduceBuf,       bufBS);
            pipe.InitBuffer(scalarBuf,       8 * sizeof(float));

            if (this->coreNum > 1) pipe.InitBuffer(accumBuf, this->alignedCPC * 2 * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->spatialSplitMode) {
            if (this->coreNum > 1) ProcessMultiCoreTiled();
            else ProcessSingleCoreTiled();
        } else {
            if (this->coreNum > 1) ProcessMultiCore();
            else ProcessSingleCore();
        }
    }

private:
    __aicore__ inline uint32_t GetGmIdx(uint32_t batch, uint32_t channel, uint32_t spatialOffset)
    {
        return batch * this->numChannels * this->numSpatial + channel * this->numSpatial + spatialOffset;
    }

    __aicore__ inline float ReadScalar(AscendC::GlobalTensor<bfloat16_t>& t, uint32_t idx)
    {
        return AscendC::ToFloat(t.GetValue(idx));
    }

    __aicore__ inline void CastToFloat(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<bfloat16_t>& src, uint32_t n)
    {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, n);
    }

    __aicore__ inline void CastFromFloat(AscendC::LocalTensor<bfloat16_t>& dst, AscendC::LocalTensor<float>& src, uint32_t n)
    {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, n);
    }

    __aicore__ inline uint32_t GetSpatialCount(uint32_t tileIdx)
    { return (tileIdx + 1 == this->spatialTileNum) ? this->spatialTailSize : this->spatialTileSize; }

    __aicore__ inline void LoadBatch(AscendC::LocalTensor<bfloat16_t>& gradOutLoc, AscendC::LocalTensor<bfloat16_t>& inputLoc,
                                      uint32_t batchIdx, uint32_t ch)
    {
        uint32_t gmBaseB  = GetGmIdx(batchIdx, ch, 0);
        uint32_t blockLen = static_cast<uint32_t>(this->numSpatial * this->typeLength);
        int32_t  rightPad = static_cast<int32_t>(this->paddedSpatial - this->numSpatial);
        AscendC::DataCopyExtParams cp{1, blockLen, 0, 0, 0};
        if (rightPad > 0) {
            AscendC::DataCopyPadExtParams<bfloat16_t> pp{true, 0, static_cast<uint8_t>(rightPad), 0};
            AscendC::DataCopyPad(gradOutLoc, gradOutputGm[gmBaseB], cp, pp);
            AscendC::DataCopyPad(inputLoc,   inputGm[gmBaseB],       cp, pp);
        } else {
            AscendC::DataCopyPadExtParams<bfloat16_t> pp{false, 0, 0, 0};
            AscendC::DataCopyPad(gradOutLoc, gradOutputGm[gmBaseB], cp, pp);
            AscendC::DataCopyPad(inputLoc,   inputGm[gmBaseB],       cp, pp);
        }
    }

    __aicore__ inline void LoadSpatialTile(AscendC::LocalTensor<bfloat16_t>& gradOutLoc, AscendC::LocalTensor<bfloat16_t>& inputLoc,
                                            uint32_t batchIdx, uint32_t ch, uint32_t tileIdx, uint32_t spatialCount)
    {
        uint32_t spatialStart = tileIdx * this->spatialTileSize;
        uint32_t gmBaseB = GetGmIdx(batchIdx, ch, spatialStart);
        uint32_t alignElems = 32u / sizeof(bfloat16_t);
        uint32_t tileAligned = ((spatialCount + alignElems - 1) / alignElems) * alignElems;
        uint32_t padElems = tileAligned - spatialCount;
        AscendC::DataCopyExtParams cp{1, spatialCount * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> pp{padElems != 0, 0, static_cast<uint8_t>(padElems), static_cast<bfloat16_t>(0)};
        AscendC::DataCopyPad(gradOutLoc, gradOutputGm[gmBaseB], cp, pp);
        AscendC::DataCopyPad(inputLoc,   inputGm[gmBaseB],       cp, pp);
    }

    // ========== 整空间 AccumulateChannel ==========
    __aicore__ inline void AccumulateChannel(uint32_t ch, float saveMeanVal, float saveInvstdVal,
                                              float& sumG, float& sumGXhat)
    {
        sumG = 0.0f; sumGXhat = 0.0f;
        float cG = 0.0f, cX = 0.0f;
        for (uint32_t grp = 0; grp < this->batchGroups; ++grp) {
            uint32_t numB = (grp == this->batchGroups - 1) ? this->tailBatches : this->batchesPerTile;
            uint32_t grpStart = grp * this->batchesPerTile;
            for (uint32_t b = 0; b < numB; ++b) {
                AscendC::LocalTensor<bfloat16_t> gLoc = inQueueGradOut.AllocTensor<bfloat16_t>();
                AscendC::LocalTensor<bfloat16_t> iLoc = inQueueInput.AllocTensor<bfloat16_t>();
                LoadBatch(gLoc, iLoc, grpStart + b, ch);
                inQueueGradOut.EnQue(gLoc); inQueueInput.EnQue(iLoc);
                AscendC::LocalTensor<bfloat16_t> gDq = inQueueGradOut.DeQue<bfloat16_t>();
                AscendC::LocalTensor<bfloat16_t> iDq = inQueueInput.DeQue<bfloat16_t>();
                AscendC::LocalTensor<float> gFloat  = gradOutFloatBuf.Get<float>();
                AscendC::LocalTensor<float> iFloat  = inputFloatBuf.Get<float>();
                AscendC::LocalTensor<float> xhat     = xhatBuf.Get<float>();
                AscendC::LocalTensor<float> t1       = tmpBuf.Get<float>();
                AscendC::LocalTensor<float> reduceWs = reduceBuf.Get<float>();
                AscendC::LocalTensor<float> scalar   = scalarBuf.Get<float>();

                CastToFloat(gFloat, gDq, this->numSpatial);
                CastToFloat(iFloat, iDq, this->numSpatial);
                AscendC::Adds(xhat, iFloat, -saveMeanVal, this->numSpatial);
                AscendC::Muls(xhat, xhat, saveInvstdVal, this->numSpatial);
                AscendC::Mul(t1, gFloat, xhat, this->numSpatial);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::ReduceSum<float>(scalar, gFloat, reduceWs, this->numSpatial);
                AscendC::PipeBarrier<PIPE_V>(); Bf16WaitVectorScalarSync();
                { float y = scalar.GetValue(0) - cG; float t = sumG + y; cG = (t - sumG) - y; sumG = t; }

                AscendC::ReduceSum<float>(scalar, t1, reduceWs, this->numSpatial);
                AscendC::PipeBarrier<PIPE_V>(); Bf16WaitVectorScalarSync();
                { float y = scalar.GetValue(0) - cX; float t = sumGXhat + y; cX = (t - sumGXhat) - y; sumGXhat = t; }

                inQueueGradOut.FreeTensor(gDq); inQueueInput.FreeTensor(iDq);
            }
        }
    }

    // ========== 整空间 WriteGradInput ==========
    __aicore__ inline void WriteGradInput(uint32_t ch, float saveMeanVal, float saveInvstdVal,
                                           float weightVal, float gBar, float gXhatBar)
    {
        float weightInvstd = weightVal * saveInvstdVal;
        for (uint32_t grp = 0; grp < this->batchGroups; ++grp) {
            uint32_t numB = (grp == this->batchGroups - 1) ? this->tailBatches : this->batchesPerTile;
            uint32_t grpStart = grp * this->batchesPerTile;
            for (uint32_t b = 0; b < numB; ++b) {
                AscendC::LocalTensor<bfloat16_t> gLoc = inQueueGradOut.AllocTensor<bfloat16_t>();
                AscendC::LocalTensor<bfloat16_t> iLoc = inQueueInput.AllocTensor<bfloat16_t>();
                LoadBatch(gLoc, iLoc, grpStart + b, ch);
                inQueueGradOut.EnQue(gLoc); inQueueInput.EnQue(iLoc);
                AscendC::LocalTensor<bfloat16_t> gDq = inQueueGradOut.DeQue<bfloat16_t>();
                AscendC::LocalTensor<bfloat16_t> iDq = inQueueInput.DeQue<bfloat16_t>();
                AscendC::LocalTensor<float> gFloat  = gradOutFloatBuf.Get<float>();
                AscendC::LocalTensor<float> iFloat  = inputFloatBuf.Get<float>();
                AscendC::LocalTensor<float> xhat     = xhatBuf.Get<float>();
                AscendC::LocalTensor<float> t1       = tmpBuf.Get<float>();

                CastToFloat(gFloat, gDq, this->numSpatial);
                CastToFloat(iFloat, iDq, this->numSpatial);
                AscendC::Adds(xhat, iFloat, -saveMeanVal, this->numSpatial);
                AscendC::Muls(xhat, xhat, saveInvstdVal, this->numSpatial);
                AscendC::Muls(t1, xhat, gXhatBar, this->numSpatial);
                AscendC::Adds(t1, t1, gBar, this->numSpatial);
                AscendC::Sub(t1, gFloat, t1, this->numSpatial);
                AscendC::Muls(t1, t1, weightInvstd, this->numSpatial);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::LocalTensor<bfloat16_t> dxLoc = outQueueGradInput.AllocTensor<bfloat16_t>();
                CastFromFloat(dxLoc, t1, this->numSpatial);
                outQueueGradInput.EnQue(dxLoc);
                AscendC::LocalTensor<bfloat16_t> dxDq = outQueueGradInput.DeQue<bfloat16_t>();
                uint32_t gmBaseB = GetGmIdx(grpStart + b, ch, 0);
                AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(this->numSpatial * this->typeLength), 0, 0, 0};
                AscendC::DataCopyPad(gradInputGm[gmBaseB], dxDq, cp);
                outQueueGradInput.FreeTensor(dxDq);

                inQueueGradOut.FreeTensor(gDq); inQueueInput.FreeTensor(iDq);
            }
        }
    }

    // ========== 空间分块 AccumulateChannelTiled ==========
    __aicore__ inline void AccumulateChannelTiled(uint32_t ch, float saveMeanVal, float saveInvstdVal,
                                                   float weightVal, float& sumG, float& sumGXhat)
    {
        sumG = 0.0f; sumGXhat = 0.0f;
        float cG = 0.0f, cX = 0.0f;
        for (uint32_t grp = 0; grp < this->batchGroups; ++grp) {
            uint32_t numB = (grp == this->batchGroups - 1) ? this->tailBatches : this->batchesPerTile;
            uint32_t grpStart = grp * this->batchesPerTile;
            for (uint32_t tileIdx = 0; tileIdx < this->spatialTileNum; ++tileIdx) {
                uint32_t spCnt = GetSpatialCount(tileIdx);
                for (uint32_t b = 0; b < numB; ++b) {
                    AscendC::LocalTensor<bfloat16_t> gLoc = inQueueGradOut.AllocTensor<bfloat16_t>();
                    AscendC::LocalTensor<bfloat16_t> iLoc = inQueueInput.AllocTensor<bfloat16_t>();
                    LoadSpatialTile(gLoc, iLoc, grpStart + b, ch, tileIdx, spCnt);
                    inQueueGradOut.EnQue(gLoc); inQueueInput.EnQue(iLoc);
                    AscendC::LocalTensor<bfloat16_t> gDq = inQueueGradOut.DeQue<bfloat16_t>();
                    AscendC::LocalTensor<bfloat16_t> iDq = inQueueInput.DeQue<bfloat16_t>();
                    AscendC::LocalTensor<float> gF = gradOutFloatBuf.Get<float>();
                    AscendC::LocalTensor<float> iF = inputFloatBuf.Get<float>();
                    AscendC::LocalTensor<float> xh = xhatBuf.Get<float>();
                    AscendC::LocalTensor<float> t1 = tmpBuf.Get<float>();
                    AscendC::LocalTensor<float> rw = reduceWsBuf.Get<float>();
                    AscendC::LocalTensor<float> sc = scalarBuf.Get<float>();

                    CastToFloat(gF, gDq, spCnt);
                    CastToFloat(iF, iDq, spCnt);
                    AscendC::Adds(xh, iF, -saveMeanVal, spCnt);
                    AscendC::Muls(xh, xh, saveInvstdVal, spCnt);

                    AscendC::ReduceSum<float>(sc, gF, rw, spCnt);
                    AscendC::PipeBarrier<PIPE_V>(); Bf16WaitVectorScalarSync();
                    { float y = sc.GetValue(0) - cG; float t = sumG + y; cG = (t - sumG) - y; sumG = t; }

                    AscendC::Mul(t1, gF, xh, spCnt);
                    AscendC::ReduceSum<float>(sc, t1, rw, spCnt);
                    AscendC::PipeBarrier<PIPE_V>(); Bf16WaitVectorScalarSync();
                    { float y = sc.GetValue(0) - cX; float t = sumGXhat + y; cX = (t - sumGXhat) - y; sumGXhat = t; }

                    inQueueGradOut.FreeTensor(gDq); inQueueInput.FreeTensor(iDq);
                }
            }
        }
    }

    // ========== 空间分块 WriteGradInputTiled ==========
    __aicore__ inline void WriteGradInputTiled(uint32_t ch, float saveMeanVal, float saveInvstdVal,
                                                float weightVal, float gBar, float gXhatBar)
    {
        float weightInvstd = weightVal * saveInvstdVal;
        for (uint32_t grp = 0; grp < this->batchGroups; ++grp) {
            uint32_t numB = (grp == this->batchGroups - 1) ? this->tailBatches : this->batchesPerTile;
            uint32_t grpStart = grp * this->batchesPerTile;
            for (uint32_t tileIdx = 0; tileIdx < this->spatialTileNum; ++tileIdx) {
                uint32_t spCnt = GetSpatialCount(tileIdx);
                for (uint32_t b = 0; b < numB; ++b) {
                    AscendC::LocalTensor<bfloat16_t> gLoc = inQueueGradOut.AllocTensor<bfloat16_t>();
                    AscendC::LocalTensor<bfloat16_t> iLoc = inQueueInput.AllocTensor<bfloat16_t>();
                    LoadSpatialTile(gLoc, iLoc, grpStart + b, ch, tileIdx, spCnt);
                    inQueueGradOut.EnQue(gLoc); inQueueInput.EnQue(iLoc);
                    AscendC::LocalTensor<bfloat16_t> gDq = inQueueGradOut.DeQue<bfloat16_t>();
                    AscendC::LocalTensor<bfloat16_t> iDq = inQueueInput.DeQue<bfloat16_t>();
                    AscendC::LocalTensor<float> gF = gradOutFloatBuf.Get<float>();
                    AscendC::LocalTensor<float> iF = inputFloatBuf.Get<float>();
                    AscendC::LocalTensor<float> xh = xhatBuf.Get<float>();
                    AscendC::LocalTensor<float> t1 = tmpBuf.Get<float>();

                    CastToFloat(gF, gDq, spCnt);
                    CastToFloat(iF, iDq, spCnt);
                    AscendC::Adds(xh, iF, -saveMeanVal, spCnt);
                    AscendC::Muls(xh, xh, saveInvstdVal, spCnt);
                    AscendC::Muls(t1, xh, gXhatBar, spCnt);
                    AscendC::Adds(t1, t1, gBar, spCnt);
                    AscendC::Sub(t1, gF, t1, spCnt);
                    AscendC::Muls(t1, t1, weightInvstd, spCnt);
                    AscendC::PipeBarrier<PIPE_V>();

                    AscendC::LocalTensor<bfloat16_t> dxLoc = outQueueGradInput.AllocTensor<bfloat16_t>();
                    CastFromFloat(dxLoc, t1, spCnt);
                    outQueueGradInput.EnQue(dxLoc);
                    AscendC::LocalTensor<bfloat16_t> dxDq = outQueueGradInput.DeQue<bfloat16_t>();
                    uint32_t gmBaseB = GetGmIdx(grpStart + b, ch, tileIdx * this->spatialTileSize);
                    AscendC::DataCopyExtParams cp{1, spCnt * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
                    AscendC::DataCopyPad(gradInputGm[gmBaseB], dxDq, cp);
                    outQueueGradInput.FreeTensor(dxDq);

                    inQueueGradOut.FreeTensor(gDq); inQueueInput.FreeTensor(iDq);
                }
            }
        }
    }

    // ========== 单核 ==========
    __aicore__ inline void ProcessSingleCore()
    {
        for (uint32_t c = 0; c < this->numChannels; ++c) {
            float sm = ReadScalar(saveMeanGm, c), iv = ReadScalar(saveInvstdGm, c), wv = ReadScalar(weightGm, c);
            float sg = 0.0f, sx = 0.0f;
            AccumulateChannel(c, sm, iv, sg, sx);
            float gB = sg / this->mFloat, gXB = sx / this->mFloat;
            WriteGradInput(c, sm, iv, wv, gB, gXB);
            gradWeightGm.SetValue(c, AscendC::ToBfloat16(sx));
            gradBiasGm.SetValue(c, AscendC::ToBfloat16(sg));
        }
    }

    __aicore__ inline void ProcessSingleCoreTiled()
    {
        for (uint32_t c = 0; c < this->numChannels; ++c) {
            float sm = ReadScalar(saveMeanGm, c), iv = ReadScalar(saveInvstdGm, c), wv = ReadScalar(weightGm, c);
            float sg = 0.0f, sx = 0.0f;
            AccumulateChannelTiled(c, sm, iv, wv, sg, sx);
            float gB = sg / this->mFloat, gXB = sx / this->mFloat;
            WriteGradInputTiled(c, sm, iv, wv, gB, gXB);
            gradWeightGm.SetValue(c, AscendC::ToBfloat16(sx));
            gradBiasGm.SetValue(c, AscendC::ToBfloat16(sg));
        }
    }

    // ========== 多核输出 ==========
    __aicore__ inline void WriteGradOutputs(AscendC::LocalTensor<float>& acc)
    {
        uint32_t pcb = this->numChannelsPerCore * sizeof(bfloat16_t);
        uint32_t alignedOff = ((pcb + 31) / 32) * (32 / sizeof(bfloat16_t));

        AscendC::LocalTensor<bfloat16_t> t = tmpBuf.Get<bfloat16_t>();
        for (uint32_t c = 0; c < this->numChannelsPerCore; ++c) {
            float wv = acc.GetValue(c);
            float bv = acc.GetValue(this->alignedCPC + c);
            t.SetValue(c, AscendC::ToBfloat16(wv));
            t.SetValue(alignedOff + c, AscendC::ToBfloat16(bv));
        }
        AscendC::PipeBarrier<PIPE_V>(); Bf16WaitVectorToMte3Sync();
        AscendC::DataCopyExtParams cp{1, pcb, 0, 0, 0};
        AscendC::DataCopyPad(gradWeightGm[this->channelStart], t, cp);
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::DataCopyPad(gradBiasGm[this->channelStart], t[alignedOff], cp);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void ProcessMultiCore()
    {
        AscendC::LocalTensor<float> acc = accumBuf.Get<float>();

        for (uint32_t c = 0; c < this->numChannelsPerCore; ++c) {
            uint32_t ch = this->channelStart + c;
            float sm = ReadScalar(saveMeanGm, ch), iv = ReadScalar(saveInvstdGm, ch);
            float sg = 0.0f, sx = 0.0f;
            AccumulateChannel(ch, sm, iv, sg, sx);
            acc.SetValue(c, sx);
            acc.SetValue(this->alignedCPC + c, sg);
        }
        for (uint32_t c = this->numChannelsPerCore; c < this->alignedCPC; ++c) {
            acc.SetValue(c, 0.0f);
            acc.SetValue(this->alignedCPC + c, 0.0f);
        }
        AscendC::PipeBarrier<PIPE_V>();

        WriteGradOutputs(acc);
        AscendC::SyncAll();

        for (uint32_t c = 0; c < this->numChannelsPerCore; ++c) {
            uint32_t ch = this->channelStart + c;
            float sm = ReadScalar(saveMeanGm, ch), iv = ReadScalar(saveInvstdGm, ch), wv = ReadScalar(weightGm, ch);
            float gXB = acc.GetValue(c) / this->mFloat;
            float gB  = acc.GetValue(this->alignedCPC + c) / this->mFloat;
            WriteGradInput(ch, sm, iv, wv, gB, gXB);
        }
        AscendC::SyncAll();
    }

    __aicore__ inline void ProcessMultiCoreTiled()
    {
        AscendC::LocalTensor<float> acc = accumBuf.Get<float>();

        for (uint32_t c = 0; c < this->numChannelsPerCore; ++c) {
            uint32_t ch = this->channelStart + c;
            float sm = ReadScalar(saveMeanGm, ch), iv = ReadScalar(saveInvstdGm, ch), wv = ReadScalar(weightGm, ch);
            float sg = 0.0f, sx = 0.0f;
            AccumulateChannelTiled(ch, sm, iv, wv, sg, sx);
            acc.SetValue(c, sx);
            acc.SetValue(this->alignedCPC + c, sg);
        }
        for (uint32_t c = this->numChannelsPerCore; c < this->alignedCPC; ++c) {
            acc.SetValue(c, 0.0f);
            acc.SetValue(this->alignedCPC + c, 0.0f);
        }
        AscendC::PipeBarrier<PIPE_V>();

        WriteGradOutputs(acc);
        AscendC::SyncAll();

        for (uint32_t c = 0; c < this->numChannelsPerCore; ++c) {
            uint32_t ch = this->channelStart + c;
            float sm = ReadScalar(saveMeanGm, ch), iv = ReadScalar(saveInvstdGm, ch), wv = ReadScalar(weightGm, ch);
            float gXB = acc.GetValue(c) / this->mFloat;
            float gB  = acc.GetValue(this->alignedCPC + c) / this->mFloat;
            WriteGradInputTiled(ch, sm, iv, wv, gB, gXB);
        }
        AscendC::SyncAll();
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueGradOut;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueInput;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueGradInput;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> gradOutFloatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inputFloatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xhatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduceWsBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> batchGBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> batchGXhatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accumBuf;

    AscendC::GlobalTensor<bfloat16_t> gradOutputGm;
    AscendC::GlobalTensor<bfloat16_t> inputGm;
    AscendC::GlobalTensor<bfloat16_t> weightGm;
    AscendC::GlobalTensor<bfloat16_t> saveMeanGm;
    AscendC::GlobalTensor<bfloat16_t> saveInvstdGm;
    AscendC::GlobalTensor<bfloat16_t> gradInputGm;
    AscendC::GlobalTensor<bfloat16_t> gradWeightGm;
    AscendC::GlobalTensor<bfloat16_t> gradBiasGm;

    uint32_t channelStart, channelEnd, numChannelsPerCore;
    uint32_t numChannels, numBatches, numSpatial;
    float    mFloat;
    uint32_t coreNum, channelsPerCore, tailChannels;
    uint32_t typeLength, elementsPerBlock;
    uint32_t spatialSplitMode;
    uint32_t spatialTileSize, spatialTileNum, spatialTailSize;
    uint32_t batchesPerTile, batchGroups, tailBatches, paddedSpatial;
    uint32_t alignedCPC;
    bool     needPerBatchLoad;
};

} // namespace NsBatchNormalizationGrad

#endif // BATCH_NORMALIZATION_GRAD_BF16_H_
