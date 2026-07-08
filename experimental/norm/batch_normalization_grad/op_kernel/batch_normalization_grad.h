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
 * \file batch_normalization_grad.h
 * \brief BatchNormalizationGrad算子Kernel实现
 *
 * 计算公式：
 * xhat = (input - save_mean) * save_invstd
 * grad_input = weight * save_invstd / m * (m * grad_output - sum(grad_output) - xhat * sum(grad_output * xhat))
 * grad_weight = sum(grad_output * xhat)
 * grad_bias = sum(grad_output)
 *
 * 支持两种计算模式：
 *   - 整空间模式：numSpatial 较小时完整空间装入 UB
 *   - 空间分块模式：numSpatial 过大时按 spatialTileSize 分块处理
 */

#ifndef BATCH_NORMALIZATION_GRAD_H_
#define BATCH_NORMALIZATION_GRAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_normalization_grad_tiling_data.h"
#include "batch_normalization_grad_tiling_key.h"

namespace NsBatchNormalizationGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline void WaitVectorScalarSync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

__aicore__ inline void WaitVectorToMte3Sync()
{
    int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
}

template <typename T>
class KernelBatchNormalizationGrad {
public:
    __aicore__ inline KernelBatchNormalizationGrad() {}

    __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR input, GM_ADDR weight,
                                GM_ADDR bias, GM_ADDR save_mean, GM_ADDR save_invstd,
                                GM_ADDR grad_input, GM_ADDR grad_weight, GM_ADDR grad_bias,
                                GM_ADDR workspace, const BatchNormalizationGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t GetGmIdx(uint32_t batch, uint32_t channel, uint32_t spatialOffset)
    {
        return batch * this->numChannels * this->numSpatial + channel * this->numSpatial + spatialOffset;
    }

    __aicore__ inline float ReadScalar(GlobalTensor<T>& t, uint32_t idx)
    { return static_cast<float>(t.GetValue(idx)); }

    __aicore__ inline void CastToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t n)
    {
        if constexpr (Std::is_same<T, float>::value) {
            Adds(dst, src, 0.0f, n);
        } else {
            // half -> float（拓宽，2B->4B）：无需舍入，必须用 CAST_NONE
            Cast(dst, src, RoundMode::CAST_NONE, n);
        }
    }

    __aicore__ inline void CastFromFloat(LocalTensor<T>& dst, LocalTensor<float>& src, uint32_t n)
    {
        if constexpr (Std::is_same<T, float>::value) {
            Adds(dst, src, 0.0f, n);
        } else {
            // float -> half（收窄，4B->2B）：必须舍入，用 CAST_RINT
            Cast(dst, src, RoundMode::CAST_RINT, n);
        }
    }

    __aicore__ inline uint32_t GetSpatialCount(uint32_t tileIdx)
    { return (tileIdx + 1 == this->spatialTileNum) ? this->spatialTailSize : this->spatialTileSize; }

    __aicore__ inline void LoadBatch(LocalTensor<T>& gradOutLoc, LocalTensor<T>& inputLoc,
                                      uint32_t batchIdx, uint32_t ch)
    {
        uint32_t gmBaseB  = GetGmIdx(batchIdx, ch, 0);
        uint32_t blockLen = static_cast<uint32_t>(this->numSpatial * this->typeLength);
        int32_t  rightPad = static_cast<int32_t>(this->paddedSpatial - this->numSpatial);
        DataCopyExtParams cp{1, blockLen, 0, 0, 0};
        if (rightPad > 0) {
            DataCopyPadExtParams<T> pp{true, 0, static_cast<uint8_t>(rightPad), 0};
            DataCopyPad(gradOutLoc, gradOutputGm[gmBaseB], cp, pp);
            DataCopyPad(inputLoc,   inputGm[gmBaseB],       cp, pp);
        } else {
            DataCopyPadExtParams<T> pp{false, 0, 0, 0};
            DataCopyPad(gradOutLoc, gradOutputGm[gmBaseB], cp, pp);
            DataCopyPad(inputLoc,   inputGm[gmBaseB],       cp, pp);
        }
    }

    __aicore__ inline void LoadSpatialTile(LocalTensor<T>& gradOutLoc, LocalTensor<T>& inputLoc,
                                            uint32_t batchIdx, uint32_t ch, uint32_t tileIdx, uint32_t spatialCount)
    {
        uint32_t spatialStart = tileIdx * this->spatialTileSize;
        uint32_t gmBaseB = GetGmIdx(batchIdx, ch, spatialStart);
        uint32_t alignElems = 32u / sizeof(T);
        uint32_t tileAligned = ((spatialCount + alignElems - 1) / alignElems) * alignElems;
        uint32_t padElems = tileAligned - spatialCount;
        DataCopyExtParams cp{1, spatialCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pp{padElems != 0, 0, static_cast<uint8_t>(padElems), static_cast<T>(0)};
        DataCopyPad(gradOutLoc, gradOutputGm[gmBaseB], cp, pp);
        DataCopyPad(inputLoc,   inputGm[gmBaseB],       cp, pp);
    }

    // ========== 整空间 AccumulateChannel ==========
    __aicore__ inline void AccumulateChannel(uint32_t ch, float saveMeanVal, float saveInvstdVal,
                                              float& sumG, float& sumGXhat)
    {
        sumG = 0.0f; sumGXhat = 0.0f;
        float cG = 0.0f, cX = 0.0f; // Kahan
        for (uint32_t grp = 0; grp < this->batchGroups; ++grp) {
            uint32_t numB = (grp == this->batchGroups - 1) ? this->tailBatches : this->batchesPerTile;
            uint32_t grpStart = grp * this->batchesPerTile;
            for (uint32_t b = 0; b < numB; ++b) {
                LocalTensor<T> gLoc = inQueueGradOut.AllocTensor<T>();
                LocalTensor<T> iLoc = inQueueInput.AllocTensor<T>();
                LoadBatch(gLoc, iLoc, grpStart + b, ch);
                inQueueGradOut.EnQue(gLoc); inQueueInput.EnQue(iLoc);
                LocalTensor<T> gDq = inQueueGradOut.DeQue<T>();
                LocalTensor<T> iDq = inQueueInput.DeQue<T>();
                LocalTensor<float> gFloat  = gradOutFloatBuf.Get<float>();
                LocalTensor<float> iFloat  = inputFloatBuf.Get<float>();
                LocalTensor<float> xhat     = xhatBuf.Get<float>();
                LocalTensor<float> t1       = tmpBuf.Get<float>();
                LocalTensor<float> reduceWs = reduceBuf.Get<float>();
                LocalTensor<float> scalar   = scalarBuf.Get<float>();

                CastToFloat(gFloat, gDq, this->numSpatial);
                CastToFloat(iFloat, iDq, this->numSpatial);
                Adds(xhat, iFloat, -saveMeanVal, this->numSpatial);
                Muls(xhat, xhat, saveInvstdVal, this->numSpatial);
                Mul(t1, gFloat, xhat, this->numSpatial);
                PipeBarrier<PIPE_V>();

                ReduceSum<float>(scalar, gFloat, reduceWs, this->numSpatial);
                PipeBarrier<PIPE_V>(); WaitVectorScalarSync();
                { float y = scalar.GetValue(0) - cG; float t = sumG + y; cG = (t - sumG) - y; sumG = t; }

                ReduceSum<float>(scalar, t1, reduceWs, this->numSpatial);
                PipeBarrier<PIPE_V>(); WaitVectorScalarSync();
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
                LocalTensor<T> gLoc = inQueueGradOut.AllocTensor<T>();
                LocalTensor<T> iLoc = inQueueInput.AllocTensor<T>();
                LoadBatch(gLoc, iLoc, grpStart + b, ch);
                inQueueGradOut.EnQue(gLoc); inQueueInput.EnQue(iLoc);
                LocalTensor<T> gDq = inQueueGradOut.DeQue<T>();
                LocalTensor<T> iDq = inQueueInput.DeQue<T>();
                LocalTensor<float> gFloat  = gradOutFloatBuf.Get<float>();
                LocalTensor<float> iFloat  = inputFloatBuf.Get<float>();
                LocalTensor<float> xhat     = xhatBuf.Get<float>();
                LocalTensor<float> t1       = tmpBuf.Get<float>();

                CastToFloat(gFloat, gDq, this->numSpatial);
                CastToFloat(iFloat, iDq, this->numSpatial);
                Adds(xhat, iFloat, -saveMeanVal, this->numSpatial);
                Muls(xhat, xhat, saveInvstdVal, this->numSpatial);
                Muls(t1, xhat, gXhatBar, this->numSpatial);
                Adds(t1, t1, gBar, this->numSpatial);
                Sub(t1, gFloat, t1, this->numSpatial);
                Muls(t1, t1, weightInvstd, this->numSpatial);
                PipeBarrier<PIPE_V>();

                LocalTensor<T> dxLoc = outQueueGradInput.AllocTensor<T>();
                CastFromFloat(dxLoc, t1, this->numSpatial);
                outQueueGradInput.EnQue(dxLoc);
                LocalTensor<T> dxDq = outQueueGradInput.DeQue<T>();
                uint32_t gmBaseB = GetGmIdx(grpStart + b, ch, 0);
                DataCopyExtParams cp{1, static_cast<uint32_t>(this->numSpatial * this->typeLength), 0, 0, 0};
                DataCopyPad(gradInputGm[gmBaseB], dxDq, cp);
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
                    LocalTensor<T> gLoc = inQueueGradOut.AllocTensor<T>();
                    LocalTensor<T> iLoc = inQueueInput.AllocTensor<T>();
                    LoadSpatialTile(gLoc, iLoc, grpStart + b, ch, tileIdx, spCnt);
                    inQueueGradOut.EnQue(gLoc); inQueueInput.EnQue(iLoc);
                    LocalTensor<T> gDq = inQueueGradOut.DeQue<T>();
                    LocalTensor<T> iDq = inQueueInput.DeQue<T>();
                    LocalTensor<float> gF = gradOutFloatBuf.Get<float>();
                    LocalTensor<float> iF = inputFloatBuf.Get<float>();
                    LocalTensor<float> xh = xhatBuf.Get<float>();
                    LocalTensor<float> t1 = tmpBuf.Get<float>();
                    LocalTensor<float> rw = reduceWsBuf.Get<float>();
                    LocalTensor<float> sc = scalarBuf.Get<float>();

                    CastToFloat(gF, gDq, spCnt);
                    CastToFloat(iF, iDq, spCnt);
                    Adds(xh, iF, -saveMeanVal, spCnt);
                    Muls(xh, xh, saveInvstdVal, spCnt);

                    ReduceSum<float>(sc, gF, rw, spCnt);
                    PipeBarrier<PIPE_V>(); WaitVectorScalarSync();
                    { float y = sc.GetValue(0) - cG; float t = sumG + y; cG = (t - sumG) - y; sumG = t; }

                    Mul(t1, gF, xh, spCnt);
                    ReduceSum<float>(sc, t1, rw, spCnt);
                    PipeBarrier<PIPE_V>(); WaitVectorScalarSync();
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
                    LocalTensor<T> gLoc = inQueueGradOut.AllocTensor<T>();
                    LocalTensor<T> iLoc = inQueueInput.AllocTensor<T>();
                    LoadSpatialTile(gLoc, iLoc, grpStart + b, ch, tileIdx, spCnt);
                    inQueueGradOut.EnQue(gLoc); inQueueInput.EnQue(iLoc);
                    LocalTensor<T> gDq = inQueueGradOut.DeQue<T>();
                    LocalTensor<T> iDq = inQueueInput.DeQue<T>();
                    LocalTensor<float> gF = gradOutFloatBuf.Get<float>();
                    LocalTensor<float> iF = inputFloatBuf.Get<float>();
                    LocalTensor<float> xh = xhatBuf.Get<float>();
                    LocalTensor<float> t1 = tmpBuf.Get<float>();

                    CastToFloat(gF, gDq, spCnt);
                    CastToFloat(iF, iDq, spCnt);
                    Adds(xh, iF, -saveMeanVal, spCnt);
                    Muls(xh, xh, saveInvstdVal, spCnt);
                    Muls(t1, xh, gXhatBar, spCnt);
                    Adds(t1, t1, gBar, spCnt);
                    Sub(t1, gF, t1, spCnt);
                    Muls(t1, t1, weightInvstd, spCnt);
                    PipeBarrier<PIPE_V>();

                    LocalTensor<T> dxLoc = outQueueGradInput.AllocTensor<T>();
                    CastFromFloat(dxLoc, t1, spCnt);
                    outQueueGradInput.EnQue(dxLoc);
                    LocalTensor<T> dxDq = outQueueGradInput.DeQue<T>();
                    uint32_t gmBaseB = GetGmIdx(grpStart + b, ch, tileIdx * this->spatialTileSize);
                    DataCopyExtParams cp{1, spCnt * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                    DataCopyPad(gradInputGm[gmBaseB], dxDq, cp);
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
            if constexpr (Std::is_same<T, float>::value) {
                gradWeightGm.SetValue(c, static_cast<T>(sx));
                gradBiasGm.SetValue(c, static_cast<T>(sg));
            } else {
                gradWeightGm.SetValue(c, ScalarCast<float, T, RoundMode::CAST_ODD>(sx));
                gradBiasGm.SetValue(c, ScalarCast<float, T, RoundMode::CAST_ODD>(sg));
            }
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
            if constexpr (Std::is_same<T, float>::value) {
                gradWeightGm.SetValue(c, static_cast<T>(sx));
                gradBiasGm.SetValue(c, static_cast<T>(sg));
            } else {
                gradWeightGm.SetValue(c, ScalarCast<float, T, RoundMode::CAST_ODD>(sx));
                gradBiasGm.SetValue(c, ScalarCast<float, T, RoundMode::CAST_ODD>(sg));
            }
        }
    }

    // ========== 多核输出 ==========
    __aicore__ inline void WriteGradOutputs(LocalTensor<float>& acc)
    {
        uint32_t pcb = this->numChannelsPerCore * this->typeLength;
        uint32_t alignedOff = ((pcb + 31) / 32) * (32 / this->typeLength);

        LocalTensor<T> t = tmpBuf.Get<T>();
        for (uint32_t c = 0; c < this->numChannelsPerCore; ++c) {
            float wv = acc.GetValue(c);
            float bv = acc.GetValue(this->alignedCPC + c);
            if constexpr (Std::is_same<T, float>::value) {
                t.SetValue(c, static_cast<T>(wv));
                t.SetValue(alignedOff + c, static_cast<T>(bv));
            } else {
                t.SetValue(c, ScalarCast<float, T, RoundMode::CAST_ODD>(wv));
                t.SetValue(alignedOff + c, ScalarCast<float, T, RoundMode::CAST_ODD>(bv));
            }
        }
        PipeBarrier<PIPE_V>(); WaitVectorToMte3Sync();
        DataCopyExtParams cp{1, pcb, 0, 0, 0};
        DataCopyPad(gradWeightGm[this->channelStart], t, cp);
        PipeBarrier<PIPE_MTE3>();
        DataCopyPad(gradBiasGm[this->channelStart], t[alignedOff], cp);
        PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void ProcessMultiCore()
    {
        LocalTensor<float> acc = accumBuf.Get<float>();

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
        PipeBarrier<PIPE_V>();

        WriteGradOutputs(acc);
        SyncAll();

        for (uint32_t c = 0; c < this->numChannelsPerCore; ++c) {
            uint32_t ch = this->channelStart + c;
            float sm = ReadScalar(saveMeanGm, ch), iv = ReadScalar(saveInvstdGm, ch), wv = ReadScalar(weightGm, ch);
            float gXB = acc.GetValue(c) / this->mFloat;
            float gB  = acc.GetValue(this->alignedCPC + c) / this->mFloat;
            WriteGradInput(ch, sm, iv, wv, gB, gXB);
        }
        SyncAll();
    }

    __aicore__ inline void ProcessMultiCoreTiled()
    {
        LocalTensor<float> acc = accumBuf.Get<float>();

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
        PipeBarrier<PIPE_V>();

        WriteGradOutputs(acc);
        SyncAll();

        for (uint32_t c = 0; c < this->numChannelsPerCore; ++c) {
            uint32_t ch = this->channelStart + c;
            float sm = ReadScalar(saveMeanGm, ch), iv = ReadScalar(saveInvstdGm, ch), wv = ReadScalar(weightGm, ch);
            float gXB = acc.GetValue(c) / this->mFloat;
            float gB  = acc.GetValue(this->alignedCPC + c) / this->mFloat;
            WriteGradInputTiled(ch, sm, iv, wv, gB, gXB);
        }
        SyncAll();
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM>  inQueueGradOut;
    TQue<QuePosition::VECIN, BUFFER_NUM>  inQueueInput;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradInput;
    TBuf<TPosition::VECCALC> gradOutFloatBuf;
    TBuf<TPosition::VECCALC> inputFloatBuf;
    TBuf<TPosition::VECCALC> xhatBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> reduceBuf;
    TBuf<TPosition::VECCALC> reduceWsBuf;
    TBuf<TPosition::VECCALC> scalarBuf;
    TBuf<TPosition::VECCALC> batchGBuf;
    TBuf<TPosition::VECCALC> batchGXhatBuf;
    TBuf<TPosition::VECCALC> accumBuf;

    GlobalTensor<T> gradOutputGm;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> saveMeanGm;
    GlobalTensor<T> saveInvstdGm;
    GlobalTensor<T> gradInputGm;
    GlobalTensor<T> gradWeightGm;
    GlobalTensor<T> gradBiasGm;

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

template <typename T>
__aicore__ inline void KernelBatchNormalizationGrad<T>::Init(
    GM_ADDR grad_output, GM_ADDR input, GM_ADDR weight,
    GM_ADDR bias, GM_ADDR save_mean, GM_ADDR save_invstd,
    GM_ADDR grad_input, GM_ADDR grad_weight, GM_ADDR grad_bias,
    GM_ADDR workspace, const BatchNormalizationGradTilingData* tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

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

    uint32_t coreIdx = GetBlockIdx();
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
    gradOutputGm.SetGlobalBuffer((__gm__ T*)grad_output, totalElements);
    inputGm.SetGlobalBuffer((__gm__ T*)input, totalElements);
    weightGm.SetGlobalBuffer((__gm__ T*)weight, this->numChannels);
    saveMeanGm.SetGlobalBuffer((__gm__ T*)save_mean, this->numChannels);
    saveInvstdGm.SetGlobalBuffer((__gm__ T*)save_invstd, this->numChannels);

    gradInputGm.SetGlobalBuffer((__gm__ T*)grad_input, totalElements);
    gradWeightGm.SetGlobalBuffer((__gm__ T*)grad_weight, this->numChannels);
    gradBiasGm.SetGlobalBuffer((__gm__ T*)grad_bias, this->numChannels);

    if (this->spatialSplitMode) {
        uint32_t alignElems = 32u / sizeof(T);
        uint32_t tileAligned = ((this->spatialTileSize + alignElems - 1) / alignElems) * alignElems;
        if (tileAligned < alignElems) tileAligned = alignElems;

        pipe.InitBuffer(inQueueGradOut, BUFFER_NUM, tileAligned * sizeof(T));
        pipe.InitBuffer(inQueueInput,   BUFFER_NUM, tileAligned * sizeof(T));
        pipe.InitBuffer(outQueueGradInput, BUFFER_NUM, tileAligned * sizeof(T));
        pipe.InitBuffer(gradOutFloatBuf, tileAligned * sizeof(float));
        pipe.InitBuffer(inputFloatBuf,   tileAligned * sizeof(float));
        pipe.InitBuffer(xhatBuf,         tileAligned * sizeof(float));
        uint32_t tBufSz1 = tileAligned * sizeof(float);
        uint32_t outAl1  = 32 / this->typeLength;
        uint32_t outNd1  = ((this->numChannelsPerCore + outAl1 - 1) / outAl1) * outAl1;
        uint32_t outBt1  = 2 * outNd1 * this->typeLength;
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
        pipe.InitBuffer(inQueueGradOut, BUFFER_NUM, this->paddedSpatial * sizeof(T));
        pipe.InitBuffer(inQueueInput,   BUFFER_NUM, this->paddedSpatial * sizeof(T));
        pipe.InitBuffer(outQueueGradInput, BUFFER_NUM, this->paddedSpatial * sizeof(T));

        uint32_t bufBS = this->paddedSpatial * sizeof(float);
        uint32_t outAlign = 32 / this->typeLength;
        uint32_t outNeed = ((this->numChannelsPerCore + outAlign - 1) / outAlign) * outAlign;
        uint32_t outBytes = 2 * outNeed * this->typeLength;
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

template <typename T>
__aicore__ inline void KernelBatchNormalizationGrad<T>::Process()
{
    if (this->spatialSplitMode) {
        if (this->coreNum > 1) ProcessMultiCoreTiled();
        else ProcessSingleCoreTiled();
    } else {
        if (this->coreNum > 1) ProcessMultiCore();
        else ProcessSingleCore();
    }
}

} // namespace NsBatchNormalizationGrad

#include "batch_normalization_grad_bf16.h"

#endif // BATCH_NORMALIZATION_GRAD_H_
