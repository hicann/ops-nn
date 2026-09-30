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
 * \file ms_deform_attn_generic.h
 * \brief
 */
#ifndef MS_DEFORM_ATTN_GENERIC_H
#define MS_DEFORM_ATTN_GENERIC_H

#include <type_traits>
#include "kernel_operator.h"
using namespace AscendC;

template <typename TilingDataT, typename T>
class KernelMultiScaleDeformableAttn {
public:
    __aicore__ inline KernelMultiScaleDeformableAttn() {}

    __aicore__ inline void Init(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valuLevelStartIndex,
                                GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output, GM_ADDR workspace,
                                const TilingDataT* __restrict tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        dataAlign = blockBytes / sizeof(T);

        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        embedDims = tiling_data->embedDims;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        batchSize = tiling_data->batchSize;
        coreNum = tiling_data->coreNum;

        numLevelsAlign = AlignUp(numLevels, dataAlign);

        maxUbNum = (useUbSize / sizeof(float) - threeBuffer * numLevelsAlign - fourBuffer * embedDims) /
                   (numQuerieBuffer + numEmbedBuffer * embedDims);
        maxUbNum = maxUbNum / dataAlign * dataAlign;
        numQueriesper = DivCeil(numQueries, maxUbNum);
        numQueriestail = numQueries - (numQueriesper - 1) * maxUbNum;
        numQueriestail = AlignUp(numQueriestail, dataAlign);
        numQueriesAlign = numQueries <= maxUbNum ? numQueriestail : maxUbNum;

        stagingStride = (embedDims > dataAlign) ? embedDims : dataAlign;

        taskNum = batchSize * numHeads * numLevels * numPoints * numQueriesper;
        taskNumPerCore = DivCeil(taskNum, coreNum);

        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        weightStride0 = numQueries;
        weightStride1 = numPoints * weightStride0;
        weightStride2 = numLevels * weightStride1;
        weightStride3 = numHeads * weightStride2;

        valueStride0 = embedDims;
        valueStride1 = numKeys * valueStride0;
        valueStride2 = numHeads * valueStride1;
        wStride = embedDims;

        outputStride0 = embedDims;
        outputStride1 = numHeads * outputStride0;
        outputStride2 = numQueries * outputStride1;

        copyOutParams = {1, (uint32_t)(embedDims * sizeof(T)), 0, 0, 0};
        copyOutParamsFloat = {1, (uint32_t)(embedDims * sizeof(float)), 0, 0, 0};
        copyInParamsV3 = {1, (uint32_t)(embedDims * sizeof(T)), 0, 0, 0};

        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        if constexpr (!std::is_same_v<T, float>) {
            eventIdStagingVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        }

        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), batchSize * numKeys * numHeads * embedDims);
        valueSpatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueSpatialShapes), numLevels * TWO);
        valueLevelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valuLevelStartIndex), numLevels);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(samplingLocations),
                                   batchSize * numQueries * numHeads * numLevels * numPoints * TWO);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attentionWeights),
                                           batchSize * numQueries * numHeads * numLevels * numPoints);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), batchSize * numQueries * numHeads * embedDims);
        if constexpr (!std::is_same_v<T, float>) {
            workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
                                        batchSize * numQueries * numHeads * embedDims);
        }
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(shapeUb, twoBuffer * numLevelsAlign * sizeof(int32_t));
        pipe->InitBuffer(offsetUb, numLevelsAlign * sizeof(int32_t));
        pipe->InitBuffer(zerosUb, fourBuffer * numQueriesAlign * embedDims * sizeof(float));
        pipe->InitBuffer(outputUb, numQueriesAlign * embedDims * sizeof(float));

        if constexpr (!std::is_same_v<T, float>) {
            pipe->InitBuffer(attentionWeightsUb, twoBuffer * numQueriesAlign * sizeof(float));
            pipe->InitBuffer(locWUb, twoBuffer * numQueriesAlign * sizeof(float));
            pipe->InitBuffer(locHUb, twoBuffer * numQueriesAlign * sizeof(float));
            pipe->InitBuffer(stagingInUb, FOUR * stagingStride * sizeof(float));
            pipe->InitBuffer(castOut1Ub, passChunkMax * sizeof(T));
            pipe->InitBuffer(castOut2Ub, passChunkMax * sizeof(T));
        } else {
            pipe->InitBuffer(attentionWeightsUb, numQueriesAlign * sizeof(float));
            pipe->InitBuffer(locWUb, numQueriesAlign * sizeof(float));
            pipe->InitBuffer(locHUb, numQueriesAlign * sizeof(float));
        }

        pipe->InitBuffer(imUb, twoBuffer * numQueriesAlign * sizeof(float));
        pipe->InitBuffer(lowUb, twoBuffer * numQueriesAlign * sizeof(int32_t));
        pipe->InitBuffer(lowFloatUb, twoBuffer * numQueriesAlign * sizeof(float));
        pipe->InitBuffer(distLowUb, twoBuffer * numQueriesAlign * sizeof(float));
        pipe->InitBuffer(w4Ub, numQueriesAlign * sizeof(float));

        pipe->InitBuffer(wv1Ub, embedDims * sizeof(float));
        pipe->InitBuffer(wv2Ub, embedDims * sizeof(float));
        pipe->InitBuffer(wv3Ub, embedDims * sizeof(float));
        pipe->InitBuffer(wv4Ub, embedDims * sizeof(float));
    }

    __aicore__ inline void GetLocalTensor()
    {
        attentionWeightLocal = attentionWeightsUb.Get<float>();
        shapesLocal = shapeUb.Get<int32_t>();
        offsetLocal = offsetUb.Get<int32_t>();

        locWLocal = locWUb.Get<float>();
        locHLocal = locHUb.Get<float>();

        imLocal = imUb.Get<float>();
        lowLocal = lowUb.Get<int32_t>();
        lowFloatLocal = lowFloatUb.Get<float>();
        zerosLocal = zerosUb.Get<float>();
        distLowLocal = distLowUb.Get<float>();

        w4Local = w4Ub.Get<float>();
        wv1Local = wv1Ub.Get<float>();
        wv2Local = wv2Ub.Get<float>();
        wv3Local = wv3Ub.Get<float>();
        wv4Local = wv4Ub.Get<float>();

        outputLocal = outputUb.Get<float>();

        if constexpr (!std::is_same_v<T, float>) {
            locWHalfView = locWUb.Get<T>();
            locHHalfView = locHUb.Get<T>();
            attnWeightHalfView = attentionWeightsUb.Get<T>();
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if constexpr (!std::is_same_v<T, float>) {
            InitOutput<float>(workspaceGm, batchSize * numQueries * numHeads * embedDims, 0);
        } else {
            InitOutput<T>(outputGm, batchSize * numQueries * numHeads * embedDims, 0);
        }
        if ASCEND_IS_AIV {
            SyncAll();
        }
    }

    __aicore__ inline void Process()
    {
        uint64_t startIdx = startOffset;
        nqloop = startIdx % numQueriesper;
        startIdx = startIdx / numQueriesper;
        point = startIdx % numPoints;
        startIdx = startIdx / numPoints;
        level = startIdx % numLevels;
        startIdx = startIdx / numLevels;
        head = startIdx % numHeads;
        batch = startIdx / numHeads;

        DataCopyPad(shapesLocal, valueSpatialShapesGm, {1, (uint32_t)(TWO * numLevels * sizeof(int32_t)), 0, 0, 0},
                    padParamsInt);
        DataCopyPad(offsetLocal, valueLevelStartIndexGm, {1, (uint32_t)(numLevels * sizeof(int32_t)), 0, 0, 0},
                    padParamsInt);

        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        if constexpr (!std::is_same_v<T, float>) {
            SetFlag<HardEvent::V_MTE2>(eventIdStagingVToMte2);
        }
        for (uint64_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            Compute(taskIdx);
        }
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        if constexpr (!std::is_same_v<T, float>) {
            CastOutputPass();
        }
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        if constexpr (!std::is_same_v<T, float>) {
            pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdStagingVToMte2);
        }
    }

private:
    __aicore__ inline void CopyCastValue(LocalTensor<float> dst, LocalTensor<T> staging, uint64_t offset)
    {
        WaitFlag<HardEvent::V_MTE2>(eventIdStagingVToMte2);
        DataCopyPad(staging, valueGm[offset], copyInParamsV3, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast<float, T>(dst, staging, RoundMode::CAST_NONE, embedDims);
        SetFlag<HardEvent::V_MTE2>(eventIdStagingVToMte2);
    }

    __aicore__ inline void ComputeGradSeparate(float distHH, float distHW, float distLH, float distLW, float w1,
                                               float w2, float w3, float w4, float attentionWeight);

    __aicore__ inline void ComputeGradTogether(float distLH, float distLW, float w1, float w2, float w3, float w4,
                                               float attentionWeight);

    __aicore__ inline void GridSampleCompute();

    __aicore__ inline void ScalarUpdate(uint64_t taskIdx)
    {
        if (taskIdx > startOffset) {
            nqloop += 1;
            point += (nqloop / numQueriesper);
            nqloop %= numQueriesper;
            level += (point / numPoints);
            point %= numPoints;
            head += (level / numLevels);
            level %= numLevels;
            batch += (head / numHeads);
            head %= numHeads;
        }
        levelStartId = offsetLocal.GetValue(level);
        h = shapesLocal.GetValue(level * TWO);
        w = shapesLocal.GetValue(level * TWO + 1);

        // 校验 valueGm 读区间 [levelStartId, levelStartId + h*w) 不越出 numKeys；h/w 非正时 guard 恒假、零读取
        if (static_cast<int64_t>(levelStartId) < 0 ||
            static_cast<int64_t>(levelStartId) + h * w > static_cast<int64_t>(numKeys)) {
            AscendC::Trap();
        }

        offsetWeight = batch * weightStride3 + head * weightStride2 + level * weightStride1 + point * weightStride0;
        offsetLocation = TWO * offsetWeight;
        thisCycleNumAlign = (nqloop == numQueriesper - 1) ? numQueriestail : maxUbNum;
        thisCycleNum = (nqloop == numQueriesper - 1) ? (numQueries - (numQueriesper - 1) * maxUbNum) : maxUbNum;
        offsetValue = batch * valueStride2 + head * valueStride1 + levelStartId * valueStride0;
        hStride = w * wStride;
        copyInParamsV1 = {2, (uint32_t)(2 * embedDims * sizeof(T)), (uint32_t)((hStride - 2 * embedDims) * sizeof(T)),
                          0, 0};
        copyInParamsV2 = {1, (uint32_t)(thisCycleNum * sizeof(T)), 0, 0, 0};
    }

    __aicore__ inline void Compute(uint64_t taskIdx);

    __aicore__ inline void CastOutputPass();

private:
    TPipe* pipe;
    GlobalTensor<T> valueGm, locationGm, attentionWeightsGm, outputGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<int32_t> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> attentionWeightsUb, shapeUb, offsetUb;
    TBuf<TPosition::VECCALC> locWUb, locHUb, imUb, lowUb, lowFloatUb;
    TBuf<TPosition::VECCALC> zerosUb, distLowUb, w4Ub;
    TBuf<TPosition::VECCALC> wv1Ub, wv2Ub, wv3Ub, wv4Ub, outputUb;
    TBuf<TPosition::VECCALC> stagingInUb, castOut1Ub, castOut2Ub;

    uint32_t coreNum;
    uint32_t curBlockIdx;
    uint32_t taskNum, taskNumPerCore;
    uint32_t dataAlign, blockBytes = 32;
    uint32_t queryOffset, queryOffsetv;
    uint32_t v1Id = 0, v2Id = 1, v3Id = 2, v4Id = 3;
    uint32_t thisCycleNum, thisCycleNumAlign, maxUbNum;
    uint32_t twoBuffer = 2, TWO = 2, threeBuffer = 3, fourBuffer = 4, eightBuffer = 8;
    uint32_t numQuerieBuffer = 12, numEmbedBuffer = 5;
    uint32_t useUbSize = 190 * 1024;
    uint32_t stagingStride = 0;
    uint32_t passChunkMax = 1024;
    uint64_t batchSize, numKeys, numHeads, embedDims, numLevels, numQueries, numPoints;
    uint64_t numQueriesAlign, numQueriesper, numQueriestail, numLevelsAlign;
    uint64_t batch, query, head, level, point, nqloop;
    uint64_t startOffset, endOffset;
    uint64_t weightStride0, weightStride1, weightStride2, weightStride3;
    uint64_t valueStride0, valueStride1, valueStride2;
    uint64_t outputStride0, outputStride1, outputStride2;
    uint64_t levelStartId;
    uint64_t offsetValue, offsetWeight, offsetLocation, wStride, hStride;
    uint64_t hLowPtrOffset, wLowPtrOffset;
    int64_t h, w, hLow, wLow;

    float hIm, wIm;
    float w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    float CONV_CONSTANT = -0.5f;

    LocalTensor<float> lowFloatLocal;
    LocalTensor<float> distLowLocal;
    LocalTensor<float> locWLocal, locHLocal;
    LocalTensor<float> imLocal;
    LocalTensor<float> zerosLocal;
    LocalTensor<float> attentionWeightLocal;
    LocalTensor<float> wv1Local, wv2Local, wv3Local, wv4Local, w4Local;
    LocalTensor<float> outputLocal;
    LocalTensor<int32_t> shapesLocal, offsetLocal;
    LocalTensor<int32_t> lowLocal;

    LocalTensor<T> locWHalfView, locHHalfView;
    LocalTensor<T> attnWeightHalfView;

    DataCopyExtParams copyInParamsV1, copyInParamsV2, copyInParamsV3, copyOutParams, copyOutParamsFloat;
    DataCopyPadExtParams<int32_t> padParamsInt{false, 0, 0, 0};
    DataCopyPadExtParams<T> padParamsFloat{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParamsWsIn{false, 0, 0, 0};
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV;
    event_t eventIdStagingVToMte2;
};

#endif // MS_DEFORM_ATTN_GENERIC_H
