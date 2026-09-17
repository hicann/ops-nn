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
 * \file multi_scale_deformable_attention_grad.h
 * \brief
 */
#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

constexpr bool kWsPassEnable = true;

template <typename T>
class MultiScaleDeformableAttentionGrad {
public:
    __aicore__ inline MultiScaleDeformableAttentionGrad(){};
    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                                GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                                GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm, GM_ADDR grad_attn_weight_gm,
                                GM_ADDR grad_value_ws_gm,
                                const MultiScaleDeformableAttentionGradTilingData* __restrict tiling_data,
                                TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        blockBytes = 32;
        dataAlign = blockBytes / sizeof(T);

        ParseTilingData(tiling_data);

        numLevelsAlign = AlignUp(numLevels, dataAlign);
        numQueriesper = DivCeil(numQueries, maxUbNum);
        numQueriestail = numQueries - (numQueriesper - 1) * maxUbNum;
        numQueriestail = AlignUp(numQueriestail, dataAlign);
        numQueriesAlign = numQueries <= maxUbNum ? numQueriestail : maxUbNum;

        stagingStride = AlignUp(embedDims, dataAlign);

        taskNum = batchSize * numHeads * numLevels * numPoints * numQueriesper;
        if (isDeterministic) {
            uint64_t taskPerBH = numLevels * numPoints * numQueriesper;
            uint64_t totalBH = batchSize * numHeads;
            taskNumPerCore = DivCeil(totalBH, coreNum) * taskPerBH;
        } else {
            taskNumPerCore = DivCeil(taskNum, coreNum);
        }

        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        gradOutStride0 = embedDims;
        gradOutStride1 = numHeads * gradOutStride0;
        gradOutStride2 = numQueries * gradOutStride1;

        weightStride0 = numQueries;
        weightStride1 = numPoints * weightStride0;
        weightStride2 = numLevels * weightStride1;
        weightStride3 = numHeads * weightStride2;

        valueStride0 = embedDims;
        valueStride1 = numHeads * valueStride0;
        valueStride2 = numKeys * valueStride1;
        wStride = numHeads * embedDims;

        baseOffsetUb = numQueriesAlign * embedDims;
        copyOutParams = {1, (uint32_t)(embedDims * sizeof(T)), 0, 0, 0};
        copyInParams = {1, (uint32_t)(embedDims * sizeof(T)), 0, 0, 0};
        copyOutParamsWs = {1, (uint32_t)(embedDims * sizeof(float)), 0, 0, 0};

        AllocEvents();
        InitGlobalBuffers(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_loc_gm, attn_weight_gm,
                          grad_output_gm, grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm, grad_value_ws_gm);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(shapeUb, twoBuffer * numLevelsAlign * sizeof(float));
        pipe->InitBuffer(offsetUb, numLevelsAlign * sizeof(float));

        pipe->InitBuffer(zerosUb, eightBuffer * numQueriesAlign * embedDims * sizeof(float));
        pipe->InitBuffer(topGradUb, numQueriesAlign * embedDims * sizeof(float));
        pipe->InitBuffer(mid1Ub, numQueriesAlign * embedDims * sizeof(float));
        pipe->InitBuffer(mid2Ub, numQueriesAlign * embedDims * sizeof(float));
        pipe->InitBuffer(mid3Ub, numQueriesAlign * embedDims * sizeof(float));
        pipe->InitBuffer(mid4Ub, numQueriesAlign * embedDims * sizeof(float));

        if constexpr (!std::is_same<T, float>::value) {
            pipe->InitBuffer(attentionWeightsUb, twoBuffer * numQueriesAlign * sizeof(float));
            pipe->InitBuffer(tmpXUb, twoBuffer * numQueriesAlign * sizeof(float));
            pipe->InitBuffer(tmpYUb, twoBuffer * numQueriesAlign * sizeof(float));
            pipe->InitBuffer(weightSumUb, twoBuffer * numQueriesAlign * sizeof(float));
            pipe->InitBuffer(locWUb, twoBuffer * numQueriesAlign * sizeof(float));
            pipe->InitBuffer(locHUb, twoBuffer * numQueriesAlign * sizeof(float));
        } else {
            pipe->InitBuffer(attentionWeightsUb, numQueriesAlign * sizeof(float));
            pipe->InitBuffer(tmpXUb, numQueriesAlign * sizeof(float));
            pipe->InitBuffer(tmpYUb, numQueriesAlign * sizeof(float));
            pipe->InitBuffer(weightSumUb, numQueriesAlign * sizeof(float));
            pipe->InitBuffer(locWUb, numQueriesAlign * sizeof(float));
            pipe->InitBuffer(locHUb, numQueriesAlign * sizeof(float));
        }
        pipe->InitBuffer(imUb, twoBuffer * numQueriesAlign * sizeof(float));
        pipe->InitBuffer(lowUb, twoBuffer * numQueriesAlign * sizeof(int32_t));
        pipe->InitBuffer(lowFloatUb, twoBuffer * numQueriesAlign * sizeof(float));
        pipe->InitBuffer(distLowUb, twoBuffer * numQueriesAlign * sizeof(float));
        pipe->InitBuffer(w4Ub, numQueriesAlign * sizeof(float));

        pipe->InitBuffer(tmpAUb, embedDims * sizeof(float));
        pipe->InitBuffer(tmpBUb, embedDims * sizeof(float));
        pipe->InitBuffer(wv1Ub, embedDims * sizeof(float));
        pipe->InitBuffer(wv2Ub, embedDims * sizeof(float));
        pipe->InitBuffer(wv3Ub, embedDims * sizeof(float));
        pipe->InitBuffer(wv4Ub, embedDims * sizeof(float));

        if constexpr (!std::is_same<T, float>::value) {
            pipe->InitBuffer(stagingInUb, FOUR * stagingStride * sizeof(float));
        }
    }

    __aicore__ inline void GetLocalTensor()
    {
        attentionWeightLocal = attentionWeightsUb.Get<float>();
        shapesLocal = shapeUb.Get<int32_t>();
        offsetLocal = offsetUb.Get<int32_t>();
        xLocal = tmpXUb.Get<float>();
        yLocal = tmpYUb.Get<float>();
        weightSumLocal = weightSumUb.Get<float>();
        topGradLocal = topGradUb.Get<float>();
        locWLocal = locWUb.Get<float>();
        locHLocal = locHUb.Get<float>();

        imLocal = imUb.Get<float>();
        lowLocal = lowUb.Get<int32_t>();
        lowFloatLocal = lowFloatUb.Get<float>();
        zerosLocal = zerosUb.Get<float>();
        distLowLocal = distLowUb.Get<float>();

        tmpALocal = tmpAUb.Get<float>();
        tmpBLocal = tmpBUb.Get<float>();

        w4Local = w4Ub.Get<float>();
        wv1Local = wv1Ub.Get<float>();
        wv2Local = wv2Ub.Get<float>();
        wv3Local = wv3Ub.Get<float>();
        wv4Local = wv4Ub.Get<float>();

        mid1Local = mid1Ub.Get<float>();
        mid2Local = mid2Ub.Get<float>();
        mid3Local = mid3Ub.Get<float>();
        mid4Local = mid4Ub.Get<float>();

        if constexpr (!std::is_same<T, float>::value) {
            locWHalfView = locWUb.Get<T>();
            locHHalfView = locHUb.Get<T>();
            attnWeightHalfView = attentionWeightsUb.Get<T>();
            xHalfView = tmpXUb.Get<T>();
            yHalfView = tmpYUb.Get<T>();
            weightSumHalfView = weightSumUb.Get<T>();
        }
    }

    __aicore__ inline void ClearOutput()
    {
        switch (curBlockIdx) {
            case 0:
                if constexpr (!std::is_same<T, float>::value) {
                    InitOutput<float>(gradValueWsGm, batchSize * numKeys * numHeads * embedDims, 0);
                } else {
                    InitOutput<T>(gradValueGm, batchSize * numKeys * numHeads * embedDims, 0);
                }
                break;
            case 1:
                InitOutput<T>(gradLocationGm, batchSize * numQueries * numHeads * numLevels * TWO * numPoints, 0);
                break;
            case 2:
                InitOutput<T>(gradWeightGm, batchSize * numQueries * numHeads * numLevels * numPoints, 0);
                break;
            default:
                break;
        }
        PipeBarrier<PIPE_MTE3>();
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
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        for (uint64_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            Compute(taskIdx);
        }
        WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        if constexpr (!std::is_same<T, float>::value) {
            if constexpr (kWsPassEnable) {
                CastGradValuePass();
            }
        }
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMteWeight);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3X);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3Y);
        pipe->ReleaseEventID<HardEvent::MTE3_S>(eventIdMte3ToS);
        pipe->ReleaseEventID<HardEvent::MTE2_S>(eventIdMte2ToS);
        pipe->ReleaseEventID<HardEvent::V_S>(eventIdVToS);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdStagVToMte2);
    }

private:
    __aicore__ inline void ParseTilingData(const MultiScaleDeformableAttentionGradTilingData* __restrict tiling_data)
    {
        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        embedDims = tiling_data->embedDims;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        batchSize = tiling_data->batchSize;
        maxUbNum = tiling_data->maxUbNum;
        coreNum = tiling_data->coreNum;
        isDeterministic = tiling_data->isDeterministic;
    }

    __aicore__ inline void AllocEvents()
    {
        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMteWeight = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3X = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3Y = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_S>());
        eventIdMte2ToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_S>());
        eventIdVToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_S>());
        eventIdStagVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(eventIdStagVToMte2);
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                                             GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                                             GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm,
                                             GM_ADDR grad_attn_weight_gm, GM_ADDR grad_value_ws_gm)
    {
        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value_gm), batchSize * numKeys * numHeads * embedDims);
        valueSpatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(spatial_shapes_gm), numLevels * TWO);
        valueLevelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(level_start_index_gm), numLevels);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sampling_loc_gm),
                                   batchSize * numQueries * numHeads * numLevels * numPoints * TWO);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attn_weight_gm),
                                           batchSize * numQueries * numHeads * numLevels * numPoints);
        gradOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_output_gm),
                                     batchSize * numQueries * numHeads * embedDims);

        gradValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_value_gm),
                                    batchSize * numKeys * numHeads * embedDims);
        gradLocationGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_sampling_loc_gm),
                                       batchSize * numQueries * numHeads * numLevels * TWO * numPoints);
        gradWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_attn_weight_gm),
                                     batchSize * numQueries * numHeads * numLevels * numPoints);
        gradValueWsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(grad_value_ws_gm),
                                      batchSize * numKeys * numHeads * embedDims);
    }

    template <bool AddH, bool AddW>
    __aicore__ inline void ComputeGrad(const LocalTensor<float>& wvLocal, const LocalTensor<float>& mid, uint32_t vId,
                                       float distH, float distW, uint64_t hPtrOffset, uint64_t wPtrOffset, float w)
    {
        if constexpr (!std::is_same<T, float>::value) {
            auto stagingInHalf = stagingInUb.Get<T>();
            WaitFlag<HardEvent::V_MTE2>(eventIdStagVToMte2);
            DataCopyPad(stagingInHalf, valueGm[offsetValue + hPtrOffset + wPtrOffset], copyInParams, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(mid[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w, embedDims);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast<float, T>(zerosLocal[vId * embedDims + queryOffsetv + FOUR * baseOffsetUb], stagingInHalf,
                           RoundMode::CAST_NONE, embedDims);
            SetFlag<HardEvent::V_MTE2>(eventIdStagVToMte2);
        } else {
            DataCopyPad(zerosLocal[vId * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                        valueGm[offsetValue + hPtrOffset + wPtrOffset], copyInParams, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(mid[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w, embedDims);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }

        Muls(wvLocal, zerosLocal[vId * embedDims + queryOffsetv + FOUR * baseOffsetUb], w, embedDims);
        Muls(tmpALocal, zerosLocal[vId * embedDims + queryOffsetv + FOUR * baseOffsetUb], distW, embedDims);
        Muls(tmpBLocal, zerosLocal[vId * embedDims + queryOffsetv + FOUR * baseOffsetUb], distH, embedDims);
        if (AddH) {
            Add(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDims);
        } else {
            Sub(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDims);
        }
        if (AddW) {
            Add(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDims);
        } else {
            Sub(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDims);
        }

        if constexpr (!std::is_same<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            if (isDeterministic) {
                PipeBarrier<PIPE_MTE3>();
            }
            DataCopyPad(gradValueWsGm[offsetValue + hPtrOffset + wPtrOffset], mid[queryOffset], copyOutParamsWs);
        } else {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            if (isDeterministic) {
                PipeBarrier<PIPE_MTE3>();
            }
            DataCopyPad(gradValueGm[offsetValue + hPtrOffset + wPtrOffset], mid[queryOffset], copyOutParams);
        }
    }

    __aicore__ inline void ComputeGradSeparate(float distHH, float distHW, float distLH, float distLW, float w1,
                                               float w2, float w3, float w4, float attentionWeight)
    {
        Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb], topGradLocal[query * embedDims], attentionWeight,
             embedDims);
        if (hLow >= 0 && wLow >= 0) {
            ComputeGrad<false, false>(wv1Local, mid1Local, v1Id, distHH, distHW, hLowPtrOffset, wLowPtrOffset, w1);
        }
        if (hLow >= 0 && wLow < w - 1) {
            ComputeGrad<false, true>(wv2Local, mid2Local, v2Id, distHH, distLW, hLowPtrOffset, wLowPtrOffset + wStride,
                                     w2);
        }
        if (hLow < h - 1 && wLow >= 0) {
            ComputeGrad<true, false>(wv3Local, mid3Local, v3Id, distLH, distHW, hLowPtrOffset + hStride, wLowPtrOffset,
                                     w3);
        }
        if (hLow < h - 1 && wLow < w - 1) {
            ComputeGrad<true, true>(wv4Local, mid4Local, v4Id, distLH, distLW, hLowPtrOffset + hStride,
                                    wLowPtrOffset + wStride, w4);
        }
        Add(wv1Local, wv1Local, wv2Local, embedDims);
        Add(wv3Local, wv3Local, wv4Local, embedDims);
        Add(wv1Local, wv1Local, wv3Local, embedDims);
        Mul(zerosLocal[queryOffset + gradWeightId * baseOffsetUb], topGradLocal[query * embedDims], wv1Local,
            embedDims);
    }

    __aicore__ inline void ComputeGradTogether(float distLH, float distLW, float w1, float w2, float w3, float w4,
                                               float attentionWeight)
    {
        if constexpr (!std::is_same<T, float>::value) {
            auto stagingInHalf = stagingInUb.Get<T>();
            WaitFlag<HardEvent::V_MTE2>(eventIdStagVToMte2);
            DataCopyPad(stagingInHalf, valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], copyInParams,
                        padParamsFloat);
            DataCopyPad(stagingInHalf[stagingStride], valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride],
                        copyInParams, padParamsFloat);
            DataCopyPad(stagingInHalf[twoBuffer * stagingStride],
                        valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], copyInParams, padParamsFloat);
            DataCopyPad(stagingInHalf[3 * stagingStride],
                        valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride], copyInParams,
                        padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb], topGradLocal[query * embedDims],
                 attentionWeight, embedDims);

            Muls(mid1Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w1, embedDims);
            Muls(mid2Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w2, embedDims);
            Muls(mid3Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w3, embedDims);
            Muls(mid4Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w4, embedDims);

            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast<float, T>(zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], stagingInHalf,
                           RoundMode::CAST_NONE, embedDims);
            Cast<float, T>(zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                           stagingInHalf[stagingStride], RoundMode::CAST_NONE, embedDims);
            Cast<float, T>(zerosLocal[v3Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                           stagingInHalf[twoBuffer * stagingStride], RoundMode::CAST_NONE, embedDims);
            Cast<float, T>(zerosLocal[v4Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                           stagingInHalf[3 * stagingStride], RoundMode::CAST_NONE, embedDims);
            SetFlag<HardEvent::V_MTE2>(eventIdStagVToMte2);
        } else {
            DataCopyPad(zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                        valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], copyInParams, padParamsFloat);
            DataCopyPad(zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                        valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride], copyInParams, padParamsFloat);
            DataCopyPad(zerosLocal[v3Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                        valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], copyInParams, padParamsFloat);
            DataCopyPad(zerosLocal[v4Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                        valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride], copyInParams,
                        padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb], topGradLocal[query * embedDims],
                 attentionWeight, embedDims);

            Muls(mid1Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w1, embedDims);
            Muls(mid2Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w2, embedDims);
            Muls(mid3Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w3, embedDims);
            Muls(mid4Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w4, embedDims);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }

        Muls(wv1Local, zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], w1, embedDims);
        Muls(wv2Local, zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], w2, embedDims);
        Muls(wv3Local, zerosLocal[v3Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], w3, embedDims);
        Muls(wv4Local, zerosLocal[v4Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], w4, embedDims);

        Sub(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
            zerosLocal[v3Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
            zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], embedDims);
        Sub(tmpALocal, zerosLocal[v4Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
            zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], embedDims);
        Sub(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
            zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
            zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], embedDims);
        Sub(tmpALocal, tmpALocal, zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], embedDims);
        Muls(tmpBLocal, tmpALocal, distLH, embedDims);
        Muls(tmpALocal, tmpALocal, distLW, embedDims);
        Add(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
            zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDims);
        Add(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
            zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDims);

        Add(wv1Local, wv1Local, wv2Local, embedDims);
        Add(wv3Local, wv3Local, wv4Local, embedDims);
        Add(wv1Local, wv1Local, wv3Local, embedDims);
        Mul(zerosLocal[queryOffset + gradWeightId * baseOffsetUb], topGradLocal[query * embedDims], wv1Local,
            embedDims);

        if constexpr (!std::is_same<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            if (isDeterministic) {
                PipeBarrier<PIPE_MTE3>();
            }
            DataCopyPad(gradValueWsGm[offsetValue + hLowPtrOffset + wLowPtrOffset], mid1Local[queryOffset],
                        copyOutParamsWs);
            DataCopyPad(gradValueWsGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride], mid2Local[queryOffset],
                        copyOutParamsWs);
            DataCopyPad(gradValueWsGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], mid3Local[queryOffset],
                        copyOutParamsWs);
            DataCopyPad(gradValueWsGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride],
                        mid4Local[queryOffset], copyOutParamsWs);
        } else {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            if (isDeterministic) {
                PipeBarrier<PIPE_MTE3>();
            }
            DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], mid1Local[queryOffset],
                        copyOutParams);
            DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride], mid2Local[queryOffset],
                        copyOutParams);
            DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], mid3Local[queryOffset],
                        copyOutParams);
            DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride],
                        mid4Local[queryOffset], copyOutParams);
        }
    }

    __aicore__ inline void GridSampleCompute()
    {
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        SetAtomicAdd<float>();
        for (query = 0; query < thisCycleNum; query++) {
            queryOffset = query * embedDims;
            queryOffsetv = query * FOUR * embedDims;
            hIm = imLocal.GetValue(thisCycleNumAlign + query);
            wIm = imLocal.GetValue(query);
            if (hIm > -1 && wIm > -1 && hIm < h && wIm < w) {
                hLow = lowLocal.GetValue(thisCycleNumAlign + query);
                wLow = lowLocal.GetValue(query);
                hLowPtrOffset = hLow * hStride;
                wLowPtrOffset = wLow * wStride;
                float distH = distLowLocal.GetValue(thisCycleNumAlign + query);
                float distW = distLowLocal.GetValue(query);
                float attenWeight = attentionWeightLocal.GetValue(query);
                w4 = w4Local.GetValue(query);
                w1 = w4 + 1 - distH - distW;
                w2 = distW - w4;
                w3 = distH - w4;
                if (hLow >= 0 && wLow >= 0 && hLow < h - 1 && wLow < w - 1) {
                    ComputeGradTogether(distH, distW, w1, w2, w3, w4, attenWeight);
                } else {
                    Duplicate(wv1Local, (float)0, embedDims);
                    Duplicate(wv2Local, (float)0, embedDims);
                    Duplicate(wv3Local, (float)0, embedDims);
                    Duplicate(wv4Local, (float)0, embedDims);
                    ComputeGradSeparate(1 - distH, 1 - distW, distH, distW, w1, w2, w3, w4, attenWeight);
                }
            }
        }
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

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
        if (h <= 0 || w <= 0 || levelStartId > numKeys ||
            levelStartId + static_cast<uint64_t>(h) * static_cast<uint64_t>(w) > numKeys) {
            ascendc_assert(false,
                           "MSDA grad invalid spatial shape at level %u: H=%d, W=%d, levelStartId=%u, "
                           "H*W=%d, numKeys=%u\n",
                           level, h, w, levelStartId, h * w, numKeys);
            Trap();
            return;
        }
        offsetGrad = batch * gradOutStride2 + nqloop * maxUbNum * gradOutStride1 + head * gradOutStride0;
        offsetWeight = batch * weightStride3 + head * weightStride2 + level * weightStride1 + point * weightStride0;
        offsetLocation = TWO * offsetWeight;
        thisCycleNumAlign = (nqloop == numQueriesper - 1) ? numQueriestail : maxUbNum;
        thisCycleNum = (nqloop == numQueriesper - 1) ? (numQueries - (numQueriesper - 1) * maxUbNum) : maxUbNum;
        offsetValue = batch * valueStride2 + levelStartId * valueStride1 + head * valueStride0;
        hStride = w * wStride;
        copyParams = {1, (uint32_t)(thisCycleNum * sizeof(T)), 0, 0, 0};
        sumParams = {thisCycleNum, embedDims, embedDims};
    }

    __aicore__ inline void Compute(uint64_t taskIdx)
    {
        ScalarUpdate(taskIdx);

        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        Duplicate(zerosLocal, (float)0, eightBuffer * numQueriesAlign * embedDims);

        if constexpr (!std::is_same<T, float>::value) {
            DataCopyPad(locWHalfView[twoBuffer * numQueriesAlign], locationGm[offsetLocation + nqloop * maxUbNum],
                        copyParams, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast<float, T>(locWLocal, locWHalfView[twoBuffer * numQueriesAlign], RoundMode::CAST_NONE,
                           thisCycleNumAlign);
        } else {
            DataCopyPad(locWLocal, locationGm[offsetLocation + nqloop * maxUbNum], copyParams, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }
        Muls(imLocal, locWLocal, (float)w, thisCycleNumAlign);

        if constexpr (!std::is_same<T, float>::value) {
            DataCopyPad(locHHalfView[twoBuffer * numQueriesAlign],
                        locationGm[offsetLocation + numQueries + nqloop * maxUbNum], copyParams, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast<float, T>(locHLocal, locHHalfView[twoBuffer * numQueriesAlign], RoundMode::CAST_NONE,
                           thisCycleNumAlign);
        } else {
            DataCopyPad(locHLocal, locationGm[offsetLocation + numQueries + nqloop * maxUbNum], copyParams,
                        padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }
        Muls(imLocal[thisCycleNumAlign], locHLocal, (float)h, thisCycleNumAlign);

        if constexpr (!std::is_same<T, float>::value) {
            DataCopyPad(attnWeightHalfView[twoBuffer * numQueriesAlign],
                        attentionWeightsGm[offsetWeight + nqloop * maxUbNum], copyParams, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast<float, T>(attentionWeightLocal, attnWeightHalfView[twoBuffer * numQueriesAlign], RoundMode::CAST_NONE,
                           thisCycleNumAlign);
        } else {
            DataCopyPad(attentionWeightLocal, attentionWeightsGm[offsetWeight + nqloop * maxUbNum], copyParams,
                        padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }

        if constexpr (!std::is_same<T, float>::value) {
            auto stagingInHalf = stagingInUb.Get<T>();
            for (query = 0; query < thisCycleNum; query++) {
                auto stagingSlot = stagingInHalf[(query % TWO) * stagingStride];
                WaitFlag<HardEvent::V_MTE2>(eventIdStagVToMte2);
                DataCopyPad(stagingSlot, gradOutputGm[offsetGrad + query * gradOutStride1], copyInParams,
                            padParamsFloat);
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                Cast<float, T>(topGradLocal[query * embedDims], stagingSlot, RoundMode::CAST_NONE, embedDims);
                SetFlag<HardEvent::V_MTE2>(eventIdStagVToMte2);
            }
        } else {
            for (query = 0; query < thisCycleNum; query++) {
                DataCopyPad(topGradLocal[query * embedDims], gradOutputGm[offsetGrad + query * gradOutStride1],
                            copyInParams, padParamsFloat);
            }
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }

        Adds(imLocal, imLocal, CONV_CONSTANT, TWO * thisCycleNumAlign);
        Cast(lowLocal, imLocal, RoundMode::CAST_FLOOR, TWO * thisCycleNumAlign);
        Cast(lowFloatLocal, lowLocal, RoundMode::CAST_NONE, TWO * thisCycleNumAlign);
        Sub(distLowLocal, imLocal, lowFloatLocal, TWO * thisCycleNumAlign);
        Mul(w4Local, distLowLocal[thisCycleNumAlign], distLowLocal, thisCycleNumAlign);
        SetFlag<HardEvent::V_S>(eventIdVToS);

        if constexpr (std::is_same<T, float>::value) {
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }
        GridSampleCompute();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);

        Mul(zerosLocal[gradWWeightId * baseOffsetUb], zerosLocal[topGradValueId * baseOffsetUb],
            zerosLocal[gradWWeightId * baseOffsetUb], thisCycleNum * embedDims);
        Mul(zerosLocal[gradHWeightId * baseOffsetUb], zerosLocal[topGradValueId * baseOffsetUb],
            zerosLocal[gradHWeightId * baseOffsetUb], thisCycleNum * embedDims);
        Muls(zerosLocal[gradWWeightId * baseOffsetUb], zerosLocal[gradWWeightId * baseOffsetUb], (float)w,
             thisCycleNum * embedDims);
        Muls(zerosLocal[gradHWeightId * baseOffsetUb], zerosLocal[gradHWeightId * baseOffsetUb], (float)h,
             thisCycleNum * embedDims);

        WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);

        if constexpr (!std::is_same<T, float>::value) {
            Sum(weightSumLocal, zerosLocal[gradWeightId * baseOffsetUb], sumParams);
            PipeBarrier<PIPE_V>();
            Cast<T, float>(weightSumHalfView[twoBuffer * numQueriesAlign], weightSumLocal, RoundMode::CAST_RINT,
                           thisCycleNumAlign);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
            Sum(xLocal, zerosLocal[gradWWeightId * baseOffsetUb], sumParams);
            PipeBarrier<PIPE_V>();
            Cast<T, float>(xHalfView[twoBuffer * numQueriesAlign], xLocal, RoundMode::CAST_RINT, thisCycleNumAlign);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
            Sum(yLocal, zerosLocal[gradHWeightId * baseOffsetUb], sumParams);
            PipeBarrier<PIPE_V>();
            Cast<T, float>(yHalfView[twoBuffer * numQueriesAlign], yLocal, RoundMode::CAST_RINT, thisCycleNumAlign);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Y);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
            DataCopyPad(gradWeightGm[offsetWeight + nqloop * maxUbNum], weightSumHalfView[twoBuffer * numQueriesAlign],
                        copyParams);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
            DataCopyPad(gradLocationGm[offsetLocation + nqloop * maxUbNum], xHalfView[twoBuffer * numQueriesAlign],
                        copyParams);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Y);
            DataCopyPad(gradLocationGm[offsetLocation + numQueries + nqloop * maxUbNum],
                        yHalfView[twoBuffer * numQueriesAlign], copyParams);
        } else {
            Sum(weightSumLocal, zerosLocal[gradWeightId * baseOffsetUb], sumParams);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
            Sum(xLocal, zerosLocal[gradWWeightId * baseOffsetUb], sumParams);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
            Sum(yLocal, zerosLocal[gradHWeightId * baseOffsetUb], sumParams);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Y);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
            DataCopyPad(gradWeightGm[offsetWeight + nqloop * maxUbNum], weightSumLocal, copyParams);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
            DataCopyPad(gradLocationGm[offsetLocation + nqloop * maxUbNum], xLocal, copyParams);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Y);
            DataCopyPad(gradLocationGm[offsetLocation + numQueries + nqloop * maxUbNum], yLocal, copyParams);
        }
        SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    }

    __aicore__ inline void CastGradValuePass()
    {
        if ASCEND_IS_AIV {
            SyncAll();
        }
        uint64_t total = batchSize * numKeys * numHeads * embedDims;
        uint64_t perCore = DivCeil(total, (uint64_t)coreNum);
        uint64_t rangeStart = curBlockIdx * perCore;
        if (rangeStart >= total) {
            return;
        }
        uint64_t rangeEnd = rangeStart + perCore;
        if (rangeEnd > total) {
            rangeEnd = total;
        }
        uint64_t chunk = (uint64_t)numQueriesAlign * embedDims;
        if (chunk > passChunkMax) {
            chunk = passChunkMax;
        }
        uint64_t numIter = DivCeil(rangeEnd - rangeStart, chunk);
        auto out0 = mid1Ub.Get<T>();
        auto out1 = mid2Ub.Get<T>();
        uint64_t firstLen = (rangeEnd - rangeStart < chunk) ? (rangeEnd - rangeStart) : chunk;
        DataCopyPad(zerosLocal, gradValueWsGm[rangeStart], {1, (uint32_t)(firstLen * sizeof(float)), 0, 0, 0},
                    padParamsWsIn);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        for (uint64_t i = 0; i < numIter; i++) {
            uint64_t off = rangeStart + i * chunk;
            uint32_t cur = (uint32_t)((off + chunk > rangeEnd) ? (rangeEnd - off) : chunk);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            if (i % 2 == 0) {
                Cast<T, float>(out0, zerosLocal, RoundMode::CAST_RINT, cur);
            } else {
                Cast<T, float>(out1, zerosLocal[chunk], RoundMode::CAST_RINT, cur);
            }
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            if (i % 2 == 0) {
                DataCopyPad(gradValueGm[off], out0, {1, (uint32_t)(cur * sizeof(T)), 0, 0, 0});
            } else {
                DataCopyPad(gradValueGm[off], out1, {1, (uint32_t)(cur * sizeof(T)), 0, 0, 0});
            }
            if (i + 1 < numIter) {
                uint64_t nextOff = off + chunk;
                uint64_t nextLen = (nextOff + chunk > rangeEnd) ? (rangeEnd - nextOff) : chunk;
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                if ((i + 1) % 2 == 0) {
                    DataCopyPad(zerosLocal, gradValueWsGm[nextOff], {1, (uint32_t)(nextLen * sizeof(float)), 0, 0, 0},
                                padParamsWsIn);
                } else {
                    DataCopyPad(zerosLocal[chunk], gradValueWsGm[nextOff],
                                {1, (uint32_t)(nextLen * sizeof(float)), 0, 0, 0}, padParamsWsIn);
                }
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            }
        }
    }

private:
    TPipe* pipe;
    GlobalTensor<T> valueGm;
    GlobalTensor<T> locationGm;
    GlobalTensor<T> attentionWeightsGm;
    GlobalTensor<T> gradOutputGm;
    GlobalTensor<T> gradValueGm;
    GlobalTensor<T> gradLocationGm;
    GlobalTensor<T> gradWeightGm;
    GlobalTensor<float> gradValueWsGm;
    GlobalTensor<int32_t> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> attentionWeightsUb, shapeUb, offsetUb, topGradUb;
    TBuf<TPosition::VECCALC> tmpXUb, tmpYUb, weightSumUb;
    TBuf<TPosition::VECCALC> zerosUb;
    TBuf<TPosition::VECCALC> locWUb, locHUb, imUb, lowUb, lowFloatUb;
    TBuf<TPosition::VECCALC> distLowUb, w4Ub;
    TBuf<TPosition::VECCALC> tmpAUb, tmpBUb;
    TBuf<TPosition::VECCALC> stagingInUb;

    uint32_t coreNum;
    uint32_t embedDims;
    uint32_t curBlockIdx;
    uint64_t isDeterministic = 0;
    uint32_t dataAlign, blockBytes;
    uint32_t stagingStride = 0;
    uint32_t passChunkMax = 1024;
    uint32_t hOffsetUb, baseOffsetUb, queryOffset, queryOffsetv;
    uint32_t gradHWeightId = 0, gradWWeightId = 1, topGradValueId = 2, gradWeightId = 3;
    uint32_t v1Id = 0, v2Id = 1, v3Id = 2, v4Id = 3;
    uint32_t thisCycleNum, thisCycleNumAlign, maxUbNum;
    uint32_t twoBuffer = 2, TWO = 2, FOUR = 4, eightBuffer = 8;
    uint64_t batchSize, numKeys, numHeads, numLevels, numQueries, numPoints;
    uint64_t numQueriesAlign, numQueriesper, numQueriestail, numLevelsAlign;
    uint64_t batch, query, head, level, point, nqloop;
    uint64_t taskNum, taskNumPerCore;
    uint64_t startOffset, endOffset;
    uint64_t gradOutStride0, gradOutStride1, gradOutStride2;
    uint64_t weightStride0, weightStride1, weightStride2, weightStride3;
    uint64_t valueStride0, valueStride1, valueStride2;
    uint64_t levelStartId;
    uint64_t offsetValue, offsetWeight, offsetLocation, offsetGrad, wStride, hStride;
    uint64_t hLowPtrOffset, wLowPtrOffset;
    int64_t h, w, hLow, wLow;

    float hIm, wIm;
    float w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    float CONV_CONSTANT = -0.5f;

    LocalTensor<float> lowFloatLocal;
    LocalTensor<float> xLocal, yLocal;
    LocalTensor<float> distLowLocal;
    LocalTensor<float> locWLocal, locHLocal;
    LocalTensor<float> imLocal;
    LocalTensor<float> zerosLocal;
    LocalTensor<float> weightSumLocal, tmpALocal, tmpBLocal;
    LocalTensor<float> topGradLocal, attentionWeightLocal;
    LocalTensor<float> wv1Local, wv2Local, wv3Local, wv4Local, w4Local;
    LocalTensor<float> mid1Local, mid2Local, mid3Local, mid4Local;
    LocalTensor<int32_t> shapesLocal, offsetLocal;
    LocalTensor<int32_t> lowLocal;

    LocalTensor<T> locWHalfView, locHHalfView;
    LocalTensor<T> attnWeightHalfView;
    LocalTensor<T> xHalfView, yHalfView;
    LocalTensor<T> weightSumHalfView;

    SumParams sumParams;
    DataCopyExtParams copyParams, copyInParams, copyOutParams, copyOutParamsWs;
    DataCopyPadExtParams<int32_t> padParamsInt{false, 0, 0, 0};
    DataCopyPadExtParams<T> padParamsFloat{false, 0, 0, 0};
    DataCopyPadExtParams<float> padParamsWsIn{false, 0, 0, 0};
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV, eventIdVToMteWeight, eventIdVToMte3X,
        eventIdVToMte3Y, eventIdMte3ToS, eventIdMte2ToS, eventIdVToS, eventIdStagVToMte2;

    TBuf<TPosition::VECCALC> wv1Ub, wv2Ub, wv3Ub, wv4Ub, mid1Ub, mid2Ub, mid3Ub, mid4Ub;
};
