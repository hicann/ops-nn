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
 * \file multi_scale_deformable_attention_grad_impl.h
 * \brief
 */
#ifndef MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_IMPL_H
#define MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_IMPL_H

#include "multi_scale_deformable_attention_grad.h"

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::ComputeGradSeparate(float distHH, float distHW,
                                                                                 float distLH, float distLW, float w1,
                                                                                 float w2, float w3, float w4,
                                                                                 float attentionWeight)
{
    Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb], topGradLocal[query * embedDims], attentionWeight,
         embedDims);
    if (hLow >= 0 && wLow >= 0) {
        ComputeGrad<false, false>(wv1Local, mid1Local, v1Id, distHH, distHW, hLowPtrOffset, wLowPtrOffset, w1);
    }
    if (hLow >= 0 && wLow < w - 1) {
        ComputeGrad<false, true>(wv2Local, mid2Local, v2Id, distHH, distLW, hLowPtrOffset, wLowPtrOffset + wStride, w2);
    }
    if (hLow < h - 1 && wLow >= 0) {
        ComputeGrad<true, false>(wv3Local, mid3Local, v3Id, distLH, distHW, hLowPtrOffset + hStride, wLowPtrOffset, w3);
    }
    if (hLow < h - 1 && wLow < w - 1) {
        ComputeGrad<true, true>(wv4Local, mid4Local, v4Id, distLH, distLW, hLowPtrOffset + hStride,
                                wLowPtrOffset + wStride, w4);
    }
    Add(wv1Local, wv1Local, wv2Local, embedDims);
    Add(wv3Local, wv3Local, wv4Local, embedDims);
    Add(wv1Local, wv1Local, wv3Local, embedDims);
    Mul(zerosLocal[queryOffset + gradWeightId * baseOffsetUb], topGradLocal[query * embedDims], wv1Local, embedDims);
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::ComputeGradTogether(float distLH, float distLW, float w1,
                                                                                 float w2, float w3, float w4,
                                                                                 float attentionWeight)
{
    if constexpr (!std::is_same<T, float>::value) {
        auto stagingInHalf = stagingInUb.Get<T>();
        WaitFlag<HardEvent::V_MTE2>(eventIdStagVToMte2);
        DataCopyPad(stagingInHalf, valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], copyInParams, padParamsFloat);
        DataCopyPad(stagingInHalf[stagingStride], valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride],
                    copyInParams, padParamsFloat);
        DataCopyPad(stagingInHalf[twoBuffer * stagingStride],
                    valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], copyInParams, padParamsFloat);
        DataCopyPad(stagingInHalf[v4Id * stagingStride],
                    valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride], copyInParams,
                    padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb], topGradLocal[query * embedDims], attentionWeight,
             embedDims);

        Muls(mid1Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w1, embedDims);
        Muls(mid2Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w2, embedDims);
        Muls(mid3Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w3, embedDims);
        Muls(mid4Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w4, embedDims);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast<float, T>(zerosLocal[v1Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], stagingInHalf,
                       RoundMode::CAST_NONE, embedDims);
        Cast<float, T>(zerosLocal[v2Id * embedDims + queryOffsetv + FOUR * baseOffsetUb], stagingInHalf[stagingStride],
                       RoundMode::CAST_NONE, embedDims);
        Cast<float, T>(zerosLocal[v3Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                       stagingInHalf[twoBuffer * stagingStride], RoundMode::CAST_NONE, embedDims);
        Cast<float, T>(zerosLocal[v4Id * embedDims + queryOffsetv + FOUR * baseOffsetUb],
                       stagingInHalf[v4Id * stagingStride], RoundMode::CAST_NONE, embedDims);
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

        Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb], topGradLocal[query * embedDims], attentionWeight,
             embedDims);

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
    Add(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
        tmpBLocal, embedDims);
    Add(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
        tmpALocal, embedDims);

    Add(wv1Local, wv1Local, wv2Local, embedDims);
    Add(wv3Local, wv3Local, wv4Local, embedDims);
    Add(wv1Local, wv1Local, wv3Local, embedDims);
    Mul(zerosLocal[queryOffset + gradWeightId * baseOffsetUb], topGradLocal[query * embedDims], wv1Local, embedDims);

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
        DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], mid1Local[queryOffset], copyOutParams);
        DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride], mid2Local[queryOffset],
                    copyOutParams);
        DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], mid3Local[queryOffset],
                    copyOutParams);
        DataCopyPad(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride],
                    mid4Local[queryOffset], copyOutParams);
    }
}

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::GridSampleCompute()
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

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::Compute(uint64_t taskIdx)
{
    ScalarUpdate(taskIdx);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    Duplicate(zerosLocal, (float)0, eightBuffer * numQueriesAlign * embedDims);

    if constexpr (!std::is_same<T, float>::value) {
        DataCopyPad(locWHalfView[twoBuffer * numQueriesAlign], locationGm[offsetLocation + nqloop * maxUbNum],
                    copyParams, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast<float, T>(locWLocal, locWHalfView[twoBuffer * numQueriesAlign], RoundMode::CAST_NONE, thisCycleNumAlign);
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
        Cast<float, T>(locHLocal, locHHalfView[twoBuffer * numQueriesAlign], RoundMode::CAST_NONE, thisCycleNumAlign);
    } else {
        DataCopyPad(locHLocal, locationGm[offsetLocation + numQueries + nqloop * maxUbNum], copyParams, padParamsFloat);
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
            DataCopyPad(stagingSlot, gradOutputGm[offsetGrad + query * gradOutStride1], copyInParams, padParamsFloat);
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

template <typename T>
__aicore__ inline void MultiScaleDeformableAttentionGrad<T>::CastGradValuePass()
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
        if (i % twoBuffer == 0) {
            Cast<T, float>(out0, zerosLocal, RoundMode::CAST_RINT, cur);
        } else {
            Cast<T, float>(out1, zerosLocal[chunk], RoundMode::CAST_RINT, cur);
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        if (i % twoBuffer == 0) {
            DataCopyPad(gradValueGm[off], out0, {1, (uint32_t)(cur * sizeof(T)), 0, 0, 0});
        } else {
            DataCopyPad(gradValueGm[off], out1, {1, (uint32_t)(cur * sizeof(T)), 0, 0, 0});
        }
        if (i + 1 < numIter) {
            uint64_t nextOff = off + chunk;
            uint64_t nextLen = (nextOff + chunk > rangeEnd) ? (rangeEnd - nextOff) : chunk;
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            if ((i + 1) % twoBuffer == 0) {
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

#endif // MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_IMPL_H
