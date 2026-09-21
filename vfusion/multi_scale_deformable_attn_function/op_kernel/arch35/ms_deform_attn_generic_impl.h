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
 * \file ms_deform_attn_generic_impl.h
 * \brief
 */
#ifndef MS_DEFORM_ATTN_GENERIC_IMPL_H
#define MS_DEFORM_ATTN_GENERIC_IMPL_H

#include "ms_deform_attn_generic.h"

template <typename TilingDataT, typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn<TilingDataT, T>::ComputeGradSeparate(float distHH, float distHW,
                                                                                           float distLH, float distLW,
                                                                                           float w1, float w2, float w3,
                                                                                           float w4,
                                                                                           float attentionWeight)
{
    if constexpr (!std::is_same_v<T, float>) {
        auto stagingInHalf = stagingInUb.Get<T>();
        if (hLow >= 0 && wLow >= 0) {
            CopyCastValue(zerosLocal[v1Id * embedDims + queryOffsetv], stagingInHalf,
                          offsetValue + hLowPtrOffset + wLowPtrOffset);
            Muls(wv1Local, zerosLocal[v1Id * embedDims + queryOffsetv], w1, embedDims);
        }
        if (hLow >= 0 && wLow < w - 1) {
            CopyCastValue(zerosLocal[v2Id * embedDims + queryOffsetv], stagingInHalf,
                          offsetValue + hLowPtrOffset + wLowPtrOffset + wStride);
            Muls(wv2Local, zerosLocal[v2Id * embedDims + queryOffsetv], w2, embedDims);
        }
        if (hLow < h - 1 && wLow >= 0) {
            CopyCastValue(zerosLocal[v3Id * embedDims + queryOffsetv], stagingInHalf,
                          offsetValue + hLowPtrOffset + hStride + wLowPtrOffset);
            Muls(wv3Local, zerosLocal[v3Id * embedDims + queryOffsetv], w3, embedDims);
        }
        if (hLow < h - 1 && wLow < w - 1) {
            CopyCastValue(zerosLocal[v4Id * embedDims + queryOffsetv], stagingInHalf,
                          offsetValue + hLowPtrOffset + hStride + wLowPtrOffset + wStride);
            Muls(wv4Local, zerosLocal[v4Id * embedDims + queryOffsetv], w4, embedDims);
        }
    } else {
        if (hLow >= 0 && wLow >= 0) {
            DataCopyPad(zerosLocal[v1Id * embedDims + queryOffsetv],
                        valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], copyInParamsV3, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(wv1Local, zerosLocal[v1Id * embedDims + queryOffsetv], w1, embedDims);
        }
        if (hLow >= 0 && wLow < w - 1) {
            DataCopyPad(zerosLocal[v2Id * embedDims + queryOffsetv],
                        valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride], copyInParamsV3, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(wv2Local, zerosLocal[v2Id * embedDims + queryOffsetv], w2, embedDims);
        }
        if (hLow < h - 1 && wLow >= 0) {
            DataCopyPad(zerosLocal[v3Id * embedDims + queryOffsetv],
                        valueGm[offsetValue + hLowPtrOffset + hStride + wLowPtrOffset], copyInParamsV3, padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(wv3Local, zerosLocal[v3Id * embedDims + queryOffsetv], w3, embedDims);
        }
        if (hLow < h - 1 && wLow < w - 1) {
            DataCopyPad(zerosLocal[v4Id * embedDims + queryOffsetv],
                        valueGm[offsetValue + hLowPtrOffset + hStride + wLowPtrOffset + wStride], copyInParamsV3,
                        padParamsFloat);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Muls(wv4Local, zerosLocal[v4Id * embedDims + queryOffsetv], w4, embedDims);
        }
    }

    Add(wv1Local, wv1Local, wv2Local, embedDims);
    Add(wv3Local, wv3Local, wv4Local, embedDims);
    Add(wv1Local, wv1Local, wv3Local, embedDims);
    Muls(outputLocal[queryOffset], wv1Local, attentionWeight, embedDims);

    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    if constexpr (!std::is_same_v<T, float>) {
        DataCopyPad(
            workspaceGm[batch * outputStride2 + (nqloop * maxUbNum + query) * outputStride1 + head * outputStride0],
            outputLocal[queryOffset], copyOutParamsFloat);
    } else {
        DataCopyPad(
            outputGm[batch * outputStride2 + (nqloop * maxUbNum + query) * outputStride1 + head * outputStride0],
            outputLocal[queryOffset], copyOutParams);
    }
}

template <typename TilingDataT, typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn<TilingDataT, T>::ComputeGradTogether(float distLH, float distLW,
                                                                                           float w1, float w2, float w3,
                                                                                           float w4,
                                                                                           float attentionWeight)
{
    if constexpr (!std::is_same_v<T, float>) {
        auto stagingInHalf = stagingInUb.Get<T>();
        WaitFlag<HardEvent::V_MTE2>(eventIdStagingVToMte2);
        DataCopyPad(stagingInHalf, valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], copyInParamsV3,
                    padParamsFloat);
        DataCopyPad(stagingInHalf[stagingStride], valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride],
                    copyInParamsV3, padParamsFloat);
        DataCopyPad(stagingInHalf[twoBuffer * stagingStride],
                    valueGm[offsetValue + hLowPtrOffset + hStride + wLowPtrOffset], copyInParamsV3, padParamsFloat);
        DataCopyPad(stagingInHalf[v4Id * stagingStride],
                    valueGm[offsetValue + hLowPtrOffset + hStride + wLowPtrOffset + wStride], copyInParamsV3,
                    padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast<float, T>(zerosLocal[v1Id * embedDims + queryOffsetv], stagingInHalf, RoundMode::CAST_NONE, embedDims);
        Cast<float, T>(zerosLocal[v2Id * embedDims + queryOffsetv], stagingInHalf[stagingStride], RoundMode::CAST_NONE,
                       embedDims);
        Cast<float, T>(zerosLocal[v3Id * embedDims + queryOffsetv], stagingInHalf[twoBuffer * stagingStride],
                       RoundMode::CAST_NONE, embedDims);
        Cast<float, T>(zerosLocal[v4Id * embedDims + queryOffsetv], stagingInHalf[v4Id * stagingStride],
                       RoundMode::CAST_NONE, embedDims);
        SetFlag<HardEvent::V_MTE2>(eventIdStagingVToMte2);
    } else {
        DataCopyPad(zerosLocal[queryOffsetv], valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], copyInParamsV1,
                    padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

    Muls(wv1Local, zerosLocal[v1Id * embedDims + queryOffsetv], w1, embedDims);
    Muls(wv2Local, zerosLocal[v2Id * embedDims + queryOffsetv], w2, embedDims);
    Muls(wv3Local, zerosLocal[v3Id * embedDims + queryOffsetv], w3, embedDims);
    Muls(wv4Local, zerosLocal[v4Id * embedDims + queryOffsetv], w4, embedDims);

    Add(wv1Local, wv1Local, wv2Local, embedDims);
    Add(wv3Local, wv3Local, wv4Local, embedDims);
    Add(wv1Local, wv1Local, wv3Local, embedDims);
    Muls(outputLocal[queryOffset], wv1Local, attentionWeight, embedDims);

    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    if constexpr (!std::is_same_v<T, float>) {
        DataCopyPad(
            workspaceGm[batch * outputStride2 + (nqloop * maxUbNum + query) * outputStride1 + head * outputStride0],
            outputLocal[queryOffset], copyOutParamsFloat);
    } else {
        DataCopyPad(
            outputGm[batch * outputStride2 + (nqloop * maxUbNum + query) * outputStride1 + head * outputStride0],
            outputLocal[queryOffset], copyOutParams);
    }
}

template <typename TilingDataT, typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn<TilingDataT, T>::GridSampleCompute()
{
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    if constexpr (!std::is_same_v<T, float>) {
        SetAtomicAdd<float>();
    } else {
        SetAtomicAdd<T>();
    }
    for (query = 0; query < thisCycleNum; query++) {
        queryOffset = query * embedDims;
        queryOffsetv = query * fourBuffer * embedDims;
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

template <typename TilingDataT, typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn<TilingDataT, T>::Compute(uint64_t taskIdx)
{
    ScalarUpdate(taskIdx);

    Duplicate(zerosLocal, (float)0, fourBuffer * numQueriesAlign * embedDims);

    if constexpr (!std::is_same_v<T, float>) {
        DataCopyPad(locWHalfView[twoBuffer * numQueriesAlign], locationGm[offsetLocation + nqloop * maxUbNum],
                    copyInParamsV2, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast<float, T>(locWLocal, locWHalfView[twoBuffer * numQueriesAlign], RoundMode::CAST_NONE, thisCycleNumAlign);
    } else {
        DataCopyPad(locWLocal, locationGm[offsetLocation + nqloop * maxUbNum], copyInParamsV2, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    Muls(imLocal, locWLocal, (float)w, thisCycleNumAlign);

    if constexpr (!std::is_same_v<T, float>) {
        DataCopyPad(locHHalfView[twoBuffer * numQueriesAlign],
                    locationGm[offsetLocation + numQueries + nqloop * maxUbNum], copyInParamsV2, padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast<float, T>(locHLocal, locHHalfView[twoBuffer * numQueriesAlign], RoundMode::CAST_NONE, thisCycleNumAlign);
    } else {
        DataCopyPad(locHLocal, locationGm[offsetLocation + numQueries + nqloop * maxUbNum], copyInParamsV2,
                    padParamsFloat);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    Muls(imLocal[thisCycleNumAlign], locHLocal, (float)h, thisCycleNumAlign);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    if constexpr (!std::is_same_v<T, float>) {
        DataCopyPad(attnWeightHalfView[twoBuffer * numQueriesAlign],
                    attentionWeightsGm[offsetWeight + nqloop * maxUbNum], copyInParamsV2, padParamsFloat);
    } else {
        DataCopyPad(attentionWeightLocal, attentionWeightsGm[offsetWeight + nqloop * maxUbNum], copyInParamsV2,
                    padParamsFloat);
    }

    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    Adds(imLocal, imLocal, CONV_CONSTANT, TWO * thisCycleNumAlign);
    Cast(lowLocal, imLocal, RoundMode::CAST_FLOOR, TWO * thisCycleNumAlign);
    Cast(lowFloatLocal, lowLocal, RoundMode::CAST_NONE, TWO * thisCycleNumAlign);
    Sub(distLowLocal, imLocal, lowFloatLocal, TWO * thisCycleNumAlign);
    Mul(w4Local, distLowLocal[thisCycleNumAlign], distLowLocal, thisCycleNumAlign);

    if constexpr (!std::is_same_v<T, float>) {
        Cast<float, T>(attentionWeightLocal, attnWeightHalfView[twoBuffer * numQueriesAlign], RoundMode::CAST_NONE,
                       thisCycleNumAlign);
    } else {
    }
    GridSampleCompute();
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
}

template <typename TilingDataT, typename T>
__aicore__ inline void KernelMultiScaleDeformableAttn<TilingDataT, T>::CastOutputPass()
{
    if ASCEND_IS_AIV {
        SyncAll();
    }
    uint64_t total = batchSize * numQueries * numHeads * embedDims;
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
    auto out0 = castOut1Ub.Get<T>();
    auto out1 = castOut2Ub.Get<T>();
    uint64_t numIter = DivCeil(rangeEnd - rangeStart, chunk);
    uint64_t firstLen = (rangeEnd - rangeStart < chunk) ? (rangeEnd - rangeStart) : chunk;
    DataCopyPad(zerosLocal, workspaceGm[rangeStart], {1, (uint32_t)(firstLen * sizeof(float)), 0, 0, 0}, padParamsWsIn);
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
            DataCopyPad(outputGm[off], out0, {1, (uint32_t)(cur * sizeof(T)), 0, 0, 0});
        } else {
            DataCopyPad(outputGm[off], out1, {1, (uint32_t)(cur * sizeof(T)), 0, 0, 0});
        }
        if (i + 1 < numIter) {
            uint64_t nextOff = off + chunk;
            uint64_t nextLen = (nextOff + chunk > rangeEnd) ? (rangeEnd - nextOff) : chunk;
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            if ((i + 1) % twoBuffer == 0) {
                DataCopyPad(zerosLocal, workspaceGm[nextOff], {1, (uint32_t)(nextLen * sizeof(float)), 0, 0, 0},
                            padParamsWsIn);
            } else {
                DataCopyPad(zerosLocal[chunk], workspaceGm[nextOff], {1, (uint32_t)(nextLen * sizeof(float)), 0, 0, 0},
                            padParamsWsIn);
            }
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        }
    }
}

#endif // MS_DEFORM_ATTN_GENERIC_IMPL_H
