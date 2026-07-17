/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ms_deform_attn_simt.h
 * \brief MultiScaleDeformableAttnFunction SIMT implementation for ascend950
 */

#ifndef MS_DEFORM_ATTN_SIMT_H
#define MS_DEFORM_ATTN_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "multi_scale_deformable_attn_function_tiling_data.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

namespace MsdaSimt {
using namespace AscendC;

constexpr uint32_t THREAD_DIM = 512;

__simt_callee__ __aicore__ __attribute__((always_inline)) inline static int64_t Floor(float x)
{
    int64_t i = static_cast<int64_t>(x);
    return (x < 0.0f && static_cast<float>(i) != x) ? i - 1 : i;
}

// SIMT context requires explicit conversion functions for half/bfloat16_t
template <typename T>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline static float ToFloat(T val)
{
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else {
        return __bfloat162float(val);
    }
}

template <typename T>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline static T FromFloat(float val)
{
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(val);
    } else {
        return __float2bfloat16(val);
    }
}

template <typename T>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline static float SampleBilinearPoint(
    __gm__ T* valueGm, __gm__ T* samplingLocGm, __gm__ T* attnWeightGm, int64_t valBase, int64_t locBase,
    int64_t weightBase, int64_t headEmbed, int64_t l, int64_t p, int64_t numPoints, int32_t hL, int32_t wL,
    int32_t levelStart)
{
    float xNorm = ToFloat<T>(samplingLocGm[locBase + (l * numPoints + p) * 2]);
    float yNorm = ToFloat<T>(samplingLocGm[locBase + (l * numPoints + p) * 2 + 1]);
    float attnW = ToFloat<T>(attnWeightGm[weightBase + l * numPoints + p]);

    float x = xNorm * static_cast<float>(wL) - 0.5f;
    float y = yNorm * static_cast<float>(hL) - 0.5f;

    int64_t x0 = Floor(x);
    int64_t y0 = Floor(y);
    int64_t x1 = x0 + 1;
    int64_t y1 = y0 + 1;

    float ax = x - static_cast<float>(x0);
    float ay = y - static_cast<float>(y0);

    const int64_t valLevelBase = valBase + static_cast<int64_t>(levelStart) * headEmbed;
    const float oneMinusAy = 1.0f - ay;
    const float oneMinusAx = 1.0f - ax;

    float sampled = 0.0f;
    if (y0 >= 0 && y0 < hL && x0 >= 0 && x0 < wL) {
        sampled += oneMinusAy * oneMinusAx * ToFloat<T>(valueGm[valLevelBase + (y0 * wL + x0) * headEmbed]);
    }
    if (y0 >= 0 && y0 < hL && x1 >= 0 && x1 < wL) {
        sampled += oneMinusAy * ax * ToFloat<T>(valueGm[valLevelBase + (y0 * wL + x1) * headEmbed]);
    }
    if (y1 >= 0 && y1 < hL && x0 >= 0 && x0 < wL) {
        sampled += ay * oneMinusAx * ToFloat<T>(valueGm[valLevelBase + (y1 * wL + x0) * headEmbed]);
    }
    if (y1 >= 0 && y1 < hL && x1 >= 0 && x1 < wL) {
        sampled += ay * ax * ToFloat<T>(valueGm[valLevelBase + (y1 * wL + x1) * headEmbed]);
    }

    return attnW * sampled;
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void MsdaSimtForwardFunc(
    int64_t totalOutput, __gm__ T* valueGm, __gm__ int32_t* spatialShapesGm, __gm__ int32_t* levelStartIndexGm,
    __gm__ T* samplingLocGm, __gm__ T* attnWeightGm, __gm__ T* outputGm, int64_t bs, int64_t numKeys, int64_t numHeads,
    int64_t embedDims, int64_t numLevels, int64_t numQueries, int64_t numPoints)
{
    const int64_t headEmbed = numHeads * embedDims;
    const int64_t locBatchStride = numQueries * numHeads * numLevels * numPoints * 2;
    const int64_t locQueryStride = numHeads * numLevels * numPoints * 2;
    const int64_t locHeadStride = numLevels * numPoints * 2;
    const int64_t weightBatchStride = numQueries * numHeads * numLevels * numPoints;
    const int64_t weightQueryStride = numHeads * numLevels * numPoints;
    const int64_t weightHeadStride = numLevels * numPoints;

    for (int64_t idx = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); idx < totalOutput;
         idx += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        int64_t d = idx % embedDims;
        int64_t hd = idx / embedDims;
        int64_t h = hd % numHeads;
        int64_t bq = hd / numHeads;
        int64_t q = bq % numQueries;
        int64_t b = bq / numQueries;

        const int64_t valBase = b * numKeys * headEmbed + h * embedDims + d;
        const int64_t locBase = b * locBatchStride + q * locQueryStride + h * locHeadStride;
        const int64_t weightBase = b * weightBatchStride + q * weightQueryStride + h * weightHeadStride;

        float sum = 0.0f;

        for (int64_t l = 0; l < numLevels; l++) {
            int32_t hL = spatialShapesGm[l * 2];
            int32_t wL = spatialShapesGm[l * 2 + 1];
            int32_t levelStart = levelStartIndexGm[l];

            for (int64_t p = 0; p < numPoints; p++) {
                sum += SampleBilinearPoint<T>(valueGm, samplingLocGm, attnWeightGm, valBase, locBase, weightBase,
                                              headEmbed, l, p, numPoints, hL, wL, levelStart);
            }
        }

        outputGm[idx] = FromFloat<T>(sum);
    }
}

template <typename T>
class MsdaSimtKernel {
public:
    __aicore__ inline MsdaSimtKernel(const MsdaRegBaseTilingData* __restrict tilingData) : tilingData_(tilingData) {}

    __aicore__ inline void Init(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
                                GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output)
    {
        valueGm_.SetGlobalBuffer((__gm__ T*)value);
        spatialShapesGm_.SetGlobalBuffer((__gm__ int32_t*)valueSpatialShapes);
        levelStartIndexGm_.SetGlobalBuffer((__gm__ int32_t*)valueLevelStartIndex);
        samplingLocGm_.SetGlobalBuffer((__gm__ T*)samplingLocations);
        attnWeightGm_.SetGlobalBuffer((__gm__ T*)attentionWeights);
        outputGm_.SetGlobalBuffer((__gm__ T*)output);
    }

    __aicore__ inline void Process()
    {
        int64_t bs = tilingData_->batchSize;
        int64_t numKeys = tilingData_->numKeys;
        int64_t numHeads = tilingData_->numHeads;
        int64_t embedDims = tilingData_->embedDims;
        int64_t numLevels = tilingData_->numLevels;
        int64_t numQueries = tilingData_->numQueries;
        int64_t numPoints = tilingData_->numPoints;

        int64_t totalOutput = bs * numQueries * numHeads * embedDims;

        Simt::VF_CALL<MsdaSimtForwardFunc<T>>(
            Simt::Dim3(THREAD_DIM), totalOutput, (__gm__ T*)valueGm_.GetPhyAddr(),
            (__gm__ int32_t*)spatialShapesGm_.GetPhyAddr(), (__gm__ int32_t*)levelStartIndexGm_.GetPhyAddr(),
            (__gm__ T*)samplingLocGm_.GetPhyAddr(), (__gm__ T*)attnWeightGm_.GetPhyAddr(),
            (__gm__ T*)outputGm_.GetPhyAddr(), bs, numKeys, numHeads, embedDims, numLevels, numQueries, numPoints);
    }

private:
    const MsdaRegBaseTilingData* tilingData_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<int32_t> spatialShapesGm_;
    GlobalTensor<int32_t> levelStartIndexGm_;
    GlobalTensor<T> samplingLocGm_;
    GlobalTensor<T> attnWeightGm_;
    GlobalTensor<T> outputGm_;
};

} // namespace MsdaSimt

#endif // MS_DEFORM_ATTN_SIMT_H
