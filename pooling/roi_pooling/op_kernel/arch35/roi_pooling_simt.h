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
 * \file roi_pooling_simt.h
 * \brief SIMT kernel implementation for roi_pooling
 */
#ifndef ROI_POOLING_SIMT_H
#define ROI_POOLING_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/math_constants.h"
#include "simt_api/asc_fp16.h"
#include "roi_pooling_tiling_data.h"
#include "roi_pooling_tiling_key.h"

namespace NsRoiPooling {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;
constexpr int32_t ROI_COLS = 5; // [batch_idx, x1, y1, x2, y2]

template <typename T>
__simt_callee__ __aicore__ inline float ToFloat(T val)
{
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        return __half2float(val);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T FromFloat(float val)
{
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        return __float2half(val);
    }
}

// ========== 辅助函数：clip 到 [0, upper] ==========
// val 和 upper 均为 int64_t：与 int64_t 的 H/W 及 float→int64 转换结果对齐。
__simt_callee__ __aicore__ inline int64_t ClipInt(int64_t val, int64_t upper)
{
    if (val < 0)
        return 0;
    if (val > upper)
        return upper;
    return val;
}

// ========== Bin 区域 max 扫描 ==========
// 在 float 域做 max 比较，避免 half -INF 和 half > 比较的潜在问题
// bin 坐标为 int64_t：与 ClipInt 返回类型对齐
template <typename T>
__simt_callee__ __aicore__ inline float ScanBinForMax(__gm__ T* x_gm, int64_t xBase, int64_t W, int64_t binY1,
                                                      int64_t binY2, int64_t binX1, int64_t binX2)
{
    float maxVal = -ASCRT_INF_F; // float -INF
    for (int64_t h = binY1; h < binY2; h++) {
        int64_t rowBase = xBase + h * W;
        for (int64_t w = binX1; w < binX2; w++) {
            float val = ToFloat(x_gm[rowBase + w]);
            if (val > maxVal) {
                maxVal = val;
            }
        }
    }
    return maxVal;
}

// ========== 处理单个输出元素 ==========
template <typename T>
__simt_callee__ __aicore__ inline void ProcessOneOutputElement(int64_t idx, int64_t N, int64_t C, int64_t H, int64_t W,
                                                               int64_t pooledH, int64_t pooledW, float spatialScaleH,
                                                               float spatialScaleW, int64_t strideC, int64_t strideN,
                                                               int64_t xStrideC, int64_t xStrideN, __gm__ T* x_gm,
                                                               __gm__ T* rois_gm, __gm__ T* y_gm)
{
    // ============ 1. 索引分解：idx → (n, c, ph, pw) ============
    // N/C/pooledH/pooledW 均为 int64_t，全程 int64 运算，无截断
    int64_t n = idx / strideN;
    int64_t rem1 = idx - n * strideN;
    int64_t c = rem1 / strideC;
    int64_t rem2 = rem1 - c * strideC;
    int64_t ph = rem2 / pooledW;
    int64_t pw = rem2 - ph * pooledW;

    // ============ 2. 读取 rois 行，提升到 float ============
    float roiBatchF = ToFloat(rois_gm[n * ROI_COLS + 0]);
    float roiX1 = ToFloat(rois_gm[n * ROI_COLS + 1]);
    float roiY1 = ToFloat(rois_gm[n * ROI_COLS + 2]);
    float roiX2 = ToFloat(rois_gm[n * ROI_COLS + 3]);
    float roiY2 = ToFloat(rois_gm[n * ROI_COLS + 4]);

    // ============ 3. batchIdx 越界双侧保护 ============
    int64_t batchIdx = static_cast<int64_t>(roiBatchF);
    if (batchIdx < 0 || batchIdx >= N) {
        y_gm[idx] = FromFloat<T>(0.0f);
        return;
    }

    // ============ 4. ROI 坐标映射（roundf 取整为 int，无 +1 偏移）============
    //   roi_start = round(coord * spatial_scale)  ← int64 类型
    //   roi_end   = round(coord * spatial_scale)  ← int64 类型，无 +1 偏移
    //   +1 偏移在 roi_width 上（见步骤5），不在坐标上
    //   y 方向用 spatialScaleH，x 方向用 spatialScaleW
    int64_t roiStartW = static_cast<int64_t>(roundf(roiX1 * spatialScaleW));
    int64_t roiStartH = static_cast<int64_t>(roundf(roiY1 * spatialScaleH));
    int64_t roiEndW = static_cast<int64_t>(roundf(roiX2 * spatialScaleW)); // ← 无 +1 偏移
    int64_t roiEndH = static_cast<int64_t>(roundf(roiY2 * spatialScaleH)); // ← 无 +1 偏移

    // ============ 5. ROI 尺寸（int64 运算，malformed 强制非空）============
    //   +1 偏移在这里（Fast R-CNN 标准），不在坐标上
    //   roiEndW/roiStartW 均为 int64_t，减法+1 不会溢出。
    int64_t roiWidth = roiEndW - roiStartW + 1;
    int64_t roiHeight = roiEndH - roiStartH + 1;
    if (roiWidth < 1)
        roiWidth = 1;
    if (roiHeight < 1)
        roiHeight = 1;

    // ============ 6. Bin 大小（float，基于 int64 roi_width）============
    float binSizeW = static_cast<float>(roiWidth) / static_cast<float>(pooledW);
    float binSizeH = static_cast<float>(roiHeight) / static_cast<float>(pooledH);

    // ============ 7. Bin 边界 floor/ceil → int64 → + roiStart → clip ============
    int64_t binX1 = ClipInt(static_cast<int64_t>(floorf(static_cast<float>(pw) * binSizeW)) + roiStartW, W);
    int64_t binY1 = ClipInt(static_cast<int64_t>(floorf(static_cast<float>(ph) * binSizeH)) + roiStartH, H);
    int64_t binX2 = ClipInt(static_cast<int64_t>(ceilf(static_cast<float>(pw + 1) * binSizeW)) + roiStartW, W);
    int64_t binY2 = ClipInt(static_cast<int64_t>(ceilf(static_cast<float>(ph + 1) * binSizeH)) + roiStartH, H);

    // ============ 8. Max pooling ============
    if (binY2 <= binY1 || binX2 <= binX1) {
        // 空 bin 输出 0
        y_gm[idx] = FromFloat<T>(0.0f);
    } else {
        int64_t xBase = batchIdx * xStrideN + c * xStrideC;
        float maxVal = ScanBinForMax<T>(x_gm, xBase, W, binY1, binY2, binX1, binX2);
        y_gm[idx] = FromFloat<T>(maxVal);
    }
}

// ========== 主计算 VF ==========
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpRoiPoolingSimtKernel(
    int64_t totalElements, int64_t N, int64_t K, int64_t C, int64_t H, int64_t W, int64_t pooledH, int64_t pooledW,
    float spatialScaleH, float spatialScaleW, __gm__ T* x_gm, __gm__ T* rois_gm, __gm__ T* y_gm)
{
    // 预计算 stride（固定除数，VF 内计算，避免重复乘法）
    // N/C/H/W/pooledH/pooledW 均为 int64_t，乘法天然 int64，无需 static_cast
    const int64_t strideC = pooledH * pooledW;
    const int64_t strideN = C * strideC;
    const int64_t xStrideC = H * W;
    const int64_t xStrideN = C * xStrideC;

    // Grid-Stride 循环：每个线程独占处理一个输出元素
    for (int64_t idx = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x); idx < totalElements;
         idx += static_cast<int64_t>(blockDim.x * gridDim.x)) {
        ProcessOneOutputElement<T>(idx, N, C, H, W, pooledH, pooledW, spatialScaleH, spatialScaleW, strideC, strideN,
                                   xStrideC, xStrideN, x_gm, rois_gm, y_gm);
    }
}

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR rois, GM_ADDR roi_actual_num, GM_ADDR y, GM_ADDR workspace,
                               GM_ADDR tiling, const RoiPoolingTilingData* tilingData)
{
    __gm__ T* x_gm = (__gm__ T*)x;
    __gm__ T* rois_gm = (__gm__ T*)rois;
    __gm__ T* y_gm = (__gm__ T*)y;
    // roi_actual_num 本算子未使用（保留接口对齐），不取地址
    (void)roi_actual_num;
    (void)workspace;
    (void)tiling;
    asc_vf_call<OpRoiPoolingSimtKernel<T>>(dim3(THREAD_NUM), tilingData->totalElements, tilingData->N, tilingData->K,
                                           tilingData->C, tilingData->H, tilingData->W, tilingData->pooledH,
                                           tilingData->pooledW, tilingData->spatialScaleH, tilingData->spatialScaleW,
                                           x_gm, rois_gm, y_gm);
}

} // namespace NsRoiPooling

#endif
