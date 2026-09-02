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
 * \file roi_pooling_tiling_data.h
 * \brief tiling data struct for roi_pooling
 */
#ifndef ROI_POOLING_TILING_DATA_H
#define ROI_POOLING_TILING_DATA_H

struct RoiPoolingTilingData {
    int64_t totalElements; // 输出总元素数 = K * C * pooled_h * pooled_w（Grid-Stride 遍历上界）
    int64_t needCoreNum;   // 实际启动核数
    int64_t N;             // 特征图 batch 数 = x.shape[0]（用于 kernel 内 batchIdx 越界检查）
    int64_t K;             // ROI 数量 = rois.shape[0]
    int64_t C;             // 通道数 = x.shape[1]
    int64_t H;             // 特征图高 = x.shape[2]
    int64_t W;             // 特征图宽 = x.shape[3]
    int64_t pooledH;       // 池化输出高（来自属性 pooled_h）
    int64_t pooledW;       // 池化输出宽（来自属性 pooled_w）
    float spatialScaleH;   // y 方向缩放因子（来自属性 spatial_scale_h）
    float spatialScaleW;   // x 方向缩放因子（来自属性 spatial_scale_w）
};

#endif
