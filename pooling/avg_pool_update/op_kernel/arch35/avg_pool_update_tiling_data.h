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
 * \file avg_pool_update_tiling_data.h
 * \brief Tiling data struct for avg_pool_update operator
 */

#ifndef AVG_POOL_UPDATE_TILING_DATA_H_
#define AVG_POOL_UPDATE_TILING_DATA_H_

struct AvgPoolUpdateTilingData {
    // 核数划分
    int64_t totalNum = 0;    // 输出元素总数 = N * C * out_h * out_w
    int32_t needCoreNum = 0; // 实际启动核数

    // 输出 shape (x1)：int64_t 避免 kernel 侧 idx*stride 溢出
    int64_t outH = 0;
    int64_t outW = 0;

    // 输入 shape (x2)
    int64_t inputH = 0;
    int64_t inputW = 0;

    // 池化参数（值域小，int32_t 即可）
    int32_t kH = 0;
    int32_t kW = 0;
    int32_t strideH = 0;
    int32_t strideW = 0;

    // Padding（通过 UB 传参到 VF）
    int64_t padT = 0;
    int64_t padB = 0;
    int64_t padL = 0;
    int64_t padR = 0;

    // 格式标志
    int32_t isNhwc = 0; // 0=NCHW, 1=NHWC
    int64_t outC = 0;   // NHWC 索引分解需要
};

#endif // AVG_POOL_UPDATE_TILING_DATA_H_
