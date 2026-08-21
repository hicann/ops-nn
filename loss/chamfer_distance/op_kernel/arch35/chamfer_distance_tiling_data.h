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
 * \file chamfer_distance_tiling_data.h
 * \brief
 */
#ifndef CHAMFER_DISTANCE_TILING_DATA_H
#define CHAMFER_DISTANCE_TILING_DATA_H

#include <cstdint>

struct ChamferDistanceArch35TilingData {
    int64_t b = 0;            // batch 数
    int64_t n = 0;            // 每个点集的点数
    int64_t taskNum = 0;      // 查询点总数 = b * n
    int64_t realCoreNum = 0;  // 实际使用核数
    int64_t tasksPerCore = 0; // 主核负责的查询点数
    int64_t tailTasks = 0;    // 尾核负责的查询点数
    int64_t colsPerChunk = 0; // 被查集合一次驻留 UB 的点数(VL 对齐)
    int64_t chunkNum = 0;     // 被查集合的分段数
};

#endif // CHAMFER_DISTANCE_TILING_DATA_H
