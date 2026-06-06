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
 * \file prelu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _PRELU_TILING_DATA_H_
#define _PRELU_TILING_DATA_H_

#include <cstdint>

struct PreluTilingData {
    int64_t totalLength = 0;   // 总元素数量
    int64_t usedCoreNum = 0;   // 实际使用的 AIV 核数
    int64_t formerNum = 0;     // scalar路径使用 formerLength 的核数
    int64_t formerLength = 0;  // scalar路径前段每核处理的元素数量
    int64_t tailLength = 0;    // scalar路径尾段每核处理的元素数量
    int64_t tileLength = 0;    // 每次 UB 循环处理的元素数量

    int64_t channelSize = 1;       // C
    int64_t innerSize = 1;         // L，真实搬运长度
    int64_t innerSizeAligned = 1;  // AlignUp(L, 32 / sizeof(T))
    int64_t baseRows = 0;          // channel路径每核基础row数
    int64_t extraRows = 0;         // channel路径前extraRows个核各多处理1个row
    int64_t tilesPerRow = 0;       // split-L parallel路径每个row的L维tile数
    int64_t baseTasks = 0;         // split-L parallel路径每核基础task数
    int64_t extraTasks = 0;        // split-L parallel路径前extraTasks个核各多处理1个task
};
#endif
