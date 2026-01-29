/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dual_level_quant_batch_matmul_tiling_data.h
 * \brief
 */
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_DATA_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

enum class L2CacheMode : std::uint32_t
{
    L2_CACHE_DEFAULT = 0x00,
    A_L2_CACHE_DISABLE = 0x01,
    B_L2_CACHE_DISABLE = 0x02,
    ALL_L2_CACHE_DISABLE = 0x03,
};

// tiling data注意8B对齐，尽量手动添加对齐的保留字段
#pragma pack(push, 8)
struct DualLevelQuantBatchMatmulBasicTilingData {
    uint8_t l1BufferNum = 2; // memory bound场景有不同
    bool hasBias = false;
    L2CacheMode l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT; // L2Cache默认使能
    uint32_t usedCoreNum = 0;
    uint32_t mSize = 0;
    uint32_t nSize = 0;
    uint32_t kSize = 0;
    uint32_t mL1Size = 256;
    uint32_t nL1Size = 256;
    uint32_t kL1Size = 512;
    uint32_t level0GroupSize = 512;

    // 尾核优化相关参数，优化尾块的多核利用率
    uint32_t mTailTile = 0; // m方向尾块重切分次数
    uint32_t nTailTile = 0; // n方向尾块重切分次数
    // 尾块优化相关参数，尾块负载均衡优化
    uint32_t mBaseTailSplitCnt = 1; // m方向尾块重切分的块数，有多少个非标准的边缘块
    uint32_t nBaseTailSplitCnt = 1; // n方向尾块重切分的块数
    uint32_t mTailMain = 1;         // 尾块重切分选择的m方向基本块大小
    uint32_t nTailMain = 1;         // 尾块重切分选择的n方向基本块大小
};
#pragma pack(pop)

#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_TILING_DATA_H
