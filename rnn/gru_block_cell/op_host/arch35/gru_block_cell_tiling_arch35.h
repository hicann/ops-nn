/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_RNN_GRU_BLOCK_CELL_TILING_ARCH35_H
#define OPS_RNN_GRU_BLOCK_CELL_TILING_ARCH35_H

namespace optiling {

// 编译期平台快照：TilingPrepare 期查询落盘，运行期 GetPlatformInfo() 不可用时回退。
// coreNum 为 AIC 核数（MIX 1:2 下 blockDim 以 AIC 块为单位，勿用 AIV 数——2× 会
// 超配物理 cluster 触发跨核握手停摆）。其余为字节数。
// L0A 行界（880）不在此列：它是 LOAD2D 实测表征界，非容量查询值。
struct GruBlockCellCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
};

} // namespace optiling

#endif
