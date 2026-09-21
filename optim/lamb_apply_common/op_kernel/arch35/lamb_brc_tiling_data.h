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
 * \file lamb_brc_tiling_data.h
 * \brief LAMB 族「多入多出 + 任意 numpy 广播」逐元素算子的共用 tiling 数据
 *
 * host 与 kernel 引用同一个模板实例, 布局天然一致(host 侧 memcpy 整个 POD 下发)。
 */

#ifndef LAMB_BRC_TILING_DATA_H
#define LAMB_BRC_TILING_DATA_H

#include <cstdint>

// GE 原型最大轴数(与 silu_grad / reverse_v2 等同仓算子一致)。规划过程只做轴合并与块内切分,
// 不会增加轴数, 因此这个尺寸对任何合法输入都够用 —— 不存在"超了就拒收"的分支。
constexpr uint32_t LAMB_BRC_MAX_DIM = 8;

// 硬件常量(非经验阈值): UB 搬运的最小块 32B、向量寄存器 256B。kernel 侧用平台接口
// GetUbBlockSize()/GetVRegSize() 静态校验二者一致, 避免两侧取值走偏。
constexpr uint32_t LAMB_BRC_GM_BLOCK_BYTES = 32;
constexpr uint32_t LAMB_BRC_VREG_BYTES = 256;

// inKind 取值
constexpr uint32_t LAMB_BRC_KIND_SCALAR = 0; // 单元素, UB 槽位铺一次后常驻
constexpr uint32_t LAMB_BRC_KIND_SAME = 1;   // 与输出同形, 连续搬入
constexpr uint32_t LAMB_BRC_KIND_BRC = 2;    // 行块内需展开广播

// tilingKey(数据内的分支, 与 binary 的 tilingKey 无关)
constexpr uint32_t LAMB_BRC_KEY_FLAT = 0;  // 无需广播的输入 -> 纯线性分片, 不受行块大小约束
constexpr uint32_t LAMB_BRC_KEY_BLOCK = 1; // 有需广播的输入 -> 按输出行块处理

template <uint32_t NIN, uint32_t NOUT>
struct LambBrcTilingData {
    uint64_t totalNum = 0;                           // 输出元素总数
    uint64_t totalRows = 0;                          // 分块模式下的行块总数
    uint64_t effStride[NIN * LAMB_BRC_MAX_DIM] = {}; // [i][j] 输入i在外层轴j的源步长(广播轴为0)
    uint64_t srcBlockLen[NIN] = {};                  // 输入i在一个行块内的源元素数
    uint64_t perCoreElems = 0;                       // 平铺模式每核元素数(已对齐 32B)
    uint64_t rowsPerCore = 0;                        // 分块模式每核行块数(已对齐 32B)
    uint32_t tileLen = 0;                            // 由 ubSize 反解的分片元素数
    uint32_t blockLen = 0;                           // 一个输出行块的元素数
    uint32_t rowsPerTile = 0;                        // 一个分片内批量处理的行块数
    uint32_t innerChunk = 0;                         // 最内轴装不下时的块内切分长度(0=不切)
    uint32_t innerChunkCnt = 0;                      // 最内轴被切成几段
    uint32_t usedCoreNum = 0;
    uint32_t splitAxis = 0; // 前导行维个数
    uint32_t collapsedRank = 0;
    uint32_t tilingKey = 0;
    uint32_t inKind[NIN] = {};
    uint32_t outShape[LAMB_BRC_MAX_DIM] = {};
    uint32_t inShape[NIN * LAMB_BRC_MAX_DIM] = {};

    static constexpr uint32_t SlotNum() { return NIN + NOUT + 1; } // 入 + 出 + 1 广播暂存
};

#endif // LAMB_BRC_TILING_DATA_H
