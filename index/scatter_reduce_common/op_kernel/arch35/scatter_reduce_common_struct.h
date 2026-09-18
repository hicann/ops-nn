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
 * \file scatter_reduce_common_struct.h
 * \brief Shared definitions for the non-nd scatter reduce ops (ScatterMax/Min/Mul/Div).
 *        Independent implementation (does not depend on scatter_add). Modern direct-struct tiling.
 */
#ifndef SCATTER_REDUCE_COMMON_STRUCT_H
#define SCATTER_REDUCE_COMMON_STRUCT_H

#include <cstdint>

namespace ScatterReduceCommon {
// reduce mode, passed as a template parameter to the kernel
constexpr uint64_t MODE_MAX = 0;
constexpr uint64_t MODE_MIN = 1;
constexpr uint64_t MODE_MUL = 2;
constexpr uint64_t MODE_DIV = 3;

// tiling key template modes
// UB 访问与掩码均按 block 对齐; 该文件 host/kernel 共用, 故不引 AscendC 命名空间,
// 取值与 AscendC::ONE_BLK_SIZE 一致(kernel 侧有静态断言校验)。
constexpr uint32_t UB_BLOCK_BYTES = 32;
// 位掩码 1 bit/元素, 故一个 block 覆盖 UB_BLOCK_BYTES * 8 个元素。

constexpr uint64_t TPL_ADDR_32 = 0; // index address fits in uint32
constexpr uint64_t TPL_ADDR_64 = 1; // index address needs uint64

// SIMT atomic tiling data (direct struct; read in kernel via GET_TILING_DATA_WITH_STRUCT).
// semantics: for each index entry m, var[indices[m]] (a slice of sliceSize elems) is reduced with
// updates[m] by the reduce mode. The total update-element space (indicesNum * sliceSize) is split
// across cores (block tiling) then iterated in UB chunks.
struct ScatterReduceSimtTilingData {
    uint64_t blockNum;            // number of cores actually used
    uint64_t blockTilingSize;     // update elements handled per front core
    uint64_t tailBlockTilingSize; // update elements handled on the last used core
    uint64_t sliceSize;           // elements per index slice (product of var tail dims)
    uint64_t varFirstDim;         // var dim0 size (index bound)
    uint64_t scalarPath; // 1 = 整个 var + indices 能同时放进 UB, 走单核标量路径; host 按实测 UB 判定
    uint64_t sortTile;   // 排序分片元素数, host 按 UB 实测容量反解并 32 对齐(不写死)
    uint64_t ubChunkMax; // phase-3 列切分上限, host 按 UB 实测容量与每列存活开销反解
};
} // namespace ScatterReduceCommon

#endif // SCATTER_REDUCE_COMMON_STRUCT_H
