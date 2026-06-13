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
constexpr uint64_t TPL_ADDR_32 = 0;  // index address fits in uint32
constexpr uint64_t TPL_ADDR_64 = 1;  // index address needs uint64

// SIMT atomic tiling data (direct struct; read in kernel via GET_TILING_DATA_WITH_STRUCT).
// semantics: for each index entry m, var[indices[m]] (a slice of sliceSize elems) is reduced with
// updates[m] by the reduce mode. The total update-element space (indicesNum * sliceSize) is split
// across cores (block tiling) then iterated in UB chunks.
struct ScatterReduceSimtTilingData {
    uint64_t blockNum;             // number of cores actually used
    uint64_t blockTilingSize;      // update elements handled per front core
    uint64_t tailBlockTilingSize;  // update elements handled on the last used core
    uint64_t sliceSize;            // elements per index slice (product of var tail dims)
    uint64_t varFirstDim;          // var dim0 size (index bound)
};
}  // namespace ScatterReduceCommon

#endif  // SCATTER_REDUCE_COMMON_STRUCT_H
