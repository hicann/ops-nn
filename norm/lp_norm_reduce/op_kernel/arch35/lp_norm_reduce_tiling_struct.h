/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// LpNormReduce_package/op_kernel/arch35/lp_norm_reduce_tiling_struct.h
// =============================================================================
//
// ROLE: 【兼容转发头】LpNormReduce TilingData 汇总结构体已按
//   docs/LpNormReduce/design/TilingData.md 落码于同目录
//   lp_norm_reduce_tiling_data.h（LpNormReduceTilingData / Base+Group 共用 +
//   LpNormReduceEmptyTilingData / Empty 独立，MAX_PATTERN_RANK=9）；本文件仅作
//   向后兼容转发（既有 include 路径与 tests/tiling 安装树引用），
//   不再独立定义任何字段。
//
// =============================================================================

#ifndef LP_NORM_REDUCE_TILING_STRUCT_H_
#define LP_NORM_REDUCE_TILING_STRUCT_H_

// TilingData 汇总结构体（与被测 tiling / kernel 两侧共用同一布局）
#include "lp_norm_reduce_tiling_data.h"

#endif // LP_NORM_REDUCE_TILING_STRUCT_H_
