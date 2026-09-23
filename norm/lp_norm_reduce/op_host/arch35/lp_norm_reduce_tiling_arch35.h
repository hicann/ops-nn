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
// LpNormReduce_package/op_host/arch35/lp_norm_reduce_tiling_arch35.h
// =============================================================================
//
// ROLE: Tiling compile-time info header for arch35 (Ascend 950 series).
//   This header defines the LpNormReduceCompileInfo struct that is used during
//   tiling preparation. The struct carries platform information (core count,
//   UB size) from the TilingPrepareForLpNormReduce function to the tiling
//   runtime. This info is read-only during tiling and is set during the
//   graph compilation phase.
//
//   The arch35 directory indicates this is for the daVinci arch 3.5 which
//   corresponds to the Ascend 950 series NPU.
//
// CONTENTS:
//   - LpNormReduceCompileInfo struct — compile-time platform info carrier
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : LpNormReduce   — struct name prefix
//   snake_case   : lp_norm_reduce  — filename
//   UPPER_SNAKE  : LP_NORM_REDUCE  — header guard (used here)
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   LpNormReduceCompileInfo → FooBarCompileInfo     (struct name)
//   LP_NORM_REDUCE → FOO_BAR                         (header guard)
//   lp_norm_reduce → foo_bar                         (filename)
//   arch35 → appropriate arch directory for target hardware
//   optiling namespace: keep (standard CANN tiling namespace)
//
// =============================================================================

#ifndef OPS_MATH_SCALE_OP_HOST_ARCH35_LP_NORM_REDUCE_TILING_ARCH35_H
#define OPS_MATH_SCALE_OP_HOST_ARCH35_LP_NORM_REDUCE_TILING_ARCH35_H

// Include the tiling data struct definition — this is shared between
// host-side tiling and device-side kernel (same struct, different compilation).
// (TilingData 汇总结构体定义于 lp_norm_reduce_tiling_data.h)
#include "../../op_kernel/arch35/lp_norm_reduce_tiling_data.h"

namespace optiling {

// ---------------------------------------------------------------------------
// LpNormReduceCompileInfo — platform information for tiling compile phase
//
// This struct is populated by TilingPrepareForLpNormReduce() with platform
// hardware information (number of AIV cores, UB memory size). It is then
// passed to the tiling function so it can make decisions about data
// partitioning and parallelism.
//
// Fields:
//   coreNum — number of available AIV (AI Vector) cores on the NPU
//   ubSize  — size of Unified Buffer (UB) in bytes per core
//             UB is the fast on-chip memory used for kernel computation
//
// To create FooBar: rename to FooBarCompileInfo.
//   Most operators need the same fields (coreNum, ubSize).
//   Add fields here if your operator needs additional compile-time info.
// ---------------------------------------------------------------------------
struct LpNormReduceCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

} // namespace optiling

#endif
