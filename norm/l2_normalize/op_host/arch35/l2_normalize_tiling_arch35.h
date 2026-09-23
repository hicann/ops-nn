/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_host/arch35/l2_normalize_tiling_arch35.h
// =============================================================================
//
// ROLE: Tiling compile-time info header for arch35 (Ascend 950 series).
//   This header defines the L2NormalizeCompileInfo struct used by the
//   TilingParse registration. The struct is
//   intentionally empty: platform parameters (AIV core count, UB size,
//   block size, cache line size) are fetched fresh from
//   context->GetPlatformInfo() inside the TilingFunc on every invocation
//   and are not cached across compilations.
//
//   The arch35 directory indicates this is for the daVinci arch 3.5 which
//   corresponds to the Ascend 950 series NPU.
//
// CONTENTS:
//   - L2NormalizeCompileInfo struct — TilingParse carrier (empty)
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : L2Normalize   — struct name prefix
//   snake_case   : l2_normalize  — filename
//   UPPER_SNAKE  : L2_NORMALIZE  — header guard (used here)
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   L2NormalizeCompileInfo → FooBarCompileInfo     (struct name)
//   L2_NORMALIZE → FOO_BAR                         (header guard)
//   l2_normalize → foo_bar                         (filename)
//   arch35 → appropriate arch directory for target hardware
//   optiling namespace: keep (standard CANN tiling namespace)
//
// =============================================================================

#ifndef OPS_NN_L2_NORMALIZE_TILING_ARCH35_H
#define OPS_NN_L2_NORMALIZE_TILING_ARCH35_H

// Include the tiling data struct definition — this is shared between
// host-side tiling and device-side kernel (same struct, different compilation).
#include "../../op_kernel/arch35/l2_normalize_tiling_struct.h"

namespace optiling {

// ---------------------------------------------------------------------------
// L2NormalizeCompileInfo — TilingParse 载体
//
// 空结构（范式 euclidean_norm 同款）：本算子无跨次编译缓存信息，平台参数
// （AIV 核数 / UB 大小 / block / cache line）每次经 TilingFunc 内
// GetPlatformInfo() 现取，不挂 CompileInfo 缓存；后续若做 binary 复用，
// 缓存字段在此结构扩展并在 TilingParseForL2Normalize 解析。
// ---------------------------------------------------------------------------
struct L2NormalizeCompileInfo {};

} // namespace optiling

#endif
