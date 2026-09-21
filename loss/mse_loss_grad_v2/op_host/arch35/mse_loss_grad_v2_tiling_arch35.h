/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_grad_v2_tiling_arch35.h
 * \brief arch35 tiling compile-time info: MseLossGradV2CompileInfo carries platform
 *        facts (AIV core count, UB size) from TilingPrepareForMseLossGradV2 to the
 *        tiling runtime. Read-only during tiling.
 */

#ifndef MSE_LOSS_GRAD_V2_TILING_ARCH35_H_
#define MSE_LOSS_GRAD_V2_TILING_ARCH35_H

// Include the tiling data struct definition — this is shared between
// host-side tiling and device-side kernel (same struct, different compilation).
#include "../../op_kernel/arch35/mse_loss_grad_v2_tiling_data.h"

namespace optiling {

// ---------------------------------------------------------------------------
// MseLossGradV2CompileInfo — platform information for tiling compile phase
//
// This struct is populated by TilingPrepareForMseLossGradV2() with platform
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
struct MseLossGradV2CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

} // namespace optiling

#endif
