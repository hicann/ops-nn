/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// gn_training_update/op_kernel/gn_training_update.cpp
// =============================================================================
//
// ROLE: Ascend C kernel entry point for GnTrainingUpdate.
//   This file contains the __global__ kernel function that runs on the NPU.
//   It is the bridge between the CANN runtime and the kernel implementation:
//
//   1. Receives GM_ADDR pointers for inputs, outputs, workspace, and tiling data
//   2. REGISTER_NONE_TILING: registers with the AICore runtime
//   3. KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY): sets task type to AIV (AI Vector)
//   4. Reads the tiling data via GET_TILING_DATA_WITH_STRUCT(TilingDataN, td, tiling)
//   5. Instantiates GnTrainingUpdateKernel<DTYPE_X, RANK> based on RANK template parameter
//   6. Calls kernel.Init() to set up buffers, then kernel.Process() to compute
//
//   The template parameter RANK (4 or 8) is selected by the tiling key at runtime.
//   DTYPE_X is auto-configured by the Ascend C runtime based on input dtype.
//
// CONTENTS:
//   - using TilingData4 = GnTrainingUpdateTilingData<4>  (type alias for rank-4 tiling data)
//   - using TilingData8 = GnTrainingUpdateTilingData<8>  (type alias for rank-8 tiling data)
//   - template<int RANK> __global__ void gn_training_update(...)  (the kernel entry)
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : GnTrainingUpdate   — GnTrainingUpdateKernel, GnTrainingUpdateTilingData
//   snake_case   : gn_training_update  — filename, kernel function name
//   UPPER_SNAKE  : GN_TRAINING_UPDATE  — (not directly used)
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   GnTrainingUpdateKernel → FooBarKernel
//   GnTrainingUpdateTilingData → FooBarTilingData
//   gn_training_update → foo_bar            (kernel function name, filename)
//   gn_training_update → foo_bar    (filename, referenced in op_def.cpp's ExtendCfgInfo)
//   TilingData4/TilingData8: rename to match new operator
//
// KEY MACROS AND CONVENTIONS:
//   __global__ __aicore__  — Ascend C attribute for NPU kernel functions
//   GM_ADDR                — global memory address type (pointer to device DRAM)
//   REGISTER_NONE_TILING   — registers kernel with AICore runtime (no tiling key)
//   KERNEL_TASK_TYPE_DEFAULT — sets the task type for this kernel
//   GET_TILING_DATA_WITH_STRUCT(StructType, varName, tilingAddr) — deserializes
//     tiling data from the tiling buffer into the struct
//
// =============================================================================

#include "kernel_operator.h"                         // Ascend C kernel framework (AscendC:: namespace)
#include "arch35/gn_training_update_tiling_struct.h" // GnTrainingUpdateTilingData<RANK> struct
#include "arch35/gn_training_update_struct.h"        // GN_TRAINING_UPDATE_RANK_4/8 constants
#include "arch35/gn_training_update_kernel.h"        // GnTrainingUpdateKernel<T, RANK> implementation

// Type aliases for the two possible tiling data sizes.
// TilingData4 is used when tensor rank ≤ 4.
// TilingData8 is used when tensor rank is 5-8.
// These match the tiling key selection in gn_training_update_tiling_arch35.cpp.
using TilingData4 = GnTrainingUpdateTilingData<GN_TRAINING_UPDATE_RANK_4>;
using TilingData8 = GnTrainingUpdateTilingData<GN_TRAINING_UPDATE_RANK_8>;

// ===========================================================================
// __global__ __aicore__ void gn_training_update(
//     GM_ADDR x, sum, square_sum, scale, offset, mean, variance,
//     GM_ADDR y, batch_mean, batch_variance, workspace, tiling)
//
// This is the main NPU kernel function. Each AIV core executes this once.
//
// Template parameter: RANK (4 or 8) — auto-determined by tiling key dispatch.
//
// Parameters (all GM_ADDR = global memory pointers on device):
//   x            — input activation tensor
//   sum          — per-(N,G) sums of x
//   square_sum   — per-(N,G) squared sums of x
//   scale/offset — affine parameters (may be unused if has_affine==0)
//   mean/variance — reserved inputs (not used on A5; stats computed in-kernel)
//   y            — normalized output tensor
//   batch_mean/batch_variance — per-(N,G) statistics outputs
//   workspace    — workspace buffer (not used by this kernel)
//   tiling       — tiling data buffer (GnTrainingUpdateTilingData<RANK>)
//
// Lifecycle:
//   1. Bundle input/output pointers into arrays for the kernel
//   2. REGISTER_NONE_TILING: registers the kernel task with the AICore runtime
//   3. GET_TILING_DATA_WITH_STRUCT: reads tiling parameters from buffer
//   4. Create GnTrainingUpdateKernel<DTYPE_X, RANK> instance
//   5. kernel.Init(): set up GlobalTensor views, UB buffers, NDDMA params
//   6. kernel.Process(): execute the tiled group-normalization computation
//
// DTYPE_X: auto-configured Ascend C macro that expands to the input data type
//   (e.g., float, half/BFloat16) based on the actual runtime data type.
// ===========================================================================
template <int RANK>
__global__ __aicore__ void gn_training_update(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale, GM_ADDR offset,
                                              GM_ADDR mean, GM_ADDR variance, GM_ADDR y, GM_ADDR batch_mean,
                                              GM_ADDR batch_variance, GM_ADDR workspace, GM_ADDR tiling)
{
    // Bundle input pointers into an array for the kernel.
    // Slots: 0=x, 1=sum, 2=square_sum, 3=scale, 4=offset, 5=mean, 6=variance
    // (scale/offset may be null when hasAffine==0; mean/variance are reserved
    // placeholder slots, not consumed by the compute).
    GM_ADDR ins[7] = {x, sum, square_sum, scale, offset, mean, variance};
    // Bundle output pointers: 0=y, 1=batch_mean, 2=batch_variance
    GM_ADDR outs[3] = {y, batch_mean, batch_variance};
    (void)workspace;

    // REGISTER_NONE_TILING: Registers the kernel with the AICore runtime.
    // "NONE_TILING" means the tiling data is read manually via GET_TILING_DATA_WITH_STRUCT
    // rather than being auto-tiled by the framework. This gives full control over tiling.
    REGISTER_NONE_TILING;

    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY):
    //   Sets this kernel to run on AIV (AI Vector) cores only.
    //   Other options include KERNEL_TYPE_AICORE (vector + cube) or KERNEL_TYPE_AICUBE.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // Kernel dispatch (design/Kernel.md §1): RANK selects the matching
    // TilingData instantiation; DTYPE_X is injected by the CANN framework.
    if constexpr (RANK == GN_TRAINING_UPDATE_RANK_4) {
        // Rank ≤ 4: use GnTrainingUpdateTilingData<GN_TRAINING_UPDATE_RANK_4>
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        GnTrainingUpdateKernel<DTYPE_X, GN_TRAINING_UPDATE_RANK_4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        // Rank 5-8: use GnTrainingUpdateTilingData<GN_TRAINING_UPDATE_RANK_8>
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        GnTrainingUpdateKernel<DTYPE_X, GN_TRAINING_UPDATE_RANK_8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
