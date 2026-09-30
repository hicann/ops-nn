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
// bn3d_training_update_grad_package/op_host/arch35/bn3d_training_update_grad_tiling_arch35.h
// =============================================================================
//
// ROLE: Tiling header for arch35 (Ascend 950 series).
//   Bridges the shared TilingData struct (defined in the op_kernel arch35
//   tiling_struct.h, included below) into the host tiling translation unit.
//   Platform facts (coreNum / ubSize) are read directly inside the tiling
//   function via GetPlatformInfo(), so no compile-info carrier struct is
//   defined here and no TilingParse is registered (HOST-4).
//
//   The arch35 directory indicates this is for the daVinci arch 3.5 which
//   corresponds to the Ascend 950 series NPU.
//
// =============================================================================

#ifndef OPS_MATH_SCALE_OP_HOST_ARCH35_BN3D_TRAINING_UPDATE_GRAD_TILING_ARCH35_H
#define OPS_MATH_SCALE_OP_HOST_ARCH35_BN3D_TRAINING_UPDATE_GRAD_TILING_ARCH35_H

// Include the tiling data struct definition — this is shared between
// host-side tiling and device-side kernel (same struct, different compilation).
#include "../../op_kernel/arch35/bn3d_training_update_grad_tiling_struct.h"

namespace optiling {

// BN3DTrainingUpdateGradCompileInfo — empty compile-info type.
//
// A TilingParse<CompileInfo> registration is required by the op-tiling compile
// interface (TbeOptilingPyInterfaceNew), so this type exists solely as the
// template parameter. It carries NO platform facts: platform info (coreNum /
// ubSize) is read directly inside TilingFuncBN3DTrainingUpdateGrad via
// GetPlatformInfo(), never through compile info (HOST-4).
struct BN3DTrainingUpdateGradCompileInfo {};

} // namespace optiling

#endif
