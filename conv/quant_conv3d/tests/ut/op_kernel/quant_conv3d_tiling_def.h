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
 * \file quant_conv3d_tiling_def.h
 * \brief Kernel UT tiling header for quant_conv3d (uses Conv3DV2TilingDataV2).
 */

#ifndef QUANT_CONV3D_TILING_DEF_H
#define QUANT_CONV3D_TILING_DEF_H

#include <cstring>
#include "kernel_tiling/kernel_tiling.h"
#include "conv3d_v2/conv3d_v2_tiling_data.h"

#define __forceinline__

namespace Ops {
namespace NN {
namespace Conv3dV2 {

inline void InitQuantConv3dTilingData(uint8_t* tiling, Conv3DV2TilingDataV2* constData)
{
    memcpy(constData, tiling, sizeof(Conv3DV2TilingDataV2));
}

} // namespace Conv3dV2
} // namespace NN
} // namespace Ops

#define GET_TILING_DATA(tilingData, tilingArg) \
    Ops::NN::Conv3dV2::Conv3DV2TilingDataV2 tilingData; \
    Ops::NN::Conv3dV2::InitQuantConv3dTilingData(tilingArg, &tilingData)

#endif // QUANT_CONV3D_TILING_DEF_H
