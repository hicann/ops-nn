/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file celu_v3_arch32.cpp
 * \brief CeluV3 kernel entry point (arch32)
 *
 * Template parameter D_T_X maps to data type:
 *   - float:       TilingKey 0 (direct fp32 computation)
 *   - half:        TilingKey 1 (cast to fp32 -> compute -> cast back)
 *   - bfloat16_t:  TilingKey 2 (cast to fp32 -> compute -> cast back)
 *
 * Kernel function signature: self, out, workspace, tiling
 */

#include "celu_v3.h"

template <typename D_T_X>
__global__ __aicore__ void celu_v3(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CeluV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(CeluV3TilingData, tilingData, tiling);
    NsCeluV3::CeluV3<D_T_X> op;
    op.Init(self, out, &tilingData);
    op.Process();
}
