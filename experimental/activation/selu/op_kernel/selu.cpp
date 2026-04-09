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
 * \file selu.cpp
 * \brief Selu kernel entry point (arch32)
 *
 * Template parameter D_T_X maps to data type:
 *   - float:       TilingKey 0 (direct fp32 computation)
 *   - half:        TilingKey 1 (direct fp16 computation)
 *   - bfloat16_t:  TilingKey 2 (cast to fp32 -> compute -> cast back)
 *   - int32_t:     TilingKey 3 (cast to fp32 -> compute -> cast back)
 *   - int8_t:      TilingKey 4 (cast to fp16 -> compute -> cast back)
 *
 * Kernel function signature: x, y, workspace, tiling
 */

#include "selu.h"

template <typename D_T_X>
__global__ __aicore__ void selu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SeluTilingData, tilingData, tiling);
    NsSelu::Selu<D_T_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
