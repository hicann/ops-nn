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

/*!
 * \file softplus_v2_apt.cpp
 * \brief SoftplusV2 kernel entry (arch35/ascend950)
 *
 * Template parameter:
 *   - D_T_X: data type (float / half / bfloat16_t)
 */

#include "arch35/softplus_v2.h"

template <typename D_T_X>
__global__ __aicore__ void softplus_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftplusV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftplusV2TilingData, tilingData, tiling);
    NsSoftplusV2::SoftplusV2<D_T_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
