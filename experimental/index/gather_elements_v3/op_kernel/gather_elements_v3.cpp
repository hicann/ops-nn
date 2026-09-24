/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN
 * Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not
 * use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 * OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */
/*!
 * \file gather_elements_v3.cpp
 * \brief
 */

#include "gather_elements_v3.h"

template <uint32_t schMode>
__global__ __aicore__ void gather_elements_v3(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GatherElementsV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(GatherElementsV3TilingData, tilingData, tiling);

    NsGatherElementsV3::GatherElementsV3<DTYPE_X> op;
    op.Init(x, index, y, &tilingData);
    op.Process();
}
