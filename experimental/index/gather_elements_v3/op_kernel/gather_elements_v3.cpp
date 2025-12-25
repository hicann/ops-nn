/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
