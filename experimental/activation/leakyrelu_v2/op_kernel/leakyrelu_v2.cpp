/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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
 * \file leakyrelu_v2.cpp
 * \brief
 */

#include "leakyrelu_v2.h"

template <uint32_t schMode>
__global__ __aicore__ void leakyrelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LeakyreluV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(LeakyreluV2TilingData, tilingData, tiling);
    NsLeakyreluV2::LeakyreluV2<DTYPE_X> op; 
    op.Init(x, y, &tilingData);      
    op.Process();                    
}