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
 * \file renorm_v2.cpp
 * \brief
 */
#include "renorm_v2.h"

template <uint32_t schMode>
__global__ __aicore__ void renorm_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RenormV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(RenormV2TilingData, tilingData, tiling);

    NsRenormV2::RenormV2<DTYPE_X> op;      // 算子kernel实例获取
    op.Init(x, y, workspace, &tilingData); // 算子kernel实例初始化
    op.Process();                          // 算子kernel实例执行
}
