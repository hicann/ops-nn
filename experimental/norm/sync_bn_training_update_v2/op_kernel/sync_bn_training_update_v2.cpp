/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
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
 * \file sync_bn_training_update_v2.cpp
 * \brief
 */
#include "sync_bn_training_update_v2.h"

template <uint32_t schMode>
__global__ __aicore__ void sync_bn_training_update_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SyncBnTrainingUpdateV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SyncBnTrainingUpdateV2TilingData, tilingData, tiling);
    NsSyncBnTrainingUpdateV2::SyncBnTrainingUpdateV2<float> op; // 算子kernel实例获取
    op.Init(x1, x2, x3, y1, y2, y3, workspace, &tilingData);      // 算子kernel实例初始化
    op.Process();                       // 算子kernel实例执行
}