/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deformable_offsets.cpp
 * \brief deformable_offsets kernel main
 */

#include "arch35/deformable_offsets.h"

#define TILING_SIMT_COMMON_KEY 1000

using namespace DeformableOffsets;
using namespace AscendC;

extern "C" __global__ __aicore__ void deformable_offsets(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILING_SIMT_COMMON_KEY)) {
        DeformableOffset<DTYPE_X> deformableOffsetObject;
        deformableOffsetObject.Init(x, offsets, y, workspace, &tilingData);
        deformableOffsetObject.Process();
    }
}
