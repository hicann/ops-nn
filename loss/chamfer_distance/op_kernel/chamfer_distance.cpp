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
 * \file chamfer_distance.cpp
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/chamfer_distance_nd.h"
#include "arch35/chamfer_distance_tiling_data.h"

using namespace AscendC;

// xyz1/xyz2/dist1/dist2 同 dtype, 由 def 的 dtype profile 驱动框架注入 DTYPE_XYZ1 并分别实例化
// (float / half / bfloat16_t); idx 恒 int32。dtype 不进 TilingKey, 见 arch35/chamfer_distance_tiling_key.h。
__global__ __aicore__ void chamfer_distance(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist1, GM_ADDR dist2, GM_ADDR idx1,
                                            GM_ADDR idx2, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ChamferDistanceArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(ChamferDistanceArch35TilingData, tilingData, tiling);

    TPipe pipe;
    ChamferDistance::ChamferDistanceND<DTYPE_XYZ1> op;
    op.Init(xyz1, xyz2, dist1, dist2, idx1, idx2, &tilingData, &pipe);
    op.Process();
}
