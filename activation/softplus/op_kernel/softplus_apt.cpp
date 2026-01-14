/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softplus_apt.cpp
 * \brief softplus impl
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "../inc/platform.h"
#include "arch35/softplus_dag.h"
#include "arch35/softplus_tilingdata.h"

using namespace AscendC;
using namespace SoftplusOp;

KERNEL_API void softplus(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SoftplusTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftplusTilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        ElementwiseSch<0UL, SoftplusDag<DTYPE_X, float>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(x, y);
        sch.Process();
    }
    return;
}
