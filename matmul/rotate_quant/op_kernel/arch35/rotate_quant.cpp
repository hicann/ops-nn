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
 * \file rotate_quant.cpp
 * \brief Unified kernel entry point for RotateQuant operator
 */

#include "kernel_operator.h"
#include "rotate_quant_basic_cmct.h"

using namespace AscendC;
using namespace RotateQuantNS;
using namespace Cmct;
using namespace Cmct::Gemm;
using RotateQuantAptOpt::RotateQuantAptTilingData;

extern "C" __global__ __aicore__ void rotate_quant(
    GM_ADDR x, GM_ADDR rot, GM_ADDR alpha, GM_ADDR y, GM_ADDR scale, GM_ADDR workSpace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RotateQuantAptTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(RotateQuantAptTilingData, tilingData, tiling);

    __gm__ uint8_t* userWorkspace = GetUserWorkspace(workSpace);
    if (userWorkspace == nullptr) {
        return;
    }

    RotateQuantCmctKernel<DTYPE_X, DTYPE_Y, DTYPE_SCALE, layout::RowMajor>(
        x, rot, alpha, y, scale, userWorkspace, tilingData);
}
