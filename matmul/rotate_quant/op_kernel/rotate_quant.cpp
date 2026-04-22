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
#include "rotate_quant.h"

using namespace AscendC;
using namespace RotateQuantOpt;

#ifdef __CCE_KT_TEST__
#define KERNEL_LINKAGE static
#else
#define KERNEL_LINKAGE extern "C"
#endif

KERNEL_LINKAGE __global__
    __aicore__ void rotate_quant(GM_ADDR x, GM_ADDR rot, GM_ADDR y, GM_ADDR scale, GM_ADDR workSpace, GM_ADDR tiling)
{
    if (x == nullptr || rot == nullptr || y == nullptr || scale == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(RotateQuantTilingData);
    GET_TILING_DATA(tilingData, tiling);
    __gm__ uint8_t* userWorkspace = GetUserWorkspace(workSpace);
    if (userWorkspace == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    RotateQuantOpt::InitParams params {x, rot, y, scale, userWorkspace, &pipe, &tilingData};

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X>;
    using MT = matmul::MatmulImpl<aType, bType, cType, cType, CFG_MDL>;
    
    MT mm;
    mm.Init(&tilingData.matmulTiling, &pipe);

    RotateQuant<DTYPE_X, DTYPE_Y, MT> op(mm);
    op.Init(params);
    op.Process();
}
