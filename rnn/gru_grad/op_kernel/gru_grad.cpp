/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_kernel.cpp
 * \brief GRU反向算子 Kernel entry
 */

#include "gru_grad.h"
#include "gru_grad_tiling_data.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

extern "C" __global__ __aicore__ void gru_grad(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h,
                                               GM_ADDR output_h, GM_ADDR reset_gate, GM_ADDR update_gate,
                                               GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy, GM_ADDR dh,
                                               GM_ADDR batch_sizes, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input,
                                               GM_ADDR dw_hidden, GM_ADDR db_input, GM_ADDR db_hidden,
                                               GM_ADDR workspace, GM_ADDR gruGradTiling)
{
    // gru_grad 不区分 dtype, 统一用一份 tiling; 直接从 buffer 还原 GruGradTilingData
    REGISTER_TILING_DEFAULT(GruGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(GruGradTilingData, tilingData, gruGradTiling);

    const TCubeTiling* dgateMMTiling = &(tilingData.dgateMMParam);
    const TCubeTiling* dwIhMMTiling = &(tilingData.dwIhMMParam);
    const TCubeTiling* dwHhMMTiling = &(tilingData.dwHhMMParam);
    const TCubeTiling* dxMMTiling = &(tilingData.dxMMParam);

    GruGradKernel<DTYPE_X> gruGradOp;

    REGIST_MATMUL_OBJ(&gruGradOp.pipe, GetSysWorkSpacePtr(), gruGradOp.dgateMM, dgateMMTiling, gruGradOp.dwIhMM,
                      dwIhMMTiling, gruGradOp.dwHhMM, dwHhMMTiling, gruGradOp.dxMM, dxMMTiling);

    gruGradOp.Init(x, w_input, w_hidden, init_h, output_h, reset_gate, update_gate, new_gate, h_n, dy, dh, batch_sizes,
                   dx, dh_prev, dw_input, dw_hidden, db_input, db_hidden, &tilingData, workspace);

    gruGradOp.Process();
}
