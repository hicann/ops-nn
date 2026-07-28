/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mse_loss_v2.cpp
 * \brief MSELossV2 kernel entry (arch35 / Ascend950)
 *
 * In UT mode (DTYPE_X defined), a non-template extern "C" entry is provided.
 * In device mode, a template entry is provided for TilingKey dispatch (dtype, buffer_mode).
 */

#include "mse_loss_v2.h"

using namespace AscendC;

#ifdef DTYPE_X
// UT entry: non-template, DTYPE_X set via compile flag -DDTYPE_X=float/half
extern "C" __global__ __aicore__ void mse_loss_v2(GM_ADDR input, GM_ADDR target, GM_ADDR output, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    NsMseLossV2::MseLossV2<DTYPE_X, 0> op;
    op.Init(input, target, output, workspace, &tilingData);
    op.Process();
}
#else
// Device entry: template for TilingKey dispatch
template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void mse_loss_v2(GM_ADDR input, GM_ADDR target, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MSELossV2Arch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(MSELossV2Arch35TilingData, tilingData, tiling);
    NsMseLossV2::MseLossV2<D_T_X, BUFFER_MODE> op;
    op.Init(input, target, output, workspace, &tilingData);
    op.Process();
}
#endif
