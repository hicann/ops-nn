/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
 * \file hard_swish_grad_apt.cpp
 * \brief HardSwishGrad kernel entry (arch35)
 *
 * In UT mode (DTYPE_X defined), a non-template extern "C" entry is provided.
 * In device mode, a template entry is provided for TilingKey dispatch.
 */

#include "arch35/hard_swish_grad.h"

using namespace AscendC;

#ifdef DTYPE_X
// UT entry: non-template, DTYPE_X set via compile flag -DDTYPE_X=float/half
extern "C" __global__ __aicore__ void hard_swish_grad(
    GM_ADDR grad_output, GM_ADDR self, GM_ADDR grad_input,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    NsHardSwishGrad::HardSwishGrad<DTYPE_X, 0> op;
    op.Init(grad_output, self, grad_input, &tilingData);
    op.Process();
}
#else
// Device entry: template for TilingKey dispatch
template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void hard_swish_grad(
    GM_ADDR grad_output, GM_ADDR self, GM_ADDR grad_input,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HardSwishGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardSwishGradTilingData, tilingData, tiling);
    NsHardSwishGrad::HardSwishGrad<D_T_X, BUFFER_MODE> op;
    op.Init(grad_output, self, grad_input, &tilingData);
    op.Process();
}
#endif
