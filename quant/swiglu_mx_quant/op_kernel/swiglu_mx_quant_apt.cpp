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
 * \file swiglu_mx_quant.cpp
 * \brief Kernel entry point for SwiGLU + MX quantization operator
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/swiglu_mx_quant_tiling_data.h"
#include "arch35/swiglu_mx_quant_common.h"
#include "arch35/swiglu_mx_quant_last_last.h"

#define SWIGLU_MX_QUANT_LAST_LAST_NO_GROUP_INDEX       1000
#define SWIGLU_MX_QUANT_LAST_LAST_GROUP_INDEX_INT32    1100
#define SWIGLU_MX_QUANT_LAST_LAST_GROUP_INDEX_INT64    1200

using namespace AscendC;
extern "C" __global__ __aicore__ void swiglu_mx_quant(GM_ADDR x, GM_ADDR group_index,GM_ADDR y, GM_ADDR mxscale,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SwigluMxQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwigluMxQuantTilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
    TILING_KEY_IS(SWIGLU_MX_QUANT_LAST_LAST_NO_GROUP_INDEX);
    TILING_KEY_IS(SWIGLU_MX_QUANT_LAST_LAST_GROUP_INDEX_INT32);
    TILING_KEY_IS(SWIGLU_MX_QUANT_LAST_LAST_GROUP_INDEX_INT64);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif
#if TILING_KEY_VAR == SWIGLU_MX_QUANT_LAST_LAST_NO_GROUP_INDEX

    // activate_dim=-1, axis=-1 group_index不存在
    SwigluMxQuant::SwigluMxQuantLastLast<DTYPE_X, DTYPE_Y, int32_t, false> op;
    op.Init(x, group_index, y, mxscale, usrWorkspace, &tilingData, &pipe);
    op.Process();
#elif TILING_KEY_VAR == SWIGLU_MX_QUANT_LAST_LAST_GROUP_INDEX_INT32
    // activate_dim=-1, axis=-1 group_index存在且为int32
    SwigluMxQuant::SwigluMxQuantLastLast<DTYPE_X, DTYPE_Y, int32_t, true> op;
    op.Init(x, group_index,y, mxscale, usrWorkspace, &tilingData, &pipe);
    op.Process();
#elif TILING_KEY_VAR == SWIGLU_MX_QUANT_LAST_LAST_GROUP_INDEX_INT64
    // activate_dim=-1, axis=-1 group_index存在且为int64
    SwigluMxQuant::SwigluMxQuantLastLast<DTYPE_X, DTYPE_Y, int64_t, true> op;
    op.Init(x, group_index,y, mxscale, usrWorkspace, &tilingData, &pipe);
    op.Process();
#endif
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
