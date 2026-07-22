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
 * \file mx_to_block_mx_quant.cpp
 * \brief Kernel entry point for MxToBlockMxQuant operator.
 * \details Template programming:
 *   - rowMode (1 bit): 0 = -2 axis is multiple of 64 (aligned), 1 = not aligned (has tail rows)
 *   - DTYPE_X / DTYPE_Y: compile-time macros from binary config
 *
 *   The compiler generates separate binaries for each (rowMode, DTYPE_X, DTYPE_Y) combination.
 */

#include "mx_to_block_mx_quant_common.h"
#include "mx_to_block_mx_quant.h"
#include "mx_to_block_mx_quant_struct.h"
#include "mx_to_block_mx_quant_tilingdata.h"
#define FLOAT_OVERFLOW_MODE_CTRL 60
using namespace MxToBlockMxQuantOp;

/**
 * @brief MxToBlockMxQuant kernel entry.
 * @tparam rowMode TPL_ROW_ALIGNED(0) or TPL_ROW_NOT_ALIGNED(1)
 *
 * DTYPE_X and DTYPE_Y are compile-time macros determined by the binary config.
 */
template <uint64_t rowMode>
__global__ __aicore__ void mx_to_block_mx_quant(GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MxToBlockMxQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(MxToBlockMxQuantTilingData, tilingData, tiling);

    (void)workspace;

    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();

    MxToBlockMxQuant<DTYPE_X, DTYPE_Y, rowMode> op;
    op.Init(x, mxScale, y, scale1, scale2, &tilingData);
    op.Process();

    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
