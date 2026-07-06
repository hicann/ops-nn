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
 * \file anti_mx_quant.cpp
 * \brief Kernel entry point for AntiMxQuant operator.
 * \details Template programming:
 *   - axisMode (1 bit): 0=tail axis, 1=non-tail axis (reserved)
 *   - DTYPE_X / DTYPE_Y: compile-time macros from binary config
 *
 *   The compiler generates separate binaries for each (axisMode, DTYPE_X, DTYPE_Y)
 *   combination, matching the entries in anti_mx_quant_binary.json.
 */

#include "arch35/anti_mx_quant_tail_axis.h"
#include "arch35/anti_mx_quant_struct.h"
#include "arch35/anti_mx_quant_tilingdata.h"
#define FLOAT_OVERFLOW_MODE_CTRL 60
using namespace AntiMxQuant;

/**
 * @brief AntiMxQuant kernel entry.
 * @tparam axisMode TPL_AXIS_TAIL(0) or TPL_AXIS_NOT_TAIL(1)
 *
 * DTYPE_X and DTYPE_Y are compile-time macros determined by the binary config.
 * They map to the 12 dtype combinations declared in anti_mx_quant_binary.json.
 */
template <uint64_t axisMode>
__global__ __aicore__ void anti_mx_quant(GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AntiMxQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(AntiMxQuantTilingData, tilingData, tiling);

    (void)workspace; // Tail-axis path does not use workspace

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if constexpr (axisMode == TPL_AXIS_TAIL) {
        AntiMxQuantTailAxis<DTYPE_X, DTYPE_Y> op;
        op.Init(x, mxScale, y, &tilingData);
        op.Process();
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
