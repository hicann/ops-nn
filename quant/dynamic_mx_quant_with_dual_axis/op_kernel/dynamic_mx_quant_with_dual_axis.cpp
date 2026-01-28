/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant_with_dual_axis.cpp
 * \brief
 */

#include "arch35/dynamic_mx_quant_with_dual_axis_base.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace DynamicMxQuantWithDualAxis;
using namespace DynamicMxQuantWithDualAxisOp;

namespace {
template <uint64_t roundMode>
struct RoundModeMapper {
    static constexpr AscendC::RoundMode value = []() {
        // fp8只支持rint
        if constexpr (IsSameType<DTYPE_Y1, fp8_e4m3fn_t>::value) {
            return AscendC::RoundMode::CAST_RINT;
        } else if constexpr (IsSameType<DTYPE_Y1, fp8_e5m2_t>::value) {
            return AscendC::RoundMode::CAST_RINT;
        } else {
            if constexpr (roundMode == TPL_RINT) {
                return AscendC::RoundMode::CAST_RINT;
            } else if constexpr (roundMode == TPL_FLOOR) {
                return AscendC::RoundMode::CAST_FLOOR;
            } else if constexpr (roundMode == TPL_ROUND) {
                return AscendC::RoundMode::CAST_ROUND;
            } else {
                return AscendC::RoundMode::CAST_RINT;
            }
        }
    }();
};
} // namespace

template <uint64_t mode, uint64_t roundMode, uint64_t scaleAlg>
__global__ __aicore__ void dynamic_mx_quant_with_dual_axis(
    GM_ADDR x, GM_ADDR y1, GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3101)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantWithDualAxisTilingData, tilingData, tiling);
    TPipe pipe;
    constexpr AscendC::RoundMode ascendcRoundMode = RoundModeMapper<roundMode>::value;
    DynamicMxQuantWithDualAxisBase<DTYPE_X, DTYPE_Y1, DTYPE_Y2, ascendcRoundMode, scaleAlg> op(&tilingData, &pipe);
    op.Init(x, y1, mxScale1, y2, mxScale2);
    op.Process();

#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}