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
 * \file swiglu_mx_quant_with_dual_axis_apt.cpp
 * \brief SwigluMxQuantWithDualAxis kernel entry (regbase mode)
 */

#include "arch35/swiglu_mx_quant_with_dual_axis_regbase.h"
#include "arch35/swiglu_mx_quant_with_dual_axis_tilingdata.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace SwigluMxQuantWithDualAxis;
using namespace SwigluMxQuantWithDualAxisOp;

namespace {
template <uint64_t roundMode>
struct RoundModeMapper {
    static constexpr AscendC::RoundMode value = []() {
        // fp8 only supports rint
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

template <uint64_t mode, uint64_t roundMode, uint64_t scaleAlg, uint64_t isGroupIdx>
__global__ __aicore__ void swiglu_mx_quant_with_dual_axis(
    GM_ADDR x, GM_ADDR group_index,
    GM_ADDR y1, GM_ADDR mx_scale1,
    GM_ADDR y2, GM_ADDR mx_scale2,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    REGISTER_TILING_DEFAULT(SwigluMxQuantWithDualAxisTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwigluMxQuantWithDualAxisTilingData, tilingData, tiling);

    TPipe pipe;
    constexpr AscendC::RoundMode ascendcRoundMode = RoundModeMapper<roundMode>::value;
    SwigluMxQuantWithDualAxisBase<DTYPE_X, DTYPE_Y1, mode, ascendcRoundMode, scaleAlg, isGroupIdx> op(
        &tilingData, &pipe);
    op.Init(x, group_index, y1, mx_scale1, y2, mx_scale2);
    op.Process();

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
