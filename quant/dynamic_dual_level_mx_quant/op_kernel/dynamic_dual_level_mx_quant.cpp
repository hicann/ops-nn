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
 * \file dynamic_dual_level_mx_quant.cpp
 * \brief
 */

#include "arch35/dynamic_dual_level_mx_quant_base.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace DynamicDualLevelMxQuant;
using namespace DynamicDualLevelMxQuantOp;

namespace {
template <uint64_t roundMode>
struct RoundModeMapper {
    static constexpr AscendC::RoundMode value = []() {
        if constexpr (roundMode == TPL_RINT) {
            return AscendC::RoundMode::CAST_RINT;
        } else if constexpr (roundMode == TPL_FLOOR) {
            return AscendC::RoundMode::CAST_FLOOR;
        } else if constexpr (roundMode == TPL_ROUND) {
            return AscendC::RoundMode::CAST_ROUND;
        } else {
            return AscendC::RoundMode::CAST_RINT;
        }
    }();
};
} // namespace

template <uint64_t roundMode>
__global__ __aicore__ void dynamic_dual_level_mx_quant(
    GM_ADDR x, GM_ADDR smooth_scale, GM_ADDR y, GM_ADDR level0_scale, GM_ADDR level1_scale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3101)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    GET_TILING_DATA_WITH_STRUCT(DynamicDualLevelMxQuantTilingData, tilingData, tiling);
    TPipe pipe;
    constexpr AscendC::RoundMode ascendcRoundMode = RoundModeMapper<roundMode>::value;
    DynamicDualLevelMxQuantBase<DTYPE_X, ascendcRoundMode> op(&tilingData, &pipe);
    op.Init(x, y, level0_scale, level1_scale);
    op.Process();

#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}