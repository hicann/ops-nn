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
 * \file dynamic_block_mx_quant.cpp
 * \brief
 */

#include "arch35/dynamic_block_mx_quant_base.h"
#include "arch35/dynamic_block_mx_quant_struct.h"
#include "arch35/dynamic_block_mx_quant_tilingdata.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace DynamicBlockMxQuant;
using namespace DynamicBlockMxQuantOp;

namespace {
template <uint64_t roundMode>
struct RoundModeMapper {
    static constexpr AscendC::RoundMode value = []() {
        // fp8只支持rint
        if constexpr (IsSameType<DTYPE_Y, fp8_e4m3fn_t>::value) {
            return AscendC::RoundMode::CAST_RINT;
        } else if constexpr (IsSameType<DTYPE_Y, fp8_e5m2_t>::value) {
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

template <uint64_t roundMode, uint64_t scaleAlg>
__global__ __aicore__ void dynamic_block_mx_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale1, GM_ADDR mxScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    REGISTER_TILING_DEFAULT(DynamicBlockMxQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(DynamicBlockMxQuantTilingData, tilingData, tiling);
    TPipe pipe;
    constexpr AscendC::RoundMode ascendcRoundMode = RoundModeMapper<roundMode>::value;
    DynamicBlockMxQuantBase<DTYPE_X, DTYPE_Y, ascendcRoundMode, scaleAlg> op(&tilingData, &pipe);
    op.Init(x, y, mxScale1, mxScale2);
    op.Process();

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}