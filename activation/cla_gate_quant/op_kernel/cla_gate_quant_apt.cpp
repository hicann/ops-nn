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
 * \file cla_gate_quant_apt.cpp
 * \brief ClaGateQuant kernel entry
 */

#include "arch35/cla_gate_quant_dual_axis.h"
#include "arch35/cla_gate_quant_single_axis.h"
#include "arch35/cla_gate_quant_tilingdata.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace ClaGateQuant;
using namespace ClaGateQuantOp;

namespace {
template <uint64_t roundMode>
struct RoundModeMapper {
    static constexpr AscendC::RoundMode value = []() {
        // FP8 only supports rint; FP4 can use rint/floor/round.
        if constexpr (IsSameType<DTYPE_ROW_DATA, fp8_e4m3fn_t>::value ||
                      IsSameType<DTYPE_ROW_DATA, fp8_e5m2_t>::value) {
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

// The axis bit in the tiling key selects the single- or dual-axis implementation
// at compile time.
template <uint64_t dualAxisFlag, typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode,
          uint64_t scaleAlg>
struct AxisOpSelector;

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
struct AxisOpSelector<TPL_DUAL_AXIS, xDtype, rowDataDtype, roundMode, scaleAlg> {
    using Type = ClaGateQuantDualAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>;
};

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
struct AxisOpSelector<TPL_SINGLE_AXIS, xDtype, rowDataDtype, roundMode, scaleAlg> {
    using Type = ClaGateQuantTailAxisBase<xDtype, rowDataDtype, roundMode, scaleAlg>;
};
} // namespace

template <uint64_t dualAxisFlag, uint64_t roundMode, uint64_t scaleAlg>
__global__ __aicore__ void cla_gate_quant(GM_ADDR global_attn, GM_ADDR local_attn, GM_ADDR global_gate_logits,
                                          GM_ADDR local_gate_logits, GM_ADDR row_data, GM_ADDR row_scale,
                                          GM_ADDR col_data, GM_ADDR col_scale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ClaGateQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClaGateQuantTilingData, tilingData, tiling);
    TPipe pipe;
    constexpr AscendC::RoundMode ascendcRoundMode = RoundModeMapper<roundMode>::value;
    using OpType = typename AxisOpSelector<dualAxisFlag, DTYPE_GLOBAL_ATTN, DTYPE_ROW_DATA, ascendcRoundMode,
                                           scaleAlg>::Type;
    OpType op(&tilingData, &pipe);
    op.Init(global_attn, local_attn, global_gate_logits, local_gate_logits, row_data, row_scale, col_data, col_scale);
    op.Process();
}
