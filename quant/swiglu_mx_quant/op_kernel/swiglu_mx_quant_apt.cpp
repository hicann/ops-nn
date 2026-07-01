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
#include "arch35/swiglu_mx_quant_tiling_key.h"
#include "arch35/swiglu_mx_quant_tiling_data.h"
#include "arch35/swiglu_mx_quant_common.h"
#include "arch35/swiglu_mx_quant_axis_last.h"
#include "arch35/swiglu_mx_quant_axis_not_last.h"

using namespace AscendC;
using namespace SwigluMxQuantOp;

namespace {
template <uint64_t roundMode>
struct RoundModeMapper {
    static constexpr AscendC::RoundMode value = []() {
        // fp8 only supports rint
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

template <template<typename, typename, typename, bool, AscendC::RoundMode, bool> class OpClass,
          bool IsActLast, uint64_t GroupIndexType, uint64_t RoundMode>
__aicore__ inline void LaunchQuantOp(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale,
                                     GM_ADDR usrWorkspace, const SwigluMxQuantTilingData* tilingData, TPipe* pipe)
{
    constexpr AscendC::RoundMode ascendcRoundMode = RoundModeMapper<RoundMode>::value;
    if constexpr (GroupIndexType == TPL_NO_GROUP_INDEX) {
        OpClass<DTYPE_X, DTYPE_Y, int32_t, false, ascendcRoundMode, IsActLast> op;
        op.Init(x, group_index, y, mxscale, usrWorkspace, tilingData, pipe);
        op.Process();
    } else if constexpr (GroupIndexType == TPL_GROUP_INDEX_INT32) {
        OpClass<DTYPE_X, DTYPE_Y, int32_t, true, ascendcRoundMode, IsActLast> op;
        op.Init(x, group_index, y, mxscale, usrWorkspace, tilingData, pipe);
        op.Process();
    } else {
        OpClass<DTYPE_X, DTYPE_Y, int64_t, true, ascendcRoundMode, IsActLast> op;
        op.Init(x, group_index, y, mxscale, usrWorkspace, tilingData, pipe);
        op.Process();
    }
}
} // namespace

template <uint64_t groupIndexType, uint64_t axisLast, uint64_t activateDimLast, uint64_t roundMode>
__global__ __aicore__ void swiglu_mx_quant(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale,
                                            GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SwigluMxQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwigluMxQuantTilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if constexpr (activateDimLast == TPL_ACTIVATE_LAST && axisLast == TPL_AXIS_LAST) {
        LaunchQuantOp<SwigluMxQuant::SwigluMxQuantAxisLast, true, groupIndexType, roundMode>(
            x, group_index, y, mxscale, usrWorkspace, &tilingData, &pipe);
    }
    if constexpr (activateDimLast == TPL_ACTIVATE_LAST && axisLast == TPL_AXIS_NOT_LAST) {
        LaunchQuantOp<SwigluMxQuant::SwigluMxQuantAxisNotLast, true, groupIndexType, roundMode>(
            x, group_index, y, mxscale, usrWorkspace, &tilingData, &pipe);
    }
    if constexpr (activateDimLast == TPL_ACTIVATE_NOT_LAST && axisLast == TPL_AXIS_LAST) {
        LaunchQuantOp<SwigluMxQuant::SwigluMxQuantAxisLast, false, groupIndexType, roundMode>(
            x, group_index, y, mxscale, usrWorkspace, &tilingData, &pipe);
    }
    if constexpr (activateDimLast == TPL_ACTIVATE_NOT_LAST && axisLast == TPL_AXIS_NOT_LAST) {
        LaunchQuantOp<SwigluMxQuant::SwigluMxQuantAxisNotLast, false, groupIndexType, roundMode>(
            x, group_index, y, mxscale, usrWorkspace, &tilingData, &pipe);
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
