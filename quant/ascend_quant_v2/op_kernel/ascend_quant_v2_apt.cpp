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
 * \file ascend_quant_v2_regbase_apt.cpp
 * \brief ascend_quant_v2 kernel enter
 */

#include "kernel_operator.h"
#include "arch35/ascend_quant_v2_struct.h"
#include "../quantize/arch35/quantize_per_channel_regbase.h"
#include "../quantize/arch35/quantize_per_channel_no_offset_regbase.h"
#include "../quantize/arch35/quantize_per_tensor_regbase.h"
#include "../quantize/arch35/quantize_per_tensor_no_offset_regbase.h"
#include "../quantize/arch35/quantize_per_head_regbase.h"
#include "../quantize/arch35/quantize_per_head_no_offset_regbase.h"
#include "../quantize/arch35/quantize_tilingdata.h"
#define FLOAT_OVERFLOW_MODE_CTRL 60

namespace {
using namespace AscendC;
using namespace AscendQuantV2Op;

template <uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void AscendQuantV2PerChannel(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizeOp::QuantizePerChannelRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_X, DTYPE_Y, 2, RoundMode, SqrtMode> op;
    op.Init(x, scale, offset, y, &tilingData);
    op.Process();
}

template <uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void AscendQuantV2PerChannelNoOffset(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizeOp::QuantizePerChannelNoOffsetRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_X, DTYPE_Y, 2, RoundMode, SqrtMode> op;
    op.Init(x, scale, offset, y, &tilingData);
    op.Process();
}

template <uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void AscendQuantV2PerTensor(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizeOp::QuantizePerTensorRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_X, DTYPE_Y, 2, RoundMode, SqrtMode> op;
    op.Init(x, scale, offset, y, &tilingData);
    op.Process();
}

template <uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void AscendQuantV2PerTensorNoOffset(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizeOp::QuantizePerTensorNoOffsetRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_X, DTYPE_Y, 2, RoundMode, SqrtMode> op;
    op.Init(x, scale, offset, y, &tilingData);
    op.Process();
}

template <uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void AscendQuantV2PerHead(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizeOp::QuantizePerHeadRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_X, DTYPE_Y, 2, RoundMode, SqrtMode> op;
    op.Init(x, scale, offset, y, &tilingData);
    op.Process();
}

template <uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void AscendQuantV2PerHeadNoOffset(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizeOp::QuantizePerHeadNoOffsetRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_X, DTYPE_Y, 2, RoundMode, SqrtMode> op;
    op.Init(x, scale, offset, y, &tilingData);
    op.Process();
}
}; // namespace

/**
 * \brief quantize kernel main entry
 */
template <uint64_t perMode, uint64_t zeroPointsType, uint64_t RoundMode, uint64_t SqrtMode>
__global__ __aicore__ void ascend_quant_v2(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    #if (__NPU_ARCH__ == 3101)
        int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    #endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (perMode == TPL_PER_TENSOR) {
        if constexpr (zeroPointsType == TPL_NO_OFFSET) {
            AscendQuantV2PerTensorNoOffset<RoundMode, SqrtMode>(x, scale, offset, y, workspace, tiling);
        } else {
            AscendQuantV2PerTensor<RoundMode, SqrtMode>(x, scale, offset, y, workspace, tiling);
        }
    } else if constexpr (perMode == TPL_PER_CHANNEL) {
        if constexpr (zeroPointsType == TPL_NO_OFFSET) {
            AscendQuantV2PerChannelNoOffset<RoundMode, SqrtMode>(x, scale, offset, y, workspace, tiling);
        } else {
            AscendQuantV2PerChannel<RoundMode, SqrtMode>(x, scale, offset, y, workspace, tiling);
        }
    } else if constexpr (perMode == TPL_PER_HEAD) {
        if constexpr (zeroPointsType == TPL_NO_OFFSET) {
            AscendQuantV2PerHeadNoOffset<RoundMode, SqrtMode>(x, scale, offset, y, workspace, tiling);
        } else {
            AscendQuantV2PerHead<RoundMode, SqrtMode>(x, scale, offset, y, workspace, tiling);
        }
    }
    #if (__NPU_ARCH__ == 3101)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
    #endif
}