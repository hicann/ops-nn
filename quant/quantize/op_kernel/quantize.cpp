/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize.cpp
 * \brief quantize kernel enter
 */

#include "kernel_operator.h"
#include "arch35/quantize_per_channel_regbase.h"
#include "arch35/quantize_per_channel_no_offset_regbase.h"
#include "arch35/quantize_per_channel_nddma_regbase.h"
#include "arch35/quantize_per_channel_nddma_no_offset_regbase.h"
#include "arch35/quantize_per_tensor_regbase.h"
#include "arch35/quantize_per_tensor_no_offset_regbase.h"
#include "arch35/quantize_per_head_regbase.h"
#include "arch35/quantize_per_head_no_offset_regbase.h"
#include "arch35/quantize_struct.h"
#include "arch35/quantize_tilingdata.h"

namespace {
using namespace AscendC;
using namespace QuantizeOp;

template <typename zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannel(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizePerChannelRegbase<DTYPE_X, DTYPE_SCALES, zeroPointsType, DTYPE_Y, DivMode, RoundMode, SqrtMode> op;
    op.Init(x, scales, zero_points, y, &tilingData);
    op.Process();
}

template <typename zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNoOffset(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizePerChannelNoOffsetRegbase<DTYPE_X, DTYPE_SCALES, zeroPointsType, DTYPE_Y, DivMode, RoundMode, SqrtMode> op;
    op.Init(x, scales, zero_points, y, &tilingData);
    op.Process();
}

template <typename zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNddma(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizePerChannelNddmaRegbase<DTYPE_X, DTYPE_SCALES, zeroPointsType, DTYPE_Y, DivMode, RoundMode, SqrtMode> op;
    op.Init(x, scales, zero_points, y, &tilingData);
    op.Process();
}

template <typename zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNddmaNoOffset(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizePerChannelNddmaNoOffsetRegbase<DTYPE_X, DTYPE_SCALES, zeroPointsType, DTYPE_Y, DivMode, RoundMode, SqrtMode>
        op;
    op.Init(x, scales, zero_points, y, &tilingData);
    op.Process();
}

template <typename zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerTensor(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizePerTensorRegbase<DTYPE_X, DTYPE_SCALES, zeroPointsType, DTYPE_Y, DivMode, RoundMode, SqrtMode> op;
    op.Init(x, scales, zero_points, y, &tilingData);
    op.Process();
}

template <typename zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerTensorNoOffset(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizePerTensorNoOffsetRegbase<DTYPE_X, DTYPE_SCALES, zeroPointsType, DTYPE_Y, DivMode, RoundMode, SqrtMode> op;
    op.Init(x, scales, zero_points, y, &tilingData);
    op.Process();
}

template <typename zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerHead(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizePerHeadRegbase<DTYPE_X, DTYPE_SCALES, zeroPointsType, DTYPE_Y, DivMode, RoundMode, SqrtMode> op;
    op.Init(x, scales, zero_points, y, &tilingData);
    op.Process();
}

template <typename zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerHeadNoOffset(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QuantizeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    QuantizePerHeadNoOffsetRegbase<DTYPE_X, DTYPE_SCALES, zeroPointsType, DTYPE_Y, DivMode, RoundMode, SqrtMode> op;
    op.Init(x, scales, zero_points, y, &tilingData);
    op.Process();
}
}; // namespace

/**
 * \brief quantize kernel main entry
 */
template <uint64_t perMode, uint64_t zeroPointsType, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__global__ __aicore__ void quantize(
    GM_ADDR x, GM_ADDR scales, GM_ADDR zero_points, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    using ZeroType = typename TypeFromEnum<zeroPointsType>::type;
    if constexpr (perMode == TPL_PER_TENSOR) {
        if constexpr (zeroPointsType == TPL_NONE) {
            QuantizePerTensorNoOffset<ZeroType, DivMode, RoundMode, SqrtMode>(
                x, scales, zero_points, y, workspace, tiling);
        } else {
            QuantizePerTensor<ZeroType, DivMode, RoundMode, SqrtMode>(x, scales, zero_points, y, workspace, tiling);
        }
    } else if constexpr (perMode == TPL_PER_CHANNEL) {
        if constexpr (zeroPointsType == TPL_NONE) {
            QuantizePerChannelNoOffset<ZeroType, DivMode, RoundMode, SqrtMode>(
                x, scales, zero_points, y, workspace, tiling);
        } else {
            QuantizePerChannel<ZeroType, DivMode, RoundMode, SqrtMode>(x, scales, zero_points, y, workspace, tiling);
        }
    } else if constexpr (perMode == TPL_PER_HEAD) {
        if constexpr (zeroPointsType == TPL_NONE) {
            QuantizePerHeadNoOffset<ZeroType, DivMode, RoundMode, SqrtMode>(
                x, scales, zero_points, y, workspace, tiling);
        } else {
            QuantizePerHead<ZeroType, DivMode, RoundMode, SqrtMode>(x, scales, zero_points, y, workspace, tiling);
        }
    } else if constexpr (perMode == TPL_PER_CHANNEL_NDDMA) {
        if constexpr (zeroPointsType == TPL_NONE) {
            QuantizePerChannelNddmaNoOffset<ZeroType, DivMode, RoundMode, SqrtMode>(
                x, scales, zero_points, y, workspace, tiling);
        } else {
            QuantizePerChannelNddma<ZeroType, DivMode, RoundMode, SqrtMode>(
                x, scales, zero_points, y, workspace, tiling);
        }
    }
}