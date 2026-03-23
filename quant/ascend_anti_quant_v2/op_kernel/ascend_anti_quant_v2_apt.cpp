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
 * \file ascend_anti_quant_v2_apt.cpp
 * \brief ascend_anti_quant_v2 kernel enter
 */

#include "kernel_operator.h"
#include "arch35/ascend_anti_quant_v2_struct.h"
#include "arch35/ascend_anti_quant_v2_tilingdata.h"
#include "arch35/ascend_anti_quant_v2_per_channel_no_offset_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_channel_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_tensor_no_offset_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_tensor_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_head_no_offset_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_head_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_channel_nddma_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_channel_nddma_no_offset_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_head_nddma_regbase.h"
#include "arch35/ascend_anti_quant_v2_per_head_nddma_no_offset_regbase.h"

using namespace AscendC;
using namespace AscendAntiQuantV2;
using namespace AscendAntiQuantV2Op;

template <uint64_t perMode, uint64_t zeroPointsType, uint64_t sqrtMode>
__global__ __aicore__ void ascend_anti_quant_v2(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (perMode == TPL_PER_TENSOR) {
        REGISTER_TILING_DEFAULT(AscendAntiQuantV2TilingData);
        GET_TILING_DATA_WITH_STRUCT(AscendAntiQuantV2TilingData, tilingData, tiling);
        if constexpr (zeroPointsType == TPL_HAS_OFFSET) {
            AscendAntiQuantV2PerTensorRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        } else {
            AscendAntiQuantV2PerTensorNoOffsetRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        }
    } else if constexpr (perMode == TPL_PER_CHANNEL) {
        REGISTER_TILING_DEFAULT(AscendAntiQuantV2TilingData);
        GET_TILING_DATA_WITH_STRUCT(AscendAntiQuantV2TilingData, tilingData, tiling);
        if constexpr (zeroPointsType == TPL_HAS_OFFSET) {
            AscendAntiQuantV2PerChannelRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        } else {
            AscendAntiQuantV2PerChannelNoOffsetRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        }
    } else if constexpr (perMode == TPL_PER_HEAD) {
        REGISTER_TILING_DEFAULT(AscendAntiQuantV2TilingData);
        GET_TILING_DATA_WITH_STRUCT(AscendAntiQuantV2TilingData, tilingData, tiling);
        if constexpr (zeroPointsType == TPL_HAS_OFFSET) {
            AscendAntiQuantV2PerHeadRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        } else {
            AscendAntiQuantV2PerHeadNoOffsetRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        }
    } else if constexpr (perMode == TPL_PER_CHANNEL_NDDMA) {
        REGISTER_TILING_DEFAULT(AscendAntiQuantV2TilingData);
        GET_TILING_DATA_WITH_STRUCT(AscendAntiQuantV2TilingData, tilingData, tiling);
        if constexpr (zeroPointsType == TPL_HAS_OFFSET) {
            AscendAntiQuantV2PerChannelNddmaRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        } else {
            AscendAntiQuantV2PerChannelNddmaNoOffsetRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        }
    } else if constexpr (perMode == TPL_PER_HEAD_NDDMA) {
        REGISTER_TILING_DEFAULT(AscendAntiQuantV2TilingData);
        GET_TILING_DATA_WITH_STRUCT(AscendAntiQuantV2TilingData, tilingData, tiling);
        if constexpr (zeroPointsType == TPL_HAS_OFFSET) {
            AscendAntiQuantV2PerHeadNddmaRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        } else {
            AscendAntiQuantV2PerHeadNddmaNoOffsetRegbase<DTYPE_X, DTYPE_SCALE, DTYPE_SCALE, DTYPE_Y, sqrtMode> op(&tilingData);
            op.Init(x, scale, offset, y);
            op.Process();
        }
    }
}