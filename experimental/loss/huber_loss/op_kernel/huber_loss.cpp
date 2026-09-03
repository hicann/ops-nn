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
 * \file huber_loss.cpp
 * \brief HuberLoss kernel entry
 *
 * The entry is a schedule-mode template instantiation and nothing else.
 * dtype arrives as DTYPE_INPUT from the build system; the none/reduce split is
 * the tiling key; mean versus sum is the divisor field in TilingData.
 */
#include "huber_loss.h"
// Must be visible in this translation unit. The kernel build rewrites the
// entry into <name>_<key>_tilingkey, and without the ASCENDC_TPL declaration
// in scope the generated wrapper fails with "no matching function for call to
// huber_loss_0_tilingkey". The mode values themselves live in the tiling
// data header, so the tiling arithmetic can share them without dragging this
// framework header along.
#include "huber_loss_tiling_key.h"

// Fallback so this file also compiles outside the operator build, where the
// build system has not defined the dtype macros.
#ifndef DTYPE_INPUT
#define DTYPE_INPUT float
#endif

template <uint32_t schMode>
__global__ __aicore__ void huber_loss(GM_ADDR input, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HuberLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(HuberLossTilingData, tilingData, tiling);
    NsHuberLoss::HuberLoss<DTYPE_INPUT, schMode> op;
    op.Init(input, target, loss, workspace, tilingData);
    op.Process();
}
