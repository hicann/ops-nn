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
 * \file full_mn_kernel_common.h
 * \brief Public constants, type aliases and helpers for FullMNKernel (tilingKey=501).
 */

#ifndef FULL_MN_KERNEL_COMMON_H
#define FULL_MN_KERNEL_COMMON_H

#include "kernel_operator.h"
#include "../../../common/arch35/conv_util.h"
#include "../../../common/arch35/conv_common.h"

using namespace AscendC;
using namespace conv;

namespace full_mn {

// ===== L0 buffer sizes (bytes) =====
// Toolchain sys_constants.h has no __NPU_ARCH__ (9201/9202) entry, so keep local sizes
// consistent with the arch35 L0 physical layout (L0A/L0B 64KB, L0C 256KB).
constexpr uint32_t FULL_MN_L0A_SIZE = 65536;
constexpr uint32_t FULL_MN_L0B_SIZE = 65536;
constexpr uint32_t FULL_MN_L0C_SIZE = 262144;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

// L0 ping-pong half sizes
constexpr uint32_t AL0_BUF_BYTES = FULL_MN_L0A_SIZE / DOUBLE_BUFFER_NUM;
constexpr uint32_t BL0_BUF_BYTES = FULL_MN_L0B_SIZE / DOUBLE_BUFFER_NUM;

// L0C element count (single buffer, M/N full-load → one output block)
// L0C physical storage is 256KB. GetDstType<half/bfloat16_t/float>::Type = float (4 bytes) on 950.
constexpr uint32_t L0C_ELEMS = FULL_MN_L0C_SIZE / sizeof(float); // 65536

// ===== Event ID constants for MTE2_MTE1 sync =====
// AL1 segment ping-pong: event 0/1 (Buf[0]/Buf[1] preload done)
constexpr event_t EVT_AL1_BASE = static_cast<event_t>(0);
// BL1 segment ping-pong: event 2/3 (Buf[0]/Buf[1] preload done)
constexpr event_t EVT_BL1_BASE = static_cast<event_t>(2);
// batch 边界屏障：MTE1 段内全部 L1 读（loadAl0_/loadBl0_）完成后，下一 batch 的 MTE2 写
// （loadAl1Seg_/loadBl1Seg5HD_ 覆写同一 al1/bl1 缓冲）才允许执行——防跨 batch L1 读写竞争
constexpr event_t EVT_BATCH_BARRIER = static_cast<event_t>(4);

const uint32_t CONST_VALUE_4 = 4;
} // namespace full_mn

#endif // FULL_MN_KERNEL_COMMON_H
