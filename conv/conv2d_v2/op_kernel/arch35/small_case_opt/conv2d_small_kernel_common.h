/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Shared constants, dtype traits and L1-layout carrier for the stateless
 * conv2d small-kernel pipelines (full_load / cin_split / fm_partload).
 *
 * Architecture (mirrors the smallcasedemo refactor):
 *   Pipeline (entry) -> Scheduler (BlockCtx) -> Processor (orchestration)
 *   -> TileOps (stateless leaf operations).
 *
 * Unlike the demo, this keeps the FULL ops-nn feature set: groups>1, NHWC
 * in/out, dual-output, inner-batch, NCHW/FRACTAL_Z weight, per-channel scale and
 * vector relu.
 */
#ifndef CONV2D_SMALL_KERNEL_COMMON_H
#define CONV2D_SMALL_KERNEL_COMMON_H

#include "kernel_operator.h"
#include "../conv2d_v2_util.h"
#include "../../../common/arch35/conv_common.h"

namespace small_conv {

using namespace AscendC;
using namespace conv;

// ---- L0 tile geometry ----
constexpr uint32_t GN0 = 16;
constexpr uint32_t GM0 = 16;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t AL0_BUF_BYTES = static_cast<uint32_t>(ASC_L0A_SIZE) / DOUBLE_BUFFER_NUM;
constexpr uint32_t BL0_BUF_BYTES = static_cast<uint32_t>(ASC_L0B_SIZE) / DOUBLE_BUFFER_NUM;

constexpr uint32_t PAD_LIST_NUM = 4;
constexpr uint32_t PAD_BOTTOM_VALUE = 255;
constexpr uint32_t FLOAT_ONE_FIXED_POINT = 1065353216UL; // IEEE-754 float 1.0
constexpr uint32_t INVALID_GROUP_ITER = ~0u;

// ---- Events (full_load) ----
constexpr event_t EVT_MTE2_DONE = static_cast<event_t>(2);
constexpr event_t EVT_BATCH_BUF0 = static_cast<event_t>(3);
constexpr event_t EVT_BATCH_BUF1 = static_cast<event_t>(4);
constexpr event_t EVT_BIAS_DONE = static_cast<event_t>(5);
constexpr event_t EVT_GROUP_L1_DONE = static_cast<event_t>(6);
constexpr event_t EVT_GROUP_BIAS_DONE = static_cast<event_t>(7);

// ---- Events (cin_split / fm_partload) ----
constexpr event_t EVT_WBS_DONE = static_cast<event_t>(0);
constexpr event_t EVT_FMAP_BUF0 = static_cast<event_t>(1);
constexpr event_t EVT_FMAP_BUF1 = static_cast<event_t>(2);

// cin-split thresholds (mirror ops-nn)
constexpr uint32_t CIN_L1_SPLIT_QUARTER = 4;
constexpr uint32_t CIN_L1_SPLIT_HALF = 2;
constexpr uint32_t MIN_GK0_MULTIPLIER_PER_SPLIT_1X1 = 2;
constexpr uint32_t FMP_BLOCK_BYTES = 32;

// Tiling pBufferFlag bit3 (mirrors conv_util.h AL1_DB_IDX): A1 double-buffering
// switch. FM-partload derives its per-buffer size from the tiling (aL1SpaceSize),
// so it must honor this bit — a single A1 buffer with serialized cin-block
// preloads when the bit is off. Self-splitting pipelines (cin-split), whose
// per-buffer size comes from chunk geometry, always keep the double buffer.
constexpr uint32_t AL1_DB_FLAG = 0x08;

// Dtype traits: GK0/GK0Fmap for weight/fmap, L0cT (arch-dependent).
template <typename FmapT_, typename WeightT_, typename BiasT_, typename OutputT_, typename Output1T_>
struct SmallTraits {
    using FmapType = FmapT_;
    using WeightType = WeightT_;
    using BiasType = BiasT_;
    using OutputType = OutputT_;
    using Output1Type = Output1T_;
    using ScaleType = uint64_t;
    using ReluWeightType = float;
#if defined(__DAV_35_FAMILY__)
    using L0cT = int32_t;
#else
    using L0cT = float;
#endif
    static constexpr uint32_t GK0 = static_cast<uint32_t>(C0_SIZE) / sizeof(WeightT_);
    static constexpr uint32_t GK0Fmap = static_cast<uint32_t>(C0_SIZE) / sizeof(FmapT_);
    static constexpr uint32_t L0C_ELEMS = static_cast<uint32_t>(ASC_L0C_SIZE) / sizeof(L0cT);
    static constexpr uint32_t AL0_BUF_ELEMS = AL0_BUF_BYTES / sizeof(FmapT_);
    static constexpr uint32_t BL0_BUF_ELEMS = BL0_BUF_BYTES / sizeof(WeightT_);
};

// L1 byte offsets for the per-core buffers (full dual-output chain).
struct L1Layout {
    uint32_t al1ElemCount;
    uint32_t al1BufBytes;
    uint32_t bl1ElemCount;
    uint32_t bl1OffBytes;
    uint32_t biasL1OffBytes;
    uint32_t scale0L1OffBytes;
    uint32_t reluWeight0L1OffBytes;
    uint32_t scale1L1OffBytes;
    uint32_t reluWeight1L1OffBytes;
    uint32_t endBytes; // end of the whole chain; used for L1 capacity checks
};

// Channel-wise bias/scale/relu-weight L1 full-load with right padding.
template <typename ChannelWiseT>
__aicore__ inline void LoadChannelWiseL1FullLoad(const LocalTensor<ChannelWiseT>& tensorL1,
                                                 const GlobalTensor<ChannelWiseT>& tensorGm, uint64_t loadNum)
{
    uint64_t byteNum = sizeof(ChannelWiseT);
    // Zero-fill first: DataCopyPad only right-pads to 32B, while the BT copy
    // reads up to 64B alignment, so the 32B..64B gap must be pre-cleared.
    InitConstValueParams<ChannelWiseT> initParams(1, static_cast<uint16_t>(AlignB(loadNum, BLOCK_L0_N) * 4 / C0_SIZE),
                                                  0, 0); // 4 for 64 bit align.
    InitConstValue<ChannelWiseT>(tensorL1, initParams);
    PipeBarrier<PIPE_MTE2>();
    DataCopyParams dataCopyParams(1, loadNum * byteNum, 0, 0);
    uint8_t rightPadding = static_cast<uint8_t>(AlignB(loadNum * byteNum, PADDING_ALIGN_SIZE) / byteNum - loadNum);
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    DataCopyPad<ChannelWiseT>(tensorL1, tensorGm, dataCopyParams, padParams);
}

} // namespace small_conv

#endif // CONV2D_SMALL_KERNEL_COMMON_H
