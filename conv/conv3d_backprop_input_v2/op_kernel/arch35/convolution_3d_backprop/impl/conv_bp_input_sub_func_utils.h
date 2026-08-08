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
 * \file conv_bp_input_sub_func_utils.h
 * \brief Common constants, arithmetic utilities and InitZeroValue for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_UTILS_H
#define CONV3D_BP_INPUT_SUB_FUNC_UTILS_H

#include "../../../../inc/platform.h"
#include "../../../conv3d_backprop_input_v2_arch35_tiling_key.h"
#include "../../../../inc/macro.h"

using AscendC::DivCeil;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::NdDmaConfig;

namespace Convolution3DBackpropFunc {
const static uint64_t DQ_SCALAR_ONE = 0x3F800000;    // float 1.0
const static uint64_t DQ_SCALAR_QF_ONE = 0x37800000; // 1 / 2 ^ 16
constexpr uint8_t FLAG_MTE1_ID_1 = 6;
constexpr uint8_t FLAG_MTE1_ID_2 = 7;
constexpr uint8_t FLAG_FIXP_ID = 8;
constexpr uint8_t FLAG_MTE2_VEC_ID = 9;
constexpr uint8_t SYNC_MODE = 4;
constexpr uint8_t CROSS_CORE_FLAG_ID_MAX = 16;
constexpr uint8_t GROUP_NDDMA_DIM_NUM = 4;
constexpr uint8_t SUB_KERNEL_NUM = 4;
constexpr uint8_t INDEX_0 = 0;
constexpr uint8_t INDEX_1 = 1;
constexpr uint8_t INDEX_2 = 2;
constexpr uint8_t INDEX_3 = 3;
constexpr uint8_t VEC_NUM = 2;
constexpr uint8_t ONE_BLK_SHIFT_SIZE = 5;
constexpr uint8_t C04_COUT_SIZE = 4;
constexpr uint8_t C04_SHIFT_SIZE = 2;
constexpr uint8_t MASK_REG_WIDTH = AscendC::VECTOR_REG_WIDTH >> 3;
constexpr NdDmaConfig nddmaConfig = {false};
constexpr FixpipeConfig CFG_COLUMN_MAJOR_UB = {CO2Layout::COLUMN_MAJOR, true};

constexpr uint32_t UB_SIZE = AscendC::TOTAL_UB_SIZE;
constexpr uint32_t SHIFT_BIT_4 = 4;
constexpr uint8_t SHIFT_VALUE_LEN = 58;

static __aicore__ inline uint32_t Div16(uint32_t a) { return a >> SHIFT_BIT_4; }

static __aicore__ inline uint32_t DivCeil16(uint32_t a) { return (a + 15) >> SHIFT_BIT_4; }

static __aicore__ inline uint32_t AlignUp16(uint32_t a) { return DivCeil16(a) << SHIFT_BIT_4; }

static __aicore__ inline uint32_t AlignDown(uint32_t a, uint32_t rnd) { return ((a) == 0 ? 0 : ((a / rnd) * rnd)); }

static __aicore__ inline uint32_t AlignUpByDtype(uint32_t a, uint32_t dtypeBit)
{
    return ((a + ((1 << dtypeBit) - 1)) >> dtypeBit) << dtypeBit;
}

template <class Intf>
static __aicore__ inline uint32_t DivCeilC0(Intf* self, const uint32_t a)
{
    return (a + self->ctx.tiling_->c0 - 1) >> self->ctx.tiling_->c0BitsB;
}

template <class Intf>
static __aicore__ inline uint32_t AlignUpC0(Intf* self, const uint32_t a)
{
    return ((a + self->ctx.tiling_->c0 - 1) >> self->ctx.tiling_->c0BitsB) << self->ctx.tiling_->c0BitsB;
}

template <class Intf>
static __aicore__ inline uint32_t DivHkWk(Intf* self, uint32_t a)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        return self->ctx.splitHkWkList_[self->ctx.splitIndex_] > 1 ?
                   a / self->ctx.splitHkWkList_[self->ctx.splitIndex_] :
                   a;
    } else {
        return self->ctx.singleShapeHWk_ > 1 ? a / self->ctx.singleShapeHWk_ : a;
    }
}

template <class Intf>
static __aicore__ inline uint32_t DivCeilHkWk(Intf* self, uint32_t a)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        return self->ctx.splitHkWkList_[self->ctx.splitIndex_] > 1 ?
                   (a + self->ctx.splitHkWkList_[self->ctx.splitIndex_] - 1) /
                       self->ctx.splitHkWkList_[self->ctx.splitIndex_] :
                   a;
    } else {
        return self->ctx.singleShapeHWk_ > 1 ? (a + self->ctx.singleShapeHWk_ - 1) / self->ctx.singleShapeHWk_ : a;
    }
}

template <class DType>
static __aicore__ inline uint32_t DivDtypeByte(uint32_t a)
{
    if constexpr (std::is_same<DType, bfloat16_t>::value || std::is_same<DType, half>::value ||
                  std::is_same<DType, uint16_t>::value) {
        return a >> 1; // 除以2字节
    } else if constexpr (std::is_same<DType, float>::value || std::is_same<DType, uint32_t>::value) {
        return a >> 2; // 2: 除以4字节
    } else {           // hifloat8_t || fp8_e4m3fn_t
        return a;
    }
}

template <class Intf, typename SrcType>
__aicore__ inline void InitZeroValue(Intf* self, const LocalTensor<SrcType>& buf, bool useOffsetX = false)
{
    uint32_t len = buf.GetSize() * sizeof(SrcType);
    uint16_t padValue = 0;
    if constexpr (std::is_same<SrcType, int8_t>::value) {
        if (useOffsetX) {
            uint8_t offsetX = static_cast<uint8_t>(self->ctx.tiling_->offsetX);
            padValue = (static_cast<uint16_t>(offsetX)) << 8 | (static_cast<uint16_t>(offsetX));
        }
    }
    if constexpr (std::is_same<SrcType, hifloat8_t>::value || std::is_same<SrcType, fp8_e4m3fn_t>::value ||
                  std::is_same<SrcType, int8_t>::value) {
        InitConstValue(buf.template ReinterpretCast<uint16_t>(), {1, static_cast<uint16_t>(len >> 5), 0, padValue});
    } else {
        AscendC::InitConstValueParams<SrcType> initConstValueParams;
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = len >> 5;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = (SrcType)(0);
        InitConstValue(buf, initConstValueParams);
    }
    PipeBarrier<PIPE_MTE2>();
}

} // namespace Convolution3DBackpropFunc

#endif
