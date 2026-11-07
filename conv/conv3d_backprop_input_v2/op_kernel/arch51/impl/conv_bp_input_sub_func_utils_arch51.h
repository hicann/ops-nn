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
 * \file conv_bp_input_sub_func_utils_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_UTILS_ARCH51_H
#define CONV3D_BP_INPUT_SUB_FUNC_UTILS_ARCH51_H

#include "../../../inc/platform.h"
#include "../../../inc/macro.h"

using AscendC::DivCeil;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

namespace Convolution3DBackpropFunc {
constexpr uint64_t V516_FBUF_BURST_UNIT_BYTES = 32;
constexpr uint64_t V516_FBUF_PRE_QUANT_BLOCK_BYTES = 128;
constexpr uint32_t V516_L1_RES_TOTAL_BYTES = 256 * 1024;
constexpr uint32_t V516_L1_RES_HALF_BYTES = V516_L1_RES_TOTAL_BYTES / 2;

constexpr uint32_t SHIFT_BIT_4 = 4;

static __aicore__ inline uint32_t DivCeil16(uint32_t a) { return (a + 15) >> SHIFT_BIT_4; }

static __aicore__ inline uint32_t AlignUp16(uint32_t a) { return DivCeil16(a) << SHIFT_BIT_4; }

template <class Intf>
static __aicore__ inline uint32_t DivCeilC0(Intf* self, const uint32_t a)
{
    return (a + self->ctx.tiling_->c0 - 1) >> self->ctx.tiling_->c0BitsB;
}

template <class Intf>
static __aicore__ inline uint32_t DivHkWk(Intf* self, uint32_t a)
{
    return self->ctx.singleShapeHWk_ > 1 ? a / self->ctx.singleShapeHWk_ : a;
}

template <class Intf>
static __aicore__ inline uint32_t DivCeilHkWk(Intf* self, uint32_t a)
{
    return self->ctx.singleShapeHWk_ > 1 ? (a + self->ctx.singleShapeHWk_ - 1) / self->ctx.singleShapeHWk_ : a;
}

constexpr uint8_t B2_TRANSPOSE_AND_REVERSE = 0;
constexpr uint8_t B2_REVERSE_ONLY = 1;
constexpr uint8_t B2_TRANSPOSE_ONLY = 2;
constexpr uint8_t B2_NO_TRANSPOSE_NO_REVERSE = 3;

template <class Intf>
static __aicore__ inline uint8_t CalcLoadB2Condition(const Intf* self)
{
    uint32_t kernelHW = self->ctx.tiling_->hk * self->ctx.tiling_->wk;
    if (self->ctx.tiling_->dkHkWk == 1) {
        return B2_TRANSPOSE_ONLY; // kernel 1×1×1，无需逆序
    }
    if (self->ctx.tiling_->baseN >= kernelHW) {
        return B2_REVERSE_ONLY; // NDHWC 且 baseN ≥ kernelHW，只逆序不转置
    }
    return B2_TRANSPOSE_AND_REVERSE;
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
        Fill(buf.template ReinterpretCast<uint16_t>(), {1, static_cast<uint16_t>(len >> 5), 0, padValue});
    } else {
        AscendC::InitConstValueParams<SrcType> initConstValueParams;
        initConstValueParams.repeatTimes = 1;
        initConstValueParams.blockNum = len >> 5;
        initConstValueParams.dstGap = 0;
        initConstValueParams.initValue = (SrcType)(0);
        Fill(buf, initConstValueParams);
    }
    PipeBarrier<PIPE_MTE2>();
}

} // namespace Convolution3DBackpropFunc

#endif
