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
 * \file add_rms_norm_dynamic_mx_quant_common.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H
#define ADD_RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "../inc/platform.h"
#include "op_kernel/platform_util.h"
#include "add_rms_norm_dynamic_mx_quant_tiling_data.h"

namespace AddRmsNormDynamicMxQuant {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::Add;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t OUT_ALL = 256;
constexpr int64_t MX_STEP_PROCESS_NUM = 256;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint16_t MX_FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t MX_FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t MX_ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MX_FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t MX_FP8_E4M3_MAX = 0x3b124925;

constexpr int32_t NEG_ZERO = 0x80000000;
constexpr float ONE_FOURTH = 0.25;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t FP32_BIAS = 127;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr int64_t MODE_ROUND = 0;
constexpr int64_t MODE_FLOOR = 1;
constexpr int64_t MODE_RINT = 4;

constexpr static uint32_t VL_F32 = platform::GetVRegSize() / sizeof(float); // 64
constexpr static uint32_t BLOCK_F32_ALIGN_NUM = Ops::Base::GetUbBlockSize() / sizeof(float);  // 8
constexpr static uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize();

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

__aicore__ inline uint64_t CeilAlign(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y * y;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T_IN>
__aicore__ inline void LoadTensorForDtypeTIn(
    __local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
    }
}

template <typename T_OUT>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T_OUT* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_OUT, float>::value) {
        DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        RegTensor<T_OUT> xOut;
        Cast<T_OUT, float, castTraitB322B16>(xOut, src, preg);
        DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
    }
}

} // namespace AddRmsNormDynamicMxQuant
#endif // ADD_RMS_NORM_DYNAMIC_MX_QUANT_COMMON_H