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
 * \file dequant_swiglu_quant_common.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_COMMON_H
#define DEQUANT_SWIGLU_QUANT_COMMON_H

#include "kernel_operator.h"

namespace DequantSwigluQuantV35Ops {
using namespace AscendC;

constexpr static uint32_t DOUBLE_BUFFER = 2;
constexpr static uint32_t BLOCK_SIZE = 32;  // 32B
constexpr static uint32_t BLOCK_ELEM_B32 = BLOCK_SIZE / sizeof(float);  // 32bit数据类型对齐32B需要的元素个数
constexpr static uint32_t BLOCK_ELEM_B8 = BLOCK_SIZE / sizeof(int8_t); // 8bit数据类型对齐32B需要的元素个数 float8和int8的占用字节数一样
constexpr static uint32_t INT8SYMBOL = 2;  // dstType:int8
constexpr static uint32_t FLOAT8E5M2SYMBOL = 35;  // dstType:float8e5m2
constexpr static uint32_t FLOAT8E4M3SYMBOL = 36;  // dstType:float8e4m3
constexpr static uint32_t FLOAT4E2M1SYMBOL = 40;  // dstType:float4e2m1
constexpr static uint32_t FLOAT4E1M2SYMBOL = 41;  // dstType:float4e1m2
constexpr static int64_t SWI_FACTOR = 2;
constexpr static int64_t FP4_WEIGHT = 2;

static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP8 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::UNKNOWN
};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_RINT = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_ROUND = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_ROUND
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_FLOOR = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_FLOOR
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_CEIL = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_CEIL
};
static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_TO_FP4_TRUNC = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_TRUNC
};
static constexpr AscendC::MicroAPI::CastTrait CAST_INT32_TO_FP32 = {
  AscendC::MicroAPI::RegLayout::UNKNOWN,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_INT16 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::NO_SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_RINT
};
static constexpr AscendC::MicroAPI::CastTrait CAST_INT16_TO_FP16 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::UNKNOWN,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_ROUND
};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP16_TO_INT8 = {
  AscendC::MicroAPI::RegLayout::ZERO,
  AscendC::MicroAPI::SatMode::SAT,
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  AscendC::RoundMode::CAST_TRUNC
};
static constexpr AscendC::MicroAPI::DivSpecificMode DIV_MODE = {
  AscendC::MicroAPI::MaskMergeMode::ZEROING,
  true,
};

}
#endif  // DEQUANT_SWIGLU_QUANT_COMMON_H