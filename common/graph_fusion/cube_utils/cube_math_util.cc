/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_utils/cube_math_util.h"

namespace ops {
namespace {
const uint16_t kFp16ExpBias = 15;
const uint32_t kFp32ExpBias = 127;
const uint16_t kFp16ManLen = 10;
const uint32_t kFp32ManLen = 23;
const uint32_t kFp32SignIndex = 31;
const uint16_t kFp16ManMask = 0x03FF;
const uint16_t kFp16ManHideBit = 0x0400;
const uint16_t kFp16MaxExp = 0x001F;
const uint32_t kFp32MaxMan = 0x7FFFFF;
const size_t kInt16BitsNum = 16;
constexpr int shift = 13;
constexpr int shiftSign = 16;
constexpr int infN = 0x7F800000; // fp32 infinity
constexpr int maxN = 0x477FE000; // max fp16 normal as a fp32
constexpr int minN = 0x38800000; // min fp16 normal as a fp32
constexpr int signN = 0x80000000; // fp32 sign bit
constexpr int infC = infN >> shift;
constexpr int nanN = (infC + 1) << shift; // minimum fp16 nan as a fp32
constexpr int maxC = maxN >> shift;
constexpr int minC = minN >> shift;
constexpr int signC = signN >> shiftSign; // fp16 sign bit
constexpr int32_t mulN = 0x52000000; // (1 << 23) / minN
constexpr int32_t mulC = 0x33800000; // minN / (1 << (23 - shift))
constexpr int32_t subC = 0x003FF; // max fp32 subnormal down shifted
constexpr int32_t norC = 0x00400; // min fp32 normal down shifted
constexpr int32_t maxD = infC - maxC - 1;
constexpr int32_t minD = minC - subC - 1;
}

using TransformUtil = union {
  float value_f;
  int value_s;
  uint32_t value_us;
};

uint16_t Fp32ToFp16(const float &fp32_value) {
  TransformUtil data, data_s;
  data.value_f = fp32_value;
  uint32_t sign = data.value_s & signN;
  data.value_s ^= sign;
  sign >>= shiftSign; // logical shift
  data_s.value_s = mulN;
  data_s.value_s = static_cast<int>(data_s.value_f * data.value_f); // correct subnormals
  int vals_lt_minn = -(data.value_s < minN);
  data.value_s ^= (data_s.value_s ^ data.value_s) & vals_lt_minn;
  int vals_lt_infn = (data.value_s < infN);
  int vals_gt_maxn = (data.value_s > maxN);
  data.value_s ^= (infN ^ data.value_s) & -(vals_lt_infn & vals_gt_maxn);
  int vals_lt_nann = (data.value_s < nanN);
  int vals_gt_infn = (data.value_s > infN);
  data.value_s ^= (nanN ^ data.value_s) & -(vals_lt_nann & vals_gt_infn);
  data.value_us >>= shift; // logical shift
  int vals_gt_maxc = -(data.value_s > maxC);
  data.value_s ^= ((data.value_s - maxD) ^ data.value_s) & vals_gt_maxc;
  int vals_gt_subc = -(data.value_s > subC);
  data.value_s ^= ((data.value_s - minD) ^ data.value_s) & vals_gt_subc;
  return data.value_us | sign;
}
}  // namespace fe
