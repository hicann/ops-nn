/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_FP16_T_H_
#define COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_FP16_T_H_

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace ops {
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_11 = 11;
constexpr uint32_t BITSHIFT_15 = 15;
/// @ingroup fp16 basic parameter
/// @brief   fp16 exponent bias
constexpr uint32_t FP16_EXP_BIAS = 15;
/// @ingroup fp16 basic parameter
/// @brief   the exponent bit length of fp16 is 5
constexpr uint32_t FP16_MAN_LEN = 10;
/// @ingroup fp16 basic parameter
/// @brief   bit index of sign in fp16
constexpr uint32_t FP16_SIGN_INDEX = 15;
/// @ingroup fp16 basic parameter
/// @brief   sign mask of fp16         (1 00000 00000 00000)
constexpr uint32_t FP16_SIGN_MASK = 0x8000;
/// @ingroup fp16 basic parameter
/// @brief   exponent mask of fp16     (  11111 00000 00000)
constexpr uint32_t FP16_EXP_MASK = 0x7C00;
/// @ingroup fp16 basic parameter
/// @brief   mantissa mask of fp16     (        11111 11111)
constexpr uint32_t FP16_MAN_MASK = 0x03FF;
/// @ingroup fp16 basic parameter
/// @brief   conceal bit of mantissa of fp16(   1 00000 00000)
constexpr uint32_t FP16_MAN_HIDE_BIT = 0x0400;
/// @ingroup fp16 basic parameter
/// @brief   maximum value            (0111 1011 1111 1111)
constexpr uint32_t FP16_MAX = 0x7BFF;
/// @ingroup fp16 basic parameter
/// @brief   absolute maximum value   (0111 1111 1111 1111)
constexpr uint32_t FP16_ABS_MAX = 0x7FFF;
/// @ingroup fp16 basic parameter
/// @brief   maximum exponent value of fp16 is 15(11111)
constexpr int16_t FP16_MAX_EXP = 0x001F;
/// @ingroup fp16 basic parameter
/// @brief   maximum mantissa value of fp16(11111 11111)
constexpr uint32_t FP16_MAX_MAN = 0x03FF;
/// @ingroup fp16 basic operator
/// @brief   get sign of fp16
#define FP16_EXTRAC_SIGN(x) (((x) >> 15) & 1)
/// @ingroup fp16 basic operator
/// @brief   get exponent of fp16
#define FP16_EXTRAC_EXP(x) (((x) >> 10) & FP16_MAX_EXP)
/// @ingroup fp16 basic operator
/// @brief   get mantissa of fp16
#define FP16_EXTRAC_MAN(x) ((((x) >> 0) & 0x3FF) | (((((x) >> 10) & 0x1F) > 0 ? 1 : 0) * 0x400))
/// @ingroup fp16 basic operator
/// @brief   constructor of fp16 from sign exponent and mantissa
#define FP16_CONSTRUCTOR(s, e, m) (((s) << FP16_SIGN_INDEX) | ((e) << FP16_MAN_LEN) | ((m) & FP16_MAX_MAN))
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is a denormalized value
#define FP16_IS_DENORM(x) ((((x) & FP16_EXP_MASK) == 0))
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is infinite
#define FP16_IS_INF(x) (((x) & FP16_ABS_MAX) == FP16_ABS_MAX)
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is invalid
#define FP16_IS_INVALID(x) (((x) & FP16_EXP_MASK) == FP16_EXP_MASK)
/// @ingroup fp32 basic parameter
/// @brief   fp32 exponent bias
constexpr uint32_t FP32_EXP_BIAS = 127;
/// @ingroup fp32 basic parameter
/// @brief   the mantissa bit length of float/fp32 is 23
constexpr uint32_t FP32_MAN_LEN = 23;
/// @ingroup fp32 basic parameter
/// @brief   bit index of sign in float/fp32
constexpr uint32_t FP32_SIGN_INDEX = 31;
/// @ingroup fp32 basic parameter
/// @brief   sign mask of fp32         (1 0000 0000  0000 0000 0000 0000 000)
constexpr uint32_t FP32_SIGN_MASK = 0x80000000u;
/// @ingroup fp32 basic parameter
/// @brief   exponent mask of fp32     (  1111 1111  0000 0000 0000 0000 000)
constexpr uint32_t FP32_EXP_MASK = 0x7F800000u;
/// @ingroup fp32 basic parameter
/// @brief   mantissa mask of fp32     (             1111 1111 1111 1111 111)
constexpr uint32_t FP32_MAN_MASK = 0x007FFFFFu;
/// @ingroup fp32 basic parameter
/// @brief   conceal bit of mantissa of fp32      (  1  0000 0000 0000 0000 000)
constexpr uint32_t FP32_MAN_HIDE_BIT = 0x00800000u;
/// @ingroup fp32 basic parameter
/// @brief   absolute maximum value    (0 1111 1111  1111 1111 1111 1111 111)
constexpr uint32_t FP32_ABS_MAX = 0x7FFFFFFFu;
/// @ingroup fp32 basic parameter
/// @brief   maximum mantissa value of fp32    (1111 1111 1111 1111 1111 111)
constexpr uint32_t FP32_MAX_MAN = 0x7FFFFFu;
/// @ingroup fp32 basic operator
/// @brief   constructor of fp32 from sign exponent and mantissa
#define FP32_CONSTRUCTOR(s, e, m) (((s) << FP32_SIGN_INDEX) | ((e) << FP32_MAN_LEN) | ((m) & FP32_MAX_MAN))
constexpr uint32_t INT16_T_MAX = 0x7FFF;
/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 16 bits length (1111 1111 1111 1111)
constexpr uint32_t BIT_LEN16_MAX = 0xFFFF;
/// @ingroup fp16_t enum
/// @brief   round mode of last valid digital
constexpr uint32_t ROUND_TO_NEAREST = 0;

/// @ingroup fp16_t
/// @brief   Half precision float
///          bit15:       1 bit SIGN      +---+-----+------------+
///          bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
///          bit0-9:      10bit MAN       +---+-----+------------+
struct tagFp16 {
  uint16_t val;

 public:
  /// @ingroup fp16_t constructor
  /// @brief   Constructor without any param(default constructor)
  tagFp16(void) { 
    val = 0x0u; 
  }
  template <typename T>
  tagFp16(const T& value) {
    *this = value;
  }
  /// @ingroup fp16_t constructor
  /// @brief   Constructor with an uint16_t value
  tagFp16(const uint16_t &uiVal) : val(uiVal) {
  }
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] fVal float object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert float to fp16_t
  /// @return  Return fp16_t result from fVal
  tagFp16 &operator=(const float &fVal);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] uiVal uint16_t object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert uint16_t to fp16_t
  /// @return  Return fp16_t result from uiVal
  tagFp16 &operator=(const uint16_t &uiVal);
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to float/fp32
  /// @return  Return float/fp32 value of fp16_t
  operator float() const;
  /// @ingroup fp16_t conversion
  /// @brief   Override convert operator to convert fp16_t to int16_t
  /// @return  Return int16_t value of fp16_t
  operator int16_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint16_t
  /// @return  Return uint16_t value of fp16_t
  operator uint16_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to float/fp32
  /// @return  Return float/fp32 value of fp16_t
  float ToFloat();
  int16_t ToInt16();
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to uint16_t
  /// @return  Return uint16_t value of fp16_t
  uint16_t ToUInt16();
};
using fp16_t = tagFp16;

/// @ingroup fp16_t public method
/// @param [in]     val signature is negative
/// @param [in|out] s   sign of fp16_t object
/// @param [in|out] e   exponent of fp16_t object
/// @param [in|out] m   mantissa of fp16_t object
/// @brief   Extract the sign, exponent and mantissa of a fp16_t object
void ExtractFP16(const uint16_t &val, uint16_t &s, int16_t &e, uint16_t &m);
uint16_t ManRoundToNearest(bool bit0, bool bit1, bool bitLeft, uint16_t man, uint16_t shift = 0);
int16_t GetManBitLength(uint16_t man);
}  // namespace ops
#endif  // COMMON_GRAPH_FUSION_CUBE_UTILS_CUBE_FP16_T_H_
