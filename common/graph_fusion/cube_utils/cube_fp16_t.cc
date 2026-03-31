/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_utils/cube_fp16_t.h"

namespace ops {
/**
 * @ingroup fp16_t global filed
 * @brief   round mode of last valid digital
 */
constexpr uint32_t g_round_mode = ROUND_TO_NEAREST;
constexpr uint32_t FLOATVALUEBITSIZE = 11;
constexpr uint32_t BITSHIFT_4BYTE_BITSIZE = 32;
/// @ingroup fp16_t public method
/// @param [in] bit0    whether the last preserved bit is 1 before round
/// @param [in] bit1    whether the abbreviation's highest bit is 1
/// @param [in] bitLeft whether the abbreviation's bits which not contain highest bit grater than 0
/// @param [in] man     mantissa of a fp16_t or float number, support types: uint16_t/uint32_t/uint64_t
/// @param [in] shift   abbreviation bits
/// @brief    Round fp16_t or float mantissa to nearest value
/// @return   Returns true if round 1,otherwise false;
uint16_t ManRoundToNearest(bool bit0, bool bit1, bool bitLeft, uint16_t man, uint16_t shift) {
  man = (man >> shift) + ((bit1 && (bitLeft || bit0)) ? 1 : 0);
  return man;
}
/// @ingroup fp16_t public method
/// @param [in] man    mantissa of a float number, support types: uint16_t/uint32_t/uint64_t
/// @brief   Get bit length of a uint32_t number
/// @return  Return bit length of man
int16_t GetManBitLength(uint16_t man) {
  int16_t len = 0;
  while (man) {
    man >>= 1;
    len++;
  }
  return len;
}

void ExtractFP16(const uint16_t& val, uint16_t &s, int16_t &e, uint16_t &m) {
  // 1.Extract
  s = FP16_EXTRAC_SIGN(val);
  e = FP16_EXTRAC_EXP(val);
  m = FP16_EXTRAC_MAN(val);

  // Denormal
  if (e == 0) {
    e = 1;
  }
}
/**
 * @ingroup fp16_t static method
 * @param [in] man       truncated mantissa
 * @param [in] shiftOut left shift bits based on ten bits
 * @brief   judge whether to add one to the result while converting fp16_t to other datatype
 * @return  Return true if add one, otherwise false
 */
static bool IsRoundOne(uint64_t man, uint16_t truncLen) {
  uint64_t mask0 = 0x4;
  uint64_t mask1 = 0x2;
  uint64_t mask2;
  uint16_t shiftOut = truncLen - DIM_2;
  mask0 = mask0 << shiftOut;
  mask1 = mask1 << shiftOut;
  mask2 = mask1 - 1;

  bool lastBit = ((man & mask0) > 0);
  bool truncHigh = 0;
  bool truncLeft = 0;
  if (ROUND_TO_NEAREST == g_round_mode) {
    truncHigh = ((man & mask1) > 0);
    truncLeft = ((man & mask2) > 0);
  }
  return (truncHigh && (truncLeft || lastBit));
}
/**
 * @ingroup fp16_t public method
 * @param [in] exp       exponent of fp16_t value
 * @param [in] man       exponent of fp16_t value
 * @brief   normalize fp16_t value
 * @return
 */
static void Fp16Normalize(int16_t& exp, uint16_t& man) {
  if (exp >= FP16_MAX_EXP) {
    exp = FP16_MAX_EXP - 1;
    man = FP16_MAX_MAN;
  } else if (exp == 0 && man == FP16_MAN_HIDE_BIT) {
    exp++;
    man = 0;
  }
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to float/fp32
 * @return  Return float/fp32 value of fpVal which is the value of fp16_t object
 */
static float Fp16ToFloat(const uint16_t& fpVal) {
  float ret;
  uint16_t hfSign, hfMan;
  int16_t hfExp;
  ExtractFP16(fpVal, hfSign, hfExp, hfMan);
  while (hfMan && !(hfMan & FP16_MAN_HIDE_BIT)) {
    hfMan <<= 1;
    hfExp--;
  }
  uint32_t sRet, eRet, mRet, fVal;
  sRet = hfSign;
  if (!hfMan) {
    eRet = 0;
    mRet = 0;
  } else {
    eRet = static_cast<uint32_t>(hfExp) - FP16_EXP_BIAS + FP32_EXP_BIAS;
    mRet = hfMan & FP16_MAN_MASK;
    mRet = mRet << (FP32_MAN_LEN - FP16_MAN_LEN);
  }
  fVal = FP32_CONSTRUCTOR(sRet, eRet, mRet);
  ret = *(reinterpret_cast<float *>(&fVal));
  return ret;
}

static uint16_t GetUint16ValByMan(uint16_t sRet, const uint64_t &longIntM, const uint16_t &shiftOut) {
  bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
  uint16_t mRet = static_cast<uint16_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN16_MAX);
  if (needRound && mRet < INT16_T_MAX) {
    mRet++;
  }
  if (sRet) {
    mRet = (~mRet) + 1;
  }
  if (mRet == 0) {
    sRet = 0;
  }
  uint16_t retV = (sRet << BITSHIFT_15) | (mRet);
  return retV;
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to int16_t
 * @return  Return int16_t value of fpVal which is the value of fp16_t object
 */
static int16_t Fp16ToInt16(const uint16_t& fpVal) {
  int16_t ret;
  uint16_t retV, sRet, hfE, hfM;
  sRet = FP16_EXTRAC_SIGN(fpVal);
  hfE = FP16_EXTRAC_EXP(fpVal);
  hfM = FP16_EXTRAC_MAN(fpVal);
  if (FP16_IS_DENORM(fpVal)) {  // Denormalized number
    retV = 0;
    ret = *(reinterpret_cast<uint8_t *>(&retV));
    return ret;
  }
  uint64_t longIntM = hfM;
  uint8_t overflowFlag = 0;
  uint16_t shiftOut = 0;
  if (FP16_IS_INVALID(fpVal)) {  // Inf or NaN
    overflowFlag = 1;
  } else {
    while (hfE != FP16_EXP_BIAS) {
      if (hfE > FP16_EXP_BIAS) {
        hfE--;
        longIntM = longIntM << 1;
        if (sRet == 1 && longIntM > 0x2000000Lu) {  // sign=1,negative number(<0)
          longIntM = 0x2000000Lu;                   // 10(fp16_t-man)+15(int16)=25bit
          overflowFlag = 1;
          break;
        } else if (sRet != 1 && longIntM >= 0x1FFFFFFLu) {  // sign=0,positive number(>0) Overflow
          longIntM = 0x1FFFFFFLu;                           // 10(fp16_t-man)+15(int16)=25bit
          overflowFlag = 1;
          break;
        }
      } else {
        hfE++;
        shiftOut++;
      }
    }
  }
  if (overflowFlag == 1) {
    retV = INT16_T_MAX + sRet;
  } else {
    retV = GetUint16ValByMan(sRet, longIntM, shiftOut);
  }
  ret = *(reinterpret_cast<int16_t *>(&retV));
  return ret;
}
/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to uint16_t
 * @return  Return uint16_t value of fpVal which is the value of fp16_t object
 */
static uint16_t Fp16ToUInt16(const uint16_t& fpVal) {
  uint16_t ret, sRet, mRet = 0;
  uint16_t hfE, hfM;
  sRet = FP16_EXTRAC_SIGN(fpVal);
  hfE = FP16_EXTRAC_EXP(fpVal);
  hfM = FP16_EXTRAC_MAN(fpVal);
  if (FP16_IS_DENORM(fpVal)) {  // Denormalized number
    return 0;
  }
  if (FP16_IS_INVALID(fpVal)) {  // Inf or NaN
    mRet = ~0;
  } else {
    uint64_t longIntM = hfM;
    uint16_t shiftOut = 0;

    while (hfE != FP16_EXP_BIAS) {
      if (hfE > FP16_EXP_BIAS) {
        hfE--;
        longIntM = longIntM << 1;
      } else {
        hfE++;
        shiftOut++;
      }
    }
    bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
    mRet = static_cast<uint16_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN16_MAX);
    if (needRound && mRet != BIT_LEN16_MAX) {
      mRet++;
    }
  }
  if (sRet == 1) {  // Negative number
    mRet = 0;
  }
  ret = mRet;
  return ret;
}

fp16_t& fp16_t::operator=(const float& fVal) {
  uint16_t sRet, mRet;
  int16_t eRet;
  uint32_t eF, mF;
  uint32_t ui32V = *(const_cast<uint32_t *>(reinterpret_cast<const uint32_t *>(&fVal)));  // 1:8:23bit sign:exp:man
  uint32_t mLenDelta;

  sRet = static_cast<uint16_t>((ui32V & FP32_SIGN_MASK) >> FP32_SIGN_INDEX);  // 4Byte->2Byte
  eF = (ui32V & FP32_EXP_MASK) >> FP32_MAN_LEN;                    // 8 bit exponent
  mF = (ui32V & FP32_MAN_MASK);                                    // 23 bit mantissa dont't need to care about denormal
  mLenDelta = FP32_MAN_LEN - FP16_MAN_LEN;

  bool needRound = false;
  // Exponent overflow/NaN converts to signed inf/NaN
  if (eF > 0x8Fu) {  // 0x8Fu:142=127+15
    eRet = FP16_MAX_EXP - 1;
    mRet = FP16_MAX_MAN;
  } else if (eF <= 0x70u) {  // 0x70u:112=127-15 Exponent underflow converts to denormalized half or signed zero
    eRet = 0;
    if (eF >= 0x67) {  // 0x67:103=127-24 Denormal
      mF = (mF | FP32_MAN_HIDE_BIT);
      uint16_t shiftOut = FP32_MAN_LEN;
      uint64_t mTmp = (static_cast<uint64_t>(mF)) << (eF - 0x67);

      needRound = IsRoundOne(mTmp, shiftOut);
      mRet = static_cast<uint16_t>(mTmp >> shiftOut);
      if (needRound) {
        mRet++;
      }
    } else if (eF == 0x66 && mF > 0) {  // 0x66:102 Denormal 0<f_v<min(Denormal)
      mRet = 1;
    } else {
      mRet = 0;
    }
  } else {  // Regular case with no overflow or underflow
    eRet = static_cast<int16_t>(eF - 0x70u);

    needRound = IsRoundOne(mF, mLenDelta);
    mRet = static_cast<uint16_t>(mF >> mLenDelta);
    if (needRound) {
      mRet++;
    }
    if (mRet & FP16_MAN_HIDE_BIT) {
      eRet++;
    }
  }

  Fp16Normalize(eRet, mRet);
  val = FP16_CONSTRUCTOR(sRet, static_cast<uint16_t>(eRet), mRet);
  return *this;
}

fp16_t& fp16_t::operator=(const uint16_t& uiVal) {
  if (uiVal == 0) {
    val = 0;
  } else {
    int16_t eRet;
    uint16_t mRet = uiVal;
    uint16_t mMin = FP16_MAN_HIDE_BIT;
    uint16_t mMax = mMin << 1;
    uint16_t len = static_cast<uint16_t>(GetManBitLength(mRet));
    if (len > FLOATVALUEBITSIZE) {
      eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
      uint32_t mTrunc;
      uint32_t truncMask = 1;
      uint16_t eTmp = len - FLOATVALUEBITSIZE;
      for (int i = 1; i < eTmp; i++) {
        truncMask = (truncMask << 1) + 1;
      }
      mTrunc = (mRet & truncMask) << (BITSHIFT_4BYTE_BITSIZE - eTmp);
      for (int i = 0; i < eTmp; i++) {
        mRet = (mRet >> 1);
        eRet = eRet + 1;
      }
      bool bLastBit = ((mRet & 1) > 0);
      bool bTruncHigh = 0;
      bool bTruncLeft = 0;
      if (ROUND_TO_NEAREST == g_round_mode) {  // trunc
        bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
        bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
      }
      mRet = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mRet);
      while (mRet >= mMax || eRet < 0) {
        mRet = mRet >> 1;
        eRet = eRet + 1;
      }
      if (FP16_IS_INVALID(val)) {
        val = FP16_MAX;
      }
    } else {
      eRet = FP16_EXP_BIAS;
      mRet = mRet << (DIM_11 - len);
      eRet = eRet + (len - 1);
    }
    val = FP16_CONSTRUCTOR(0u, static_cast<uint16_t>(eRet), mRet);
  }
  return *this;
}
// convert
fp16_t::operator float() const {
  return Fp16ToFloat(val);
}
fp16_t::operator int16_t() const {
  return Fp16ToInt16(val);
}
fp16_t::operator uint16_t() const {
  return Fp16ToUInt16(val);
}
float fp16_t::ToFloat() {
  return Fp16ToFloat(val);
}
int16_t fp16_t::ToInt16() {
  return Fp16ToInt16(val);
}
uint16_t fp16_t::ToUInt16() {
  return Fp16ToUInt16(val);
}
}  // namespace ops
