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
 * \file math_util.cc
 * \brief function of math_util
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include "math_util.h"

namespace Ops {
namespace NN {
namespace Conv {

int32_t MathUtil::GetGcd(int32_t param1, int32_t param2) {
  // get greatest common divisor of param1 and param2
  if (param1 < param2) {
    std::swap(param1, param2);
  }
  if (param2 == 0) {
    return param1;
  }
  if (param1 % param2 == 0) {
    return param2;
  } else {
    return GetGcd(param2, param1 - param2);
  }
}

void MathUtil::GetFactors(std::vector<int32_t> &factor_list, int64_t src_num, int32_t max_factor) {
  int32_t max_num = static_cast<int32_t>(std::min(src_num, static_cast<int64_t>(max_factor)));
  for (int32_t factor = 1; factor <= max_num; factor++) {
    if (src_num % factor == 0) {
      factor_list.push_back(factor);
    }
  }
}

int64_t MathUtil::Lcm(int64_t param1, int64_t param2) {
  int64_t pram1_lcm = param1;
  int64_t pram2_lcm = param2;
  int64_t temp = pram1_lcm * pram2_lcm;
  int64_t param1_temp = pram1_lcm;
  while (pram1_lcm % pram2_lcm != 0) {
    param1_temp = pram1_lcm;
    pram1_lcm = pram2_lcm;
    pram2_lcm = param1_temp % pram2_lcm;
  }
  return temp / pram2_lcm;
}

int64_t MathUtil::Lcm(int32_t param1, int32_t param2) {
  return MathUtil::Lcm(static_cast<int64_t>(param1), static_cast<int64_t>(param2));
}

}  // namespace Conv
}  // namespace NN
}  // namespace Ops
