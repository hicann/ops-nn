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
 * \file math_util.h
 * \brief
 */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace Ops {
namespace NN {
namespace Conv {

class MathUtil {
 public:
  static int32_t GetGcd(int32_t param1, int32_t param2);
  static void GetFactors(std::vector<int32_t> &factor_list, int64_t src_num, int32_t max_factor);
  static int64_t Lcm(int64_t param1, int64_t param2);
  static int64_t Lcm(int32_t param1, int32_t param2);
};

}  // namespace Conv
}  // namespace NN
}  // namespace Ops
