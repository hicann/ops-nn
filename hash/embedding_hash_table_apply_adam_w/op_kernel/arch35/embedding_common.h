/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"
#include "../../inc/hashtable_common.h"

#define EMBEDDING_THREAD_NUM 512
#define TABLE_FLAG_NUM 3

using namespace AscendC;

__aicore__ inline uint32_t ROUND_UP8(const uint32_t x) {
  constexpr uint32_t ROUND_SIZE = 8;
  if (x % ROUND_SIZE != 0) {
    return (x / ROUND_SIZE + 1) * ROUND_SIZE;
  }
  return x;
}