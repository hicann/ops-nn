/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MERGE_SORT_CONSTANTS_H
#define MERGE_SORT_CONSTANTS_H

#include <cstdint>
#include "op_kernel/platform_util.h"

namespace MergeSortConstants {

constexpr uint32_t FP32_DTYPE_BYTES = 4;
constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();

constexpr uint32_t MERGE_LIST_MAX_NUM = 4;
constexpr uint32_t TWO_WAY_MERGE_LIST_NUM = 2;
constexpr uint32_t THREE_WAY_MERGE_LIST_NUM = 3;
constexpr uint32_t MERGE_INTRA_BUFFER_NUM = 2;
constexpr uint32_t MERGE_MORE_BUFFER_NUM = 1;
constexpr uint32_t MERGE_WORKSPACE_BUFFER_NUM = 2;
constexpr uint32_t MERGE_SORT_WORKSPACE_PARAM = 5U;

constexpr int32_t XOR_OP_VALUE_FP = 0x80000000;
constexpr int16_t XOR_OP_VALUE_HALF = 0x8000;

constexpr uint32_t DEALING_CONCAT_NUM_ONCE = 16;
constexpr uint32_t DEALING_SORT_NUM_ONCE = 32;
constexpr uint32_t DEALING_EXTRACT_NUM_ONCE = 32;

} // namespace MergeSortConstants

#endif
