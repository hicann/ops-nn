/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "opdev/op_log.h"
#ifndef OP_API_INC_OP_MATH_UTIL_H_
#define OP_API_INC_OP_MATH_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

inline int64_t Ceil(int64_t x, int64_t y) {
  if (y == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
    return INT64_MIN;
  }
  return ((x + y - 1) / y) * y;
}

inline int64_t CeilDiv(int64_t x, int64_t y) {
  if (y == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
    return INT64_MIN;
  }
  return (x + y - 1) / y;
}

#ifdef __cplusplus
}
#endif
#endif