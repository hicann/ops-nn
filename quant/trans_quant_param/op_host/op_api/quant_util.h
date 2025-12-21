/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_SRC_LEVEL2_QUANT_UTIL_H_
#define OP_API_SRC_LEVEL2_QUANT_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

// 将scale转为硬件需要的uint64_t类型数据
aclnnStatus TransQuantScaleToM1(const float *scaleArray, uint64_t scaleSize, uint64_t **quantParam);

// 将offset转为硬件需要的uint64_t类型数据
aclnnStatus TransQuantOffsetToToM1(const float *offsetArray, uint64_t offsetSize, uint64_t **quantParam);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_SRC_LEVEL2_QUANT_UTIL_H_