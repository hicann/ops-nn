
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_TRANS_QUANT_PARAM_H_
#define OP_API_INC_LEVEL2_ACLNN_TRANS_QUANT_PARAM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 将scale数据从float类型转换为硬件需要的uint64_t类型
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnTransQuantParam(const float* scaleArray, uint64_t scaleSize, const float* offsetArray,
                                           uint64_t offsetSize, uint64_t** quantParam, uint64_t* quantParamSize);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_TRANS_QUANT_PARAM_H_
