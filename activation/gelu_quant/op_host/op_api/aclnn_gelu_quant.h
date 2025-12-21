/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GELU_QUANT_H_
#define OP_API_INC_LEVEL2_ACLNN_GELU_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGeluQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] self: 待进行GeluQuant计算的入参。npu device侧的aclTensor，
 * 数据类型支持float16、 bfloat16、float,数据格式支持ND，支持非连续的Tensor。
 * @param [in] inputScaleOptional: npu device侧的aclTensor，数据类型支持float16、 bfloat16、float
 * @param [in] inputOffsetOptional: npu device侧的aclTensor，数据类型支持float16、 bfloat16、float
 * @param [in] approximate:  host侧的aclScalar，数据类型string，仅支持 "none","tanh"
 * @param [in] quantMode:  host侧的aclScalar，数据类型string，仅支持 "static","dynamic"
 * @param [in] roundMode:  host侧的aclScalar，数据类型string，支持 "rint","round","hybrid","floor","ceil","trunc"
 * @param [in] dstType:  host侧的aclScalar, 数据类型int, 输入范围为{2, 34, 35, 36}，分别对应输出y的数据类型为{2: INT8,
 * 34: HIFLOAT8, 35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}
 * @param [in] y: GeluQuant计算的出参。npu device侧的aclTensor，
 * 数据类型支持int8, hifloat8, float8_e4m3fn, float8_e5m2, 数据格式支持ND，不支持非连续的Tensor。
 * @param [in] outScaleOptional: GeluQuant计算的出参。npu device侧的aclTensor，
 * 数据类型支持float32, 数据格式支持ND，支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeluQuantGetWorkspaceSize(
    const aclTensor* self, const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional,
    const char* approximate, const char* quantMode, const char* roundMode, int64_t dstType, const aclTensor* y,
    const aclTensor* outScaleOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGeluQuant的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGeluQuant获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnGeluQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_GELU_QUANT_H_
