/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SHRINK_H_
#define OP_API_INC_SHRINK_H_

#include "aclnn/aclnn_base.h"
#include"aclnn_util.h"

#ifdef __cplusplus
extern "C"{
#endif

/**
 * @brief aclnnShrink的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer、aclnn_ops_train
 * 功能描述：对输入张量进行非线性变换，根据输入值self与阈值lambd的关系，对输入通过偏移量bias进行缩放和偏移处理。
 * 计算公式：如下
 * $$
 * out=
 * \begin{cases}
 * x-bias, if x > lambd \\
 * x+bias, if x < -lambd \\
 * 0, otherwise \\
 * \end{cases}
 * $$
 * 参数描述：
 * @param [in]   self
 * 输入Tensor，数据类型支持FLOAT，FLOAT16。支持非连续Tensor，数据格式支持ND。
 * @param [in]   lambd
 * 输入Scalar，数据类型支持FLOAT。
 * @param [in]   bias
 * 输入Scalar，数据类型支持FLOAT。
 * @param [in]   out
 * 输出Tensor，数据类型支持FLOAT，FLOAT16。支持非连续Tensor，数据格式支持ND。
 * @param [out]  workspaceSize   返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor         返回op执行器，包含了算子计算流程。
 * @return       aclnnStatus      返回状态码
 */
ACLNN_API aclnnStatus aclnnShrinkGetWorkspaceSize(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnShrink的第二段接口，用于执行计算。
 * 功能描述：对输入张量进行非线性变换，根据输入值self与阈值lambd的关系，对输入通过偏移量bias进行缩放和偏移处理。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnShrinkGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnShrink(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif //OP_API_INC_SHRINK_H_