/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CHUNKLOSSBACKWARD_H_
#define OP_API_INC_CHUNKLOSSBACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnChunkLossBackward的第一段接口，根据计算流程计算workspace大小并生成执行器。
 * @domain aclnn_ops_train
 * 算子功能：ChunkLoss的反向传播。
 *
 * @param [in] gradOutput(const aclTensor*): 损失的上游梯度。数据类型支持FLOAT，数据格式支持ND，shape为[1]，
 * 支持非连续Tensor。
 * @param [in] hidden(const aclTensor*): 隐藏状态。数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，shape为[N,H]，
 * 支持非连续Tensor。
 * @param [in] weight(const aclTensor*): 投影权重。数据类型与hidden一致，数据格式支持ND，shape为[V,H]，
 * 支持非连续Tensor。
 * @param [in] labels(const aclTensor*): 每个token的目标类别索引。数据类型支持INT64，数据格式支持ND，shape为[N]，
 * 支持非连续Tensor。非ignoreIndex元素的取值范围为[0,V)。
 * @param [in] softmaxMax(const aclTensor*): 正向保存的逐token词表得分最大值。数据类型支持FLOAT，数据格式支持ND，
 * shape为[N]，支持非连续Tensor。
 * @param [in] softmaxSum(const aclTensor*): 正向保存的逐token稳定化指数和。数据类型支持FLOAT，数据格式支持ND，
 * shape为[N]，支持非连续Tensor，元素须为有限正值。
 * @param [in] alpha(double): 损失归一化系数。须为有限正数，转换为FLOAT后仍须大于0。
 * @param [in] ignoreIndex(int64_t): 不参与权重梯度计算的标签值。
 * @param [out] gradWeightOut(aclTensor*): 权重梯度。数据类型与weight一致，数据格式支持ND，shape为[V,H]，
 * 支持非连续Tensor。
 * @param [out] workspaceSize(uint64_t*): 返回用户需要在NPU Device侧申请的workspace大小。
 * @param [out] executor(aclOpExecutor**): 返回算子执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnChunkLossBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* hidden,
                                                             const aclTensor* weight, const aclTensor* labels,
                                                             const aclTensor* softmaxMax, const aclTensor* softmaxSum,
                                                             double alpha, int64_t ignoreIndex,
                                                             aclTensor* gradWeightOut, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);

/**
 * @brief aclnnChunkLossBackward的第二段接口，用于执行计算。
 * @param [in] workspace(void*): 在NPU Device侧申请的workspace内存地址。
 * @param [in] workspaceSize(uint64_t): 在NPU Device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor(aclOpExecutor*): 算子执行器，包含算子计算流程。
 * @param [in] stream(aclrtStream): ACL Stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnChunkLossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_CHUNKLOSSBACKWARD_H_
