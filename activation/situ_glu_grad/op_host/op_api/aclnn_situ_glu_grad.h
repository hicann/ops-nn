/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SITU_GLU_GRAD_H_
#define OP_API_INC_LEVEL2_ACLNN_SITU_GLU_GRAD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSituGluGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成SiTU门控线性单元（SiTU Gated Linear Unit）激活函数的反向梯度计算。
 * 根据上游梯度gradY和前向输入x，计算输入x的梯度grad_x。
 *
 * @param [in] gradY: 上游梯度，shape在dim对应维度上为x的一半、其他维度与x一致，dtype FLOAT/FLOAT16/BFLOAT16。
 *                    支持空Tensor。
 * @param [in] x: 前向输入，shape在dim对应维度上为偶数，dtype同gradY。支持空Tensor。
 * @param [in] dim: 切分维度序号，取值范围[-x.dim(), x.dim()-1]。
 * @param [in] beta: SiTU门控部分的缩放系数，建议为非零值。
 * @param [in] linearBeta: up路径线性tanh的缩放系数，小于等于0时up路径不进行变换。
 * @param [in] activateLeft: 切分x时gate是否为前半部分。
 * @param [out] out: 输入梯度grad_x，shape与x一致，dtype同x。支持空Tensor与非连续Tensor。不支持空指针。
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSituGluGradGetWorkspaceSize(const aclTensor* gradY, const aclTensor* x, int64_t dim,
                                                       double beta, double linearBeta, bool activateLeft,
                                                       const aclTensor* out, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnSituGluGrad的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在Device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSituGluGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_SITU_GLU_GRAD_H_
