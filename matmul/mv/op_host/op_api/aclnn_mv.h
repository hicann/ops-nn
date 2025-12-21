/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_MV_H_
#define OP_API_INC_LEVEL2_ACLNN_MV_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 计算矩阵input与向量vec的乘积
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128类型。支持
 * [非连续的Tensor](#)，shape为n*m的二维张量，数据格式支持ND([参考](#))。
 * @param [in] vec: npu device侧的aclTensor。数据类型支持FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128类型，且数据类型与
 * self保持一致。支持[非连续的Tensor](#)，shape为长度为m的一维张量，数据格式支持ND([参考](#))。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT、DOUBLE、COMPLEX64、COMPLEX128类型，且数据类型与self
 * 保持一致。支持[非连续的Tensor](#)，shape为长度为n的一维张量，数据格式支持ND([参考](#))。
 * @param [in] cubeMathType(INT8,
 * 计算输入)：INT8类型的枚举值，用于判断Cube单元应该使用哪种计算逻辑进行运算，0：KEEP_DTYPE, 保
 * 持输入数据类型进行计算，若输入是FP32，Ascend910系列芯片暂不支持，该值为0时会报错。1：ALLOW_FP32_DOWN_PRECISION，允许转换输入
 * 数据类型降低精度计算，若输入为FP32，在Ascend910系列芯片上转换为FP16进行计算，在Ascend910B系列芯片上转换为HF16进行计算。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMvGetWorkspaceSize(
    const aclTensor* self, const aclTensor* vec, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: aclnnMv的第二段接口，用于执行计算
 *
 * 算子功能： 计算矩阵input与向量vec的乘积
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMvGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MV_H_