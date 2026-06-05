/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_RMS_NORM_GRAD_QUANT_H_
#define OP_API_INC_LEVEL2_RMS_NORM_GRAD_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRmsNormGradQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：RmsNormGrad与Quant融合算子，在反向传播过程中计算输入张量的梯度，
 * 并对dx结果进行量化，减少搬入搬出操作。
 *
 * @param [in] dy: 反向传回的梯度，数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND。
 * @param [in] x: 正向算子的输入，数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND。
 * @param [in] rstd: 正向算子的中间计算结果Rms(x)，数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] gamma: 正向算子的缩放因子，数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND。
 * @param [in] scalesX: 量化缩放因子，数据类型支持FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND。
 * @param [in] offsetXOptional: 可选参数，量化零点，数据类型支持INT32，数据格式支持ND。
 * @param [in] quantMode: 量化模式，字符串类型，当前支持"static"。
 * @param [in] divMode: bool类型，必选参数，决定量化公式是否使用除法。
 * @param [out] dxOut: 输入x的梯度，数据类型支持HIFLOAT8、INT8，数据格式支持ND。
 * @param [out] dgammaOut: gamma的梯度，数据类型支持FLOAT32，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRmsNormGradQuantGetWorkspaceSize(
    const aclTensor* dy, const aclTensor* x, const aclTensor* rstd, const aclTensor* gamma,
    const aclTensor* scalesX, const aclTensor* offsetXOptional,
    char* quantMode, bool divMode,
    aclTensor* dxOut, aclTensor* dgammaOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRmsNormGradQuant的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRmsNormGradQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRmsNormGradQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_RMS_NORM_GRAD_QUANT_H_
