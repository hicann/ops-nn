/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_SITU_MX_QUANT_H_
#define OP_API_INC_SITU_MX_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSituMxQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：将Situ激活函数与动态MX（Microscaling）量化融合，对输入x进行Situ激活后，
 * 对激活结果做MX量化，输出量化后的结果y和scale（E8M0）。
 *
 * @param [in] x: npu device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，
 * shape支持1-8维，最后一维需要是2的倍数，不支持空Tensor，数据格式支持ND。
 * @param [in] beta: host侧的double类型，Situ激活的beta参数，必须大于0。默认1.0。
 * @param [in] linearBeta: host侧的double类型，Situ激活的linear_beta参数，
 * 当值≤0时不启用。默认0.0。
 * @param [in] activateLeft: host侧的bool类型，是否对输入的左半部分做Situ激活。默认false。
 * @param [in] axis: host侧的int64_t类型，量化轴，当前仅支持-1。默认-1。
 * @param [in] dstType: host侧的int64_t类型，输出y的数据类型，
 * 输入范围为{35, 36}，分别对应{35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。默认36。
 * @param [in] roundModeOptional: host侧的char*类型，量化舍入模式，
 * 支持"rint"，FP8输出仅支持"rint"，传入nullptr时按默认值"rint"处理。默认"rint"。
 * @param [out] yOut: npu device侧的aclTensor，量化后的输出，shape为x.shape[:-1]+[H]，
 * 其中H=x.shape[-1]/2，数据类型由dstType决定（FLOAT8_E4M3FN或FLOAT8_E5M2）。
 * @param [out] yScaleOut: npu device侧的aclTensor，MX量化的scale（E8M0格式），
 * shape为x.shape[:-1]+[ceil(H/64), 2]，数据类型FLOAT8_E8M0。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSituMxQuantGetWorkspaceSize(const aclTensor* x, double beta, double linearBeta,
                                                       bool activateLeft, int64_t axis, int64_t dstType,
                                                       char* roundModeOptional, const aclTensor* yOut,
                                                       const aclTensor* yScaleOut, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnSituMxQuant的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存地址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnSituMxQuantGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSituMxQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_SITU_MX_QUANT_H_
