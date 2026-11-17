/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_DEQUANT_SITU_QUANT_H_
#define OP_API_INC_DEQUANT_SITU_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDequantSituQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：在Situ激活函数前后添加dequant和quant操作，实现x的DequantSituQuant计算。
 * 计算流程为：反量化 -> Situ激活 -> 量化。
 *
 * @param [in] x: npu device侧的aclTensor，数据类型支持INT8、INT32、BFLOAT16、FLOAT16，数据格式支持ND，
 * 不支持空Tensor。x为INT8时维度为2-8维，为INT32/BFLOAT16/FLOAT16时维度为2维，最后一维需要是2的倍数。
 * @param [in] weightScaleOptional: npu device侧的aclTensor，反量化的weight scale，数据类型FLOAT，数据格式ND，
 * shape为(2H,)或(1,)。INT8和INT32必选，BFLOAT16/FLOAT16不使用。
 * @param [in] activationScaleOptional: npu device侧的aclTensor，反量化的activation scale，数据类型FLOAT，
 * 数据格式ND，shape为(1,)。仅INT32必选，其他数据类型不使用。
 * @param [in] biasOptional: npu device侧的aclTensor，反量化的bias，数据类型FLOAT，数据格式ND，
 * shape为(2H,)或(1,)。INT8和INT32可选，BFLOAT16/FLOAT16不使用。
 * @param [in] quantScaleOptional: npu device侧的aclTensor，量化的scale，数据类型FLOAT，数据格式ND，
 * shape为(H,)或(1,)。quantTypeOptional为static时必选；为dynamic时可选，作为smooth scale使用。仅INT8使用。
 * @param [in] quantOffsetOptional: npu device侧的aclTensor，量化的offset，数据类型FLOAT，数据格式ND，
 * shape同quantScaleOptional。仅static量化使用且为可选输入，缺省时按0处理；dynamic时不使用，须为空。
 * @param [in] groupIndexOptional: npu device侧的aclTensor，MoE分组的group index，数据类型INT64，数据格式ND，
 * 1维tensor。INT32/BFLOAT16/FLOAT16可选，INT8不使用。
 * @param [in] beta: host侧的double类型，Situ激活的beta参数，不能为0，默认4.0。
 * @param [in] linearBeta: host侧的double类型，Situ激活的linear_beta参数，当值<=0时不启用，默认25.0。
 * @param [in] activateLeft: host侧的bool类型，是否对输入的左半部分做Situ激活，默认true。
 * @param [in] quantTypeOptional: host侧的char*类型，量化模式，仅支持{"static", "dynamic"}，传入nullptr时按默认值
 * "dynamic"处理。
 * @param [out] yOut: npu device侧的aclTensor，量化后的输出，数据类型INT8，数据格式ND，不支持空Tensor。
 * INT8时shape为x.shape[:-1]+[H]；其他数据类型时shape为[M, H]。其中H=x.shape[-1]/2。
 * @param [out] yScaleOut: npu device侧的aclTensor，动态量化的scale，数据类型FLOAT，数据格式ND，不支持空Tensor。
 * INT8时shape为x.shape[:-1]；其他数据类型时shape为[M]。quantTypeOptional为static时输出无意义值。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnDequantSituQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weightScaleOptional, const aclTensor* activationScaleOptional,
    const aclTensor* biasOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* groupIndexOptional, double beta, double linearBeta, bool activateLeft, char* quantTypeOptional,
    const aclTensor* yOut, const aclTensor* yScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDequantSituQuant的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnDequantSituQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_DEQUANT_SITU_QUANT_H_
