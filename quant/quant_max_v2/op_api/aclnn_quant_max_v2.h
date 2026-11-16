/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_QUANT_MAX_V2_H_
#define OP_API_INC_LEVEL2_ACLNN_QUANT_MAX_V2_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMaxV2GetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu
 * device侧的aclTensor，算子输入的Tensor，shape维度为2~8维，Device侧的aclTensor，数据类型支持FLOAT32、FLOAT16、BFLOAT16。
 * 支持空Tensor，支持非连续的Tensor，数据格式支持ND。
 * @param [in] xScaleOptional: npu
 * 算子输入的量化缩放系数（乘性，等于常规量化scale的倒数1/s），Device侧的aclTensor，数据类型支持FLOAT32。
 * 传nullptr表示动态量化，量化系数按本组amax现算；非nullptr表示静态量化。
 * shape第一维大小是分组数numGroups（Per-Tensor场景为1），传入时numGroups须与groupListOptional的第一维一致。
 * 不支持空Tensor，支持非连续的Tensor，数据格式支持ND。
 * @param [in] groupListOptional: npu
 * 算子输入的groupList，Device侧的aclTensor，数据类型支持INT64，支持非连续的Tensor，数据格式支持ND。
 * shape只支持一维，且维度大小等于numGroups，取值范围1~384。groupListOptional[g]表示前g+1组在x第0维上的累计结束位置（cumsum语义），
 * 边界要求单调不减、位于[0, x.shape[0]]，最后一个边界等于x.shape[0]，相邻边界相等表示空组；
 * 该数值约束由调用方保证，违反约束的结果未定义。传nullptr表示Per-Tensor场景（整个Tensor视为单组）。
 * 不支持空Tensor。
 * @param [in]
 * roundMode：输出yOut转换到目标类型时的舍入模式，Host侧的char*。支持"round"、"hybrid"、"rint"，
 * 且须与dstType匹配：DT_HIFLOAT8支持"round"、"hybrid"，DT_FLOAT8_E5M2和DT_FLOAT8_E4M3FN仅支持"rint"。
 * @param [in]
 * dstType：输出yOut的数据类型对应的枚举值，Host侧的int64_t。支持DT_HIFLOAT8(34)，DT_FLOAT8_E5M2(35)，DT_FLOAT8_E4M3FN(36)。
 * @param [in]
 * dstTypeMax：目标类型的最大有限值，Host侧的float，默认15.0，取值范围[0, 32768]。仅动态量化参与系数计算；
 * dstType为DT_FLOAT8_E5M2或DT_FLOAT8_E4M3FN时按类型最大值（57344/448）推导，传入值不参与计算。
 * @param [in] yOut: npu
 * 量化后的输出Tensor，shape和x保持一致，类型由dstType指定。支持非连续的Tensor，数据格式支持ND。
 * @param [in] amaxOut: npu
 * 量化前分组内x绝对值的最大值，必选输出，Device侧的aclTensor，shape第一维大小是numGroups（Per-Tensor场景为1），
 * 数据类型支持FLOAT32、FLOAT16、BFLOAT16，并且数据类型要和x保持一致。支持非连续的Tensor，数据格式支持ND。
 * @param [in] yScaleOutOptional: npu
 * 量化使用的y_scale（等于amax除以dstTypeMax，满足x ≈ yOut × yScale），Device侧的aclTensor，数据类型支持FLOAT32，
 * 支持非连续的Tensor（内部Contiguous计算后ViewCopy按原strides回写），数据格式支持ND。
 * 动态量化（xScaleOptional为nullptr）时必选，shape第一维大小是numGroups（Per-Tensor场景为1）；
 * 静态量化时必须传nullptr（不输出y_scale，非nullptr会被拒绝）。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnQuantMaxV2GetWorkspaceSize(const aclTensor* x, const aclTensor* xScaleOptional,
                                                      const aclTensor* groupListOptional, const char* roundMode,
                                                      int64_t dstType, float dstTypeMax, const aclTensor* yOut,
                                                      const aclTensor* amaxOut, const aclTensor* yScaleOutOptional,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantMaxV2的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnQuantMaxV2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnQuantMaxV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_QUANT_MAX_V2_H_
