/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_CLA_GATE_QUANT_H_
#define OP_API_INC_ACLNN_CLA_GATE_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnClaGateQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：实现CLA两路head-wise gate加权合并与动态块量化的组合计算。先对Global/CLA分支与Local/SWA分支的
 * Attention输出分别施加Sigmoid门控并加权合并，再将合并结果reshape为[T, K]（K = N*D）进行基于块的动态量化，
 * 输出低精度的FP8/FP4量化数据和对应的E8M0缩放因子。dualAxisFlag为false（默认）时为单轴量化，仅在K方向
 * （[1,32] block，row-wise）输出一套量化数据；为true时为双轴量化，同时输出row-wise与col-wise两套量化数据。
 *
 * @param [in] globalAttn: Global/CLA分支Attention输出，shape为[T, N, D]，dtype为BF16/FP16。
 * @param [in] localAttn: Local/SWA分支Attention输出，shape与dtype同globalAttn。
 * @param [in] globalGateLogits: Global gate的Sigmoid前值，shape为[T, N]，dtype同globalAttn。
 * @param [in] localGateLogits: Local gate的Sigmoid前值，shape与dtype同globalGateLogits。
 * @param [in] roundMode: 量化数据转换的舍入模式，FP8输出（dstType为35/36）仅支持"rint"，
 *            FP4输出（dstType为40/41）支持"rint"/"floor"/"round"，传入空指针时默认"rint"。
 * @param [in] scaleAlg: 缩放因子的计算方法，0为OCP实现，1为cuBLAS向上取整算法（仅FP8），FP4输出仅支持0。
 * @param [in] dstType: 量化数据的数据类型，取值{35, 36, 40, 41}，分别对应FLOAT8_E5M2、FLOAT8_E4M3FN、
 *            FLOAT4_E2M1、FLOAT4_E1M2；dstType为40/41时K = N*D必须可被4整除。
 * @param [in] inputAttnLayout: 输入globalAttn/localAttn的排布格式，当前仅支持"TND"，传入空指针时默认"TND"。
 * @param [in] dualAxisFlag: 是否双轴量化，false时仅输出row-wise结果，colDataOut/colScaleOut必须传空指针。
 * @param [out] rowDataOut: Row-wise量化数据，shape为[T, N*D]，dtype由dstType决定。
 * @param [out] rowScaleOut: Row-wise的E8M0缩放因子，shape为[T, ceil(N*D/64), 2]，dtype为FLOAT8_E8M0。
 * @param [out] colDataOut: Col-wise量化数据，shape为[T, N*D]，dualAxisFlag为false时传空指针。
 * @param [out] colScaleOut: Col-wise的E8M0缩放因子，shape为[ceil(T/64), N*D, 2]，dualAxisFlag为false时传空指针。
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnClaGateQuantGetWorkspaceSize(
    const aclTensor* globalAttn, const aclTensor* localAttn, const aclTensor* globalGateLogits,
    const aclTensor* localGateLogits, const char* roundMode, int64_t scaleAlg, int64_t dstType,
    const char* inputAttnLayout, bool dualAxisFlag, const aclTensor* rowDataOut, const aclTensor* rowScaleOut,
    const aclTensor* colDataOut, const aclTensor* colScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnClaGateQuant的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在Device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnClaGateQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CLA_GATE_QUANT_H_
