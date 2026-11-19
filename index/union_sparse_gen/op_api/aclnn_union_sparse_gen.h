/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_UNION_SPARSE_GEN_H
#define OP_API_INC_ACLNN_UNION_SPARSE_GEN_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUnionSparseGen的第一段接口，计算workspace大小并创建执行器。
 * @domain aclnn_ops_infer
 *
 * 将同一请求内的query块列表合并为升序并集，输出实际块数和逐token的query归属位掩码。
 * H为稀疏head数，T为Q存储行数，K为每行块容量，B为请求数，G为输出组数，L为最大逻辑块数。
 * 输出容量U=min(unionSize*(K+residualBlockMode),L)。所有Tensor均为npu device侧ND格式，支持非连续Tensor。
 * @param [in] sparseBlockIdx: INT32
 * Tensor，shape为[H,T,K]，每行有效前缀保存逻辑KV块编号；isInputIdxSorted=true时须升序。
 * @param [in] sparseBlockCount: INT32 Tensor，shape为[H,T]，给出各输入行的有效前缀长度。
 * @param [in] cuSqlenQOptional: 可为nullptr的INT64 Tensor，shape为[B+1]，表示Q存储前缀和。
 * 不传时使用T/B等宽槽位；若seqUsedQOptional也不传，则有效Q和槽宽均为maxQSeqlen。
 * @param [in] cuSqlenKvOptional: 可为nullptr的INT64 Tensor，shape为[B+1]，表示KV存储前缀和。
 * 不传且seqUsedKvOptional也不传时，有效KV为maxKvSeqlen。Q/KV独立选择长度来源。
 * @param [in] seqUsedQOptional: 可为nullptr的INT32 Tensor，shape为[B]，优先指定各请求有效Q长度；
 * 未传时使用Q前缀和差分或定长maxQ；传入时不得超过对应Q存储长度。
 * @param [in] seqUsedKvOptional: 可为nullptr的INT32 Tensor，shape为[B]，优先指定各请求有效KV长度；
 * 未传时使用KV前缀和差分或定长maxKV；传入且有KV前缀和时不得超过对应KV存储长度。
 * @param [in] blockShape: Q/KV块粒度的aclIntArray，支持[1,Y]，Y为4、8、16或128。
 * @param [in] unionSize: 同一请求内每组query行数，当前仅支持8，组不跨请求。
 * @param [in] maskType: 0表示非因果；1表示右下对齐因果，额外限制KV位置p<KV-Q+qLocal+1。
 * 两种模式均保留query归属关系和有效KV边界，residualBlockMode=0时不改变块并集与count。
 * @param [in] maxQSeqlen:
 * 有任意Q长度Tensor时忽略其任意int64值；否则必须非负，指定定长Q槽宽，B=ceil(T/maxQSeqlen)，最后一槽取剩余Q行。
 * @param [in] maxKvSeqlen:
 * 有任意KV长度Tensor时忽略其任意int64值；否则必须非负，指定定长KV，ceil(maxKvSeqlen/Y)<=INT32_MAX。
 * @param [in] layoutSparsePattern: 稀疏模式布局，当前仅支持4，即[H,T,K]。
 * @param [in] isInputIdxSorted: 有效输入前缀已升序排列的承诺，支持true或false。
 * @param [in] residualBlockMode:
 * 尾块处理方式，0仅处理显式块；1过滤非完整可见显式块并补入不完整可见尾块，maskType=0按完整KV长度补尾。
 * @param [in] totalqUnionNum: 形状提示；ACLNN直调以unionSparseIdx.shape[1]作为输出组数G。
 * @param [in] maxLogicalBlocks: 形状提示；ACLNN直调以unionSparseIdx.shape[2]作为输出容量U，须容纳补尾后的并集。
 * @param [in] unionSparseIdx: 预分配的INT32输出Tensor，shape为[H,G,U]，升序去重块列表，padding为-1。
 * @param [in] unionSparseCount: 预分配的INT32输出Tensor，shape为[H,G]，保存各组实际并集块数。
 * @param [in] exactMask: 预分配的INT32输出Tensor，shape为[H,G,U*Y]，低unionSize位分别对应组内query；
 * 未使用输出位置和越过有效KV长度的token为0。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace字节数。
 * @param [out] executor: 返回包含算子计算流程的执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUnionSparseGenGetWorkspaceSize(
    const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount, const aclTensor* cuSqlenQOptional,
    const aclTensor* cuSqlenKvOptional, const aclTensor* seqUsedQOptional, const aclTensor* seqUsedKvOptional,
    const aclIntArray* blockShape, int64_t unionSize, int64_t maskType, int64_t maxQSeqlen, int64_t maxKvSeqlen,
    int64_t layoutSparsePattern, bool isInputIdxSorted, int64_t residualBlockMode, int64_t totalqUnionNum,
    int64_t maxLogicalBlocks, aclTensor* unionSparseIdx, aclTensor* unionSparseCount, aclTensor* exactMask,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUnionSparseGen的第二段接口，执行块并集和逐token位掩码计算。
 * @param [in] workspace: 按第一段接口返回大小申请的npu device侧workspace内存起址。
 * @param [in] workspaceSize: workspace字节数，由aclnnUnionSparseGenGetWorkspaceSize获取。
 * @param [in] executor: 第一段接口返回的执行器，包含算子计算流程。
 * @param [in] stream: 执行计算的acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUnionSparseGen(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
