/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_gru.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_GRU_H_
#define OP_API_INC_LEVEL2_ACLNN_GRU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGRU的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能：实现门控循环单元（Gated Recurrent Unit, GRU）计算。
 * @param [in] input: 输入的序列数据，数据类型支持：FLOAT16、FLOAT32，format支持：ND。
 *                   定长模式（batchSizes为nullptr）：shape为(T, B, I)或batchFirst=true时为(B, T, I)。
 *                   不定长模式（batchSizes非nullptr）：shape为(sum(batch_size), I)。
 * @param [in] params: 权重和偏置参数列表，数据类型与input保持一致，format支持：ND。
 *                     列表长度 = 2 * bScale * dScale * numLayers，其中 bScale = hasBias ? 2 : 1，dScale = bidirection ?
 * 2 : 1。 排列顺序：先层后方向，每组内依次为 W_ih、W_hh、b_ih（有偏置时）、b_hh（有偏置时）。 W_ih 形状为 [3H,
 * I]（首层）或 [3H, D*H]（非首层），W_hh 形状为 [3H, H]，偏置形状为 [3H]。
 * @param [in] hx: 初始隐状态，对应公式中的 h_0。可选参数，传入空指针表示初始隐状态为零向量。
 *                 数据类型与input保持一致，shape为(L*D, B, H)。
 * @param [in] batchSizes: 不定长序列的batch大小数组，对应PackedSequence模式。可选参数，传入空指针表示使用定长模式。
 *                         数据类型支持：INT64，shape为(T)。
 *                         传入非空指针时，进入PackedSequence模式，input形状必须为2D (sum(batch_size), I)。
 * @param [in] hasBias: 是否使用偏置。hasBias为true时params列表包含偏置参数，否则不包含。
 * @param [in] numLayers: GRU层数，必须为正整数。
 * @param [in] dropout: dropout概率，当前版本未使用，保留参数。
 * @param [in] train: 是否训练模式。train为true时输出各门控中间结果（rOut、zOut、nOut、hnOut、hOut），否则传入空指针。
 * @param [in] bidirection: 是否双向GRU。bidirection为true时，每层包含正向和反向两个方向，D=2，否则D=1。
 * @param [in] batchFirst: 是否batch优先。batchFirst为true时，input和output的shape为(B, T, *)维度，否则为(T, B, *)维度。
 *                          仅在定长模式（batchSizes为nullptr）下生效。
 * @param [out] output: 输出的隐状态序列，数据类型与input保持一致，format支持：ND。
 *                      定长模式：shape为(T, B, D*H)或batchFirst=true时为(B, T, D*H)。
 *                      不定长模式：shape为(sum(batch_size), D*H)。
 * @param [out] hy: 最终隐状态，数据类型与input保持一致，shape为(L*D, B, H)。
 * @param [out] rOut: 训练模式下输出重置门中间结果，数据类型与input保持一致，shape与output一致。
 *                    非训练模式下传入空指针。
 * @param [out] zOut: 训练模式下输出更新门中间结果，数据类型与input保持一致，shape与output一致。
 *                    非训练模式下传入空指针。
 * @param [out] nOut: 训练模式下输出新门中间结果，数据类型与input保持一致，shape与output一致。
 *                    非训练模式下传入空指针。
 * @param [out] hnOut: 训练模式下输出隐状态与新门乘积中间结果，数据类型与input保持一致，shape与output一致。
 *                      非训练模式下传入空指针。
 * @param [out] hOut: 训练模式下输出隐状态中间结果，数据类型与input保持一致，shape与output一致。
 *                     非训练模式下传入空指针。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnGRUGetWorkspaceSize(const aclTensor* input, const aclTensorList* params, const aclTensor* hx,
                                               const aclTensor* batchSizes, bool hasBias, int64_t numLayers,
                                               double dropout, bool train, bool bidirection, bool batchFirst,
                                               aclTensor* output, aclTensor* hy, aclTensorList* rOut,
                                               aclTensorList* zOut, aclTensorList* nOut, aclTensorList* hnOut,
                                               aclTensorList* hOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGRU的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGRUGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnGRU(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_GRU_H_