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
 * \file aclnn_gru_backward.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_GRU_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_GRU_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGRUBackward第一段接口，根据计算流程计算workspace大小。
 * @domain aclnn_ops_train
 *
 * @param input [in] GRU输入序列 x，shape 见下：
 *   - 若 batchSizes 有效: [timeStep * batchSize, inputSize]
 *   - 若 batchSizes 为空: [timeStep, batchSize, inputSize] 或 [batchSize, timeStep, inputSize]
 * @param params [in] 每层权重和偏置 TensorList。
 *   bidirection=True 时 D=2，否则 D=1；hasBias=True 时 B=2，否则 B=1。
 *   列表长度 = D * B * numLayers。
 *   hasBias+bidirection 均为 True 时排布为:
 *     [w_ih_0, w_hh_0, b_ih_0, b_hh_0, w_ih_rev_0, w_hh_rev_0, b_ih_rev_0, b_hh_rev_0, ...]
 *   w_ih: [3*hiddenSize, cur_inputSize]
 *   w_hh: [3*hiddenSize, hiddenSize]
 *   b_ih/b_hh: [3*hiddenSize]
 * @param hx [in] 每层初始hidden状态，沿第0维按先方向后逐层堆叠的单个 tensor。
 *   shape: [D * numLayers, batchSize, hiddenSize]。
 *   内部按层/方向 Slice 切分 (与 aclnnGRU 前向 hx 一致)。
 * @param dy [in] 最后一层输出 hidden 的梯度。
 *   双向时数据沿最后一维按前后向排布。
 *   shape: [timeStep, batchSize, hiddenSize * D] 或 [batchSize, timeStep, hiddenSize * D]
 * @param dh [in] T时刻从下一个时间步传来的梯度。
 *   多层双向时沿第0维按先双向后逐层排布。
 *   shape: [numLayers * D, batchSize, hiddenSize]
 * @param r [in] 每步重置门激活值 TensorList，长度 D * numLayers。
 *   每个 tensor shape: [timeStep, batchSize, hiddenSize]
 * @param z [in] 每步更新门激活值 TensorList，长度 D * numLayers。
 *   每个 tensor shape: [timeStep, batchSize, hiddenSize]
 * @param n [in] 每步候选隐藏状态激活值 TensorList，长度 D * numLayers。
 *   每个 tensor shape: [timeStep, batchSize, hiddenSize]
 * @param hn [in] 每步候选隐藏状态中间值 TensorList，长度 D * numLayers。
 *   对应公式中的 W_{hn}*h_{t-1} + b_{hn}。
 *   每个 tensor shape: [timeStep, batchSize, hiddenSize]
 * @param h [in] 每步隐藏状态 TensorList，长度 D * numLayers。
 *   每个 tensor shape: [timeStep, batchSize, hiddenSize]
 * @param batchSizes [in] 变长序列各时刻有效 batch 数，可选。
 *   不支持时传 nullptr。shape: [timeStep]
 * @param hasBias [in] 是否有偏置 b
 * @param numLayers [in] GRU 层数，> 0
 * @param bidirection [in] 是否双向
 * @param batchFirst [in] input/dy/dxOut 的 batch 是否在第一维
 * @param dxOut [out] 输入 input 上的梯度，shape 与 input 一致
 * @param dhPrevOut [out] 初始 hidden 的梯度。
 *   多层双向时沿第0维按先双向后逐层排布。
 *   shape: [D * numLayers, batchSize, hiddenSize]
 * @param dparamsOut [out] 权重和偏置梯度 TensorList，排布与 params 一致
 */
ACLNN_API aclnnStatus aclnnGRUBackwardGetWorkspaceSize(
    const aclTensor* input, const aclTensorList* params, const aclTensor* hx, const aclTensor* dy, const aclTensor* dh,
    const aclTensorList* r, const aclTensorList* z, const aclTensorList* n, const aclTensorList* hn,
    const aclTensorList* h, const aclTensor* batchSizes, bool hasBias, int64_t numLayers, bool bidirection,
    bool batchFirst, aclTensor* dxOut, aclTensor* dhPrevOut, aclTensorList* dparamsOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnGRUBackward第二段接口，执行计算。
 */
ACLNN_API aclnnStatus aclnnGRUBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_GRU_BACKWARD_H_
