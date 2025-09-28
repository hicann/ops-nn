/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_fake_quant_per_channel_affine_cachemask.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_CHANNEL_AFFINE_CACHEMASK_H_
#define OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_CHANNEL_AFFINE_CACHEMASK_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFakeQuantPerChannelAffineCachemask的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：
 *   对于输入数据self，使用scale和zero_point对输入self在指定轴axis上进行伪量化处理，并根据quant_min和quant_max对
 * 伪量化输出进行值域更新，最终返回结果out及对应位置掩码mask。
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A1[(self)] -->B1(l0op::Contiguous)-->C1(l0op::Transpose)-->D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A2[(scale)] -->B2(l0op::Contiguous)-->D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A3[(zeroPoint)]-->B3(l0op::Contiguous)-->D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A4((quantMin)) --> D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A5((quantMax)) --> D(l0op::FakeQuantPerChannelAffineCachemask)
 *   A6((axis))-->C1(l0op::Transpose)
 *   D(l0op::FakeQuantPerChannelAffineCachemask)-->C2(l0op::Transpose)-->E1(l0op::ViewCopy)-->F1[(out)]
 *   D(l0op::FakeQuantPerChannelAffineCachemask)-->C3(l0op::Transpose)-->EF2(l0op::ViewCopy)-->F2[(mask)]
 * ```
 * @param [in] self:
 * Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32。支持非连续的Tensor，[数据格式](../../../docs/context/数据格式.md)支持ND。
 * @param [in] scale: Device侧的aclTensor，表示输入伪量化的缩放系数。数据类型支持FLOAT16、FLOAT32，size大小为1。
 * @param [in] zeroPoint: Device侧的aclTensor，表示输入伪量化的零基准参数。数据类型支持INT32，size大小为1。
 * @param [in] axis: Host侧的整型，表示计算维度，数据类型支持INT。
 * @param [in] quantMin: Host侧的整型，表示输入数据伪量化后的最小值，数据类型支持INT。
 * @param [in] quantMax: Host侧的整型，表示输入数据伪量化后的最大值，数据类型支持INT。
 * @param [out] out:
 * Device侧的aclTensor，数据类型支持LOAT16、FLOAT32，支持非连续Tensor，[数据格式](../../../docs/context/数据格式.md)支持ND。
 * @param [out] mask: Device侧的aclTensor，数据类型支持BOOL，支持非连续Tensor，[数据格式](../../../docs/context/数据格式.md)支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, int64_t axis, int64_t quantMin,
    int64_t quantMax, aclTensor* out, aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFakeQuantPerChannelAffineCachemask的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnFakeQuantPerChannelAffineCachemask获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_FAKE_QUANT_PER_CHANNEL_AFFINE_CACHEMASK_H_
