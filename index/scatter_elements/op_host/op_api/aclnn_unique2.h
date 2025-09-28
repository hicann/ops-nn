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
 * \file aclnn_unique2.h
 * \brief
 */
#ifndef OP_API_INC_UNIQUE2_H_
#define OP_API_INC_UNIQUE2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUnique2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：返回输入张量中的独特元素
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B ---> F([UniqueWithCountsAndSorting])
 *     C[(sorted)] --->F
 *     D[(returnInverse)] --->F
 *     E[(returnCounts)] --->F
 *     F --> G([valueOut])
 *     F --> H([inverseOut])
 *     F --> I([countsOut])
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持BOOL, FLOAT, FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16,
 * INT32, UINT32, UINT64, INT64，支持非连续的Tensor，数据格式支持ND。
 * @param [in] sorted: 可选参数，默认False，表示是否对 valueOut 按升序进行排序。
 * @param [in] returnInverse: 可选参数，默认False，表示是否返回输入数据中各个元素在 valueOut 中的下标。
 * @param [in] returnCounts: 可选参数，默认False，表示是否返回 valueOut 中每个独特元素在原输入Tensor中的数目。
 * @param [in] valueOut: npu device侧的aclTensor, 第一个输出张量，输入张量中的唯一元素，数据类型支持BOOL, FLOAT,
 * FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64，数据格式支持ND。
 * @param [in] inverseOut: npu
 * device侧的aclTensor，第二个输出张量，当returnInversie为True时有意义，返回self中各元素在valueOut中出现的位置下
 *                      标，数据类型支持INT64，shape与self保持一致
 * @param [in] countsOut: npu
 * device侧的aclTensor，第三个输出张量，当returnCounts为True时有意义，返回valueOut中各元素在self中出现的次数，数据
 *                     类型支持INT64，shape与valueOut保持一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUnique2GetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse,
                                                   bool returnCounts, aclTensor* valueOut, aclTensor* inverseOut,
                                                   aclTensor* countsOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief aclnnUnique2的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnUnique2GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnUnique2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNIQUE2_H_