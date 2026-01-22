/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_REPEAT_INTERLEAVE_H_
#define OP_API_INC_LEVEL2_ACLNN_REPEAT_INTERLEAVE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

// 无dim, tensor repeats
/**
 * @brief aclnnRepeatInterleave的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、BFLOAT16类型。
 * 支持空tensor, 支持非连续的tensor。数据格式支持ND。
 * @param [in] repeats: npu device侧的aclTensor。数据类型支持INT64。repeats只能为0D / 1D Tensor。如果为1D Tensor，那么
 * repeats的size必须为1或self的元素个数。支持空tensor，支持非连续的tensor。数据格式支持ND。
 * @param [in] outputSize: 进行重复后的tensor最终大小。数据类型为INT64。当repeats中只有一个元素时，outputSize =
 * self的元素 个数 * repeats的值。当repeats中有多个值时，outputSize = repeats的值之和。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32类型。数据
 * 类型需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveGetWorkspaceSize(
    const aclTensor* self, const aclTensor* repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: aclnnRepeatInterleave的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRepeatInterleaveGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRepeatInterleave(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

// 有dim, tensor repeats
/**
 * @brief aclnnRepeatInterleaveWithDim的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、BFLOAT16类型。
 * 支持空tensor， 支持非连续的tensor。数据格式支持ND。
 * @param [in] repeats: npu device侧的aclTensor。数据类型支持INT64。repeats只能为0D / 1D tensor。如果为1D tensor，那么
 * repeats的size必须为1或self的dim维度的size。支持空tensor，支持非连续的tensor。数据格式支持ND。
 * @param [in] dim: 进行重复的维度，数据类型为INT64。范围为[-self的维度数量, self的维度数量-1]。
 * @param [in] outputSize:
 * dim维度再进行重复后的最终大小。数据类型为INT64。如果repeats中有多个值，则outputSize值必须为repeats
 * 的求和。如果repeats只有一个元素时，则outputSize值必须为repeats * self的dim维度size。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32类型。数据
 * 类型需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(
    const aclTensor* self, const aclTensor* repeats, int64_t dim, int64_t outputSize, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnRepeatInterleaveWithDim的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRepeatInterleaveWithDimGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRepeatInterleaveWithDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

// 无dim, int repeats
/**
 * @brief aclnnRepeatInterleaveInt的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、BFLOAT16类型。
 * 支持空tensor, 支持非连续的tensor。数据格式支持ND。
 * @param [in] repeats: 重复的次数。数据类型为INT64。repeats的值必须为自然数。
 * @param [in] outputSize: 进行重复后的tensor最终大小。数据类型为INT64。outputSize必须等于self的元素个数 * repeats的值。
 * @param [in] out: npu device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32类型。
 * 数据类型需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveIntGetWorkspaceSize(
    const aclTensor* self, int64_t repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: aclnnRepeatInterleaveInt的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRepeatInterleaveIntGetWorkspaceSize
 * 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRepeatInterleaveInt(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

// 有dim, int repeats
/**
 * @brief aclnnRepeatInterleaveIntWithDim的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、BFLOAT16类型。
 * 支持空tensor， 支持非连续的tensor。数据格式支持ND。
 * @param [in] repeats: 重复的次数。数据类型为INT64。repeats的值必须为自然数。
 * @param [in] dim: 进行重复的维度，数据类型为INT64。范围为[-self的维度数量, self的维度数量-1]。
 * @param [in] outputSize: dim维度再进行重复后的最终大小。数据类型为INT64。outputSize值必须为repeats *
 * self的dim维度size。
 * @param [in] out: npu device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32类型。
 * 数据类型需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(
    const aclTensor* self, int64_t repeats, int64_t dim, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: aclnnRepeatInterleaveIntWithDim的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口
 * aclnnRepeatInterleaveIntWithDimGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRepeatInterleaveIntWithDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

// repeat_interleave.Tensor
/**
 * @brief aclnnRepeatInterleaveTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] repeats: npu device侧的aclTensor。数据类型支持INT32、INT64。repeats只能为1D Tensor
 * (包括shape=[0,]的场景)。 支持非连续的tensor。数据格式支持ND。
 * @param [in] outputSize: 进行重复后的tensor最终大小。数据类型为INT64。outputSize必须等于repeats的值之和。
 * @param [in] out: npu device侧的aclTensor，数据类型支持INT64类型。数据类型需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveTensorGetWorkspaceSize(
    const aclTensor* repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnRepeatInterleaveTensor的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成repeatinterleave操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRepeatInterleaveTensorGetWorkspaceSize
 * 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnRepeatInterleaveTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_REPEAT_INTERLEAVE_H_