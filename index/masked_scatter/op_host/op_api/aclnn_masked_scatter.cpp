/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file aclnn_masked_scatter.cpp
 * \brief
 */
#include "aclnn_masked_scatter.h"
#include "masked_scatter.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* MaskedScatter 算子的完整计算流程如下:
 * selfRef                               mask                    source
 *   |                                  |                          /
 *   \                                  /                         /
 * Contiguous(workspace_0)    Contiguous(workspace_1)     Contiguous(workspace_2)
 *      \                             /                       /
 *          \                 Cast(workspace_3)             /
 *             \                 /                        /
 *                      MaskedScatter(workspace_4)
 *                                 |
 *                             ViewCopy
 *                                  |
 *                                result
 */

constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_SELFREF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_SELFREF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static bool CheckNotNull(const aclTensor* selfRef, const aclTensor* mask, const aclTensor* source)
{
    OP_CHECK_NULL(selfRef, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(source, return false);

    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_SELFREF_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_SELFREF_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* selfRef, const aclTensor* mask, const aclTensor* source)
{
    auto supportList = GetDtypeSupportList();
    // 检查selfRef的数据类型是否在masked_scatter算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, supportList, return false);
    // 检查mask的数据类型是否在maskedScatter算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(mask, MASK_DTYPE_SUPPORT_LIST, return false);
    // 检查selfRef和source的数据类型是否相同
    if (selfRef->GetDataType() != source->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "source dtype %s should be same with selfRef [%s].",
            op::ToString(source->GetDataType()).GetString(), op::ToString(selfRef->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool isMaskCanExpandToSelfRef(const aclTensor* selfRef, const aclTensor* mask, op::Shape* broadcastShape)
{
    op::Shape selfRefShape = selfRef->GetViewShape();
    size_t selfRefShapeLen = selfRefShape.GetDimNum();
    for (size_t i = 0; i < selfRefShapeLen; i++) {
        if (broadcastShape->GetDim(i) != selfRefShape.GetDim(i)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "target selfRef sizes %s ,source mask sizes %s",
                op::ToString(selfRef->GetViewShape()).GetString(), op::ToString(mask->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* selfRef, const aclTensor* mask, const aclTensor* source)
{
    OP_CHECK_MAX_DIM(selfRef, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(mask, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(source, MAX_DIM_LEN, return false);

    Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(selfRef, mask, broadcastShape, return false);

    if (!isMaskCanExpandToSelfRef(selfRef, mask, &broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "the number of dimensions in selfRef tensor must be greater or equal to the mask tensor");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* selfRef, const aclTensor* mask, const aclTensor* source)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(selfRef, mask, source), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(selfRef, mask, source), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入形状是否满足
    CHECK_RET(CheckShape(selfRef, mask, source), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* mask, const aclTensor* source, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceMaskedScatter, DFX_IN(selfRef, mask, source), DFX_OUT(selfRef));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(selfRef, mask, source);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (selfRef->IsEmpty() || mask->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor
    auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入mask转换成连续的tensor
    auto maskContiguous = l0op::Contiguous(mask, uniqueExecutor.get());
    CHECK_RET(maskContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 当输入source不是空tensor时，将其换成连续的tensor
    const aclTensor* sourceContiguous = nullptr;
    if (source->IsEmpty()) {
        sourceContiguous = source;
    } else {
        sourceContiguous = l0op::Contiguous(source, uniqueExecutor.get());
        CHECK_RET(sourceContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将输入mask的数据类型转换成bool数据类型
    auto maskBool = l0op::Cast(maskContiguous, DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(maskBool != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用MaskedScatter算子
    auto maskedScatterOpOut = l0op::MaskedScatter(selfRefContiguous, maskBool, sourceContiguous, uniqueExecutor.get());
    CHECK_RET(maskedScatterOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(maskedScatterOpOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceMaskedScatter(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceMaskedScatter);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
