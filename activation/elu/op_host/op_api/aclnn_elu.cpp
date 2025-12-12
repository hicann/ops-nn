/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "aclnn_elu.h"
#include "elu.h"
#include "level0/threshold.h"
#include "level0/muls.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_BF16};

static inline bool CheckNotNull(
    const aclTensor* self, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    const aclTensor* out)
{
    // self、alpha、scale、inputScale、out不能为空指针
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(inputScale, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(
    const aclTensor* self, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    const aclTensor* out)
{
    // 检查数据类型是否在Elu算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    // 检查scale的数据类型能否转换为FLOAT
    if (!CanCast(scale->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "scale dtype %s can not cast to float32.",
            ToString(scale->GetDataType()).GetString());
        return false;
    }

    // 检查alpha的数据类型能否转换为FLOAT
    if (!CanCast(alpha->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "alpha dtype %s can not cast to float32.",
            ToString(alpha->GetDataType()).GetString());
        return false;
    }

    // 检查inputScale的数据类型能否转换为FLOAT
    if (!CanCast(inputScale->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "inputScale dtype %s can not cast to float32.",
            ToString(inputScale->GetDataType()).GetString());
        return false;
    }

    // 检查self的数据类型能否转换为out的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);

    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, alpha, scale, inputScale, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, alpha, scale, inputScale, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(
    const aclTensor* self, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, alpha, scale, inputScale, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果self是空tensor，则out也是空tensor，直接返回
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的Tensor
    auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Elu算子kernel
    const aclTensor* eluOut;
    if (std::isinf(alpha->ToFloat()) || std::isnan(alpha->ToFloat()) || std::isinf(inputScale->ToFloat()) || std::isnan(inputScale->ToFloat())) {
        aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
        aclGetDataType(contiguousSelf, &dataType);
        auto oriDataType = dataType;
        if (oriDataType == aclDataType::ACL_DOUBLE || oriDataType == aclDataType::ACL_FLOAT16 || oriDataType == aclDataType::ACL_BF16) {
            dataType = aclDataType::ACL_FLOAT;
        }
        float zeroValue = 0.0f;
        const aclScalar* zeroScalar = aclCreateScalar(&zeroValue, dataType);

        float negAlpha = (alpha->ToFloat()) * float(-1.0);
        const aclScalar* negAlphaScalar = aclCreateScalar(&negAlpha, dataType);
        const aclTensor* thresholdOut;
        if (oriDataType == aclDataType::ACL_DOUBLE || oriDataType == aclDataType::ACL_FLOAT16 ||
            oriDataType == aclDataType::ACL_BF16) {
            auto castSelf = l0op::Cast(contiguousSelf, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(castSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

            thresholdOut = l0op::Threshold(castSelf, zeroScalar, negAlphaScalar, uniqueExecutor.get());
        } else {
            thresholdOut = l0op::Threshold(contiguousSelf, zeroScalar, negAlphaScalar, uniqueExecutor.get());
        }
        CHECK_RET(thresholdOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        eluOut = l0op::Muls(thresholdOut, scale->ToFloat(), uniqueExecutor.get());

        // 释放资源
        aclDestroyScalar(zeroScalar);
        aclDestroyScalar(negAlphaScalar);
    } else {
        eluOut = l0op::Elu(contiguousSelf, alpha->ToFloat(), scale->ToFloat(), inputScale->ToFloat(), uniqueExecutor.get());
    }

    CHECK_RET(eluOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(eluOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEluGetWorkspaceSize(
    const aclTensor* self, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnElu, DFX_IN(self, alpha, scale, inputScale), DFX_OUT(out));
    return GetWorkspaceSizeCommon(self, alpha, scale, inputScale, out, workspaceSize, executor);
}

aclnnStatus aclnnElu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnElu);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceEluGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceElu, DFX_IN(selfRef, alpha, scale, inputScale), DFX_OUT(selfRef));
    return GetWorkspaceSizeCommon(selfRef, alpha, scale, inputScale, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceElu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceElu);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
