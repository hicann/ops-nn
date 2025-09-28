/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_div_scalar_v2.h"
#include "foreach_div_scalar_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                    DataType::DT_FLOAT16,
                                                                                    DataType::DT_BF16};

static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                    DataType::DT_DOUBLE};

static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT16_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                    DataType::DT_DOUBLE};

static const std::initializer_list<DataType> EMPTY_LIST  = {};

static inline bool CheckNotNull(const aclTensorList* self, const aclScalar* scalar, const aclTensorList* out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(scalar, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckFormat(const aclTensorList* self, const aclTensorList* out) {
    for (uint64_t i = 0; i < self->Size(); i++) {
        // self格式不能是私有格式
        if (IsPrivateFormat((*self)[i]->GetStorageFormat()) || IsPrivateFormat((*out)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
            return false;
        }
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented",
            op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
    return EMPTY_LIST;
  }
}

static inline bool CheckDtypeValid(const aclTensorList* self, const aclScalar* scalar, const aclTensorList* out) {
    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented",
            op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return false;
    }

    if (self->Size() == 0) {
        return true;
    }

    // checkself input dtype, and check the releation of input and out
    auto selfDtyte = (*self)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*self)[0], dtypeSupportList, return false);
    for (uint64_t i = 0; i < self->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*self)[i], selfDtyte, return false);
	}

    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], selfDtyte, return false);
	}

    // check the releation of self and scalar
    if (selfDtyte == DataType::DT_BF16 || selfDtyte == DataType::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT_SUPPORT_LIST, return false);
    } else if (selfDtyte == DataType::DT_FLOAT16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT16_SUPPORT_LIST, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* self, const aclTensorList* out) {
    // tensor 维度检查
    for (uint64_t i = 0; i < self->Size(); i++) {
        OP_CHECK_MAX_DIM((*self)[i], MAX_SUPPORT_DIMS_NUMS, return false);
    }

    // self和out的shape必须一致
    for (uint64_t i = 0; i < self->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*self)[i], (*out)[i], return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* self, const aclScalar* scalar, const aclTensorList* out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, scalar, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, scalar, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Format是否满足约束
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachDivScalarV2GetWorkspaceSize(const aclTensorList *x,  const aclScalar *scalar, const aclTensorList *out, uint64_t *workspaceSize,
                                             aclOpExecutor **executor) {
     // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, scalar, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensorlist处理
    if (x->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    std::vector<const aclTensor *> tensorsVec;
    for (size_t i = 0; i < x->Size(); ++i) {
        auto secondContiguous = l0op::Contiguous((*x)[i], uniqueExecutor.get());
        CHECK_RET(secondContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorsVec.push_back(secondContiguous);
    }
    auto contiguousTensors = uniqueExecutor.get()->AllocTensorList(tensorsVec.data(), tensorsVec.size());
    CHECK_RET(contiguousTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // sclar to tensor
    const aclTensor* otherTensor;
    if ((*x)[0]->GetDataType() ==  DataType::DT_BF16) {
        otherTensor = uniqueExecutor.get()->ConvertToTensor(scalar, DataType::DT_FLOAT);
    } else {
        otherTensor = uniqueExecutor.get()->ConvertToTensor(scalar, (*x)[0]->GetDataType());
    }

    // 调用l0算子ForeachDivScalarV2进行计算
    auto result = l0op::ForeachDivScalarV2(contiguousTensors, otherTensor, out, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachDivScalarV2GetWorkspaceSize(
    const aclTensorList *x,
    const aclScalar *scalar,
    aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnForeachDivScalarV2, DFX_IN(x, scalar), DFX_OUT(out));
    return ExecForeachDivScalarV2GetWorkspaceSize(x, scalar, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachDivScalarV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnForeachDivScalarV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
