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
 * \file aclnn_foreach_addcmul_scalar_v2.cpp
 * \brief
 */

#include "aclnn_foreach_addcmul_scalar_v2.h"
#include "foreach_addcmul_scalar_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> FOREACH_ADDCMUL_SCALAR_V2_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                    DataType::DT_FLOAT16,
                                                                                    DataType::DT_INT32,
                                                                                    DataType::DT_BF16};

static const std::initializer_list<DataType> FOREACH_ADDCMUL_SCALAR_FLOAT_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                    DataType::DT_DOUBLE};

static const std::initializer_list<DataType> FOREACH_ADDCMUL_SCALAR_FLOAT16_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                    DataType::DT_DOUBLE};

static const std::initializer_list<DataType> FOREACH_ADDCMUL_SCALAR_FLOAT16_SUPPORT_LIST_910D = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> FOREACH_SCALAR_INT_SUPPORT_LIST = {DataType::DT_INT32,
                                                                    DataType::DT_INT64};

static const std::initializer_list<DataType> EMPTY_LIST  = {};

static inline bool ForeachAddcMulScalarV2CheckNotNull(
    const aclTensorList* self, const aclTensorList* x2, const aclTensorList* x3, const aclScalar* scalar,
    const aclTensorList* out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x3, return false);
    OP_CHECK_NULL(scalar, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool ForeachAddcMulScalarV2CheckFormat(
    const aclTensorList* self, const aclTensorList* x2, const aclTensorList* x3, const aclTensorList* out) {
    for (uint64_t i = 0; i < self->Size(); i++) {
        // self格式不能是私有格式
        if (IsPrivateFormat((*self)[i]->GetStorageFormat()) || IsPrivateFormat((*x2)[i]->GetStorageFormat()) ||
            IsPrivateFormat((*x3)[i]->GetStorageFormat()) || IsPrivateFormat((*out)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
            return false;
        }
    }
    return true;
}

static const std::initializer_list<DataType>& ForeachAddcMulScalarV2GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 ||
      Ops::NN::AclnnUtil::IsRegbase()) {
    return FOREACH_ADDCMUL_SCALAR_V2_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %u is not implemented",
            static_cast<uint32_t>(GetCurrentPlatformInfo().GetCurNpuArch()));
    return EMPTY_LIST;
  }
}

static inline bool ForeachAddcMulScalarV2CheckDtypeValid(const aclTensorList* self, const aclTensorList* x2, const aclTensorList* x3,
                                   const aclScalar* scalar, const aclTensorList* out) {
    const auto& dtypeSupportList = ForeachAddcMulScalarV2GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %u is not implemented",
            static_cast<uint32_t>(GetCurrentPlatformInfo().GetCurNpuArch()));
        return false;
    }

    if (self->Size() == 0) {
        return true;
    }

    // checkself input dtype, and check the releation of input and out
    OP_CHECK_NULL((*self)[0], return false);
    auto selfDtype = (*self)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*self)[0], dtypeSupportList, return false);

    // check the releation of self and scalar
    if (selfDtype == DataType::DT_BF16 || selfDtype == DataType::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_ADDCMUL_SCALAR_FLOAT_SUPPORT_LIST, return false);
    } else if (selfDtype == DataType::DT_FLOAT16) {
        if (Ops::NN::AclnnUtil::IsRegbase()) {
            OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_ADDCMUL_SCALAR_FLOAT16_SUPPORT_LIST_910D, return false);
        } else {
            OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_ADDCMUL_SCALAR_FLOAT16_SUPPORT_LIST, return false);
        }
    } else if (selfDtype == DataType::DT_INT32) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_INT_SUPPORT_LIST, return false);
    }

    for (uint64_t i = 0; i < self->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*self)[i], selfDtype, return false);
	}

    for (uint64_t i = 0; i < x2->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x2)[i], selfDtype, return false);;
	}

    for (uint64_t i = 0; i < x3->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x3)[i], selfDtype, return false);
	}
    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], selfDtype, return false);
	}
    return true;
}

static inline bool ForeachAddcMulScalarV2CheckShape(const aclTensorList* self, const aclTensorList* x2, const aclTensorList* x3, const aclTensorList* out) {
    // tensorlist size检查
    OP_CHECK_TENSORLIST_SIZE_EQUAL(self, x2, return false);
    OP_CHECK_TENSORLIST_SIZE_EQUAL(self, x3, return false);
    OP_CHECK_TENSORLIST_SIZE_EQUAL(self, out, return false);

    // tensorlist中tensor shape一致性检查
    for (uint64_t i = 0; i < self->Size(); i++) {
       OP_CHECK_NULL((*self)[i], return false);
       OP_CHECK_NULL((*x2)[i], return false);
       OP_CHECK_NULL((*x3)[i], return false);
       OP_CHECK_SHAPE_NOT_EQUAL((*self)[i], (*x2)[i], return false);
       OP_CHECK_SHAPE_NOT_EQUAL((*self)[i], (*x3)[i], return false);
    }

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

static inline aclnnStatus ForeachAddcMulScalarV2CheckParams(const aclTensorList* self, const aclTensorList* x2, const aclTensorList* x3, const aclScalar* scalar, const aclTensorList* out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(ForeachAddcMulScalarV2CheckNotNull(self, x2, x3, scalar, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(ForeachAddcMulScalarV2CheckDtypeValid(self, x2, x3, scalar, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(ForeachAddcMulScalarV2CheckShape(self, x2, x3, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Format是否满足约束
    CHECK_RET(ForeachAddcMulScalarV2CheckFormat(self, x2, x3, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachAddcmulScalarV2GetWorkspaceSize(const aclTensorList *x1, const aclTensorList *x2, const aclTensorList *x3, const aclScalar *scalar, const aclTensorList *out, uint64_t *workspaceSize,
                                             aclOpExecutor **executor) {
     // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = ForeachAddcMulScalarV2CheckParams(x1, x2, x3, scalar, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensorlist处理
    if (x1->Size() == 0 || x2->Size() == 0 || x3->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    std::vector<const aclTensor *> tensorsVec1;
    for (size_t i = 0; i < x1->Size(); ++i) {
        auto secondContiguous = l0op::Contiguous((*x1)[i], uniqueExecutor.get());
        CHECK_RET(secondContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorsVec1.push_back(secondContiguous);
    }
    auto contiguousTensorsX1 = uniqueExecutor.get()->AllocTensorList(tensorsVec1.data(), tensorsVec1.size());
    CHECK_RET(contiguousTensorsX1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::vector<const aclTensor *> tensorsVec2;
    for (size_t i = 0; i < x2->Size(); ++i) {
        auto secondContiguous = l0op::Contiguous((*x2)[i], uniqueExecutor.get());
        CHECK_RET(secondContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorsVec2.push_back(secondContiguous);
    }
    auto contiguousTensorsX2 = uniqueExecutor.get()->AllocTensorList(tensorsVec2.data(), tensorsVec2.size());
    CHECK_RET(contiguousTensorsX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::vector<const aclTensor *> tensorsVec3;
    for (size_t i = 0; i < x3->Size(); ++i) {
        auto secondContiguous = l0op::Contiguous((*x3)[i], uniqueExecutor.get());
        CHECK_RET(secondContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorsVec3.push_back(secondContiguous);
    }
    auto contiguousTensorsX3 = uniqueExecutor.get()->AllocTensorList(tensorsVec3.data(), tensorsVec3.size());
    CHECK_RET(contiguousTensorsX3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // sclar to tensor
    const aclTensor* otherTensor;
    if ((*x1)[0]->GetDataType() ==  DataType::DT_BF16) {
        otherTensor = uniqueExecutor.get()->ConvertToTensor(scalar, DataType::DT_FLOAT);
    } else if (Ops::NN::AclnnUtil::IsRegbase() &&
               (*x1)[0]->GetDataType() == DataType::DT_FLOAT16 &&
               (scalar->GetDataType() == DataType::DT_FLOAT || scalar->GetDataType() == DataType::DT_DOUBLE)) {
        otherTensor = uniqueExecutor.get()->ConvertToTensor(scalar, DataType::DT_FLOAT);
    } else {
        otherTensor = uniqueExecutor.get()->ConvertToTensor(scalar, (*x1)[0]->GetDataType());
    }
    CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 调用l0算子ForeachAddcmulScalarV2进行计算
    auto result = l0op::ForeachAddcmulScalarV2(contiguousTensorsX1, contiguousTensorsX2, contiguousTensorsX3, otherTensor, out, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_INVALID);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachAddcmulScalarV2GetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    const aclTensorList *x3,
    const aclScalar *scalar,
    aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnForeachAddcmulScalarV2, DFX_IN(x1, x2, x3, scalar), DFX_OUT(out));
    return ExecForeachAddcmulScalarV2GetWorkspaceSize(x1, x2, x3, scalar, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachAddcmulScalarV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnForeachAddcmulScalarV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
