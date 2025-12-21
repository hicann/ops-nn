/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shrink.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_shrink.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out) {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static inline bool CheckNotNull(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias, const aclTensor* out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(lambd, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static inline bool CheckShape(const aclTensor* self,const aclTensor* out){
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}
static inline bool CheckLambdValue(const aclScalar* lambd){
    if(lambd->ToFloat() < 0.0f){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "lambd should be greater or equal to 0, but found to be [%f].", lambd->ToFloat());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias, const aclTensor* out){
    CHECK_RET(CheckNotNull(self, lambd, bias, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckLambdValue(lambd), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecShrinkGetWorkspaceSize(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor){
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, lambd, bias, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if(self->IsEmpty()){
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto shrinkOutRet = l0op::Shrink(selfContiguous, lambd->ToFloat(), bias->ToFloat(), uniqueExecutor.get());
    CHECK_RET(shrinkOutRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(shrinkOutRet, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnShrinkGetWorkspaceSize(const aclTensor* self, const aclScalar* lambd, const aclScalar* bias, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor){
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnShrink, DFX_IN(self, lambd, bias), DFX_OUT(out));
    return ExecShrinkGetWorkspaceSize(self, lambd, bias,out,workspaceSize, executor);
}
aclnnStatus aclnnShrink(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream){
    L2_DFX_PHASE_2(aclnnShrink);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
} // namespace op;
#endif

