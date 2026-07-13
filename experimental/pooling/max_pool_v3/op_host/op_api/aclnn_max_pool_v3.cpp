/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_pool_v3.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "max_pool_v3.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"
#include "op_api/level2_base_caculation.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                 op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static aclnnStatus CheckParams(const aclTensor* x, const aclIntArray* ksize, const aclIntArray* strides,
                               const aclIntArray* pads, const aclTensor* out)
{
    CHECK_RET(CheckNotNull2Tensor(x, out), ACLNN_ERR_PARAM_NULLPTR);
    if (ksize == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strides == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto supportList = GetDtypeSupportListV1(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
    CHECK_RET(CheckDtypeValidActivation(x, out, supportList), ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_MAX_DIM(x, MAX_SUPPORT_DIMS_NUMS, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxPoolV3GetWorkspaceSize(const aclTensor* x, const aclIntArray* ksize, const aclIntArray* strides,
                                           const aclIntArray* pads, const aclScalar* ceilMode, aclTensor* out,
                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnMaxPoolV3, DFX_IN(x, ksize, strides, pads), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, ksize, strides, pads, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    size_t dimSize = x->GetViewShape().GetDimNum();
    auto reshapeX = ReshapeSelfValueGetActivation(x, dimSize, xContiguous, uniqueExecutor);

    bool ceilModeBool = false;
    if (ceilMode != nullptr) {
        ceilModeBool = (ceilMode->ToInt64() != 0);
    }
    auto poolOut = l0op::MaxPoolV3(reshapeX, ksize, strides, pads, ceilModeBool, out, uniqueExecutor.get());
    CHECK_RET(poolOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxPoolV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxPoolV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
