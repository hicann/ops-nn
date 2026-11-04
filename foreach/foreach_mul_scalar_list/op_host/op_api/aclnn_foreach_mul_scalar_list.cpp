/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "aclnn_foreach_mul_scalar_list.h"
#include "foreach_mul_scalar_list.h"
#include "../../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "../../../foreach_utils/op_host/foreach_null_check_helper.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

static aclnnStatus CheckParams(const aclTensorList* x, const aclScalarList* scalars, const aclTensorList* out)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(scalars != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorListNotNull(x, "x"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(x->Size() > 0 && out->Size() > 0 && scalars->Size() > 0, ACLNN_ERR_PARAM_INVALID);
    const std::initializer_list<DataType> tensorTypes = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16,
                                                         DataType::DT_INT32};
    const std::initializer_list<DataType> scalarTypes = {DataType::DT_FLOAT, DataType::DT_INT64, DataType::DT_DOUBLE};
    if (x->Size() > 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT((*x)[0], tensorTypes, return ACLNN_ERR_PARAM_INVALID);
        if (out->Size() > 0) {
            OP_CHECK_DTYPE_NOT_MATCH((*out)[0], (*x)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
        }
    }
    for (uint64_t i = 0; i < scalars->Size(); ++i) {
        OP_CHECK_NULL((*scalars)[i], return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT((*scalars)[i], scalarTypes, return ACLNN_ERR_PARAM_INVALID);
    }
    for (uint64_t i = 0; i < x->Size(); ++i) {
        CHECK_RET(!IsPrivateFormat((*x)[i]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID);
        if (i < scalars->Size()) {
            if ((*x)[i]->GetDataType() == DataType::DT_INT32) {
                OP_CHECK_DTYPE_NOT_MATCH((*scalars)[i], DataType::DT_INT64, return ACLNN_ERR_PARAM_INVALID);
            } else {
                const auto scalarType = (*scalars)[i]->GetDataType();
                if (scalarType != DataType::DT_FLOAT && scalarType != DataType::DT_DOUBLE) {
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
    }
    for (uint64_t i = 0; i < out->Size(); ++i) {
        CHECK_RET(!IsPrivateFormat((*out)[i]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachMulScalarListGetWorkspaceSize(const aclTensorList* x, const aclScalarList* scalars,
                                                      const aclTensorList* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachMulScalarList, DFX_IN(x, scalars), DFX_OUT(out));
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(x, scalars, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (x->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto contiguousX = ForeachMakeContiguousTensorList(x, uniqueExecutor.get());
    CHECK_RET(contiguousX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Reuse compatible outputs and make strided outputs contiguous before computation.
    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* scalarTensor = nullptr;
    if ((*x)[0]->GetDataType() == DataType::DT_INT32) {
        std::vector<int64_t> values;
        for (uint64_t i = 0; i < scalars->Size(); ++i) {
            values.push_back((*scalars)[i]->ToInt64());
        }
        scalarTensor = uniqueExecutor->ConvertToTensor(values.data(), values.size(), DataType::DT_INT64);
    } else {
        std::vector<float> values;
        for (uint64_t i = 0; i < scalars->Size(); ++i) {
            values.push_back((*scalars)[i]->ToFloat());
        }
        scalarTensor = uniqueExecutor->ConvertToTensor(values.data(), values.size(), DataType::DT_FLOAT);
    }
    CHECK_RET(scalarTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto result = l0op::ForeachMulScalarList(contiguousX, scalarTensor, contiguousOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_TILING_ERROR);
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachMulScalarList(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachMulScalarList);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
