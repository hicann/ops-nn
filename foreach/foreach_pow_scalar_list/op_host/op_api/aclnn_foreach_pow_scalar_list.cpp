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
#include "aclnn_foreach_pow_scalar_list.h"
#include "foreach_pow_scalar_list_l0op.h"
#include "../../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "../../../foreach_utils/op_host/foreach_null_check_helper.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/make_op_executor.h"
#include "base/err_msg.h"
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
    CHECK_RET(scalars->Size() == x->Size(), ACLNN_ERR_PARAM_INVALID);
    // out 的 Tensor 个数需大于等于 x 的 Tensor 个数，多出的 out 块不写入（与同族接口口径一致）
    if (out->Size() < x->Size()) {
        const std::string
            reason = ("The out tensorlist must contain at least as many tensors as the x tensorlist, but got out=" +
                      std::to_string(out->Size()) + " and x=" + std::to_string(x->Size()) + ".");
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ0038",
                                  (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
                                  (std::vector<const char*>{"out", "aclnnForeachPowScalarList",
                                                            (std::to_string(out->Size())).c_str(), reason.c_str()}));
        return ACLNN_ERR_PARAM_INVALID;
    }
    const std::initializer_list<DataType> tensorTypes = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16,
                                                         DataType::DT_INT32};
    const std::initializer_list<DataType> floatScalarTypes = {DataType::DT_FLOAT, DataType::DT_DOUBLE};
    OP_CHECK_DTYPE_NOT_SUPPORT((*x)[0], tensorTypes, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH((*out)[0], (*x)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    for (uint64_t i = 0; i < x->Size(); ++i) {
        OP_CHECK_DTYPE_NOT_MATCH((*x)[i], (*x)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(!IsPrivateFormat((*x)[i]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID);
    }
    for (uint64_t i = 0; i < out->Size(); ++i) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], (*x)[0]->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(!IsPrivateFormat((*out)[i]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID);
    }
    // out[i] 的 shape size 需大于等于 x[i]（资料口径），避免向更小的输出 buffer 写入
    for (uint64_t i = 0; i < x->Size(); ++i) {
        const int64_t outSize = (*out)[i]->GetViewShape().GetShapeSize();
        const int64_t xSize = (*x)[i]->GetViewShape().GetShapeSize();
        if (outSize < xSize) {
            const std::string reason = ("The shape size of the " + std::to_string(i) +
                                        "th tensor in out must be greater than or equal to that of x, but got " +
                                        std::to_string(outSize) + " and " + std::to_string(xSize) + ".");
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
            REPORT_PREDEFINED_ERR_MSG("EZ0015",
                                      (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
                                      (std::vector<const char*>{"out", "aclnnForeachPowScalarList",
                                                                (std::to_string(outSize)).c_str(), reason.c_str()}));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // scalars与x逐位配对：INT32张量仅接受INT64标量，浮点张量接受FLOAT/DOUBLE（DOUBLE内部转FLOAT32参与计算）
    for (uint64_t i = 0; i < scalars->Size(); ++i) {
        OP_CHECK_NULL((*scalars)[i], return ACLNN_ERR_PARAM_INVALID);
        if ((*x)[i]->GetDataType() == DataType::DT_INT32) {
            OP_CHECK_DTYPE_NOT_MATCH((*scalars)[i], DataType::DT_INT64, return ACLNN_ERR_PARAM_INVALID);
        } else {
            OP_CHECK_DTYPE_NOT_SUPPORT((*scalars)[i], floatScalarTypes, return ACLNN_ERR_PARAM_INVALID);
        }
    }
    return ACLNN_SUCCESS;
}

static const aclTensor* ConvertScalarsToTensor(const aclScalarList* scalars, DataType dataType, aclOpExecutor* executor)
{
    // scalars列表在kernel侧是一个shape为[N]的tensor，需将标量列表转换为单个tensor
    if (dataType == DataType::DT_INT64) {
        std::vector<int64_t> scalarsVec;
        scalarsVec.reserve(scalars->Size());
        for (uint64_t i = 0; i < scalars->Size(); i++) {
            scalarsVec.push_back((*scalars)[i]->ToInt64());
        }
        return executor->ConvertToTensor(scalarsVec.data(), scalarsVec.size(), dataType);
    }
    std::vector<float> scalarsVec;
    scalarsVec.reserve(scalars->Size());
    for (uint64_t i = 0; i < scalars->Size(); i++) {
        scalarsVec.push_back((*scalars)[i]->ToFloat());
    }
    return executor->ConvertToTensor(scalarsVec.data(), scalarsVec.size(), dataType);
}

aclnnStatus aclnnForeachPowScalarListGetWorkspaceSize(const aclTensorList* x, const aclScalarList* scalars,
                                                      const aclTensorList* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachPowScalarList, DFX_IN(x, scalars), DFX_OUT(out));
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(x, scalars, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousX = ForeachMakeContiguousTensorList(x, uniqueExecutor.get());
    CHECK_RET(contiguousX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // scalars转tensor，dtype按x的dtype映射：INT32->INT64，其余->FLOAT
    auto scalarsDtype = ((*x)[0]->GetDataType() == DataType::DT_INT32) ? DataType::DT_INT64 : DataType::DT_FLOAT;
    auto scalarsTensor = ConvertScalarsToTensor(scalars, scalarsDtype, uniqueExecutor.get());
    CHECK_RET(scalarsTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子ForeachPowScalarList进行计算，输出到连续buffer
    auto result = l0op::ForeachPowScalarList(contiguousX, scalarsTensor, contiguousOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_TILING_ERROR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachPowScalarList(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachPowScalarList);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
