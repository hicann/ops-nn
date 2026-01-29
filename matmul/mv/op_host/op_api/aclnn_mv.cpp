/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_mv.cpp
 * \brief
 */

#include "aclnn_mv.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/reshape.h"
#include "level0/unsqueeze.h"
#include "level0/zero_op.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"

using namespace Ops::NN;
using namespace op;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

namespace{
    inline bool CheckNotNull(const aclTensor* self, const aclTensor* vec, const aclTensor* out)
    {
        OP_CHECK_NULL(self, return false);
        OP_CHECK_NULL(vec, return false);
        OP_CHECK_NULL(out, return false);
        return true;
    }

    static const aclTensor* MVProcess(const aclTensor* self, const aclTensor* vec, const aclTensor* out, int8_t cubeMathType, MmOpInfo& mmOpInfo, aclOpExecutor* executor){
        return MatmulCommonProcess(self, vec, nullptr, out, cubeMathType, mmOpInfo, executor, false);
    }
}

// 1. vec、out的数据类型要与self一致  2. self的dtype为FP16/BF16/FP32。
static bool CheckDtype(const aclTensor* self, const aclTensor* vec, const aclTensor* out)
{
    // 1. vec、out的数据类型要与self一致
    // 先校验out和vec dtype一致
    OP_CHECK_DTYPE_NOT_SAME(out, vec, return false);
    // self不为空tensor时，校验self和vec dtype一致
    if (!self->IsEmpty()) {
        OP_CHECK_DTYPE_NOT_SAME(self, vec, return false);
        // 2. self的dtype为FP16/BF16/FP32。
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

// 1. vec、out的数据类型要与self一致
static bool CheckDtypeSame(const aclTensor* self, const aclTensor* vec, const aclTensor* out)
{
    // 1. vec、out的数据类型要与self一致
    // 先校验out和vec dtype一致
    OP_CHECK_DTYPE_NOT_SAME(out, vec, return false);
    // self不为空tensor时，校验self和vec dtype一致
    if (!self->IsEmpty()) {
        OP_CHECK_DTYPE_NOT_SAME(self, vec, return false);
    }
    return true;
}

// shape: self必须为2维: n x m, vec必须为1维：m, out必须为1维：n
static bool CheckShape(const aclTensor* self, const aclTensor* vec, const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(self, 2, return false);
    OP_CHECK_WRONG_DIMENSION(vec, 1, return false);
    OP_CHECK_WRONG_DIMENSION(out, 1, return false);

    int64_t selfShapeN = self->GetViewShape().GetDim(0);
    int64_t selfShapeM = self->GetViewShape().GetDim(1);
    op::Shape expectVecShape = vec->GetViewShape();
    expectVecShape.SetDim(0, selfShapeM);
    op::Shape expectOutShape = out->GetViewShape();
    expectOutShape.SetDim(0, selfShapeN);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(vec, expectVecShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectOutShape, return false);
    return true;
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* vec, aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, vec, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. self的dtype支持 + vec、out的数据类型要与self一致
    CHECK_RET(CheckDtype(self, vec, out), ACLNN_ERR_PARAM_INVALID);
    // 3. shape: self必须为2维: n x m, vec必须为1维：m, out必须为1维：n
    CHECK_RET(CheckShape(self, vec, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查cubeMathType
    CHECK_RET(CheckMathType(self, vec, cubeMathType), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputParams(const aclTensor* self, const aclTensor* vec, aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, vec, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. self的dtype支持 + vec、out的数据类型要与self一致
    CHECK_RET(CheckDtypeSame(self, vec, out), ACLNN_ERR_PARAM_INVALID);

    //  self dtype 按soc校验。
    auto socRule = SocMatMulRule::getInstance();
    CHECK_RET(socRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(socRule->CheckInput(self, vec, nullptr, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    // 3. shape: self必须为2维: n x m, vec必须为1维：m, out必须为1维：n
    CHECK_RET(CheckShape(self, vec, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}


aclnnStatus aclnnMvGetWorkspaceSize(
    const aclTensor* self, const aclTensor* vec, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMv, DFX_IN(self, vec, cubeMathType), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckInputParams(self, vec, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // mv (n x m, m) -> n 。n为0时，返回空tensor
    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // mv (n x m, m) -> n 。n不为0时，返回全为0的tensor
    if (self->IsEmpty()) {
        auto output = uniqueExecutor.get()->AllocTensor(out->GetViewShape(), out->GetDataType());
        auto zeroOut = l0op::ZerosLike(output, uniqueExecutor.get());
        CHECK_RET(zeroOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyOut = l0op::ViewCopy(zeroOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vecContiguous = l0op::Contiguous(vec, uniqueExecutor.get());
    CHECK_RET(vecContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    FVector<int64_t> dimData{-1};
    auto dims = uniqueExecutor.get()->AllocIntArray(dimData.data(), dimData.size());
    vecContiguous = l0op::UnsqueezeNd(vecContiguous, dims, uniqueExecutor.get());
    CHECK_RET(vecContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    MmOpInfo opinfo;
    auto matmulOut = MVProcess(selfContiguous, vecContiguous, out, cubeMathType, opinfo, uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    matmulOut = l0op::Cast(matmulOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    matmulOut = l0op::Reshape(matmulOut, out->GetViewShape(), uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(matmulOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMv);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
