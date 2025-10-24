/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gemm.h"
#include "level0/add.h"
#include "level0/axpy.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "matmul/mat_mul_v3/op_host/op_api/matmul.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "level0/muls.h"
#include "aclnn_kernels/transpose.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace Ops::NN;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* A, const aclTensor* B, const aclTensor* C, const aclTensor* out)
{
    OP_CHECK_NULL(C, return false);
    OP_CHECK_NULL(A, return false);
    OP_CHECK_NULL(B, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static inline bool CheckDtypeValid(const aclTensor* C, const aclTensor* A, const aclTensor* B, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(A, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(B, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(C, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);

    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && (C->GetDataType() == op::DataType::DT_BF16 || A->GetDataType() == op::DataType::DT_BF16 ||
                      B->GetDataType() == op::DataType::DT_BF16 || out->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], A is %s, B is %s, C is %s, out is %s",
            op::ToString(socVersion).GetString(), op::ToString(A->GetDataType()).GetString(),
            op::ToString(B->GetDataType()).GetString(), op::ToString(C->GetDataType()).GetString(),
            op::ToString(out->GetDataType()).GetString());
        return false;
    }

    return true;
}

static inline bool CheckMatmul(const aclTensor* A, const aclTensor* B, int64_t transA, int64_t transB)
{
    // check A dims number is 2
    OP_CHECK_WRONG_DIMENSION(A, 2, return false);

    // check B dims number is 2
    OP_CHECK_WRONG_DIMENSION(B, 2, return false);

    // check whether matrices can be multiplied
    auto kDimA = transA ? (A->GetViewShape())[0] : (A->GetViewShape())[1];
    auto kDimB = transB ? (B->GetViewShape())[1] : (B->GetViewShape())[0];
    if (kDimA != kDimB) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different.");
        return false;
    }

    return true;
}

static inline bool CheckBroadcast(
    const aclTensor* A, const aclTensor* B, const aclTensor* C, int64_t transA, int64_t transB)
{
    auto mDim = transA ? (A->GetViewShape())[1] : (A->GetViewShape())[0];
    auto nDim = transB ? (B->GetViewShape())[0] : (B->GetViewShape())[1];
    op::Shape matmulShape = {mDim, nDim};
    OP_CHECK_BROADCAST_WITH_SHAPE(C, matmulShape, return false);

    return true;
}

// 假设A是 n x m，B是 m x p，out必须是 n x p    如果n / p为0，那么out为empty即可
static inline bool CheckOutShape(
    const aclTensor* A, const aclTensor* B, int64_t transA, int64_t transB, const aclTensor* out)
{
    auto mDim = transA ? A->GetViewShape().GetDim(1) : A->GetViewShape().GetDim(0);
    auto nDim = transB ? B->GetViewShape().GetDim(0) : B->GetViewShape().GetDim(1);

    int64_t out_m = out->GetViewShape().GetDim(0);
    int64_t out_n = out->GetViewShape().GetDim(1);

    if (mDim == 0 || nDim == 0) {
        return out->IsEmpty();
    }
    return mDim == out_m && nDim == out_n;
}

static inline bool CheckMathType(const aclTensor* A, const aclTensor* B, int8_t cubeMathType)
{
    bool AFloat = A->GetDataType() == DataType::DT_FLOAT;
    bool BFloat = B->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = AFloat || BFloat ? DataType::DT_FLOAT : A->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static aclnnStatus CheckParams(
    const aclTensor* A, const aclTensor* B, const aclTensor* C, int64_t transA, int64_t transB, const aclTensor* out,
    int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(A, B, C, out), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(A, B, C, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查A和B是否为2维，且是否满足相乘条件
    CHECK_RET(CheckMatmul(A, B, transA, transB), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查C和A@B是否能broadcast
    CHECK_RET(CheckBroadcast(A, B, C, transA, transB), ACLNN_ERR_PARAM_INVALID);

    // 5. out必须和A@B的shape一致
    CHECK_RET(CheckOutShape(A, B, transA, transB, out), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查cubeMathType
    CHECK_RET(CheckMathType(A, B, cubeMathType), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// A: m x k, B: k x n  ->   m x n 是否为空tensor，为空tensor返回true
static inline bool CheckMulResIsEmpty(const aclTensor* A, const aclTensor* B, int64_t transA, int64_t transB)
{
    auto mDim = transA ? A->GetViewShape().GetDim(1) : A->GetViewShape().GetDim(0);
    auto nDim = transB ? B->GetViewShape().GetDim(0) : B->GetViewShape().GetDim(1);
    return mDim == 0 || nDim == 0;
}

// A * B结果为空tensor，返回 beta * C 并且broadcast成out的shape
static aclnnStatus GemmMulEmptyProcess(const aclTensor* C, float beta, aclTensor* out, aclOpExecutor* executor)
{
    auto CContiguous = l0op::Contiguous(C, executor);
    CHECK_RET(CContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mulOut = l0op::Muls(CContiguous, beta, executor);
    CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // broadcast成和out一个shape
    if (mulOut->GetViewShape() != out->GetViewShape()) {
        int64_t tensorSize = (int64_t)(out->GetViewShape().GetDimNum());
        std::vector<int64_t> tensorShape(tensorSize);
        for (int64_t i = 0; i < tensorSize; i++) {
            tensorShape[i] = (out->GetViewShape())[i];
        }
        auto outShape = executor->AllocIntArray(tensorShape.data(), tensorSize);
        mulOut = l0op::BroadcastTo(mulOut, outShape, executor);
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyResult = l0op::ViewCopy(mulOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor* AddProcess(
    const aclTensor* mulOut, const aclTensor* matmulOut, float alpha, aclOpExecutor* executor)
{
    // Add算子需要对C和A两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteTypeAdd = op::PromoteType(mulOut->GetDataType(), matmulOut->GetDataType());
    auto mulOutCasted = l0op::Cast(mulOut, promoteTypeAdd, executor);
    CHECK_RET(mulOutCasted != nullptr, nullptr);
    auto matmulOutCasted = l0op::Cast(matmulOut, promoteTypeAdd, executor);
    CHECK_RET(matmulOutCasted != nullptr, nullptr);

    // 进行Add计算
    const aclTensor* addOut = nullptr;
    if (std::abs(alpha - 1.0f) <= std::numeric_limits<float>::epsilon()) {
        addOut = l0op::Add(mulOutCasted, matmulOutCasted, executor);
    } else {
        addOut = l0op::Axpy(mulOutCasted, matmulOutCasted, alpha, executor);
    }
    return addOut;
}

aclnnStatus aclnnGemmGetWorkspaceSize(
    const aclTensor* A, const aclTensor* B, const aclTensor* C, float alpha, float beta, int64_t transA, int64_t transB,
    aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGemm, DFX_IN(A, B, C, alpha, beta, transA, transB, cubeMathType), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(A, B, C, transA, transB, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果C是空tensor，返回空tensor。如果A 和B是空tensor，则乘积也是空tensor，返回空tensor
    if (C->IsEmpty() || CheckMulResIsEmpty(A, B, transA, transB)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 如果A mk 和B kn 是空tensor，但是mn不是空tensor, 返回Beta C
    if (!CheckMulResIsEmpty(A, B, transA, transB) && (A->IsEmpty() || B->IsEmpty())) {
        auto GemmRes = GemmMulEmptyProcess(C, beta, out, uniqueExecutor.get());
        CHECK_RET(GemmRes == ACLNN_SUCCESS, GemmRes);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* castOut = nullptr;
    if (std::abs(beta - 0.0f) <= std::numeric_limits<float>::epsilon()) {
        auto matmulOut = ExecMmOpWithTrans(A, B, transA, transB, cubeMathType, uniqueExecutor.get());
        CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto mulOut = matmulOut;
        if (std::abs(alpha - 1.0f) >= std::numeric_limits<float>::epsilon()) {
            mulOut = l0op::Muls(matmulOut, alpha, uniqueExecutor.get());
            CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        castOut = l0op::Cast(mulOut, out->GetDataType(), uniqueExecutor.get());
    } else {
        // beta * C
        auto CContiguous = l0op::Contiguous(C, uniqueExecutor.get());
        CHECK_RET(CContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto mulOut = CContiguous;
        if (std::abs(beta - 1.0f) >= std::numeric_limits<float>::epsilon()) {
            mulOut = l0op::Muls(CContiguous, beta, uniqueExecutor.get());
            CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // matmul (alpha*matmulOut + mulOut)
        auto matmulOut = ExecMmOpWithTrans(A, B, transA, transB, cubeMathType, uniqueExecutor.get());
        CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto addOut = AddProcess(mulOut, matmulOut, alpha, uniqueExecutor.get());
        CHECK_RET(addOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        castOut = l0op::Cast(addOut, out->GetDataType(), uniqueExecutor.get());
    }

    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGemm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGemm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif