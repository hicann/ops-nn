/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_addmv.h"
#include "level0/add.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/mul.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"
#include "level0/reduce_sum_op.h"
#include "level0/logical_and.h"
#include "level0/logical_or.h"
#include "level0/reduce_any.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace Ops::NN;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

#define MATRIX_DIM 2
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> addmvDtypeSupportList = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64,  DataType::DT_FLOAT16, DataType::DT_INT16,
    DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_BOOL};
static const std::initializer_list<DataType> matmulDtypeSupportList = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static bool CheckNotNull(
    const aclTensor* self, const aclTensor* mat, const aclTensor* vec, const aclScalar* alpha, const aclScalar* beta,
    const aclTensor* out)
{
    // 检查输入的数据类型是否在算子的支持列表内
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(mat, return false);
    OP_CHECK_NULL(vec, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(beta, return false);

    // 检查输出的数据类型是否在算子的支持列表内
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* mat, const aclTensor* vec, const aclScalar* alpha, const aclScalar* beta,
    const aclTensor* out)
{
    // 检查self的数据类型是否在addmv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, addmvDtypeSupportList, return false);

    // 检查mat的数据类型是否在addmv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(mat, addmvDtypeSupportList, return false);

    // 检查vec的数据类型是否在addmv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(vec, addmvDtypeSupportList, return false);

    // 检查alpha的数据类型是否在addmv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(alpha, addmvDtypeSupportList, return false);

    // 检查beta的数据类型是否在addmv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(beta, addmvDtypeSupportList, return false);

    // 检查out的数据类型是否在addmv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, addmvDtypeSupportList, return false);

    return true;
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static bool CheckPromoteType(
    const aclTensor* self, const aclTensor* mat, const aclTensor* vec, const aclScalar* alpha, const aclScalar* beta,
    const aclTensor* out)
{
    // 检查mat和vec能否做数据类型推导
    DataType promoteType = PromoteType(mat->GetDataType(), vec->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Mat dtype %s and Vec dtype %s can not promote dtype.",
            ToString(mat->GetDataType()).GetString(), ToString(vec->GetDataType()).GetString());
        return false;
    }

    // 检查alpha和mat/vec能否做数据类型推导
    DataType promoteType2 = PromoteType(alpha->GetDataType(), promoteType);
    if (promoteType2 == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Alpha dtype %s and Mat/Vec dtype %s can not promote dtype.",
            ToString(alpha->GetDataType()).GetString(), ToString(promoteType).GetString());
        return false;
    }

    // 检查self和beta能否做数据类型推导
    DataType promoteType3 = PromoteType(beta->GetDataType(), self->GetDataType());
    if (promoteType3 == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and Beta dtype %s can not promote dtype.",
            ToString(self->GetDataType()).GetString(), ToString(beta->GetDataType()).GetString());
        return false;
    }

    // 检查self和promote_type能否做数据类型推导
    DataType promoteTypeFinal = PromoteType(promoteType3, promoteType2);
    if (promoteTypeFinal == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "part1(beta*self) dtype %s and part2(alpha*(mat@vec)) dtype %s can not promote dtype.",
            ToString(promoteType3).GetString(), ToString(promoteType2).GetString());
        return false;
    }

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteTypeFinal, out->GetDataType(), return false);

    return true;
}

static bool CheckTensorDimAndSize(
    const aclTensor* self, const aclTensor* mat, const aclTensor* vec, const aclTensor* out)
{
    // 需要根据算子实际情况添加校验
    Shape selfShape = self->GetViewShape();
    Shape matShape = mat->GetViewShape();
    Shape vecShape = vec->GetViewShape();
    Shape outShape = out->GetViewShape();
    if ((selfShape.GetDimNum() > 1) || (matShape.GetDimNum() != MATRIX_DIM) || (vecShape.GetDimNum() != 1) ||
        (outShape.GetDimNum() != 1)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expect input tensor dim [0/1,2,1,1], but receive self [%zu], mat [%zu], vec [%zu], out [%zu].",
            selfShape.GetDimNum(), matShape.GetDimNum(), vecShape.GetDimNum(), outShape.GetDimNum());
        return false;
    }

    if ((matShape.GetDim(1) != vecShape.GetDim(0)) || (matShape.GetDim(0) != outShape.GetDim(0)) ||
        (selfShape.GetDimNum() != 0 && selfShape.GetDim(0) != 1 && matShape.GetDim(0) != selfShape.GetDim(0))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Input tensor shape not statisify, current shape : self [%s], mat [%s], vec [%s], out [%s].",
            op::ToString(selfShape).GetString(), op::ToString(matShape).GetString(), op::ToString(vecShape).GetString(),
            op::ToString(outShape).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* mat, const aclTensor* vec, const aclScalar* alpha, const aclScalar* beta,
    const aclTensor* out, int8_t cubeMathType)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat, vec, alpha, beta, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, mat, vec, alpha, beta, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, mat, vec, alpha, beta, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入是否满足矩阵乘法和加法要求
    CHECK_RET(CheckTensorDimAndSize(self, mat, vec, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查cubeMathType
    CHECK_RET(CheckMathType(mat, vec, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline const aclTensor* MulOrLogAndCalculation(
    const aclTensor* lTensor, const aclTensor* rTensor, DataType dtype, aclOpExecutor* executor)
{
    return dtype == DataType::DT_BOOL ? l0op::LogicalAnd(lTensor, rTensor, executor) :
                                        l0op::Mul(lTensor, rTensor, executor);
}

static inline const aclTensor* AddOrLogOrCalculation(
    const aclTensor* lTensor, const aclTensor* rTensor, DataType dtype, aclOpExecutor* executor)
{
    return dtype == DataType::DT_BOOL ? l0op::LogicalOr(lTensor, rTensor, executor) :
                                        l0op::Add(lTensor, rTensor, executor);
}

static inline const aclTensor* MulScalar(const aclScalar* alpha, const aclTensor* vecResult, aclOpExecutor* executor)
{
    // 当alpha!=1 需要做一个乘法
    auto promoteType = PromoteType(alpha->GetDataType(), vecResult->GetDataType());
    auto alphaTensor = executor->ConvertToTensor(alpha, promoteType);
    CHECK_RET(alphaTensor != nullptr, nullptr);
    // 将输入vecResult的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto vecResultCast = l0op::Cast(vecResult, promoteType, executor);
    CHECK_RET(vecResultCast != nullptr, nullptr);
    auto vecFinalResult = MulOrLogAndCalculation(vecResultCast, alphaTensor, promoteType, executor);
    return vecFinalResult;
}

const aclTensor* GetMatMulResult(
    const aclTensor* mat, const aclTensor* vec, int8_t cubeMathType, const aclScalar* alpha, aclOpExecutor* executor)
{
    // 将输入vec转换成连续的tensor
    auto vecContiguous = l0op::Contiguous(vec, executor);
    CHECK_RET(vecContiguous != nullptr, nullptr);

    // 将输入mat转换成连续的tensor
    auto matContiguous = l0op::Contiguous(mat, executor);
    CHECK_RET(matContiguous != nullptr, nullptr);

    // 调用matmul之前需要把vec转换成2维矩阵
    const int64_t appendDim[] = {1};
    aclIntArray* vecToMatShape = executor->AllocIntArray(appendDim, 1);
    auto mat2 = l0op::UnsqueezeNd(vecContiguous, vecToMatShape, executor);

    // 调用matmul之前需要对mat和vec做数据类型转换，以满足运算条件
    auto promoteType = PromoteType(matContiguous->GetDataType(), mat2->GetDataType());

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto matCasted = l0op::Cast(matContiguous, promoteType, executor);
    CHECK_RET(matCasted != nullptr, nullptr);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto mat2Casted = l0op::Cast(mat2, promoteType, executor);
    CHECK_RET(mat2Casted != nullptr, nullptr);

    // 调用接口进行矩阵乘法运算
    auto matMulOut = ExecMmOp(matCasted, mat2Casted, cubeMathType, executor);
    CHECK_RET(matMulOut != nullptr, nullptr);
    // 将矩阵乘结果转换为vec
    auto vecResult = l0op::SqueezeNd(matMulOut, vecToMatShape, executor);
    CHECK_RET(vecResult != nullptr, nullptr);
    // 当alpha!=1 需要做一个乘法
    if (std::abs(alpha->ToFloat() - 1.0f) > std::numeric_limits<float>::epsilon()) {
        vecResult = MulScalar(alpha, vecResult, executor);
        CHECK_RET(vecResult != nullptr, nullptr);
    }
    return vecResult;
}

const aclTensor* GetMulResult(
    const aclTensor* mat, const aclTensor* vec, const aclScalar* alpha, aclOpExecutor* executor)
{
    // 将输入vec转换成连续的tensor
    auto vecContiguous = l0op::Contiguous(vec, executor);
    CHECK_RET(vecContiguous != nullptr, nullptr);

    // 将输入mat转换成连续的tensor
    auto matContiguous = l0op::Contiguous(mat, executor);
    CHECK_RET(matContiguous != nullptr, nullptr);

    // 调用matmul之前需要对mat和vec做数据类型转换，以满足运算条件
    auto promoteType = PromoteType(matContiguous->GetDataType(), vecContiguous->GetDataType());

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto matCasted = l0op::Cast(matContiguous, promoteType, executor);
    CHECK_RET(matCasted != nullptr, nullptr);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto vecCasted = l0op::Cast(vecContiguous, promoteType, executor);
    CHECK_RET(vecCasted != nullptr, nullptr);

    // 调用接口进行乘法运算
    auto matMulOut = MulOrLogAndCalculation(matCasted, vecCasted, promoteType, executor);
    CHECK_RET(matMulOut != nullptr, nullptr);
    // 将矩阵乘结果转换为vec
    const int64_t reduceDim[] = {1};
    aclIntArray* reduceDimArray = executor->AllocIntArray(reduceDim, 1);
    const aclTensor* vecResult = nullptr;
    if (promoteType == DataType::DT_BOOL) {
        vecResult = l0op::ReduceAny(matMulOut, reduceDimArray, false, executor);
    } else {
        vecResult = l0op::ReduceSumOp(matMulOut, reduceDimArray, false, executor);
    }

    CHECK_RET(vecResult != nullptr, nullptr);
    if (std::abs(alpha->ToFloat() - 1.0f) > std::numeric_limits<float>::epsilon()) {
        vecResult = MulScalar(alpha, vecResult, executor);
    }
    return vecResult;
}
static inline bool IsMatMulSupportDtype(const aclTensor* mat, const aclTensor* vec)
{
    auto promoteType = PromoteType(mat->GetDataType(), vec->GetDataType());
    if (CheckType(promoteType, matmulDtypeSupportList)) {
        return true;
    }
    return false;
}

aclnnStatus aclnnAddmvGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat, const aclTensor* vec, const aclScalar* alpha, const aclScalar* beta,
    aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddmv, DFX_IN(self, mat, vec, alpha, beta, cubeMathType), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, mat, vec, alpha, beta, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* matMulRet = nullptr;
    if (std::abs(alpha->ToFloat() - 0.0f) > std::numeric_limits<float>::epsilon() &&
        (!(mat->IsEmpty() || vec->IsEmpty()))) {
        if (IsMatMulSupportDtype(mat, vec)) {
            matMulRet = GetMatMulResult(mat, vec, cubeMathType, alpha, uniqueExecutor.get());
        } else {
            matMulRet = GetMulResult(mat, vec, alpha, uniqueExecutor.get());
        }
        CHECK_RET(matMulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* mulRet = nullptr;
    if (matMulRet == nullptr || std::abs(beta->ToFloat() - 0.0f) > std::numeric_limits<float>::epsilon()) {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (std::abs(beta->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) {
            mulRet = selfContiguous;
        } else {
            mulRet = MulScalar(beta, selfContiguous, uniqueExecutor.get());
        }
    }

    const aclTensor* vecResultFinal = nullptr;
    if (mulRet != nullptr && matMulRet != nullptr) {
        // Add算子需要对self和other两个输入做隐式数据类型转换，根据具体算子语义按需调用
        auto promoteType3 = PromoteType(mulRet->GetDataType(), matMulRet->GetDataType());
        // 将第一项的数据类型转换成推导后的数据类型
        auto mulRetCasted = l0op::Cast(mulRet, promoteType3, uniqueExecutor.get());
        CHECK_RET(mulRetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将第二项的数据类型转换成推导后的数据类型
        auto matMulRetCasted = l0op::Cast(matMulRet, promoteType3, uniqueExecutor.get());
        CHECK_RET(matMulRetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 调用加法进行计算
        vecResultFinal = AddOrLogOrCalculation(matMulRetCasted, mulRetCasted, promoteType3, uniqueExecutor.get());
        CHECK_RET(vecResultFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (mulRet != nullptr) {
        vecResultFinal = mulRet;
    } else {
        vecResultFinal = matMulRet;
    }

    // 将计算结果转换成输出out的数据类型
    auto finalRet = l0op::Cast(vecResultFinal, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(finalRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(finalRet, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddmv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddmv);
    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
