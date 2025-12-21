/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_baddbmm.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "level0/add.h"
#include "level0/axpy.h"
#include "batch_matmul.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/muls.h"

#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"

using namespace Ops::NN;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static const int32_t SHAPE_LIMIT = 3;
static const int32_t FIRST_DIM = 0;
static const int32_t PENULTIMATE_DIM = 2;
static const int32_t LAST_DIM = 1;

static inline bool CheckInputNotNull(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(batch1, return false);
    OP_CHECK_NULL(batch2, return false);
    OP_CHECK_NULL(beta, return false);
    OP_CHECK_NULL(alpha, return false);
    return true;
}

static inline bool CheckOutputNotNull(const aclTensor* out)
{
    OP_CHECK_NULL(out, return false);
    return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> dtypeSupportListWithoutBf16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclTensor* out)
{
    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto dtypeList = bf16flag ? dtypeSupportList : dtypeSupportListWithoutBf16;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(batch1, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(batch2, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || batch1->GetDataType() == op::DataType::DT_BF16 ||
                      batch2->GetDataType() == op::DataType::DT_BF16 || out->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], self is %s, mat1 is %s, mat2 is %s, out is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(batch1->GetDataType()).GetString(), op::ToString(batch2->GetDataType()).GetString(),
            op::ToString(out->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* selfTensor, const aclTensor* batch1Tensor, const aclTensor* batch2Tensor)
{
    // check bmm shape
    OP_CHECK_WRONG_DIMENSION(batch1Tensor, SHAPE_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(batch2Tensor, SHAPE_LIMIT, return false);

    auto batch1DimNum = batch1Tensor->GetViewShape().GetDimNum();
    auto batch2DimNum = batch2Tensor->GetViewShape().GetDimNum();
    const op::Shape batch1 = batch1Tensor->GetViewShape();
    const op::Shape batch2 = batch2Tensor->GetViewShape();
    // batch1DimNum - LAST_DIM means the last element, batch2DimNum - PENULTIMATE_DIM means the penultimate element
    if (batch1[batch1DimNum - LAST_DIM] != batch2[batch2DimNum - PENULTIMATE_DIM]) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "batch1's last dim and batch2's penultimate dim shoule be same, batch1 [%ld], batch2 [%ld].",
            batch1[batch1DimNum - LAST_DIM], batch2[batch2DimNum - PENULTIMATE_DIM]);
        return false;
    }

    if (!CheckBatchDimBroadcast(batch1DimNum, batch2DimNum, batch1, batch2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self's batch dim and mat2's batch dim can not broadcast");
        return false;
    }

    // check self is empty or not
    bool selfIsEmpty = selfTensor->IsEmpty();
    // check batch1@batch2 is empty or not
    bool matIsEmpty =
        (batch1[FIRST_DIM] == 0 || batch1[SHAPE_LIMIT - PENULTIMATE_DIM] == 0 || batch2[SHAPE_LIMIT - LAST_DIM] == 0) ?
            true :
            false;

    if ((selfIsEmpty && !matIsEmpty) || (!selfIsEmpty && matIsEmpty)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "self and batch1@batch2 should be both empty tensors, or not, self is [%d], batch1@batch2 is [%d].",
            selfIsEmpty, matIsEmpty);
        return false;
    }

    return true;
}

static bool CheckBroadCast(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclTensor* out)
{
    op::Shape broadcastShape;
    op::Shape bmmShape = {(batch1->GetViewShape())[0], (batch1->GetViewShape())[1], (batch2->GetViewShape())[2]};
    if (!BroadcastInferShape(self->GetViewShape(), bmmShape, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of self and batch1@batch2 can't broadcast.");
        return false;
    }

    if (!BroadcastInferShape(self->GetViewShape(), broadcastShape, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of self and other can't broadcast.");
        return false;
    }

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }

    if ((broadcastShape[FIRST_DIM] != 1 && (batch1->GetViewShape())[FIRST_DIM] == 1) ||
        (broadcastShape[SHAPE_LIMIT - PENULTIMATE_DIM] != 1 &&
         (batch1->GetViewShape())[SHAPE_LIMIT - PENULTIMATE_DIM] == 1) ||
        (broadcastShape[SHAPE_LIMIT - LAST_DIM] != 1 && (batch2->GetViewShape())[SHAPE_LIMIT - LAST_DIM] == 1)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "batch1@batch2's shape should be same with broadcastShape, broadcastShape [%s], bmmShape [%s]",
            op::ToString(broadcastShape).GetString(), op::ToString(bmmShape).GetString());
        return false;
    }

    return true;
}

static inline bool CheckFormat(const aclTensor* batch1, const aclTensor* batch2, const aclTensor* out)
{
    if (batch1->GetViewFormat() != batch2->GetViewFormat() || batch1->GetViewFormat() != out->GetViewFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format of all input and output should be equal, batch1 [%s], batch2 [%s], out [%s].",
            op::ToString(batch1->GetViewFormat()).GetString(), op::ToString(batch2->GetViewFormat()).GetString(),
            op::ToString(out->GetViewFormat()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha, const aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查输入参数是否为空指针
    CHECK_RET(CheckInputNotNull(self, batch1, batch2, beta, alpha), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckOutputNotNull(out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, batch1, batch2, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查batch1和batch2是否满足条件
    CHECK_RET(CheckShape(self, batch1, batch2), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和batch1@batch2是否能broadcast
    CHECK_RET(CheckBroadCast(self, batch1, batch2, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查batch1, batch2和out的format是否一致，self存在与其他输入format不一样的情况
    CHECK_RET(CheckFormat(batch1, batch2, out), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查cubeMathType
    CHECK_RET(CheckMathType(batch1, batch2, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* bmmProcessEmptyTensor(const aclTensor* self, const aclTensor* mat2, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    // 获取self的第1维度、第2维度和mat2的最后1维度作为输出shape
    op::Shape outShape = {selfShape.GetDim(0), selfShape.GetDim(1), mat2Shape.GetDim(2)};
    auto out = executor->AllocTensor(outShape, self->GetDataType());
    OP_LOGI("Returning an empty tensor without actually doing calculation");
    return out;
}
} // namespace

aclnnStatus aclnnBaddbmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBaddbmm, DFX_IN(self, batch1, batch2, beta, alpha, cubeMathType), DFX_OUT(out));
    // 参数检查
    auto ret = CheckParams(self, batch1, batch2, beta, alpha, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 输入空Tensor处理方法
    if ((batch1->GetViewShape())[FIRST_DIM] == 0 || (batch1->GetViewShape())[SHAPE_LIMIT - PENULTIMATE_DIM] == 0 ||
        (batch2->GetViewShape())[SHAPE_LIMIT - LAST_DIM] == 0) {
        auto emptyOut = bmmProcessEmptyTensor(batch1, batch2, uniqueExecutor.get());
        CHECK_RET(emptyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(emptyOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，获取计算过程中需要使用的workspace大小
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* castOut = nullptr;
    if (std::abs(beta->ToFloat() - 0.0f) <= std::numeric_limits<float>::epsilon()) {
        bool isBaddbmm = true;
        auto bmmOut = ExecBmmOp(batch1, batch2, out, cubeMathType, uniqueExecutor.get(), isBaddbmm);
        CHECK_RET(bmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 做bmmOut和alpha的Muls操作
        auto mulOut = l0op::Muls(bmmOut, alpha->ToFloat(), uniqueExecutor.get());
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果转换成输出out的数据类型
        castOut = l0op::Cast(mulOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // beta * self
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto mulOut = l0op::Muls(selfContiguous, beta->ToFloat(), uniqueExecutor.get());
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        bool isBaddbmm = true;
        auto bmmOut = ExecBmmOp(batch1, batch2, out, cubeMathType, uniqueExecutor.get(), isBaddbmm);
        CHECK_RET(bmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // Add算子需要对两个输入做隐式数据类型转换，根据具体算子语义按需调用
        auto promoteTypeAdd = op::PromoteType(mulOut->GetDataType(), bmmOut->GetDataType());

        // 将输入的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        auto mulOutCasted = l0op::Cast(mulOut, promoteTypeAdd, uniqueExecutor.get());
        CHECK_RET(mulOutCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入batch1的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        auto bmmOutCasted = l0op::Cast(bmmOut, promoteTypeAdd, uniqueExecutor.get());
        CHECK_RET(bmmOutCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 进行Add计算
        const aclTensor* addOut = nullptr;
        if (std::abs(alpha->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) {
            addOut = l0op::Add(mulOutCasted, bmmOutCasted, uniqueExecutor.get());
        } else {
            addOut = l0op::Axpy(mulOutCasted, bmmOutCasted, alpha->ToFloat(), uniqueExecutor.get());
        }
        CHECK_RET(addOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果转换成输出out的数据类型
        castOut = l0op::Cast(addOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBaddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBaddbmm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnBaddbmmGetWorkspaceSize(
        selfRef, batch1, batch2, beta, alpha, out, cubeMathType, workspaceSize, executor);
}

aclnnStatus aclnnInplaceBaddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceBaddbmm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif