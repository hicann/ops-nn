/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "group_norm_silu_quant.h"      
#include "aclnn_kernels/contiguous.h"
#include "level0/ones_like.h"
#include "level0/zero_op.h"
#include "aclnn_kernels/cast.h"
#include "level0/fill.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "aclnn_group_norm_silu_quant.h"
#include "aclnn/aclnn_base.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> QUANTSCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static const int64_t ONE_DIM = 1;

static bool CheckNotNull(
    const aclTensor* self, const aclTensor* quantScale, const aclTensor* out, const aclTensor* meanOut, const aclTensor* rstdOut)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(quantScale, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(meanOut, return false);
    OP_CHECK_NULL(rstdOut, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, const aclTensor* quantScale, const aclTensor* out,
    const aclTensor* meanOut, const aclTensor* rstdOut)
{
    // 检查self的数据类型是否在支持列表内
    auto supportList = ASCEND910B_DTYPE_SUPPORT_LIST;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    if (gamma != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(gamma, supportList, return false);
    }
    if (beta != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(beta, supportList, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(quantScale, QUANTSCALE_DTYPE_SUPPORT_LIST, return false);
    // 检查self，gamma, beta的数据类型是否一致
    if (gamma != nullptr && beta != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(gamma, self->GetDataType(), return false);
        OP_CHECK_DTYPE_NOT_MATCH(gamma, beta->GetDataType(), return false);
    }
    // 检查out是否为int8
    OP_CHECK_DTYPE_NOT_MATCH(out, op::DataType::DT_INT8, return false);
    // 检查meanOut, rstdOut的数据类型是否与self一致
    OP_CHECK_DTYPE_NOT_MATCH(meanOut, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(rstdOut, self->GetDataType(), return false);
    return true;
}

static bool CheckAttr(const aclTensor* self, int64_t group, double eps, bool activateSilu)
{
    // 检查group是否大于0
    OP_CHECK(
    group > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected group to be greater than 0, got %ld.", group),
    return false);
    OP_CHECK(
        self->GetViewShape()[0] > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected N to be greater than 0, got %ld.", self->GetViewShape()[0]),
        return false);
    OP_CHECK(
        self->GetViewShape()[1] > 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected C to be greater than 0, got %ld.", self->GetViewShape()[1]),
        return false);
    // 检查eps是否大于0
    OP_CHECK(
        eps > 0.0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected eps to be greater than 0, got %lf.", eps), return false);
    // 检查activateSilu是否为true
    OP_CHECK(
        activateSilu == true, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected activateSilu is TRUE"), return false);
    // 检查C是否能被group整除
    OP_CHECK(
        self->GetViewShape()[1] % group == 0,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected number of channels in input to be divisible by group, "
            "but got input of shape %s and group=%ld.",
            op::ToString(self->GetViewShape()).GetString(), group),
        return false);
    return true;
}

static bool CheckShape(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, const aclTensor* quantScale, int64_t group, const aclTensor* out,
    const aclTensor* meanOut, const aclTensor* rstdOut)
{
    // 检查x维度是否大于等于2
    OP_CHECK_MIN_DIM(self, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    // 检查x维度是否小于等于8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    // 检查out的shape是否与self不同。
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

    // 检查meanOut与rstdOut的shape是否为(N, group)
    auto mvShape = op::Shape({self->GetViewShape()[0], group});
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(meanOut, mvShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(rstdOut, mvShape, return false);

    // 检查(gamma是否为空指针)或者(gamma维度是否为1且元素数量是否等于C)
    if (gamma != nullptr &&
        (gamma->GetViewShape().GetDimNum() != 1 || gamma->GetViewShape()[0] != self->GetViewShape()[1])) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected weight to be a vector of size equal to the number of channels in "
            "input, but got weight of shape %s and input of shape %s.",
            op::ToString(gamma->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    // 检查(beta是否为空指针)或者(beta维度是否为1且元素数量是否等于C)
    if (beta != nullptr &&
        (beta->GetViewShape().GetDimNum() != 1 || beta->GetViewShape()[0] != self->GetViewShape()[1])) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected bias to be a vector of size equal to the number of channels in "
            "input, but got weight of shape %s and input of shape %s.",
            op::ToString(beta->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    // 检查(quantScale维度是否为1且元素数量是否等于1或C)
    if ((quantScale->GetViewShape().GetDimNum() != 1 || (quantScale->GetViewShape()[0] != 1 &&
         quantScale->GetViewShape()[0] != self->GetViewShape()[1]))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected quantScale to be a vector of size equal to the number of channels in "
            "input, but got quantScale of shape %s and input of shape %s.",
            op::ToString(quantScale->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, const aclTensor* quantScale, int64_t group,
    double eps, bool activateSilu, aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, quantScale, out, meanOut, rstdOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, gamma, beta, quantScale, out, meanOut, rstdOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(self, gamma, beta, quantScale, group, out, meanOut, rstdOut), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查属性是否满足约束条件
    CHECK_RET(CheckAttr(self, group, eps, activateSilu), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(aclTensor* out, float val, aclOpExecutor* executor)
{
    FVector<int64_t> shape;
    size_t dimNum = out->GetViewShape().GetDimNum();
    for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = out->GetViewShape().GetDim(idx);
        shape.push_back(tmpVal);
    }
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());
    CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    CHECK_RET(valTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor* GetGamma(
    const aclTensor* self, const aclTensor* gamma, int64_t size, aclOpExecutor* executor)
{
    const aclTensor* gammaContiguous = nullptr;
    if (gamma) {
        gammaContiguous = l0op::Contiguous(gamma, executor);
    } else {
        auto gammaTensor = executor->AllocTensor(op::Shape({size}), self->GetDataType(), self->GetViewFormat());
        CHECK_RET(gammaTensor != nullptr, nullptr);
        gammaContiguous = l0op::OnesLike(gammaTensor, executor);
    }
    return gammaContiguous;
}

static const aclTensor* GetBeta(
    const aclTensor* self, const aclTensor* beta, int64_t size, aclOpExecutor* executor)
{
    const aclTensor* betaContiguous = nullptr;
    if (beta) {
        betaContiguous = l0op::Contiguous(beta, executor);
    } else {
        auto betaTensor = executor->AllocTensor(op::Shape({size}), self->GetDataType(), self->GetViewFormat());
        CHECK_RET(betaTensor != nullptr, nullptr);
        betaContiguous = l0op::ZerosLike(betaTensor, executor);
    }
    return betaContiguous;
}

static const aclTensor* GetQuantScale(
    const aclTensor* quantScale, aclOpExecutor* executor)
{
    //quantScale不应该为空，逻辑与gamma、beta不一样
    const aclTensor* quantScaleContiguous = nullptr;
    quantScaleContiguous = l0op::Contiguous(quantScale, executor);
    return quantScaleContiguous;
}

aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, const aclTensor* quantScale, int64_t group, double eps, bool activateSilu,
    aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGroupNormSiluQuant, DFX_IN(self, gamma, beta, quantScale, group, eps, activateSilu), DFX_OUT(out, meanOut, rstdOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, gamma, beta, quantScale, group, eps, activateSilu, out, meanOut, rstdOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空tensor支持
    if (self->IsEmpty()) {
        // meanOut在输入空Tensor情况下，输出0
        ret = FillScalar(meanOut, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        // rstdOut在输入空Tensor情况下，输出NAN
        ret = FillScalar(rstdOut, NAN, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self, gamma, beta转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gammaContiguous = GetGamma(selfContiguous, gamma, selfContiguous->GetViewShape()[1], uniqueExecutor.get());
    auto betaContiguous = GetBeta(selfContiguous, beta, selfContiguous->GetViewShape()[1], uniqueExecutor.get());
    auto quantScaleContiguous = GetQuantScale(quantScale, uniqueExecutor.get());
    CHECK_RET(gammaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(betaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(quantScaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用GroupNormSiluQuant计算
    auto result = l0op::GroupNormSiluQuant(
        selfContiguous, gammaContiguous, betaContiguous, quantScaleContiguous, group, static_cast<float>(eps), activateSilu,
        uniqueExecutor.get());

    auto y1 = std::get<0>(result);
    CHECK_RET(y1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mean = std::get<1>(result);
    CHECK_RET(mean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto rstd = std::get<2>(result);
    CHECK_RET(rstd != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto outViewCopyResult = l0op::ViewCopy(y1, out, uniqueExecutor.get());
    CHECK_RET(outViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
    auto meanOutViewCopyResult = l0op::ViewCopy(mean, meanOut, uniqueExecutor.get());
    CHECK_RET(meanOutViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出rstdOut上，rstdOut可能是非连续的tensor
    auto rstdOutViewCopyResult = l0op::ViewCopy(rstd, rstdOut, uniqueExecutor.get());
    CHECK_RET(rstdOutViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupNormSiluQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupNormSiluQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
