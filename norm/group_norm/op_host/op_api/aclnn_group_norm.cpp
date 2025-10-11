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
 * \file aclnn_group_norm.cpp
 * \brief
 */

#include "norm/group_norm_silu/op_host/op_api/group_norm_silu.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/ones_like.h"
#include "level0/zero_op.h"
#include "aclnn_kernels/reshape.h"
#include "level0/add.h"
#include "level0/rsqrt.h"
#include "level0/fill.h"
#include "group_norm.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/level2_base.h"
#include "aclnn_group_norm.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static const int64_t ONE_DIM = 1;
static const int64_t Y_INDEX = 0;
static const int64_t MEAN_INDEX = 1;
static const int64_t RSTD_INDEX = 2;
static const int64_t TWO_DIM = 2;

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        socVersion == SocVersion::ASCEND910_95) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, const aclTensor* out,
    const aclTensor* meanOut, const aclTensor* rstdOut)
{
    // 检查self的数据类型是否在支持列表内
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查gamma, beta, out, meanOut, rstdOut的数据类型是否与out一致
    if (gamma != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(gamma, self->GetDataType(), return false);
    }
    if (beta != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(beta, self->GetDataType(), return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(meanOut, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(rstdOut, self->GetDataType(), return false);
    return true;
}

static int64_t GetTensorNumel(const aclTensor* x, size_t startIdx)
{
    size_t xShapeDim = x->GetViewShape().GetDimNum();
    int64_t xShapeSize = 1;
    for (size_t i = startIdx; i < xShapeDim; i++) {
        xShapeSize *= x->GetViewShape().GetDim(i);
    }
    return xShapeSize;
}

static bool CheckAttr(const aclTensor* self, int64_t C, int64_t group, double eps)
{
    // 检查group是否大于0
    OP_CHECK(
        group > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected group to be greater than 0, got %ld.", group),
        return false);
    // 检查C是否能被group整除
    OP_CHECK(
        C % group == 0,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected number of channels in input to be divisible by group, "
            "but got input of shape %s and group=%ld.",
            op::ToString(self->GetViewShape()).GetString(), group),
        return false);
    // 检查eps是否大于0
    OP_CHECK(
        eps > 0.0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected eps to be greater than 0, got %lf.", eps), return false);
    return true;
}

static bool CheckShape(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t N, int64_t C, int64_t HxW,
    int64_t group, const aclTensor* out, const aclTensor* meanOut, const aclTensor* rstdOut)
{
    // 检查x维度是否大于等于2
    OP_CHECK_MIN_DIM(self, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    // 检查x维度是否小于等于8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    // 检查out的shape是否与self不同。
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

    // 检查meanOut与rstdOut的shape是否为(N, group)
    auto mvShape = op::Shape({N, group});
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(meanOut, mvShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(rstdOut, mvShape, return false);

    // 检查self的N维度元素数量是否等于入参N
    OP_CHECK(
        self->GetViewShape()[0] == N,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected self.shape[0] == N to be true, but got false."), return false);

    // 检查self的C维度元素数量是否等于入参C
    OP_CHECK(
        self->GetViewShape()[1] == C,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected self.shape[1] == C to be true, but got false."), return false);

    // 检查self除N, C维度外的元素数量是否等于HxW
    OP_CHECK(
        GetTensorNumel(self, TWO_DIM) == HxW,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected the size of self outside the N and C dimensions == HxW to be "
            "true, but got false."),
        return false);

    // 检查self的元素数量是否等于 N * C * HxW
    int64_t selfShapeSize = GetTensorNumel(self, 0);
    OP_CHECK(
        selfShapeSize == N * C * HxW,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected self.numel() == N * C * HxW to be true, but got false."),
        return false);

    // 检查(gamma是否为空指针)或者(gamma维度是否为1且元素数量是否等于C)
    if (gamma != nullptr && (gamma->GetViewShape().GetDimNum() != 1 || gamma->GetViewShape()[0] != C)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected gamma to be a vector of size equal to the number of channels in "
            "input, but got gamma of shape %s and input of shape %s.",
            op::ToString(gamma->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    // 检查(beta是否为空指针)或者(beta维度是否为1且元素数量是否等于C)
    if (beta != nullptr && (beta->GetViewShape().GetDimNum() != 1 || beta->GetViewShape()[0] != C)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected beta to be a vector of size equal to the number of channels in "
            "input, but got beta of shape %s and input of shape %s.",
            op::ToString(beta->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of self and out should be equal, but got self [%s], out [%s]",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    auto selfFormat = self->GetStorageFormat();
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(selfFormat) || selfFormat == Format::FORMAT_NHWC || selfFormat == Format::FORMAT_HWCN ||
        selfFormat == Format::FORMAT_NDHWC) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCL、NCHW、NCDHW.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t N, int64_t C, int64_t HxW,
    int64_t group, double eps, aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull4Tensor(self, out, meanOut, rstdOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, gamma, beta, out, meanOut, rstdOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查属性是否满足约束条件
    CHECK_RET(CheckAttr(self, C, group, eps), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否支持
    CHECK_RET(CheckShape(self, gamma, beta, N, C, HxW, group, out, meanOut, rstdOut), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetGamma(const aclTensor* self, const aclTensor* gamma, int64_t size, aclOpExecutor* executor)
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

static const aclTensor* GetBeta(const aclTensor* self, const aclTensor* beta, int64_t size, aclOpExecutor* executor)
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

static std::tuple<aclTensor*, aclTensor*, aclTensor*> GroupNormOutCastProcess(
    const aclTensor* x, const aclTensor* y, const aclTensor* mean, const aclTensor* variance, aclOpExecutor* executor)
{
    aclTensor* yCast = const_cast<aclTensor*>(y);
    aclTensor* meanCast = const_cast<aclTensor*>(mean);
    aclTensor* varianceCast = const_cast<aclTensor*>(variance);
    if (x->GetDataType() == op::DataType::DT_FLOAT16 &&
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
        yCast = const_cast<aclTensor*>(l0op::Cast(y, op::DataType::DT_FLOAT16, executor));
        meanCast = const_cast<aclTensor*>(l0op::Cast(mean, op::DataType::DT_FLOAT16, executor));
        varianceCast = const_cast<aclTensor*>(l0op::Cast(variance, op::DataType::DT_FLOAT16, executor));
    }

    return std::tie(yCast, meanCast, varianceCast);
}

static const aclTensor* ReshapeProcess(const aclTensor* self, int64_t N, int64_t group, aclOpExecutor* executor)
{
    int64_t shapeValue[2] = {N, group};
    aclIntArray* shapeArray = executor->AllocIntArray(shapeValue, 2);
    CHECK_RET(shapeArray != nullptr, nullptr);
    auto selfReshape = l0op::Reshape(self, shapeArray, executor);
    return selfReshape;
}

static const aclTensor* RstdProcess(
    const aclTensor* variance, int64_t N, int64_t group, double eps, aclOpExecutor* executor)
{
    // 计算rstd
    FVector<float> valVector = {static_cast<float>(eps)};
    auto epsTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), variance->GetDataType());
    CHECK_RET(epsTensor != nullptr, nullptr);
    auto addResult = l0op::Add(variance, epsTensor, executor);
    CHECK_RET(addResult != nullptr, nullptr);
    auto rsqrtResult = l0op::Rsqrt(addResult, executor);
    CHECK_RET(rsqrtResult != nullptr, nullptr);
    auto rsqrtReshape = ReshapeProcess(rsqrtResult, N, group, executor);
    return rsqrtReshape;
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

aclnnStatus aclnnGroupNormGetWorkspaceSize(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta, int64_t N, int64_t C, int64_t HxW,
    int64_t group, double eps, aclTensor* out, aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGroupNorm, DFX_IN(self, gamma, beta, N, C, HxW, group, eps), DFX_OUT(out, meanOut, rstdOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, gamma, beta, N, C, HxW, group, eps, out, meanOut, rstdOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空tensor支持
    if (self->IsEmpty()) {
        // 空Tensor情况下，meanOut输出0, rstdOut输出NAN
        ret = FillScalar(meanOut, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = FillScalar(rstdOut, NAN, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self, gamma, beta转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gammaContiguous = GetGamma(selfContiguous, gamma, C, uniqueExecutor.get());
    CHECK_RET(gammaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto betaContiguous = GetBeta(selfContiguous, beta, C, uniqueExecutor.get());
    CHECK_RET(betaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 根据芯片确定调用GroupNorm，还是GroupNormSilu
    std::tuple<aclTensor*, aclTensor*, aclTensor*> result;
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    bool isNormSocLists =
        (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
         socVersion == SocVersion::ASCEND310P || socVersion == SocVersion::ASCEND910_95);
    if (isNormSocLists) {
        result = l0op::GroupNormSilu(
            selfContiguous, gammaContiguous, betaContiguous, group, static_cast<float>(eps), false,
            uniqueExecutor.get());
    } else {
        const aclTensor* xCast = selfContiguous;
        const aclTensor* gammaCast = gammaContiguous;
        const aclTensor* betaCast = betaContiguous;
        if (selfContiguous->GetDataType() == op::DataType::DT_FLOAT16 && socVersion == SocVersion::ASCEND910) {
            xCast = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(xCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            gammaCast = l0op::Cast(gammaContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(gammaCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            betaCast = l0op::Cast(betaContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
            CHECK_RET(betaCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        const bool isTraining = true;
        auto resultGN = l0op::GroupNorm(
            xCast, gammaCast, betaCast, N, group, static_cast<float>(eps), isTraining, uniqueExecutor.get());
        auto yResult = std::get<Y_INDEX>(resultGN);
        auto meanResult = std::get<MEAN_INDEX>(resultGN);
        auto varianceResult = std::get<RSTD_INDEX>(resultGN);
        CHECK_RET(
            (yResult != nullptr) && (meanResult != nullptr) && (varianceResult != nullptr), ACLNN_ERR_INNER_NULLPTR);

        result = GroupNormOutCastProcess(selfContiguous, yResult, meanResult, varianceResult, uniqueExecutor.get());
    }

    auto y = std::get<0>(result);
    CHECK_RET(y != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mean = std::get<1>(result);
    CHECK_RET(mean != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto variance = std::get<2>(result);
    CHECK_RET(variance != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto outViewCopyResult = l0op::ViewCopy(y, out, uniqueExecutor.get());
    CHECK_RET(outViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (isNormSocLists) {
        // 固定写法，将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
        auto meanOutViewCopyResult = l0op::ViewCopy(mean, meanOut, uniqueExecutor.get());
        CHECK_RET(meanOutViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出rstdOut上，rstdOut可能是非连续的tensor
        auto rstdOutViewCopyResult = l0op::ViewCopy(variance, rstdOut, uniqueExecutor.get());
        CHECK_RET(rstdOutViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // mean reshpe
        auto meanReshape = ReshapeProcess(mean, N, group, uniqueExecutor.get());
        CHECK_RET(meanReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 计算rstd
        auto rsqrtResult = RstdProcess(variance, N, group, eps, uniqueExecutor.get());
        CHECK_RET(rsqrtResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出meanOut上，meanOut可能是非连续的tensor
        auto meanOutViewCopyResult = l0op::ViewCopy(meanReshape, meanOut, uniqueExecutor.get());
        CHECK_RET(meanOutViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出rstdOut上，rstdOut可能是非连续的tensor
        auto rstdOutViewCopyResult = l0op::ViewCopy(rsqrtResult, rstdOut, uniqueExecutor.get());
        CHECK_RET(rstdOutViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupNorm);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
