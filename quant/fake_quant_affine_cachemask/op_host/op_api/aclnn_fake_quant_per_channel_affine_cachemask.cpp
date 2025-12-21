/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_fake_quant_per_channel_affine_cachemask.cpp
 * \brief
 */

#include "aclnn_fake_quant_per_channel_affine_cachemask.h"
#include "fake_quant_affine_cachemask.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static bool CheckNotNull(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, const aclTensor* out,
    const aclTensor* mask)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(zeroPoint, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(mask, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, const aclTensor* out,
    const aclTensor* mask)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(zeroPoint, op::DataType::DT_INT32, return false);

    op::DataType promoteType = op::PromoteType(self->GetDataType(), scale->GetDataType());
    if (!CanCast(DataType(out->GetDataType()), promoteType)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Promote dtype %s can't be cast to the desired output type %s.",
            op::ToString(promoteType).GetString(), op::ToString(DataType(out->GetDataType())).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_MATCH(mask, op::DataType::DT_BOOL, return false);
    return true;
}

static inline int64_t MakeWrapDim(int64_t dim, int64_t dimPostExpr)
{
    // 支持0维tensor
    if (dimPostExpr <= 0) {
        dimPostExpr = 1;
    }
    if (dim < 0) {
        dim += dimPostExpr;
    }
    return dim;
}

static bool CheckShape(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, int64_t axis, const aclTensor* out,
    const aclTensor* mask)
{
    auto inputShape = self->GetViewShape();
    auto scaleShape = scale->GetViewShape();
    auto zeroPointShape = zeroPoint->GetViewShape();

    auto scaleDimNum = scaleShape.GetDimNum();
    auto zeroPointDimNum = zeroPointShape.GetDimNum();
    if (scaleDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale should be a 1-D tensor");
        return false;
    }
    if (zeroPointDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "zero point should be a 1-D tensor");
        return false;
    }
    if (scaleShape.GetDim(0) != zeroPointShape.GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale and zeroPoint need to have the same dimensions");
        return false;
    }

    int64_t selfDimNum = static_cast<int64_t>(inputShape.GetDimNum());
    if (selfDimNum == 0 && axis != 0 && axis != -1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-1, 0], but got %ld)", axis);
        return false;
    } else if (selfDimNum > 0 && (axis < -selfDimNum || axis >= selfDimNum)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dimension out of range (expected to be in range of [-%ld, %ld],"
            "but got %ld)",
            selfDimNum, selfDimNum - 1, axis);
        return false;
    }

    int64_t positiveDim = MakeWrapDim(axis, selfDimNum);
    auto expectSize = inputShape.GetDim(positiveDim);
    if (expectSize != scaleShape.GetDim(0) || expectSize != zeroPointShape.GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimensions of scale and zero-point are not consistant with input tensor");
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, mask, return false);
    return true;
}

static bool CheckQuantValue(int64_t quantMin, int64_t quantMax)
{
    if (quantMin > quantMax) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "`quantMin` should be less than or equal to `quantMax`.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, int64_t axis, int64_t quantMin,
    int64_t quantMax, const aclTensor* out, const aclTensor* mask)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, scale, zeroPoint, out, mask), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, scale, zeroPoint, out, mask), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入shape
    CHECK_RET(CheckShape(self, scale, zeroPoint, axis, out, mask), ACLNN_ERR_PARAM_INVALID);

    // 4. 校验quant_min和quant_max的大小是否合法
    CHECK_RET(CheckQuantValue(quantMin, quantMax), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclIntArray* GetTransposeArray(const aclTensor* self, int64_t axis, aclOpExecutor* executor)
{
    int64_t dimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    int64_t positiveDim = MakeWrapDim(axis, dimNum);
    std::vector<int64_t> perm(dimNum, 0);
    for (int64_t i = 0; i < dimNum; i++) {
        perm[i] = i;
    }
    std::swap(perm[positiveDim], perm[0]);
    return executor->AllocIntArray(perm.data(), dimNum);
}

aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(
    const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, int64_t axis, int64_t quantMin,
    int64_t quantMax, aclTensor* out, aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnFakeQuantPerChannelAffineCachemask, DFX_IN(self, scale, zeroPoint, axis, quantMin, quantMax),
        DFX_OUT(out, mask));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, scale, zeroPoint, axis, quantMin, quantMax, out, mask);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || out->IsEmpty() || mask->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto promoteType = op::PromoteType(self->GetDataType(), scale->GetDataType());
    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto scaleCasted = l0op::Cast(scale, promoteType, uniqueExecutor.get());
    CHECK_RET(scaleCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclTensor* fakeQuantOut = nullptr;
    aclTensor* fakeQuantMask = nullptr;
    if (axis != 0) {
        aclIntArray* axes = GetTransposeArray(selfCasted, axis, uniqueExecutor.get());
        CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 对self进行transpose
        auto selfTranspose = const_cast<aclTensor*>(l0op::Transpose(selfCasted, axes, uniqueExecutor.get()));
        CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto result = l0op::FakeQuantAffineCachemask(
            selfTranspose, scaleCasted, zeroPoint, quantMin, quantMax, uniqueExecutor.get());
        auto fakeOut = std::get<0>(result);
        auto maskOut = std::get<1>(result);
        CHECK_RET(fakeOut != nullptr && maskOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将结果values_transpose进行transpose，转换成正确的shape
        fakeQuantOut = const_cast<aclTensor*>(l0op::Transpose(fakeOut, axes, uniqueExecutor.get()));

        // 将结果indices_transpose进行transpose，转换成正确的shape
        fakeQuantMask = const_cast<aclTensor*>(l0op::Transpose(maskOut, axes, uniqueExecutor.get()));
    } else {
        auto result = l0op::FakeQuantAffineCachemask(
            selfCasted, scaleCasted, zeroPoint, quantMin, quantMax, uniqueExecutor.get());
        fakeQuantOut = std::get<0>(result);
        fakeQuantMask = std::get<1>(result);
    }
    CHECK_RET(fakeQuantOut != nullptr && fakeQuantMask != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto fakeCastOut = l0op::Cast(fakeQuantOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(fakeCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto fakeViewCopyResult = l0op::ViewCopy(fakeCastOut, out, uniqueExecutor.get());
    CHECK_RET(fakeViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto maskViewCopyResult = l0op::ViewCopy(fakeQuantMask, mask, uniqueExecutor.get());
    CHECK_RET(maskViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFakeQuantPerChannelAffineCachemask);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
