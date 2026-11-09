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
#include "aclnn_kernels/transpose.h"
#include "opdev/tensor_view_utils.h"
#include "fake_quant_common.h"
#include "aclnn_util.h"

using namespace op;
using namespace FakeQuantCommon;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> ZERO_POINT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint,
                         const aclTensor* out, const aclTensor* mask)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(zeroPoint, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(mask, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint,
                            const aclTensor* out, const aclTensor* mask)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, DTYPE_SUPPORT_LIST, return false);
    // out is an output contract too: docs only allow FLOAT16/FLOAT32. CanCast alone is not
    // enough because float->int / float->bool are considered castable and silently truncate.
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(zeroPoint, ZERO_POINT_DTYPE_SUPPORT_LIST, return false);
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

static bool CheckShape(const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, int64_t axis,
                       const aclTensor* out, const aclTensor* mask)
{
    auto inputShape = self->GetViewShape();
    auto scaleShape = scale->GetViewShape();
    auto zeroPointShape = zeroPoint->GetViewShape();

    // 文档声明 self/out/mask 支持 0-8 维（ND）：越界返回 161002（P1-5）
    if (self->GetViewShape().GetDimNum() > 8) { // ND 格式 0-8 维
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension of self should be in range [0, 8], but got %zu",
                self->GetViewShape().GetDimNum());
        return false;
    }

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
    if (selfDimNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "self of FakeQuantPerChannelAffineCachemask must have rank >= 1; rank-0 is not supported.");
        return false;
    }
    if (axis < -selfDimNum || axis >= selfDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Dimension out of range (expected to be in range of [-%ld, %ld],"
                "but got %ld)",
                selfDimNum, selfDimNum - 1, axis);
        return false;
    }

    int64_t positiveDim = MakeWrapDim(axis, selfDimNum);
    // 0 维 self 无 GetDim(0) 可读，视为单元素，要求 scale/zeroPoint 长度为 1
    int64_t expectSize = (selfDimNum == 0) ? 1 : inputShape.GetDim(positiveDim);
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
    constexpr int64_t QUANT_BOUND_LIMIT = 16777215; // 2^24 - 1，与 Host/tiling 同口径
    if (quantMin < -QUANT_BOUND_LIMIT || quantMax > QUANT_BOUND_LIMIT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quant bounds out of representable range [-%ld, %ld].", QUANT_BOUND_LIMIT,
                QUANT_BOUND_LIMIT);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, int64_t axis,
                               int64_t quantMin, int64_t quantMax, const aclTensor* out, const aclTensor* mask)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, scale, zeroPoint, out, mask), ACLNN_ERR_PARAM_NULLPTR);

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
    // 0 维（标量）self：无维度可置换（axis 合法域已由 CheckShape 限定为 {0,-1}），
    // 返回空置换数组；原实现对空 vector 取下标 perm[0] 属未定义行为
    if (dimNum == 0) {
        return executor->AllocIntArray(nullptr, 0);
    }
    int64_t positiveDim = MakeWrapDim(axis, dimNum);
    std::vector<int64_t> perm(dimNum, 0);
    for (int64_t i = 0; i < dimNum; i++) {
        perm[i] = i;
    }
    std::swap(perm[positiveDim], perm[0]);
    return executor->AllocIntArray(perm.data(), dimNum);
}

static std::pair<aclTensor*, aclTensor*> RunFakeQuantDispatch(const aclTensor* selfContiguous,
                                                              const aclTensor* scaleContiguous,
                                                              const aclTensor* zeroPointContiguous, int64_t axis,
                                                              int64_t quantMin, int64_t quantMax, aclOpExecutor* exec)
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        int64_t selfDimNum = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
        int64_t positiveAxis = MakeWrapDim(axis, selfDimNum);
        auto result = l0op::FakeQuantAffineCachemask(selfContiguous, scaleContiguous, zeroPointContiguous, positiveAxis,
                                                     quantMin, quantMax, exec);
        return {std::get<0>(result), std::get<1>(result)};
    }
    // 0 维（标量）self 无维度可置换：axis∈{0,-1} 均等价于 axis=0，直接走无
    // Transpose 路径（避免空置换数组进入 Transpose 及其回转）
    const int64_t selfDimNum = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
    if (selfDimNum == 0) {
        auto result = l0op::FakeQuantAffineCachemask(selfContiguous, scaleContiguous, zeroPointContiguous, 0, quantMin,
                                                     quantMax, exec);
        return {std::get<0>(result), std::get<1>(result)};
    }
    if (axis != 0) {
        aclIntArray* axes = GetTransposeArray(selfContiguous, axis, exec);
        if (axes == nullptr)
            return {nullptr, nullptr};

        auto selfTranspose = const_cast<aclTensor*>(l0op::Transpose(selfContiguous, axes, exec));
        if (selfTranspose == nullptr)
            return {nullptr, nullptr};

        auto result = l0op::FakeQuantAffineCachemask(selfTranspose, scaleContiguous, zeroPointContiguous, 0, quantMin,
                                                     quantMax, exec);
        auto fakeOut = std::get<0>(result);
        auto maskOut = std::get<1>(result);
        if (fakeOut == nullptr || maskOut == nullptr)
            return {nullptr, nullptr};

        auto outT = const_cast<aclTensor*>(l0op::Transpose(fakeOut, axes, exec));
        auto maskT = const_cast<aclTensor*>(l0op::Transpose(maskOut, axes, exec));
        return {outT, maskT};
    }
    auto result = l0op::FakeQuantAffineCachemask(selfContiguous, scaleContiguous, zeroPointContiguous, 0, quantMin,
                                                 quantMax, exec);
    return {std::get<0>(result), std::get<1>(result)};
}

aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(const aclTensor* self, const aclTensor* scale,
                                                                    const aclTensor* zeroPoint, int64_t axis,
                                                                    int64_t quantMin, int64_t quantMax, aclTensor* out,
                                                                    aclTensor* mask, uint64_t* workspaceSize,
                                                                    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnFakeQuantPerChannelAffineCachemask, DFX_IN(self, scale, zeroPoint, axis, quantMin, quantMax),
                   DFX_OUT(out, mask));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, scale, zeroPoint, axis, quantMin, quantMax, out, mask);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || out->IsEmpty() || mask->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const auto& [selfContiguous, scaleContiguous, zeroPointContiguous] = GetContiguousInput(self, scale, zeroPoint,
                                                                                            uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr && scaleContiguous != nullptr && zeroPointContiguous != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto [fakeQuantOut, fakeQuantMask] = RunFakeQuantDispatch(selfContiguous, scaleContiguous, zeroPointContiguous,
                                                              axis, quantMin, quantMax, uniqueExecutor.get());
    return FinalizeOutput(fakeQuantOut, fakeQuantMask, out, mask, workspaceSize, executor, uniqueExecutor);
}

aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFakeQuantPerChannelAffineCachemask);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
