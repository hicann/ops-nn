/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adaptive_avg_pool2d_assist_matrix.h"
#include "adaptive_avg_pool3d.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/mul.h"
#include "level0/reduce_mean.h"
#include "level0/shape_op.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_adaptive_avg_pool2d.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportListOrigin = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> dtypeSupportListAscend910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const int64_t leftMatrixIndex = 0;
static const int64_t rightMatrixIndex = 1;
static const int64_t mulMatrixIndex = 2;
static const int64_t chwShapeSize = 3;
static const int64_t nchwShapeSize = 4;
static const int64_t ncdhwShapeSize = 5;
static const int64_t assistMatrixNums = 3;
static const int64_t outputSizeLimit = 2;
static const int64_t dimD = 2;
static const int64_t INDEX_DIM0 = 0;
static const int64_t INDEX_DIM1 = 1;
static const int64_t INDEX_DIM2 = 2;
static const int64_t INDEX_DIM3 = 3;
static const int64_t INDEX_DIM4 = 4;

static bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool IsSocVersion910B_310P()
{
    if ((op::GetCurrentPlatformInfo().GetSocVersion() >= op::SocVersion::ASCEND910B &&
         op::GetCurrentPlatformInfo().GetSocVersion() <= op::SocVersion::ASCEND910E) ||
        op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P) {
        return true;
    }
    return false;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return dtypeSupportListAscend910B;
    } else {
        return dtypeSupportListOrigin;
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);
    // 检查out和self数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

// 校验输入Tensor Shape, AdaptiveAvgPool2D支持3维或者4维
static bool CheckInputOutputShape(const aclTensor* self, const aclTensor* out)
{
    auto inputShape = self->GetViewShape();
    auto outputShape = out->GetViewShape();
    size_t inputDimNum = inputShape.GetDimNum();
    size_t outputDimNum = outputShape.GetDimNum();
    if (inputDimNum != chwShapeSize && inputDimNum != nchwShapeSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dims %zu should be in 3 or 4.", inputDimNum);
        return false;
    }
    if (inputDimNum != outputDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out dims %zu should equal to self dims %zu.", outputDimNum, inputDimNum);
        return false;
    }
    // nc_dim index offset is 2
    for (uint64_t i = 0; i < outputDimNum - 2; i++) {
        if (outputShape[i] != inputShape[i]) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Out_shape[%lu]: %ld should equal to input_shape[%lu]: %ld", i, outputShape[i],
                i, inputShape[i]);
            return false;
        }
    }
    return true;
}

// 校验输入输出格式， AdaptiveAvgPool2D只支持NCHW,NCL
static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and out should be equal, input [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only don't support private format.");
        return false;
    }

    if (self->GetStorageFormat() != Format::FORMAT_NCHW && self->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCHW or NCL.");
        return false;
    }

    if ((self->GetStorageFormat() == Format::FORMAT_NCL && self->GetViewShape().GetDimNum() != chwShapeSize) ||
        (self->GetStorageFormat() == Format::FORMAT_NCHW && self->GetViewShape().GetDimNum() != nchwShapeSize)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self is %zuD tensor but the format is [%s].", self->GetViewShape().GetDimNum(),
            op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    if ((out->GetStorageFormat() == Format::FORMAT_NCL && out->GetViewShape().GetDimNum() != chwShapeSize) ||
        (out->GetStorageFormat() == Format::FORMAT_NCHW && out->GetViewShape().GetDimNum() != nchwShapeSize)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "out is %zuD tensor but the format is [%s].", out->GetViewShape().GetDimNum(),
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

static bool CheckOutputSize(const aclIntArray* outputSize, const aclTensor* out)
{
    uint64_t size = outputSize->Size();
    if (size != outputSizeLimit) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "OutputSize length should be 2.");
        return false;
    }
    auto outputShape = out->GetViewShape();
    size_t outputDimNum = outputShape.GetDimNum();
    size_t offset = outputDimNum == chwShapeSize ? 1 : 2;
    for (size_t i = 0; i < size; ++i) {
        if ((*outputSize)[i] != outputShape[i + offset]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputSize value should match the outputShape value.");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 校验输入tensor维度/校验是否为空tensor
    CHECK_RET(CheckInputOutputShape(self, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入输出数据类型
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);
    // 5. 校验outputSize是否合法
    CHECK_RET(CheckOutputSize(outputSize, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclTensor* Reformat(const aclTensor* input, op::Format format)
{
    aclTensor* inputCast = const_cast<aclTensor*>(input);
    inputCast->SetStorageFormat(format);
    inputCast->SetViewFormat(format);
    inputCast->SetOriginalFormat(format);
    return inputCast;
}

// at::mean_out(result, self, {self.dim() - 2, self.dim() - 1}, true);
// NCHW ---> NC11
// Inputsize 为[1, 1], 计算图如下:
// self ---> Contiguous ---> ReduceMean ---> ViewCopy ---> out
static aclnnStatus DoReduceMean(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    int64_t dimNum = self->GetViewShape().GetDimNum();
    int64_t reduceAxes[] = {dimNum - 2, dimNum - 1};
    aclIntArray* axes = executor->AllocIntArray(reduceAxes, 2);

    // ReduceMean支持ND,需要先设置下format
    aclTensor* input = const_cast<aclTensor*>(self);
    input = Reformat(self, Format::FORMAT_ND);

    // 固定写法，将输出self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(input, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto reduceMeanResult = l0op::ReduceMean(selfContiguous, axes, true, executor);
    CHECK_RET(reduceMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto format = dimNum == chwShapeSize ? Format::FORMAT_NCL : Format::FORMAT_NCHW;
    reduceMeanResult = Reformat(reduceMeanResult, format);
    // 调用ViewCopy将ReduceMean的结果拷贝到out上
    auto viewCopyResult = l0op::ViewCopy(reduceMeanResult, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// InputSize不为[1, 1], AdaptiveAvgPool2D走融合算子流程，会拆分成多个算子进行实现
static aclnnStatus DoAdaptiveAvgPool2D(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, aclOpExecutor* executor)
{
    if (IsSocVersion910B_310P()) {
        // 将2d参数转换为3d可以使用的参数
        int64_t newOutputSizeData[] = {1, (*outputSize)[0], (*outputSize)[1]};
        aclIntArray* newOutputSize = executor->AllocIntArray(newOutputSizeData, 3);
        auto inputContiguous = l0op::Contiguous(self, executor);
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // reshape(将3d的input转变为4d并增加D=1维)
        op::Shape inputContiguousShape = inputContiguous->GetViewShape();
        auto inputDimNum = inputContiguousShape.GetDimNum();
        std::vector<int64_t> valueShape(ncdhwShapeSize);
        valueShape[0] = inputDimNum == nchwShapeSize ? inputContiguousShape.GetDim(0) : 1;
        valueShape[1] = inputDimNum == nchwShapeSize ? inputContiguousShape.GetDim(1) : inputContiguousShape.GetDim(0);
        valueShape[dimD] = 1;
        for (int64_t i = inputDimNum - outputSizeLimit; i < inputDimNum; i++) {
            valueShape[ncdhwShapeSize - inputDimNum + i] = inputContiguousShape.GetDim(i);
        }
        auto reshapeShape = executor->AllocIntArray(valueShape.data(), ncdhwShapeSize);
        CHECK_RET(reshapeShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputContiguousReshape = l0op::Reshape(inputContiguous, reshapeShape, executor);
        CHECK_RET(inputContiguousReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将C维度转置到最后面
        std::vector<int64_t> valuePerm{INDEX_DIM0, INDEX_DIM2, INDEX_DIM3, INDEX_DIM4, INDEX_DIM1};
        auto perm = executor->AllocIntArray(valuePerm.data(), ncdhwShapeSize);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputReshapeTran = l0op::Transpose(inputContiguousReshape, perm, executor);
        CHECK_RET(inputReshapeTran != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto poolResult = l0op::AdaptiveAvgPool3d(inputReshapeTran, newOutputSize, executor);
        CHECK_RET(poolResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将C维度转置回原位置
        std::vector<int64_t> valuePermRes{INDEX_DIM0, INDEX_DIM4, INDEX_DIM1, INDEX_DIM2, INDEX_DIM3};
        auto permRes = executor->AllocIntArray(valuePermRes.data(), ncdhwShapeSize);
        CHECK_RET(permRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto resTran = l0op::Transpose(poolResult, permRes, executor);
        CHECK_RET(resTran != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto resShapeVector = op::ToShapeVector(out->GetViewShape());
        aclIntArray* resShapeArray = executor->AllocIntArray(resShapeVector.data(), resShapeVector.size());
        CHECK_RET(resShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto resTranReshape = l0op::Reshape(resTran, resShapeArray, executor);
        CHECK_RET(resTranReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(resTranReshape, out, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }
    aclTensor* input = const_cast<aclTensor*>(self);
    input = Reformat(self, Format::FORMAT_ND);
    auto inputContiguous = l0op::Contiguous(input, executor);
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0 Shape接口,将输入Tensor的Shape信息存储到新的Tensor中
    auto shapeResult = l0op::Shape_op(inputContiguous, executor);
    CHECK_RET(shapeResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::array<aclTensor*, assistMatrixNums> assistMatrixTensor =
        l0op::AdaptiveAvgPool2dAssistMatrix(shapeResult, inputContiguous, outputSize, executor);
    CHECK_RET(assistMatrixTensor[leftMatrixIndex] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(assistMatrixTensor[rightMatrixIndex] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(assistMatrixTensor[mulMatrixIndex] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto leftMatrix = assistMatrixTensor[leftMatrixIndex];
    auto rightMatrix = assistMatrixTensor[rightMatrixIndex];
    auto mulMatrix = assistMatrixTensor[mulMatrixIndex];

    size_t dimNum = self->GetViewShape().GetDimNum();
    aclIntArray* nchwShape = nullptr;
    if (dimNum == chwShapeSize) {
        const int64_t appendDim[] = {0};
        nchwShape = executor->AllocIntArray(appendDim, 1);
    } else {
        const int64_t appendDim[] = {0, 1};
        const int64_t appendDimSize = 2;
        nchwShape = executor->AllocIntArray(appendDim, appendDimSize);
    }

    auto leftMatrixUnsqueeze = l0op::UnsqueezeNd(leftMatrix, nchwShape, executor);
    CHECK_RET(leftMatrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto rightMatrixUnsqueeze = l0op::UnsqueezeNd(rightMatrix, nchwShape, executor);
    CHECK_RET(rightMatrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mulMatrixUnsqueeze = l0op::UnsqueezeNd(mulMatrix, nchwShape, executor);
    CHECK_RET(mulMatrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* leftMatrixBmmv2 = nullptr;
    const aclTensor* rightMatrixBmmv2 = nullptr;
    auto leftMatrixCast = l0op::Cast(leftMatrixUnsqueeze, self->GetDataType(), executor);
    CHECK_RET(leftMatrixCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto rightMatrixCast = l0op::Cast(rightMatrixUnsqueeze, self->GetDataType(), executor);
    CHECK_RET(rightMatrixCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    leftMatrixBmmv2 = ExecBatchMatmulOp(inputContiguous, leftMatrixCast, inputContiguous, true, true, 1, executor);
    CHECK_RET(leftMatrixBmmv2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    rightMatrixBmmv2 = ExecBatchMatmulOp(leftMatrixBmmv2, rightMatrixCast, inputContiguous, true, false, 1, executor);
    CHECK_RET(rightMatrixBmmv2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mulMatrixCast = l0op::Cast(mulMatrixUnsqueeze, rightMatrixBmmv2->GetDataType(), executor);
    CHECK_RET(mulMatrixCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mulResult = l0op::Mul(rightMatrixBmmv2, mulMatrixCast, executor);
    CHECK_RET(mulResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mulResultCast = l0op::Cast(mulResult, self->GetDataType(), executor);
    CHECK_RET(mulResultCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto format = dimNum == chwShapeSize ? Format::FORMAT_NCL : Format::FORMAT_NCHW;
    mulResultCast = Reformat(mulResultCast, format);

    auto viewCopyResult = l0op::ViewCopy(mulResultCast, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
static aclnnStatus DoFusionAdapativeAvgPool2D(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, aclOpExecutor* executor)
{
    int64_t hValue = (*outputSize)[0];
    int64_t wValue = (*outputSize)[1];
    if (hValue == 1 && wValue == 1) {
        return DoReduceMean(self, out, executor);
    }
    return DoAdaptiveAvgPool2D(self, outputSize, out, executor);
}

aclnnStatus aclnnAdaptiveAvgPool2dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaptiveAvgPool2d, DFX_IN(self, outputSize), DFX_OUT(out));
    // 固定写法，参数检查
    auto ret = CheckParams(self, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    ret = DoFusionAdapativeAvgPool2D(self, outputSize, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveAvgPool2d(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaptiveAvgPool2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
