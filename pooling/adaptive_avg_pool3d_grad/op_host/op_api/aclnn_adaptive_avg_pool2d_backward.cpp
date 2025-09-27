/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <numeric>
#include <vector>
#include "aclnn/aclnn_base.h"
#include "aclnn_adaptive_avg_pool2d_backward.h"
#include "adaptive_avg_pool3d_backward.h"
#include "level0/adaptive_avg_pool2d_assist_matrix.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "level0/shape_op.h"
#include "level0/unsqueeze.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/fast_vector.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t hDimIndex = -2;
static const int64_t wDimIndex = -1;
static const int64_t hDimIndexFromLast = 2;
static const int64_t wDimIndexFromLast = 1;
static const int64_t hwShapeSizes = 2;
static const int64_t chwShapesizes = 3;
static const int64_t nchwShapesizes = 4;
static const int64_t reshapeSizes = 4;
static const int64_t leftMaitrixIndex = 0;
static const int64_t rightMaitrixIndex = 1;
static const int64_t vmWeightIndex = 2;
static const int64_t assistMatrixNums = 3;
static const int64_t POOL_DIMNUM = 3;

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}
namespace{
static bool CheckInputDims(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* /* out */)
{
    auto selfDimNum = self->GetViewShape().GetDimNum();
    auto gradOutputDimNum = gradOutput->GetViewShape().GetDimNum();

    const op::Shape selfShape = self->GetViewShape();
    const op::Shape gradOutputShape = gradOutput->GetViewShape();

    if (gradOutputDimNum != selfDimNum) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "DimNum of inputs should be equal, gradOutput [%zu], self [%zu]", gradOutputDimNum,
            selfDimNum);
        return false;
    }
    for (size_t i = 0; i < selfDimNum; i++) {
        if (selfShape.GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self'dims is invalid, self No.[%lu] dim is [%d].", i + 1, 0);
            return false;
        }
    }
    for (size_t i = 0; i < gradOutputDimNum; i++) {
        if (gradOutputShape.GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput'dims is invalid, self No.[%lu] dim is [%d].", i + 1, 0);
            return false;
        }
    }
    return true;
}
}


static bool CheckInputOutputShape(const aclTensor* self, const aclTensor* out)
{
    size_t outDimNum = out->GetViewShape().GetDimNum();
    size_t selfDimNum = self->GetViewShape().GetDimNum();
    const op::Shape selfShape = self->GetViewShape();
    const op::Shape outShape = out->GetViewShape();

    if (outDimNum != chwShapesizes && outDimNum != nchwShapesizes) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dims %zu should be in 3 or 4.", outDimNum);
        return false;
    }
    if (outDimNum != selfDimNum) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "DimNum of inputs should be equal, out [%zu], self [%zu]", outDimNum, selfDimNum);
        return false;
    }
    // 2 is dim offset
    if (outDimNum > 2U) {
        for (size_t i = 0; i < outDimNum - 2U; i++) {
            if (outShape[i] != selfShape[i]) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "Out_shape[%zu]: %ld should equal to input_shape[%zu]: %ld", i, outShape[i], i,
                    selfShape[i]);
                return false;
            }
        }
    }

    return true;
}

static const std::initializer_list<op::DataType> ORIGIN_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ORIGIN_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    // 检查gradOutput、self的数据类型是否在AdaptiveAvgPool2dBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查gradOutput和self数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* self)
{
    if (gradOutput->GetStorageFormat() != self->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of inputs should be equal, gradOutput [%s], self [%s]",
            op::ToString(gradOutput->GetStorageFormat()).GetString(),
            op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(gradOutput->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format don't support private format.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* y)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, y), ACLNN_ERR_PARAM_NULLPTR);

    // 检查输入参数维度
    CHECK_RET(CheckInputDims(gradOutput, self, y), ACLNN_ERR_PARAM_INVALID);

    // 检查输入输出参shape
    CHECK_RET(CheckInputOutputShape(self, y), ACLNN_ERR_PARAM_INVALID);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持
    CHECK_RET(CheckFormat(gradOutput, self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static int64_t adaptive_avg_pool2d_backward_safe_size(const aclTensor* self)
{
    FVector<int64_t, hwShapeSizes> dims = {hDimIndex, wDimIndex};

    int64_t size = 1;
    if (self->IsEmpty()) {
        return size;
    }
    int64_t dim_post_expr = self->GetViewShape().GetDimNum();
    for (int64_t ndim : dims) {
        if (dim_post_expr <= 0) {
            dim_post_expr = 1; // this will make range [-1, 0]
        }
        ndim += dim_post_expr;
        size *= self->GetViewShape().GetDim(ndim);
    }

    return size;
}

static aclTensor* SetFormat(const aclTensor* self, op::Format format)
{
    aclTensor* input_cast = const_cast<aclTensor*>(self);
    input_cast->SetStorageFormat(format);
    input_cast->SetViewFormat(format);
    input_cast->SetOriginalFormat(format);
    return input_cast;
}

static inline aclnnStatus checkRetReFormat(bool cond, aclnnStatus status, const aclTensor* acltensor, op::Format format)
{
    if (!cond) {
        SetFormat(acltensor, format);
        return status;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DoFusionAdaptiveAvgPool2DBackward(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        // 将2d参数转换为3d可以使用的参数
        auto inputContiguous = l0op::Contiguous(gradOutput, executor);
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfContiguous = l0op::Contiguous(self, executor);
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // reshape(展平NC为1维，增加d维)
        op::Shape inputContiguousShape = inputContiguous->GetViewShape();
        op::Shape selfContiguousShape = selfContiguous->GetViewShape();
        auto inputDimNum = inputContiguousShape.GetDimNum();
        std::vector<int64_t> valueShape(reshapeSizes);
        valueShape[0] = -1;
        valueShape[1] = 1;
        auto self3dDimNum = inputDimNum + 1;
        std::vector<int64_t> selfShape(self3dDimNum);
        auto unsqueezeDim = self3dDimNum - hDimIndexFromLast - 1;
        selfShape[unsqueezeDim] = 1;
        for (int64_t i = 0; i < hDimIndexFromLast; i++) {
            valueShape[reshapeSizes - hDimIndexFromLast + i] =
                inputContiguousShape.GetDim(inputDimNum - hDimIndexFromLast + i);
            selfShape[self3dDimNum - hDimIndexFromLast + i] =
                selfContiguousShape.GetDim(inputDimNum - hDimIndexFromLast + i);
        }
        for (uint64_t i = 0; i < unsqueezeDim; i++) {
            selfShape[i] = selfContiguousShape.GetDim(i);
        }
        auto reshapeShape = executor->AllocIntArray(valueShape.data(), reshapeSizes);
        CHECK_RET(reshapeShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputContiguousReshape = l0op::Reshape(inputContiguous, reshapeShape, executor);
        CHECK_RET(inputContiguousReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfReshapeShape = executor->AllocIntArray(selfShape.data(), self3dDimNum);
        CHECK_RET(selfReshapeShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfContiguousReshape = l0op::Reshape(selfContiguous, selfReshapeShape, executor);
        CHECK_RET(selfContiguousReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将DHW维度转置到前面
        std::vector<int64_t> valuePerm(reshapeSizes);
        for (int64_t i = 0; i < static_cast<int64_t>(reshapeSizes); i++) {
            valuePerm[i] = (i + reshapeSizes - POOL_DIMNUM) % reshapeSizes;
        }
        auto perm = executor->AllocIntArray(valuePerm.data(), reshapeSizes);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputReshapeTran = l0op::Transpose(inputContiguousReshape, perm, executor);
        CHECK_RET(inputReshapeTran != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto poolResult = l0op::AdaptiveAvgPool3dGrad(inputReshapeTran, selfContiguousReshape, executor);
        CHECK_RET(poolResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将DHW维度转置到后面
        std::vector<int64_t> valuePermRes(reshapeSizes);
        for (int64_t i = 0; i < reshapeSizes; i++) {
            valuePermRes[i] = (i + POOL_DIMNUM) % reshapeSizes;
        }
        auto permRes = executor->AllocIntArray(valuePermRes.data(), reshapeSizes);
        CHECK_RET(permRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto resTran = l0op::Transpose(poolResult, permRes, executor);
        CHECK_RET(resTran != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto resReshapeSize = out->GetViewShape().GetDimNum();
        std::vector<int64_t> resShapeValue(resReshapeSize);
        for (int64_t i = 0; i < resReshapeSize; i++) {
            resShapeValue[i] = out->GetViewShape().GetDim(i);
        }
        auto resShape = executor->AllocIntArray(resShapeValue.data(), resReshapeSize);
        CHECK_RET(resShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto resTranReshape = l0op::Reshape(resTran, resShape, executor);
        CHECK_RET(resTranReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(resTranReshape, out, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }
    // Save gradOutput origin format
    Format gradOutputOriFormat = gradOutput->GetStorageFormat();

    // Generate  out_size
    auto input = SetFormat(gradOutput, Format::FORMAT_ND);

    int64_t gradOutputShape[2];
    auto gradOutputShapes = gradOutput->GetViewShape();
    auto gradOutputDim = gradOutputShapes.GetDimNum();
    gradOutputShape[0] = gradOutputShapes.GetDim(gradOutputDim - hDimIndexFromLast);
    gradOutputShape[1] = gradOutputShapes.GetDim(gradOutputDim - wDimIndexFromLast);
    aclIntArray* output_size = executor->AllocIntArray(gradOutputShape, hwShapeSizes);

    auto selfShape = l0op::Shape_op(self, executor);
    checkRetReFormat(selfShape != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);

    auto selfShapeCast = l0op::Cast(selfShape, DataType::DT_INT64, executor);
    checkRetReFormat(selfShapeCast != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);

    // Generate Assist Matrix
    std::array<aclTensor*, assistMatrixNums> constList =
        l0op::AdaptiveAvgPool2dAssistMatrix(selfShapeCast, self, output_size, executor);

    auto vmWeight = constList[vmWeightIndex];
    auto leftMaitrix = constList[leftMaitrixIndex];
    auto rightMaitrix = constList[rightMaitrixIndex];

    checkRetReFormat(vmWeight != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);
    checkRetReFormat(leftMaitrix != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);
    checkRetReFormat(rightMaitrix != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);
    CHECK_RET(vmWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(leftMaitrix != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(rightMaitrix != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* vmWeightCast = vmWeight;
    if (gradOutput->GetDataType() != vmWeight->GetDataType()) {
        vmWeightCast = l0op::Cast(vmWeight, gradOutput->GetDataType(), executor);
        CHECK_RET(vmWeightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // start mul
    auto vmOut = l0op::Mul(gradOutput, vmWeightCast, executor);
    CHECK_RET(vmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    checkRetReFormat(vmOut != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);

    // UnsqueezeNd
    size_t dimNums = self->GetViewShape().GetDimNum();
    aclIntArray* nchwShape = nullptr;
    if (dimNums == chwShapesizes) {
        const int64_t append_dim[] = {0};
        nchwShape = executor->AllocIntArray(append_dim, 1);
    } else {
        const int64_t append_dim[] = {0, 1};
        nchwShape = executor->AllocIntArray(append_dim, hwShapeSizes);
    }
    auto leftMaitrixUnsqueeze = l0op::UnsqueezeNd(leftMaitrix, nchwShape, executor);
    CHECK_RET(leftMaitrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
    checkRetReFormat(leftMaitrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);
    auto rightMaitrixUnsqueeze = l0op::UnsqueezeNd(rightMaitrix, nchwShape, executor);
    CHECK_RET(rightMaitrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
    checkRetReFormat(rightMaitrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);

    if (vmOut->GetDataType() != leftMaitrixUnsqueeze->GetDataType()) {
        leftMaitrixUnsqueeze = l0op::Cast(leftMaitrixUnsqueeze, vmOut->GetDataType(), executor);
        CHECK_RET(leftMaitrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 创建BMM left
    aclTensor* bmmOutDesc = const_cast<aclTensor*>(input);
    auto bmmOutForward = ExecBatchMatmulOp(vmOut, leftMaitrixUnsqueeze, bmmOutDesc, true, false, 1, executor);
    CHECK_RET(bmmOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR);
    checkRetReFormat(bmmOutForward != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);

    if (bmmOutForward->GetDataType() != rightMaitrixUnsqueeze->GetDataType()) {
        rightMaitrixUnsqueeze = l0op::Cast(rightMaitrixUnsqueeze, bmmOutForward->GetDataType(), executor);
        CHECK_RET(rightMaitrixUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 创建BMM right
    auto bmmOut = ExecBatchMatmulOp(bmmOutForward, rightMaitrixUnsqueeze, bmmOutDesc, true, true, 1, executor);
    CHECK_RET(bmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    checkRetReFormat(bmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);

    auto bmmOutReformat = SetFormat(bmmOut, Format::FORMAT_ND);
    auto view_copy_result = l0op::ViewCopy(bmmOutReformat, out, executor);
    CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    checkRetReFormat(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR, gradOutput, gradOutputOriFormat);

    // reset gradOutput format
    (void)SetFormat(gradOutput, gradOutputOriFormat);
    return ACLNN_SUCCESS;
}

static aclnnStatus DoAdapativeAvgPool2DBackward(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    auto gradOutputShapes = gradOutput->GetViewShape();
    auto gradOutputDim = gradOutputShapes.GetDimNum();
    if (gradOutputShapes.GetDim(gradOutputDim - hDimIndexFromLast) == 1 &&
        gradOutputShapes.GetDim(gradOutputDim - wDimIndexFromLast) == 1) {
        // fill value tensor
        float fillVaule = 1.0 / (adaptive_avg_pool2d_backward_safe_size(self));
        aclScalar* value = executor->AllocScalar(fillVaule);
        const aclTensor* castTensor = executor->ConvertToTensor(value, static_cast<op::DataType>(self->GetDataType()));
        // fill dims tensor
        size_t dimNum = self->GetViewShape().GetDimNum();
        FVector<int64_t> dimTmp;
        for (size_t idx = 0; idx < dimNum; idx++) {
            int64_t tmpVal = self->GetViewShape().GetDim(idx);
            dimTmp.push_back(tmpVal);
        }

        aclIntArray* shapeArray = executor->AllocIntArray(dimTmp.data(), dimTmp.size());
        const aclTensor* dims =
            executor->ConvertToTensor(dimTmp.data(), dimTmp.size(), static_cast<op::DataType>(ACL_INT64));
        auto fillOut = l0op::Fill(dims, castTensor, shapeArray, executor);
        CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto mulOpOut = l0op::Mul(fillOut, gradOutput, executor);
        CHECK_RET(mulOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将计算结果转换成输出out的数据类型
        auto castOut = l0op::Cast(mulOpOut, out->GetDataType(), executor);
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto view_copy_result = l0op::ViewCopy(castOut, out, executor);
        CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    } else {
        return DoFusionAdaptiveAvgPool2DBackward(gradOutput, self, out, executor);
    }
}

aclnnStatus aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaptiveAvgPool2dBackward, DFX_IN(gradOutput, self), DFX_OUT(out));

    // 参数检查
    auto ret = CheckParams(gradOutput, self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty() || gradOutput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto rets = DoAdapativeAvgPool2DBackward(gradOutputContiguous, selfContiguous, out, uniqueExecutor.get());
    CHECK_RET(rets == ACLNN_SUCCESS, rets);

    // // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveAvgPool2dBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaptiveAvgPool2dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
