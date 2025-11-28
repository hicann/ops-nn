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
 * \file aclnn_mv.cpp
 * \brief
 */

#include "aclnn_kernels/contiguous.h"
#include "level0/shape_op.h"
#include "aclnn_kernels/transdata.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "adaptive_max_pool2d.h"
#include "adaptive_max_pool3d.h"
#include "level0/max_pool3d_with_argmax_v2.h"
#include "aclnn_adaptive_max_pool2d.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t AXIS_DIM = 0;
static constexpr size_t OUTPUT_SIZE_NUM = 2;
static constexpr size_t CHW_DIM_NUM = 3;
static constexpr size_t NCHW_DIM_NUM = 4;
static constexpr size_t NCDHW_DIM_NUM = 5;
static constexpr int64_t INDEX_DIM0 = 0;
static constexpr int64_t INDEX_DIM1 = 1;
static constexpr int64_t INDEX_DIM2 = 2;
static constexpr int64_t INDEX_DIM3 = 3;
static constexpr int64_t INDEX_DIM4 = 4;
static constexpr int64_t DIM_D = 2;
constexpr size_t MAX_VECTOR_SIZE = 32;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsSelfDtypeDouble(const aclTensor* self)
{
    if (self->GetDataType() == op::DataType::DT_DOUBLE) {
        return true;
    }
    return false;
}

static bool IsSocVersion910B()
{
    if (op::GetCurrentPlatformInfo().GetSocVersion() >= op::SocVersion::ASCEND910B &&
        op::GetCurrentPlatformInfo().GetSocVersion() <= op::SocVersion::ASCEND910E) {
        return true;
    }
    return false;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return DTYPE_SUPPORT_LIST_ASCEND910B;
    } else {
        return DTYPE_SUPPORT_LIST;
    }
}

static bool CheckNotNull(
    const aclTensor* self, const aclIntArray* outputSize, const aclTensor* output, const aclTensor* indices)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(indices, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* output, const aclTensor* indices)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(output, self->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(indices, op::DataType::DT_INT64, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* output, const aclTensor* indices)
{
    size_t dimNum = self->GetViewShape().GetDimNum();
    if (dimNum != CHW_DIM_NUM && dimNum != NCHW_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dims %zu should be in 3 or 4.", dimNum);
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(output, indices, return false);
    return true;
}

static bool CheckInpuNullTensor(const aclTensor* self)
{
    auto inputShape = self->GetViewShape();
    size_t dimNum = inputShape.GetDimNum();

    for (size_t i = 1; i < dimNum; ++i) {
        if (inputShape.GetDim(i) <= 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "adaptive_max_pooling2d():expected input to have non-empty spatiak dimensions, "
                "but input has sizes %zu with dimension %zu being empty.",
                dimNum, i);
            return false;
        }
    }
    return true;
}

static bool CheckOutputSize(const aclIntArray* outputSize)
{
    uint64_t size = outputSize->Size();
    if (size != OUTPUT_SIZE_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputSize length should be 2, but now is %lu.", size);
        return false;
    }
    for (size_t i = 0; i < size; ++i) {
        if ((*outputSize)[i] <= 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "outputSize value should greater than 0, but the sizes of %zu is %ld.", i,
                (*outputSize)[i]);
            return false;
        }
    }
    return true;
}

static bool CheckInputFormat(const aclTensor* self)
{
    auto input_format = self->GetViewFormat();
    if (input_format != op::Format::FORMAT_NHWC && input_format != op::Format::FORMAT_NCHW) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input format only support NHWC[%d] or NCHW[%d], but now is %d.",
            op::Format::FORMAT_NHWC, op::Format::FORMAT_NCHW, input_format);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclIntArray* outputSize, const aclTensor* output, const aclTensor* indices)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, outputSize, output, indices), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入输出数据类型
    CHECK_RET(CheckDtypeValid(self, output, indices), ACLNN_ERR_PARAM_INVALID);

    // 3. 校验shape
    CHECK_RET(CheckShape(self, output, indices), ACLNN_ERR_PARAM_INVALID);

    // 4. 校验输入tensor是否为空
    CHECK_RET(CheckInpuNullTensor(self), ACLNN_ERR_PARAM_INVALID);

    // 5. 校验output_size是否合法
    CHECK_RET(CheckOutputSize(outputSize), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveMaxPool2dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaptiveMaxPool2d, DFX_IN(self, outputSize), DFX_OUT(outputOut, indicesOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, outputSize, outputOut, indicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    if (IsSocVersion910B() && !(IsSelfDtypeDouble(self))) {
        // 将2d参数转换为3d可以使用的参数
        int64_t newOutputSizeData[] = {1, (*outputSize)[0], (*outputSize)[1]};
        aclIntArray* newOutputSize = uniqueExecutor.get()->AllocIntArray(newOutputSizeData, 3);

        auto inputContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // NHWC -> NCHW
        if (self->GetViewFormat() == op::Format::FORMAT_NHWC) {
            std::vector<int64_t> valuePerm{INDEX_DIM0, INDEX_DIM3, INDEX_DIM1, INDEX_DIM2};
            auto perm = uniqueExecutor.get()->AllocIntArray(valuePerm.data(), NCHW_DIM_NUM);
            CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
            inputContiguous = l0op::Transpose(inputContiguous, perm, uniqueExecutor.get());
            CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // reshape NCHW/NCL -> NCDHW
        op::Shape inputContiguousShape = inputContiguous->GetViewShape();
        size_t inputDimNum = inputContiguousShape.GetDimNum();
        std::vector<int64_t> valueShape(NCDHW_DIM_NUM);
        valueShape[0] = inputDimNum == NCHW_DIM_NUM ? inputContiguousShape.GetDim(0) : 1;
        valueShape[1] = inputDimNum == NCHW_DIM_NUM ? inputContiguousShape.GetDim(1) : inputContiguousShape.GetDim(0);
        valueShape[DIM_D] = 1;
        for (size_t i = inputDimNum - OUTPUT_SIZE_NUM; i < inputDimNum; i++) {
            valueShape[NCDHW_DIM_NUM - inputDimNum + i] = inputContiguousShape.GetDim(i);
        }
        auto reshapeShape = uniqueExecutor.get()->AllocIntArray(valueShape.data(), NCDHW_DIM_NUM);
        CHECK_RET(reshapeShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputContiguousReshape = l0op::Reshape(inputContiguous, reshapeShape, uniqueExecutor.get());
        CHECK_RET(inputContiguousReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        int64_t dValueRemainder = inputContiguousReshape->GetViewShape()[INDEX_DIM2] % (*newOutputSize)[INDEX_DIM0];
        int64_t hValueRemainder = inputContiguousReshape->GetViewShape()[INDEX_DIM3] % (*newOutputSize)[INDEX_DIM1];
        int64_t wValueRemainder = inputContiguousReshape->GetViewShape()[INDEX_DIM4] % (*newOutputSize)[INDEX_DIM2];
        if (dValueRemainder == 0 && hValueRemainder == 0 && wValueRemainder == 0) {
            size_t kernelDSize = inputContiguousReshape->GetViewShape()[INDEX_DIM2] / (*newOutputSize)[INDEX_DIM0];
            size_t kernelHSize = inputContiguousReshape->GetViewShape()[INDEX_DIM3] / (*newOutputSize)[INDEX_DIM1];
            size_t kernelWSize = inputContiguousReshape->GetViewShape()[INDEX_DIM4] / (*newOutputSize)[INDEX_DIM2];
            std::vector<int64_t> kernelSizeArr = {};
            kernelSizeArr.push_back(kernelDSize);
            kernelSizeArr.push_back(kernelHSize);
            kernelSizeArr.push_back(kernelWSize);
            const aclIntArray* kernelSize = aclCreateIntArray(kernelSizeArr.data(), kernelSizeArr.size());
            const aclIntArray* stride = aclCreateIntArray(kernelSizeArr.data(), kernelSizeArr.size());
            std::vector<int64_t> paddingArr = {0, 0, 0};
            const aclIntArray* padding = aclCreateIntArray(paddingArr.data(), paddingArr.size());
            std::vector<int64_t> dilationArr = {1, 1, 1};
            const aclIntArray* dilation = aclCreateIntArray(dilationArr.data(), dilationArr.size());
            bool ceilMode = false;
            std::string dataFormat = "NCDHW";

            inputContiguousReshape =
                l0op::ReFormat(inputContiguousReshape, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
            CHECK_RET(inputContiguousReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto [outResult, indicesResult] = l0op::MaxPool3DWithArgmaxV2Ncdhw(
                inputContiguousReshape, kernelSize, stride, padding, dilation, ceilMode, dataFormat,
                uniqueExecutor.get());
            CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto indicesResultCast = l0op::Cast(indicesResult, op::DataType::DT_INT64, uniqueExecutor.get());
            CHECK_RET(indicesResultCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto resShapeVector = op::ToShapeVector(outputOut->GetViewShape());
            aclIntArray* resShapeArray =
                uniqueExecutor.get()->AllocIntArray(resShapeVector.data(), resShapeVector.size());
            CHECK_RET(resShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto resTranReshapeOut = l0op::Reshape(outResult, resShapeArray, uniqueExecutor.get());
            CHECK_RET(resTranReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            resTranReshapeOut = l0op::ReFormat(resTranReshapeOut, outputOut->GetStorageFormat(), uniqueExecutor.get());
            CHECK_RET(resTranReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto viewCopyResultOut = l0op::ViewCopy(resTranReshapeOut, outputOut, uniqueExecutor.get());
            CHECK_RET(viewCopyResultOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto resTranReshapeIndices = l0op::Reshape(indicesResultCast, resShapeArray, uniqueExecutor.get());
            CHECK_RET(resTranReshapeIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

            resTranReshapeIndices =
                l0op::ReFormat(resTranReshapeIndices, indicesOut->GetStorageFormat(), uniqueExecutor.get());
            CHECK_RET(resTranReshapeIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto viewCopyResultIndices = l0op::ViewCopy(resTranReshapeIndices, indicesOut, uniqueExecutor.get());
            CHECK_RET(viewCopyResultIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
            // 释放资源
            aclDestroyIntArray(kernelSize);
            aclDestroyIntArray(stride);
            aclDestroyIntArray(padding);
            aclDestroyIntArray(dilation);
        } else {
            auto [outResult, indicesResult] =
                l0op::AdaptiveMaxPool3d(inputContiguousReshape, newOutputSize, uniqueExecutor.get());
            CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto indicesResultCast = l0op::Cast(indicesResult, op::DataType::DT_INT64, uniqueExecutor.get());
            CHECK_RET(indicesResultCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto resShapeVector = op::ToShapeVector(outputOut->GetViewShape());
            aclIntArray* resShapeArray =
                uniqueExecutor.get()->AllocIntArray(resShapeVector.data(), resShapeVector.size());
            CHECK_RET(resShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto resTranReshapeOut = l0op::Reshape(outResult, resShapeArray, uniqueExecutor.get());
            CHECK_RET(resTranReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto viewCopyResultOut = l0op::ViewCopy(resTranReshapeOut, outputOut, uniqueExecutor.get());
            CHECK_RET(viewCopyResultOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

            auto resTranReshapeIndices = l0op::Reshape(indicesResultCast, resShapeArray, uniqueExecutor.get());
            CHECK_RET(resTranReshapeIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto viewCopyResultIndices = l0op::ViewCopy(resTranReshapeIndices, indicesOut, uniqueExecutor.get());
            CHECK_RET(viewCopyResultIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }

        // 固定写法，获取计算过程中需要使用的workspace大小
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* selfNewFormat = selfContiguous;
    if (self->GetViewShape().GetDimNum() != NCHW_DIM_NUM) {
        auto selfNd = l0op::UnsqueezeNd(selfContiguous, AXIS_DIM, uniqueExecutor.get());
        CHECK_RET(selfNd != nullptr, ACLNN_ERR_INNER_NULLPTR);

        selfNewFormat = l0op::ReFormat(selfNd, static_cast<op::Format>(ACL_FORMAT_NCHW));
        CHECK_RET(selfNewFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    CHECK_RET(CheckInputFormat(selfNewFormat), ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::AdaptiveMaxPool2d(selfNewFormat, outputSize, uniqueExecutor.get());
    const aclTensor* outputRst = std::get<0>(result);
    const aclTensor* indicesRst = std::get<1>(result);

    CHECK_RET(outputRst != nullptr && indicesRst != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outputNewFormat = outputRst;
    auto indicesNewFormat = indicesRst;
    if (self->GetViewShape().GetDimNum() != NCHW_DIM_NUM) {
        auto outputNewShape = l0op::SqueezeNd(outputRst, AXIS_DIM, uniqueExecutor.get());
        CHECK_RET(outputNewShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto indicesNewShape = l0op::SqueezeNd(indicesRst, AXIS_DIM, uniqueExecutor.get());
        CHECK_RET(indicesNewShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        outputNewFormat = l0op::ReFormat(outputNewShape, outputOut->GetViewFormat());
        CHECK_RET(outputNewFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        indicesNewFormat = l0op::ReFormat(indicesNewShape, indicesOut->GetViewFormat());
        CHECK_RET(indicesNewFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // check output shape
    CHECK_RET(CheckReduceOutShape(outputNewFormat, outputOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckReduceOutShape(indicesNewFormat, indicesOut), ACLNN_ERR_PARAM_INVALID);
    auto viewCopyOutputResult = l0op::ViewCopy(outputNewFormat, outputOut, uniqueExecutor.get());
    CHECK_RET(viewCopyOutputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyIndicesResult = l0op::ViewCopy(indicesNewFormat, indicesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveMaxPool2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaptiveMaxPool2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
