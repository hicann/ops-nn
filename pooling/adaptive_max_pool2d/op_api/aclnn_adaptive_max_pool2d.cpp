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
 * \file aclnn_adaptive_max_pool2d.cpp
 * \brief
 */

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "adaptive_max_pool2d.h"
#include "level0/max_pool3d_with_argmax_v2.h"
#include "../../adaptive_max_pool3d/op_api/adaptive_max_pool3d.h"
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
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        return true;
    }
    return false;
}

static bool IsSocVersion910D()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return true;
    }
    return false;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (IsSocVersion910B() || IsSocVersion910D()) {
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

static inline const aclTensor* View3Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // CHW -> unsqueeze -> reformat -> NCDHW
    // unsqueeze input into 4D
    const aclTensor* unsqueezedInput = l0op::UnsqueezeNd(input, 1, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // unsqueeze input into 5D
    auto unsqueezedInput5D = l0op::UnsqueezeNd(unsqueezedInput, static_cast<int64_t>(0), executor);
    CHECK_RET(unsqueezedInput5D != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput5D, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View4Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCHW -> unsqueeze -> reformat -> NCDHW
    // unsqueeze input into 5D
    auto unsqueezedInput = l0op::UnsqueezeNd(input, 2, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View5Das3D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // NCDHW -> squeeze -> reformat -> CHW
    // squeeze out into 4D
    const aclTensor* squeezedInput = l0op::SqueezeNd(input, 2, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // squeeze out into 3D
    auto squeezedInput3D = l0op::SqueezeNd(squeezedInput, static_cast<int64_t>(0), executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCL
    auto reformatInput = l0op::ReFormat(squeezedInput3D, format);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View5Das4D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // NCDHW -> squeeze -> reformat -> NCHW
    // squeeze out into 3D
    auto squeezedInput = l0op::SqueezeNd(input, 2, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor* View3Das4D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCL -> unsqueeze -> reformat -> NCHW
    // unsqueeze input into 4D
    const int64_t appendDim[] = {0};
    aclIntArray* dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static aclnnStatus ProcessAndCopyResults(
    const aclTensor* outResult, 
    const aclTensor* indicesResultCast,
    aclTensor* outputOut, 
    aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    auto resShapeVector = op::ToShapeVector(outputOut->GetViewShape());
    aclIntArray* resShapeArray =
        executor->AllocIntArray(resShapeVector.data(), resShapeVector.size());
    CHECK_RET(resShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto resTranReshapeOut = l0op::Reshape(outResult, resShapeArray, executor);
    CHECK_RET(resTranReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    resTranReshapeOut = l0op::ReFormat(resTranReshapeOut, outputOut->GetStorageFormat(), executor);
    CHECK_RET(resTranReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResultOut = l0op::ViewCopy(resTranReshapeOut, outputOut, executor);
    CHECK_RET(viewCopyResultOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto resTranReshapeIndices = l0op::Reshape(indicesResultCast, resShapeArray, executor);
    CHECK_RET(resTranReshapeIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    resTranReshapeIndices =
        l0op::ReFormat(resTranReshapeIndices, indicesOut->GetStorageFormat(), executor);
    CHECK_RET(resTranReshapeIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResultIndices = l0op::ViewCopy(resTranReshapeIndices, indicesOut, executor);
    CHECK_RET(viewCopyResultIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    return ACLNN_SUCCESS;
}

// 转MaxPool3D
static aclnnStatus HandleMaxPool3DCase(
    const aclTensor* inputContiguousReshape, 
    const std::vector<int64_t>& kernelSizeArr,
    aclTensor* outputOut, 
    aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    const aclIntArray* kernelSize = aclCreateIntArray(kernelSizeArr.data(), kernelSizeArr.size());
    const aclIntArray* stride = aclCreateIntArray(kernelSizeArr.data(), kernelSizeArr.size());
    std::vector<int64_t> paddingArr = {0, 0, 0};
    const aclIntArray* padding = aclCreateIntArray(paddingArr.data(), paddingArr.size());
    std::vector<int64_t> dilationArr = {1, 1, 1};
    const aclIntArray* dilation = aclCreateIntArray(dilationArr.data(), dilationArr.size());
    bool ceilMode = false;
    std::string dataFormat = "NCDHW";

    inputContiguousReshape =
        l0op::ReFormat(inputContiguousReshape, op::Format::FORMAT_NCDHW, executor);
    CHECK_RET(inputContiguousReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto [outResult, indicesResult] = l0op::MaxPool3DWithArgmaxV2Ncdhw(
        inputContiguousReshape, kernelSize, stride, padding, dilation, ceilMode, dataFormat,
        executor);
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesResultCast = l0op::Cast(indicesResult, op::DataType::DT_INT64, executor);
    CHECK_RET(indicesResultCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto ret = ProcessAndCopyResults(outResult, indicesResultCast, outputOut, indicesOut, executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    aclDestroyIntArray(kernelSize);
    aclDestroyIntArray(stride);
    aclDestroyIntArray(padding);
    aclDestroyIntArray(dilation);
    
    return ACLNN_SUCCESS;
}

// AdaptiveMaxPool2d的处理逻辑
static aclnnStatus ProcessAdaptiveMaxPoolResults(
    const aclTensor* outResult, 
    const aclTensor* indicesResult,
    aclTensor* outputOut, 
    aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    auto indicesResultCast = l0op::Cast(indicesResult, op::DataType::DT_INT64, executor);
    CHECK_RET(indicesResultCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto resShapeVector = op::ToShapeVector(outputOut->GetViewShape());
    aclIntArray* resShapeArray =
        executor->AllocIntArray(resShapeVector.data(), resShapeVector.size());
    CHECK_RET(resShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto resTranReshapeOut = l0op::Reshape(outResult, resShapeArray, executor);
    CHECK_RET(resTranReshapeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultOut = l0op::ViewCopy(resTranReshapeOut, outputOut, executor);
    CHECK_RET(viewCopyResultOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto resTranReshapeIndices = l0op::Reshape(indicesResultCast, resShapeArray, executor);
    CHECK_RET(resTranReshapeIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResultIndices = l0op::ViewCopy(resTranReshapeIndices, indicesOut, executor);
    CHECK_RET(viewCopyResultIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    return ACLNN_SUCCESS;
}

// NHWC->NCHW
static aclnnStatus ConvertNHWCtoNCHW(
    const aclTensor*& inputContiguous,
    const aclTensor* self,
    aclOpExecutor* executor)
{
    if (self->GetViewFormat() == op::Format::FORMAT_NHWC) {
        std::vector<int64_t> valuePerm{INDEX_DIM0, INDEX_DIM3, INDEX_DIM1, INDEX_DIM2};
        auto perm = executor->AllocIntArray(valuePerm.data(), NCHW_DIM_NUM);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        inputContiguous = l0op::Transpose(inputContiguous, perm, executor);
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus Handle910DCase(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    auto inputContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // NHWC -> NCHW
    auto ret = ConvertNHWCtoNCHW(inputContiguous, self, executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // reshape NCHW/NCL -> NCDHW
    op::Shape inputContiguousShape = inputContiguous->GetViewShape();
    int64_t inputDimNum = static_cast<int64_t>(inputContiguousShape.GetDimNum());
    int64_t hiValue = inputDimNum == 3 ? inputContiguous->GetViewShape()[1] : inputContiguous->GetViewShape()[2];
    int64_t wiValue = inputDimNum == 3 ? inputContiguous->GetViewShape()[2] : inputContiguous->GetViewShape()[3];

    int64_t hValueRemainder = hiValue % (*outputSize)[0];
    int64_t wValueRemainder = wiValue % (*outputSize)[1];
    bool ifTranMaxPool3D = hValueRemainder == 0 && wValueRemainder == 0;
    const aclTensor* inputContiguousReshape = inputContiguous;
    if (inputDimNum == 3 && !ifTranMaxPool3D) { // 转ada 2d
        inputContiguousReshape = View3Das4D(inputContiguous, executor);
    } else if (ifTranMaxPool3D) { // 转max3d
        inputContiguousReshape = inputDimNum == 3 ? View3Das5D(inputContiguous, executor) :
                                 View4Das5D(inputContiguous, executor);
    }
    if (ifTranMaxPool3D) {
        int64_t kernelDSize = 1;
        int64_t kernelHSize = inputContiguousReshape->GetViewShape()[INDEX_DIM3] / (*outputSize)[INDEX_DIM0];
        int64_t kernelWSize = inputContiguousReshape->GetViewShape()[INDEX_DIM4] / (*outputSize)[INDEX_DIM1];
        std::vector<int64_t> kernelSizeArr = {kernelDSize, kernelHSize, kernelWSize};
        ret = HandleMaxPool3DCase(inputContiguousReshape, kernelSizeArr, outputOut, indicesOut,
                                  executor);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    } else {
        auto [outResult, indicesResult] =
            l0op::AdaptiveMaxPool2d(inputContiguousReshape, outputSize, executor);
        CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        
        ret = ProcessAdaptiveMaxPoolResults(outResult, indicesResult, outputOut, indicesOut, executor);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus Handle910BCase(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    // 将2d参数转换为3d可以使用的参数
    int64_t newOutputSizeData[] = {1, (*outputSize)[0], (*outputSize)[1]};
    aclIntArray* newOutputSize = executor->AllocIntArray(newOutputSizeData, 3);

    auto inputContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // NHWC -> NCHW
    auto ret = ConvertNHWCtoNCHW(inputContiguous, self, executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // reshape NCHW/NCL -> NCDHW
    op::Shape inputContiguousShape = inputContiguous->GetViewShape();
    int64_t inputDimNum = static_cast<int64_t>(inputContiguousShape.GetDimNum());
    std::vector<int64_t> valueShape(NCDHW_DIM_NUM);
    valueShape[0] = inputDimNum == static_cast<int64_t>(NCHW_DIM_NUM) ? inputContiguousShape.GetDim(0) : 1;
    valueShape[1] = inputDimNum == static_cast<int64_t>(NCHW_DIM_NUM) ? inputContiguousShape.GetDim(1) :
                    inputContiguousShape.GetDim(0);
    valueShape[DIM_D] = 1;
    for (int64_t i = inputDimNum - static_cast<int64_t>(OUTPUT_SIZE_NUM); i < inputDimNum; i++) {
        valueShape[NCDHW_DIM_NUM - inputDimNum + i] = inputContiguousShape.GetDim(i);
    }
    auto reshapeShape = executor->AllocIntArray(valueShape.data(), NCDHW_DIM_NUM);
    CHECK_RET(reshapeShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputContiguousReshape = l0op::Reshape(inputContiguous, reshapeShape, executor);
    CHECK_RET(inputContiguousReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t dValueRemainder = inputContiguousReshape->GetViewShape()[INDEX_DIM2] % (*newOutputSize)[INDEX_DIM0];
    int64_t hValueRemainder = inputContiguousReshape->GetViewShape()[INDEX_DIM3] % (*newOutputSize)[INDEX_DIM1];
    int64_t wValueRemainder = inputContiguousReshape->GetViewShape()[INDEX_DIM4] % (*newOutputSize)[INDEX_DIM2];
    if (dValueRemainder == 0 && hValueRemainder == 0 && wValueRemainder == 0) {
        int64_t kernelDSize = inputContiguousReshape->GetViewShape()[INDEX_DIM2] / (*newOutputSize)[INDEX_DIM0];
        int64_t kernelHSize = inputContiguousReshape->GetViewShape()[INDEX_DIM3] / (*newOutputSize)[INDEX_DIM1];
        int64_t kernelWSize = inputContiguousReshape->GetViewShape()[INDEX_DIM4] / (*newOutputSize)[INDEX_DIM2];
        std::vector<int64_t> kernelSizeArr = {kernelDSize, kernelHSize, kernelWSize};
        
        ret = HandleMaxPool3DCase(inputContiguousReshape, kernelSizeArr, outputOut, indicesOut, executor);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    } else {
        auto [outResult, indicesResult] =
            l0op::AdaptiveMaxPool3d(inputContiguousReshape, newOutputSize, executor);
        CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        
        ret = ProcessAdaptiveMaxPoolResults(outResult, indicesResult, outputOut, indicesOut, executor);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus HandleGeneralCase(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut,
    aclOpExecutor* executor)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* selfNewFormat = selfContiguous;
    if (self->GetViewShape().GetDimNum() != NCHW_DIM_NUM) {
        auto selfNd = l0op::UnsqueezeNd(selfContiguous, AXIS_DIM, executor);
        CHECK_RET(selfNd != nullptr, ACLNN_ERR_INNER_NULLPTR);

        selfNewFormat = l0op::ReFormat(selfNd, static_cast<op::Format>(ACL_FORMAT_NCHW));
        CHECK_RET(selfNewFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    CHECK_RET(CheckInputFormat(selfNewFormat), ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::AdaptiveMaxPool2d(selfNewFormat, outputSize, executor);
    const aclTensor* outputRst = std::get<0>(result);
    const aclTensor* indicesRst = std::get<1>(result);

    CHECK_RET(outputRst != nullptr && indicesRst != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outputNewFormat = outputRst;
    auto indicesNewFormat = indicesRst;
    if (self->GetViewShape().GetDimNum() != NCHW_DIM_NUM) {
        auto outputNewShape = l0op::SqueezeNd(outputRst, AXIS_DIM, executor);
        CHECK_RET(outputNewShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto indicesNewShape = l0op::SqueezeNd(indicesRst, AXIS_DIM, executor);
        CHECK_RET(indicesNewShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        outputNewFormat = l0op::ReFormat(outputNewShape, outputOut->GetViewFormat());
        CHECK_RET(outputNewFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

        indicesNewFormat = l0op::ReFormat(indicesNewShape, indicesOut->GetViewFormat());
        CHECK_RET(indicesNewFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // check output shape
    CHECK_RET(CheckReduceOutShape(outputNewFormat, outputOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckReduceOutShape(indicesNewFormat, indicesOut), ACLNN_ERR_PARAM_INVALID);
    auto viewCopyOutputResult = l0op::ViewCopy(outputNewFormat, outputOut, executor);
    CHECK_RET(viewCopyOutputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyIndicesResult = l0op::ViewCopy(indicesNewFormat, indicesOut, executor);
    CHECK_RET(viewCopyIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
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

    if (IsSocVersion910D() && !(IsSelfDtypeDouble(self))) {
        CHECK_RET(
            Handle910DCase(self, outputSize, outputOut, indicesOut, uniqueExecutor.get()) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);
    } else if (IsSocVersion910B() && !(IsSelfDtypeDouble(self))) {
        CHECK_RET(Handle910BCase(self, outputSize, outputOut, indicesOut, uniqueExecutor.get()) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);
    } else {
        CHECK_RET(HandleGeneralCase(self, outputSize, outputOut, indicesOut, uniqueExecutor.get()) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);
    }
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
