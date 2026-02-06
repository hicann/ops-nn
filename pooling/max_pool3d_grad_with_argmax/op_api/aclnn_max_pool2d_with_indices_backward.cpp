/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "aclnn_max_pool2d_with_indices_backward.h"
#include "max_pool_grad_with_argmax_v1.h"
#include "pooling/max_pool_grad_with_argmax_v3/op_api/max_pool_grad_with_argmax_v3.h"
#include "max_pool3d_grad_with_argmax.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

using TupleArrayInput = std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>;

static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_NULL_LIST = {};
// 输入为ND场景下，1980不支持任何数据类型
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_910_LIST = {};
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_910B_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_REGBASE_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
// 输入为5HD场景下
static const std::initializer_list<op::DataType> MASK_SELF_OUT_DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MASK_SELF_OUT_DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_INDICES_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_REGBASE_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static const int DIMENTION_3 = 3;
static const int DIMENTION_4 = 4;
static const int SIZE_1 = 1;
static const int SIZE_2 = 2;
static const int SIZE_3 = 3;

static const int TUPLE_INX_GRAD = 0;
static const int TUPLE_INX_SELF = 1;
static const int TUPLE_INX_INDICES = 2;

static const int DIM_INX_0 = 0;
static const int DIM_INX_1 = 1;
static const int DIM_INX_2 = 2;
static const int DIM_INX_3 = 3;
static const int DIM_INX_4 = 4;

static const int INDICES_TYPE_CONVERT = 2;
static const int64_t BLOCKSIZE = 16;
static const int KERNEL_SIZE_32 = 32;
static const int KERNEL_SIZE_64 = 64;

static const inline std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion(
    const bool isMask, const bool isMask2ND)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_2201: {
            return (
                isMask    ? MASK_SELF_OUT_DTYPE_SUPPORT_910B_LIST :
                isMask2ND ? MASK_SELF_OUT_DTYPE_SUPPORT_910B_LIST :
                            SELF_OUT_DTYPE_SUPPORT_910B_LIST);
        }
        case NpuArch::DAV_3510: {
            return SELF_OUT_DTYPE_SUPPORT_REGBASE_LIST;
        }
        case NpuArch::DAV_1001: {
            return (isMask ? MASK_SELF_OUT_DTYPE_SUPPORT_910_LIST : SELF_OUT_DTYPE_SUPPORT_910_LIST);
        }
        default: {
            return SELF_OUT_DTYPE_NULL_LIST;
        }
    }
}

static const inline std::initializer_list<op::DataType> GetIndicesDtypeSupportListBySocVersion(const bool isMask)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_3510:
            return INDICES_DTYPE_SUPPORT_REGBASE_LIST;
        case NpuArch::DAV_2201:
        case NpuArch::DAV_1001:
        default: {
            return (isMask ? MASK_INDICES_DTYPE_SUPPORT_LIST : INDICES_DTYPE_SUPPORT_LIST);
        }
    }
}

static bool CheckNotNullPtr(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, aclTensor* gradInput)
{
    // gradOutput、self、indices、kernelSize、stride、padding、dilation、gradInput不能为空指针
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(dilation, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput,
    const bool isMask, const bool isMask2ND)
{
    // 检查self的数据类型是否在支持列表内
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion(isMask, isMask2ND);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(self, gradOutput, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, gradInput, return false);

    // 检查indices的数据类型是否在支持列表内
    const std::initializer_list<op::DataType> indicesDtypeSupportList = GetIndicesDtypeSupportListBySocVersion(isMask);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, indicesDtypeSupportList, return false);

    return true;
}

static bool CheckFormat(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if ((self->GetStorageFormat() != gradOutput->GetStorageFormat()) ||
        (self->GetStorageFormat() != gradInput->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format of self and gradOutput and gradInput should be same, self[%s], gradOutput[%s], gradInput[%s].",
            op::ToString(self->GetStorageFormat()).GetString(),
            op::ToString(gradOutput->GetStorageFormat()).GetString(),
            op::ToString(gradInput->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(gradInput->GetStorageFormat()) ||
        op::IsPrivateFormat(gradOutput->GetStorageFormat()) || op::IsPrivateFormat(indices->GetStorageFormat())) {
        if (Ops::NN::AclnnUtil::IsRegbase(curArch)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW or NHWC");
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCHW");
        }

        return false;
    }

    // 仅支持NCHW、CHW
    if (self->GetViewShape().GetDimNum() != DIMENTION_3 && self->GetViewShape().GetDimNum() != DIMENTION_4) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "3D or 4D tensor expected for self but got dim num:%zu",
            self->GetViewShape().GetDimNum());
        return false;
    }

    if (gradInput->GetViewShape().GetDimNum() != DIMENTION_3 && gradInput->GetViewShape().GetDimNum() != DIMENTION_4) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "3D or 4D tensor expected for gradInput but got dim num:%zu",
            gradInput->GetViewShape().GetDimNum());
        return false;
    }

    if (gradOutput->GetViewShape().GetDimNum() != DIMENTION_3 &&
        gradOutput->GetViewShape().GetDimNum() != DIMENTION_4) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "3D or 4D tensor expected for gradOutput but got dim num:%zu",
            gradOutput->GetViewShape().GetDimNum());
        return false;
    }

    if (indices->GetViewShape().GetDimNum() != DIMENTION_3 && indices->GetViewShape().GetDimNum() != DIMENTION_4) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "3D or 4D tensor expected for indices but got dim num:%zu",
            indices->GetViewShape().GetDimNum());
        return false;
    }

    return true;
}

// size只能是1或者2
static inline bool CheckAttrSize1Or2(const aclIntArray* param)
{
    if ((param->Size() != 1) && (param->Size() != SIZE_2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "param size must be a single int, or a tuple of two ints");
        return false;
    }
    return true;
}

// size只能是0或1或者2
static inline bool CheckAttrSize0Or1Or2(const aclIntArray* param)
{
    if ((param->Size() != 0) && (param->Size() != 1) && (param->Size() != SIZE_2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "param size must be a single int, or a tuple of two ints");
        return false;
    }
    return true;
}

// Integer division rounding to -Infinity
static inline int64_t div_rtn(const int64_t x, const int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        --q;
    };
    return q;
}

// 计算经过MaxPool后的shape的h和w（n,c与input一致，不用计算）
static inline int64_t PoolingOutShape(
    const int64_t inputSize, const int64_t kernelSize, const int64_t pad_l, const int64_t pad_r, const int64_t stride,
    const int64_t dilation, const bool ceil_mode)
{
    int64_t outputSize =
        div_rtn(inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 + (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode) {
        if ((outputSize - 1) * stride >= inputSize + pad_l) {
            --outputSize;
        }
    }
    return outputSize;
}

static int64_t CeilDivUp(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return valueNum;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

static bool CheckGradInputAndIndicesShapeIsMask(
    const int64_t outHeight, const int64_t outWidth, const bool is3d, int64_t nInputPlane, bool isMask2ND, bool isMask,
    const int64_t kH, const int64_t kW, int64_t nBatch, const aclTensor* indices, const aclTensor* gradOutput,
    op::Shape calcOutShape)
{
    if (isMask) {
        const int64_t blockSize = 16;
        const int64_t maskH = kH * kW;
        const int64_t maskW = (CeilDivUp(outHeight * outWidth, blockSize) + 1) * INDICES_TYPE_CONVERT * BLOCKSIZE;
        const op::Shape calcIndicesShape =
            is3d ? op::Shape({nInputPlane, maskH, maskW}) : op::Shape({nBatch, nInputPlane, maskH, maskW});
        // 判断indices的shape与推导出的输出shape是否相等
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(indices, calcIndicesShape, return false);
    } else {
        if (!isMask2ND) {
            // indices与gradOutput的shape需要一致
            OP_CHECK_SHAPE_NOT_EQUAL(indices, gradOutput, return false);
        } else {
            if (calcOutShape.GetShapeSize() * sizeof(int32_t) >
                indices->GetViewShape().GetShapeSize() * sizeof(int8_t)) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "calIndicesSize should be smaller than indicesSize, but got calIndicesShapeSize:%ld, indices:%ld ",
                    calcOutShape.GetShapeSize(), indices->GetViewShape().GetShapeSize());
                return false;
            }
        }
    }

    return true;
}

static bool CheckGradInputAndIndicesShape(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const int64_t kH, const int64_t kW,
    const int64_t padH, const int64_t padW, const int64_t dH, const int64_t dW, const int64_t dilationH,
    const int64_t dilationW, const bool ceilMode, const bool isMask, bool isMask2ND, const aclTensor* gradInput)
{
    // gradInput与self的shape需要一致
    OP_CHECK_SHAPE_NOT_EQUAL(gradInput, self, return false);

    const bool is3d = self->GetViewShape().GetDimNum() == DIMENTION_3;
    const int64_t nBatch = is3d ? 1 : self->GetViewShape().GetDim(DIM_INX_0);
    int64_t nInputPlane = is3d ? self->GetViewShape().GetDim(DIM_INX_0) : self->GetViewShape().GetDim(DIM_INX_1);
    int64_t height = is3d ? self->GetViewShape().GetDim(DIM_INX_1) : self->GetViewShape().GetDim(DIM_INX_2);
    int64_t width = is3d ? self->GetViewShape().GetDim(DIM_INX_2) : self->GetViewShape().GetDim(DIM_INX_3);
    if (self->GetStorageFormat() == ge::FORMAT_NHWC) {
        nInputPlane = is3d ? self->GetViewShape().GetDim(DIM_INX_2) : self->GetViewShape().GetDim(DIM_INX_3);
        height = is3d ? self->GetViewShape().GetDim(DIM_INX_0) : self->GetViewShape().GetDim(DIM_INX_1);
        width = is3d ? self->GetViewShape().GetDim(DIM_INX_1) : self->GetViewShape().GetDim(DIM_INX_2);
    }
    OP_CHECK(
        ((nInputPlane != 0) && (height != 0) && (width != 0)),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected tensor\
        with optional 0 dim batch size, but got nInputPlane:%ld, height:%ld, width:%ld",
            nInputPlane, height, width),
        return false);
    OP_CHECK(
        padH <= ((kH - 1) * dilationH + 1) / 2,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "pad should be at most half of\
        effective kernel size, but got padH=%ld, kH=%ld and dilationH=%ld",
            padH, kH, dilationH),
        return false);
    OP_CHECK(
        padW <= ((kW - 1) * dilationW + 1) / 2,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "pad should be at most half of\
        effective kernel size, but got padW=%ld, kW=%ld and dilationW=%ld",
            padW, kW, dilationW),
        return false);

    const int64_t outHeight = PoolingOutShape(height, kH, padH, padH, dH, dilationH, ceilMode);
    const int64_t outWidth = PoolingOutShape(width, kW, padW, padW, dW, dilationW, ceilMode);
    OP_CHECK(
        (outHeight > 0 && outWidth > 0),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Given input size %ldx%ldx%ld, calc out size %ldx%ldx%ld", nInputPlane, height,
            width, nInputPlane, outHeight, outWidth),
        return false);

    op::Shape calcOutShape =
        is3d ? op::Shape({nInputPlane, outHeight, outWidth}) : op::Shape({nBatch, nInputPlane, outHeight, outWidth});
    if (self->GetStorageFormat() == ge::FORMAT_NHWC) {
        calcOutShape = is3d ? op::Shape({outHeight, outWidth, nInputPlane}) :
                              op::Shape({nBatch, outHeight, outWidth, nInputPlane});
    } // 判断out的shape与推导出的输出shape是否相等
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradOutput, calcOutShape, return false);

    CheckGradInputAndIndicesShapeIsMask(
        outHeight, outWidth, is3d, nInputPlane, isMask2ND, isMask, kH, kW, nBatch, indices, gradOutput, calcOutShape);

    return true;
}

static bool CheckShape(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, const bool isMask, bool isMask2ND)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(
        CheckAttrSize1Or2(kernelSize),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of kernelSize must be one or two, but got %zu.", kernelSize->Size()),
        return false);
    OP_CHECK(
        CheckAttrSize0Or1Or2(stride),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of stride must be zero, one or two, but got %zu.", stride->Size()),
        return false);
    OP_CHECK(
        CheckAttrSize1Or2(padding),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of padding must be one or two, but got %zu.", padding->Size()),
        return false);
    OP_CHECK(
        CheckAttrSize1Or2(dilation),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of dilation must be one or two, but got %zu.", dilation->Size()),
        return false);

    // 校验kernel的值
    const aclIntArray& kernelRef = *kernelSize;
    const int64_t kH = kernelRef[0];
    const int64_t kW = (kernelRef.Size() == 1) ? kH : kernelRef[1];
    OP_CHECK(
        ((kH > 0) && (kW > 0)),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of kernel should be greater than 0, but got kh:%ld, kw:%ld", kH, kW),
        return false);

    // 校验stride的值，stride的数值不能等于0
    const aclIntArray& strideRef = *stride;
    const int64_t dH = (strideRef.Size() == 0) ? kH : strideRef[0];
    const int64_t dW = (strideRef.Size() == 0) ? kW : ((strideRef.Size() == 1) ? dH : strideRef[1]);
    OP_CHECK(
        ((dH > 0) && (dW > 0)),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of stride should be greater than 0, but got dh:%ld, dw:%ld", dH, dW),
        return false);

    // 校验padding的值，padding的数值不能大于 kernelSize/2
    const aclIntArray& paddingRef = *padding;
    const int64_t padH = paddingRef[0];
    const int64_t padW = (paddingRef.Size() == 1) ? padH : paddingRef[1];
    const int RATIO_KERNEL_PAD = 2;
    OP_CHECK(
        ((padH >= 0) && (padW >= 0) && (padH <= (kH / RATIO_KERNEL_PAD)) && (padW <= (kW / RATIO_KERNEL_PAD))),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "pad should be >= 0 and <= half of kernel size, but got kh:%ld, kw:%ld, padH:%ld, padW:%ld", kH, kW, padH,
            padW),
        return false);

    // dilation目前算子只支持取值1
    const aclIntArray& dilationRef = *dilation;
    const int64_t dilationH = dilationRef[0];
    const int64_t dilationW = (dilationRef.Size() == 1) ? dilationH : dilationRef[1];
    if (Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        OP_CHECK(
            ((dilationH > 0) && (dilationW > 0)),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "dilation should be greater than 0, but got dilationH:%ld, dilationW:%ld",
                dilationH, dilationW),
            return false);
    } else {
        OP_CHECK(
            ((dilationH == 1) && (dilationW == 1)),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "dilation only support 1, but got dilationH:%ld, dilationW:%ld", dilationH,
                dilationW),
            return false);
    }

    // 检查GradInput和indices的shape
    const bool ret = CheckGradInputAndIndicesShape(
        gradOutput, self, indices, kH, kW, padH, padW, dH, dW, dilationH, dilationW, ceilMode, isMask, isMask2ND,
        gradInput);
    CHECK_RET(ret, false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, const bool isMask, bool isMask2ND)
{
    // 检查参数是否为空指针
    CHECK_RET(
        CheckNotNullPtr(gradOutput, self, indices, kernelSize, stride, padding, dilation, gradInput),
        ACLNN_ERR_PARAM_NULLPTR);

    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, indices, gradInput, isMask, isMask2ND), ACLNN_ERR_PARAM_INVALID);

    // 检查数据格式是否支持
    CHECK_RET(CheckFormat(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 检查数据维度是否合法
    CHECK_RET(
        CheckShape(
            gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput, isMask, isMask2ND),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline const aclTensor* View3Das4D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCL -> unsqueeze -> reformat -> NCHW
    // unsqueeze input into 4D
    const int64_t dim = 0;
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dim, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
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

static inline const aclTensor* View4Das3D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // NCHW -> squeeze -> reformat -> NCL
    // squeeze out into 3D
    const int64_t dim = 0;
    auto squeezedInput = l0op::SqueezeNd(input, dim, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCL
    auto reformatInput = l0op::ReFormat(squeezedInput, format);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static op::DataType GetCastDtypeForMask(const aclTensor* self, bool isMask2ND)
{
    /*
    fp32转fp16场景：910soc aclnn自定义mask语义接口通过转fp16方式兼容float32
    bf16转fp32场景：l0op::MaxPoolGradWithArgmaxV1不支持bf16，910B或910_93通过转fp32方式支持bf16
    其他场景：GetCastDtypeForMask返回为输入原数据类型，调l0op::Cast不会做cast
    */
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    auto selfType = self->GetDataType();
    if ((curArch == NpuArch::DAV_1001) && (selfType == op::DataType::DT_FLOAT)) {
        return op::DataType::DT_FLOAT16;
    }
    if (selfType == op::DataType::DT_BF16) {
        return op::DataType::DT_FLOAT;
    }
    if (isMask2ND && selfType == op::DataType::DT_FLOAT16) {
        return op::DataType::DT_FLOAT;
    }

    return selfType;
}

// 检查tuple <gradOut, self, indices>里的元素是否为非null。true表示为非null，false表示为null
static bool CheckTupleArrayNullptr(TupleArrayInput tensorTupleArray)
{
    return (
        (std::get<TUPLE_INX_GRAD>(tensorTupleArray) != nullptr) &&
        (std::get<TUPLE_INX_SELF>(tensorTupleArray) != nullptr) &&
        (std::get<TUPLE_INX_INDICES>(tensorTupleArray) != nullptr));
}

static const aclTensor* IndicesInputProcess(const aclTensor* indices, aclOpExecutor* executor)
{
    const bool isIndices3D = indices->GetViewShape().GetDimNum() == DIMENTION_3;

    Shape indxViewShape = indices->GetViewShape();
    int64_t n_batch = isIndices3D ? 1 : indxViewShape.GetDim(DIM_INX_0);
    int64_t n_plane = isIndices3D ? indxViewShape.GetDim(DIM_INX_0) : indxViewShape.GetDim(DIM_INX_1);
    int64_t indices_height = isIndices3D ? indxViewShape.GetDim(DIM_INX_1) : indxViewShape.GetDim(DIM_INX_2);
    int64_t indices_width = isIndices3D ? indxViewShape.GetDim(DIM_INX_2) : indxViewShape.GetDim(DIM_INX_3);
    op::Shape dimsViewShape = {n_batch, n_plane, indices_height, indices_width / INDICES_TYPE_CONVERT / BLOCKSIZE};
    op::Shape dimsStorageShape = {
        n_batch, n_plane, indices_height, indices_height, indices_width / INDICES_TYPE_CONVERT / BLOCKSIZE, BLOCKSIZE};

    auto indicesNew = executor->AllocTensor(
        dimsStorageShape, dimsViewShape, op::DataType::DT_UINT16, op::Format::FORMAT_NC1HWC0, op::Format::FORMAT_NCHW);
    CHECK_RET(indicesNew != nullptr, nullptr);
    indicesNew->SetFromWorkspace(false);
    indicesNew->SetStorageAddr(indices->GetStorageAddr());

    return indicesNew;
}

static const aclTensor* IndicesInputProcess3D(
    const aclTensor* indices, const aclTensor* gradOutput, aclOpExecutor* executor)
{
    // 处理2D属性适配3D
    Shape gradOutputShape = gradOutput->GetViewShape();
    int64_t n_batch = gradOutputShape.GetDim(DIM_INX_0);
    int64_t n_plane = gradOutputShape.GetDim(DIM_INX_1);
    int64_t indices_depth = gradOutputShape.GetDim(DIM_INX_2);
    int64_t indices_height = gradOutputShape.GetDim(DIM_INX_3);
    int64_t indices_width = gradOutputShape.GetDim(DIM_INX_4);
    op::Shape dimsViewShape = {n_batch, n_plane, indices_depth, indices_height, indices_width};
    op::Shape dimsStorageShape = {n_batch, n_plane, indices_depth, indices_height, indices_width};

    auto indicesNew = executor->AllocTensor(
        dimsStorageShape, dimsViewShape, op::DataType::DT_INT32, op::Format::FORMAT_NCDHW, op::Format::FORMAT_NCDHW);
    CHECK_RET(indicesNew != nullptr, nullptr);
    indicesNew->SetFromWorkspace(false);
    indicesNew->SetStorageAddr(indices->GetStorageAddr());

    return indicesNew;
}

static TupleArrayInput InputProcess(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const bool isMask, bool isMask2ND,
    aclOpExecutor* executor)
{
    if (isMask) {
        /* 自定义mask语义接口时，将输入转为5HD格式 */
        int64_t groups = 1;
        TupleArrayInput nullptrArray =
            std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>(nullptr, nullptr, nullptr);

        auto castGradOutput = l0op::Cast(gradOutput, GetCastDtypeForMask(self, isMask2ND), executor);
        OP_CHECK(
            castGradOutput != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The gradOutput Cast return nullptr."),
            return nullptrArray);
        auto castSelf = l0op::Cast(self, GetCastDtypeForMask(self, isMask2ND), executor);
        OP_CHECK(
            castSelf != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The self Cast return nullptr."),
            return nullptrArray);

        auto gradOutputTrans = l0op::TransDataSpecial(castGradOutput, Format::FORMAT_NC1HWC0, groups, executor);
        OP_CHECK(
            gradOutputTrans != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The gradOutput TransDataSpecial return nullptr."), return nullptrArray);

        auto selfTrans = l0op::TransDataSpecial(castSelf, Format::FORMAT_NC1HWC0, groups, executor);
        OP_CHECK(
            selfTrans != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The self TransDataSpecial return nullptr."),
            return nullptrArray);

        // 通过创建一个数据地址指向indices数据地址，但数据类型为UINT16的tensor，将返回结果的数据类型从UINT16转换为INT8
        auto indicesNew = IndicesInputProcess(indices, executor);
        OP_CHECK(
            indicesNew != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The indices process return nullptr."),
            return nullptrArray);

        return std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>(gradOutputTrans, selfTrans, indicesNew);
    } else {
        TupleArrayInput nullptrArray =
            std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>(nullptr, nullptr, nullptr);
        auto castGradOutput = l0op::Cast(gradOutput, GetCastDtypeForMask(self, isMask2ND), executor);
        OP_CHECK(
            castGradOutput != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The gradOutput Cast return nullptr."),
            return nullptrArray);
        auto castSelf = l0op::Cast(self, GetCastDtypeForMask(self, isMask2ND), executor);
        OP_CHECK(
            castSelf != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The self Cast return nullptr."),
            return nullptrArray);
        const bool isGrad3D = gradOutput->GetViewShape().GetDimNum() == DIMENTION_3;
        Shape gradViewShape = gradOutput->GetViewShape();
        const int64_t nBatch = isGrad3D ? 1 : gradViewShape.GetDim(DIM_INX_0);
        const int64_t nPlane = isGrad3D ? gradViewShape.GetDim(DIM_INX_0) : gradViewShape.GetDim(DIM_INX_1);
        const int64_t heightOut = isGrad3D ? gradViewShape.GetDim(DIM_INX_1) : gradViewShape.GetDim(DIM_INX_2);
        const int64_t widthOut = isGrad3D ? gradViewShape.GetDim(DIM_INX_2) : gradViewShape.GetDim(DIM_INX_3);
        op::Shape dimsViewShape = {nBatch, nPlane, heightOut, widthOut};
        op::Shape dimsStorageShape = {nBatch, nPlane, heightOut, widthOut};
        auto indicesNew = executor->AllocTensor(
            dimsStorageShape, dimsViewShape, op::DataType::DT_INT32, op::Format::FORMAT_NCHW, op::Format::FORMAT_NCHW);
        OP_CHECK(
            indicesNew != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The indices process return nullptr."),
            return nullptrArray);
        indicesNew->SetFromWorkspace(false);
        indicesNew->SetStorageAddr(indices->GetStorageAddr());
        return std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>(castGradOutput, castSelf, indicesNew);
    }
}

static const aclTensor* OutputProcess(
    const aclTensor* gradInput, const aclTensor* self, const bool isMask, aclOpExecutor* executor)
{
    if (isMask) {
        /* 自定义mask语义接口时，将输出从5HD转为NCHW格式 */
        int64_t groups = 1;
        auto gradInputTrans = l0op::TransDataSpecial(gradInput, Format::FORMAT_NCHW, groups, executor);
        OP_CHECK(
            gradInputTrans != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The gradInput TransDataSpecial return nullptr."), return nullptr);

        auto castGradInput = l0op::Cast(gradInputTrans, self->GetDataType(), executor);
        OP_CHECK(
            castGradInput != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The gradOutput Cast return nullptr."),
            return nullptr);

        return castGradInput;
    } else {
        auto castGradInput = l0op::Cast(gradInput, self->GetDataType(), executor);
        OP_CHECK(
            castGradInput != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The gradOutput Cast return nullptr."),
            return nullptr);

        return castGradInput;
    }
}

static aclnnStatus ExecMaxPool2dWithIndicesBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, const bool isMask, bool isMask2ND, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(
        gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput, isMask, isMask2ND);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查是否是空tensor
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutpoutContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutpoutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 自定义mask语义接口时，indices不支持不连续Tensor
    auto indicesContiguous = isMask ? indices : l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果是3d，需要扩到4d再调用MaxPoolGradWithArgMaxV1接口
    const bool isSelf3D = self->GetViewShape().GetDimNum() == DIMENTION_3;

    auto selfUnsqueezed = isSelf3D ? View3Das4D(selfContiguous, uniqueExecutor.get()) : selfContiguous;
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputUnsqueezed =
        isSelf3D ? View3Das4D(gradOutpoutContiguous, uniqueExecutor.get()) : gradOutpoutContiguous;
    CHECK_RET(gradOutputUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 自定义mask语义接口时，indices不在此处进行维度转换
    auto indicesUnsqueezed =
        (isSelf3D && !isMask) ? View3Das4D(indicesContiguous, uniqueExecutor.get()) : indicesContiguous;
    CHECK_RET(indicesUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 对输入进行处理，包括5HD格式转换
    auto inputResult =
        InputProcess(gradOutputUnsqueezed, selfUnsqueezed, indicesUnsqueezed, isMask, isMask2ND, uniqueExecutor.get());
    CHECK_RET(CheckTupleArrayNullptr(inputResult), ACLNN_ERR_INNER_NULLPTR);

    auto maxPoolGradResult = l0op::MaxPoolGradWithArgmaxV1(
        std::get<TUPLE_INX_GRAD>(inputResult), std::get<TUPLE_INX_SELF>(inputResult),
        std::get<TUPLE_INX_INDICES>(inputResult), kernelSize, stride, padding, dilation, ceilMode,
        uniqueExecutor.get());
    CHECK_RET(maxPoolGradResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputResult = OutputProcess(maxPoolGradResult, self, isMask, uniqueExecutor.get());
    CHECK_RET(gradInputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& dstFormat = gradInput->GetStorageFormat();
    auto gradResultSqueezed = isSelf3D ? View4Das3D(gradInputResult, dstFormat, uniqueExecutor.get()) : gradInputResult;
    CHECK_RET(gradResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(gradResultSqueezed, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

static std::tuple<aclIntArray*, aclIntArray*, aclIntArray*, aclIntArray*> ProcessAttrTo3DParams(
    const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, aclOpExecutor* executor)
{
    // 处理 kernel 尺寸
    const aclIntArray& kernelRef = *kernelSize;
    const int64_t kernelH = kernelRef[0];
    const int64_t kernelW = (kernelRef.Size() == 1) ? kernelH : kernelRef[1];

    // 处理步长
    const aclIntArray& strideRef = *stride;
    const int64_t strideH = (strideRef.Size() == 0) ? kernelH : strideRef[0];
    const int64_t strideW = (strideRef.Size() == 0) ? kernelW : ((strideRef.Size() == 1) ? strideH : strideRef[1]);

    // 处理填充
    const aclIntArray& paddingRef = *padding;
    const int64_t paddingH = paddingRef[0];
    const int64_t paddingW = (paddingRef.Size() == 1) ? paddingH : paddingRef[1];

    // 构建三维参数数组
    FVector<int64_t> kernelSizeData{1, kernelH, kernelW};
    FVector<int64_t> strideSizeData{1, strideH, strideW};
    FVector<int64_t> paddingSizeData{0, paddingH, paddingW};
    FVector<int64_t> dilationSizeData{1, 1, 1};

    // 分配内存并返回
    aclIntArray* kernelSize3D = executor->AllocIntArray(kernelSizeData.data(), SIZE_3);
    aclIntArray* stride3D = executor->AllocIntArray(strideSizeData.data(), SIZE_3);
    aclIntArray* padding3D = executor->AllocIntArray(paddingSizeData.data(), SIZE_3);
    aclIntArray* dilation3D = executor->AllocIntArray(dilationSizeData.data(), SIZE_3);
    return {kernelSize3D, stride3D, padding3D, dilation3D};
}

static aclnnStatus ExecMaxPool2dWithIndicesBackwardGetWorkspaceSizeV3(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(
        gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput, false, false);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查是否是空tensor
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutpoutContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutpoutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果self是3d，需要扩到4d再调用MaxPoolGradWithArgMaxV3接口
    const bool isSelf3D = self->GetViewShape().GetDimNum() == DIMENTION_3;

    auto selfUnsqueezed =
        isSelf3D ? View3Das4D(selfContiguous, uniqueExecutor.get()) : selfContiguous; // 正向此处是：self
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputUnsqueezed = isSelf3D ? View3Das4D(gradOutpoutContiguous, uniqueExecutor.get()) :
                                           gradOutpoutContiguous; // 正向此处是：gradOutpout
    CHECK_RET(gradOutputUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 自定义mask语义接口时，indices不在此处进行维度转换
    auto indicesUnsqueezed =
        isSelf3D ? View3Das4D(indicesContiguous, uniqueExecutor.get()) : indicesContiguous; // 正向此处是：indices
    CHECK_RET(indicesUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::string dataFormat = (self->GetStorageFormat() == ge::FORMAT_NHWC) ? "NHWC" : "NCHW";
    auto maxPoolGradResult = l0op::MaxPoolGradWithArgmaxV3(
        gradOutputUnsqueezed, selfUnsqueezed, indicesUnsqueezed, kernelSize, stride, padding, indices->GetDataType(),
        dilation, ceilMode, dataFormat, uniqueExecutor.get());
    CHECK_RET(maxPoolGradResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputResult = OutputProcess(maxPoolGradResult, self, false, uniqueExecutor.get());
    CHECK_RET(gradInputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& dstFormat = gradInput->GetStorageFormat();
    auto gradResultSqueezed = isSelf3D ? View4Das3D(gradInputResult, dstFormat, uniqueExecutor.get()) : gradInputResult;
    CHECK_RET(gradResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(gradResultSqueezed, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecMaxPool3dWithIndicesBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, const bool isMask, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(
        gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput, isMask, false);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查是否是空tensor
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutpoutContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutpoutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果是3d/4d，需要扩到5d再调用MaxPoolGradWithArgMaxV1接口
    const bool isSelf3D = selfContiguous->GetViewShape().GetDimNum() == DIMENTION_3;
    auto selfUnsqueezed = isSelf3D ? View3Das5D(selfContiguous, uniqueExecutor.get()) :
                                     View4Das5D(selfContiguous, uniqueExecutor.get()); // 检查下是否会出现其他情况
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputUnsqueezed = isSelf3D ? View3Das5D(gradOutpoutContiguous, uniqueExecutor.get()) :
                                           View4Das5D(gradOutpoutContiguous, uniqueExecutor.get());
    CHECK_RET(gradOutputUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 取其中有效数据转为5HD
    auto indicesUnsqueezed = isMask ?
                                 IndicesInputProcess3D(indicesContiguous, gradOutputUnsqueezed, uniqueExecutor.get()) :
                                 (isSelf3D ? View3Das5D(indicesContiguous, uniqueExecutor.get()) :
                                             View4Das5D(indicesContiguous, uniqueExecutor.get()));
    CHECK_RET(indicesUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 处理输入属性适配5D
    auto [kernel3D, stride3D, padding3D, dilation3D] =
        ProcessAttrTo3DParams(kernelSize, stride, padding, uniqueExecutor.get());
    CHECK_RET(
        (kernel3D != nullptr) && (stride3D != nullptr) && (padding3D != nullptr) && (dilation3D != nullptr),
        ACLNN_ERR_INNER_NULLPTR);
    auto maxPoolGradResult = l0op::MaxPool3DGradWithArgmax(
        gradOutputUnsqueezed, selfUnsqueezed, indicesUnsqueezed, kernel3D, stride3D, padding3D, dilation3D, ceilMode,
        uniqueExecutor.get());
    CHECK_RET(maxPoolGradResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& dstFormat = gradInput->GetStorageFormat();
    auto gradResultSqueezed = isSelf3D ? View5Das3D(maxPoolGradResult, dstFormat, uniqueExecutor.get()) :
                                         View5Das4D(maxPoolGradResult, dstFormat, uniqueExecutor.get());
    CHECK_RET(gradResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(gradResultSqueezed, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(
        !(Ops::NN::AclnnUtil::IsRegbase(curArch)),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Please use aclnnMaxPool2dWithIndicesBackward."),
        return ACLNN_ERR_PARAM_INVALID);
    // 如果kernelSize符合切ND要求，在这里把isMask设置为false
    OP_CHECK_NULL(kernelSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(gradOutput, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(
        CheckAttrSize1Or2(kernelSize),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "param size must be a single int, or a tuple of two ints.\
                    stride can be empty. kernelSize:%zu, stride:%zu, padding:%zu dilation:%zu",
            kernelSize->Size(), stride->Size(), padding->Size(), dilation->Size()),
        return ACLNN_ERR_PARAM_NULLPTR);
    const aclIntArray& kernelRef = *kernelSize;
    const int64_t kH = kernelRef[0];
    const int64_t kW = (kernelRef.Size() == 1) ? kH : kernelRef[1];
    if (!(kH == 1 && kW == 1) && (curArch == NpuArch::DAV_2201)) {
        L2_DFX_PHASE_1(
            aclnnMaxPool2dWithMaskBackward,
            DFX_IN(gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode), DFX_OUT(gradInput));
        return ExecMaxPool3dWithIndicesBackwardGetWorkspaceSize(
            gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput, true, workspaceSize,
            executor);
    } else {
        L2_DFX_PHASE_1(
            aclnnMaxPool2dWithMaskBackward,
            DFX_IN(gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode), DFX_OUT(gradInput));
        return ExecMaxPool2dWithIndicesBackwardGetWorkspaceSize(
            gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput, true, false,
            workspaceSize, executor);
    }
}

aclnnStatus aclnnMaxPool2dWithMaskBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxPool2dWithMaskBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    aclnnStatus ret;
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    L2_DFX_PHASE_1(
        aclnnMaxPool2dWithIndicesBackward,
        DFX_IN(gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode), DFX_OUT(gradInput));

    if (Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        ret = ExecMaxPool2dWithIndicesBackwardGetWorkspaceSizeV3(
            gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput, workspaceSize,
            executor);
    } else {
        ret = ExecMaxPool3dWithIndicesBackwardGetWorkspaceSize(
            gradOutput, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradInput, false, workspaceSize,
            executor);
    }

    return ret;
}

aclnnStatus aclnnMaxPool2dWithIndicesBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxPool2dWithIndicesBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif