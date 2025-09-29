/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_pool2d_with_indices.h"
#include "level0/max_pool_with_argmax_v1.h"
#include "max_pool3d_with_argmax_v2.h"
#include "level0/max_pool_with_argmax_v3.h"
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

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

using TupleArrayOut = std::tuple<const aclTensor*, const aclTensor*>;

static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_NULL_LIST = {};
// 输入为ND场景下，1980不支持任何数据类型
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_910_LIST = {};
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> SELF_OUT_DTYPE_SUPPORT_310P_LIST = {op::DataType::DT_FLOAT};
// 输入为5HD场景下
static const std::initializer_list<op::DataType> MASK_SELF_OUT_DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MASK_SELF_OUT_DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_INDICES_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64, op::DataType::DT_INT32};

static const int DIMENTION_3 = 3;
static const int DIMENTION_4 = 4;
static const int SIZE_1 = 1;
static const int SIZE_2 = 2;
static const int SIZE_3 = 3;

static const int DIM_INX_0 = 0;
static const int DIM_INX_1 = 1;
static const int DIM_INX_2 = 2;
static const int DIM_INX_3 = 3;

static const int INDICES_TYPE_CONVERT = 2;
static const int64_t BLOCKSIZE = 16;
static const int KERNEL_SIZE_32 = 32;
static const int KERNEL_SIZE_64 = 64;

namespace {
static bool CheckNotNullPtr(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, aclTensor* out, aclTensor* indices)
{
    // self、kernelSize、stride、padding、dilation、out、indices不能为空指针
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(dilation, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(indices, return false);
    return true;
}

static const inline std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion(bool isMask, bool isMask2ND)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_95:
            return MASK_SELF_OUT_DTYPE_SUPPORT_910B_LIST;
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910B: {
            return (
                isMask    ? MASK_SELF_OUT_DTYPE_SUPPORT_910B_LIST :
                isMask2ND ? MASK_SELF_OUT_DTYPE_SUPPORT_910B_LIST :
                            SELF_OUT_DTYPE_SUPPORT_910B_LIST);
        }
        case SocVersion::ASCEND910: {
            return (isMask ? MASK_SELF_OUT_DTYPE_SUPPORT_910_LIST : SELF_OUT_DTYPE_SUPPORT_910_LIST);
        }
        case SocVersion::ASCEND310P: {
            return (isMask ? SELF_OUT_DTYPE_SUPPORT_310P_LIST : SELF_OUT_DTYPE_SUPPORT_910_LIST);
        }
        default: {
            return SELF_OUT_DTYPE_NULL_LIST;
        }
    }
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* out, const aclTensor* indices, bool isMask, bool isMask2ND)
{
    // 检查self的数据类型是否在支持列表内
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion(isMask, isMask2ND);
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    // 检查indices的数据类型是否在支持列表内
    if (isMask) {
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, MASK_INDICES_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_DTYPE_SUPPORT_LIST, return false);
    }

    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and output should be same, self [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    OP_CHECK(
        !op::IsPrivateFormat(self->GetStorageFormat()),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCHW or CHW"), return false);

    // 仅支持NCHW、CHW
    if (self->GetViewShape().GetDimNum() != DIMENTION_3 && self->GetViewShape().GetDimNum() != DIMENTION_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "3D or 4D tensor expected for inputs");
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

// Integer division rounding to -Infinity
static inline int64_t div_rtn(int64_t x, int64_t y)
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
    int64_t inputSize, int64_t kernelSize, int64_t pad_l, int64_t pad_r, int64_t stride, int64_t dilation,
    bool ceil_mode)
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

static bool CheckOutAndIndicesShape(
    const aclTensor* self, int64_t kH, int64_t kW, int64_t padH, int64_t padW, int64_t dH, int64_t dW,
    int64_t dilationH, int64_t dilationW, bool ceilMode, bool isMask, bool isMask2ND, const aclTensor* out,
    const aclTensor* indices)
{
    const bool is3d = self->GetViewShape().GetDimNum() == DIMENTION_3;
    const int64_t nBatch = is3d ? 1 : self->GetViewShape().GetDim(DIM_INX_0);

    uint32_t nInputPlaneIdx = is3d ? DIM_INX_0 : DIM_INX_1;
    uint32_t heightIdx = is3d ? DIM_INX_1 : DIM_INX_2;
    uint32_t widthIdx = is3d ? DIM_INX_2 : DIM_INX_3;
    if (self->GetStorageFormat() == ge::FORMAT_NHWC) {
        nInputPlaneIdx += DIM_INX_2;
        heightIdx--;
        widthIdx--;
    }
    const int64_t nInputPlane = self->GetViewShape().GetDim(nInputPlaneIdx);
    const int64_t height = self->GetViewShape().GetDim(heightIdx);
    const int64_t width = self->GetViewShape().GetDim(widthIdx);

    OP_CHECK(
        ((nInputPlane != 0) && (height != 0) && (width != 0)),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected tensor with optional 0 dim batch size, but got nInputPlane:%ld, height:%ld, width:%ld",
            nInputPlane, height, width),
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
    }

    // 判断out的shape与推导出的输出shape是否相等
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, calcOutShape, return false);

    if (isMask) {
        const int64_t maskH = kH * kW;
        // UINT16转为INT8，indices shape的最后一维的大小需*2
        const int64_t maskW = (CeilDivUp(outHeight * outWidth, BLOCKSIZE) + 1) * INDICES_TYPE_CONVERT * BLOCKSIZE;
        const op::Shape calcIndicesShape =
            is3d ? op::Shape({nInputPlane, maskH, maskW}) : op::Shape({nBatch, nInputPlane, maskH, maskW});
        // 判断indices的shape与推导出的输出shape是否相等
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(indices, calcIndicesShape, return false);
    } else {
        if (!isMask2ND) {
            // indices与out的shape需要一致
            OP_CHECK_SHAPE_NOT_EQUAL(indices, out, return false);
        } else {
            // ND场景下，推导出的Indices的shape与输出Out的shape相等
            if (calcOutShape.GetShapeSize() * sizeof(int32_t) >
                indices->GetViewShape().GetShapeSize() * sizeof(int8_t)) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "calIndicesShapeSize should be smaller than indicesShapeSize, but got calIndicesShapeSize:%ld, "
                    "indices:%ld ",
                    calcOutShape.GetShapeSize(), indices->GetViewShape().GetShapeSize());
                return false;
            }
        }
    }
    return true;
}
static bool CheckShape(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, bool isMask, bool isMask2ND)
{
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

    // 校验padding的值，padding的数值不能小于 0
    if (padH < 0 || padW < 0) {
        OP_LOGW(
            "Negative pad inputs are not supported, and the output precision is not guaranteed when the pad is "
            "negative.");
    }

    // dilation目前算子只支持取值1
    const aclIntArray& dilationRef = *dilation;
    const int64_t dilationH = dilationRef[0];
    const int64_t dilationW = (dilationRef.Size() == 1) ? dilationH : dilationRef[1];
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
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

    // 检查out和indices的shape
    const bool ret = CheckOutAndIndicesShape(
        self, kH, kW, padH, padW, dH, dW, dilationH, dilationW, ceilMode, isMask, isMask2ND, out, indices);
    CHECK_RET(ret, false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, bool isMask, bool isMask2ND)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNullPtr(self, kernelSize, stride, padding, dilation, out, indices), ACLNN_ERR_PARAM_NULLPTR);

    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out, indices, isMask, isMask2ND), ACLNN_ERR_PARAM_INVALID);

    // 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查数据维度是否合法
    CHECK_RET(
        CheckShape(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, isMask, isMask2ND),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 检查tuple <values, indices>里的元素是否为非null。true表示为非null，false表示为null
static bool CheckTupleNullptr(std::tuple<const aclTensor*, const aclTensor*> tensorTuple)
{
    static const int RESULT_NUM = 2;
    if (std::tuple_size<decltype(tensorTuple)>::value != RESULT_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The length of tuple returned by MaxPoolWithArgmaxV1 is not 2.");
        return false;
    }

    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
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

static inline const aclTensor* View4Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCHW -> unsqueeze -> reformat -> NC1HW(NCDHW, D=1)
    // unsqueeze input into 5D
    const int64_t dim = 2; // Unsqueeze at dimension 2
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dim, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View5Das4D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // NC1HW(NCDHW, D=1) -> squeeze -> reformat -> NCHW
    // squeeze out into 4D
    const int64_t dim = 2; // Squeeze out dimension 0
    auto squeezedInput = l0op::SqueezeNd(input, dim, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View3Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // CHW -> unsqueeze -> reformat -> 1C1HW(NCDHW, N=1, D=1)
    // unsqueeze input into 5D
    FVector<int64_t> dimData{0, 2}; // Unsqueeze at dimension 0, 2
    const aclIntArray* dim2 = executor->AllocIntArray(dimData.data(), 2);
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dim2, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View5Das3D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // 1C1HW(NCDHW, N=1, D=1) -> squeeze -> reformat -> CHW
    // squeeze out into 3D
    FVector<int64_t> dimData{0, 2}; // Unsqueeze at dimension 0, 2
    const aclIntArray* dim2 = executor->AllocIntArray(dimData.data(), 2);
    auto squeezedInput = l0op::SqueezeNd(input, dim2, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format, executor);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static op::DataType GetCastDtypeForMask(const aclTensor* self, bool isMask2ND)
{
    /*
    fp32转fp16场景：910soc aclnn自定义mask语义接口通过转fp16方式兼容float32
    bf16转fp32场景：l0op::MaxPoolWithArgmaxV1不支持bf16，910B或910_93通过转fp32方式支持bf16
    其他场景：GetCastDtypeForMask返回为输入原数据类型，调l0op::Cast不会做cast
    */
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto selfType = self->GetDataType();
    if (socVersion == SocVersion::ASCEND910 && selfType == op::DataType::DT_FLOAT) {
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

static const aclTensor* SelfPreProcess(const aclTensor* self, bool isMask, bool isMask2ND, aclOpExecutor* executor)
{
    auto castSelf = l0op::Cast(self, GetCastDtypeForMask(self, isMask2ND), executor);
    OP_CHECK(castSelf != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The self Cast return nullptr."), return nullptr);
    if (isMask) {
        /* 自定义mask语义接口时，将输入转换为5HD */
        int64_t groups = 1;
        auto selfTrans = l0op::TransDataSpecial(castSelf, Format::FORMAT_NC1HWC0, groups, executor);
        OP_CHECK(
            selfTrans != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The TransDataSpecial return nullptr."),
            return nullptr);
        return selfTrans;
    } else {
        return castSelf;
    }
}

static TupleArrayOut OutputProcess(
    TupleArrayOut output, const aclTensor* self, const aclTensor* indicesInput, bool isMask, aclOpExecutor* executor)
{
    TupleArrayOut nullptrArray = std::tuple<const aclTensor*, const aclTensor*>(nullptr, nullptr);
    const aclTensor* castOut = nullptr;
    if (isMask) {
        /* 自定义接口时，将输出out从5HD转为NCHW格式 */
        int64_t groups = 1;
        auto outTrans = l0op::TransDataSpecial(std::get<0>(output), Format::FORMAT_NCHW, groups, executor);
        OP_CHECK(
            outTrans != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The out TransDataSpecial return nullptr."),
            return nullptrArray);

        castOut = l0op::Cast(outTrans, self->GetDataType(), executor);
        OP_CHECK(
            castOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The out Cast return nullptr."), return nullptrArray);
    } else {
        castOut = l0op::Cast(std::get<0>(output), self->GetDataType(), executor);
        OP_CHECK(
            castOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The out Cast return nullptr."), return nullptrArray);
    }
    /* 由于算子输出的indices的数据内存，是按照NCHW方式格式排布的mask值。通过指定输出indices内存地址的方式，
    ** 直接将算子结果放到指定地址，无需进行ViewCopy */
    auto indicesOutput = std::get<1>(output);
    indicesOutput->SetStorageAddr(indicesInput->GetStorageAddr());
    indicesOutput->SetFromWorkspace(false);
    return std::tuple<const aclTensor*, const aclTensor*>(castOut, indicesOutput);
}

static aclnnStatus ExecMaxPool2dWithIndicesGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, bool isMask, bool isMask2ND,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, isMask, isMask2ND);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查是否是空tensor（算子不支持空tensor）
    if (self->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果self是3d，需要扩到4d，再调用MaxPoolWithArgMaxV1接口
    const bool isSelf3D = self->GetViewShape().GetDimNum() == DIMENTION_3;
    auto selfUnsqueezed = isSelf3D ? View3Das4D(selfContiguous, uniqueExecutor.get()) : selfContiguous;
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 对self进行预处理，如果是自定义接口，则转化为5HD格式
    auto selfPreResult = SelfPreProcess(selfUnsqueezed, isMask, isMask2ND, uniqueExecutor.get());
    CHECK_RET(selfPreResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 返回tuple，包含out和indices
    auto maxPoolResult =
        l0op::MaxPoolWithArgmaxV1(selfPreResult, kernelSize, stride, padding, dilation, ceilMode, uniqueExecutor.get());
    CHECK_RET(CheckTupleNullptr(maxPoolResult), ACLNN_ERR_INNER_NULLPTR);

    // 对返回结果进行处理，如果是自定义接口，则将out从5HD格式转换为NCHW格式
    auto outputResult = OutputProcess(maxPoolResult, self, indices, isMask, uniqueExecutor.get());
    CHECK_RET(CheckTupleNullptr(outputResult), ACLNN_ERR_INNER_NULLPTR);

    auto maxPoolOutResult = std::get<0>(outputResult);
    CHECK_RET(maxPoolOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& dstFormat = out->GetStorageFormat();
    auto outSqueezed = isSelf3D ? View4Das3D(maxPoolOutResult, dstFormat, uniqueExecutor.get()) : maxPoolOutResult;
    CHECK_RET(outSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto outResult = l0op::ViewCopy(outSqueezed, out, uniqueExecutor.get());
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

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

static aclnnStatus ExecMaxPool2dWithIndicesReshape3DGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, [[maybe_unused]]bool isMask,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, true, false);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查是否是空tensor（算子不支持空tensor）
    if (self->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果self是3d，需要扩到4d，再调用MaxPoolWithArgMaxV1接口
    const bool isSelf3D = self->GetViewShape().GetDimNum() == DIMENTION_3;
    auto selfUnsqueezed =
        isSelf3D ? View3Das5D(selfContiguous, uniqueExecutor.get()) : View4Das5D(selfContiguous, uniqueExecutor.get());

    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 更新2D属性适配3D
    auto [kernel3D, stride3D, padding3D, dilation3D] =
        ProcessAttrTo3DParams(kernelSize, stride, padding, uniqueExecutor.get());
    CHECK_RET(
        (kernel3D != nullptr) && (stride3D != nullptr) && (padding3D != nullptr) && (dilation3D != nullptr),
        ACLNN_ERR_INNER_NULLPTR);
    // 调用3D接口
    std::string dataFormat = "NCDHW";
    auto [outResult, indicesResult] = l0op::MaxPool3DWithArgmaxV2Ncdhw(
        selfUnsqueezed, kernel3D, stride3D, padding3D, dilation3D, ceilMode, dataFormat, uniqueExecutor.get());
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const op::Format& outFormat = out->GetStorageFormat();
    auto outResultSqueezed = isSelf3D ? View5Das3D(outResult, outFormat, uniqueExecutor.get()) :
                                        View5Das4D(outResult, outFormat, uniqueExecutor.get());
    CHECK_RET(outResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* indicesResultSqueezed = isSelf3D ? View5Das3D(indicesResult, outFormat, uniqueExecutor.get()) :
                                                        View5Das4D(indicesResult, outFormat, uniqueExecutor.get());
    CHECK_RET(indicesResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    indicesResultSqueezed->SetStorageAddr(indices->GetStorageAddr());
    indicesResultSqueezed->SetFromWorkspace(false);
    auto outViewCopyResult = l0op::ViewCopy(outResultSqueezed, out, uniqueExecutor.get());
    CHECK_RET(outViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecMaxPool3dWithIndicesGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, false, false);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查是否是空tensor（算子不支持空tensor）
    if (self->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果self是3d，需要扩到4d，再调用MaxPoolWithArgMaxV1接口
    const bool isSelf3D = self->GetViewShape().GetDimNum() == DIMENTION_3;

    auto selfUnsqueezed =
        isSelf3D ? View3Das5D(selfContiguous, uniqueExecutor.get()) : View4Das5D(selfContiguous, uniqueExecutor.get());
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // change attr
    const aclIntArray& kernelRef = *kernelSize;
    const int64_t kernelH = kernelRef[0];
    const int64_t kernelW = (kernelRef.Size() == 1) ? kernelH : kernelRef[1];

    const aclIntArray& strideRef = *stride;
    const int64_t strideH = (strideRef.Size() == 0) ? kernelH : strideRef[0];
    const int64_t strideW = (strideRef.Size() == 0) ? kernelW : ((strideRef.Size() == 1) ? strideH : strideRef[1]);

    const aclIntArray& paddingRef = *padding;
    const int64_t paddingH = paddingRef[0];
    const int64_t paddingW = (paddingRef.Size() == 1) ? paddingH : paddingRef[1];

    FVector<int64_t> kernelSizeData{1, kernelH, kernelW};
    FVector<int64_t> strideSizeData{1, strideH, strideW};
    FVector<int64_t> paddingSizeData{0, paddingH, paddingW};
    FVector<int64_t> dilationSizeData{1, 1, 1};

    aclIntArray* kernelSize3 = uniqueExecutor.get()->AllocIntArray(kernelSizeData.data(), 3);
    aclIntArray* stride3 = uniqueExecutor.get()->AllocIntArray(strideSizeData.data(), 3);
    aclIntArray* padding3 = uniqueExecutor.get()->AllocIntArray(paddingSizeData.data(), 3);
    aclIntArray* dilation3 = uniqueExecutor.get()->AllocIntArray(dilationSizeData.data(), 3);

    std::string dataFormat = "NCDHW";
    // Returns a tuple containing out and indices
    auto [outResult, indicesResult] = l0op::MaxPool3DWithArgmaxV2Ncdhw(
        selfUnsqueezed, kernelSize3, stride3, padding3, dilation3, ceilMode, dataFormat, uniqueExecutor.get());

    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& outFormat = out->GetStorageFormat();
    auto outResultSqueezed = isSelf3D ? View5Das3D(outResult, outFormat, uniqueExecutor.get()) :
                                        View5Das4D(outResult, outFormat, uniqueExecutor.get());
    CHECK_RET(outResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& indicesFormat = indices->GetStorageFormat();
    auto indicesResultSqueezed = isSelf3D ? View5Das3D(indicesResult, indicesFormat, uniqueExecutor.get()) :
                                            View5Das4D(indicesResult, outFormat, uniqueExecutor.get());
    CHECK_RET(indicesResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CheckReduceOutShape(outResultSqueezed, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckReduceOutShape(indicesResultSqueezed, indices), ACLNN_ERR_PARAM_INVALID);

    if (indicesResultSqueezed->GetDataType() != indices->GetDataType()) {
        indicesResultSqueezed = l0op::Cast(indicesResultSqueezed, indices->GetDataType(), uniqueExecutor.get());
        CHECK_RET(indicesResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // If the out is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewOutCopyResult = l0op::ViewCopy(outResultSqueezed, out, uniqueExecutor.get());
    CHECK_RET(viewOutCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If the indices is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewIndicesCopyResult = l0op::ViewCopy(indicesResultSqueezed, indices, uniqueExecutor.get());
    CHECK_RET(viewIndicesCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecMaxPool2dWithIndicesGetWorkspaceSizeV3(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, const bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, false, false);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查是否是空tensor（算子不支持空tensor）
    if (self->IsEmpty() || out->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // self不支持非连续，如果self是3d，需要扩到4d，再调用MaxPoolWithArgMaxV3接口
    const bool isSelf3D = self->GetViewShape().GetDimNum() == DIMENTION_3;
    auto selfUnsqueezed = isSelf3D ? View3Das4D(selfContiguous, uniqueExecutor.get()) : selfContiguous;
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::string dataFormat = (self->GetStorageFormat() == ge::FORMAT_NHWC) ? "NHWC" : "NCHW";

    // 返回tuple，包含out和indices
    auto maxPoolResult = l0op::MaxPoolWithArgmaxV3(
        selfUnsqueezed, kernelSize, stride, padding, indices->GetDataType(), dilation, ceilMode, dataFormat,
        uniqueExecutor.get());
    CHECK_RET(CheckTupleNullptr(maxPoolResult), ACLNN_ERR_INNER_NULLPTR);

    auto maxPoolOutResult = std::get<0>(maxPoolResult);
    CHECK_RET(maxPoolOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& dstFormat = out->GetStorageFormat();
    auto outSqueezed = isSelf3D ? View4Das3D(maxPoolOutResult, dstFormat, uniqueExecutor.get()) : maxPoolOutResult;
    CHECK_RET(outSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto outResult = l0op::ViewCopy(outSqueezed, out, uniqueExecutor.get());
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto maxPoolIndicesResult = std::get<1>(maxPoolResult);
    CHECK_RET(maxPoolIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesSqueezed =
        isSelf3D ? View4Das3D(maxPoolIndicesResult, dstFormat, uniqueExecutor.get()) : maxPoolIndicesResult;
    CHECK_RET(indicesSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 如果出参indices是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto indicesResult = l0op::ViewCopy(indicesSqueezed, indices, uniqueExecutor.get());
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    // 如果kernelSize符合切ND要求，在这里把isMask设置为false
    OP_CHECK_NULL(kernelSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckAttrSize1Or2(kernelSize), ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "On the 910_95 chip, the aclnnMaxPool2dWithMaskGetWorkspaceSize interface has been deprecated, "
            "please use the aclnnMaxPool2dWithIndices interface."),
        return ACLNN_ERR_PARAM_INVALID);
    const aclIntArray& kernelRef = *kernelSize;
    const int64_t kH = kernelRef[0];
    const int64_t kW = (kernelRef.Size() == 1) ? kH : kernelRef[1];
    if (!(kH == 1 && kW == 1) && (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                                  GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93)) {
        L2_DFX_PHASE_1(
            aclnnMaxPool2dWithMask, DFX_IN(self, kernelSize, stride, padding, dilation, ceilMode),
            DFX_OUT(out, indices));
        return ExecMaxPool2dWithIndicesReshape3DGetWorkspaceSize(
            self, kernelSize, stride, padding, dilation, ceilMode, out, indices, false, workspaceSize, executor);
    } else {
        return ExecMaxPool2dWithIndicesGetWorkspaceSize(
            self, kernelSize, stride, padding, dilation, ceilMode, out, indices, true, false, workspaceSize, executor);
    }
}

aclnnStatus aclnnMaxPool2dWithMask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxPool2dWithMask);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnMaxPool2dWithIndicesGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnMaxPool2dWithIndices, DFX_IN(self, kernelSize, stride, padding, dilation, ceilMode),
        DFX_OUT(out, indices));
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return ExecMaxPool2dWithIndicesGetWorkspaceSizeV3(
            self, kernelSize, stride, padding, dilation, ceilMode, out, indices, workspaceSize, executor);
    } else {
        return ExecMaxPool3dWithIndicesGetWorkspaceSize(
            self, kernelSize, stride, padding, dilation, ceilMode, out, indices, workspaceSize, executor);
    }
}

aclnnStatus aclnnMaxPool2dWithIndices(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxPool2dWithIndices);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
