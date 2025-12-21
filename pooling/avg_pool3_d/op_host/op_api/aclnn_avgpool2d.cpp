/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_avgpool2d.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/framework_op.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "pooling.h"
#include "avgpool_update.h"
#include "level0/reduce_mean.h"
#include "avgpool_v2.h"
#include "aclnn_kernels/transdata.h"
#include "level0/muls.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "avgpool3d.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "cube_util.h"

using namespace op;

namespace {
template <typename T>
static const aclTensor *GenWeightIdentity(
    aclOpExecutor *executor, int32_t blockK, int32_t blockN, op::DataType dataType, T areaFactor)
{
    int32_t enLarge = std::max(blockK, blockN);
    if (blockK == 0 || blockN == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockN or blockK is 0");
    }
    int32_t coutOpt = enLarge / blockN;
    int32_t cinOpt = enLarge / blockK;
    int64_t matrixSizeWeight = (int64_t)cinOpt * blockK * coutOpt * blockN;
    std::unique_ptr<T[]> input = std::make_unique<T[]>(matrixSizeWeight);
    if (input.get() == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "nullptr error");
    }
    for (int32_t j = 0; j < matrixSizeWeight; j++) {
        input[j] = 0;
    }

    // Write data to the fractal weight.
    int32_t baseNIdx = blockK;
    int32_t baseCout1OptIdx = baseNIdx * blockN;
    int32_t baseCin1OptIdx = baseCout1OptIdx * coutOpt;
    for (int32_t cin1OptIdx = 0; cin1OptIdx < cinOpt; cin1OptIdx++) {
        for (int32_t cout1OptIdx = 0; cout1OptIdx < coutOpt; cout1OptIdx++) {
            for (int32_t nIdx = 0; nIdx < blockN; nIdx++) {
                int32_t kIdx = cout1OptIdx * blockN + nIdx - cin1OptIdx * blockK;
                if (kIdx < 0 || kIdx >= blockK) {
                    continue;
                }
                int32_t valueIdx = cin1OptIdx * baseCin1OptIdx + cout1OptIdx * baseCout1OptIdx + nIdx * baseNIdx + kIdx;
                if (valueIdx >= matrixSizeWeight) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "valueIdx error");
                }

                input[valueIdx] = areaFactor;
            }
        }
    }

    op::Shape weightShape = op::Shape({ cinOpt, coutOpt, blockN, blockK });
    auto weight = executor->ConvertToTensor(input.get(), matrixSizeWeight, dataType);
    const_cast<aclTensor *>(weight)->SetOriginalShape(weightShape);
    const_cast<aclTensor *>(weight)->SetOriginalFormat(Format::FORMAT_FRACTAL_Z);
    const_cast<aclTensor *>(weight)->SetStorageShape(weightShape);
    const_cast<aclTensor *>(weight)->SetStorageFormat(Format::FORMAT_FRACTAL_Z);
    const_cast<aclTensor *>(weight)->SetViewShape(weightShape);
    const_cast<aclTensor *>(weight)->SetViewFormat(Format::FORMAT_FRACTAL_Z);

    return weight;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif
namespace {
static const uint32_t NCHW_N_IDX = 0;
static const uint32_t NCHW_C_IDX = 1;
static const uint32_t NCHW_H_IDX = 2;
static const uint32_t NCHW_W_IDX = 3;

static const uint32_t CHW_H_IDX = 1;
static const uint32_t CHW_W_IDX = 2;

static const int64_t AVGV2_KERNEL_H_MAX = 255;
static const int64_t AVGV2_KERNEL_W_MAX = 255;
static const int64_t AVGV2_KERNEL_SIZE_H_MUL_W = 255;
static const int64_t AVGV2_KERNEL_SIZE = 20;
static const int64_t AVGV2_STRIDE_SIZE = 63;
static const int64_t AVGV2_DIVISOR_OVERRIDE_MAX = 255;
static const int64_t AVGV2_DIVISOR_OVERRIDE_MIN = -255;

static const uint32_t FP16_BLOCK_K_SIZE = 16;
static const uint32_t FP16_BLOCK_N_SIZE = 16;
static const uint32_t FP32_BLOCK_K_SIZE = 8;
static const uint32_t FP32_BLOCK_N_SIZE = 16;

static const int64_t POOLING_SUM_MODE = 2;
static const int64_t DIM_NUM_3D = 3;
static const int64_t DIM_NUM_4D = 4;

static const int64_t INDEX_1 = 1;
static const int64_t INDEX_2 = 2;
static const int64_t INDEX_3 = 3;

static const int64_t PAD_DIM_2 = 2;
static const int64_t STRIDE_DIM_2 = 2;

static const int64_t DIM0 = 0;
static const int64_t DIM1 = 1;
static const int64_t DIM2 = 2;
static const int64_t DIM3 = 3;
static const int64_t DIM4 = 4;

const int64_t kKernelSizeHIdx = 0;
const int64_t kKernelSizeWIdx = 1;
const int64_t kPaddingUpIdx = 0;
const int64_t kPaddingLeftIdx = 1;

const int64_t cMaxDims = 65536;

static const int64_t LONG_STRIP_MULTIPLIER = 2;

static const char *FORMAT_NCHW_STR = "NCHW";

static const int64_t BIG_KERNEL_SUM_LIMIT = 10240;
static const int64_t BIG_KERNEL_SINGLE_LIMIT = 1024;
static const int64_t BIG_KERNEL_CALC_LIMIT = 1.0e+10;
static const int64_t BIG_KERNEL_CHANNEL_LIMIT = 64;

struct AvgPoolUtilInfo {
    vector<int64_t> ksize;
    vector<int64_t> strides;
    vector<int64_t> pads;

    int64_t inputN = 0;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t ksizeH = 0;
    int64_t ksizeW = 0;
    int64_t stridesH = 0;
    int64_t stridesW = 0;

    int64_t divisor = 0;

    int64_t outputH = 0;
    int64_t outputW = 0;
};

using AvgPoolInfo = AvgPoolUtilInfo;

static aclnnStatus GetPoolInfo(
    const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding,
    const int64_t divisorOverride, const aclTensor *out, AvgPoolInfo &avgPoolInfo)
{
    for (size_t i = 0; i < kernel->Size(); i++) {
        avgPoolInfo.ksize.emplace_back((*kernel)[i]);
    }

    for (size_t i = 0; i < stride->Size(); i++) {
        avgPoolInfo.strides.emplace_back((*stride)[i]);
    }

    for (size_t i = 0; i < padding->Size(); i++) {
        avgPoolInfo.pads.emplace_back((*padding)[i]);
    }

    avgPoolInfo.divisor = divisorOverride;
    if (out->GetViewShape().GetDimNum() == DIM_NUM_4D) { // NCHW
        avgPoolInfo.outputH = out->GetViewShape().GetDim(NCHW_H_IDX);
        avgPoolInfo.outputW = out->GetViewShape().GetDim(NCHW_W_IDX);
    } else { // CHW
        avgPoolInfo.outputH = out->GetViewShape().GetDim(CHW_H_IDX);
        avgPoolInfo.outputW = out->GetViewShape().GetDim(CHW_W_IDX);
    }

    string formatStr = op::ToString(self->GetStorageFormat()).GetString();
    op::Shape shape = self->GetViewShape();

    if (formatStr == FORMAT_NCHW_STR) {
        avgPoolInfo.inputN = shape.GetDim(NCHW_N_IDX);
        avgPoolInfo.inputC = shape.GetDim(NCHW_C_IDX);
        avgPoolInfo.inputH = shape.GetDim(NCHW_H_IDX);
        avgPoolInfo.inputW = shape.GetDim(NCHW_W_IDX);
        avgPoolInfo.ksizeH = avgPoolInfo.ksize[NCHW_H_IDX];
        avgPoolInfo.ksizeW = avgPoolInfo.ksize[NCHW_W_IDX];
        avgPoolInfo.stridesH = avgPoolInfo.strides[NCHW_H_IDX];
        avgPoolInfo.stridesW = avgPoolInfo.strides[NCHW_W_IDX];
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format:%s not support.", formatStr.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static const aclTensor *View4Das3D(const aclTensor *input, aclOpExecutor *executor)
{
    // NCHW -> squeeze -> reformat -> NCL
    // squeeze out into 3D
    const int64_t removeDim[] = {0};
    aclIntArray *dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCL
    auto reformatInput = l0op::ReFormat(squeezedInput, Format::FORMAT_NCL);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor *View3Das4D(const aclTensor *input, aclOpExecutor *executor)
{
    // NCL -> unsqueeze -> reformat -> NCHW
    // unsqueeze input into 4D
    const int64_t appendDim[] = {0};
    aclIntArray *dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static bool CheckNotNullPtr(
    const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding,
    aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(kernel, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    // 根据API定义，需要列出所能支持的所有dtype
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);

    // 检查self的数据类型是否和out一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

    return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out)
{
    // 检查self的format是否为NCHW/NCL
    if (self->GetStorageFormat() != Format::FORMAT_NCHW && self->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self format should be NCHW or NCL. Actual: self is [%s].",
            op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    // 检查out的format是否为NCHW/NCL
    if (out->GetStorageFormat() != Format::FORMAT_NCHW && out->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out format should be NCHW or NCL. Actual: out is [%s].",
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    // 检查当输入是4D时格式是否为NCHW
    if (self->GetViewShape().GetDimNum() == DIM_NUM_4D && (self->GetStorageFormat() != Format::FORMAT_NCHW ||
        out->GetStorageFormat() != Format::FORMAT_NCHW)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and out should all be NCHW when 4D. Actual: self [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

   // 检查当输入是3D时格式是否为NCL
    if (self->GetViewShape().GetDimNum() == DIM_NUM_3D && (self->GetStorageFormat() != Format::FORMAT_NCL ||
        out->GetStorageFormat() != Format::FORMAT_NCL)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and out should all be NCL when 3D. Actual: self [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support format [%s].", op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

static bool IfSupport2dTo3d() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return true;
  }
  return false;
}

// 计算经过avgpool2d后的shape的h和w（n,c与input一致，不用计算）
static inline int64_t PoolingOutShape(
    const int64_t inputSize, const int64_t kernelSize, const int64_t padL, const int64_t stride, const bool ceilMode) {
    int64_t outputSize = (stride == 0) ? -1 :
                         (inputSize + padL * PAD_DIM_2 - kernelSize + (ceilMode ? stride - 1 : 0)) / stride + 1;

    if (ceilMode) {
        if ((outputSize - 1) * stride >= inputSize + padL) {
            --outputSize;
        }
    }
    return outputSize;
}

static bool CheckPaddingValidAvgPool2D(const aclIntArray *kernelSize, const aclIntArray *padding) {
    const int64_t kernelH = (*kernelSize)[kKernelSizeHIdx];
    const int64_t kernelPaddingSize = 1;
    const int64_t MULTIPLIER = 2;
    // 1表示kernelSize长度为1
    const int64_t kernelW = kernelSize->Size() == kernelPaddingSize ? (*kernelSize)[kKernelSizeHIdx] :
                                                                      (*kernelSize)[kKernelSizeWIdx];
    OP_CHECK(padding->Size() != 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding is empty"), return false);
    const int64_t paddingH = (*padding)[kPaddingUpIdx];
    const int64_t paddingW = padding->Size() == kernelPaddingSize ? (*padding)[kPaddingUpIdx] :
                                                                    (*padding)[kPaddingLeftIdx];
    // MULTIPLIER 表示paddingH不能大于kernelH的一半，下同
    if (kernelH < MULTIPLIER * paddingH) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "value of paddingH should be at most half of kernelH. Actual: paddingH is [%ld],kernelH is [%ld].",
                paddingH, kernelH);
        return false;
    }
    if (kernelW < MULTIPLIER * paddingW) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "value of paddingW should be at most half of kernelW. Actual: paddingW is [%ld],"
                "kernelW is [%ld].",
                paddingW, kernelW);
        return false;
    }
    return true;
}

static bool CheckOutputShape(
    const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding,
    const bool ceilMode, const aclTensor *out) {
    const bool is3d = self->GetViewShape().GetDimNum() == DIM_NUM_3D;
    const int64_t nBatch = is3d ? 1 : self->GetViewShape().GetDim(0);
    const int64_t nInputPlane = is3d ? self->GetViewShape().GetDim(0) : self->GetViewShape().GetDim(INDEX_1);
    const int64_t height = is3d ? self->GetViewShape().GetDim(INDEX_1) : self->GetViewShape().GetDim(INDEX_2);
    const int64_t width = is3d ? self->GetViewShape().GetDim(INDEX_2) : self->GetViewShape().GetDim(INDEX_3);
    OP_CHECK(((nInputPlane != 0) && (height != 0) && (width != 0)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Expected tensor with optional 0 dim batch size,\
                     but got nInputPlane:%ld, height:%ld, width:%ld", nInputPlane, height, width),
             return false);
    const int64_t kH = (*kernel)[0];
    const int64_t kW = kernel->Size() == 1 ? kH : (*kernel)[1];

    const int64_t sH = stride->Size() == 0 ? kH : (*stride)[0];
    const int64_t sW = stride->Size() == 0 ? kW : stride->Size() == 1 ? sH : (*stride)[1];

    const int64_t padH = (*padding)[0];
    const int64_t padW = padding->Size() == 1 ? padH : (*padding)[1];

    const int64_t outHeight = PoolingOutShape(height, kH, padH, sH, ceilMode);
    const int64_t outWidth = PoolingOutShape(width, kW, padW, sW, ceilMode);

    OP_CHECK((outHeight > 0 && outWidth > 0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Given input size %ldx%ldx%ld, calc out size %ldx%ldx%ld",
                     nInputPlane, height, width, nInputPlane, outHeight, outWidth),
             return false);

    const op::Shape calcOutShape = is3d ? op::Shape({nInputPlane, outHeight, outWidth})
                                        : op::Shape({nBatch, nInputPlane, outHeight, outWidth});
    // 判断out的shape与推导出的输出shape是否相等
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, calcOutShape, return false);
    return true;
}

static bool CheckShape(
    const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding,
    const bool ceilmode, const aclTensor *out)
{
    // 检查self/out是否是3D或4D
    if (!((self->GetViewShape().GetDimNum() == DIM_NUM_3D && out->GetViewShape().GetDimNum() == DIM_NUM_3D) ||
        (self->GetViewShape().GetDimNum() == DIM_NUM_4D && out->GetViewShape().GetDimNum() == DIM_NUM_4D))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and out must maintain same format, support 3D or 4D.");
        return false;
    }

    bool validDims = self->GetViewShape().GetDim(1) != 0 && self->GetViewShape().GetDim(DIM2) != 0;
    if (!((self->GetViewShape().GetDimNum() == DIM_NUM_3D && self->GetViewShape().GetDim(0) != 0 && validDims) ||
        (self->GetViewShape().GetDimNum() == DIM_NUM_4D && validDims && self->GetViewShape().GetDim(DIM3) != 0))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor with optional 0 dim batch size for input, but got: %s.",
        op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    if (kernel->Size() < 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Length of kernel is less than 1.");
        return false;
    }

    if (padding->Size() != 1 && padding->Size() != PAD_DIM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected length of padding is 1 or 2, but got %lu.", padding->Size());
        return false;
    }

    if (stride->Size() > STRIDE_DIM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected length of stride is 0 or 1 or 2, but got: %lu.", stride->Size());
        return false;
    }
    bool result = CheckPaddingValidAvgPool2D(kernel, padding);
    CHECK_RET(result, false);
    // 检查out的shape是否符合预期输出
    bool ret =
        CheckOutputShape(self, kernel, stride, padding, ceilmode, out);
    CHECK_RET(ret, false);

    return true;
}

static bool CheckValue(
    const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding,
    const aclTensor *out)
{
    // 检查参数每一个维度的值是否合法
    for (size_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
        if (self->GetViewShape().GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of self is negative.");
            return false;
        }
    }

    for (size_t i = 0; i < out->GetViewShape().GetDimNum(); i++) {
        if (out->GetViewShape().GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of out is negative.");
            return false;
        }
    }

    for (size_t i = 0; i < kernel->Size(); i++) {
        if ((*kernel)[i] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of kernel is less than or equal to 0.");
            return false;
        }
    }

    for (size_t i = 0; i < stride->Size(); i++) {
        if ((*stride)[i] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of stride is less than or equal to 0.");
            return false;
        }
    }

    for (size_t i = 0; i < padding->Size(); i++) {
        if ((*padding)[i] < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of padding is negtive.");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor *self, const aclIntArray *kernelSize, const aclIntArray *stride, const aclIntArray *padding,
    const bool ceilMode, aclTensor *out, int8_t cubeMathType)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNullPtr(self, kernelSize, stride, padding, out), ACLNN_ERR_PARAM_NULLPTR);

    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查参数值是否合法
    CHECK_RET(CheckValue(self, kernelSize, stride, padding, out), ACLNN_ERR_PARAM_INVALID);

    // 检查数据维度是否合法
    CHECK_RET(CheckShape(self, kernelSize, stride, padding, ceilMode, out), ACLNN_ERR_PARAM_INVALID);

    // 检查cubeMathType
    CHECK_RET(CheckCubeMathType(self->GetDataType(), cubeMathType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CheckGlobalPool(AvgPoolInfo &avgPoolInfo)
{
    if ((avgPoolInfo.inputH == avgPoolInfo.ksizeH) && (avgPoolInfo.inputW == avgPoolInfo.ksizeW)) {
        for (size_t i = 0; i < avgPoolInfo.pads.size(); i++) {
            if (avgPoolInfo.pads[i] != 0) {
                return false;
            }
        }

        return true;
    }

    return false;
}

static bool CheckAvgPool(const AvgPoolInfo &avgPoolInfo)
{
    bool aiCoreKsize = avgPoolInfo.ksizeH <= AVGV2_KERNEL_H_MAX && avgPoolInfo.ksizeW <= AVGV2_KERNEL_W_MAX;
    bool isSupportKernel = (avgPoolInfo.ksizeH * avgPoolInfo.ksizeW <= AVGV2_KERNEL_SIZE_H_MUL_W ||
        (avgPoolInfo.ksizeH <= AVGV2_KERNEL_SIZE && avgPoolInfo.ksizeW <= AVGV2_KERNEL_SIZE));
    bool isDivisorOverrideValid = (avgPoolInfo.divisor >= AVGV2_DIVISOR_OVERRIDE_MIN) &&
        (avgPoolInfo.divisor <= AVGV2_DIVISOR_OVERRIDE_MAX);

    if(IfSupport2dTo3d() && !isSupportKernel) {
        return false;
    }
    else if(!isSupportKernel && avgPoolInfo.outputH != 1 && avgPoolInfo.outputW == 1) {
        return false;
    }

    if (!aiCoreKsize) {
        return false;
    }
    if (!isDivisorOverrideValid) {
        return false;
    }

    return true;
}

static bool CheckNeedChangeTo3D(const AvgPoolInfo &avgPoolInfo)
{
    if((avgPoolInfo.inputW / avgPoolInfo.inputH < LONG_STRIP_MULTIPLIER ) &&
       (avgPoolInfo.inputH / avgPoolInfo.inputW  < LONG_STRIP_MULTIPLIER))
    {
        return false;
    }
    return true;
}

static aclnnStatus HandleOut(const aclTensor *out, const aclTensor *castOut, UniqueExecutor &uniqueExecutor){
    if (out->GetViewShape().GetDimNum() == DIM_NUM_3D) {
        const aclTensor *out3d = nullptr;
        out3d  = View4Das3D(castOut, uniqueExecutor.get());
        auto result = l0op::ViewCopy(out3d, out, uniqueExecutor.get());
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    } else {
        auto result = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

// 构建全局average pooling, 通过vector实现
static aclnnStatus BuildGlobalPoolGraph(
    UniqueExecutor &uniqueExecutor, const aclTensor *self, const aclIntArray *kernelSize, const int64_t divisorOverride,
    const aclTensor *out)
{
    // 进行ReduceMean计算
    FVector<int64_t> axesVec { NCHW_H_IDX, NCHW_W_IDX }; // 默认是NCHW
    auto axesArray = uniqueExecutor.get()->AllocIntArray(axesVec.data(), 2);
    bool keepDims = true; // 维度一直保持不变
    auto reduceOut = l0op::ReduceMean(self, axesArray, keepDims, uniqueExecutor.get());
    CHECK_RET(reduceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *transOutput = nullptr;
    int64_t group = 1;
    if (divisorOverride != 0) {
        int64_t kw = (*kernelSize)[2];
        int64_t kh = (*kernelSize)[3];
        float factor = static_cast<float>(float(1.0 * kw * kh) / divisorOverride);
        auto mulOut = l0op::Muls(reduceOut, factor, uniqueExecutor.get());
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto reformatOut = l0op::ReFormat(mulOut, Format::FORMAT_NCHW);
        CHECK_RET(reformatOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transOutput = l0op::TransData(reformatOut, self->GetViewFormat(), group, uniqueExecutor.get());
        CHECK_RET(transOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto reformatOut = l0op::ReFormat(reduceOut, Format::FORMAT_NCHW);
        CHECK_RET(reformatOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transOutput = l0op::TransData(reformatOut, self->GetViewFormat(), group, uniqueExecutor.get());
        CHECK_RET(transOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(transOutput, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return HandleOut(out, castOut, uniqueExecutor);
}

static bool CheckNeedUpdate(bool countIncludePad, int64_t divisorOverride)
{
    if (divisorOverride != 0) {
        return false;
    }

    if (countIncludePad) {
        return false;
    } else {
        return true;
    }
}

static const aclTensor *GenWeightDynamic(
    const aclTensor *self, bool exclusivePads, const aclIntArray *kernelSize, int64_t divisorOverride,
    aclOpExecutor *executor)
{
    float areaFactor = 0;
    if (exclusivePads) {
        areaFactor = 1.0;
    }
    else {
        int64_t kw = (*kernelSize)[2];
        int64_t kh = (*kernelSize)[3];
        areaFactor = static_cast<float>(1.0 / (kw * kh));
    }

    if (divisorOverride != 0) {
        areaFactor = static_cast<float>(1.0 / divisorOverride);
    }

    int32_t blockK = 0;
    int32_t blockN = 0;
    // op::Shape weightShape;
    auto dataType = self->GetDataType();
    if (DataType::DT_FLOAT == dataType) {
        blockK = FP32_BLOCK_K_SIZE;
        blockN = FP32_BLOCK_N_SIZE;
        return GenWeightIdentity(executor, blockK, blockN, dataType, areaFactor);
    } else if (DataType::DT_FLOAT16 == dataType) {
        blockK = FP16_BLOCK_K_SIZE;
        blockN = FP16_BLOCK_N_SIZE;
        fp16_t areaFactorFp16 = areaFactor;
        return GenWeightIdentity(executor, blockK, blockN, dataType, areaFactorFp16);
    }
    // 其他类型暂不支持
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dtype %s do not support", op::ToString(self->GetDataType()).GetString());
    return nullptr;
}

// 构建局部average pooling计算图, 通过Cube实现
static aclnnStatus BuildAvgPoolGraph(
    UniqueExecutor &uniqueExecutor, const aclTensor *self, const aclIntArray *kernelSize, const aclIntArray *stride,
    const aclIntArray *pad, const int64_t ceilMode, const bool countIncludePad, const int64_t divisorOverride,
    const aclTensor *out)
{
    // 固定写法，将输入self格式转换成NC1HWC0
    int64_t groups = 1;
    auto inputNC1HWC0 = l0op::TransData(self, Format::FORMAT_NC1HWC0, groups, uniqueExecutor.get());
    CHECK_RET(inputNC1HWC0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 先构造辅助矩阵weight
    const aclTensor *weight = nullptr;
    bool exclusivePads = CheckNeedUpdate(countIncludePad, divisorOverride);
    weight = GenWeightDynamic(self, exclusivePads, kernelSize, divisorOverride, uniqueExecutor.get());
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto deviceTensor = op::CopyToNpu(weight, uniqueExecutor.get());

    // 进行Pooling
    int64_t mode = POOLING_SUM_MODE;
    bool globalPooling = false;

    const aclTensor *poolingOut = l0op::Pooling5Hd(inputNC1HWC0, deviceTensor, mode, globalPooling, kernelSize,
        stride, pad, ceilMode, FORMAT_NCHW_STR, uniqueExecutor.get());
    CHECK_RET(poolingOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (exclusivePads) {
        // pad不参与运算，插入AvgPoolUpdate算子
        const char *paddingMode = "CALCULATED";
        poolingOut = l0op::AvgPoolUpdate5Hd(poolingOut, inputNC1HWC0, kernelSize, stride, paddingMode, pad,
                                              FORMAT_NCHW_STR, ceilMode, exclusivePads, uniqueExecutor.get());
        CHECK_RET(poolingOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将输出的格式从NC1HWC0转换成NCHW
    auto transOutput = l0op::TransData(poolingOut, Format::FORMAT_NCHW, groups, uniqueExecutor.get());
    CHECK_RET(transOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(transOutput, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return HandleOut(out, castOut, uniqueExecutor);
}

static bool CheckBigKernel(const aclIntArray *kernelSize, const aclTensor *avgpoolIn, const aclTensor *out)
{
    int64_t sumK = (*kernelSize)[DIM0] * (*kernelSize)[DIM1] * (*kernelSize)[DIM2];
    bool bigKernel = ((*kernelSize)[DIM0] > BIG_KERNEL_SINGLE_LIMIT || (*kernelSize)[DIM1] > BIG_KERNEL_SINGLE_LIMIT || 
                        (*kernelSize)[DIM2] > BIG_KERNEL_SINGLE_LIMIT) && sumK > BIG_KERNEL_SUM_LIMIT;

    auto avgpoolInShape = avgpoolIn->GetViewShape();
    int64_t channel = avgpoolInShape.GetDim(DIM1);

    auto avgpoolOutShape = out->GetViewShape();
    auto outDimNum = avgpoolOutShape.GetDimNum();
    int64_t oH = outDimNum == DIM_NUM_4D ? avgpoolOutShape.GetDim(DIM2) : avgpoolOutShape.GetDim(DIM1);
    int64_t oW = outDimNum == DIM_NUM_4D ? avgpoolOutShape.GetDim(DIM3) : avgpoolOutShape.GetDim(DIM2);
    int64_t windowsCalc = oH *  oW * sumK;
    // 非全局平均池化下，kernel较大，C通道小于64，且达到一定计算数据量时走Big Kernel分支
    if (bigKernel && windowsCalc > BIG_KERNEL_CALC_LIMIT && channel < BIG_KERNEL_CHANNEL_LIMIT){
        return true;
    }
    return false;
}

// 构建averagepool3d计算图, 通过Vector实现
static aclnnStatus BuildAvgPool2dTo3dGraph(
    UniqueExecutor &uniqueExecutor, const aclTensor *self, const aclIntArray *kernelSize, const aclIntArray *stride,
    const aclIntArray *pad, const bool ceilMode, const bool countIncludePad, const int64_t divisorOverride,
    const aclTensor *out)
{
    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* avgpoolIn = selfContiguous;

    //Reshape 4D to 5D N,C,H,W->1,NC,1,H,W
    auto shape = op::ToShapeVector(self->GetViewShape());
    FVector<int64_t> newShape = {1, shape[DIM0]*shape[DIM1], 1, shape[DIM2], shape[DIM3]};
    aclIntArray* shapeArray = uniqueExecutor->AllocIntArray(newShape.data(), newShape.size());
    CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    avgpoolIn = l0op::Reshape(avgpoolIn, shapeArray, uniqueExecutor.get());
    CHECK_RET(avgpoolIn != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::string dataFormat = "NCDHW";
    bool isBigKernelFlag = CheckBigKernel(kernelSize, avgpoolIn, out);
    if (!isBigKernelFlag){
        dataFormat = "NDHWC";
        //Transpose 1,NC,D,H,W->1,D,H,W,NC
        FVector<int64_t> inputDims = { DIM0, DIM2, DIM3, DIM4, DIM1};
        auto permPre = uniqueExecutor->AllocIntArray(inputDims.data(), inputDims.size());
        CHECK_RET(permPre != nullptr, ACLNN_ERR_INNER_NULLPTR);

        avgpoolIn = l0op::Transpose(avgpoolIn, permPre, uniqueExecutor.get());
        CHECK_RET(avgpoolIn != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    //执行L0算子
    auto avgpoolOut = l0op::AvgPool3D(avgpoolIn, kernelSize, stride, pad, ceilMode,
        countIncludePad, divisorOverride, dataFormat, uniqueExecutor.get());
    CHECK_RET(avgpoolOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (!isBigKernelFlag){
        //Transpose NDHWC->NCDHW
        FVector<int64_t> outputDims = { DIM0, DIM4, DIM1, DIM2, DIM3};
        auto permPost = uniqueExecutor->AllocIntArray(outputDims.data(), outputDims.size());
        CHECK_RET(permPost != nullptr, ACLNN_ERR_INNER_NULLPTR);

        avgpoolOut = l0op::Transpose(avgpoolOut, permPost, uniqueExecutor.get());
        CHECK_RET(avgpoolOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    
    //Reshape 5D to 4D or 3D NCDHW->NCHW(NCL)
    shape = op::ToShapeVector(out->GetViewShape());
    shapeArray = uniqueExecutor->AllocIntArray(shape.data(), shape.size());
    CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    avgpoolOut = l0op::Reshape(avgpoolOut, shapeArray, uniqueExecutor.get());
    CHECK_RET(avgpoolOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ViewCopy(avgpoolOut, out, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus BuildAiCpuGraph(
    UniqueExecutor &uniqueExecutor, const aclTensor *self, const aclIntArray *kernelSize, const aclIntArray *strides,
    const aclIntArray *paddings, const bool ceilMode, const bool exclusive, const int64_t divisorOverride,
    const aclTensor *out)
{
    static const std::string paddingMode = "CALCULATED";
    const bool globalPooling = false;

    static const std::string dataFormat = FORMAT_NCHW_STR;
    auto avgPoolingOut = l0op::AvgPoolNchw(self, kernelSize, strides, paddingMode, paddings, dataFormat,
        globalPooling, ceilMode, exclusive, divisorOverride, uniqueExecutor.get());

    auto castOut = l0op::Cast(avgPoolingOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return HandleOut(out, castOut, uniqueExecutor);
}

static aclnnStatus ProcessDataTypeConvert(
    const aclTensor *self, const int8_t cubeMathType, UniqueExecutor &uniqueExecutor, const aclTensor *self4d,
    const aclTensor* &cast4dTensor)
{
    CHECK_RET(self != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(self4d != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (!IsCubeSupportFp32() && DataType::DT_FLOAT == self->GetDataType()) { // 芯片不支持FP32且输入是FP32
        if (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_FP16) { // 允许降精度，则转为FP16
            cast4dTensor = l0op::Cast(self4d, DataType::DT_FLOAT16, uniqueExecutor.get());
            CHECK_RET(cast4dTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else { // cubeMathType == KEEP_DTYPE or USE_HF32，不允许降精度，则报错
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cubeMathType cannot support fp32 in this platform.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (IsCubeSupportFp32() && DataType::DT_FLOAT == self->GetDataType()) { // 芯片支持FP32且输入是FP32
        if (cubeMathType == USE_FP16) { // 允许降精度，则转为FP16
            cast4dTensor = l0op::Cast(self4d, DataType::DT_FLOAT16, uniqueExecutor.get());
            CHECK_RET(cast4dTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else if (cubeMathType != KEEP_DTYPE) { // cubeMathType == USE_HF32 or ALLOW_FP32_DOWN_PRECISION
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cubeMathType cannot support hf32 in this platform.");
            return ACLNN_ERR_PARAM_INVALID;
        } else { // KEEP_DTYPE
            cast4dTensor = self4d;
        }
    } else if (IsCubeSupportFp32() && DataType::DT_BF16 == self->GetDataType()) { // 芯片支持FP32,等同判断支持bf16,且输入是bf16
        // 临时方案: 忽略精度模式的判断，升精度，转为FP32，才能调用fp32的算子能力，算子暂不支持bf16能力。
        cast4dTensor = l0op::Cast(self4d, DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(cast4dTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else { // 输入是FP16
        cast4dTensor = self4d;
    }

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnAvgPool2dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* strides, const aclIntArray* paddings,
    const bool ceilMode, const bool countIncludePad, const int64_t divisorOverride, const int8_t cubeMathType,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAvgPool2d, DFX_IN(self, kernelSize, strides, paddings,\
                   ceilMode, countIncludePad, divisorOverride, cubeMathType), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, kernelSize, strides, paddings, ceilMode, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果是空tensor，直接返回
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果输入是3D(NCL)，转化为4D(NCHW)
    const aclTensor *self4d = nullptr;
    if (self->GetViewShape().GetDimNum() == DIM_NUM_3D) {
        self4d = View3Das4D(selfContiguous, uniqueExecutor.get());
    } else {
        self4d = selfContiguous;
    }

    // 对kernel,stride,pad等参数做扩维
    const int64_t kH = (*kernelSize)[0];
    const int64_t kW = kernelSize->Size() == 1 ? kH : (*kernelSize)[1];
    FVector<int64_t> newKernel { 1, 1, kH, kW};
    const aclIntArray *kernel4 = uniqueExecutor.get()->AllocIntArray(newKernel.data(), 4);

    const int64_t sH = strides->Size() == 0 ? kH : (*strides)[0];
    const int64_t sW = strides->Size() == 0 ? kW : strides->Size() == 1 ? sH : (*strides)[1];
    FVector<int64_t> newStride { 1, 1, sH, sW};
    const aclIntArray *stride4 = uniqueExecutor.get()->AllocIntArray(newStride.data(), 4);

    const int64_t padH = (*paddings)[0];
    const int64_t padW = paddings->Size() == 1 ? padH : (*paddings)[1];
    FVector<int64_t> newPad { padH, padH, padW, padW};
    const aclIntArray *paddings4 = uniqueExecutor.get()->AllocIntArray(newPad.data(), 4);

    // 获取AvgPool2d基本算子信息
    AvgPoolInfo avgPoolInfo;
    ret = GetPoolInfo(self4d, kernel4, stride4, paddings4, divisorOverride, out, avgPoolInfo);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool isSupport2dTo3d = IfSupport2dTo3d();
    OP_LOGD("Current isSupport2dTo3d flag is: %d", isSupport2dTo3d);

    auto cDims = self4d->GetViewShape().GetDim(NCHW_C_IDX);
    // 计算图构建
    if (CheckGlobalPool(avgPoolInfo)) {
        const aclTensor *cast4dTensor = nullptr;
        ret = ProcessDataTypeConvert(self, cubeMathType, uniqueExecutor, self4d, cast4dTensor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        // 构建global average pooling计算图, 通过vector实现
        CHECK_RET(ACLNN_SUCCESS == BuildGlobalPoolGraph(uniqueExecutor, cast4dTensor, kernel4, divisorOverride, out),
                                                        ACLNN_ERR_INNER);
    } else if(!isSupport2dTo3d) {
        const aclTensor *cast4dTensor = nullptr;
        ret = ProcessDataTypeConvert(self, cubeMathType, uniqueExecutor, self4d, cast4dTensor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        if (CheckAvgPool(avgPoolInfo)) {
            int64_t ceilModeInt = 0;
            if (!ceilMode) ceilModeInt = static_cast<int64_t>(1);
            // 构建常规average pooling计算图, 通过Cube实现
            CHECK_RET(ACLNN_SUCCESS == BuildAvgPoolGraph(uniqueExecutor, cast4dTensor, kernel4, stride4, paddings4,
                ceilModeInt, countIncludePad, divisorOverride, out), ACLNN_ERR_INNER);
        } else {
            // 构建走AICPU分支的计算图
            const bool exclusive = !countIncludePad;
            CHECK_RET(ACLNN_SUCCESS == BuildAiCpuGraph(uniqueExecutor, cast4dTensor, kernel4, stride4, paddings4,
                ceilMode, exclusive, divisorOverride, out), ACLNN_ERR_INNER);
        }
    } else {
        if (!CheckNeedChangeTo3D(avgPoolInfo) && CheckAvgPool(avgPoolInfo) && !ceilMode && cDims < cMaxDims) {
            const aclTensor *cast4dTensor = nullptr;
            ret = ProcessDataTypeConvert(self, cubeMathType, uniqueExecutor, self4d, cast4dTensor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
            int64_t ceilModeInt = 0;
            if (!ceilMode) ceilModeInt = static_cast<int64_t>(1);
            // 构建常规average pooling计算图, 通过Cube实现
            CHECK_RET(ACLNN_SUCCESS == BuildAvgPoolGraph(uniqueExecutor, cast4dTensor, kernel4, stride4, paddings4,
                ceilModeInt, countIncludePad, divisorOverride, out), ACLNN_ERR_INNER);
        } else {
            // 构建走3D分支

            const int64_t kD = 1;
            FVector<int64_t> newKernel3d { kD, kH, kW };
            const aclIntArray *kernelList = uniqueExecutor.get()->AllocIntArray(newKernel3d.data(), newKernel3d.size());

            const int64_t sD = 1;
            FVector<int64_t> newStride3d { sD, sH, sW };
            const aclIntArray *strideList = uniqueExecutor.get()->AllocIntArray(newStride3d.data(), newStride3d.size());

            const int64_t padD = 0;
            FVector<int64_t> newPad3d { padD, padH, padW};
            const aclIntArray *padList = uniqueExecutor.get()->AllocIntArray(newPad3d.data(), newPad3d.size());

            CHECK_RET(ACLNN_SUCCESS == BuildAvgPool2dTo3dGraph(uniqueExecutor, self4d, kernelList, strideList, padList,
                ceilMode, countIncludePad, divisorOverride, out), ACLNN_ERR_INNER);
        }
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAvgPool2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAvgPool2d);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
