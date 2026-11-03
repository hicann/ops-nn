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
 * \file aclnn_modulate_backward.cpp
 * \brief
 */
#include "aclnn_modulate_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "modulate_grad.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

static const int64_t MAX_SUPPORT_DIM = 3;
static const int64_t SCALE_DIM = 2;
static const int64_t FIRST_DIM = 0;
static const int64_t MIDDLE_DIM = 1;
static const int64_t LAST_DIM = 2;

static const std ::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std ::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST_910C{
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std ::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std ::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_910C{
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* grad_output, const aclTensor* input, const aclTensor* grad_input)
{
    OP_CHECK_NULL(grad_output, return false);
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(grad_input, return false);
    return true;
}

static bool CheckNotNullForScaleAndShift(const aclTensor* tensor)
{
    OP_CHECK_NULL(tensor, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* grad_output, const aclTensor* input, const aclTensor* scale,
                            const aclTensor* shift, const aclTensor* grad_input, const aclTensor* grad_scale,
                            const aclTensor* grad_shift)
{
    bool is910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion ::ASCEND910B);
    const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST = is910BSocVersion ? GRAD_DTYPE_SUPPORT_LIST_910B :
                                                                                       GRAD_DTYPE_SUPPORT_LIST_910C;
    const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST = is910BSocVersion ? OUT_DTYPE_SUPPORT_LIST_910B :
                                                                                      OUT_DTYPE_SUPPORT_LIST_910C;

    OP_CHECK_DTYPE_NOT_SUPPORT(grad_output, GRAD_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, GRAD_DTYPE_SUPPORT_LIST, return false);
    if (CheckNotNullForScaleAndShift(scale)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scale, GRAD_DTYPE_SUPPORT_LIST, return false);
    }
    if (CheckNotNullForScaleAndShift(shift)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(shift, GRAD_DTYPE_SUPPORT_LIST, return false);
    }
    if (CheckNotNullForScaleAndShift(grad_scale)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(grad_scale, OUT_DTYPE_SUPPORT_LIST, return false);
    }
    if (CheckNotNullForScaleAndShift(grad_shift)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(grad_shift, OUT_DTYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(grad_input, OUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckDtypeConsistency(const aclTensor* grad_output, const aclTensor* input, const aclTensor* scale,
                                  const aclTensor* shift, const aclTensor* grad_input, const aclTensor* grad_scale,
                                  const aclTensor* grad_shift)
{
    auto refDtype = grad_output->GetDataType();
    if (input->GetDataType() != refDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input dtype %d does not match grad_output dtype %d",
                static_cast<int32_t>(input->GetDataType()), static_cast<int32_t>(refDtype));
        return false;
    }
    if (CheckNotNullForScaleAndShift(scale) && scale->GetDataType() != refDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale dtype %d does not match grad_output dtype %d",
                static_cast<int32_t>(scale->GetDataType()), static_cast<int32_t>(refDtype));
        return false;
    }
    if (CheckNotNullForScaleAndShift(shift) && shift->GetDataType() != refDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shift dtype %d does not match grad_output dtype %d",
                static_cast<int32_t>(shift->GetDataType()), static_cast<int32_t>(refDtype));
        return false;
    }
    if (grad_input->GetDataType() != refDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad_input dtype %d does not match grad_output dtype %d",
                static_cast<int32_t>(grad_input->GetDataType()), static_cast<int32_t>(refDtype));
        return false;
    }
    if (CheckNotNullForScaleAndShift(grad_scale) && grad_scale->GetDataType() != refDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad_scale dtype %d does not match grad_output dtype %d",
                static_cast<int32_t>(grad_scale->GetDataType()), static_cast<int32_t>(refDtype));
        return false;
    }
    if (CheckNotNullForScaleAndShift(grad_shift) && grad_shift->GetDataType() != refDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad_shift dtype %d does not match grad_output dtype %d",
                static_cast<int32_t>(grad_shift->GetDataType()), static_cast<int32_t>(refDtype));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* grad_output, const aclTensor* input, const aclTensor* scale,
                        const aclTensor* shift, const aclTensor* grad_input, const aclTensor* grad_scale,
                        const aclTensor* grad_shift)
{
    if (grad_output->GetStorageFormat() != Format::FORMAT_ND || input->GetStorageFormat() != Format::FORMAT_ND ||
        grad_input->GetStorageFormat() != Format::FORMAT_ND ||
        (scale != nullptr && scale->GetStorageFormat() != Format::FORMAT_ND) ||
        (shift != nullptr && shift->GetStorageFormat() != Format::FORMAT_ND) ||
        (grad_scale != nullptr && grad_scale->GetStorageFormat() != Format::FORMAT_ND) ||
        (grad_shift != nullptr && grad_shift->GetStorageFormat() != Format::FORMAT_ND)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "All input and output tensors only support ND format");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* grad_output, const aclTensor* input, const aclTensor* scale,
                       const aclTensor* shift, const aclTensor* grad_input, const aclTensor* grad_scale,
                       const aclTensor* grad_shift)
{
    const auto gradOutputShape = grad_output->GetViewShape();
    if (gradOutputShape.GetDimNum() != MAX_SUPPORT_DIM || input->GetViewShape().GetDimNum() != MAX_SUPPORT_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad_output and input must be 3-dimensional tensors");
        return false;
    }
    if (gradOutputShape != input->GetViewShape() || gradOutputShape != grad_input->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad_output, input and grad_input must have the same shape");
        return false;
    }

    if (scale == nullptr && grad_scale != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad_scale requires scale to be provided");
        return false;
    }
    if (scale != nullptr) {
        const auto scaleShape = scale->GetViewShape();
        if (scaleShape.GetDimNum() != SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale must be a 2-dimensional tensor");
            return false;
        }
        if (gradOutputShape.GetDim(FIRST_DIM) != scaleShape.GetDim(FIRST_DIM) ||
            gradOutputShape.GetDim(LAST_DIM) != scaleShape.GetDim(MIDDLE_DIM) ||
            (grad_scale != nullptr && scaleShape != grad_scale->GetViewShape())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale or grad_scale shape does not match grad_output");
            return false;
        }
    }

    if (shift == nullptr && grad_shift != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad_shift requires shift to be provided");
        return false;
    }
    if (shift != nullptr) {
        const auto shiftShape = shift->GetViewShape();
        if (shiftShape.GetDimNum() != SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shift must be a 2-dimensional tensor");
            return false;
        }
        if (gradOutputShape.GetDim(FIRST_DIM) != shiftShape.GetDim(FIRST_DIM) ||
            gradOutputShape.GetDim(LAST_DIM) != shiftShape.GetDim(MIDDLE_DIM) ||
            (grad_shift != nullptr && shiftShape != grad_shift->GetViewShape())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shift or grad_shift shape does not match grad_output");
            return false;
        }
    }

    if (grad_output->IsEmpty() &&
        ((scale != nullptr && !scale->IsEmpty()) || (shift != nullptr && !shift->IsEmpty()))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when grad_output is empty, scale and shift must be empty or absent");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* grad_output, const aclTensor* input, const aclTensor* scale,
                               const aclTensor* shift, const aclTensor* grad_input, const aclTensor* grad_scale,
                               const aclTensor* grad_shift)
{
    // 1.scale和shift参数可以为空，所以只检查grad_out input grad_input这三个参数是否为空指针
    CHECK_RET(CheckNotNull(grad_output, input, grad_input), ACLNN_ERR_PARAM_NULLPTR);
    // 2.检查数据类型是否在支持的范围内
    CHECK_RET(CheckDtypeValid(grad_output, input, scale, shift, grad_input, grad_scale, grad_shift),
              ACLNN_ERR_PARAM_INVALID);
    // 3.检查数据类型一致性
    CHECK_RET(CheckDtypeConsistency(grad_output, input, scale, shift, grad_input, grad_scale, grad_shift),
              ACLNN_ERR_PARAM_INVALID);
    // 4.检查数据格式
    CHECK_RET(CheckFormat(grad_output, input, scale, shift, grad_input, grad_scale, grad_shift),
              ACLNN_ERR_PARAM_INVALID);
    // 5.检查shape及可选输入输出关系
    CHECK_RET(CheckShape(grad_output, input, scale, shift, grad_input, grad_scale, grad_shift),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnModulateBackwardGetWorkspaceSize(const aclTensor* grad_output, const aclTensor* input,
                                                  const aclTensor* scale, const aclTensor* shift,
                                                  const aclTensor* grad_input, const aclTensor* grad_scale,
                                                  const aclTensor* grad_shift, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnModulateBackward, DFX_IN(grad_output, input, scale, shift),
                   DFX_OUT(grad_input, grad_scale, grad_shift));
    auto uniqueExecutor = CREATE_EXECUTOR();
    // CREATE_EXECUTOR() 内部使用普通 new，分配失败会抛 bad_alloc，此处 null 检查不可达，故删除。
    auto ret = CheckParams(grad_output, input, scale, shift, grad_input, grad_scale, grad_shift);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (grad_output->IsEmpty() || input->IsEmpty() || grad_input->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto grad_outputContiguous = l0op::Contiguous(grad_output, uniqueExecutor.get());
    CHECK_RET(grad_outputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto grad_outputCasted = l0op::Cast(grad_outputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(grad_outputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputCasted = l0op::Cast(inputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(inputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* scaleCasted = nullptr;
    const aclTensor* shiftCasted = nullptr;
    std::tuple<const aclTensor*, const aclTensor*, const aclTensor*> modulateBackWardresult;
    if (CheckNotNullForScaleAndShift(scale)) {
        auto scaleContiguous = l0op::Contiguous(scale, uniqueExecutor.get());
        CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        scaleCasted = l0op::Cast(scaleContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(scaleCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (CheckNotNullForScaleAndShift(shift)) {
        auto shiftContiguous = l0op::Contiguous(shift, uniqueExecutor.get());
        CHECK_RET(shiftContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        shiftCasted = l0op::Cast(shiftContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(shiftCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    modulateBackWardresult = l0op::ModulateGrad(grad_outputCasted, inputCasted, scaleCasted, shiftCasted,
                                                uniqueExecutor.get());
    const aclTensor* grad_inputresult = std::get<0>(modulateBackWardresult);
    const aclTensor* grad_scaleresult = std::get<1>(modulateBackWardresult);
    const aclTensor* grad_shiftresult = std::get<2>(modulateBackWardresult);
    auto castgrad_input = l0op::Cast(grad_inputresult, grad_input->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castgrad_input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopygrad_input = l0op::ViewCopy(castgrad_input, grad_input, uniqueExecutor.get());
    CHECK_RET(viewCopygrad_input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (grad_scale != nullptr) {
        auto castgrad_scale = l0op::Cast(grad_scaleresult, grad_input->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castgrad_scale != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopygrad_scale = l0op::ViewCopy(castgrad_scale, grad_scale, uniqueExecutor.get());
        CHECK_RET(viewCopygrad_scale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (grad_shift != nullptr) {
        auto castgrad_shift = l0op::Cast(grad_shiftresult, grad_input->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castgrad_shift != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopygrad_shift = l0op::ViewCopy(castgrad_shift, grad_shift, uniqueExecutor.get());
        CHECK_RET(viewCopygrad_shift != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnModulateBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnModulateBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
