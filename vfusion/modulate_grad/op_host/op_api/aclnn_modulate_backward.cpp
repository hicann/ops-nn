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

static const std :: initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std :: initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST_910C {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std :: initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std :: initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST_910C {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* grad_output, const aclTensor* input, const aclTensor* grad_input){
    OP_CHECK_NULL(grad_output,return false);
    OP_CHECK_NULL(input,return false);
    OP_CHECK_NULL(grad_input,return false);
    return true;
}

static bool CheckNotNullForScaleAndShift(const aclTensor* tensor){
    OP_CHECK_NULL(tensor,return false);
    return true;  
}

static bool CheckDtypeValid(const aclTensor* grad_output,const aclTensor* input, const aclTensor* scale,const aclTensor* shift,const aclTensor* grad_input,const aclTensor* grad_scale,const aclTensor* grad_shift){
    bool is910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion :: ASCEND910B);
    const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST=
        is910BSocVersion ? GRAD_DTYPE_SUPPORT_LIST_910B : GRAD_DTYPE_SUPPORT_LIST_910C;
    const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST=
        is910BSocVersion ? OUT_DTYPE_SUPPORT_LIST_910B : OUT_DTYPE_SUPPORT_LIST_910C;
    
    OP_CHECK_DTYPE_NOT_SUPPORT(grad_output,GRAD_DTYPE_SUPPORT_LIST,return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(input,GRAD_DTYPE_SUPPORT_LIST,return false);
    if (CheckNotNullForScaleAndShift(scale)){
        OP_CHECK_DTYPE_NOT_SUPPORT(scale,GRAD_DTYPE_SUPPORT_LIST,return false);    
    }
    if (CheckNotNullForScaleAndShift(shift)){
        OP_CHECK_DTYPE_NOT_SUPPORT(shift,GRAD_DTYPE_SUPPORT_LIST,return false);    
    }
    if (CheckNotNullForScaleAndShift(grad_scale)){
        OP_CHECK_DTYPE_NOT_SUPPORT(grad_scale,OUT_DTYPE_SUPPORT_LIST,return false);    
    }
    if (CheckNotNullForScaleAndShift(grad_shift)){
        OP_CHECK_DTYPE_NOT_SUPPORT(grad_shift,OUT_DTYPE_SUPPORT_LIST,return false);    
    }   

    OP_CHECK_DTYPE_NOT_SUPPORT(grad_input,OUT_DTYPE_SUPPORT_LIST,return false);
    return true;
}

static bool CheckMaxDimension(const aclTensor *grad_output, const aclTensor *input, const aclTensor *scale, const aclTensor *shift){
    op::Shape gradoutputShape = grad_output->GetViewShape();
    size_t gradoutputDimNum = gradoutputShape.GetDimNum();
    op::Shape inputShape = input->GetViewShape();
    size_t inputDimNum = inputShape.GetDimNum();
    if (gradoutputDimNum !=MAX_SUPPORT_DIM || inputDimNum !=MAX_SUPPORT_DIM){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of self must be 3");
        return false;
    }
    if ((grad_output->IsEmpty() || input->IsEmpty()) && (!scale->IsEmpty() || !shift->IsEmpty())){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when grad_output or input is empty, scale and shift must be empty");
        return false;
    }

    return true;
}

static bool CheckDimension(const aclTensor *grad_output, const aclTensor *input, const aclTensor *scale, const aclTensor *shift){
    auto gradoutput_Shape = grad_output->GetViewShape();
    //检查Format
    if(input->GetStorageFormat() != Format::FORMAT_ND || grad_output->GetStorageFormat() != Format::FORMAT_ND || scale->GetStorageFormat() != Format::FORMAT_ND || shift->GetStorageFormat() != Format::FORMAT_ND ){
        OP_LOGW("Format only support ND");
    }
    if (CheckNotNullForScaleAndShift(scale)){
        auto scale_Shape = scale->GetViewShape();
        if(scale_Shape.GetDimNum() != SCALE_DIM){
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,"scale dim is not equal 2");
            return false;
        }
        if(gradoutput_Shape.GetDim(0) != scale_Shape.GetDim(FIRST_DIM) || gradoutput_Shape.GetDim(LAST_DIM) != scale_Shape.GetDim(MIDDLE_DIM)){
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,"gradoutput_Shape dim0 or dim 2 not match scale dim 0 or dim 1");
            return false;
        }
    }
    if (CheckNotNullForScaleAndShift(shift)){
        auto shift_Shape = shift->GetViewShape();
        if(shift_Shape.GetDimNum() != SCALE_DIM){
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,"shift dim is not equal 2");
            return false;
        }
        if(gradoutput_Shape.GetDim(FIRST_DIM) != shift_Shape.GetDim(FIRST_DIM) || gradoutput_Shape.GetDim(LAST_DIM) != shift_Shape.GetDim(MIDDLE_DIM)){
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,"gradoutput_Shape dim0 or dim 2 not match shift dim 0 or dim 1");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* grad_output,const aclTensor* input, const aclTensor* scale,const aclTensor* shift,const aclTensor* grad_input,const aclTensor* grad_scale,const aclTensor* grad_shift){
    //1.scale和shift参数可以为空，所以只检查grad_out input grad_input这三个参数是否为空指针
    CHECK_RET(CheckNotNull(grad_output, input, grad_input),ACLNN_ERR_PARAM_NULLPTR);
    //2.检查数据类型是否在支持的范围内
    CHECK_RET(CheckDtypeValid(grad_output, input, scale, shift, grad_input, grad_scale, grad_shift),ACLNN_ERR_PARAM_INVALID);
    //3.检查维度是否超过8维 在这里不会超过8维
    CHECK_RET(CheckMaxDimension(grad_output, input, scale, shift),ACLNN_ERR_PARAM_INVALID);
    //4.检查维度是否匹配
    CHECK_RET(CheckDimension(grad_output, input, scale, shift),ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnModulateBackwardGetWorkspaceSize(const aclTensor *grad_output,const aclTensor *input,const aclTensor *scale, const aclTensor *shift,const aclTensor *grad_input,const aclTensor *grad_scale, const aclTensor *grad_shift,uint64_t *workspaceSize,aclOpExecutor **executor){
    OP_CHECK_COMM_INPUT(workspaceSize,executor);
    L2_DFX_PHASE_1(aclnnModulateBackward, DFX_IN(grad_output,input,scale,shift), DFX_OUT(grad_input,grad_scale,grad_shift));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(grad_output,input,scale,shift,grad_input,grad_scale,grad_shift);
    CHECK_RET(ret == ACLNN_SUCCESS,ret);
    if (grad_output-> IsEmpty() || input->IsEmpty() || grad_input->IsEmpty()){
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;}
    auto grad_outputContiguous =l0op::Contiguous(grad_output,uniqueExecutor.get());
    CHECK_RET(grad_outputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto grad_outputCasted = l0op::Cast(grad_outputContiguous,op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(grad_outputCasted !=nullptr,ACLNN_ERR_INNER_NULLPTR);
    auto inputContiguous =l0op::Contiguous(input,uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputCasted = l0op::Cast(inputContiguous,op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(inputCasted !=nullptr,ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* scaleCasted = nullptr;
    const aclTensor* shiftCasted = nullptr;
    std::tuple<const aclTensor*,const aclTensor*,const aclTensor*> modulateBackWardresult;
    if(CheckNotNullForScaleAndShift(scale)){
        auto scaleContiguous =l0op::Contiguous(scale,uniqueExecutor.get());
        CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        scaleCasted = l0op::Cast(scaleContiguous,op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(scaleCasted !=nullptr,ACLNN_ERR_INNER_NULLPTR);}
    if(CheckNotNullForScaleAndShift(shift)){
        auto shiftContiguous =l0op::Contiguous(shift,uniqueExecutor.get());
        CHECK_RET(shiftContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        shiftCasted = l0op::Cast(shiftContiguous,op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(shiftCasted !=nullptr,ACLNN_ERR_INNER_NULLPTR);}

    modulateBackWardresult = l0op::ModulateGrad(grad_outputCasted,inputCasted,scaleCasted,shiftCasted, uniqueExecutor.get());
    const aclTensor *grad_inputresult = std::get<0>(modulateBackWardresult);
    const aclTensor *grad_scaleresult = std::get<1>(modulateBackWardresult);
    const aclTensor *grad_shiftresult = std::get<2>(modulateBackWardresult);
    auto castgrad_input = l0op::Cast(grad_inputresult,grad_input->GetDataType(),uniqueExecutor.get());
    CHECK_RET(castgrad_input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopygrad_input = l0op::ViewCopy(castgrad_input,grad_input,uniqueExecutor.get());
    CHECK_RET(viewCopygrad_input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (grad_scale != nullptr){
        auto castgrad_scale = l0op::Cast(grad_scaleresult,grad_input->GetDataType(),uniqueExecutor.get());
        CHECK_RET(castgrad_scale != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopygrad_scale = l0op::ViewCopy(castgrad_scale,grad_scale,uniqueExecutor.get());
        CHECK_RET(viewCopygrad_scale != nullptr, ACLNN_ERR_INNER_NULLPTR);}
    if (grad_shift != nullptr){
        auto castgrad_shift = l0op::Cast(grad_shiftresult,grad_input->GetDataType(),uniqueExecutor.get());
        CHECK_RET(castgrad_shift != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopygrad_shift = l0op::ViewCopy(castgrad_shift,grad_shift,uniqueExecutor.get());
        CHECK_RET(viewCopygrad_shift != nullptr, ACLNN_ERR_INNER_NULLPTR);}
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnModulateBackward(void *workspace, uint64_t workspaceSize,aclOpExecutor *executor, const aclrtStream stream){
    L2_DFX_PHASE_2(aclnnModulateBackward);
    return CommonOpExecutorRun(workspace,workspaceSize,executor,stream);
}

