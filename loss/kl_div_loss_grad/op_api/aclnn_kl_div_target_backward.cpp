/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kl_div_target_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/reduce_sum_op.h"
#include "level0/broadcast_to.h"
#include "level0/add.h"
#include "level0/log.h"
#include "level0/sub.h"
#include "level0/mul.h"
#include "level0/exp.h"
#include "level0/div.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "loss/common/level2_base_loss.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

int64_t GetElementNum(const op::Shape &shape)
{
    int64_t ret = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        ret *= shape[i];
    }
    return ret;
}

enum Reduction
{
    None = 0,
    Mean = 1,
    Sum = 2,
    Batchmean = 3,
    End
};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(SocVersion socVersion)
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    return emptyDtypes;
}

static bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* gradTarget)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    const auto& supportList = GetSupportDtypeList(socVersion);

    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), gradTarget->GetDataType(), return false);
    // 检查gradOutput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    // 检查target的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);
    // 检查gradTarget的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradTarget, supportList, return false);
    // target和gradTarget数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(gradTarget, target->GetDataType(), return false);

    return true;
}

constexpr size_t MAX_DIM_LEN = 8;

static bool CheckShape(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* gradTarget)
{
    OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(target, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(gradTarget, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    op::Shape broadcastGradShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, target, broadcastShape, return false);
    if (!BroadcastInferShape(gradOutput->GetViewShape(), broadcastShape, broadcastGradShape) ||
        broadcastShape != broadcastGradShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Except shape of gradOutput must broadcast to %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(gradOutput->GetViewShape()).GetString());
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(self, gradTarget, return false);

    return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* gradTarget) {
  // 如果输入格式是私有格式，记录日志，直接报错
  if (op::IsPrivateFormat(gradOutput->GetStorageFormat()) || op::IsPrivateFormat(self->GetStorageFormat()) || 
      op::IsPrivateFormat(target->GetStorageFormat()) || op::IsPrivateFormat(gradTarget->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW、NCL. Actual: gradOutput [%s], self [%s], target [%s], gradTarget [%s] ",
            ToString(gradOutput->GetStorageFormat()).GetString(), ToString(self->GetStorageFormat()).GetString(),
            ToString(target->GetStorageFormat()).GetString(), ToString(gradTarget->GetStorageFormat()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, aclTensor* gradTarget)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull4Tensor(gradOutput, self, target, gradTarget), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, target, gradTarget), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输出输出shape
    CHECK_RET(CheckShape(gradOutput, self, target, gradTarget), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持
    CHECK_RET(CheckFormat(gradOutput, self, target, gradTarget), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclIntArray* ComputeBroadcastShapeLossBackward(const op::Shape broadcastShape, aclOpExecutor* executor)
{
    int64_t tensorSize = (int64_t)(broadcastShape.GetDimNum());
    std::vector<int64_t> tensorShape(tensorSize);
    for (int i = 0; i < tensorSize; i++) {
        tensorShape[i] = broadcastShape[i];
    }
    return executor->AllocIntArray(tensorShape.data(), tensorSize);
}

static const aclTensor* BroadcastTensor(const aclTensor* self, const op::Shape broadcastShape, aclOpExecutor* executor)
{
    // 如果self的shape与broadcast的不一致，进行BroadcastTo
    if (self->GetViewShape() != broadcastShape) {
        auto broadcastShapeIntArray = ComputeBroadcastShapeLossBackward(broadcastShape, executor);
        if (broadcastShapeIntArray != nullptr) {
            return l0op::BroadcastTo(self, broadcastShapeIntArray, executor);
        }
    }
    return self;
}

static const aclTensor* ReduceSumTensor(const aclTensor* grad, const op::Shape gradTargetShape, aclOpExecutor* executor)
{
    // 如果grad的shape与gradTargetShape不一致，进行ReduceSum
    if (grad->GetViewShape() != gradTargetShape) {
        size_t gradTargetDimNum = gradTargetShape.GetDimNum();
        size_t gradDimNum = grad->GetViewShape().GetDimNum();
        size_t startDim = gradDimNum - gradTargetDimNum;
        size_t dimIdx = startDim;
        std::vector<int64_t> appendDim;
        for (size_t i = 0; i < startDim; ++i) {
            appendDim.push_back(i);
        }
        for (size_t j = startDim; j < gradDimNum; ++j) {
            if (gradTargetShape[j - startDim] != (grad->GetViewShape())[j]) {
                appendDim.push_back(j);
                dimIdx++;
            }
        }
        auto axes = executor->AllocIntArray(appendDim.data(), dimIdx);
        auto gradTarget = l0op::ReduceSumOp(grad, axes, true, executor);
        CHECK_RET(gradTarget != nullptr, nullptr);
        auto gradTargetShapeIntArray = ComputeBroadcastShapeLossBackward(gradTargetShape, executor);
        return l0op::Reshape(gradTarget, gradTargetShapeIntArray, executor);
    }
    return grad;
}

static const aclTensor* ComputeGradForKlDiv(
    const aclTensor* targetBroadcast, const aclTensor* selfCasted, 
    const aclTensor* gradOutputCasted, bool logTarget, aclOpExecutor* executor)
{
    auto oneScalar = executor->AllocScalar(1.0);
    auto oneTensor = executor->ConvertToTensor(oneScalar, op::DataType::DT_FLOAT);
    if (!logTarget) {
        const float LOG_BASE = -1.0f;
        const float LOG_SCALE = 1.0f;
        const float LOG_SHIFT = 0.0f;
        auto logOut = l0op::Log(targetBroadcast, LOG_BASE, LOG_SCALE, LOG_SHIFT, executor);
        auto logPlusOne = l0op::Add(logOut, oneTensor, executor);
        auto termSubSelf = l0op::Sub(logPlusOne, selfCasted, executor);
        return l0op::Mul(gradOutputCasted, termSubSelf, executor);
    } else {
        auto expOut = l0op::Exp(targetBroadcast, executor);
        auto logPlusOne = l0op::Add(targetBroadcast, oneTensor, executor);
        auto termSubSelf = l0op::Sub(logPlusOne, selfCasted, executor);
        auto mulExp = l0op::Mul(expOut, termSubSelf, executor);
        return l0op::Mul(gradOutputCasted, mulExp, executor);
    }
}

static aclnnStatus ExecuteKlDivTargetBackward(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction, bool logTarget,
    aclTensor* gradTarget, aclOpExecutor* executor)
{
    auto promoteType = op::DataType::DT_FLOAT; 
 
    // 固定写法，将输入gradOutput转换成连续的tensor	 
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, executor);	 
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);	 
 
    // 将输入gradoutput的数据类型转换成隐式数据类型，根据具体算子语义按需调用 
    auto gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, executor); 
    CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR); 

    // 固定写法，将输入self转换成连续的tensor	 
    auto selfContiguous = l0op::Contiguous(self, executor);	 
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
 
    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用 
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, executor); 
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR); 
 
    // 固定写法，将输入target转换成连续的tensor	 
    auto targetContiguous = l0op::Contiguous(target, executor);	 
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);	 
 
    // 将输入target的数据类型转换成隐式数据类型，根据具体算子语义按需调用	 
    auto targetCasted = l0op::Cast(targetContiguous, promoteType, executor);	 
    CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    op::Shape broadcastShape;
    BroadcastInferShape(target->GetViewShape(), self->GetViewShape(), broadcastShape);

    // 判断self是否需要进行broadcast
    auto targetBroadcast = BroadcastTensor(targetCasted, broadcastShape, executor);
    CHECK_RET(targetBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行计算
    auto grad = ComputeGradForKlDiv(targetBroadcast, selfCasted, gradOutputCasted, logTarget, executor);
    CHECK_RET(grad != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (reduction == Mean) {
        auto totalElementScalar = executor->AllocScalar(GetElementNum(grad->GetViewShape()));
        auto totalElementTensor = executor->ConvertToTensor(totalElementScalar, promoteType);
        grad = l0op::Div(grad, totalElementTensor, executor);
        CHECK_RET(grad != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (reduction == Batchmean){
        auto batchSizeScalar = executor->AllocScalar(grad->GetViewShape()[0]);
        auto batchSizeTensor = executor->ConvertToTensor(batchSizeScalar, promoteType);
        grad = l0op::Div(grad, batchSizeTensor, executor);
        CHECK_RET(grad != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    
    // 根据grad的shape是否与gradTarget的shape相同，判断是否需要reduce
    auto gradReduce = ReduceSumTensor(grad, gradTarget->GetViewShape(), executor);
    CHECK_RET(gradReduce != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出gradTarget的数据类型
    auto castOut = l0op::Cast(gradReduce, gradTarget->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出gradTarget上，gradTarget可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, gradTarget, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKlDivTargetBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction, bool logTarget,
    aclTensor* gradTarget, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnKlDivTargetBackward, DFX_IN(gradOutput, self, target, reduction, logTarget), DFX_OUT(gradTarget));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, target, gradTarget);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    ret = ExecuteKlDivTargetBackward(gradOutput, self, target, reduction, logTarget, gradTarget, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKlDivTargetBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKlDivTargetBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
