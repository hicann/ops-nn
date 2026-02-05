/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kl_div_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "kl_div_loss_grad.h"
#include "level0/reduce_sum_op.h"
#include "level0/broadcast_to.h"
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

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

enum Reduction
{
    None = 0,
    Mean = 1,
    Sum = 2,
    Batchmean = 3,
    End
};

static const char* REDUCTION_NONE = "none";
static const char* REDUCTION_MEAN = "mean";
static const char* REDUCTION_SUM = "sum";
static const char* REDUCTION_BATCHMEAN = "batchmean";

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(SocVersion socVersion)
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        socVersion == SocVersion::ASCEND950) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    if (socVersion == SocVersion::ASCEND910 || socVersion == SocVersion::ASCEND310P) {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
    return emptyDtypes;
}

static bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* out)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    const auto& supportList = GetSupportDtypeList(socVersion);

    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
    // 检查gradOutput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    // 检查target的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(target, supportList, return false);
    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    return true;
}

constexpr size_t MAX_DIM_LEN = 8;

static bool CheckShape(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(target, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

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
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull4Tensor(gradOutput, self, target, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, target, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出输出shape
    CHECK_RET(CheckShape(gradOutput, self, target, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const char* GetReductionStr(int64_t reduction)
{
    if (reduction == None) {
        return REDUCTION_NONE;
    } else if (reduction == Mean) {
        return REDUCTION_MEAN;
    } else if (reduction == Sum) {
        return REDUCTION_SUM;
    } else if (reduction == Batchmean) {
        return REDUCTION_BATCHMEAN;
    } else {
        return REDUCTION_NONE;
    }
}

static const aclTensor* BroadcastTensor(const aclTensor* self, const op::Shape broadcastShape, aclOpExecutor* executor)
{
    // 如果self的shape与broadcast的不一致，进行BroadcastTo
    if (self->GetViewShape() != broadcastShape) {
        auto broadcastShapeIntArray = GetBroadcastShapeLossBackward(broadcastShape, executor);
        if (broadcastShapeIntArray != nullptr) {
            return l0op::BroadcastTo(self, broadcastShapeIntArray, executor);
        }
    }
    return self;
}

static const aclTensor* ReduceSumTensor(const aclTensor* grad, const op::Shape outShape, aclOpExecutor* executor)
{
    // 如果grad的shape与outShape不一致，进行ReduceSum
    if (grad->GetViewShape() != outShape) {
        size_t outDimNum = outShape.GetDimNum();
        size_t gradDimNum = grad->GetViewShape().GetDimNum();
        size_t startDim = gradDimNum - outDimNum;
        size_t dimIdx = startDim;
        std::vector<int64_t> appendDim;
        for (size_t i = 0; i < startDim; ++i) {
            appendDim.push_back(i);
        }
        for (size_t j = startDim; j < gradDimNum; ++j) {
            if (outShape[j - startDim] != (grad->GetViewShape())[j]) {
                appendDim.push_back(j);
                dimIdx++;
            }
        }
        auto axes = executor->AllocIntArray(appendDim.data(), dimIdx);
        auto out = l0op::ReduceSumOp(grad, axes, true, executor);
        CHECK_RET(out != nullptr, nullptr);
        auto outShapeIntArray = GetBroadcastShapeLossBackward(outShape, executor);
        return l0op::Reshape(out, outShapeIntArray, executor);
    }
    return grad;
}

aclnnStatus aclnnKlDivBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction, bool logTarget,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnKlDivBackward, DFX_IN(gradOutput, self, target, reduction, logTarget), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, target, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        // 根据实际支持情况补充
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

    // 固定写法，将输入target转换成连续的tensor
    auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputCasted = gradOutputContiguous;
    auto selfCasted = selfContiguous;
    auto targetCasted = targetContiguous;
    if (!(gradOutput->GetDataType() == op::DataType::DT_FLOAT16 && self->GetDataType() == op::DataType::DT_FLOAT16 && target->GetDataType() == op::DataType::DT_FLOAT16)) {
        auto promoteType = op::DataType::DT_FLOAT;

        // 将输入gradoutput的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入target的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        targetCasted = l0op::Cast(targetContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    op::Shape broadcastShape;
    BroadcastInferShape(target->GetViewShape(), self->GetViewShape(), broadcastShape);

    // 判断self是否需要进行broadcast
    auto selfBroadcast = BroadcastTensor(selfCasted, broadcastShape, uniqueExecutor.get());
    CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行计算
    auto grad = l0op::KlDivLossGrad(
        gradOutputCasted, selfBroadcast, targetCasted, GetReductionStr(reduction), logTarget, uniqueExecutor.get());
    CHECK_RET(grad != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 根据grad的shape是否与out的shape相同，判断是否需要reduce
    auto gradReduce = ReduceSumTensor(grad, out->GetViewShape(), uniqueExecutor.get());
    CHECK_RET(gradReduce != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(gradReduce, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKlDivBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKlDivBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
