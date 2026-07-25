/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_nll_loss.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/div.h"
#include "level0/fill.h"
#include "nll_loss.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::string REDUCTION_NONE = "none";
static const std::string REDUCTION_MEAN = "mean";
static const std::string REDUCTION_SUM = "sum";
static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;
static const int64_t MAX_SELF_DIM_NUM = 2;
static const int64_t OUT_COUNTS = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                 op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> TARGET_DTYPE_SUPPORT_LIST = {DataType::DT_INT64, DataType::DT_UINT8,
                                                                              DataType::DT_INT32};

static bool CheckNotNull(const aclTensor* self, const aclTensor* target, const aclTensor* weight, const aclTensor* out,
                         const aclTensor* totalWeightOut)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(target, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(totalWeightOut, return false);
    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510: {
            return ASCEND910B_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* target, const aclTensor* weight,
                            const aclTensor* out, const aclTensor* totalWeightOut)
{
    OP_CHECK_DTYPE_NOT_MATCH(weight, self->GetDataType(), return false);
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), totalWeightOut->GetDataType(), return false);

    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(target, TARGET_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckReduction(int64_t reduction)
{
    // 检查self和other能否做数据类型推导
    if (reduction > REDUCTION_SUM_NUM || reduction < REDUCTION_NONE_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reduction should be between 0 and 2, but current is %ld.", reduction);
        return false;
    }
    return true;
}

static void CheckFormat(const aclTensor* self)
{
    // 检查self和other能否做数据类型推导
    if (self->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("Format of self gets [%s], this format may lead to precision failure.",
                op::ToString(self->GetStorageFormat()).GetString());
    }
}

static bool CheckShape(const aclTensor* self, const aclTensor* target, const aclTensor* weight, const aclTensor* out,
                       const aclTensor* totalWeightOut, int64_t reduction)
{
    size_t selfDimNum = self->GetViewShape().GetDimNum();
    OP_CHECK(selfDimNum > 0 && selfDimNum <= MAX_SELF_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor should be 1D or 2D."), return false);

    size_t targetDimNum = target->GetViewShape().GetDimNum();
    OP_CHECK(targetDimNum <= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "0D or 1D target tensor expected, multi-target not supported."),
             return false);

    bool noBatchDim = selfDimNum == 1 && targetDimNum == 0;
    bool oneBatchDim = selfDimNum == 2 && targetDimNum == 1;
    OP_CHECK(noBatchDim || (oneBatchDim && self->GetViewShape().GetDim(0) == target->GetViewShape().GetDim(0)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "when self is 1D, target need be 0D; when self is 2D, target need be 1D, "
                     "and target first dimension need equal to self first dimension, but (got self: %s, target: %s)",
                     op::ToString(self->GetViewShape()).GetString(), op::ToString(target->GetViewShape()).GetString()),
             return false);

    const auto nClasses = self->GetViewShape().GetDim(selfDimNum - 1);
    OP_CHECK(weight->GetViewShape().GetDimNum() == 1 && weight->GetViewShape().GetDim(0) == nClasses,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Weight tensor should be defined for all %ld classes but got weight tensor of shape: %s", nClasses,
                     op::ToString(weight->GetViewShape()).GetString()),
             return false);

    if (reduction == 0 && selfDimNum == MAX_SELF_DIM_NUM) {
        OP_CHECK(
            out->GetViewShape().GetDimNum() == 1 && out->GetViewShape().GetDim(0) == self->GetViewShape().GetDim(0),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expect out shape [%ld], but got: %s.", self->GetViewShape().GetDim(0),
                    op::ToString(out->GetViewShape()).GetString()),
            return false);
    } else {
        OP_CHECK(out->GetViewShape().GetDimNum() <= 1 && out->GetViewShape().GetShapeSize() == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected a single element out tensor, but got: %s",
                         op::ToString(out->GetViewShape()).GetString()),
                 return false);
    }

    OP_CHECK(totalWeightOut->GetViewShape().GetShapeSize() == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of totalWeightOut tensor should be [1], but current is %s.",
                     op::ToString(totalWeightOut->GetViewShape()).GetString()),
             return false);

    if (self->IsEmpty() && nClasses == 0 && target->GetViewShape().GetShapeSize() > 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Target is out of bounds");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* target, const aclTensor* weight,
                               int64_t reduction, const aclTensor* out, const aclTensor* totalWeightOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, target, weight, out, totalWeightOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, target, weight, out, totalWeightOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查reduction是否符合规则
    CHECK_RET(CheckReduction(reduction), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出输出shape
    CHECK_RET(CheckShape(self, target, weight, out, totalWeightOut, reduction), ACLNN_ERR_PARAM_INVALID);

    CheckFormat(self);

    return ACLNN_SUCCESS;
}

static const std::string& GetReductionStr(int64_t reduction)
{
    if (reduction == 0) {
        return REDUCTION_NONE;
    } else if (reduction == 1) {
        return REDUCTION_MEAN;
    } else {
        return REDUCTION_SUM;
    }
}

static aclnnStatus FillScalar(aclTensor* out, float val, aclOpExecutor* executor)
{
    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());

    FVector<int64_t> tmp = {1};
    auto dims = executor->ConvertToTensor(tmp.data(), tmp.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());

    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus NLLLossEmptyTensorCompute(int64_t reduction, aclTensor* out, aclTensor* totalWeightOut,
                                             aclOpExecutor* executor)
{
    aclnnStatus ret;
    if (reduction == 0) {
        return ACLNN_SUCCESS;
    } else if (reduction == 1) {
        ret = FillScalar(out, NAN, executor);
    } else {
        ret = FillScalar(out, 0, executor);
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = FillScalar(totalWeightOut, 0, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

// contiguous + cast (+ 1D unsqueeze for self) of the three inputs.
static aclnnStatus NLLLossPrepareInputs(const aclTensor* self, const aclTensor* target, const aclTensor* weight,
                                        aclOpExecutor* executor, const aclTensor** selfReshape,
                                        const aclTensor** targetCasted, const aclTensor** weightCast)
{
    bool regbase = Ops::NN::AclnnUtil::IsRegbase();
    op::DataType promoteType = regbase ? self->GetDataType() :
                                         (self->GetDataType() == op::DataType::DT_BF16 ? op::DataType::DT_BF16 :
                                                                                         op::DataType::DT_FLOAT);

    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfCast = l0op::Cast(selfContiguous, promoteType, executor);
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *selfReshape = (regbase || self->GetViewShape().GetDimNum() != 1) ?
                       selfCast :
                       l0op::UnsqueezeNd(selfCast, static_cast<int64_t>(0), executor);
    CHECK_RET(*selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto targetContiguous = l0op::Contiguous(target, executor);
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    op::DataType targetPromoteType = regbase ?
                                         target->GetDataType() :
                                         (target->GetDataType() == op::DataType::DT_INT64 ? op::DataType::DT_INT64 :
                                                                                            op::DataType::DT_INT32);
    *targetCasted = l0op::Cast(targetContiguous, targetPromoteType, executor);
    CHECK_RET(*targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightContiguous = l0op::Contiguous(weight, executor);
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *weightCast = l0op::Cast(weightContiguous, promoteType, executor);
    CHECK_RET(*weightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// cast + view-copy the loss (and total weight when reduction != none) to the user outputs.
static aclnnStatus NLLLossCopyToOutput(const aclTensor* loss, const aclTensor* totalWeight, int64_t reduction,
                                       aclTensor* out, aclTensor* totalWeightOut, aclOpExecutor* executor)
{
    auto castOut = l0op::Cast(loss, out->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (reduction != 0) {
        auto castTotalWeightOut = l0op::Cast(totalWeight, totalWeightOut->GetDataType(), executor);
        CHECK_RET(castTotalWeightOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyTotalweightResult = l0op::ViewCopy(castTotalWeightOut, totalWeightOut, executor);
        CHECK_RET(viewCopyTotalweightResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNLLLossGetWorkspaceSize(const aclTensor* self, const aclTensor* target, const aclTensor* weight,
                                         int64_t reduction, int64_t ignoreIndex, aclTensor* out,
                                         aclTensor* totalWeightOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNLLLoss, DFX_IN(self, target, weight, reduction, ignoreIndex), DFX_OUT(out, totalWeightOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(self, target, weight, reduction, out, totalWeightOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        ret = NLLLossEmptyTensorCompute(reduction, out, totalWeightOut, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* selfReshape = nullptr;
    const aclTensor* targetCasted = nullptr;
    const aclTensor* weightCast = nullptr;
    ret = NLLLossPrepareInputs(self, target, weight, uniqueExecutor.get(), &selfReshape, &targetCasted, &weightCast);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    std::array<const aclTensor*, OUT_COUNTS> lossOut = l0op::NLLLoss(
        selfReshape, targetCasted, weightCast, GetReductionStr(reduction), ignoreIndex, uniqueExecutor.get());
    CHECK_RET(lossOut[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(lossOut[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* loss;
    if (self->GetViewShape().GetDimNum() == 1 && reduction == 0) {
        loss = Ops::NN::AclnnUtil::IsRegbase() ?
                   lossOut[0] :
                   l0op::SqueezeNd(lossOut[0], static_cast<int64_t>(0), uniqueExecutor.get());
    } else {
        loss = lossOut[0];
    }
    CHECK_RET(loss != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ret = NLLLossCopyToOutput(loss, lossOut[1], reduction, out, totalWeightOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNLLLoss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNLLLoss);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
