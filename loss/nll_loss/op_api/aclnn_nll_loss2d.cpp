/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_nll_loss2d.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/div.h"
#include "level0/fill.h"
#include "nll_loss.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::string REDUCTION_NONE = "none";
static const std::string REDUCTION_MEAN = "mean";
static const std::string REDUCTION_SUM = "sum";

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 根据API定义，需要列出target所能支持的所有dtype
static const std::initializer_list<op::DataType> TARGET_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT64, DataType::DT_UINT8, DataType::DT_INT32};

static const std::initializer_list<DataType>& GetSelfDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    return ASCEND910_DTYPE_SUPPORT_LIST;
}

static inline bool CheckNotNull(
    const aclTensor* self, const aclTensor* target, const aclTensor* weight, const aclTensor* out,
    const aclTensor* totalWeightOut)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(target, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(totalWeightOut, return false);

    return true;
}

static void CheckFormat(const aclTensor* self)
{
    // 检查self和other能否做数据类型推导
    if (self->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW(
            "Format of self gets [%s], this format may lead to precision failure.",
            op::ToString(self->GetStorageFormat()).GetString());
    }
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* target, const aclTensor* weight, const aclTensor* out,
    const aclTensor* totalWeightOut)
{
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, GetSelfDtypeSupportList(), return false);

    // 检查类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(self, weight->GetDataType(), return false);

    // 检查self能否转换成out类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), totalWeightOut->GetDataType(), return false);

    // 检查target的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(target, TARGET_DTYPE_SUPPORT_LIST, return false);
    return true;
}

constexpr int64_t MIN_REDUCTION = 0;
constexpr int64_t MEDIAN_REDUCTION = 1;
constexpr int64_t MAX_REDUCTION = 2;
constexpr int64_t PERMUTE_DIM4 = 4;
static inline bool CheckReduction(int64_t reduction)
{
    // 检查self和other能否做数据类型推导
    if (reduction > MAX_REDUCTION || reduction < MIN_REDUCTION) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reduction should be between 0 and 2, but current is [%ld].", reduction);
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* self, const aclTensor* target, const aclTensor* weight, const aclTensor* out,
    const aclTensor* totalWeightOut, int64_t reduction)
{
    size_t selfDimNum = self->GetViewShape().GetDimNum();
    OP_CHECK(
        selfDimNum == 4, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor dim should be 4D, other dim not supported."),
        return false);

    const auto nClasses = self->GetViewShape().GetDim(1);
    OP_CHECK(
        weight->GetViewShape().GetDim(0) == nClasses,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor x[1] should be equal to weight[0]."), return false);

    size_t targetDimNum = target->GetViewShape().GetDimNum();
    OP_CHECK(
        targetDimNum == 3, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "target tensor dim should be 3D, other dim not supported."),
        return false);

    const auto input0 = self->GetViewShape().GetDim(0);
    const auto input2 = self->GetViewShape().GetDim(2);
    const auto input3 = self->GetViewShape().GetDim(3);

    const auto target0 = target->GetViewShape().GetDim(0);
    const auto target1 = target->GetViewShape().GetDim(1);
    const auto target2 = target->GetViewShape().GetDim(2);

    OP_CHECK(
        input0 == target0 && input2 == target1 && input3 == target2,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "size mismatch got input: [%s], target: [%s].",
            op::ToString(self->GetViewShape()).GetString(), op::ToString(target->GetViewShape()).GetString()),
        return false);

    if (reduction == 0) {
        OP_CHECK_SHAPE_NOT_EQUAL(out, target, return false);
    } else {
        OP_CHECK(
            out->Size() == 1,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Shape of out tensor should be [1], but current is [%s].",
                op::ToString(out->GetViewShape()).GetString()),
            return false);
    }

    OP_CHECK(
        totalWeightOut->Size() == 1,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of totalWeightOut tensor should be [1], but current is [%s].",
            op::ToString(totalWeightOut->GetViewShape()).GetString()),
        return false);
    return true;
}

/**
 * param类型检查
 */
static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* target, const aclTensor* weight, int64_t reduction, const aclTensor* out,
    const aclTensor* totalWeightOut)
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
    FVector<int64_t> tmp = {1};
    auto dims = executor->ConvertToTensor(tmp.data(), tmp.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus NLLLossEmptyTensorCompute(
    int64_t reduction, aclTensor* out, aclTensor* totalWeightOut, aclOpExecutor* executor)
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

aclnnStatus aclnnNLLLoss2dGetWorkspaceSize(
    const aclTensor* self, const aclTensor* target, const aclTensor* weight, int64_t reduction, int64_t ignoreIndex,
    aclTensor* out, aclTensor* totalWeightOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNLLLoss2d, DFX_IN(self, target, weight, reduction, ignoreIndex), DFX_OUT(out, totalWeightOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, target, weight, reduction, out, totalWeightOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // normal_ API的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty() || target->IsEmpty() || weight->IsEmpty()) {
        // 根据实际支持情况补充
        ret = NLLLossEmptyTensorCompute(reduction, out, totalWeightOut, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    op::DataType promoteType;
    if (socVersion == SocVersion::ASCEND910_95) {
        promoteType = self->GetDataType();
    } else {
        promoteType = self->GetDataType() == op::DataType::DT_BF16 ? op::DataType::DT_BF16 : op::DataType::DT_FLOAT;
    }
    // self -> selfContiguous -> selfFormat -> selfPermute -> selfReShape
    // [Contiguous] 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCast = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // target --> targetContiguous --> targetShape
    // [Contiguous]，将输入target转换成连续的tensor
    op::DataType targetPromoteType;
    auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入weight转换成连续的tensor
    auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightCasted = l0op::Cast(weightContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(weightCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* loss;
    const aclTensor* totalWeight;

    // [permute] 完成4维矩阵的维度转换
    if (socVersion != SocVersion::ASCEND910_95) {
        // [ReFormat] 类型转换
        auto selfFormat = l0op::ReFormat(selfCast, static_cast<op::Format>(ACL_FORMAT_ND));
        const aclTensor* selfPermute;
        if (selfFormat->GetViewShape().GetDim(1) == 1) {
            // 不需要转换
            selfPermute = selfFormat;
        } else {
            const int64_t permuteList[] = {0, 2, 3, 1};
            selfPermute = l0op::Transpose(
                selfFormat, uniqueExecutor.get()->AllocIntArray(permuteList, PERMUTE_DIM4), uniqueExecutor.get());
            CHECK_RET(selfPermute != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // [reshape] 维度转换为2维
        const int64_t reshapeList[] = {-1, self->GetViewShape().GetDim(1)};
        auto selfReshape =
            l0op::Reshape(selfPermute, uniqueExecutor.get()->AllocIntArray(reshapeList, 2), uniqueExecutor.get());
        CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // [reshape] 完成维度降维为一维
        const int64_t tarReshapeList[] = {-1};
        auto targetReshape = l0op::Reshape(
            targetContiguous, uniqueExecutor.get()->AllocIntArray(tarReshapeList, 1), uniqueExecutor.get());

        // [Cast] 进行类型转换
        targetPromoteType =
            target->GetDataType() == op::DataType::DT_INT64 ? op::DataType::DT_INT64 : op::DataType::DT_INT32;

        auto targetCasted = l0op::Cast(targetReshape, targetPromoteType, uniqueExecutor.get());
        CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 进行nllloss计算
        std::array<const aclTensor*, 2> lossOut = l0op::NLLLoss(
            selfReshape, targetCasted, weightCasted, GetReductionStr(reduction), ignoreIndex, uniqueExecutor.get());
        CHECK_RET(lossOut[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(lossOut[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

        totalWeight = lossOut[1];
        if (reduction != 0) {
            loss = lossOut[0];
        } else {
            // 将輸出self按照需要重写排列
            const int64_t outSquence[] = {
                target->GetViewShape().GetDim(0), target->GetViewShape().GetDim(1), target->GetViewShape().GetDim(2)};
            size_t dims = target->GetViewShape().GetDimNum();
            loss =
                l0op::Reshape(lossOut[0], uniqueExecutor.get()->AllocIntArray(outSquence, dims), uniqueExecutor.get());
            CHECK_RET(loss != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        targetPromoteType = target->GetDataType();
        auto targetCasted = l0op::Cast(targetContiguous, targetPromoteType, uniqueExecutor.get());
        CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        std::array<const aclTensor*, 2> lossOut = l0op::NLLLoss(
            selfCast, targetCasted, weightCasted, GetReductionStr(reduction), ignoreIndex, uniqueExecutor.get());
        CHECK_RET(lossOut[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(lossOut[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

        loss = lossOut[0];
        totalWeight = lossOut[1];
    }

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(loss, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (reduction != 0) {
        // 固定写法，将计算结果转换成输出out的数据类型
        auto castTotalWeightOut = l0op::Cast(totalWeight, totalWeightOut->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castTotalWeightOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResultWeight = l0op::ViewCopy(castTotalWeightOut, totalWeightOut, uniqueExecutor.get());
        CHECK_RET(viewCopyResultWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNLLLoss2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNLLLoss2d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
