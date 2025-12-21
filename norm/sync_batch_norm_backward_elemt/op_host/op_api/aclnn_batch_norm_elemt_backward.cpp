/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/realdiv.h"
#include "level0/reduce_sum_op.h"
#include "level0/unsqueeze.h"
#include "../../../norm_common/op_host/norm_tensor_util.h"
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "batch_norm_elemt_backward.h"
#include "aclnn_batch_norm_elemt_backward.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t DIM_TWO = 2;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_BF16_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> COUNTER_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static bool CheckNotNull(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    [[maybe_unused]]const aclTensor*  weight, const aclTensor* sumDy, const aclTensor* sumDyXmu, const aclTensor* counter,
    aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(invstd, return false);
    OP_CHECK_NULL(sumDy, return false);
    OP_CHECK_NULL(sumDyXmu, return false);
    OP_CHECK_NULL(counter, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, const aclTensor* sumDy, const aclTensor* sumDyXmu, const aclTensor* counter,
    const aclTensor* gradInput)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mean, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(invstd, DTYPE_SUPPORT_LIST, return false);
    if (weight != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weight, DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(sumDy, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sumDyXmu, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(counter, COUNTER_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool Check910BDtypeValid(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_BF16_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_BF16_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mean, DTYPE_SUPPORT_BF16_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(invstd, DTYPE_SUPPORT_BF16_LIST, return false);
    if (weight != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weight, DTYPE_SUPPORT_BF16_LIST, return false);
    }
    return true;
}

static bool Check910BOtherDtypeValid(
    const aclTensor* sumDy, const aclTensor* sumDyXmu, const aclTensor* counter, const aclTensor* gradInput)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(sumDy, DTYPE_SUPPORT_BF16_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sumDyXmu, DTYPE_SUPPORT_BF16_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(counter, COUNTER_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, DTYPE_SUPPORT_BF16_LIST, return false);
    return true;
}

static bool CheckFormat(const aclTensor* gradOut, const aclTensor* input, const aclTensor* gradInput)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(gradOut->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of gradOut only support ND, NCHW, NCL, NCDHW.");
        return false;
    }
    if (op::IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of input only support ND, NCHW, NCL, NCDHW.");
        return false;
    }
    if (op::IsPrivateFormat(gradInput->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of gradInput only support ND, NCHW, NCL, NCDHW.");
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, const aclTensor* sumDy, const aclTensor* sumDyXmu, const aclTensor* counter,
    const aclTensor* gradInput)
{
    OP_CHECK_MIN_DIM(input, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(input, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gradOut, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gradInput, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(counter, MAX_SUPPORT_DIMS_NUMS, return false);
    if (input->GetViewShape()[1] == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input channel size should not be zero.");
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(gradOut, input, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradInput, input, return false);

    op::Shape expectShape = {input->GetViewShape()[1]};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(mean, expectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(invstd, expectShape, return false);
    if (weight != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, expectShape, return false);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(sumDy, expectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(sumDyXmu, expectShape, return false);
    return true;
}

static const aclTensor* Broadcast(const aclTensor* input, const aclTensor* value, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(input->GetViewShape(), value->GetViewShape(), broadcastShape) ||
        input->GetViewShape() != broadcastShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of input and value can't broadcast.");
        return nullptr;
    }
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
    auto broadcastDstTensor = executor->AllocTensor(broadcastShape, value->GetDataType());
    auto shape =
        executor->ConvertToTensor(broadcastDims.data(), broadcastDims.size(), static_cast<op::DataType>(ACL_INT64));
    return l0op::BroadcastTo(value, broadcastDstTensor, shape, executor);
}

static const aclTensor* Reshape(
    const aclTensor* input, const aclTensor* value, size_t /* dimC */, size_t dimNum, aclOpExecutor* executor)
{
    // 将参数当做C维，默认C维在第二维，在前面补一个N维度
    const int64_t appendDim[] = {0};
    aclIntArray* shape = executor->AllocIntArray(appendDim, 1);
    auto valueUnSqueeze = l0op::UnsqueezeNd(value, shape, executor);
    if (valueUnSqueeze == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsqueeze 0 error");
        return nullptr;
    }
    if (dimNum > DIM_TWO) {
        FVector<int64_t> dimVector;
        for (size_t i = 2; i < dimNum; i++) {
            // 在C维后补维，至和输入维度一致
            dimVector.push_back(i);
        }
        aclIntArray* newShape = executor->AllocIntArray(dimVector.data(), dimNum - DIM_TWO);
        valueUnSqueeze = l0op::UnsqueezeNd(valueUnSqueeze, newShape, executor);
        if (valueUnSqueeze == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsqueeze end error");
            return nullptr;
        }
    }
    return Broadcast(input, valueUnSqueeze, executor);
}

static op::DataType GetPromoteType(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, const aclTensor* sumDy, const aclTensor* sumDyXmu)
{
    if (gradOut->GetDataType() == DataType::DT_FLOAT16 && input->GetDataType() == DataType::DT_FLOAT16 &&
        mean->GetDataType() == DataType::DT_FLOAT16 && invstd->GetDataType() == DataType::DT_FLOAT16 &&
        weight->GetDataType() == DataType::DT_FLOAT16 && sumDy->GetDataType() == DataType::DT_FLOAT16 &&
        sumDyXmu->GetDataType() == DataType::DT_FLOAT16) {
        return DataType::DT_FLOAT16;
    }
    return DataType::DT_FLOAT;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, const aclTensor* sumDy, const aclTensor* sumDyXmu, aclTensor* counter,
    aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(
        CheckNotNull(gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, counter, gradInput),
        ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 || socVersion == SocVersion::ASCEND910_95) {
        CHECK_COND(
            Check910BDtypeValid(gradOut, input, mean, invstd, weight), ACLNN_ERR_PARAM_INVALID,
            "CheckDtypeValid failed!");
        CHECK_COND(
            Check910BOtherDtypeValid(sumDy, sumDyXmu, counter, gradInput), ACLNN_ERR_PARAM_INVALID,
            "CheckDtypeValid failed!");
    } else {
        CHECK_COND(
            CheckDtypeValid(gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, counter, gradInput),
            ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");
    }

    // 3. 检查数据格式是否支持
    CHECK_COND(CheckFormat(gradOut, input, gradInput), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed!");

    // 4. 检查Shape是否支持
    CHECK_COND(
        CheckShape(gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, counter, gradInput), ACLNN_ERR_PARAM_INVALID,
        "CheckShape failed!");

    return ACLNN_SUCCESS;
}

static op::DataType GetDivPromoteType(const aclTensor* self, const aclTensor* other)
{
    // RealDiv算子需要对self和other两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());

    // 下沉PTA入口操作将入参类型转化成FLOAT进行后续处理
    return (promoteType != op::DataType::DT_FLOAT) && (promoteType != op::DataType::DT_FLOAT16) ?
               op::DataType::DT_FLOAT :
               promoteType;
}

static aclnnStatus MeanByCounter(
    const aclTensor* sumDy, const aclTensor* sumDyXmu, const aclTensor* counter, const aclTensor** meanDy,
    const aclTensor** meanDyXmu, aclOpExecutor* executor)
{
    // 调用l0算子ReduceSumOp进行求和
    op::Shape shape = counter->GetViewShape();
    size_t dimDum = shape.GetDimNum();
    int64_t appendDim[dimDum];
    for (uint64_t i = 0; i < dimDum; i++) {
        appendDim[i] = i;
    }
    auto dim = executor->AllocIntArray(appendDim, dimDum);
    auto counterCasted = l0op::Cast(counter, op::DataType::DT_FLOAT, executor);
    auto reduceSumOut = l0op::ReduceSumOp(counterCasted, dim, false, executor);
    CHECK_RET(reduceSumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto promoteType = GetDivPromoteType(sumDy, reduceSumOut);

    auto sumDyCasted = l0op::Cast(sumDy, promoteType, executor);
    CHECK_RET(sumDyCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sumDyXmuCasted = l0op::Cast(sumDyXmu, promoteType, executor);
    CHECK_RET(sumDyXmuCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto reduceSumOutCasted = l0op::Cast(reduceSumOut, promoteType, executor);
    CHECK_RET(reduceSumOutCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子RealDiv进行求均值
    auto meanDyResult = l0op::RealDiv(sumDyCasted, reduceSumOutCasted, executor);
    CHECK_RET(meanDyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *meanDy = meanDyResult;

    // 调用l0算子RealDiv进行求均值
    auto meanDyXmuResult = l0op::RealDiv(sumDyXmuCasted, reduceSumOutCasted, executor);
    CHECK_RET(meanDyXmuResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *meanDyXmu = meanDyXmuResult;

    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus aclnnBatchNormElemtBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* invstd,
    const aclTensor* weight, const aclTensor* sumDy, const aclTensor* sumDyXmu, aclTensor* counter,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnBatchNormElemtBackward, DFX_IN(gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, counter),
        DFX_OUT(gradInput));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, counter, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input->IsEmpty() || gradOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 可选入参weight处理
    if (weight == nullptr) {
        weight = FillScalar(input->GetViewShape()[1], 1, uniqueExecutor.get());
        CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    size_t dimC = input->GetViewShape()[1];

    size_t dimNum = input->GetViewShape().GetDimNum();

    // 对所有入参执行非连续转连续
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanContiguous = l0op::Contiguous(mean, uniqueExecutor.get());
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto invstdContiguous = l0op::Contiguous(invstd, uniqueExecutor.get());
    CHECK_RET(invstdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sumDyContiguous = l0op::Contiguous(sumDy, uniqueExecutor.get());
    CHECK_RET(sumDyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sumDyXmuContiguous = l0op::Contiguous(sumDyXmu, uniqueExecutor.get());
    CHECK_RET(sumDyXmuContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto counterContiguous = l0op::Contiguous(counter, uniqueExecutor.get());
    CHECK_RET(counterContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 对入参执行cast
    auto promoteType = GetPromoteType(
        gradOutContiguous, inputContiguous, meanContiguous, invstdContiguous, weightContiguous, sumDyContiguous,
        sumDyXmuContiguous);
    auto gradOutCast = l0op::Cast(gradOutContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(gradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputCast = l0op::Cast(inputContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(inputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanCast = l0op::Cast(meanContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(meanCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto invstdCast = l0op::Cast(invstdContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(invstdCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightCast = l0op::Cast(weightContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(weightCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sumDyCast = l0op::Cast(sumDyContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(sumDyCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sumDyXmuCast = l0op::Cast(sumDyXmuContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(sumDyXmuCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 对sumDy和sumDyXmu求均值
    const aclTensor* meanDy;
    const aclTensor* meanDyXmu;
    auto meanResult =
        MeanByCounter(sumDyCast, sumDyXmuCast, counterContiguous, &meanDy, &meanDyXmu, uniqueExecutor.get());
    CHECK_RET(meanResult == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // 对参数执行升维和广播
    auto meanReshape = Reshape(inputCast, meanCast, dimC, dimNum, uniqueExecutor.get());
    CHECK_RET(meanReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto invstdReshape = Reshape(inputCast, invstdCast, dimC, dimNum, uniqueExecutor.get());
    CHECK_RET(invstdReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightReshape = Reshape(inputCast, weightCast, dimC, dimNum, uniqueExecutor.get());
    CHECK_RET(weightReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanDyReshape = Reshape(inputCast, meanDy, dimC, dimNum, uniqueExecutor.get());
    CHECK_RET(meanDyReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanDyXmuReshape = Reshape(inputCast, meanDyXmu, dimC, dimNum, uniqueExecutor.get());
    CHECK_RET(meanDyXmuReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanDyCast = l0op::Cast(meanDyReshape, promoteType, uniqueExecutor.get());
    CHECK_RET(meanDyCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanDyXmuCast = l0op::Cast(meanDyXmuReshape, promoteType, uniqueExecutor.get());
    CHECK_RET(meanDyXmuCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto output = l0op::SyncBatchNormBackwardElemt(
        gradOutCast, inputCast, meanReshape, invstdReshape, weightReshape, meanDyCast, meanDyXmuCast,
        uniqueExecutor.get());

    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outputCast = l0op::Cast(output, gradInput->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(outputCast, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormElemtBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchNormElemtBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif