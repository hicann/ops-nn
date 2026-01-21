/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_batch_norm_reduce.h"
#include "norm/norm_common/op_host/op_api/norm_tensor_util.h"
#include "../../../batch_norm_v3/op_host/op_api/batch_norm.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const int64_t IN_SIZE = 4;
static constexpr size_t OUT_SIZE = 1;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr size_t TENSOR_NUM = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_IN = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_OUT = {op::DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor* x, const aclTensor* sum, const aclTensor* squareSum)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(sum, return false);
    OP_CHECK_NULL(squareSum, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* sum, const aclTensor* squareSum)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST_IN, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sum, DTYPE_SUPPORT_LIST_OUT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(squareSum, DTYPE_SUPPORT_LIST_OUT, return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* sum, const aclTensor* squareSum)
{
    auto xFormat = x->GetStorageFormat();
    auto sumFormat = sum->GetStorageFormat();
    auto squareSumFormat = squareSum->GetStorageFormat();
    if (xFormat != op::Format::FORMAT_NCHW) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of x only supports [NCHW], but format is [%s].",
            op::ToString(xFormat).GetString());
        return false;
    }

    if (sumFormat != op::Format::FORMAT_ND) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of sum only supports [ND], but format is [%s].",
            op::ToString(sumFormat).GetString());
        return false;
    }
    if (squareSumFormat != op::Format::FORMAT_ND) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of squareSum only supports [ND], but format is [%s].",
            op::ToString(squareSumFormat).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* sum, const aclTensor* squareSum)
{
    auto xShape = x->GetViewShape();
    auto sumShape = sum->GetViewShape();
    auto squareSumShape = squareSum->GetViewShape();

    OP_CHECK_WRONG_DIMENSION(x, IN_SIZE, return false);
    OP_CHECK(
        sumShape.GetDimNum() == OUT_SIZE,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "It is expected dimNum of sum equals to 1, but size is %zu", sumShape.GetDimNum()),
        return false);
    OP_CHECK(
        squareSumShape.GetDimNum() == OUT_SIZE,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "It is expected dimNum of squareSum equals to 1, but size is %zu",
            sumShape.GetDimNum()),
        return false);

    int64_t inputC = xShape.GetDim(DIM_ONE);
    if (x->GetStorageFormat() == op::Format::FORMAT_NHWC) {
        inputC = xShape.GetDim(DIM_THREE);
    }
    OP_CHECK(
        sumShape.GetDim(DIM_ZERO) == inputC,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "It is expected size of sum equals to input_C(%ld), but size is %zu", inputC,
            sumShape.GetDim(DIM_ZERO)),
        return false);
    OP_CHECK(
        squareSumShape.GetDim(DIM_ZERO) == inputC,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "It is expected size of squareSum equals to input_C(%ld), but size is %zu", inputC,
            squareSumShape.GetDim(DIM_ZERO)),
        return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, aclTensor* sum, aclTensor* squareSum)
{
    // 1. 检查输入/输出的数据类型是否合法
    CHECK_RET(CheckDtypeValid(x, sum, squareSum), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入/输出的shape大小
    CHECK_RET(CheckFormat(x, sum, squareSum), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入/输出的shape大小
    CHECK_RET(CheckShape(x, sum, squareSum), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor* ResizeTo4D(const aclTensor* input, aclOpExecutor* executor)
{
    const int64_t appendDim[] = {0, 2, 3};
    aclIntArray* newShape = executor->AllocIntArray(appendDim, sizeof(appendDim) / sizeof(int64_t));

    auto inputUnsqueeze = l0op::UnsqueezeNd(input, newShape, executor);
    if (inputUnsqueeze == nullptr) {
        return inputUnsqueeze;
    }
    auto formatTensor =
        executor == nullptr ?
            const_cast<aclTensor*>(inputUnsqueeze) :
            executor->CreateView(inputUnsqueeze, inputUnsqueeze->GetViewShape(), inputUnsqueeze->GetViewOffset());
    formatTensor->SetViewFormat(Format::FORMAT_NCHW);
    formatTensor->SetOriginalFormat(Format::FORMAT_NCHW);
    formatTensor->SetStorageFormat(Format::FORMAT_NCHW);
    return formatTensor;
}

const aclTensor* ResizeTo1D(const aclTensor* input, aclOpExecutor* executor)
{
    const int64_t removeDim[] = {0, 2, 3};
    aclIntArray* newShape = executor->AllocIntArray(removeDim, sizeof(removeDim) / sizeof(int64_t));

    auto inputSqueeze = l0op::SqueezeNd(input, newShape, executor);
    if (inputSqueeze == nullptr) {
        return inputSqueeze;
    }
    auto formatTensor =
        executor == nullptr ?
            const_cast<aclTensor*>(inputSqueeze) :
            executor->CreateView(inputSqueeze, inputSqueeze->GetViewShape(), inputSqueeze->GetViewOffset());
    formatTensor->SetViewFormat(Format::FORMAT_ND);
    formatTensor->SetOriginalFormat(Format::FORMAT_ND);
    formatTensor->SetStorageFormat(Format::FORMAT_ND);
    return formatTensor;
}
} // namespace

aclnnStatus aclnnBatchNormReduceGetWorkspaceSize(
    const aclTensor* x, aclTensor* sum, aclTensor* squareSum, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBatchNormReduce, DFX_IN(x), DFX_OUT(sum, squareSum));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 检查必选输入/输出是否为空指针
    CHECK_RET(CheckNotNull(x, sum, squareSum), ACLNN_ERR_PARAM_NULLPTR);

    if (x->IsEmpty() || sum->IsEmpty() || squareSum->IsEmpty()) {
        auto ret = op::ProcessEmptyTensorWithValue(sum, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = op::ProcessEmptyTensorWithValue(squareSum, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto ret = CheckParams(x, sum, squareSum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sumNCHW = ResizeTo4D(sum, uniqueExecutor.get());
    CHECK_RET(sumNCHW != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::array<aclTensor*, TENSOR_NUM> sumTensor =
        l0op::BNTrainingReduce(xContiguous, sumNCHW->GetViewShape(), uniqueExecutor.get());

    auto sumND = ResizeTo1D(sumTensor[0], uniqueExecutor.get());
    CHECK_RET(sumND != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto squareSumND = ResizeTo1D(sumTensor[1], uniqueExecutor.get());
    CHECK_RET(squareSumND != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopySum = l0op::ViewCopy(sumND, sum, uniqueExecutor.get());
    CHECK_RET(viewCopySum != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopySquareSum = l0op::ViewCopy(squareSumND, squareSum, uniqueExecutor.get());
    CHECK_RET(viewCopySquareSum != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormReduce(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchNormReduce);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif