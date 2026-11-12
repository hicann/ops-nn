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
 * \file aclnn_thnn_fused_lstm_cell_backward.cpp
 * \brief
 */
#include "aclnn_thnn_fused_lstm_cell_backward.h"
#include "thnn_fused_lstm_cell_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "level0/zero_op.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t DIM_TWO = 2;
constexpr int64_t GATE_COUNT = 4; // i, f, j, o 四个门
constexpr int64_t BATCH_DIM = 0;
constexpr int64_t HIDDEN_DIM = 1;
constexpr int64_t OUT_DC_PREV_INDEX = 1;
constexpr int64_t OUT_DGATES_INDEX = 0;
constexpr int64_t OUT_DB_INDEX = 2;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_FLOAT16};
} // namespace

// gradHy/gradC均为可选输入，取首个非空张量作为shape/dtype基准
static const aclTensor* GetBaseTensor(const aclTensor* gradHyOptional, const aclTensor* gradCOptional,
                                      const aclTensor* cx)
{
    if (gradHyOptional != nullptr) {
        return gradHyOptional;
    }
    if (gradCOptional != nullptr) {
        return gradCOptional;
    }
    return cx;
}

// 可选梯度缺省时按零梯度参与计算（与上游Tensor?语义一致）
// 入参shape/rank已由CheckShapeValid校验为[batch, hidden_size]且秩为2，直接沿用其viewShape，不做维度索引
static const aclTensor* MakeZeroGradTensor(const aclTensor* refTensor, aclOpExecutor* executor)
{
    auto gradTmp = executor->AllocTensor(refTensor->GetViewShape(), refTensor->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(gradTmp != nullptr, nullptr);
    auto zeroGrad = l0op::ZerosLike(gradTmp, executor);
    CHECK_RET(zeroGrad != nullptr, nullptr);
    return zeroGrad;
}

// 将出参整体置零（两个可选梯度均缺省时结果恒为零，无需执行反传计算）
static bool ZeroOutTensor(aclTensor* out, aclOpExecutor* executor)
{
    auto zeroTmp = executor->AllocTensor(out->GetViewShape(), out->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(zeroTmp != nullptr, false);
    auto zeroResult = l0op::ZerosLike(zeroTmp, executor);
    CHECK_RET(zeroResult != nullptr, false);
    auto copyResult = l0op::ViewCopy(zeroResult, out, executor);
    CHECK_RET(copyResult != nullptr, false);
    return true;
}

// gradHyOptional/gradCOptional为可选输入，允许传空指针，不参与必选参数判空
static bool CheckNotNull(const aclTensor* cx, const aclTensor* cy, const aclTensor* storage, const aclTensor* gradGates,
                         const aclTensor* gradCx, const aclTensor* gradBias)
{
    OP_CHECK_NULL(cx, return false);
    OP_CHECK_NULL(cy, return false);
    OP_CHECK_NULL(storage, return false);
    OP_CHECK_NULL(gradGates, return false);
    OP_CHECK_NULL(gradCx, return false);
    OP_CHECK_NULL(gradBias, return false);
    return true;
}

static bool CheckFormatValid(const aclTensor* gradHyOptional, const aclTensor* gradCOptional, const aclTensor* cx,
                             const aclTensor* cy, const aclTensor* storage, const aclTensor* gradGates,
                             const aclTensor* gradCx, const aclTensor* gradBias)
{
    if (gradHyOptional != nullptr && gradHyOptional->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradHyOptional format only support ND");
        return false;
    }
    if (gradCOptional != nullptr && gradCOptional->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradCOptional format only support ND");
        return false;
    }
    if (cx->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cx format only support ND");
        return false;
    }
    if (cy->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cy format only support ND");
        return false;
    }
    if (storage->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "storage format only support ND");
        return false;
    }
    if (gradGates->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradGates format only support ND");
        return false;
    }
    if (gradBias->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradBias format only support ND");
        return false;
    }
    if (gradCx->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradCx format only support ND");
        return false;
    }
    return true;
}

// 检查单个张量的数据类型支持和一致性
static bool CheckSingleTensorDtype(const aclTensor* tensor, const char* tensorName, ge::DataType baseDtype)
{
    // 检查是否在支持的数据类型列表中
    OP_CHECK_DTYPE_NOT_SUPPORT(tensor, DTYPE_SUPPORT_LIST, return false);
    // 检查数据类型一致性
    if (tensor->GetDataType() != baseDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor dtype inconsistent, expected: %s, actual: %s.", tensorName,
                op::ToString(baseDtype).GetString(), op::ToString(tensor->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckDtypeValid(const aclTensor* gradHyOptional, const aclTensor* gradCOptional, const aclTensor* cx,
                            const aclTensor* cy, const aclTensor* storage, const aclTensor* gradGates,
                            const aclTensor* gradCx, const aclTensor* gradBias)
{
    ge::DataType baseDtype = GetBaseTensor(gradHyOptional, gradCOptional, cx)->GetDataType();
    // 检查所有单个张量，可选输入为空时跳过
    if (!CheckSingleTensorDtype(cx, "cx", baseDtype))
        return false;
    if (!CheckSingleTensorDtype(cy, "cy", baseDtype))
        return false;
    if (!CheckSingleTensorDtype(storage, "storage", baseDtype))
        return false;
    if (!CheckSingleTensorDtype(gradGates, "gradGates", baseDtype))
        return false;
    if (!CheckSingleTensorDtype(gradBias, "gradBias", baseDtype))
        return false;
    if (!CheckSingleTensorDtype(gradCx, "gradCx", baseDtype))
        return false;
    if (gradHyOptional != nullptr && !CheckSingleTensorDtype(gradHyOptional, "gradHyOptional", baseDtype))
        return false;
    if (gradCOptional != nullptr && !CheckSingleTensorDtype(gradCOptional, "gradCOptional", baseDtype))
        return false;
    return true;
}

static bool ValidateInputShape(const aclTensor* input, const std::vector<int64_t>& expected_dims,
                               const char* tensorName)
{
    auto shape = input->GetViewShape();
    if (shape.GetDimNum() != expected_dims.size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor %s has wrong dimension count.", tensorName);
        return false;
    }

    for (size_t i = 0; i < expected_dims.size(); ++i) {
        if (expected_dims[i] != shape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor %s dim %zu mismatch.", tensorName, i);
            return false;
        }
    }
    return true;
};

static bool CheckShapeValid(const aclTensor* gradHyOptional, const aclTensor* gradCOptional, const aclTensor* cx,
                            const aclTensor* cy, const aclTensor* storage, bool hasBias, const aclTensor* gradGates,
                            const aclTensor* gradCx, const aclTensor* gradBias)
{
    // 以首个非空可选梯度(缺省时取cx)作为batch/hidden基准
    const aclTensor* baseTensor = GetBaseTensor(gradHyOptional, gradCOptional, cx);
    OP_CHECK_WRONG_DIMENSION(baseTensor, DIM_TWO, return false);
    auto baseShape = baseTensor->GetViewShape();
    int64_t hiddenSize = baseShape[HIDDEN_DIM];
    int64_t batch = baseShape[BATCH_DIM];
    CHECK_RET(hiddenSize > 0 && batch > 0, false);
    std::vector<int64_t> biasDim = {GATE_COUNT * hiddenSize};
    std::vector<int64_t> storageDim = {batch, GATE_COUNT * hiddenSize};
    std::vector<int64_t> hiddenDim = {batch, hiddenSize};

    bool ret = ValidateInputShape(cx, hiddenDim, "cx") && ValidateInputShape(cy, hiddenDim, "cy") &&
               ValidateInputShape(storage, storageDim, "storage") &&
               ValidateInputShape(gradGates, storageDim, "gradGates") &&
               ValidateInputShape(gradCx, hiddenDim, "gradCx");
    if (gradHyOptional != nullptr) {
        ret = ValidateInputShape(gradHyOptional, hiddenDim, "gradHyOptional") && ret;
    }
    if (gradCOptional != nullptr) {
        ret = ValidateInputShape(gradCOptional, hiddenDim, "gradCOptional") && ret;
    }
    ret = hasBias ? ValidateInputShape(gradBias, biasDim, "gradBias") && ret : ret;
    return ret;
}

static aclnnStatus CheckParams(const aclTensor* gradHyOptional, const aclTensor* gradCOptional, const aclTensor* cx,
                               const aclTensor* cy, const aclTensor* storage, bool hasBias, const aclTensor* gradGates,
                               const aclTensor* gradCx, const aclTensor* gradBias)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(cx, cy, storage, gradGates, gradCx, gradBias), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradHyOptional, gradCOptional, cx, cy, storage, gradGates, gradCx, gradBias),
              ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShapeValid(gradHyOptional, gradCOptional, cx, cy, storage, hasBias, gradGates, gradCx, gradBias),
              ACLNN_ERR_PARAM_INVALID);

    // 4. 检查format是否满足约束
    CHECK_RET(CheckFormatValid(gradHyOptional, gradCOptional, cx, cy, storage, gradGates, gradCx, gradBias),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CheckInputsNotEmpty(const aclTensor* gradHyOptional, const aclTensor* gradCOptional, const aclTensor* cx,
                                const aclTensor* cy, const aclTensor* storage)
{
    if (cx->IsEmpty() || cy->IsEmpty() || storage->IsEmpty()) {
        return false;
    }
    // 可选梯度为空指针时无数据，不参与空tensor判定
    if ((gradHyOptional != nullptr && gradHyOptional->IsEmpty()) ||
        (gradCOptional != nullptr && gradCOptional->IsEmpty())) {
        return false;
    }
    return true;
}

aclnnStatus aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(const aclTensor* gradHyOptional,
                                                           const aclTensor* gradCOptional, const aclTensor* cx,
                                                           const aclTensor* cy, const aclTensor* storage, bool hasBias,
                                                           aclTensor* gradGatesOut, aclTensor* gradCxOut,
                                                           aclTensor* gradBiasOut, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnThnnFusedLstmCellBackward, DFX_IN(gradHyOptional, gradCOptional, cx, cy, storage, hasBias),
                   DFX_OUT(gradGatesOut, gradCxOut, gradBiasOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(gradHyOptional, gradCOptional, cx, cy, storage, hasBias, gradGatesOut, gradCxOut,
                           gradBiasOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (!CheckInputsNotEmpty(gradHyOptional, gradCOptional, cx, cy, storage)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 两个可选梯度均缺省：输出梯度恒为零，直接置零后返回，不执行反传计算
    if (gradHyOptional == nullptr && gradCOptional == nullptr) {
        CHECK_RET(ZeroOutTensor(gradGatesOut, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ZeroOutTensor(gradCxOut, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
        if (hasBias) {
            CHECK_RET(ZeroOutTensor(gradBiasOut, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
        }
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* gradHyOptionalContiguous = nullptr;
    if (gradHyOptional != nullptr) {
        gradHyOptionalContiguous = l0op::Contiguous(gradHyOptional, uniqueExecutor.get());
        CHECK_RET(gradHyOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* gradCOptionalContiguous = nullptr;
    if (gradCOptional != nullptr) {
        gradCOptionalContiguous = l0op::Contiguous(gradCOptional, uniqueExecutor.get());
        CHECK_RET(gradCOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto cxContiguous = l0op::Contiguous(cx, uniqueExecutor.get());
    CHECK_RET(cxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cyContiguous = l0op::Contiguous(cy, uniqueExecutor.get());
    CHECK_RET(cyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto storageContiguous = l0op::Contiguous(storage, uniqueExecutor.get());
    CHECK_RET(storageContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 可选梯度缺省时补零，底层算子dhy/dc为必选输入
    // 零梯度与cx同shape同dtype（CheckShapeValid/CheckDtypeValid已保证一致），直接以cx为基准
    if (gradHyOptionalContiguous == nullptr) {
        gradHyOptionalContiguous = MakeZeroGradTensor(cxContiguous, uniqueExecutor.get());
        CHECK_RET(gradHyOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (gradCOptionalContiguous == nullptr) {
        gradCOptionalContiguous = MakeZeroGradTensor(cxContiguous, uniqueExecutor.get());
        CHECK_RET(gradCOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto out = l0op::ThnnFusedLstmCellGrad(gradHyOptionalContiguous, gradCOptionalContiguous, cxContiguous,
                                           cyContiguous, storageContiguous, hasBias, uniqueExecutor.get());
    CHECK_RET(out[OUT_DGATES_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(out[OUT_DC_PREV_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradGatesCopyResult = l0op::ViewCopy(out[OUT_DGATES_INDEX], gradGatesOut, uniqueExecutor.get());
    CHECK_RET(gradGatesCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradCxCopyResult = l0op::ViewCopy(out[OUT_DC_PREV_INDEX], gradCxOut, uniqueExecutor.get());
    CHECK_RET(gradCxCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (hasBias) {
        CHECK_RET(out[OUT_DB_INDEX] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto gradBiasCopyResult = l0op::ViewCopy(out[OUT_DB_INDEX], gradBiasOut, uniqueExecutor.get());
        CHECK_RET(gradBiasCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThnnFusedLstmCellBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnThnnFusedLstmCellBackward);
    //  固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
