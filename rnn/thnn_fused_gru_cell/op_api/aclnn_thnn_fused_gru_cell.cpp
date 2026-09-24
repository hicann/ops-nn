/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_thnn_fused_gru_cell.cpp
 * \brief
 */

#include "aclnn_thnn_fused_gru_cell.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_util.h"

#include "thnn_fused_gru_cell.h"

using namespace op;
using namespace l0op;

// dtype 白名单（与 IR 注册、tiling 校验链一致）
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_FLOAT};

// 必选输入/输出非空校验（executor 已由 OP_CHECK_COMM_INPUT 前置校验）；
// 可选 bias 空指针 = 缺省，合法，不检查
static bool CheckNotNull(const aclTensor* inputGates, const aclTensor* hiddenGates, const aclTensor* hx,
                         const aclTensor* hyOut, const aclTensor* storageOut)
{
    OP_CHECK_NULL(inputGates, return false);
    OP_CHECK_NULL(hiddenGates, return false);
    OP_CHECK_NULL(hx, return false);
    OP_CHECK_NULL(hyOut, return false);
    OP_CHECK_NULL(storageOut, return false);
    return true;
}

// 全部在位张量（含输出）与派发锚点 inputGates 同 dtype
static bool CheckDtypeValid(const aclTensor* inputGates, const aclTensor* hiddenGates, const aclTensor* hx,
                            const aclTensor* inputBiasOptional, const aclTensor* hiddenBiasOptional,
                            const aclTensor* hyOut, const aclTensor* storageOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputGates, DTYPE_SUPPORT_LIST, return false);
    const auto anchor = inputGates->GetDataType();
    OP_CHECK_DTYPE_NOT_MATCH(hiddenGates, anchor, return false);
    OP_CHECK_DTYPE_NOT_MATCH(hx, anchor, return false);
    if (inputBiasOptional != nullptr) { // 可选 bias 缺省
        OP_CHECK_DTYPE_NOT_MATCH(inputBiasOptional, anchor, return false);
    }
    if (hiddenBiasOptional != nullptr) { // 可选 bias 缺省
        OP_CHECK_DTYPE_NOT_MATCH(hiddenBiasOptional, anchor, return false);
    }
    OP_CHECK_DTYPE_NOT_MATCH(hyOut, anchor, return false);
    OP_CHECK_DTYPE_NOT_MATCH(storageOut, anchor, return false);
    return true;
}

// shape 契约：以 hx 为锚点，inputGates == (B, 3H)、hx == hyOut == (B, H)、storageOut == (B, 5H)，
// 在位 bias 为 rank 1 且 numel == 3H；空 Tensor（B == 0 或 H == 0）为合法 shape，由调用方短路处理。
// 注：storage format 不校验——ND 语义即支持任意 format，非连续布局由 Contiguous/ViewCopy 归一
static bool CheckShape(const aclTensor* inputGates, const aclTensor* hiddenGates, const aclTensor* hx,
                       const aclTensor* inputBiasOptional, const aclTensor* hiddenBiasOptional, const aclTensor* hyOut,
                       const aclTensor* storageOut)
{
    OP_CHECK_WRONG_DIMENSION(inputGates, 2, return false);
    OP_CHECK_WRONG_DIMENSION(hiddenGates, 2, return false);
    OP_CHECK_WRONG_DIMENSION(hx, 2, return false);
    OP_CHECK_WRONG_DIMENSION(hyOut, 2, return false);
    OP_CHECK_WRONG_DIMENSION(storageOut, 2, return false);
    if (hiddenGates->GetViewShape() != inputGates->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of hiddenGates %s must be equal to inputGates %s.",
                op::ToString(hiddenGates->GetViewShape()).GetString(),
                op::ToString(inputGates->GetViewShape()).GetString());
        return false;
    }
    const int64_t batch = hx->GetViewShape().GetDim(0);
    const int64_t hidden = hx->GetViewShape().GetDim(1);
    OP_CHECK(inputGates->GetViewShape().GetDim(0) == batch &&
                 inputGates->GetViewShape().GetDim(1) == GATES_PER_INPUT * hidden,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of inputGates %s must be (B, 3H) = (%ld, %ld) anchored on hx.",
                     op::ToString(inputGates->GetViewShape()).GetString(), static_cast<long>(batch),
                     static_cast<long>(GATES_PER_INPUT * hidden)),
             return false);
    // 输出 shape 由共享 helper 推导（与 L0 AllocTensor 单一事实源）
    op::Shape hyShape;
    op::Shape storageShape;
    OP_CHECK(ThnnFusedGruCellOutShape(hx, hyShape, storageShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Derive hyOut/storageOut shape from hx failed."), return false);
    if (hyOut->GetViewShape() != hyShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of hyOut %s must be (B, H) = (%ld, %ld), same as hx.",
                op::ToString(hyOut->GetViewShape()).GetString(), static_cast<long>(batch), static_cast<long>(hidden));
        return false;
    }
    if (storageOut->GetViewShape() != storageShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of storageOut %s must be (B, 5H) = (%ld, %ld).",
                op::ToString(storageOut->GetViewShape()).GetString(), static_cast<long>(batch),
                static_cast<long>(GATES_PER_STORAGE * hidden));
        return false;
    }
    // 在位 bias 为 rank 1 且 numel == 3H（可选 bias 缺省跳过）
    if (inputBiasOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(inputBiasOptional, 1, return false);
        OP_CHECK(inputBiasOptional->GetViewShape().GetShapeSize() == GATES_PER_INPUT * hidden,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of input_bias %s must be (3H) with numel == %ld.",
                         op::ToString(inputBiasOptional->GetViewShape()).GetString(),
                         static_cast<long>(GATES_PER_INPUT * hidden)),
                 return false);
    }
    if (hiddenBiasOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(hiddenBiasOptional, 1, return false);
        OP_CHECK(hiddenBiasOptional->GetViewShape().GetShapeSize() == GATES_PER_INPUT * hidden,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of hidden_bias %s must be (3H) with numel == %ld.",
                         op::ToString(hiddenBiasOptional->GetViewShape()).GetString(),
                         static_cast<long>(GATES_PER_INPUT * hidden)),
                 return false);
    }
    return true;
}

// 参数检查总入口：空指针 → dtype → shape
static aclnnStatus CheckParams(const aclTensor* inputGates, const aclTensor* hiddenGates, const aclTensor* hx,
                               const aclTensor* inputBiasOptional, const aclTensor* hiddenBiasOptional,
                               aclTensor* hyOut, aclTensor* storageOut)
{
    CHECK_RET(CheckNotNull(inputGates, hiddenGates, hx, hyOut, storageOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(inputGates, hiddenGates, hx, inputBiasOptional, hiddenBiasOptional, hyOut, storageOut),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(inputGates, hiddenGates, hx, inputBiasOptional, hiddenBiasOptional, hyOut, storageOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnThnnFusedGruCellGetWorkspaceSize(const aclTensor* inputGates, const aclTensor* hiddenGates,
                                                  const aclTensor* hx, const aclTensor* inputBiasOptional,
                                                  const aclTensor* hiddenBiasOptional, aclTensor* hyOut,
                                                  aclTensor* storageOut, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnThnnFusedGruCell, DFX_IN(inputGates, hiddenGates, hx, inputBiasOptional, hiddenBiasOptional),
                   DFX_OUT(hyOut, storageOut));

    // workspaceSize / executor 空指针前置校验（含日志）
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    // 芯片版本控制：kernel 为 RegBase 原生实现（AscendC::Reg / NDDMA），非 RegBase 平台直接拦截
    OP_CHECK(Ops::NN::AclnnUtil::IsRegbase(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ThnnFusedGruCell only supports RegBase platforms, e.g. Ascend950."),
             return ACLNN_ERR_PARAM_INVALID);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(inputGates, hiddenGates, hx, inputBiasOptional, hiddenBiasOptional, hyOut, storageOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理：B == 0 或 H == 0 时直接返回空输出，不下发kernel
    if (inputGates->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 转连续：可选 bias 为空指针时保持 nullptr（缺省 ≡ 全零）
    auto inputGatesContiguous = Contiguous(inputGates, uniqueExecutor.get());
    CHECK_RET(inputGatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto hiddenGatesContiguous = Contiguous(hiddenGates, uniqueExecutor.get());
    CHECK_RET(hiddenGatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto hxContiguous = Contiguous(hx, uniqueExecutor.get());
    CHECK_RET(hxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* inputBiasOptionalContiguous = nullptr;
    if (inputBiasOptional != nullptr) {
        inputBiasOptionalContiguous = Contiguous(inputBiasOptional, uniqueExecutor.get());
        CHECK_RET(inputBiasOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* hiddenBiasOptionalContiguous = nullptr;
    if (hiddenBiasOptional != nullptr) {
        hiddenBiasOptionalContiguous = Contiguous(hiddenBiasOptional, uniqueExecutor.get());
        CHECK_RET(hiddenBiasOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0算子ThnnFusedGruCell进行计算，L0 内部分配连续输出
    auto outputs = ThnnFusedGruCell(inputGatesContiguous, hiddenGatesContiguous, hxContiguous,
                                    inputBiasOptionalContiguous, hiddenBiasOptionalContiguous, uniqueExecutor.get());
    CHECK_RET(outputs[0] != nullptr && outputs[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 结果按用户布局拷回，hyOut / storageOut 支持非连续
    CHECK_RET(ViewCopy(outputs[0], hyOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopy(outputs[1], storageOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThnnFusedGruCell(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnThnnFusedGruCell);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
