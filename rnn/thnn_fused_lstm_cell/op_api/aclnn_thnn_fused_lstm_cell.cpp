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
 * \file aclnn_thnn_fused_lstm_cell.cpp
 * \brief
 */
#include "aclnn_thnn_fused_lstm_cell.h"
#include "thnn_fused_lstm_cell.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "level0/zero_op.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct AclnnParams {
    const aclTensor    *inputGates;
    const aclTensor    *hiddenGates;
    const aclTensor    *cx;
    const aclTensor    *inputBiasOptional;
    const aclTensor    *hiddenBiasOptional;
    aclTensor          *hyOut;
    aclTensor          *cyOut;
    aclTensor          *storageOut;
} aclnnParams;

struct AclnnInfo {
    int64_t B;
    int64_t H;
    ge::DataType dtype;
    aclOpExecutor *executor;
} info;

struct BaseOpOutputs {
    aclTensor          *hy;
    aclTensor          *cy;
    aclTensor          *storage;
} baseOuts;

const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
const int64_t INDEX_2 = 2;
const int64_t INDEX_4 = 4;

}  // namespace

static aclnnStatus CheckParamsNullptr()
{
    OP_CHECK_NULL(aclnnParams.inputGates, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aclnnParams.hiddenGates, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aclnnParams.cx, return ACLNN_ERR_PARAM_NULLPTR);
    if (aclnnParams.inputBiasOptional) {
        OP_CHECK(
            aclnnParams.hiddenBiasOptional != nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hiddenBias should not be a nullptr when inputBias is set."),
            return ACLNN_ERR_PARAM_NULLPTR
        );
    } else {
        OP_CHECK(
            aclnnParams.hiddenBiasOptional == nullptr,
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "inputBias should not be a nullptr when hiddenBias is set."),
            return ACLNN_ERR_PARAM_NULLPTR
        );
    }

    OP_CHECK_NULL(aclnnParams.hyOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aclnnParams.cyOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aclnnParams.storageOut, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDimsAndListLength()
{
    OP_CHECK_WRONG_DIMENSION(aclnnParams.inputGates, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(aclnnParams.hiddenGates, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(aclnnParams.cx, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    if (aclnnParams.inputBiasOptional) {
        OP_CHECK_WRONG_DIMENSION(aclnnParams.inputBiasOptional, 1, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_WRONG_DIMENSION(aclnnParams.hiddenBiasOptional, 1, return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_WRONG_DIMENSION(aclnnParams.hyOut, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(aclnnParams.cyOut, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(aclnnParams.storageOut, INDEX_2, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShapes()
{
    auto cxShape = aclnnParams.cx->GetViewShape();
    info.B = cxShape.GetDim(0);
    info.H = cxShape.GetDim(1);
    op::Shape gatesShape = {info.B, info.H * INDEX_4};
    op::Shape commonShape = {info.B, info.H};
    op::Shape biasShape = {info.H * INDEX_4};

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.inputGates, gatesShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.hiddenGates, gatesShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.cx, commonShape, return ACLNN_ERR_PARAM_INVALID);
    if (aclnnParams.inputBiasOptional) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.inputBiasOptional, biasShape, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.hiddenBiasOptional, biasShape, return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.hyOut, commonShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.cyOut, commonShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.storageOut, gatesShape, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypes()
{
    OP_CHECK_DTYPE_NOT_SUPPORT(aclnnParams.inputGates, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    info.dtype = aclnnParams.inputGates->GetDataType();
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.hiddenGates, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.cx, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    if (aclnnParams.inputBiasOptional) {
        OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.inputBiasOptional, info.dtype, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.hiddenBiasOptional, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.hyOut, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.cyOut, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.storageOut, info.dtype, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsValid()
{
    aclnnStatus ret;

    // 空指针校验
    ret = CheckParamsNullptr();
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ret, "CheckParamsNullptr failed, certain incoming nullptrs may be invalid."),
        return ret
    );

    if (aclnnParams.inputGates->IsEmpty()) return ACLNN_SUCCESS;  // 跳至空Tensor处理流程

    // tensor dim校验
    ret = CheckDimsAndListLength();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // shape关系校验
    ret = CheckShapes();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // dtype校验
    ret = CheckDtypes();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

static void MakeParamsContig()
{
    aclnnParams.inputGates = l0op::Contiguous(aclnnParams.inputGates, info.executor);
    aclnnParams.hiddenGates = l0op::Contiguous(aclnnParams.hiddenGates, info.executor);
    aclnnParams.cx = l0op::Contiguous(aclnnParams.cx, info.executor);
    if (aclnnParams.inputBiasOptional) {
        aclnnParams.inputBiasOptional = l0op::Contiguous(aclnnParams.inputBiasOptional, info.executor);
        aclnnParams.hiddenBiasOptional = l0op::Contiguous(aclnnParams.hiddenBiasOptional, info.executor);
    }
}

static aclnnStatus HandleParams()
{
    if (!aclnnParams.inputBiasOptional) {
        op::Shape biasShape = {info.H * INDEX_4};
        auto biasTmp = info.executor->AllocTensor(biasShape, info.dtype, op::Format::FORMAT_ND);
        CHECK_RET(biasTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclnnParams.inputBiasOptional = l0op::ZerosLike(biasTmp, info.executor);
        aclnnParams.hiddenBiasOptional = l0op::ZerosLike(biasTmp, info.executor);
        CHECK_RET(aclnnParams.inputBiasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(aclnnParams.hiddenBiasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus RunBaseOp()
{
    op::Shape gatesShape = {info.B, info.H * INDEX_4};
    op::Shape commonShape = {info.B, info.H};
    baseOuts.hy = info.executor->AllocTensor(commonShape, info.dtype, op::Format::FORMAT_ND);
    baseOuts.cy = info.executor->AllocTensor(commonShape, info.dtype, op::Format::FORMAT_ND);
    baseOuts.storage = info.executor->AllocTensor(gatesShape, info.dtype, op::Format::FORMAT_ND);
    auto ret = l0op::ThnnFusedLstmCell(
        aclnnParams.inputGates, aclnnParams.hiddenGates, aclnnParams.cx, aclnnParams.inputBiasOptional, aclnnParams.hiddenBiasOptional,
        baseOuts.hy, baseOuts.cy, baseOuts.storage, 
        info.executor
    );
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ret, "Run l0op::ThnnFusedLstmCell failed."),
        return ret
    );
    return ACLNN_SUCCESS;
}

static aclnnStatus ViewCopyResults()
{
    const aclTensor *res = nullptr;
    res = l0op::ViewCopy(baseOuts.hy, aclnnParams.hyOut, info.executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
    res = l0op::ViewCopy(baseOuts.cy, aclnnParams.cyOut, info.executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
    res = l0op::ViewCopy(baseOuts.storage, aclnnParams.storageOut, info.executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThnnFusedLstmCellGetWorkspaceSize(
    const aclTensor    *inputGates, 
    const aclTensor    *hiddenGates, 
    const aclTensor    *cx, 
    const aclTensor    *inputBiasOptional, 
    const aclTensor    *hiddenBiasOptional, 
    aclTensor          *hyOut, 
    aclTensor          *cyOut, 
    aclTensor          *storageOut,
    uint64_t           *workspaceSize, 
    aclOpExecutor      **executor)
{
    L2_DFX_PHASE_1(
        aclnnThnnFusedLstmCell,
        DFX_IN(inputGates, hiddenGates, cx, inputBiasOptional, hiddenBiasOptional),
        DFX_OUT(hyOut, cyOut, storageOut)
    );

    aclnnParams = AclnnParams{
        inputGates, hiddenGates, cx, inputBiasOptional, hiddenBiasOptional,
        hyOut, cyOut, storageOut
    };

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    info.executor = uniqueExecutor.get();

    aclnnStatus ret;

    // 参数校验，顺便填充info
    ret = CheckParamsValid();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 预处理
    // 空tensor处理
    if (aclnnParams.inputGates->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 转连续
    MakeParamsContig();

    // bias处理
    ret = HandleParams();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 运行底层算子
    ret = RunBaseOp();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // ViewCopy输出
    ret = ViewCopyResults();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThnnFusedLstmCell(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnThnnFusedLstmCell);
    //  固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif