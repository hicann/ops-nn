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
#include "opdev/platform.h"
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
    const aclTensor* inputGates;
    const aclTensor* hiddenGates;
    const aclTensor* cx;
    const aclTensor* inputBiasOptional;
    const aclTensor* hiddenBiasOptional;
    aclTensor* hyOut;
    aclTensor* cyOut;
    aclTensor* storageOut;
};

struct AclnnInfo {
    int64_t B;
    int64_t H;
    ge::DataType dtype;
    aclOpExecutor* executor;
};

struct BaseOpOutputs {
    aclTensor* hy;
    aclTensor* cy;
    aclTensor* storage;
};

static const std::initializer_list<op::DataType> DTYPE_COMMON_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                              op::DataType::DT_FLOAT16};
// ascend950 def 新增 bfloat16
static const std::initializer_list<op::DataType> DTYPE_950_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> GetDtypeSupportList()
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return DTYPE_950_SUPPORT_LIST;
    }
    return DTYPE_COMMON_SUPPORT_LIST;
}

const int64_t INDEX_2 = 2;
const int64_t INDEX_4 = 4;

} // namespace

static aclnnStatus CheckParamsNullptr(const AclnnParams& aclnnParams)
{
    OP_CHECK_NULL(aclnnParams.inputGates, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aclnnParams.hiddenGates, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aclnnParams.cx, return ACLNN_ERR_PARAM_NULLPTR);
    if (aclnnParams.inputBiasOptional) {
        OP_CHECK(aclnnParams.hiddenBiasOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hiddenBias should not be a nullptr when inputBias is set."),
                 return ACLNN_ERR_PARAM_NULLPTR);
    } else {
        OP_CHECK(aclnnParams.hiddenBiasOptional == nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "inputBias should not be a nullptr when hiddenBias is set."),
                 return ACLNN_ERR_PARAM_NULLPTR);
    }

    OP_CHECK_NULL(aclnnParams.hyOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aclnnParams.cyOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(aclnnParams.storageOut, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDimsAndListLength(const AclnnParams& aclnnParams)
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

static aclnnStatus CheckShapes(const AclnnParams& aclnnParams, AclnnInfo* info)
{
    auto cxShape = aclnnParams.cx->GetViewShape();
    info->B = cxShape.GetDim(0);
    info->H = cxShape.GetDim(1);
    op::Shape gatesShape = {info->B, info->H * INDEX_4};
    op::Shape commonShape = {info->B, info->H};
    op::Shape biasShape = {info->H * INDEX_4};

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.inputGates, gatesShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.hiddenGates, gatesShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.cx, commonShape, return ACLNN_ERR_PARAM_INVALID);
    if (aclnnParams.inputBiasOptional) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.inputBiasOptional, biasShape,
                                                    return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.hiddenBiasOptional, biasShape,
                                                    return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.hyOut, commonShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.cyOut, commonShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(aclnnParams.storageOut, gatesShape, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypes(const AclnnParams& aclnnParams, AclnnInfo* info)
{
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(aclnnParams.inputGates, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    info->dtype = aclnnParams.inputGates->GetDataType();
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.hiddenGates, info->dtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.cx, info->dtype, return ACLNN_ERR_PARAM_INVALID);
    if (aclnnParams.inputBiasOptional) {
        OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.inputBiasOptional, info->dtype, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.hiddenBiasOptional, info->dtype, return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.hyOut, info->dtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.cyOut, info->dtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(aclnnParams.storageOut, info->dtype, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus MakeParamsContig(AclnnParams* aclnnParams, const AclnnInfo& info)
{
    aclnnParams->inputGates = l0op::Contiguous(aclnnParams->inputGates, info.executor);
    CHECK_RET(aclnnParams->inputGates != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclnnParams->hiddenGates = l0op::Contiguous(aclnnParams->hiddenGates, info.executor);
    CHECK_RET(aclnnParams->hiddenGates != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclnnParams->cx = l0op::Contiguous(aclnnParams->cx, info.executor);
    CHECK_RET(aclnnParams->cx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (aclnnParams->inputBiasOptional) {
        aclnnParams->inputBiasOptional = l0op::Contiguous(aclnnParams->inputBiasOptional, info.executor);
        CHECK_RET(aclnnParams->inputBiasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclnnParams->hiddenBiasOptional = l0op::Contiguous(aclnnParams->hiddenBiasOptional, info.executor);
        CHECK_RET(aclnnParams->hiddenBiasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus HandleParams(AclnnParams* aclnnParams, const AclnnInfo& info)
{
    if (!aclnnParams->inputBiasOptional) {
        op::Shape biasShape = {info.H * INDEX_4};
        auto biasTmp = info.executor->AllocTensor(biasShape, info.dtype, op::Format::FORMAT_ND);
        CHECK_RET(biasTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aclnnParams->inputBiasOptional = l0op::ZerosLike(biasTmp, info.executor);
        aclnnParams->hiddenBiasOptional = l0op::ZerosLike(biasTmp, info.executor);
        CHECK_RET(aclnnParams->inputBiasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(aclnnParams->hiddenBiasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus RunBaseOp(const AclnnParams& aclnnParams, const AclnnInfo& info, BaseOpOutputs* baseOuts)
{
    op::Shape gatesShape = {info.B, info.H * INDEX_4};
    op::Shape commonShape = {info.B, info.H};
    baseOuts->hy = info.executor->AllocTensor(commonShape, info.dtype, op::Format::FORMAT_ND);
    CHECK_RET(baseOuts->hy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    baseOuts->cy = info.executor->AllocTensor(commonShape, info.dtype, op::Format::FORMAT_ND);
    CHECK_RET(baseOuts->cy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    baseOuts->storage = info.executor->AllocTensor(gatesShape, info.dtype, op::Format::FORMAT_ND);
    CHECK_RET(baseOuts->storage != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret = l0op::ThnnFusedLstmCell(aclnnParams.inputGates, aclnnParams.hiddenGates, aclnnParams.cx,
                                       aclnnParams.inputBiasOptional, aclnnParams.hiddenBiasOptional, baseOuts->hy,
                                       baseOuts->cy, baseOuts->storage, info.executor);
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE_WITHOUT_REPORT(ret, "Run l0op::ThnnFusedLstmCell failed."), return ret);
    return ACLNN_SUCCESS;
}

static aclnnStatus ViewCopyResults(const AclnnParams& aclnnParams, const AclnnInfo& info, const BaseOpOutputs& baseOuts)
{
    const aclTensor* res = nullptr;
    res = l0op::ViewCopy(baseOuts.hy, aclnnParams.hyOut, info.executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
    res = l0op::ViewCopy(baseOuts.cy, aclnnParams.cyOut, info.executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
    res = l0op::ViewCopy(baseOuts.storage, aclnnParams.storageOut, info.executor);
    CHECK_RET(res != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThnnFusedLstmCellGetWorkspaceSize(const aclTensor* inputGates, const aclTensor* hiddenGates,
                                                   const aclTensor* cx, const aclTensor* inputBiasOptional,
                                                   const aclTensor* hiddenBiasOptional, aclTensor* hyOut,
                                                   aclTensor* cyOut, aclTensor* storageOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnThnnFusedLstmCell, DFX_IN(inputGates, hiddenGates, cx, inputBiasOptional, hiddenBiasOptional),
                   DFX_OUT(hyOut, cyOut, storageOut));

    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    AclnnParams aclnnParams{inputGates,         hiddenGates, cx,    inputBiasOptional,
                            hiddenBiasOptional, hyOut,       cyOut, storageOut};
    AclnnInfo info{};
    BaseOpOutputs baseOuts{};

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    // REMOVED(unreachable): CREATE_EXECUTOR 使用普通 new，失败抛 bad_alloc，该 null 检查不可达
    info.executor = uniqueExecutor.get();

    aclnnStatus ret;

    // 空指针校验
    ret = CheckParamsNullptr(aclnnParams);
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ret, "CheckParamsNullptr failed, certain incoming nullptrs may be invalid."),
             return ret);

    // 预处理
    // tensor dim校验
    ret = CheckDimsAndListLength(aclnnParams);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // shape关系校验
    ret = CheckShapes(aclnnParams, &info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // dtype校验
    ret = CheckDtypes(aclnnParams, &info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理: 完成全部参数校验后早退;全空输入(B=0)为合法场景,
    // 空inputGates与其余张量不一致时按161002拒绝
    if (aclnnParams.inputGates->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 转连续
    ret = MakeParamsContig(&aclnnParams, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // bias处理
    ret = HandleParams(&aclnnParams, info);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 运行底层算子
    ret = RunBaseOp(aclnnParams, info, &baseOuts);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // ViewCopy输出
    ret = ViewCopyResults(aclnnParams, info, baseOuts);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThnnFusedLstmCell(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnThnnFusedLstmCell);
    //  固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
