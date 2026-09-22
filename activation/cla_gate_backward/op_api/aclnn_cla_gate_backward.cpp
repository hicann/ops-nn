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
 * \file aclnn_cla_gate_backward.cpp
 * \brief
 */
#include "aclnn_cla_gate_backward.h"
#include "cla_gate_backward.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

// 输入张量维度：三路 attention 为 [T, N, D]，gate logits 为 [T, N]。
constexpr size_t TND_DIM_NUM = 3;
constexpr size_t GATE_LOGITS_DIM_NUM = 2;

static bool CheckNotNull(const aclTensor* gradMerged, const aclTensor* globalAttn, const aclTensor* localAttn,
                         const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                         const aclTensor* gradGlobalAttnOut, const aclTensor* gradLocalAttnOut,
                         const aclTensor* gradGlobalGateLogitsOut, const aclTensor* gradLocalGateLogitsOut)
{
    OP_CHECK_NULL(gradMerged, return false);
    OP_CHECK_NULL(globalAttn, return false);
    OP_CHECK_NULL(localAttn, return false);
    OP_CHECK_NULL(globalGateLogits, return false);
    OP_CHECK_NULL(localGateLogits, return false);
    OP_CHECK_NULL(gradGlobalAttnOut, return false);
    OP_CHECK_NULL(gradLocalAttnOut, return false);
    OP_CHECK_NULL(gradGlobalGateLogitsOut, return false);
    OP_CHECK_NULL(gradLocalGateLogitsOut, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradMerged, const aclTensor* globalAttn, const aclTensor* localAttn,
                            const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                            const aclTensor* gradGlobalAttnOut, const aclTensor* gradLocalAttnOut,
                            const aclTensor* gradGlobalGateLogitsOut, const aclTensor* gradLocalGateLogitsOut)
{
    // 5 输入 dtype 必须在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradMerged, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(globalAttn, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(localAttn, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(globalGateLogits, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(localGateLogits, DTYPE_SUPPORT_LIST, return false);

    // 高进高出：5 路输入同 dtype，输出与对应输入一致
    OP_CHECK_DTYPE_NOT_SAME(globalAttn, gradMerged, return false);
    OP_CHECK_DTYPE_NOT_SAME(localAttn, gradMerged, return false);
    OP_CHECK_DTYPE_NOT_SAME(globalGateLogits, gradMerged, return false);
    OP_CHECK_DTYPE_NOT_SAME(localGateLogits, gradMerged, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradGlobalAttnOut, gradMerged, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradLocalAttnOut, gradMerged, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradGlobalGateLogitsOut, globalGateLogits, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradLocalGateLogitsOut, localGateLogits, return false);
    return true;
}

static bool CheckShapeValid(const aclTensor* gradMerged, const aclTensor* globalAttn, const aclTensor* localAttn,
                            const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                            const aclTensor* gradGlobalAttnOut, const aclTensor* gradLocalAttnOut,
                            const aclTensor* gradGlobalGateLogitsOut, const aclTensor* gradLocalGateLogitsOut)
{
    // gradMerged 必须为 3 维 [T, N, D]
    OP_CHECK_WRONG_DIMENSION(gradMerged, TND_DIM_NUM, return false);
    // 三路 TND 输入 shape 必须完全相同
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(globalAttn, gradMerged->GetViewShape(), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(localAttn, gradMerged->GetViewShape(), return false);

    // gate logits 必须为 2 维 [T, N]
    OP_CHECK_WRONG_DIMENSION(globalGateLogits, GATE_LOGITS_DIM_NUM, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(localGateLogits, globalGateLogits->GetViewShape(), return false);

    // 输出 shape 与对应输入一致
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradGlobalAttnOut, gradMerged->GetViewShape(), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradLocalAttnOut, gradMerged->GetViewShape(), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradGlobalGateLogitsOut, globalGateLogits->GetViewShape(),
                                                return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradLocalGateLogitsOut, localGateLogits->GetViewShape(), return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradMerged, const aclTensor* globalAttn, const aclTensor* localAttn,
                               const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                               const aclTensor* gradGlobalAttnOut, const aclTensor* gradLocalAttnOut,
                               const aclTensor* gradGlobalGateLogitsOut, const aclTensor* gradLocalGateLogitsOut)
{
    CHECK_RET(CheckNotNull(gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, gradGlobalAttnOut,
                           gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, gradGlobalAttnOut,
                              gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, gradGlobalAttnOut,
                              gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnClaGateBackwardGetWorkspaceSize(const aclTensor* gradMerged, const aclTensor* globalAttn,
                                                 const aclTensor* localAttn, const aclTensor* globalGateLogits,
                                                 const aclTensor* localGateLogits, const char* inputAttnLayout,
                                                 const aclTensor* gradGlobalAttnOut, const aclTensor* gradLocalAttnOut,
                                                 const aclTensor* gradGlobalGateLogitsOut,
                                                 const aclTensor* gradLocalGateLogitsOut, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnClaGateBackward,
                   DFX_IN(gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, inputAttnLayout),
                   DFX_OUT(gradGlobalAttnOut, gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut));

    auto ret = CheckParams(gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, gradGlobalAttnOut,
                           gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 输入连续化
    auto gradMergedCont = l0op::Contiguous(gradMerged, uniqueExecutor.get());
    CHECK_RET(gradMergedCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto globalAttnCont = l0op::Contiguous(globalAttn, uniqueExecutor.get());
    CHECK_RET(globalAttnCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto localAttnCont = l0op::Contiguous(localAttn, uniqueExecutor.get());
    CHECK_RET(localAttnCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto globalGateLogitsCont = l0op::Contiguous(globalGateLogits, uniqueExecutor.get());
    CHECK_RET(globalGateLogitsCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto localGateLogitsCont = l0op::Contiguous(localGateLogits, uniqueExecutor.get());
    CHECK_RET(localGateLogitsCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out = l0op::ClaGateBackward(gradMergedCont, globalAttnCont, localAttnCont, globalGateLogitsCont,
                                     localGateLogitsCont, inputAttnLayout, uniqueExecutor.get());
    CHECK_RET(out.gradGlobalAttnOut != nullptr && out.gradLocalAttnOut != nullptr &&
                  out.gradGlobalGateLogitsOut != nullptr && out.gradLocalGateLogitsOut != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto vc1 = l0op::ViewCopy(out.gradGlobalAttnOut, gradGlobalAttnOut, uniqueExecutor.get());
    CHECK_RET(vc1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vc2 = l0op::ViewCopy(out.gradLocalAttnOut, gradLocalAttnOut, uniqueExecutor.get());
    CHECK_RET(vc2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vc3 = l0op::ViewCopy(out.gradGlobalGateLogitsOut, gradGlobalGateLogitsOut, uniqueExecutor.get());
    CHECK_RET(vc3 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vc4 = l0op::ViewCopy(out.gradLocalGateLogitsOut, gradLocalGateLogitsOut, uniqueExecutor.get());
    CHECK_RET(vc4 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnClaGateBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClaGateBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
