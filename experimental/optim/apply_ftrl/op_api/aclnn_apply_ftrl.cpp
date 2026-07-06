/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file aclnn_apply_ftrl.cpp
 * \brief ApplyFtrl L2 (aclnn) API implementation -- DERIVED two-phase façade.
 *
 * Standard L2 flow:
 *   CREATE_EXECUTOR -> CheckParams -> empty-tensor fast path -> Contiguous(var/accum/
 *   linear/grad) -> l0op::ApplyFtrl -> ViewCopy each Ref back (if non-contiguous) ->
 *   GetWorkspaceSize.
 *
 * var / accum / linear are Ref tensors (in place); grad / lr / l1 / l2 / lrPower are
 * read-only.  All tensors share one dtype in {FLOAT, FLOAT16, BFLOAT16}; var/accum/
 * linear/grad share one shape; lr/l1/l2/lrPower are scalars (0-D or 1-element).
 */

#include "aclnn_apply_ftrl.h"
#include "apply_ftrl.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// Supported input dtypes (all inputs/outputs share one dtype): FLOAT / FLOAT16 / BFLOAT16.
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* varRef, const aclTensor* accumRef, const aclTensor* linearRef,
                         const aclTensor* grad, const aclTensor* lr, const aclTensor* l1, const aclTensor* l2,
                         const aclTensor* lrPower)
{
    OP_CHECK_NULL(varRef, return false);
    OP_CHECK_NULL(accumRef, return false);
    OP_CHECK_NULL(linearRef, return false);
    OP_CHECK_NULL(grad, return false);
    OP_CHECK_NULL(lr, return false);
    OP_CHECK_NULL(l1, return false);
    OP_CHECK_NULL(l2, return false);
    OP_CHECK_NULL(lrPower, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* varRef, const aclTensor* accumRef, const aclTensor* linearRef,
                            const aclTensor* grad, const aclTensor* lr, const aclTensor* l1, const aclTensor* l2,
                            const aclTensor* lrPower)
{
    // var dtype must be in the support list; every other input must match var exactly.
    OP_CHECK_DTYPE_NOT_SUPPORT(varRef, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(accumRef, varRef, return false);
    OP_CHECK_DTYPE_NOT_SAME(linearRef, varRef, return false);
    OP_CHECK_DTYPE_NOT_SAME(grad, varRef, return false);
    OP_CHECK_DTYPE_NOT_SAME(lr, varRef, return false);
    OP_CHECK_DTYPE_NOT_SAME(l1, varRef, return false);
    OP_CHECK_DTYPE_NOT_SAME(l2, varRef, return false);
    OP_CHECK_DTYPE_NOT_SAME(lrPower, varRef, return false);
    return true;
}

static bool CheckShape(const aclTensor* varRef, const aclTensor* accumRef, const aclTensor* linearRef,
                       const aclTensor* grad, const aclTensor* lr, const aclTensor* l1, const aclTensor* l2,
                       const aclTensor* lrPower)
{
    // var/accum/linear/grad must share the exact same shape.
    OP_CHECK_SHAPE_NOT_EQUAL(accumRef, varRef, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(linearRef, varRef, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(grad, varRef, return false);
    // lr/l1/l2/lrPower must be scalars (0-D or a single element).
    if (lr->Numel() != 1 || l1->Numel() != 1 || l2->Numel() != 1 || lrPower->Numel() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "lr/l1/l2/lrPower must be scalar (0-D or 1-element). Got numel lr=%ld, l1=%ld, l2=%ld, lrPower=%ld.",
                lr->Numel(), l1->Numel(), l2->Numel(), lrPower->Numel());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* varRef, const aclTensor* accumRef, const aclTensor* linearRef,
                               const aclTensor* grad, const aclTensor* lr, const aclTensor* l1, const aclTensor* l2,
                               const aclTensor* lrPower)
{
    // 1. nullptr check.
    CHECK_RET(CheckNotNull(varRef, accumRef, linearRef, grad, lr, l1, l2, lrPower), ACLNN_ERR_PARAM_NULLPTR);
    // 2. dtype check (support list + cross-input consistency).
    CHECK_RET(CheckDtypeValid(varRef, accumRef, linearRef, grad, lr, l1, l2, lrPower), ACLNN_ERR_PARAM_INVALID);
    // 3. shape check.
    CHECK_RET(CheckShape(varRef, accumRef, linearRef, grad, lr, l1, l2, lrPower), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnApplyFtrlGetWorkspaceSize(aclTensor* varRef, aclTensor* accumRef, aclTensor* linearRef,
                                           const aclTensor* grad, const aclTensor* lr, const aclTensor* l1,
                                           const aclTensor* l2, const aclTensor* lrPower, bool useLocking,
                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // useLocking maps to attribute use_locking; only false is supported.  It does not
    // affect the math and is not read by tiling/kernel, so it is traced but not lowered.
    L2_DFX_PHASE_1(aclnnApplyFtrl, DFX_IN(varRef, accumRef, linearRef, grad, lr, l1, l2, lrPower, useLocking),
                   DFX_OUT());

    // Fixed pattern: create the OpExecutor.
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Fixed pattern: parameter check.
    auto ret = CheckParams(varRef, accumRef, linearRef, grad, lr, l1, l2, lrPower);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Empty-tensor fast path (var/accum/linear/grad share one shape).
    if (varRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Fixed pattern: make the Ref/grad tensors contiguous for the kernel.
    auto varContiguous = l0op::Contiguous(varRef, uniqueExecutor.get());
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto accumContiguous = l0op::Contiguous(accumRef, uniqueExecutor.get());
    CHECK_RET(accumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto linearContiguous = l0op::Contiguous(linearRef, uniqueExecutor.get());
    CHECK_RET(linearContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
    CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // lr/l1/l2/lrPower are single-element scalars -> always contiguous; passed as-is.

    // Fixed pattern: invoke the L0 ApplyFtrl op (var/accum/linear updated in place).
    auto [varOut, accumOut, linearOut] = l0op::ApplyFtrl(varContiguous, accumContiguous, linearContiguous,
                                                         gradContiguous, lr, l1, l2, lrPower, uniqueExecutor.get());
    CHECK_RET(varOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(accumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(linearOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Fixed pattern: copy each result back onto its (possibly non-contiguous) Ref tensor.
    // When the Ref is already contiguous the kernel wrote into its own GM, so no copy needed.
    if (!IsContiguous(varRef)) {
        auto viewCopyResult = l0op::ViewCopy(varOut, varRef, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!IsContiguous(accumRef)) {
        auto viewCopyResult = l0op::ViewCopy(accumOut, accumRef, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!IsContiguous(linearRef)) {
        auto viewCopyResult = l0op::ViewCopy(linearOut, linearRef, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Fixed pattern: obtain workspace size and hand the executor over.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnApplyFtrl(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnApplyFtrl);
    // Fixed pattern: run the executor.
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
