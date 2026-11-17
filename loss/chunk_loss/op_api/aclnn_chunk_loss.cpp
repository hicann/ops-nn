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
 * \file aclnn_chunk_loss.cpp
 * \brief aclnnChunkLoss 对外 C API 实现
 *
 * 校验输入并调用 l0op::ChunkLoss 构图执行。
 */
#include "aclnn_chunk_loss.h"

#include <cmath>

#include "chunk_loss.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static bool IsNdFormat(const aclTensor* tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    return tensor->GetStorageFormat() == op::Format::FORMAT_ND || tensor->GetViewFormat() == op::Format::FORMAT_ND;
}

static aclnnStatus ViewCopyIfNeeded(const aclTensor* src, aclTensor* dst, aclOpExecutor* executor)
{
    if (src == dst) {
        return ACLNN_SUCCESS;
    }
    auto copied = l0op::ViewCopy(src, dst, executor);
    CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkLossGetWorkspaceSize(const aclTensor* hidden, const aclTensor* weight, const aclTensor* labels,
                                           double alpha, int64_t ignoreIndex, aclTensor* loss, aclTensor* maxOut,
                                           aclTensor* sumOut, aclTensor* dH, uint64_t* workspaceSize,
                                           aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnChunkLoss, DFX_IN(hidden, weight, labels, alpha, ignoreIndex),
                   DFX_OUT(loss, maxOut, sumOut, dH));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (hidden == nullptr || weight == nullptr || labels == nullptr || loss == nullptr || maxOut == nullptr ||
        sumOut == nullptr || dH == nullptr || workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!std::isfinite(alpha) || alpha <= 0.0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto hType = hidden->GetDataType();
    if ((hType != DataType::DT_FLOAT16 && hType != DataType::DT_BF16) || hType != weight->GetDataType() ||
        labels->GetDataType() != DataType::DT_INT64 || loss->GetDataType() != DataType::DT_FLOAT ||
        maxOut->GetDataType() != DataType::DT_FLOAT || sumOut->GetDataType() != DataType::DT_FLOAT ||
        dH->GetDataType() != hType) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsNdFormat(hidden) || !IsNdFormat(weight) || !IsNdFormat(labels) || !IsNdFormat(loss) || !IsNdFormat(maxOut) ||
        !IsNdFormat(sumOut) || !IsNdFormat(dH)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hidden->GetViewShape().GetDimNum() != 2 || weight->GetViewShape().GetDimNum() != 2 ||
        labels->GetViewShape().GetDimNum() != 1 || loss->GetViewShape().GetDimNum() != 0 ||
        maxOut->GetViewShape().GetDimNum() != 1 || sumOut->GetViewShape().GetDimNum() != 1 ||
        dH->GetViewShape().GetDimNum() != 2) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t m = hidden->GetViewShape().GetDim(0);
    int64_t h = hidden->GetViewShape().GetDim(1);
    int64_t v = weight->GetViewShape().GetDim(0);
    if (m < 1 || h < 1 || v < 1 || weight->GetViewShape().GetDim(1) != h || labels->GetViewShape().GetDim(0) != m ||
        maxOut->GetViewShape().GetDim(0) != m || sumOut->GetViewShape().GetDim(0) != m ||
        dH->GetViewShape().GetDim(0) != m || dH->GetViewShape().GetDim(1) != h) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto* exec = uniqueExecutor.get();
    auto hiddenCont = l0op::Contiguous(hidden, exec);
    auto weightCont = l0op::Contiguous(weight, exec);
    auto labelsCont = l0op::Contiguous(labels, exec);
    auto lossCont = l0op::Contiguous(loss, exec);
    auto maxOutCont = l0op::Contiguous(maxOut, exec);
    auto sumOutCont = l0op::Contiguous(sumOut, exec);
    auto dHCont = l0op::Contiguous(dH, exec);
    CHECK_RET(hiddenCont != nullptr && weightCont != nullptr && labelsCont != nullptr && lossCont != nullptr &&
                  maxOutCont != nullptr && sumOutCont != nullptr && dHCont != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ChunkLoss(hiddenCont, weightCont, labelsCont, static_cast<float>(alpha), ignoreIndex,
                                  const_cast<aclTensor*>(lossCont), const_cast<aclTensor*>(maxOutCont),
                                  const_cast<aclTensor*>(sumOutCont), const_cast<aclTensor*>(dHCont), exec);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(ViewCopyIfNeeded(lossCont, loss, exec) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfNeeded(maxOutCont, maxOut, exec) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfNeeded(sumOutCont, sumOut, exec) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfNeeded(dHCont, dH, exec) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkLoss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkLoss);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
