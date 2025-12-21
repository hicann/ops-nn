/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_quant_scatter_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "quant_update_scatter.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
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
namespace {
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_SELFREF = {DataType::DT_INT8};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_SELFREF_91095 = {
    DataType::DT_INT8, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_HIFLOAT8};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_INDICES = {DataType::DT_INT32, DataType::DT_INT64};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_UPDATES = {DataType::DT_BF16, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_QUANT_SCALES = {DataType::DT_BF16, DataType::DT_FLOAT};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_QUANT_ZERO_POINTS = {
    DataType::DT_BF16, DataType::DT_INT32};

static inline bool CheckNotNull(
    const aclTensor* selfRef, const aclTensor* indices, const aclTensor* updates, const aclTensor* quantScales)
{
    OP_CHECK_NULL(selfRef, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(updates, return false);
    OP_CHECK_NULL(quantScales, return false);
    return true;
}

static inline bool CheckPlatform()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(socVersion == SocVersion::ASCEND910_95,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Support for %s is not implemented.",
                op::ToString(socVersion).GetString()),
            return false);
    return true;
}

static inline bool CheckDtypeCombineValid(
    const aclTensor* updates, const aclTensor* quantScales, const aclTensor* quantZeroPoints)
{
    DataType updatesDtype = updates->GetDataType();
    DataType quantScalesDtype = quantScales->GetDataType();

    if (quantZeroPoints) {
        DataType quantZeroPointsDtype = quantZeroPoints->GetDataType();
        if (updatesDtype == DataType::DT_FLOAT16 && quantScalesDtype == DataType::DT_FLOAT &&
            quantZeroPointsDtype == DataType::DT_INT32) {
            return true;
        } else if (
            updatesDtype == DataType::DT_BF16 && quantScalesDtype == DataType::DT_BF16 &&
            quantZeroPointsDtype == DataType::DT_BF16) {
            return true;
        } else {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "(updates,quantScales,quantZeroPoints) Dtype should be (float16, float, int32) or (bfloat16, bfloat16, "
                "bfloat16), but is (%s, %s, %s).",
                op::ToString(updates->GetDataType()).GetString(), op::ToString(quantScales->GetDataType()).GetString(),
                op::ToString(quantZeroPoints->GetDataType()).GetString());
            return false;
        }
    } else {
        if (updatesDtype == DataType::DT_FLOAT16 && quantScalesDtype == DataType::DT_FLOAT) {
            return true;
        } else if (updatesDtype == DataType::DT_BF16 && quantScalesDtype == DataType::DT_BF16) {
            return true;
        } else {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "(updates,quantScales) Dtype should be (float16, float) or (bfloat16, bfloat16), but is (%s, %s).",
                op::ToString(updates->GetDataType()).GetString(), op::ToString(quantScales->GetDataType()).GetString());
            return false;
        }
    }
}

static bool CheckRoundMode(const aclTensor* selfRef, const char* roundMode)
{
    DataType dstType = selfRef->GetDataType();
    // 检查参数 roundMode 是否合法
    const std::string mode = std::string(roundMode);
    if (dstType == op::DataType::DT_HIFLOAT8) {
      if (mode != "round" && mode != "hybrid") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "check roundMode failed, roundMode[%s] not in ['round','hybrid'] for hifloat8.", mode.c_str());
        return false;
      }
    } else {
      if (mode != "rint") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "check roundMode failed, roundMode[%s] not in ['rint'] for float8_e5m2/float8_e4m3fn/int8.", mode.c_str());
        return false;
      }
    }
    return true;
}


static inline bool CheckDtypeValid(
    const aclTensor* selfRef, const aclTensor* indices, const aclTensor* updates, const aclTensor* quantScales,
    const aclTensor* quantZeroPoints)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, DTYPE_SUPPORT_LIST_SELFREF_91095, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, DTYPE_SUPPORT_LIST_SELFREF, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, DTYPE_SUPPORT_LIST_INDICES, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(updates, DTYPE_SUPPORT_LIST_UPDATES, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(quantScales, DTYPE_SUPPORT_LIST_QUANT_SCALES, return false);

    if (quantZeroPoints) {
        OP_CHECK_DTYPE_NOT_SUPPORT(quantZeroPoints, DTYPE_SUPPORT_LIST_QUANT_ZERO_POINTS, return false);
    }

    if (!CheckDtypeCombineValid(updates, quantScales, quantZeroPoints)) {
        return false;
    }
    return true;
}

static inline bool CheckShape(const aclTensor* selfRef, const aclTensor* updates)
{
    if (selfRef->GetViewShape().GetDimNum() != updates->GetViewShape().GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimension of selfRef and updates should equal");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* selfRef, const aclTensor* indices, const aclTensor* updates, const aclTensor* quantScales,
    const aclTensor* quantZeroPoints, const char* roundMode)
{
    // 1. 检查平台
    CHECK_RET(CheckPlatform(), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(selfRef, indices, updates, quantScales), ACLNN_ERR_PARAM_NULLPTR);

    // 3. 检查参数的数据类型是否符合预期
    CHECK_RET(CheckDtypeValid(selfRef, indices, updates, quantScales, quantZeroPoints), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入tensor的shape
    CHECK_RET(CheckShape(selfRef, updates), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入roundMode
    CHECK_RET(CheckRoundMode(selfRef, roundMode), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
}
aclnnStatus aclnnInplaceQuantScatterV2GetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* indices, const aclTensor* updates, const aclTensor* quantScales,
    const aclTensor* quantZeroPoints, int64_t axis, int64_t quantAxis, int64_t reduction, const char* roundMode,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnInplaceQuantScatterV2,
        DFX_IN(selfRef, indices, updates, quantScales, quantZeroPoints, axis, quantAxis, reduction, roundMode), DFX_OUT(selfRef));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(selfRef, indices, updates, quantScales, quantZeroPoints, roundMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (selfRef->IsEmpty() || indices->IsEmpty() || updates->IsEmpty() || quantScales->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入selfRef转换成连续的tensor
    auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入indices转换成连续的tensor
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入updates转换成连续的tensor
    auto updatesContiguous = l0op::Contiguous(updates, uniqueExecutor.get());
    CHECK_RET(updatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入quantScales转换成连续的tensor
    auto quantScalesContiguous = l0op::Contiguous(quantScales, uniqueExecutor.get());
    CHECK_RET(quantScalesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* quantZeroPointsContiguous = nullptr;
    if (quantZeroPoints) {
        // 将输入quantZeroPoints转换成连续的tensor
        quantZeroPointsContiguous = l0op::Contiguous(quantZeroPoints, uniqueExecutor.get());
        CHECK_RET(quantZeroPointsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const std::string reduce = "update";
    const bool reciprocalScale = false;
    // 执行L0算子
    auto quantUpdateScatterResult = l0op::QuantUpdateScatter(
        selfRefContiguous, indicesContiguous, updatesContiguous, quantScalesContiguous, quantZeroPointsContiguous,
        reduce, axis, quantAxis, uniqueExecutor.get(), reciprocalScale, roundMode);
    CHECK_RET(quantUpdateScatterResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出data上
    auto viewCopyResult = l0op::ViewCopy(quantUpdateScatterResult, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceQuantScatterV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceQuantScatterV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
