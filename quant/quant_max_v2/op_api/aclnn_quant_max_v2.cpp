/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_max_v2.h"
#include "quant_max_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include <string>
#include <cstring>
#include <cmath>

using namespace op;

constexpr int64_t ACLNN_MIN_SHAPE_RANK = 2;
constexpr int64_t ACLNN_MAX_SHAPE_RANK = 8;

static constexpr const char* K_OP_NAME = "aclnnQuantMaxV2";
constexpr int64_t MAX_GROUP_NUM = 384;
constexpr float DST_TYPE_MAX_LIMIT = 32768.0f;

// dstType 枚举：34=HIFLOAT8，35=FLOAT8_E5M2，36=FLOAT8_E4M3FN
constexpr int64_t DST_TYPE_HIFLOAT8 = 34;
constexpr int64_t DST_TYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DST_TYPE_FLOAT8_E4M3FN = 36;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                              DataType::DT_BF16};

static op::DataType GetOutputDtype(int64_t dstType)
{
    switch (dstType) {
        case DST_TYPE_HIFLOAT8:
            return DataType::DT_HIFLOAT8;
        case DST_TYPE_FLOAT8_E5M2:
            return DataType::DT_FLOAT8_E5M2;
        case DST_TYPE_FLOAT8_E4M3FN:
            return DataType::DT_FLOAT8_E4M3FN;
        default:
            return DataType::DT_UNDEFINED;
    }
}

// 工具：判断两个 tensor 的 shape 是否逐维相等。
static bool IsShapeEquals(const aclTensor* tensor1, const aclTensor* tensor2)
{
    auto shape1 = tensor1->GetViewShape();
    auto shape2 = tensor2->GetViewShape();
    if (shape1.GetDimNum() != shape2.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < shape1.GetDimNum(); i++) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}

// 1. 校验空指针。
static aclnnStatus CheckNullParams(const aclTensor* x, const aclTensor* y, const aclTensor* amaxOut,
                                   const char* roundMode)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(y, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(amaxOut, return ACLNN_ERR_PARAM_NULLPTR); // amax 为必选输出。
    return ACLNN_SUCCESS;
}

// 2. 校验 x、scale、groupList 输入参数。
static aclnnStatus CheckInputParams(const aclTensor* x, const aclTensor* scale, const aclTensor* groupList)
{
    // 校验 x 的数据类型。
    if (!CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(K_OP_NAME, "x", op::ToString(x->GetDataType()).GetString(),
                                              "The dtype of x must be within the range FLOAT/FLOAT16/BFLOAT16");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 校验 x 的维数范围 [2, 8]。
    auto xShape = x->GetViewShape();
    if (xShape.GetDimNum() < ACLNN_MIN_SHAPE_RANK || xShape.GetDimNum() > ACLNN_MAX_SHAPE_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_OP_NAME, "x", std::to_string(xShape.GetDimNum()).c_str(),
                                                 "The shape dim of x must be in range [2D, 8D]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // scale 的数据类型必须为 FLOAT。
    if (scale != nullptr && scale->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(K_OP_NAME, "scale", op::ToString(scale->GetDataType()).GetString(), "FLOAT");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // scale 必须为非空的 [num_groups]；维数错误与空张量分别报错。
    if (scale != nullptr && scale->GetViewShape().GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_OP_NAME, "scale",
                                                 std::to_string(scale->GetViewShape().GetDimNum()).c_str(),
                                                 "scale must be rank one [num_groups]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (scale != nullptr && scale->IsEmpty()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "scale", "empty", "scale does not support empty tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // groupList 的数据类型必须为 INT64。
    if (groupList != nullptr && groupList->GetDataType() != DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(K_OP_NAME, "groupList", op::ToString(groupList->GetDataType()).GetString(), "INT64");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // groupList 必须为非空的 [num_groups]；维数错误与空张量分别报错。
    if (groupList != nullptr && groupList->GetViewShape().GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(K_OP_NAME, "groupList",
                                                 std::to_string(groupList->GetViewShape().GetDimNum()).c_str(),
                                                 "groupList must be rank one [num_groups]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (groupList != nullptr && groupList->IsEmpty()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "groupList", "empty",
                                              "groupList does not support empty tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto groups = groupList == nullptr ? 1 : groupList->GetViewShape().GetDim(0);
    if (groups < 1 || groups > MAX_GROUP_NUM || (scale != nullptr && scale->GetViewShape().GetDim(0) != groups)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "groups", std::to_string(groups).c_str(),
                                              "group count must be in [1, 384] and scale length must equal groups");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 3. 校验 dstType。
static aclnnStatus CheckDstType(int64_t dstType)
{
    if (dstType != DST_TYPE_HIFLOAT8 && dstType != DST_TYPE_FLOAT8_E5M2 && dstType != DST_TYPE_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "dstType", std::to_string(dstType).c_str(),
                                              "dstType must be 34(HIFLOAT8)/35(FLOAT8_E5M2)/36(FLOAT8_E4M3FN)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 4. 校验舍入模式：FP8 仅支持 rint，HIFLOAT8 支持 round 或 hybrid。
static aclnnStatus CheckRoundMode(int64_t dstType, const char* roundMode)
{
    // roundMode 不能为空指针。
    if (roundMode == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "roundMode", "nullptr", "roundMode cannot be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // roundMode 不能为空字符串。
    if (strlen(roundMode) == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "roundMode", "<empty>", "roundMode cannot be empty string");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 校验舍入模式与输出类型是否匹配。
    if (dstType == DST_TYPE_FLOAT8_E5M2 || dstType == DST_TYPE_FLOAT8_E4M3FN) {
        // FLOAT8_E5M2 和 FLOAT8_E4M3FN 仅支持 rint。
        if (strcmp(roundMode, "rint") != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                K_OP_NAME, "roundMode", roundMode,
                "roundMode must be 'rint' for dstType FLOAT8_E5M2(35) or FLOAT8_E4M3FN(36)");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (dstType == DST_TYPE_HIFLOAT8) {
        // HIFLOAT8 仅支持 round 或 hybrid。
        if (strcmp(roundMode, "round") != 0 && strcmp(roundMode, "hybrid") != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "roundMode", roundMode,
                                                  "roundMode must be 'round' or 'hybrid' for dstType HIFLOAT8(34)");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// 5. 校验输出：amax 必选，形状为 [num_groups]；yScale 仅动态模式输出。
static aclnnStatus CheckOutputParams(const aclTensor* x, const aclTensor* groupList, bool isDynamic, int64_t dstType,
                                     const aclTensor* y, const aclTensor* amaxOut, const aclTensor* yScaleOutOptional)
{
    // y 的数据类型必须与 dstType 对应。
    auto expectedYDtype = GetOutputDtype(dstType);
    if (y->GetDataType() != expectedYDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(K_OP_NAME, "y", op::ToString(y->GetDataType()).GetString(),
                                  op::ToString(expectedYDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // y 的形状必须与 x 相同。
    if (!IsShapeEquals(y, x)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(K_OP_NAME, "y", op::ToString(y->GetViewShape()).GetString(),
                                               "The shape of y must be equal to the shape of x");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto groups = groupList == nullptr ? 1 : groupList->GetViewShape().GetDim(0);
    if (yScaleOutOptional == nullptr) {
        // 静态量化约定传入 nullptr；动态量化必须提供实际的 yScale 张量。
        if (isDynamic) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "yScale", "nullptr",
                                                  "dynamic quantization must output yScale");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (isDynamic) {
        // 动态模式：yScale 必须为 FLOAT [num_groups]；支持非连续视图（内部 Contiguous + ViewCopy 回写）。
        if (yScaleOutOptional->GetDataType() != DataType::DT_FLOAT ||
            yScaleOutOptional->GetViewShape().GetDimNum() != 1 ||
            yScaleOutOptional->GetViewShape().GetDim(0) != groups) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "yScale",
                                                  op::ToString(yScaleOutOptional->GetViewShape()).GetString(),
                                                  "yScale must be FLOAT [num_groups]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        // 静态模式：y_scale 不输出，非空一律拒绝（不写值的「容忍」会造成三处 shape 契约互斥：
        // 本层按 [num_groups] 校验、InferShape 推导 [0]、tiling 要求 dim0==0，任何非空 shape
        // 都会在某一层失败）。与 tiling/InferShape 统一为「静态只接受 nullptr」。
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "yScale",
                                              yScaleOutOptional == nullptr ? "nullptr" : "non-null tensor",
                                              "static quantization does not output yScale; pass nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // amax 的数据类型必须与 x 相同。
    if (amaxOut->GetDataType() != x->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(K_OP_NAME, "amax", op::ToString(amaxOut->GetDataType()).GetString(),
                                              "The dtype of amax must be the same as x");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // amax 的形状必须为 [num_groups]。
    const auto& shape = amaxOut->GetViewShape();
    if (shape.GetDimNum() != 1 || shape.GetDim(0) != groups) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "amax", op::ToString(shape).GetString(),
                                              "amax must be [num_groups]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// aclnn 第一段入参校验：空指针、dtype、shape、属性取值与模式契约（动态必须给
// y_scale、静态不输出），失败按错误码分级上报。
static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* scale, const aclTensor* groupList,
                               const char* roundMode, int64_t dstType, float dstTypeMax, const aclTensor* y,
                               const aclTensor* amaxOut, const aclTensor* yScaleOutOptional)
{
    // 校验空指针。
    auto ret = CheckNullParams(x, y, amaxOut, roundMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 校验输入类型和形状。
    ret = CheckInputParams(x, scale, groupList);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 校验 dstType。
    ret = CheckDstType(dstType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 校验 roundMode 取值。
    ret = CheckRoundMode(dstType, roundMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 校验 dstTypeMax：须为有限值且在 [0, 32768]（范围校验对传入值恒生效；
    // E5M2/E4M3FN 场景该值在 tiling 内被覆盖为类型最大值，实际不参与计算）。
    if (!std::isfinite(dstTypeMax) || dstTypeMax < 0.0f || dstTypeMax > DST_TYPE_MAX_LIMIT) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "dstTypeMax", std::to_string(dstTypeMax).c_str(),
                                              "dstTypeMax must be finite and in [0.0, 32768.0]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验输出类型和形状；scale 是否传入选定动态/静态模式（scale == nullptr 即动态）。
    ret = CheckOutputParams(x, groupList, scale == nullptr, dstType, y, amaxOut, yScaleOutOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ACLNN_SUCCESS;
}

// 两段式调用第一步（参数契约见 aclnn_quant_max_v2.h）：
// 校验 -> 输入/输出连续化 -> L0 组装执行计划 -> 非连续输出 ViewCopy 回写 -> 计算 workspace。
aclnnStatus aclnnQuantMaxV2GetWorkspaceSize(const aclTensor* x, const aclTensor* xScaleOptional,
                                            const aclTensor* groupListOptional, const char* roundMode, int64_t dstType,
                                            float dstTypeMax, const aclTensor* yOut, const aclTensor* amaxOut,
                                            const aclTensor* yScaleOutOptional, uint64_t* workspaceSize,
                                            aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantMaxV2, DFX_IN(x, xScaleOptional, groupListOptional, roundMode, dstType, dstTypeMax),
                   DFX_OUT(yOut, amaxOut, yScaleOutOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, xScaleOptional, groupListOptional, roundMode, dstType, dstTypeMax, yOut, amaxOut,
                           yScaleOutOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scaleContiguous = xScaleOptional == nullptr ? nullptr : l0op::Contiguous(xScaleOptional, uniqueExecutor.get());
    CHECK_RET(xScaleOptional == nullptr || scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto groupListContiguous = groupListOptional == nullptr ? nullptr :
                                                              l0op::Contiguous(groupListOptional, uniqueExecutor.get());
    CHECK_RET(groupListOptional == nullptr || groupListContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto yContiguous = l0op::Contiguous(yOut, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto amaxContiguous = l0op::Contiguous(amaxOut, uniqueExecutor.get());
    CHECK_RET(amaxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 静态模式 yScale 传 nullptr 时分配形状 [0] 的占位张量，保证 L0 三输出齐全；
    // 占位张量不会被 kernel 写入。
    auto yScaleContiguous = yScaleOutOptional == nullptr ?
                                uniqueExecutor->AllocTensor(op::Shape({0}), DataType::DT_FLOAT, op::Format::FORMAT_ND) :
                                l0op::Contiguous(yScaleOutOptional, uniqueExecutor.get());
    CHECK_RET(yScaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // roundMode 已在 CheckParams 中校验，可直接使用。
    auto opResult = l0op::QuantMaxV2(xContiguous, scaleContiguous, groupListContiguous, roundMode, dstType, dstTypeMax,
                                     yContiguous, amaxContiguous, yScaleContiguous, uniqueExecutor.get());
    CHECK_RET(opResult[0] != nullptr && opResult[1] != nullptr && opResult[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto yComputeOut = opResult[0];
    auto amaxComputeOut = opResult[1];
    auto yScaleComputeOut = opResult[2];

    // 输出非连续时，将连续计算结果拷贝到调用方张量。
    auto viewCopyResultY = l0op::ViewCopy(yComputeOut, yOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResultY != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResultAmax = l0op::ViewCopy(amaxComputeOut, amaxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResultAmax != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 动态模式：y_scale 回写到调用方原始张量（yScaleOutOptional）；连续性已在 CheckOutputParams 校验。
    if (xScaleOptional == nullptr) {
        auto viewCopyResultYScale = l0op::ViewCopy(yScaleComputeOut, yScaleOutOptional, uniqueExecutor.get());
        CHECK_RET(viewCopyResultYScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// 两段式调用第二步：在指定 stream 上下发执行计划（计划已含 ViewCopy 等回写算子）。
aclnnStatus aclnnQuantMaxV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMaxV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
