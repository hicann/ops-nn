/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_gelu_quant.h"
#include "gelu_quant.h"
#include "level0/fault_injection.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

#include <dlfcn.h>

#include <new>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t DIM_NUM_TWO = 2;
static constexpr int64_t DIM_NUM_EIGHT = 8;
static constexpr int64_t DIM_NUM_SEVEN = 7;
static constexpr int64_t NUM_TWO = 2;

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> Y_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_INT8, op::DataType::DT_HIFLOAT8};

static inline bool CheckNotNull(
    const aclTensor* self, const aclTensor* inputScaleOptional, const char* approximate,
    const char* roundMode, const aclTensor* y, const aclTensor* outScaleOptional, bool isDynQuant)
{
    OP_CHECK_NULL(self, return false);
    if (!isDynQuant) {
        OP_CHECK_NULL(inputScaleOptional, return false);
    }
    CHECK_RET(approximate != nullptr, false);
    CHECK_RET(roundMode != nullptr, false);
    OP_CHECK_NULL(y, return false);
    if (isDynQuant) {
        OP_CHECK_NULL(outScaleOptional, return false);
    }
    return true;
}

static bool IsDynamicQuant(const char* quantMode)
{
    return (strncmp(quantMode, "dynamic", strlen("dynamic")) == 0);
}

static bool GetQuantMode(const char* quantMode, bool& isDynQuant)
{
    // 检查参数 quantMode 是否合法
    OP_LOGD("Get GeluQuant QuantMode: quantMode: %s", quantMode);
    bool isValidquantMode =
        strncmp(quantMode, "dynamic", strlen("dynamic")) == 0 || strncmp(quantMode, "static", strlen("static")) == 0;
    OP_CHECK(
        isValidquantMode, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Got quantMode neither \"dynamic\" nor \"static\"."),
        return false);
    isDynQuant = IsDynamicQuant(quantMode);
    return true;
}

static bool CheckRoundMode(const char* roundMode, int64_t dstType)
{
    // 检查参数 roundMode 是否合法
    const std::string mode = std::string(roundMode);
    if (dstType == op::DataType::DT_HIFLOAT8) {
        if (mode != "round" && mode != "hybrid") {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "check roundMode failed, roundMode[%s] not in ['round','hybrid'] for hifloat8.", mode.c_str());
            return false;
        }
    } else {
        if (mode != "rint") {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "check roundMode failed, roundMode[%s] not in ['rint'] for float8_e5m2/float8_e4m3fn/int8.",
                mode.c_str());
            return false;
        }
    }
    return true;
}

static inline bool CheckPlatform()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(
        socVersion == SocVersion::ASCEND910_95,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Support for %s is not implemented.", op::ToString(socVersion).GetString()),
        return false);
    return true;
}

static bool CheckShape(
    const aclTensor* self, const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional,
    const aclTensor* y, const aclTensor* outScaleOptional, bool isDynQuant)
{
    auto selfShape = self->GetViewShape();
    auto yShape = y->GetViewShape();
    if (isDynQuant) {
        auto outScaleShape = outScaleOptional->GetViewShape();
        size_t selfDimNums = selfShape.GetDimNum();
        size_t outScaleDimNums = outScaleShape.GetDimNum();
        OP_CHECK(
            selfDimNums >= DIM_NUM_TWO && selfDimNums <= DIM_NUM_EIGHT,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input self Dims is %zu, should be 2-8D.", selfDimNums), return false);
        OP_CHECK(
            outScaleDimNums >= 1 && outScaleDimNums <= DIM_NUM_SEVEN,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output outScaleOptional Dims is %zu, should be 1-7D.", outScaleDimNums),
            return false);
        for (size_t i = 0; i < selfDimNums - static_cast<size_t>(1); i++) {
            OP_CHECK(
                (selfShape.GetDim(i) == outScaleShape.GetDim(i)),
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "The shape of outScaleOptional matches the shape of self across all dimensions except for the last "
                    "dimension."),
                return false);
        }
    } else {
        OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    if (inputScaleOptional != nullptr) {
        auto scaleShape = inputScaleOptional->GetViewShape();
        if (scaleShape.GetDimNum() != 1) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "InputScaleOptional Dims is %zu, only supported 1D.", scaleShape.GetDimNum());
            return false;
        }
    }
    if (inputOffsetOptional != nullptr) {
        auto offsetShape = inputOffsetOptional->GetViewShape();
        if (offsetShape.GetDimNum() != 1) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "inputOffsetOptional Dims is %zu, only supported 1D.",
                offsetShape.GetDimNum());
            return false;
        }
        if (inputScaleOptional != nullptr) {
            auto scaleShape = inputScaleOptional->GetViewShape();
            OP_CHECK(
                (scaleShape.GetDim(0) == offsetShape.GetDim(0)),
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "The shape of inputOffsetOptional matches the shape of inputScaleOptional."),
                return false);
        }
    }
    OP_CHECK_SHAPE_NOT_EQUAL(self, y, return false);
    return true;
}

static bool CheckDtypeCompatible(const aclTensor* x, const aclTensor* scale)
{
    if (scale->GetDataType() == op::DataType::DT_FLOAT) {
        return true;
    } else if (x->GetDataType() == op::DataType::DT_FLOAT16 && scale->GetDataType() == op::DataType::DT_BF16) {
        return false;
    } else if (x->GetDataType() == op::DataType::DT_BF16 && scale->GetDataType() == op::DataType::DT_FLOAT16) {
        return false;
    }
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional,
    const char* approximate, int64_t dstType, const aclTensor* y, const aclTensor* outScaleOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, Y_DTYPE_SUPPORT_LIST, return false);

    if (inputScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputScaleOptional, SELF_DTYPE_SUPPORT_LIST, return false);
        if (!CheckDtypeCompatible(self, inputScaleOptional)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "dtype of input self:%s is not compatible with inputScaleOptional:%s.",
                op::ToString(self->GetDataType()).GetString(),
                op::ToString(inputScaleOptional->GetDataType()).GetString());
            return false;
        }
    }

    if (inputOffsetOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputOffsetOptional, SELF_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SAME(inputScaleOptional, inputOffsetOptional, return false);
    }

    if (outScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(outScaleOptional, op::DataType::DT_FLOAT, return false);
    }

    const std::string approximateGelu = std::string(approximate);
    OP_CHECK(
        approximateGelu == "none" || approximateGelu == "tanh",
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "expected approximate equals 'none' or 'tanh', get: %s", approximateGelu.c_str()),
        return false);

    OP_CHECK(
        static_cast<int64_t>(y->GetDataType()) == dstType,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "dstType:%ld(%s) is must be the same as y dtype[%s].", dstType,
            op::ToString(static_cast<op::DataType>(dstType)).GetString(), op::ToString(y->GetDataType()).GetString()),
        return false);

    return true;
}

static bool CheckOptCombValid(
    const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional, const aclTensor* outScaleOptional,
    bool isDynQuant)
{
    bool s1Exist = (nullptr != inputScaleOptional);
    bool s2Exist = (nullptr != inputOffsetOptional);
    bool outscaleExist = (nullptr != outScaleOptional);
    OP_LOGD(
        "Got aclnnGeluQuant opt input combination (%d,%d,%d), isDyn=%d", s1Exist, s2Exist, outscaleExist, isDynQuant);

    if (isDynQuant) {
        bool validDynQuantComb = (s1Exist && s2Exist && outscaleExist) || (s1Exist && (!s2Exist) && outscaleExist) ||
                                 ((!s1Exist) && (!s2Exist) && outscaleExist);
        OP_CHECK(
            (validDynQuantComb),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid Optional input, output combination in dynamic quant mode"),
            return false);
    } else {
        bool validStcQuantComb =
            (s1Exist && s2Exist && (!outscaleExist)) || (s1Exist && (!s2Exist) && (!outscaleExist));
        OP_CHECK(
            (validStcQuantComb),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid Optional input, output combination in static quant mode"),
            return false);
    }
    return true;
}

inline static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional,
    const char* approximate, const char* quantMode, const char* roundMode, int64_t dstType, const aclTensor* y,
    const aclTensor* outScaleOptional)
{
    // 当前仅支持910_95
    CHECK_RET(CheckPlatform(), ACLNN_ERR_RUNTIME_ERROR);

    // 1. 检查参数 quantMode 是否合法
    bool isDynQuant = true;
    CHECK_RET(quantMode != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(GetQuantMode(quantMode, isDynQuant), ACLNN_ERR_PARAM_INVALID);

    // 2. 查 可选输入组合
    // 3. 检查参数是否为空指针
    CHECK_RET(
        CheckNotNull(self, inputScaleOptional, approximate, roundMode, y, outScaleOptional, isDynQuant),
        ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(
        CheckOptCombValid(inputScaleOptional, inputOffsetOptional, outScaleOptional, isDynQuant),
        ACLNN_ERR_PARAM_INVALID);
    // 4. 检查数据类型是否合法
    CHECK_RET(
        CheckDtypeValid(self, inputScaleOptional, inputOffsetOptional, approximate, dstType, y, outScaleOptional),
        ACLNN_ERR_PARAM_INVALID);

    // 5. 检查shape是否合法
    CHECK_RET(
        CheckShape(self, inputScaleOptional, inputOffsetOptional, y, outScaleOptional, isDynQuant),
        ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckRoundMode(roundMode, dstType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* GetOptTensorContiguous(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnGeluQuantGetWorkspaceSize(
    const aclTensor* self, const aclTensor* inputScaleOptional, const aclTensor* inputOffsetOptional,
    const char* approximate, const char* quantMode, const char* roundMode, int64_t dstType, const aclTensor* y,
    const aclTensor* outScaleOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnGeluQuant,
        DFX_IN(self, inputScaleOptional, inputOffsetOptional, approximate, quantMode, roundMode, dstType),
        DFX_OUT(y, outScaleOptional));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(
        self, inputScaleOptional, inputOffsetOptional, approximate, quantMode, roundMode, dstType, y, outScaleOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    bool hasEmptyTensor = self->IsEmpty() || (inputScaleOptional != nullptr && inputScaleOptional->IsEmpty()) ||
                          (inputOffsetOptional != nullptr && inputOffsetOptional->IsEmpty()) || y->IsEmpty() ||
                          (outScaleOptional != nullptr && outScaleOptional->IsEmpty());
    if (hasEmptyTensor) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Geluquant does not support empty tensor.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // x如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputScaleOptionalContiguous = GetOptTensorContiguous(inputScaleOptional, uniqueExecutor.get());
    auto inputOffsetOptionalContiguous = GetOptTensorContiguous(inputOffsetOptional, uniqueExecutor.get());
    auto result = l0op::GeluQuant(
        selfContiguous, inputScaleOptionalContiguous, inputOffsetOptionalContiguous, approximate, quantMode, roundMode,
        dstType, uniqueExecutor.get());
    const aclTensor* yOut = std::get<0>(result);
    const aclTensor* outScaleOptionalOut = std::get<1>(result);
    CHECK_RET(yOut != nullptr && outScaleOptionalOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("aclnnGeluQuant get output");
    // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult0 = l0op::ViewCopy(yOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (IsDynamicQuant(quantMode)) {
        auto viewCopyResult1 = l0op::ViewCopy(outScaleOptionalOut, outScaleOptional, uniqueExecutor.get());
        CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeluQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGeluQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif