/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_matmul.h"
#include <cstdio>
#include <cstring>

#include "matmul/common/op_host/op_api/fusedmatmul.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "log/log.h"

#include "aclnn_kernels/reshape.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/batch_matmul.h"
#include "matmul/common/op_host/op_api/batch_matmul_util.h"
#include "matmul/common/op_host/op_api/matmul.h"
#include "common/stub/op_api/level0/add.h"
#include "common/stub/op_api/level0/mul.h"

using namespace op;
using namespace Ops::NN;
namespace {
const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BUILT_IN = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static constexpr size_t DIM_LEN_MIN = 2;
static constexpr size_t DIM_LEN_MAX = 3;
static constexpr size_t DIM_LEN_MAX_RELU = 6;
static constexpr float FUSED_MATMUL_DEFAULT_SCALE_VALUE = 1.0F;

static const std::vector<const char*> kAllSupportedOpTypes = {"",         "16cast32",  "add", "mul",
                                                              "gelu_erf", "gelu_tanh", "relu"};
static const std::vector<const char*> kSupportedBiasOpTypes = {"", "16cast32", "relu", "add", "mul"};
static const std::vector<const char*> kSupportedFp32OpTypes = {"", "relu", "add", "mul"};
static const std::vector<const char*> kSupportedX3OpTypes = {"add", "mul"};
static const std::vector<const char*> kSupportedIn16CastOut32OpTypes = {"16cast32"};
static const std::vector<const char*> kSupportedEmptyTensorOpTypes = {"",          "relu", "gelu_erf",
                                                                      "gelu_tanh", "add",  "mul"};
static constexpr int64_t INNER_PRECISE_HIGH_PRECISION = 0;
static constexpr int64_t INNER_PRECISE_HIGH_PERFORMANCE = 1;

bool IsInSupportedOpTypes(const char* fusedOpType, const std::vector<const char*>& types)
{
    if (fusedOpType == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fusedOpType is nullptr");
        return false;
    }
    for (const auto& type : types) {
        if (type && fusedOpType && strcmp(fusedOpType, type) == 0) {
            return true;
        }
    }
    return false;
}

// inner_precise默认为1；2D MatMul和3D BatchMatMul在cubeMathType为USE_FP32_ADD，且fusedOpType为add/mul，
// x/x2/x3均为同一种fp16或bf16类型时，inner_precise取0。
static int64_t GetInnerPrecise(const aclTensor* x, const aclTensor* x2, const aclTensor* x3, const char* fusedOpType,
                               int8_t cubeMathType)
{
    int64_t xDimNum = x->GetViewShape().GetDimNum();
    int64_t x2DimNum = x2->GetViewShape().GetDimNum();
    bool isMatmulOrBatchMatmul = (xDimNum == DIM_LEN_MIN && x2DimNum == DIM_LEN_MIN) ||
                                 (xDimNum == DIM_LEN_MAX && x2DimNum == DIM_LEN_MAX);
    if (isMatmulOrBatchMatmul && cubeMathType == USE_FP32_ADD &&
        IsInSupportedOpTypes(fusedOpType, kSupportedX3OpTypes) &&
        (x->GetDataType() == DataType::DT_FLOAT16 || x->GetDataType() == DataType::DT_BF16) &&
        x->GetDataType() == x2->GetDataType() && x3 != nullptr && x3->GetDataType() == x->GetDataType()) {
        return INNER_PRECISE_HIGH_PRECISION;
    }
    return INNER_PRECISE_HIGH_PERFORMANCE;
}

// 校验fusedOpType是否合法
bool CheckFusedOpType(const char* fusedOpType)
{
    if (!IsInSupportedOpTypes(fusedOpType, kAllSupportedOpTypes)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "fusedOpType must be in the type of /16cast32/add/mul/gelu_erf/gelu_tanh/relu");
        return false;
    }
    return true;
}

// 校验是否为空指针
bool CheckNotNull(const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3,
                  const char* fusedOpType, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(x2, return false);
    if (bias != nullptr && !IsInSupportedOpTypes(fusedOpType, kSupportedBiasOpTypes)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "bias is not supported for the current fusedOpType");
        return false;
    }
    if (IsInSupportedOpTypes(fusedOpType, kSupportedX3OpTypes)) {
        OP_CHECK_NULL(x3, return false);
    }
    OP_CHECK_NULL(y, return false);
    return true;
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
    if (cubeMathType != USE_HF32 && promoteType == DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fusedmatmul only supports bf16/fp16/hf32, does not support fp32.");
        return false;
    }
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

// 校验是否不为NZ格式
static bool CheckFormat(const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3,
                        const aclTensor* y)
{
    if (x->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ || x2->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ ||
        y->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format does not support NZ");
        return false;
    }

    if (bias != nullptr) {
        if (bias->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format does not support NZ");
            return false;
        }
    }

    if (x3 != nullptr) {
        if (x3->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format does not support NZ");
            return false;
        }
    }
    return true;
}
// 校验数据类型是否合法
static bool CheckDtypeValid(const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3,
                            const char* fusedOpType, const aclTensor* y)
{
    auto dtypeSupportList = IsInSupportedOpTypes(fusedOpType, kSupportedFp32OpTypes) ? DTYPE_SUPPORT_LIST_BUILT_IN :
                                                                                       DTYPE_SUPPORT_LIST;
    // 检查x的数据类型是否在fusedmatmul算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, dtypeSupportList, return false);
    // 检查x2的数据类型是否在fusedmatmul算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, dtypeSupportList, return false);
    // x和x2数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(x2, x->GetDataType(), return false);
    if (IsInSupportedOpTypes(fusedOpType, kSupportedIn16CastOut32OpTypes)) {
        // y fp32 x1=x2=fp16|bf16
        std::initializer_list<op::DataType> yDtypeSupportList{op::DataType::DT_FLOAT};
        OP_CHECK_DTYPE_NOT_SUPPORT(y, yDtypeSupportList, return false);
    } else {
        // 检查y的数据类型是否在fusedmatmul算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(y, dtypeSupportList, return false);
        // x和y数据类型必须一样
        OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);
    }
    if (bias != nullptr) {
        std::initializer_list<op::DataType> biasDtypeSupportList{x->GetDataType(), op::DataType::DT_FLOAT};
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, biasDtypeSupportList, return false);
    }
    if (x3 != nullptr) {
        // 检查x3的数据类型是否在fusedmatmul算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(x3, dtypeSupportList, return false);
        OP_CHECK_DTYPE_NOT_MATCH(x3, x->GetDataType(), return false);
    }
    return true;
}

static bool CheckNoBroadcastBatchShape(const aclTensor* x, const aclTensor* x2, const aclTensor* y,
                                       const char* fusedOpType)
{
    const auto& xShape = x->GetViewShape();
    const auto& x2Shape = x2->GetViewShape();
    const auto& yShape = y->GetViewShape();
    const char* opTypeForLog = strcmp(fusedOpType, "") == 0 ? "empty" : fusedOpType;
    size_t batchDimNum = xShape.GetDimNum() - DIM_LEN_MIN;
    for (size_t i = 0; i < batchDimNum; ++i) {
        if (xShape[i] != x2Shape[i] || xShape[i] != yShape[i]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s op type only supports no-broadcast batch shape, but x batch dim[%zu] is %ld, "
                    "x2 batch dim[%zu] is %ld, y batch dim[%zu] is %ld.",
                    opTypeForLog, i, xShape[i], i, x2Shape[i], i, yShape[i]);
            return false;
        }
    }
    return true;
}

static bool CheckGeluBatchShape(const aclTensor* x)
{
    const auto& xShape = x->GetViewShape();
    if (xShape.GetDimNum() == DIM_LEN_MIN) {
        return true;
    }
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
        "aclnnFusedMatmul", "x1", FormatString("%zuD", xShape.GetDimNum()).c_str(),
        FormatString("The shape dim of %s must be %zuD for gelu op type", "x1", DIM_LEN_MIN).c_str());
    return false;
}

static inline bool CheckX3Shape(const aclTensor* x3, const aclTensor* y)
{
    // check x3 dims number is 2 or 3(bmm)
    OP_CHECK_MAX_DIM(x3, DIM_LEN_MAX, return false);
    OP_CHECK_MIN_DIM(x3, DIM_LEN_MIN, return false);

    // mm or bmm
    if (y->GetViewShape().GetDimNum() == DIM_LEN_MIN && x3->GetViewShape() != y->GetViewShape()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnFusedMatmul", "x3, y",
                                               FormatString("%s, %s", op::ToString(x3->GetViewShape()).GetString(),
                                                            op::ToString(y->GetViewShape()).GetString())
                                                   .c_str(),
                                               "The shape of x3 and y must be the same when y is 2D");
        return false;
    }

    if (x3->GetViewShape().GetDimNum() == DIM_LEN_MAX && x3->GetViewShape()[0] != 1 &&
        x3->GetViewShape()[0] != y->GetViewShape()[0]) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnFusedMatmul", "x3 batch, y batch",
            FormatString("%ld, %ld", x3->GetViewShape()[0], y->GetViewShape()[0]).c_str(),
            "The batch of x3 must be 1 or same as the batch of y");
        return false;
    }

    if (y->GetViewShape().GetDimNum() == DIM_LEN_MAX) {
        int64_t yM = y->GetViewShape()[1];
        int64_t yN = y->GetViewShape()[2];
        int64_t x3DimNum = x3->GetViewShape().GetDimNum();
        int64_t x3M = x3->GetViewShape()[x3DimNum - 2];
        int64_t x3N = x3->GetViewShape()[x3DimNum - 1];
        if (x3M != yM || x3N != yN) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnFusedMatmul", "x3 M/N, y M/N",
                                                   FormatString("[%ld,%ld], [%ld,%ld]", x3M, x3N, yM, yN).c_str(),
                                                   "The last two dimensions of x3 must match the M/N dimensions of y");
            return false;
        }
    }
    return true;
}

static bool CheckBiasShape(const aclTensor* bias)
{
    if (bias == nullptr) {
        return true;
    }
    size_t biasDimNum = bias->GetViewShape().GetDimNum();
    if (biasDimNum != 1 && biasDimNum != DIM_LEN_MIN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dimension of bias cannot be larger than 2");
        return false;
    }
    return true;
}

static bool CheckKZeroBias(const aclTensor* x, const aclTensor* bias)
{
    if (bias == nullptr) {
        return true;
    }
    const auto& xShape = x->GetViewShape();
    const size_t xDimNum = xShape.GetDimNum();
    if (xDimNum == 0 || xShape[xDimNum - 1] != 0) {
        return true;
    }
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
        "aclnnFusedMatmul", "x1", op::ToString(xShape).GetString(),
        "When optional parameter bias exists, the K-axis of x1 must be a positive number");
    return false;
}

static inline bool CheckShape(const aclTensor* x, const aclTensor* x2, const aclTensor* x3, const char* fusedOpType,
                              const aclTensor* y)
{
    bool isReluOrEmpty = (strcmp(fusedOpType, "relu") == 0 || strcmp(fusedOpType, "") == 0);
    bool isGelu = (strcmp(fusedOpType, "gelu_erf") == 0 || strcmp(fusedOpType, "gelu_tanh") == 0);
    size_t dimLenMax = isReluOrEmpty ? DIM_LEN_MAX_RELU : DIM_LEN_MAX;
    // check x dims number
    OP_CHECK_MAX_DIM(x, dimLenMax, return false);
    OP_CHECK_MIN_DIM(x, DIM_LEN_MIN, return false);

    // check x2 dims number
    OP_CHECK_MAX_DIM(x2, dimLenMax, return false);
    OP_CHECK_MIN_DIM(x2, DIM_LEN_MIN, return false);

    // check dimensions of x and x2 must be same
    if (x2->GetViewShape().GetDimNum() != x->GetViewShape().GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x dimension and x2 dimension should be the same, but x dimension is %d, x2 dimension is %d.",
                x->GetViewShape().GetDimNum(), x2->GetViewShape().GetDimNum());
        return false;
    }

    // check dimensions of x and y must be same
    if (y->GetViewShape().GetDimNum() != x->GetViewShape().GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x dimension and y dimension should be the same, but x dimension is %d, y dimension is %d.",
                x->GetViewShape().GetDimNum(), y->GetViewShape().GetDimNum());
        return false;
    }

    CHECK_RET(CheckNoBroadcastBatchShape(x, x2, y, fusedOpType), false);
    if (isGelu) {
        CHECK_RET(CheckGeluBatchShape(x), false);
    }
    if (x3 != nullptr) {
        CHECK_RET(CheckX3Shape(x3, y), false);
    }
    return true;
}

static bool HasNonDefaultScale(const aclScalar* scaleOptional)
{
    return scaleOptional != nullptr && scaleOptional->ToFloat() != FUSED_MATMUL_DEFAULT_SCALE_VALUE;
}

static bool CheckScaleParam(const aclScalar* scaleOptional, const char* name)
{
    if (scaleOptional == nullptr) {
        return true;
    }
    const auto scaleType = scaleOptional->GetDataType();
    if (scaleType != DataType::DT_FLOAT && scaleType != DataType::DT_FLOAT16 && scaleType != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s only supports float32, float16 or bfloat16.", name);
        return false;
    }
    return true;
}

static bool CheckScaleParams(const aclScalar* alphaOptional, const aclScalar* betaOptional)
{
    return CheckScaleParam(alphaOptional, "alphaOptional") && CheckScaleParam(betaOptional, "betaOptional");
}

static bool CheckFmmWithScaleAddScenario(const aclTensor* x, const aclTensor* x2, const aclTensor* bias,
                                         const aclTensor* x3, const char* fusedOpType, const aclTensor* y)
{
    if (std::strcmp(fusedOpType, "add") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha or beta scale is only supported when fusedOpType is add.");
        return false;
    }
    if (!IsNpuArch3510Series()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha or beta scale is only supported on Ascend 950.");
        return false;
    }
    if (bias != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The scale_add scenario does not support bias.");
        return false;
    }
    if (x3 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The scale_add scenario requires x3.");
        return false;
    }
    auto dataType = x->GetDataType();
    if ((dataType != DataType::DT_FLOAT16 && dataType != DataType::DT_BF16) || x2->GetDataType() != dataType ||
        x3->GetDataType() != dataType || y->GetDataType() != dataType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The scale_add scenario requires x, x2, x3 and y to have the same FP16 or BF16 dtype.");
        return false;
    }
    if (IsTransposeLastTwoDims(x) || IsTransposeLastTwoDims(x2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The scale_add scenario does not support transposed x or x2; expected x[B,M,K] and "
                "x2[B,K,N].");
        return false;
    }

    const auto& xShape = x->GetViewShape();
    const auto& x2Shape = x2->GetViewShape();
    const auto& x3Shape = x3->GetViewShape();
    const auto& yShape = y->GetViewShape();
    if (xShape.GetDimNum() != DIM_LEN_MAX || x2Shape.GetDimNum() != DIM_LEN_MAX || x3Shape.GetDimNum() != DIM_LEN_MAX ||
        yShape.GetDimNum() != DIM_LEN_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The scale_add scenario requires 3D x, x2, x3 and y.");
        return false;
    }

    bool batchMatched = xShape[0] == x2Shape[0] && xShape[0] == x3Shape[0] && xShape[0] == yShape[0];
    bool matmulMatched = xShape[2] == x2Shape[1];
    bool outputMatched = xShape[1] == x3Shape[1] && xShape[1] == yShape[1] && x2Shape[2] == x3Shape[2] &&
                         x2Shape[2] == yShape[2];
    bool positiveShape = xShape[0] > 0 && xShape[1] > 0 && xShape[2] > 0 && x2Shape[2] > 0;
    if (!batchMatched || !matmulMatched || !outputMatched || !positiveShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The scale_add scenario requires x[B,M,K] * x2[B,K,N] + x3[B,M,N] -> y[B,M,N].");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3,
                               const aclScalar* alphaOptional, const aclScalar* betaOptional, const char* fusedOpType,
                               int8_t cubeMathType, const aclTensor* y)
{
    // 检验fusedOpType类型是否合法
    CHECK_RET(CheckFusedOpType(fusedOpType), ACLNN_ERR_PARAM_INVALID);
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, x2, bias, x3, fusedOpType, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查A和B是否为2维，且是否满足matmul shape MN 与传入的x3 shape Mn相同
    CHECK_RET(CheckShape(x, x2, x3, fusedOpType, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckKZeroBias(x, bias), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckBiasShape(bias), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在支持的数据类型之内
    CHECK_RET(CheckDtypeValid(x, x2, bias, x3, fusedOpType, y), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查Format是否支持
    CHECK_RET(CheckFormat(x, x2, bias, x3, y), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查cubeMathType
    CHECK_RET(CheckMathType(x, x2, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckScaleParams(alphaOptional, betaOptional), ACLNN_ERR_PARAM_INVALID);
    if (HasNonDefaultScale(alphaOptional) || HasNonDefaultScale(betaOptional)) {
        CHECK_RET(CheckFmmWithScaleAddScenario(x, x2, bias, x3, fusedOpType, y), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static constexpr int64_t SMALL_THRESHOLD = 16;
static bool IsX3NoBatch(const aclTensor* x, const aclTensor* x2, const aclTensor* x3)
{
    if (x3 == nullptr) {
        return false;
    }
    if (x->GetViewShape().GetDimNum() != static_cast<int64_t>(DIM_LEN_MAX) ||
        x2->GetViewShape().GetDimNum() != static_cast<int64_t>(DIM_LEN_MAX)) {
        return false;
    }
    int64_t dimNum = x3->GetViewShape().GetDimNum();
    if (dimNum == static_cast<int64_t>(DIM_LEN_MIN)) {
        return true;
    }
    if (dimNum == static_cast<int64_t>(DIM_LEN_MAX) && x3->GetViewShape()[0] == 1) {
        return true;
    }
    return false;
}

static const aclTensor* BuildSplitMatmulOp(const aclTensor* x, const aclTensor* x2, const aclTensor* bias,
                                           const aclTensor* y, bool is2D, int8_t cubeMathType, aclOpExecutor* executor)
{
    if (is2D) {
        OP_LOGI("FusedMatMul split to MatmulCommonProcess.");
        MmOpInfo splitMmOpInfo;
        return MatmulCommonProcess(x, x2, bias, y, cubeMathType, splitMmOpInfo, executor, false, true);
    }
    OP_LOGI("FusedMatMul split to ExecBmmOpWithBiasV2.");
    return ExecBmmOpWithBiasV2(x, x2, bias, y, cubeMathType, executor);
}

static const aclTensor* BuildSplitEpilogueOp(const aclTensor* splitMmOut, const aclTensor* contiguousX3,
                                             const aclTensor* y, const MmOpInfo& mmOpInfo, const char* fusedOpType,
                                             aclOpExecutor* executor)
{
    auto contiguousMmOut = l0op::Contiguous(splitMmOut, executor);
    CHECK_RET(contiguousMmOut != nullptr, nullptr);
    const aclTensor* fusionOut = nullptr;
    if (contiguousX3 != nullptr) {
        if (contiguousMmOut->GetDataType() == DataType::DT_FLOAT && contiguousX3->GetDataType() != DataType::DT_FLOAT) {
            contiguousX3 = l0op::Cast(contiguousX3, DataType::DT_FLOAT, executor);
            CHECK_RET(contiguousX3 != nullptr, nullptr);
        }
        int64_t x3Numel = contiguousX3->GetViewShape().GetShapeSize();
        int64_t mmOutNumel = contiguousMmOut->GetViewShape().GetShapeSize();
        if (x3Numel <= 0 || mmOutNumel % x3Numel != 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                "aclnnFusedMatmul", "mmOutNumel, x3Numel", FormatString("%ld, %ld", mmOutNumel, x3Numel).c_str(),
                "mmOutNumel must be divisible by x3Numel in FusedMatMul split epilogue");
            return nullptr;
        }
        int64_t batchDim = mmOutNumel / x3Numel;
        auto flatMmOut = l0op::Reshape(contiguousMmOut, {batchDim, x3Numel}, executor);
        CHECK_RET(flatMmOut != nullptr, nullptr);
        auto flatX3 = l0op::Reshape(contiguousX3, {x3Numel}, executor);
        CHECK_RET(flatX3 != nullptr, nullptr);
        if (std::strcmp(fusedOpType, "add") == 0) {
            fusionOut = l0op::Add(flatMmOut, flatX3, executor);
        } else {
            fusionOut = l0op::Mul(flatMmOut, flatX3, executor);
        }
        CHECK_RET(fusionOut != nullptr, nullptr);
    } else {
        fusionOut = contiguousMmOut;
    }
    auto castOut = l0op::Cast(fusionOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);
    auto matReshape = l0op::Reshape(castOut, y->GetViewShape(), executor);
    CHECK_RET(matReshape != nullptr, nullptr);
    return matReshape;
}

/*
                 x               x2
                 |               |
            contiguous       contiguous
                 |               |
               cast             cast
                 |               |
                  \              /
                    fusedmatmul_op - add/mul/gelu_erf/gelu_tanh - contiguous - mat3
                          |
                        cast
                          |
                       output
*/
static const aclTensor* BuildSplitFusedMatMulGraph(const aclTensor* x, const aclTensor* x2, const aclTensor* bias,
                                                   const aclTensor* x3, const aclTensor* y, const MmOpInfo& mmOpInfo,
                                                   const char* fusedOpType, int8_t cubeMathType, bool isHighPrecision,
                                                   aclOpExecutor* executor)
{
    bool is2D = (x->GetViewShape().GetDimNum() == static_cast<int64_t>(DIM_LEN_MIN) &&
                 x2->GetViewShape().GetDimNum() == static_cast<int64_t>(DIM_LEN_MIN));
    const aclTensor* splitMmDesc = y;
    if (isHighPrecision) {
        splitMmDesc = executor->AllocTensor(y->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);
        CHECK_RET(splitMmDesc != nullptr, nullptr);
    }
    auto splitMmOut = BuildSplitMatmulOp(x, x2, bias, splitMmDesc, is2D, cubeMathType, executor);
    CHECK_RET(splitMmOut != nullptr, nullptr);
    auto contiguousX3 = x3;
    if (contiguousX3 != nullptr) {
        contiguousX3 = l0op::Contiguous(x3, executor);
        CHECK_RET(contiguousX3 != nullptr, nullptr);
        contiguousX3 = l0op::ReFormat(contiguousX3, op::Format::FORMAT_ND);
        CHECK_RET(contiguousX3 != nullptr, nullptr);
    }
    return BuildSplitEpilogueOp(splitMmOut, contiguousX3, y, mmOpInfo, fusedOpType, executor);
}

static const aclTensor* BuildDirectFusedMatMulOutput(const aclTensor* x, const aclTensor* x2, const aclTensor* bias,
                                                     const aclTensor* x3, float alpha, float beta, const aclTensor* y,
                                                     const MmOpInfo& mmOpInfo, const char* fusedOpType,
                                                     int64_t innerPrecise, aclOpExecutor* executor)
{
    const aclTensor* mmOut = nullptr;
    if (std::strcmp(fusedOpType, "scale_add") == 0) {
        mmOut = l0op::FusedMatMulWithScaleAddNd(x, x2, bias, x3, alpha, beta, mmOpInfo.shapeInfo.transposeX1,
                                                mmOpInfo.shapeInfo.transposeX2, mmOpInfo.enableHf32, innerPrecise,
                                                executor);
    } else if (std::strcmp(fusedOpType, "16cast32") == 0) {
        mmOut = l0op::FusedMatMul16Cast32(x, x2, bias, x3, mmOpInfo.shapeInfo.transposeX1,
                                          mmOpInfo.shapeInfo.transposeX2, mmOpInfo.enableHf32, fusedOpType,
                                          innerPrecise, executor);
    } else {
        mmOut = l0op::FusedMatMulNd(x, x2, bias, x3, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.shapeInfo.transposeX2,
                                    mmOpInfo.enableHf32, fusedOpType, innerPrecise, executor);
    }
    CHECK_RET(mmOut != nullptr, nullptr);
    // output cast
    auto castOut = l0op::Cast(mmOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);
    // Reshape to out shape
    auto matReshape = l0op::Reshape(castOut, y->GetViewShape(), executor);
    CHECK_RET(matReshape != nullptr, nullptr);
    return matReshape;
}

static const aclTensor* BuildDirectFusedMatMulGraph(const aclTensor* x, const aclTensor* x2, const aclTensor* bias,
                                                    const aclTensor* x3, float alpha, float beta, const aclTensor* y,
                                                    MmOpInfo mmOpInfo, const char* fusedOpType, int8_t cubeMathType,
                                                    bool hasScaleInput, aclOpExecutor* executor)
{
    auto selfCastOut = x;
    bool selfCastRes = ContiguousAndCast(x, selfCastOut, mmOpInfo.shapeInfo.transposeX1,
                                         mmOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCastRes, nullptr);
    // 右输入非连续转连续
    auto mat2CastOut = x2;
    bool mat2CastRes = ContiguousAndCast(x2, mat2CastOut, mmOpInfo.shapeInfo.transposeX2,
                                         mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastRes, nullptr);
    // bias非连续转连续以及转换dtype
    auto contiguousBias = bias;
    if (contiguousBias != nullptr) {
        contiguousBias = ContiguousBias(x, bias, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }
    auto selfReshapeOutput = selfCastOut;
    auto mat2ReshapeOutput = mat2CastOut;
    bool ifKEqual1 = false;
    if (x->GetViewShape().GetDimNum() > DIM_LEN_MIN) {
        // scale_add tiling currently does not support transposed x2. Keep the original x2 layout when N=1.
        const bool enableNEqual1Transpose = !hasScaleInput;
        CHECK_RET(ProcessEqual1Cases(selfCastOut, mat2CastOut, mmOpInfo, contiguousBias, mmOpInfo.shapeInfo.transposeX1,
                                     mmOpInfo.shapeInfo.transposeX2, selfReshapeOutput, mat2ReshapeOutput, executor,
                                     ifKEqual1, enableNEqual1Transpose) != -1,
                  nullptr);
    } else {
        CHECK_RET(ProcessSpecialCases(selfCastOut, mat2CastOut, mmOpInfo, contiguousBias, selfReshapeOutput,
                                      mat2ReshapeOutput, executor, ifKEqual1) != -1,
                  nullptr);
    }
    // 全部转成ND
    selfCastOut = l0op::ReFormat(selfReshapeOutput, op::Format::FORMAT_ND);
    CHECK_RET(selfCastOut != nullptr, nullptr);
    mat2CastOut = l0op::ReFormat(mat2ReshapeOutput, op::Format::FORMAT_ND);
    CHECK_RET(mat2CastOut != nullptr, nullptr);
    // x3非连续转连续
    auto contiguousX3 = x3;
    if (contiguousX3 != nullptr) {
        contiguousX3 = l0op::Contiguous(x3, executor);
        CHECK_RET(contiguousX3 != nullptr, nullptr);
        contiguousX3 = l0op::ReFormat(contiguousX3, op::Format::FORMAT_ND);
        CHECK_RET(contiguousX3 != nullptr, nullptr);
    }
    int64_t innerPrecise = GetInnerPrecise(x, x2, x3, fusedOpType, cubeMathType);
    const char* internalFusedOpType = hasScaleInput ? "scale_add" : fusedOpType;
    return BuildDirectFusedMatMulOutput(selfCastOut, mat2CastOut, contiguousBias, contiguousX3, alpha, beta, y,
                                        mmOpInfo, internalFusedOpType, innerPrecise, executor);
}

static const aclTensor* BuildFusedMatMulGraph(const aclTensor* x, const aclTensor* x2, const aclTensor* bias,
                                              const aclTensor* x3, float alpha, float beta, const aclTensor* y,
                                              const char* fusedOpType, int8_t cubeMathType, bool hasScaleInput,
                                              aclOpExecutor* executor)
{
    // 空tensor 处理，对于非16Cast32放开空tensor
    bool allowEmptyTensor = IsInSupportedOpTypes(fusedOpType, kSupportedEmptyTensorOpTypes);
    if (!allowEmptyTensor && (x->IsEmpty() || x2->IsEmpty())) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedMatmul does not support empty tensor for this fusedOpType");
        return nullptr;
    }
    // 解析当前规格matmulop支持的dtype、format能力
    MmOpInfo mmOpInfo = GetMatmulOpInfo(x, x2, nullptr, nullptr, cubeMathType);
    // 输出fp32
    if (IsInSupportedOpTypes(fusedOpType, kSupportedIn16CastOut32OpTypes)) {
        mmOpInfo.ori_info.output_dtype = DataType::DT_FLOAT;
    }
    int64_t innerPrecise = GetInnerPrecise(x, x2, x3, fusedOpType, cubeMathType);
    // Split small cases through common MatMul/BMM graph builders.
    int64_t selfDimNum = x->GetViewShape().GetDimNum();
    int64_t mat2DimNum = x2->GetViewShape().GetDimNum();
    bool isHighPrecisionBmm = innerPrecise == INNER_PRECISE_HIGH_PRECISION && selfDimNum == DIM_LEN_MAX &&
                              mat2DimNum == DIM_LEN_MAX;
    if (!hasScaleInput && IsNpuArch3510Series() && IsInSupportedOpTypes(fusedOpType, kSupportedX3OpTypes) &&
        (innerPrecise == INNER_PRECISE_HIGH_PERFORMANCE || isHighPrecisionBmm)) {
        const auto& selfShape = x->GetViewShape();
        const auto& mat2Shape = x2->GetViewShape();
        int64_t realM = selfShape[selfDimNum - 2];
        int64_t realK = selfShape[selfDimNum - 1];
        int64_t realN = mat2Shape[mat2DimNum - 1];
        bool needSplit = (realK == 1) || (realN < SMALL_THRESHOLD || realM < SMALL_THRESHOLD) || IsX3NoBatch(x, x2, x3);
        if (needSplit) {
            return BuildSplitFusedMatMulGraph(x, x2, bias, x3, y, mmOpInfo, fusedOpType, cubeMathType,
                                              isHighPrecisionBmm, executor);
        }
    }
    return BuildDirectFusedMatMulGraph(x, x2, bias, x3, alpha, beta, y, mmOpInfo, fusedOpType, cubeMathType,
                                       hasScaleInput, executor);
}

} // namespace

aclnnStatus aclnnFusedMatmulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                             const aclTensor* x3, const char* fusedOpType, int8_t cubeMathType,
                                             const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedMatmul, DFX_IN(x1, x2, bias, x3, fusedOpType, cubeMathType), DFX_OUT(y));
    OP_LOGW("aclnnFusedMatmul is being phased out. Use the fused_matmul Torch API or "
            "aclnnFusedMatmulV2 instead.");
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 固定写法，参数检查
    auto ret = CheckParams(x1, x2, bias, x3, nullptr, nullptr, fusedOpType, cubeMathType, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 构造fusedmatmul计算器
    auto matmulOut = BuildFusedMatMulGraph(x1, x2, bias, x3, FUSED_MATMUL_DEFAULT_SCALE_VALUE,
                                           FUSED_MATMUL_DEFAULT_SCALE_VALUE, y, fusedOpType, cubeMathType, false,
                                           uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (matmulOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto viewCopyResult = l0op::ViewCopy(matmulOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 获取workspace
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedMatmul);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFusedMatmulV2GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                               const aclTensor* x3, const aclScalar* alphaOptional,
                                               const aclScalar* betaOptional, const char* fusedOpType,
                                               int8_t cubeMathType, const aclTensor* y, uint64_t* workspaceSize,
                                               aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedMatmulV2, DFX_IN(x1, x2, bias, x3, alphaOptional, betaOptional, fusedOpType, cubeMathType),
                   DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto ret = CheckParams(x1, x2, bias, x3, alphaOptional, betaOptional, fusedOpType, cubeMathType, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const float alphaValue = alphaOptional == nullptr ? FUSED_MATMUL_DEFAULT_SCALE_VALUE : alphaOptional->ToFloat();
    const float betaValue = betaOptional == nullptr ? FUSED_MATMUL_DEFAULT_SCALE_VALUE : betaOptional->ToFloat();
    const bool hasScaleInput = HasNonDefaultScale(alphaOptional) || HasNonDefaultScale(betaOptional);
    auto matmulOut = BuildFusedMatMulGraph(x1, x2, bias, x3, alphaValue, betaValue, y, fusedOpType, cubeMathType,
                                           hasScaleInput, uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (matmulOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    CHECK_RET(l0op::ViewCopy(matmulOut, y, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedMatmulV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedMatmulV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
