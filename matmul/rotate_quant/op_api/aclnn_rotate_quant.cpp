/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_rotate_quant.h"
#include "rotate_quant.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "matmul/common/op_host/log_format_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "log/log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;
using namespace Ops::NN;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const char* const kOpName = "aclnnRotateQuantGetWorkspaceSize";
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr int64_t DIM_MIN = 2;
static constexpr int64_t N_DIM_ALIGNMENT = 8;
static constexpr double ZERO = 0.0;
static constexpr double SIX = 6.0;
static constexpr double TWELVE = 12.0;
constexpr uint32_t MAX_X_DIM_NUM = 7;
constexpr uint32_t MIN_ROT_DIM_NUM = 2;
constexpr uint32_t MAX_ROT_DIM_NUM = 3;
constexpr int64_t MX_SCALE_CEIL_NUM = 64;

static const std::initializer_list<op::DataType> EMPTY_LIST = {};

static const std::initializer_list<op::DataType> IN_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST_INT = {
    op::DataType::DT_INT32, op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST_MX = {
    op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2};

static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST_INT = {op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST_MX = {op::DataType::DT_FLOAT8_E8M0};

static const std::map<NpuArch, const std::initializer_list<op::DataType>*> SOC_IN_SUPPORT_DTYPES = {
    {NpuArch::DAV_2201, &IN_DTYPE_SUPPORT_LIST}, {NpuArch::DAV_3510, &IN_DTYPE_SUPPORT_LIST}};

static const std::map<NpuArch, const std::initializer_list<op::DataType>*> SOC_OUT_SUPPORT_DTYPES = {
    {NpuArch::DAV_2201, &OUT_DTYPE_SUPPORT_LIST_INT}, {NpuArch::DAV_3510, &OUT_DTYPE_SUPPORT_LIST_MX}};

static const std::map<NpuArch, const std::initializer_list<op::DataType>*> SOC_SCALE_SUPPORT_DTYPES = {
    {NpuArch::DAV_2201, &SCALE_DTYPE_SUPPORT_LIST_INT}, {NpuArch::DAV_3510, &SCALE_DTYPE_SUPPORT_LIST_MX}};

static const std::initializer_list<op::DataType>& GetDtypeSupportList(
    const std::map<NpuArch, const std::initializer_list<op::DataType>*>& socSupportDtypes)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    auto found = socSupportDtypes.find(curArch);
    if (found != socSupportDtypes.end()) {
        return *(found->second);
    }
    OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for arch %u is not implemented", static_cast<uint32_t>(curArch));
    return EMPTY_LIST;
}

static bool CheckNotNull(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* yOut, const aclTensor* scaleOut)
{
    OP_CHECK(
        x != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "x", "nullptr", FormatString("The value of %s cannot be %s", "x", "null").c_str()),
        return false);
    OP_CHECK(
        rotation != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "rotation", "nullptr", FormatString("The value of %s cannot be %s", "rotation", "null").c_str()),
        return false);
    OP_CHECK(
        yOut != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "yOut", "nullptr", FormatString("The value of %s cannot be %s", "yOut", "null").c_str()),
        return false);
    OP_CHECK(
        scaleOut != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "scaleOut", "nullptr", FormatString("The value of %s cannot be %s", "scaleOut", "null").c_str()),
        return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* yOut, const aclTensor* scaleOut)
{
    OP_CHECK(
        CheckType(x->GetDataType(), GetDtypeSupportList(SOC_IN_SUPPORT_DTYPES)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "x", op::ToString(x->GetDataType()).GetString(),
            FormatString(
                "The dtype of %s must be in %s", "x",
                op::ToString(GetDtypeSupportList(SOC_IN_SUPPORT_DTYPES)).GetString())
                .c_str()),
        return false);
    OP_CHECK(
        CheckType(rotation->GetDataType(), GetDtypeSupportList(SOC_IN_SUPPORT_DTYPES)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "rotation", op::ToString(rotation->GetDataType()).GetString(),
            FormatString(
                "The dtype of %s must be in %s", "rotation",
                op::ToString(GetDtypeSupportList(SOC_IN_SUPPORT_DTYPES)).GetString())
                .c_str()),
        return false);
    OP_CHECK(
        x->GetDataType() == rotation->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            kOpName, "x, rotation",
            FormatString(
                "%s, %s", op::ToString(x->GetDataType()).GetString(), op::ToString(rotation->GetDataType()).GetString())
                .c_str(),
            FormatString("The dtypes of %s must be the same", "x and rotation").c_str()),
        return false);

    auto outList = GetDtypeSupportList(SOC_OUT_SUPPORT_DTYPES);
    auto scaleList = GetDtypeSupportList(SOC_SCALE_SUPPORT_DTYPES);
    OP_CHECK(
        std::find(outList.begin(), outList.end(), yOut->GetDataType()) != outList.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "yOut", op::ToString(yOut->GetDataType()).GetString(),
            FormatString("The dtype of %s must be in %s", "yOut", op::ToString(outList).GetString()).c_str()),
        return false);
    OP_CHECK(
        std::find(scaleList.begin(), scaleList.end(), scaleOut->GetDataType()) != scaleList.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "scaleOut", op::ToString(scaleOut->GetDataType()).GetString(),
            FormatString("The dtype of %s must be in %s", "scaleOut", op::ToString(scaleList).GetString()).c_str()),
        return false);
    return true;
}

static bool CheckMxRotationShape(const aclTensor* x, const aclTensor* rotation)
{
    auto xShape = x->GetViewShape();
    auto rotShape = rotation->GetViewShape();

    int64_t N = xShape.GetDim(xShape.GetDimNum() - 1);
    int64_t K = rotShape.GetDim(rotShape.GetDimNum() - 1);
    OP_CHECK(
        K == 32 || K == 64 || K == 128,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "rotation last dim", FormatString("%ld", K).c_str(),
            FormatString("The value of %s must be in %s", "rotation last dim", "[32, 64, 128]").c_str()),
        return false);
    OP_CHECK(
        N % K == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "N", FormatString("%ld", N).c_str(),
            FormatString("The value of %s must be divisible by %s", "N", "K").c_str()),
        return false);

    auto xDimNum = xShape.GetDimNum();
    OP_CHECK(
        xDimNum >= 1 && xDimNum <= MAX_X_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "x", FormatString("%zu", xDimNum).c_str(),
            FormatString("The shape dim of %s must be in range [%d, %d]", "x", 1, MAX_X_DIM_NUM).c_str()),
        return false);
    auto rotDimNum = rotShape.GetDimNum();
    OP_CHECK(
        rotDimNum >= MIN_ROT_DIM_NUM && rotDimNum <= MAX_ROT_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "rotation", FormatString("%zu", rotDimNum).c_str(),
            FormatString("The shape dim of %s must be in range [%d, %d]", "rotation", MIN_ROT_DIM_NUM, MAX_ROT_DIM_NUM)
                .c_str()),
        return false);
    if (rotDimNum == MIN_ROT_DIM_NUM) {
        OP_CHECK(
            rotShape.GetDim(0) == K,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                kOpName, "rotation", op::ToString(rotShape).GetString(),
                FormatString("The shape of %s must be (%ld, %ld)", "rotation", K, K).c_str()),
            return false);
    } else {
        int64_t nKRatio = (K != 0) ? (N / K) : N;
        OP_CHECK(
            rotShape.GetDim(0) == nKRatio && rotShape.GetDim(1) == K,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                kOpName, "rotation", op::ToString(rotShape).GetString(),
                FormatString("The shape of %s must be (%ld, %ld, %ld)", "rotation", nKRatio, K, K).c_str()),
            return false);
    }
    return true;
}

static bool CheckMxOutputShape(const aclTensor* x, const aclTensor* yOut, const aclTensor* scaleOut)
{
    auto xShape = x->GetViewShape();
    int64_t N = xShape.GetDim(xShape.GetDimNum() - 1);

    if (yOut->GetViewShape() != xShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "yOut", op::ToString(yOut->GetViewShape()).GetString(),
            FormatString("The shape of %s must be %s", "yOut", op::ToString(xShape).GetString()).c_str());
        return false;
    }
    op::Shape expectScaleShape = xShape;
    expectScaleShape.SetDim(expectScaleShape.GetDimNum() - 1, (N + MX_SCALE_CEIL_NUM - 1) / MX_SCALE_CEIL_NUM);
    expectScaleShape.AppendDim(2);
    if (scaleOut->GetViewShape() != expectScaleShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "scaleOut", op::ToString(scaleOut->GetViewShape()).GetString(),
            FormatString("The shape of %s must be %s", "scaleOut", op::ToString(expectScaleShape).GetString()).c_str());
        return false;
    }
    return true;
}

static bool CheckMxShape(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* yOut, const aclTensor* scaleOut)
{
    return CheckMxRotationShape(x, rotation) && CheckMxOutputShape(x, yOut, scaleOut);
}

static bool CheckIntRotationShape(const aclTensor* x, const aclTensor* rotation)
{
    auto xShape = x->GetViewShape();
    auto rotShape = rotation->GetViewShape();
    OP_CHECK(
        xShape.GetDimNum() == DIM_MIN,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "x", FormatString("%zu", xShape.GetDimNum()).c_str(),
            FormatString("The shape dim of %s must be %d", "x", DIM_MIN).c_str()),
        return false);
    OP_CHECK(
        rotShape.GetDimNum() == DIM_MIN,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "rotation", FormatString("%zu", rotShape.GetDimNum()).c_str(),
            FormatString("The shape dim of %s must be %d", "rotation", DIM_MIN).c_str()),
        return false);
    int64_t M = xShape.GetDim(0);
    int64_t N = xShape.GetDim(1);
    int64_t K = rotShape.GetDim(1);
    OP_CHECK(
        N % N_DIM_ALIGNMENT == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "N", FormatString("%ld", N).c_str(),
            FormatString("The value of %s must be divisible by %ld", "N", N_DIM_ALIGNMENT).c_str()),
        return false);
    OP_CHECK(
        K != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "rotK", "0", FormatString("The value of %s cannot be %s", "rotK", "0").c_str()),
        return false);
    OP_CHECK(
        N % K == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "N", FormatString("%ld", N).c_str(),
            FormatString("The value of %s must be divisible by %s", "N", "K").c_str()),
        return false);
    OP_CHECK(
        rotShape.GetDim(0) == K,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "rotation", op::ToString(rotShape).GetString(),
            FormatString("The shape of %s must be a square matrix", "rotation").c_str()),
        return false);
    return true;
}

static bool CheckIntOutputShape(const aclTensor* x, const aclTensor* yOut, const aclTensor* scaleOut)
{
    auto xShape = x->GetViewShape();
    int64_t M = xShape.GetDim(0);
    int64_t N = xShape.GetDim(1);

    auto yShape = yOut->GetViewShape();
    auto yDtype = yOut->GetDataType();
    OP_CHECK(
        yShape.GetDimNum() == DIM_MIN,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "yOut", FormatString("%zu", yShape.GetDimNum()).c_str(),
            FormatString("The shape dim of %s must be %d", "yOut", DIM_MIN).c_str()),
        return false);
    OP_CHECK(
        yShape.GetDim(0) == M,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "yOut dim0", FormatString("%ld", yShape.GetDim(0)).c_str(),
            FormatString("The value of %s must equal %ld", "yOut dim0", M).c_str()),
        return false);
    if (yDtype == op::DataType::DT_INT8) {
        OP_CHECK(
            yShape.GetDim(1) == N,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "yOut dim1", FormatString("%ld", yShape.GetDim(1)).c_str(),
                FormatString("The value of %s must equal %ld for INT8 output", "yOut dim1", N).c_str()),
            return false);
    } else if (yDtype == op::DataType::DT_INT32) {
        OP_CHECK(
            yShape.GetDim(1) == N / INT4_NUMS_IN_INT32_SPACE,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "yOut dim1", FormatString("%ld", yShape.GetDim(1)).c_str(),
                FormatString(
                    "The value of %s must equal %ld for INT32 output", "yOut dim1", N / INT4_NUMS_IN_INT32_SPACE)
                    .c_str()),
            return false);
    }

    auto scaleShape = scaleOut->GetViewShape();
    OP_CHECK(
        scaleShape.GetDimNum() == 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            kOpName, "scaleOut", FormatString("%zu", scaleShape.GetDimNum()).c_str(),
            FormatString("The shape dim of %s must be %d", "scaleOut", 1).c_str()),
        return false);
    OP_CHECK(
        scaleShape.GetDim(0) == M,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "scaleOut dim0", FormatString("%ld", scaleShape.GetDim(0)).c_str(),
            FormatString("The value of %s must equal %ld", "scaleOut dim0", M).c_str()),
        return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* rotation, const aclTensor* yOut, const aclTensor* scaleOut)
{
    if (scaleOut->GetDataType() == op::DataType::DT_FLOAT8_E8M0) {
        return CheckMxShape(x, rotation, yOut, scaleOut);
    }
    return CheckIntRotationShape(x, rotation) && CheckIntOutputShape(x, yOut, scaleOut);
}

static bool CheckFormat(const aclTensor* x, const aclTensor* rotation, const aclTensor* yOut, const aclTensor* scaleOut)
{
    auto xFormat = x->GetStorageFormat();
    auto rotFormat = rotation->GetStorageFormat();
    auto yFormat = yOut->GetStorageFormat();
    auto scaleFormat = scaleOut->GetStorageFormat();
    bool noSupportFormat =
        ((xFormat == Format::FORMAT_FRACTAL_NZ) || (rotFormat == Format::FORMAT_FRACTAL_NZ) ||
         (yFormat == Format::FORMAT_FRACTAL_NZ) || (scaleFormat == Format::FORMAT_FRACTAL_NZ));
    if (noSupportFormat) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            kOpName, "x, rotation, yOut, scaleOut",
            FormatString(
                "%s, %s, %s, %s", op::ToString(xFormat).GetString(), op::ToString(rotFormat).GetString(),
                op::ToString(yFormat).GetString(), op::ToString(scaleFormat).GetString())
                .c_str(),
            FormatString("The formats of %s cannot be %s", "x, rotation, yOut, scaleOut", "FRACTAL_NZ").c_str());
        return false;
    }
    return true;
}

static bool CheckAlphaDtype(const aclTensor* alpha, const aclTensor* scaleOut)
{
    if (scaleOut->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_CHECK(
            alpha == nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "alpha", "not null",
                FormatString("The value of %s must be null when scaleAlg is not 2", "alpha").c_str()),
            return false);
        return true;
    }
    if (alpha != nullptr) {
        OP_CHECK(
            alpha->GetDataType() == op::DataType::DT_BF16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                kOpName, "alpha", op::ToString(alpha->GetDataType()).GetString(),
                FormatString("The dtype of %s must be %s", "alpha", "BF16").c_str()),
            return false);
        auto alphaShape = alpha->GetViewShape();
        OP_CHECK(
            alphaShape.GetDimNum() == 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                kOpName, "alpha", FormatString("%zu", alphaShape.GetDimNum()).c_str(),
                FormatString("The shape dim of %s must be %d", "alpha", 1).c_str()),
            return false);
        OP_CHECK(
            alphaShape.GetDim(0) == 1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                kOpName, "alpha", op::ToString(alphaShape).GetString(),
                FormatString("The shape of %s must be (%d,)", "alpha", 1).c_str()),
            return false);
    }
    return true;
}

static bool CheckAxisValid(int64_t axis, const aclTensor* x)
{
    auto xShape = x->GetViewShape();
    int64_t lastDim = xShape.GetDimNum() - 1;
    OP_CHECK(
        axis == -1 || axis == lastDim,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "axis", FormatString("%ld", axis).c_str(),
            FormatString("The value of %s must be in -1 or %ld", "axis", lastDim).c_str()),
        return false);
    return true;
}

static bool CheckRoundModeValid(const char* roundModeValue, const aclTensor* yOut)
{
    const std::string roundMode = std::string(roundModeValue);
    OP_CHECK(
        roundMode == "rint" || roundMode == "floor" || roundMode == "round",
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "round_mode", roundMode.c_str(),
            FormatString("The value of %s must be one of %s", "round_mode", "{rint, floor, round}").c_str()),
        return false);
    if (yOut->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN || yOut->GetDataType() == op::DataType::DT_FLOAT8_E5M2) {
        OP_CHECK(
            roundMode == "rint",
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "round_mode", roundMode.c_str(),
                FormatString("When yOut dtype is FLOAT8_E4M3/FLOAT8_E5M2, %s only support rint", "round_mode").c_str()),
            return false);
    }
    return true;
}

static bool CheckScaleAlgValid(int64_t scaleAlg, const aclTensor* yOut)
{
    auto yDtype = yOut->GetDataType();
    if (yDtype == op::DataType::DT_INT32 || yDtype == op::DataType::DT_INT8) {
        OP_CHECK(
            scaleAlg == 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "scaleAlg", FormatString("%ld", scaleAlg).c_str(),
                FormatString("The value of %s must be %ld for %s output", "scaleAlg", 0, "int8/int4").c_str()),
            return false);
    } else if (yDtype == op::DataType::DT_FLOAT4_E2M1) {
        OP_CHECK(
            scaleAlg == 0 || scaleAlg == 2,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "scaleAlg", FormatString("%ld", scaleAlg).c_str(),
                FormatString("The value of %s must be in %s for %s output", "scaleAlg", "{0, 2}", "MXFP4").c_str()),
            return false);
    } else {
        OP_CHECK(
            scaleAlg == 0 || scaleAlg == 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "scaleAlg", FormatString("%ld", scaleAlg).c_str(),
                FormatString("The value of %s must be in %s for %s output", "scaleAlg", "{0, 1}", "MXFP8").c_str()),
            return false);
    }
    return true;
}

static bool CheckDstTypeMaxValid(double dstTypeMax, int64_t scaleAlg, const aclTensor* yOut)
{
    auto yDtype = yOut->GetDataType();
    if (yDtype == op::DataType::DT_FLOAT4_E2M1 && scaleAlg == 2) {
        OP_CHECK(
            dstTypeMax >= SIX && dstTypeMax <= TWELVE,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "dstTypeMax", FormatString("%.17g", dstTypeMax).c_str(),
                FormatString(
                    "The value of %s must be in range [%.1f, %.1f] when scaleAlg is 2", "dstTypeMax", SIX, TWELVE)
                    .c_str()),
            return false);
    } else {
        OP_CHECK(
            dstTypeMax == ZERO,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                kOpName, "dstTypeMax", FormatString("%.17g", dstTypeMax).c_str(),
                FormatString("The value of %s must be %.1f when scaleAlg is not 2", "dstTypeMax", ZERO).c_str()),
            return false);
    }

    return true;
}

static bool CheckTransValid(bool trans)
{
    OP_CHECK(
        trans == false,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "trans", BoolToString(trans),
            FormatString("The value of %s must be %s", "trans", "false").c_str()),
        return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, const aclTensor* yOut,
    const aclTensor* scaleOut)
{
    CHECK_RET(CheckNotNull(x, rotation, yOut, scaleOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, rotation, yOut, scaleOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, rotation, yOut, scaleOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(x, rotation, yOut, scaleOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAlphaDtype(alpha, scaleOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus Int42Int32PackedTensor(const aclTensor* y, const aclTensor*& outTensor, aclOpExecutor* executor)
{
    auto viewShape = y->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    OP_LOGD("RotateQuant output real dtype is int4, pack to int32 to out.");
    return ACLNN_SUCCESS;
}

}; // namespace

aclnnStatus aclnnRotateQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, int64_t axis, char* roundMode,
    int64_t scaleAlg, double dstTypeMax, bool trans, aclTensor* yOut, aclTensor* scaleOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRotateQuant, DFX_IN(x, rotation, alpha), DFX_OUT(yOut, scaleOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, rotation, alpha, yOut, scaleOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(CheckAxisValid(axis, x), ACLNN_ERR_PARAM_INVALID);
    const char* roundModeValue = (roundMode != nullptr) ? roundMode : "rint";
    CHECK_RET(CheckRoundModeValid(roundModeValue, yOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckScaleAlgValid(scaleAlg, yOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDstTypeMaxValid(dstTypeMax, scaleAlg, yOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTransValid(trans), ACLNN_ERR_PARAM_INVALID);

    if (x->IsEmpty() || rotation->IsEmpty() || yOut->IsEmpty() || scaleOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto rotContiguous = l0op::Contiguous(rotation, uniqueExecutor.get());
    CHECK_RET(rotContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* alphaContiguous = nullptr;
    if (alpha != nullptr) {
        alphaContiguous = l0op::Contiguous(alpha, uniqueExecutor.get());
        CHECK_RET(alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto yDtype = yOut->GetDataType();
    if (yDtype == op::DataType::DT_INT32) {
        yDtype = op::DataType::DT_INT4;
    }
    int64_t dstDtype = static_cast<int64_t>(yDtype);

    std::tuple<aclTensor*, aclTensor*> result = l0op::RotateQuant(
        xContiguous, rotContiguous, alphaContiguous, axis, roundModeValue, scaleAlg, static_cast<float>(dstTypeMax),
        trans, dstDtype, yOut, scaleOut, uniqueExecutor.get());

    const aclTensor* y = std::get<0>(result);
    const aclTensor* scale = std::get<1>(result);
    CHECK_RET(y != nullptr && scale != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* yTensor = y;
    if (yDtype == op::DataType::DT_INT4) {
        ret = Int42Int32PackedTensor(y, yTensor, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    auto yResult = l0op::ViewCopy(yTensor, yOut, uniqueExecutor.get());
    CHECK_RET(yResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto scaleResult = l0op::ViewCopy(scale, scaleOut, uniqueExecutor.get());
    CHECK_RET(scaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRotateQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRotateQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
