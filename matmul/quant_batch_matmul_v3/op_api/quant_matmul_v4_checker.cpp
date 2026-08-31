/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_v4.h"
#include <dlfcn.h>
#include "aclnn_quant_matmul_v3.h"
#include "aclnn_quant_matmul_weight_nz.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "quant_matmul_v3.h"
#include "matmul/common/op_host/op_api/quant_matmul_v4.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "util/math_util.h"
#include "quant_matmul_checker.h"
#include "quant_matmul_v4_common.h"

using namespace op;
using namespace quant_matmul_v4;
using Ops::Base::CeilDiv;
using Ops::NN::BoolToString;
using Ops::NN::FormatString;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::SwapLastTwoDimValue;

namespace quant_matmul_v4 {
namespace internal {
bool CheckNotNull(TupleTensor mandatoryTensors, const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2ub(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                               const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);

    if (x1->GetDataType() != op::DataType::DT_INT8 || x2->GetDataType() != op::DataType::DT_INT8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x1 and x2 dtype should be INT8, actual dtype are %s and %s",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    if (!(scale->GetDataType() == op::DataType::DT_UINT64 || scale->GetDataType() == op::DataType::DT_INT64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale dtype should be UINT64 or INT64, actual dtype is %s",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (bias != nullptr && bias->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias dtype should be INT32, actual dtype is %s",
                op::ToString(bias->GetDataType()).GetString());
        return false;
    }
    if (pertokenScaleOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PertokenScaleOptional should be null");
        return false;
    }
    if (!(out->GetDataType() == op::DataType::DT_INT8 || out->GetDataType() == op::DataType::DT_FLOAT16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output dtype should be INT8 or FLOAT16, actual dtype is %s",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2ubPertoken(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                                       const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (x1->GetDataType() != op::DataType::DT_INT8 || x2->GetDataType() != op::DataType::DT_INT8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x1 and x2 dtype should be INT8, actual dtype are %s and %s",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (scale->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale dtype should be FLOAT, actual dtype is %s",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (bias != nullptr && bias->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias dtype should be INT32, actual dtype is %s",
                op::ToString(bias->GetDataType()).GetString());
        return false;
    }
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (pertokenScaleOptional->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PertokenScaleOptional should be FLOAT, actual dtype is %s",
                op::ToString(pertokenScaleOptional->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() != op::DataType::DT_FLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output dtype should be FLOAT16, actual dtype is %s",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForA4W4(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                                       const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (isA8W4Msd(x1, x2, scale, pertokenScaleOptional)) {
        return true;
    }

    if (x1->GetDataType() != op::DataType::DT_INT4 || x2->GetDataType() != op::DataType::DT_INT4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x1 x2 dtype should be INT4 in a4w4 scenario, actual dtype is %s %s.",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    if (pertokenScaleOptional == nullptr) {
        if (scale->GetDataType() != op::DataType::DT_UINT64 && scale->GetDataType() != op::DataType::DT_INT64) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Scale dtype should be UINT64 or INT64 in a4w4 without pertoken scale scenario, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
            return false;
        }
        if (bias != nullptr && bias->GetDataType() != op::DataType::DT_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Bias dtype should be INT32 in a4w4 without pertoken scale scenario, actual dtype is %s",
                    op::ToString(bias->GetDataType()).GetString());
            return false;
        }
    }
    if (out->GetDataType() != op::DataType::DT_FLOAT16 && out->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Output dtype should be FLOAT16 or BFLOAT16 in a4w4 scenario, actual dtype is %s.",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForPertoken(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                                           const aclTensor* out)
{
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (pertokenScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(pertokenScaleOptional, op::DataType::DT_FLOAT, return false);
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT16 &&
            out->GetDataType() != op::DataType::DT_FLOAT16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is not nullptr, bias dtype is FLOAT16, out dtype should be FLOAT16, \
actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if (out->GetDataType() != op::DataType::DT_FLOAT16 && out->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is not nullptr, out dtype should be FLOAT16 or BFLOAT16, actual dtype "
                    "is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if (out->GetDataType() == op::DataType::DT_FLOAT16 &&
            (scale->GetDataType() != op::DataType::DT_FLOAT && scale->GetDataType() != op::DataType::DT_UINT64)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is not nullptr, out dtype is FLOAT16, scale dtype should be FLOAT \
or UINT64, actual dtype is %s.",
                    op::ToString(scale->GetDataType()).GetString());
            return false;
        }
    } else {
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is nullptr, bias dtype should not be FLOAT16.");
            return false;
        }
        if (bias != nullptr && bias->GetDataType() == op::DataType::DT_FLOAT &&
            out->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is nullptr and bias dtype is FLOAT, out dtype should be BFLOAT16, \
actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
        if ((out->GetDataType() == op::DataType::DT_INT8 || out->GetDataType() == op::DataType::DT_FLOAT16) &&
            scale->GetDataType() == op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When pertokenScaleOptional is nullptr and output dtype is INT8 or FLOAT16, \
scale dtype should not be FLOAT.");
            return false;
        }
    }
    return true;
}

static inline bool CheckDtypeValidInBf16OutScenario(TupleTensor mandatoryTensors)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    bool isA8W4 = isA8W4Int(x1, x2) || isA8W4Float(x1, x2);
    bool passScaleCheckA8W4 = isA8W4 && !(scale->GetDataType() == op::DataType::DT_BF16 ||
                                          scale->GetDataType() == op::DataType::DT_FLOAT ||
                                          scale->GetDataType() == op::DataType::DT_UINT64);
    if (passScaleCheckA8W4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When the dtype of out is BFLOAT16, scale dtype should be BFLOAT16/FLOAT/UINT64 in A8W4 scenario, "
                "actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    bool passScaleCheckOthers = !isA8W4 && !(scale->GetDataType() == op::DataType::DT_BF16 ||
                                             scale->GetDataType() == op::DataType::DT_FLOAT);
    if (passScaleCheckOthers) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When the dtype of out is BFLOAT16, scale dtype should be BFLOAT16/FLOAT in A4W4/A8W8 scenario, actual "
                "dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2outForUnclassified(TupleTensor mandatoryTensors,
                                                               TupleOptional optionalTensors, const aclTensor* out)
{
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (bias != nullptr && bias->GetDataType() == op::DataType::DT_BF16 &&
        out->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When bias dtype is BFLOAT16, out dtype should be BFLOAT16, actual dtype is %s.",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    if (scale->GetDataType() == op::DataType::DT_BF16 && out->GetDataType() != op::DataType::DT_BF16 &&
        out->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When scale dtype is BFLOAT16, out dtype should be BFLOAT16 or INT32, actual dtype is %s",
                op::ToString(out->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_BF16 && !CheckDtypeValidInBf16OutScenario(mandatoryTensors)) {
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT8 &&
        !(scale->GetDataType() == op::DataType::DT_INT64 || scale->GetDataType() == op::DataType::DT_UINT64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When the dtype of out is INT8, scale dtype should be INT64 or UINT64, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT32 &&
        !(scale->GetDataType() == op::DataType::DT_FLOAT || scale->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When the dtype of out is INT32, scale dtype should be FLOAT or BFLOAT16, actual dtype is %s.",
                op::ToString(scale->GetDataType()).GetString());
        return false;
    }
    if (out->GetDataType() == op::DataType::DT_INT32 && bias != nullptr &&
        bias->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "When the dtype of out is INT32, bias dtype should be INT32, actual dtype is %s.",
                op::ToString(bias->GetDataType()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckDtypeValidOnOnlyL0c2out(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                                const aclTensor* out, bool isA4W4)
{
    // 对A4W4场景/非A4W4场景进行校验
    if (isA4W4 && !CheckDtypeValidOnOnlyL0c2outForA4W4(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    if (!CheckDtypeValidOnOnlyL0c2outForUnclassified(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    if (!CheckDtypeValidOnOnlyL0c2outForPertoken(mandatoryTensors, optionalTensors, out)) {
        return false;
    }
    return true;
}

static inline bool CheckDtypeValid(TupleTensor mandatoryTensors, TupleOptional optionalTensors, const aclTensor* out,
                                   bool isA4W4)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto pertokenScale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, SCALE_TYPE_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_TYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_TYPE_SUPPORT_LIST, return false);

    // 无芯片差异的公共校验
    if (!isA8W4Msd(x1, x2, scale, pertokenScale) && x1->GetDataType() != x2->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In non-A8W4 case, x1 and x2 dtype should be same, \
        actual x1 dtype is %s and x2 dtype is %s.",
                op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }

    if (offset != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(offset, op::DataType::DT_FLOAT, return false);
        // offset only exists if out is int8
        if (out->GetDataType() != op::DataType::DT_INT8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Offset only exists when out dtype is INT8, actual dtype is %s.",
                    op::ToString(out->GetDataType()).GetString());
            return false;
        }
    }
    // 区分芯片校验
    if ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) &&
        ((pertokenScale == nullptr && !CheckDtypeValidOnOnlyL0c2ub(mandatoryTensors, optionalTensors, out)) ||
         (pertokenScale != nullptr && !CheckDtypeValidOnOnlyL0c2ubPertoken(mandatoryTensors, optionalTensors, out)))) {
        return false;
    } else if ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
                GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) &&
               !CheckDtypeValidOnOnlyL0c2out(mandatoryTensors, optionalTensors, out, isA4W4)) {
        return false;
    }
    return true;
}

static inline bool CheckFormatInt4(const aclTensor* x1, const aclTensor* x2)
{
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 only support ND format in a4w4 scenario, but now is %s.",
                op::ToString(x1->GetStorageFormat()).GetString());
        return false;
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND && x2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 only support ND/NZ in a4w4 scenario, but now is %s.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckFormat(TupleTensor mandatoryTensors, TupleOptional optionalTensors, bool isA4W4)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x1StorageFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x1->GetStorageFormat()));
    auto x2StorageFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat()));
    bool formatValid = ((x1StorageFormat == op::Format::FORMAT_ND) ||
                        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P &&
                         pertokenScaleOptional != nullptr && x1StorageFormat == op::Format::FORMAT_FRACTAL_NZ)) &&
                       (x2StorageFormat == op::Format::FORMAT_ND || x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ) &&
                       scale->GetStorageFormat() == op::Format::FORMAT_ND;
    if (offset != nullptr) {
        formatValid = formatValid && offset->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (pertokenScaleOptional != nullptr) {
        formatValid = formatValid && pertokenScaleOptional->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (bias != nullptr) {
        formatValid = formatValid && bias->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (isA4W4) {
        formatValid = formatValid && CheckFormatInt4(x1, x2);
    }
    return formatValid;
}

bool CheckDimRange(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale, const aclTensor* out)
{
    auto x2StorageFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat()));
    int64_t x2MaxDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MAX_DIM_NUM_NZ : MAX_DIM_NUM_ND;
    int64_t x2MinDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MIN_DIM_NUM_NZ : MIN_DIM_NUM_ND;
    int64_t x2DimNum = x2->GetStorageShape().GetDimNum();
    CHECK_RET(x2DimNum >= x2MinDimNum && x2DimNum <= x2MaxDimNum, false);
    OP_CHECK_MIN_DIM(x1, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MIN_DIM(out, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MAX_DIM(x1, MAX_DIM_NUM_ND, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_NUM_ND, return false);
    if (isMx(scale)) {
        OP_CHECK_MIN_DIM(scale, MX_SCALE_DIM_NUM, return false);
        OP_CHECK_MAX_DIM(scale, MX_SCALE_MAX_DIM_NUM, return false);
    } else {
        size_t expectScaleDim = 1;
        if (scale != nullptr && scale->GetViewShape().GetDimNum() == x2->GetViewShape().GetDimNum()) {
            expectScaleDim = scale->GetViewShape().GetDimNum();
        }
        OP_CHECK_WRONG_DIMENSION(scale, expectScaleDim, return false);
    }
    OP_LOGD("QuantMatmul check dim-num range success");
    return true;
}

static int64_t InferOutputShape(const aclTensor* x1, const aclTensor* x2, std::vector<int64_t>& batchRecord)
{
    int64_t inferedOutbatchValue = 1;
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    auto& longShapeTensor = x1DimNum > x2DimNum ? x1 : x2;
    auto& shortShapeTensor = x1DimNum > x2DimNum ? x2 : x1;
    size_t validOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        auto shortDimValue = i < validOffset ? 1 : shortShapeTensor->GetViewShape().GetDim(i - validOffset);
        auto longDimValue = longShapeTensor->GetViewShape().GetDim(i);
        if (shortDimValue > 1 && longDimValue > 1 && shortDimValue != longDimValue) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Current short dim value %ld and long dim value %ld are not supported for broadcasting.",
                    shortDimValue, longDimValue);
            return OUTPUT_INFER_FAIL;
        }
        int64_t curBatchValue = static_cast<int64_t>(std::max(shortDimValue, longDimValue));
        inferedOutbatchValue = inferedOutbatchValue * curBatchValue;
        batchRecord.push_back(curBatchValue);
    }
    return inferedOutbatchValue;
}

static inline bool CheckBiasShape(const aclTensor* bias, int64_t x2NDim, const std::vector<int64_t> batchRecord,
                                  int64_t inferedOutbatchValue)
{
    auto biasDimNum = bias->GetViewShape().GetDimNum();
    // 3 is bias with batch dim-num
    if (biasDimNum != 3 && biasDimNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias dim-num should equal 3 or 1, but it is %zu.", biasDimNum);
        return false;
    }
    auto biasFirstDim = bias->GetViewShape().GetDim(0);
    if (biasDimNum == 1) {
        OP_CHECK(biasFirstDim == x2NDim,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias 1st dim should be equal to x2 n dim %ld, but it is %ld.",
                         x2NDim, biasFirstDim),
                 return false);
        return true;
    }
    auto biasSecondDim = bias->GetViewShape().GetDim(1);
    // 2 is bias last dim index
    auto biasThirdDim = bias->GetViewShape().GetDim(2);
    // output batch need to be only 1 dim when bias dim is 3
    if (batchRecord.size() != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "When bias dim-num is 3, inferred out batch dim-num should be 1, but inferred out batch dim-num is %zu.",
            batchRecord.size());
        return false;
    }
    OP_CHECK(biasFirstDim == inferedOutbatchValue,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Bias 1st dim should be equal to out batch dim, but it is %ld and inferred out batch dim is %ld.",
                     biasFirstDim, inferedOutbatchValue),
             return false);
    OP_CHECK(biasSecondDim == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias 2nd dim should be equal to 1, but it is %ld.", biasFirstDim),
             return false);
    OP_CHECK(biasThirdDim == x2NDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias last dim should be equal to x2 n dim %ld, but actually is %ld.",
                     x2NDim, biasThirdDim),
             return false);
    return true;
}

static inline bool CheckOutShape(const aclTensor* out, bool twoDimMatmulCaseFlag, int64_t x1MDim, int64_t x2NDim,
                                 const std::vector<int64_t>& batchRecord)
{
    auto outDimNum = out->GetViewShape().GetDimNum();
    int64_t outMDim = out->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    int64_t outNDim = out->GetViewShape().GetDim(outDimNum - 1);
    size_t inferedOutDimNum = batchRecord.size() + 2;
    // x1 and x2 are 2 dim and out is 3 dim speical case
    if (outMDim == 1 && inferedOutDimNum == 2 && outDimNum == 3 && twoDimMatmulCaseFlag) {
        outDimNum -= 1;
        outMDim = out->GetViewShape().GetDim(0);
    }
    if (inferedOutDimNum != outDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Inferred output dim-num %zu is not equal to actual out dim-num %zu.",
                inferedOutDimNum, outDimNum);
        return false;
    }
    OP_CHECK(
        outMDim == x1MDim,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Out 1st dim should be equal to x1 m dim, but out 1st dim is %ld, x1 m dim is %ld.", outMDim, x1MDim),
        return false);
    OP_CHECK(
        outNDim == x2NDim,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Out 2nd dim should be equal to x2 n dim, but out 2nd dim is %ld, x2 n dim is %ld.", outNDim, x2NDim),
        return false);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        OP_CHECK(out->GetViewShape().GetDim(i) == batchRecord[i],
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Output dim %ld is not equal to inferred output dim %ld at shape index %zu.",
                         out->GetViewShape().GetDim(i), batchRecord[i], i),
                 return false);
    }
    return true;
}

static inline std::tuple<int64_t, int64_t, int64_t, int64_t> GetX1X2DimValue(const aclTensor* x1, const aclTensor* x2,
                                                                             bool transposeX1, bool transposeX2)
{
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x1MDim = transposeX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
    int64_t x2KDim = transposeX2 ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = transposeX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    return std::tie(x1KDim, x1MDim, x2KDim, x2NDim);
}

static inline bool CheckDimValue(const aclTensor* scale, const aclTensor* offset,
                                 const aclTensor* pertokenScaleOptional, int64_t x2NDim, int64_t x1MDim)
{
    if (scale->GetViewShape().GetDim(0) != x2NDim && scale->GetViewShape().GetDim(0) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scale last dim should equal to x2 n dim %ld or 1, but actual is %ld.", x2NDim,
                scale->GetViewShape().GetDim(0));
        return false;
    }

    if (offset != nullptr) {
        OP_CHECK_WRONG_DIMENSION(offset, 1, return false);
        if (offset->GetViewShape().GetDim(0) != x2NDim && offset->GetViewShape().GetDim(0) != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Offset 1st dim should equal to x2 n dim %ld or 1, but actual is %ld.",
                    x2NDim, offset->GetViewShape().GetDim(0));
            return false;
        }
    }

    if (pertokenScaleOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(pertokenScaleOptional, 1, return false);
        if (pertokenScaleOptional->GetViewShape().GetDim(0) != x1MDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "PertokenScaleOptional 1st dim should be equal to x1 m dim %ld or 1, but actually is %ld.", x1MDim,
                    pertokenScaleOptional->GetViewShape().GetDim(0));
            return false;
        }
    }
    return true;
}

static inline bool MaxDimCheck(int64_t x1DimNum, int64_t x2DimNum, const op::Shape x1Shape, const op::Shape x2Shape)
{
    OP_CHECK(x1Shape[x1DimNum - 1] <= LAST_AXIS_LIMIT && x2Shape[x2DimNum - 1] <= LAST_AXIS_LIMIT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 last dim or x2 last dim is larger than 65535, x1 is %ld, x2 is %ld.",
                     x1Shape[x1DimNum - 1], x2Shape[x2DimNum - 1]),
             return false);
    return true;
}

static inline bool CheckShapeForWeightNz(const aclTensor* x1, const aclTensor* x2, bool transposeX1, bool transposeX2)
{
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetStorageShape();
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetStorageShape().GetDimNum();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x2K1Dim = transposeX2 ? x2Shape[x2DimNum - NZ_K1_INDEX_TRANS] : x2Shape[x2DimNum - NZ_K1_INDEX];
    int64_t nz_k0_value_trans = SelectNzK0Value(x2->GetDataType(), isA8W4Float(x1, x2));
    int64_t roundValue = transposeX2 ? nz_k0_value_trans : NZ_K0_VALUE_BMM_BLOCK_NUM;
    int64_t x1KDimRound = ((x1KDim + roundValue - 1) / roundValue) * roundValue;
    if (x1KDimRound != x2K1Dim * roundValue) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlignedK1 value %ld is not matched with k1 value times roundValue, which is %ld.", x1KDimRound,
                x2K1Dim * roundValue);
        return false;
    }
    return true;
}

static inline bool CheckShapeInt4(const aclTensor* x1, const aclTensor* x2, bool transposeX1, bool transposeX2,
                                  const aclTensor* bias)
{
    int64_t x1KDim, x1MDim, x2KDim, x2NDim;
    std::tie(x1KDim, x1MDim, x2KDim, x2NDim) = GetX1X2DimValue(x1, x2, transposeX1, transposeX2);
    if (!IsAligned<int64_t>(x1KDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x1_k should be a positive even number in a4w4/a8w4 scenario, but now x1_k is %ld.", x1KDim);
        return false;
    }
    if (transposeX2 && !IsAligned<int64_t>(x2KDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x2_k should be a positive even number when transposeX2 is true in a4w4 scenario, but now x2_k is %ld.",
                x2KDim);
        return false;
    }
    if (isA8W4Int(x1, x2) && x1KDim > MAX_SHAPE_SIZE_A8W4_INT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-dim must belong to [1, %ld], which is %ld", MAX_SHAPE_SIZE_A8W4_INT,
                x1KDim);
        return false;
    }
    if (!transposeX2 && !IsAligned<int64_t>(x2NDim, INT4_NUMS_IN_INT8)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "x2_n should be a positive even number when transposeX2 is false in a4w4 scenario, but now x2_n is %ld.",
            x2NDim);
        return false;
    }
    if (transposeX1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TransposeX1 should be false in a4w4/a8w4 scenario, but now is true.");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != X2_FIXED_DIM_NUM_A4W4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 should be 2-d in a4w4/a8w4, but is %zu.", x2->GetViewShape().GetDimNum());
        return false;
    }
    if (bias != nullptr && bias->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias should be 1-d in a4w4/a8w4, but is %zu.",
                bias->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

static inline bool IsValidForLargeInput(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
                                        const aclTensor* pertokenScale)
{
    if (x1->GetDataType() != op::DataType::DT_INT8 || x2->GetDataType() != op::DataType::DT_INT8) {
        return false;
    }

    if (scale->GetDataType() != op::DataType::DT_FLOAT && scale->GetDataType() != op::DataType::DT_BF16) {
        return false;
    }

    if (pertokenScale == nullptr) {
        return false;
    }

    if (pertokenScale->GetViewShape().GetDimNum() != 1 || scale->GetViewShape().GetDimNum() != 1) {
        return false;
    }

    return true;
}

static inline bool CheckShape(TupleTensor& mandatoryTensors, TupleOptional& optionalTensors, TupleAttr& boolsTrans,
                              bool isA4W4, const aclTensor* out)
{
    auto transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    auto transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim;
    int64_t x1MDim;
    int64_t x2KDim;
    int64_t x2NDim;
    std::tie(x1KDim, x1MDim, x2KDim, x2NDim) = GetX1X2DimValue(x1, x2, transposeX1, transposeX2);

    if ((isA4W4 || isA8W4Msd(x1, x2, scale, pertokenScaleOptional)) &&
        !CheckShapeInt4(x1, x2, transposeX1, transposeX2, bias)) {
        return false;
    }

    OP_CHECK(x1KDim == x2KDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 k dim and x2 k dim should be same, but x1 is %ld, x2 is %ld.", x1KDim,
                     x2KDim),
             return false);
    bool inputSupport = IsValidForLargeInput(x1, x2, scale, pertokenScaleOptional);
    if (!(op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 && inputSupport)) {
        CHECK_RET(MaxDimCheck(x1DimNum, x2DimNum, x1Shape, x2Shape), false);
    }

    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
        CHECK_RET(CheckShapeForWeightNz(x1, x2, transposeX1, transposeX2), false);
    }

    CHECK_RET(CheckDimValue(scale, offset, pertokenScaleOptional, x2NDim, x1MDim), false);

    std::vector<int64_t> batchRecord;
    int64_t inferedOutbatchValue = InferOutputShape(x1, x2, batchRecord);
    if (inferedOutbatchValue == OUTPUT_INFER_FAIL) {
        return false;
    }
    if (bias != nullptr) {
        if (!CheckBiasShape(bias, x2NDim, batchRecord, inferedOutbatchValue)) {
            return false;
        }
    }
    bool twoDimMatmulCaseFlag = x1DimNum == x2DimNum && x2DimNum == 2;
    CHECK_RET(CheckOutShape(out, twoDimMatmulCaseFlag, x1MDim, x2NDim, batchRecord), false);
    return true;
}

static inline bool CheckEmptyTensor(TupleTensor mandatoryTensors)
{
    // scale, out和可选参数已在CheckShape函数校验，无需再次校验空tensor场景。
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmul not support to process empty tensor currently.");
        return false;
    }
    return true;
}

bool CheckA8W4FloatQuantType(const aclTensor* x1, const aclTensor* x2, const aclTensor* perTokenScale,
                             const aclTensor* scale)
{
    if (isA8W4Float(x1, x2)) {
        if (!IsMicroScaling(perTokenScale, scale) && !IsTCG(perTokenScale, scale)) {
            std::string scaleInfo = std::string("perTokenScale=") +
                                    op::ToString(perTokenScale != nullptr ? perTokenScale->GetDataType() :
                                                                            op::DataType::DT_UNDEFINED)
                                        .GetString() +
                                    ", scale=" + op::ToString(scale->GetDataType()).GetString();
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                "aclnnQuantMatmulV4", "perTokenScale/scale", scaleInfo.c_str(),
                "A8W4 float scenario only supports MX quantization (perTokenScale and scale are FLOAT8_E8M0) "
                "or TCG quantization (perTokenScale is null and scale is BF16 or FLOAT16)");
            return false;
        }
    }
    return true;
}

static bool CheckA8W4TcGDtype(const aclTensor* x2Scale, const aclTensor* bias, const aclTensor* yScale)
{
    if (bias != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "bias", "not null",
            "in A8W4 scenario, when the quantization mode is t-cg, bias must be null");
        return false;
    }
    if (yScale == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "yScale", "null",
            "in A8W4 scenario, when the quantization mode is t-cg, yScale can not be null");
        return false;
    }
    if (x2Scale == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2Scale", "null",
            "in A8W4 scenario, when the quantization mode is t-cg, x2Scale can not be null");
        return false;
    }
    if (x2Scale->GetDataType() != DataType::DT_BF16 && x2Scale->GetDataType() != DataType::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2Scale",
                                              op::ToString(x2Scale->GetDataType()).GetString(),
                                              "in A8W4 scenario, the dtype of x2Scale must be BFLOAT16 or FLOAT16");
        return false;
    }
    if (!CheckType(yScale->GetDataType(), Y_SCALE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "yScale",
                                              op::ToString(yScale->GetDataType()).GetString(),
                                              FormatString("the dtype of yScale must be in dtype support list %s",
                                                           op::ToString(Y_SCALE_SUPPORT_LIST).GetString())
                                                  .c_str());
        return false;
    }
    return true;
}

static bool CheckA8W4MxDtype(const aclTensor* bias, const aclTensor* yScale, const aclTensor* out)
{
    if (bias != nullptr) {
        // Check that bias dtype is the same as out dtype.
        if (bias->GetDataType() != out->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "bias, out",
                                                   FormatString("%s, %s", op::ToString(bias->GetDataType()).GetString(),
                                                                op::ToString(out->GetDataType()).GetString())
                                                       .c_str(),
                                                   "in A8W4 scenario, when the quantization mode is mx, the dtype of "
                                                   "bias must be the same as the dtype of out");
            return false;
        }
    }
    if (yScale != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "yScale", "not null",
            "in A8W4 scenario, when the quantization mode is mx, yScale must be null");
        return false;
    }
    return true;
}

static bool CheckA8W4Dtype(const TupleTensor& mandatoryTensors, const TupleOptional& optionalTensors,
                           const aclTensor* out)
{
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (x1Scale == nullptr) {
        return CheckA8W4TcGDtype(x2Scale, bias, yScale);
    }
    if (IsMicroScaling(x1Scale, x2Scale)) {
        return CheckA8W4MxDtype(bias, yScale, out);
    }
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
        "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1Scale, x2Scale",
        FormatString("%s, %s", x1Scale == nullptr ? "null" : op::ToString(x1Scale->GetDataType()).GetString(),
                     x2Scale == nullptr ? "null" : op::ToString(x2Scale->GetDataType()).GetString())
            .c_str(),
        "in A8W4 scenario, the quantization mode must be mx or t-cg");
    return false;
}

static inline bool CheckA8W4Format(const TupleTensor& mandatoryTensors, const TupleOptional& optionalTensors,
                                   const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    CHECK_RET(x1->GetStorageFormat() == op::Format::FORMAT_ND, false);
    CHECK_RET(x2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ, false);
    CHECK_RET(x2Scale->GetStorageFormat() == op::Format::FORMAT_ND, false);

    if (x1Scale != nullptr) {
        CHECK_RET(x1Scale->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (bias != nullptr) {
        CHECK_RET(bias->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (yScale != nullptr) {
        CHECK_RET(yScale->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    CHECK_RET(out->GetStorageFormat() == op::Format::FORMAT_ND, false);
    return true;
}

static inline bool CheckA8W4ScaleX1Shape(const TupleOptional& optionalTensors, const TupleTensor& mandatoryTensors,
                                         int64_t groupDimM, int64_t groupDimK)
{
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (IsMicroScaling(x1Scale, x2Scale)) {
        // 2：x1Scale 形状为（m, groupDimK / 2, 2）
        if (x1Scale->GetViewShape().GetDim(0) != groupDimM ||
            x1Scale->GetViewShape().GetDim(1) != CeilDiv(groupDimK, 2L) ||
            x1Scale->GetViewShape().GetDim(2) != 2) { // 2: 最后一维为2
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1Scale",
                FormatString("%ld, %ld, %ld", x1Scale->GetViewShape().GetDim(0), x1Scale->GetViewShape().GetDim(1),
                             x1Scale->GetViewShape().GetDim(MAX_DIM_VALUE))
                    .c_str(),
                FormatString("the shape of x1Scale must be [%ld, %ld, 2]", groupDimM, CeilDiv(groupDimK, 2L)).c_str());
            return false;
        }
    }
    return true;
}

static inline bool CheckA8W4ScaleX2Shape(const TupleOptional& optionalTensors, const TupleTensor& mandatoryTensors,
                                         int64_t groupDimN, int64_t groupDimK, bool transposeX2)
{
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    int64_t x2ScaleReshapeFactor = 2;
    int64_t x2ScaleNDim = transposeX2 ? x2Scale->GetViewShape().GetDim(0) : x2Scale->GetViewShape().GetDim(1);
    int64_t x2ScaleGroupDim = transposeX2 ? x2Scale->GetViewShape().GetDim(1) : x2Scale->GetViewShape().GetDim(0);
    if (IsMicroScaling(x1Scale, x2Scale)) {
        // 2： x2Scale形状：（n, groupDimK / 2, 2）
        if (x2ScaleNDim != groupDimN || x2ScaleGroupDim != CeilDiv(groupDimK, x2ScaleReshapeFactor) ||
            x2Scale->GetViewShape().GetDim(MX_SCALE_LAST_DIM_INDEX) != MX_SCALE_LAST_DIM) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2Scale",
                                                  FormatString("%ld, %ld, %ld", x2ScaleNDim, x2ScaleGroupDim,
                                                               x2Scale->GetViewShape().GetDim(MX_SCALE_LAST_DIM_INDEX))
                                                      .c_str(),
                                                  FormatString("the shape of x2Scale must be [%ld, %ld, 2]", groupDimN,
                                                               CeilDiv(groupDimK, x2ScaleReshapeFactor))
                                                      .c_str());
            return false;
        }
    } else if (x2ScaleNDim != groupDimN || x2ScaleGroupDim != groupDimK) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2Scale",
            FormatString("%ld, %ld", x2ScaleNDim, x2ScaleGroupDim).c_str(),
            FormatString("the shape of x2Scale must be [%ld, %ld]", groupDimN, groupDimK).c_str());
        return false;
    }
    return true;
}

static inline bool CheckA8W4OutAndBiasShape(const TupleOptional& optionalTensors, int64_t x1MDim, int64_t x2NDim,
                                            const aclTensor* out)
{
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    if (bias != nullptr) {
        if (bias->GetViewShape().GetDim(0) != 1 || bias->GetViewShape().GetDim(1) != x2NDim) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "aclnnQuantMatmulWeightNzGetWorkspaceSize", "bias",
                FormatString("%ld, %ld", bias->GetViewShape().GetDim(0), bias->GetViewShape().GetDim(1)).c_str(),
                FormatString("the shape of bias must be [1, %ld]", x2NDim).c_str());
            return false;
        }
    }

    if (out->GetViewShape().GetDim(0) != x1MDim || out->GetViewShape().GetDim(1) != x2NDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "out",
            FormatString("%ld, %ld", out->GetViewShape().GetDim(0), out->GetViewShape().GetDim(1)).c_str(),
            FormatString("the shape of out must be [%ld, %ld]", x1MDim, x2NDim).c_str());
        return false;
    }

    if (yScale != nullptr) {
        if (yScale->GetViewShape().GetDim(1) != x2NDim || yScale->GetViewShape().GetDim(0) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "aclnnQuantMatmulWeightNzGetWorkspaceSize", "yScale",
                FormatString("%ld, %ld", yScale->GetViewShape().GetDim(0), yScale->GetViewShape().GetDim(1)).c_str(),
                FormatString("the shape of yScale must be [1, %ld]", x2NDim).c_str());
            return false;
        }
    }
    return true;
}

static inline bool CheckA8W4X1X2Shape(int64_t x1KDim, int64_t x2KDim, int64_t x2NDim, bool isMx)
{
    // CHECK x1KDim
    if (x1KDim <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1",
                                              std::to_string(x1KDim).c_str(),
                                              "the k dimension of x1 must be greater than 0");
        return false;
    }
    if (x2NDim <= 0) { // A8W4Float
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2",
                                              std::to_string(x2NDim).c_str(),
                                              "the n dimension of x2 must be greater than 0");
        return false;
    }
    if (x1KDim != x2KDim) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1 K, x2 K",
                                               FormatString("%ld, %ld", x1KDim, x2KDim).c_str(),
                                               "the k dimension of x1 and x2 must be equal");
        return false;
    }
    if (isMx && (x1KDim % SUPPORTED_MX_A8W4_K_ALIGN_NUM != 0)) { // Mx量化k方向8对齐
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1", std::to_string(x1KDim).c_str(),
            FormatString("the k dimension of x1 must be aligned to %ld for MX quantization",
                         SUPPORTED_MX_A8W4_K_ALIGN_NUM)
                .c_str());
        return false;
    }
    if (!isMx && (x1KDim % SUPPORTED_TCG_A8W4_K_ALIGN_NUM != 0 || x1KDim <= SUPPORTED_TCG_A8W4_K_ALIGN_NUM)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1", std::to_string(x1KDim).c_str(),
            FormatString("the k dimension of x1 must be aligned to %ld and greater than %ld",
                         SUPPORTED_TCG_A8W4_K_ALIGN_NUM, SUPPORTED_TCG_A8W4_K_ALIGN_NUM)
                .c_str());
        return false;
    }
    if (x2NDim % SUPPORTED_N_ALIGN_NUM != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2", std::to_string(x2NDim).c_str(),
            FormatString("in A8W4 scenario, when x2 is FRACTAL_NZ, the n dimension of x2 must be aligned to %ld",
                         SUPPORTED_N_ALIGN_NUM)
                .c_str());
        return false;
    }
    return true;
}

static inline bool CheckA8W4Shape(const TupleTensor& mandatoryTensors, const TupleOptional& optionalTensors,
                                  const TupleAttr& boolsTrans, const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    bool transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);

    auto x1DimNum = x1->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    int64_t x1KDim = transposeX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x1MDim = transposeX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];

    const op::Shape x2Shape = x2->GetViewShape();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    int64_t x2KDim = transposeX2 ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = transposeX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    if (x1MDim <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1",
                                              std::to_string(x1MDim).c_str(),
                                              "the m dimension of x1 must be greater than 0");
        return false;
    }
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    CHECK_RET(CheckA8W4X1X2Shape(x1KDim, x2KDim, x2NDim, IsMicroScaling(x1Scale, x2Scale)), false);
    int64_t groupDimK = (x2KDim + SUPPORTED_GROUP_SIZE - 1) / SUPPORTED_GROUP_SIZE;
    int64_t groupDimM = x1MDim;
    int64_t groupDimN = x2NDim;
    CHECK_RET(CheckA8W4ScaleX1Shape(optionalTensors, mandatoryTensors, groupDimM, groupDimK), false);
    CHECK_RET(CheckA8W4ScaleX2Shape(optionalTensors, mandatoryTensors, groupDimN, groupDimK, transposeX2), false);
    CHECK_RET(CheckA8W4OutAndBiasShape(optionalTensors, x1MDim, x2NDim, out), false);
    return true;
}

static inline aclnnStatus CheckParamsA8W4Float(const TupleTensor& mandatoryTensors,
                                               const TupleOptional& optionalTensors, const TupleAttr& boolsTrans,
                                               const aclTensor* out)
{
    // 1. 校验dtype是否符合要求
    CHECK_RET(CheckA8W4Dtype(mandatoryTensors, optionalTensors, out), ACLNN_ERR_PARAM_INVALID);
    // 2. 检查format是否符合要求
    CHECK_RET(CheckA8W4Format(mandatoryTensors, optionalTensors, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否符合要求
    CHECK_RET(CheckA8W4Shape(mandatoryTensors, optionalTensors, boolsTrans, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsDAV3510(TupleTensor mandatoryTensors, TupleOptional optionalTensors, TupleAttr boolsTrans,
                                      const aclTensor* out, const char* apiName)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (isA8W4Float(x1, x2)) {
        return CheckParamsA8W4Float(mandatoryTensors, optionalTensors, boolsTrans, out);
    }
    const TupleInput inputTensors = std::tie(std::get<0>(mandatoryTensors), std::get<1>(mandatoryTensors));
    const aclTensor* yScale = nullptr;
    const aclTensor* x1Offset = nullptr;
    const int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_OPTIONAL_TUPLE>(optionalTensors);
    // 5 represents the aclnnQuantMatmulV5 interface
    const int64_t interfaceType = 5;
    const TupleQuant quantTensors = std::tie(
        std::get<1>(optionalTensors), std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors), yScale, x1Offset,
        std::get<0>(optionalTensors), x1Offset, std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors), groupSize,
        interfaceType);

    int64_t groupSizeReal = groupSize;
    auto& scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (isMx(scale)) {
        QuantMatmulChecker qmmV3Checker(inputTensors, quantTensors, boolsTrans, out, false, apiName);
        qmmV3Checker.Init();
        CHECK_RET(qmmV3Checker.InferGroupSize(groupSizeReal), ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("Infer groupSize success. groupSize: %ld.", groupSizeReal);
    }
    const TupleQuant quantTuples = std::tie(
        std::get<1>(optionalTensors), std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors), yScale, x1Offset,
        std::get<0>(optionalTensors), x1Offset, std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors), groupSizeReal,
        interfaceType);

    QuantMatmulChecker qmmV3Checker(inputTensors, quantTuples, boolsTrans, out, false, apiName);
    qmmV3Checker.Init();
    return qmmV3Checker.CheckParams();
}

aclnnStatus CheckWeightNzParamsDAV3510(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                                       const aclTensor* x2Scale, const aclTensor* out)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return ACLNN_SUCCESS;
    }

    if (x1 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1", "null",
                                              "x1 can not be null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2 == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2", "null",
                                              "x2 can not be null");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const bool isInt8Input = x1->GetDataType() == op::DataType::DT_INT8 && x2->GetDataType() == op::DataType::DT_INT8;
    const bool hasUnsupportedScaleDim = (x1Scale != nullptr && x1Scale->GetViewShape().GetDimNum() != 1) ||
                                        x2Scale->GetViewShape().GetDimNum() != 1;
    if (isInt8Input && hasUnsupportedScaleDim) {
        const std::string x1ScaleDim = x1Scale == nullptr ? "null" :
                                                            FormatString("%zuD", x1Scale->GetViewShape().GetDimNum());
        const std::string scaleDims = x1ScaleDim + ", " + FormatString("%zuD", x2Scale->GetViewShape().GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1Scale, x2Scale", scaleDims.c_str(),
            "when x1 and x2 dtypes are INT8, x1Scale must be null or 1D and x2Scale must be 1D");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (isA8W4Float(x1, x2)) {
        if (!IsFormatNZ(x2)) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2", op::ToString(x2->GetStorageFormat()).GetString(),
                "in A8W4 scenario, the format of x2 must be FRACTAL_NZ, FRACTAL_NZ_C0_4 or FRACTAL_NZ_C0_32");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (out->GetDataType() != op::DataType::DT_BF16 && out->GetDataType() != op::DataType::DT_FLOAT16) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "out",
                                                  op::ToString(out->GetDataType()).GetString(),
                                                  "in A8W4 scenario, the dtype of out must be BFLOAT16 or FLOAT16");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2",
                                               op::ToString(x2->GetStorageFormat()).GetString(),
                                               "the format of x2 must be FRACTAL_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 对于torch的场景，NZ情况下，x2的k和n不能为1
    int64_t dim1 = x2->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = x2->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    if (x2->GetViewShape().GetDim(dim2) == 1 || x2->GetViewShape().GetDim(dim1) == 1) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2 K, x2 N",
            FormatString("%ld, %ld", x2->GetViewShape().GetDim(dim2), x2->GetViewShape().GetDim(dim1)).c_str(),
            "when the format of x2 is FRACTAL_NZ, the k dimension and n dimension of x2 can not be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGD("QuantMatmulWeightNz check params success.");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(TupleTensor mandatoryTensors, TupleOptional optionalTensors, TupleAttr boolsTrans, bool isA4W4,
                        const aclTensor* out, const char* apiName)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return CheckParamsDAV3510(mandatoryTensors, optionalTensors, boolsTrans, out, apiName);
    } else {
        // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
        CHECK_RET(CheckDtypeValid(mandatoryTensors, optionalTensors, out, isA4W4), ACLNN_ERR_PARAM_INVALID);

        // 2. 检查shape是否符合要求
        CHECK_RET(CheckShape(mandatoryTensors, optionalTensors, boolsTrans, isA4W4, out), ACLNN_ERR_PARAM_INVALID);

        // 3. 检查format是否符合要求
        CHECK_RET(CheckFormat(mandatoryTensors, optionalTensors, isA4W4), ACLNN_ERR_PARAM_INVALID);

        // 4. 空Tensor处理逻辑
        CHECK_RET(CheckEmptyTensor(mandatoryTensors), ACLNN_ERR_PARAM_INVALID);
    }
    OP_LOGD("QuantMatmul check params success.");
    return ACLNN_SUCCESS;
}

bool CheckInputAttrExistence(const TupleAttr& boolsTrans, const TupleTensor& mandatoryTensors,
                             const TupleOptional& optionalTensors)
{
    auto& x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_OPTIONAL_TUPLE>(optionalTensors);
    bool transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    if (transposeX1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "transposeX1",
                                              transposeX1 ? "true" : "false",
                                              "in A8W4 scenario, transposeX1 must be false");
        return false;
    }

    bool isX2Nz = ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
    if (!isX2Nz) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2",
                                               op::ToString(x2->GetStorageFormat()).GetString(),
                                               "in A8W4 scenario, the format of x2 must be FRACTAL_NZ");
        return false;
    }

    auto& x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    if (x1Scale == nullptr && transposeX2) {
        // A8W4 scenario with t-cg quant mode
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "transposeX2", transposeX2 ? "true" : "false",
            "in A8W4 scenario with NZ format, when the quantization mode is t-cg, transposeX2 must be false");
        return false;
    } else if (IsMicroScaling(x1Scale, x2Scale) && !transposeX2) {
        // A8W4 scenario with mx quant mode
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantMatmulWeightNzGetWorkspaceSize", "transposeX2", transposeX2 ? "true" : "false",
            "in A8W4 scenario with NZ format, when the quantization mode is mx, transposeX2 must be true");
        return false;
    }
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    if (groupSizeK != SUPPORTED_GROUP_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "groupSizeK",
                                              std::to_string(groupSizeK).c_str(),
                                              "in A8W4 scenario, groupSizeK must be 32");
        return false;
    }
    return true;
}

bool CheckDimRangeA8W4(const TupleTensor& mandatoryTensors, const TupleOptional& optionalTensors, const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);

    if (x1->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1",
                                                 FormatString("%zuD", x1->GetViewShape().GetDimNum()).c_str(),
                                                 "the shape dim of x1 must be 2");
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2",
                                                 FormatString("%zuD", x2->GetViewShape().GetDimNum()).c_str(),
                                                 "the shape dim of x2 must be 2");
        return false;
    }
    if (bias != nullptr && bias->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "bias",
                                                 FormatString("%zuD", bias->GetViewShape().GetDimNum()).c_str(),
                                                 "the shape dim of bias must be 2");
        return false;
    }
    if (out->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "out",
                                                 FormatString("%zuD", out->GetViewShape().GetDimNum()).c_str(),
                                                 "the shape dim of out must be 2");
        return false;
    }
    OP_LOGD("QuantMatmul check dimension range success.");
    return true;
}

bool CheckScaleDimRangeA8W4(const TupleTensor& mandatoryTensors, const TupleOptional& optionalTensors)
{
    auto x2Scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x1Scale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);

    if (IsMicroScaling(x1Scale, x2Scale)) {
        if (x1Scale->GetViewShape().GetDimNum() != MX_SCALE_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1Scale",
                                                     FormatString("%zuD", x1Scale->GetViewShape().GetDimNum()).c_str(),
                                                     "the shape dim of x1Scale must be 3");
            return false;
        }
        if (x2Scale != nullptr && x2Scale->GetViewShape().GetDimNum() != MX_SCALE_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2Scale",
                                                     FormatString("%zuD", x2Scale->GetViewShape().GetDimNum()).c_str(),
                                                     "the shape dim of x2Scale must be 3");
            return false;
        }
    } else {
        if (x1Scale != nullptr && x1Scale->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x1Scale",
                                                     FormatString("%zuD", x1Scale->GetViewShape().GetDimNum()).c_str(),
                                                     "the shape dim of x1Scale must be 2");
            return false;
        }
        if (x2Scale != nullptr && x2Scale->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "x2Scale",
                                                     FormatString("%zuD", x2Scale->GetViewShape().GetDimNum()).c_str(),
                                                     "the shape dim of x2Scale must be 2");
            return false;
        }
    }
    if (yScale != nullptr && yScale->GetViewShape().GetDimNum() != MAX_DIM_VALUE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "yScale",
                                                 FormatString("%zuD", yScale->GetViewShape().GetDimNum()).c_str(),
                                                 "the shape dim of yScale must be 2");
        return false;
    }
    OP_LOGD("QuantMatmul check scale dimension range success.");
    return true;
}

} // namespace internal
} // namespace quant_matmul_v4
