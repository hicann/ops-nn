/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_transpose_quant_batch_mat_mul.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/dot.h"
#include "level0/fill.h"
#include "matmul/common/op_host/op_api/matmul.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"
#include "transpose_quant_batch_mat_mul.h"

#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/math_util.h"

using namespace std;
using namespace op;
using namespace Ops::NN;

static const std::initializer_list<op::DataType> x1_SUPPORT_LIST_FP8 = {
    DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x2_SUPPORT_LIST_FP8 = {
    DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x1_SUPPORT_LIST_MXFP8 = {DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x2_SUPPORT_LIST_MXFP8 = {DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x1_SCALE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> x2_SCALE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};
static constexpr size_t EXPECTED_DIM = 3;
static constexpr size_t EXPECTED_SCALE_DIM = 1;
static constexpr size_t EXPECTED_MX_SCALE_DIM = 4;
static constexpr size_t EXPECTED_NZ_DIM = 5;
static constexpr int TQBMM_VALID_K = 512;
static constexpr int TQBMM_VALID_N = 128;
static const uint64_t GROUP_M_OFFSET = 32;
static const uint64_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static constexpr uint64_t K_ALIGNMENT64 = 64UL;
static const int64_t SUPPORTED_GROUP_SIZE = 32;
static const int64_t NUM_TWO = 2;
static const int64_t NUM_THREE = 3;
static const char* const OP_NAME = "aclnnTransposeQuantBatchMatMulGetWorkspaceSize";

inline static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* out, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY)
{
    OP_CHECK(x1 != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "x1", "nullptr",
            Ops::NN::FormatString("The value of %s cannot be %s", "x1", "null").c_str()),
        return false);
    OP_CHECK(x2 != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "x2", "nullptr",
            Ops::NN::FormatString("The value of %s cannot be %s", "x2", "null").c_str()),
        return false);
    OP_CHECK(x1Scale != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "x1Scale", "nullptr",
            Ops::NN::FormatString("The value of %s cannot be %s", "x1Scale", "null").c_str()),
        return false);
    OP_CHECK(x2Scale != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "x2Scale", "nullptr",
            Ops::NN::FormatString("The value of %s cannot be %s", "x2Scale", "null").c_str()),
        return false);
    OP_CHECK(permX1 != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "permX1", "nullptr",
            Ops::NN::FormatString("The value of %s cannot be %s", "permX1", "null").c_str()),
        return false);
    OP_CHECK(permX2 != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "permX2", "nullptr",
            Ops::NN::FormatString("The value of %s cannot be %s", "permX2", "null").c_str()),
        return false);
    OP_CHECK(permY != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "permY", "nullptr",
            Ops::NN::FormatString("The value of %s cannot be %s", "permY", "null").c_str()),
        return false);
    OP_CHECK(out != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "out", "nullptr",
            Ops::NN::FormatString("The value of %s cannot be %s", "out", "null").c_str()),
        return false);
    return true;
}

static inline bool IsMicroScaling(const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    if (x1Scale == nullptr || x2Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

inline static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* out,
    int32_t dtype)
{
    if (!IsMicroScaling(x1Scale, x2Scale)) {
        if (!CheckType(x1->GetDataType(), x1_SUPPORT_LIST_FP8)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "x1", op::ToString(x1->GetDataType()).GetString(),
                Ops::NN::FormatString(
                    "The dtype of %s must be in %s", "x1", op::ToString(x1_SUPPORT_LIST_FP8).GetString())
                    .c_str());
            return false;
        }
        if (!CheckType(x2->GetDataType(), x2_SUPPORT_LIST_FP8)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "x2", op::ToString(x2->GetDataType()).GetString(),
                Ops::NN::FormatString(
                    "The dtype of %s must be in %s", "x2", op::ToString(x2_SUPPORT_LIST_FP8).GetString())
                    .c_str());
            return false;
        }
    } else {
        if (!CheckType(x1->GetDataType(), x1_SUPPORT_LIST_MXFP8)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "x1", op::ToString(x1->GetDataType()).GetString(),
                Ops::NN::FormatString(
                    "The dtype of %s must be in %s", "x1", op::ToString(x1_SUPPORT_LIST_MXFP8).GetString())
                    .c_str());
            return false;
        }
        if (!CheckType(x2->GetDataType(), x2_SUPPORT_LIST_MXFP8)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "x2", op::ToString(x2->GetDataType()).GetString(),
                Ops::NN::FormatString(
                    "The dtype of %s must be in %s", "x2", op::ToString(x2_SUPPORT_LIST_MXFP8).GetString())
                    .c_str());
            return false;
        }
    }
    if (!CheckType(x1Scale->GetDataType(), x1_SCALE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "x1Scale", op::ToString(x1Scale->GetDataType()).GetString(),
            Ops::NN::FormatString(
                "The dtype of %s must be in %s", "x1Scale", op::ToString(x1_SCALE_SUPPORT_LIST).GetString())
                .c_str());
        return false;
    }
    if (!CheckType(x2Scale->GetDataType(), x2_SCALE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "x2Scale", op::ToString(x2Scale->GetDataType()).GetString(),
            Ops::NN::FormatString(
                "The dtype of %s must be in %s", "x2Scale", op::ToString(x2_SCALE_SUPPORT_LIST).GetString())
                .c_str());
        return false;
    }
    // Only support FP16 and BF16
    if (!CheckType(out->GetDataType(), OUT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "out", op::ToString(out->GetDataType()).GetString(),
            Ops::NN::FormatString(
                "The dtype of %s must be in %s", "out", op::ToString(OUT_DTYPE_SUPPORT_LIST).GetString())
                .c_str());
        return false;
    }
    // Dtype shoulde be same with out tensor data type
    if (static_cast<int32_t>(out->GetDataType()) != dtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "out", op::ToString(out->GetDataType()).GetString(),
            Ops::NN::FormatString("The dtype of %s must be out dtype(%d)", "out", dtype).c_str());
        return false;
    }
    return true;
}

inline static bool CheckScalex1Valid(const aclTensor* x1Scale, int64_t batch, int64_t m, int64_t numGroup, bool isMxFp)
{
    OP_LOGD("X1Scale %s", op::ToString(x1Scale->GetViewShape()).GetString());
    auto dimTensorScale = x1Scale->GetViewShape().GetDimNum();
    if (isMxFp) {
        if (dimTensorScale != EXPECTED_MX_SCALE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "x1Scale", Ops::NN::FormatString("%zu", dimTensorScale).c_str(),
                Ops::NN::FormatString(
                    "In %s scene, the shape dim of %s must be %d", "MXFp8", "x1Scale",
                    static_cast<int>(EXPECTED_MX_SCALE_DIM))
                    .c_str());
            return false;
        }
        if (x1Scale->GetViewShape().GetDim(0) != m || x1Scale->GetViewShape().GetDim(1) != batch ||
            x1Scale->GetViewShape().GetDim(NUM_TWO) != numGroup ||
            x1Scale->GetViewShape().GetDim(NUM_THREE) != NUM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "x1Scale", op::ToString(x1Scale->GetViewShape()).GetString(),
                Ops::NN::FormatString(
                    "In %s scene, the shape of %s must be %s", "MXFp8", "x1Scale",
                    Ops::NN::FormatString("[%ld, %ld, %ld, 2]", m, batch, numGroup).c_str())
                    .c_str());
            return false;
        }
    } else {
        if (dimTensorScale != EXPECTED_SCALE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "x1Scale", Ops::NN::FormatString("%zu", dimTensorScale).c_str(),
                Ops::NN::FormatString("The shape dim of %s must be %zu", "x1Scale", EXPECTED_SCALE_DIM).c_str());
            return false;
        }
        if (x1Scale->GetViewShape().GetDim(0) != m) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                OP_NAME, "x1Scale", Ops::NN::FormatString("%ld", x1Scale->GetViewShape().GetDim(0)).c_str(),
                Ops::NN::FormatString(
                    "%s of %s must be equal to %s of %s (%ld)", "Shape[0]", "x1Scale", "m-axis", "x1", m)
                    .c_str());
            return false;
        }
    }
    return true;
}

inline static bool CheckScalex2Valid(
    const aclTensor* x2Scale, int64_t batch, int64_t n, int64_t numGroup, const aclIntArray* permX2, bool isMxFp)
{
    OP_LOGD("X2Scale %s", op::ToString(x2Scale->GetViewShape()).GetString());
    auto dimTensorScale = x2Scale->GetViewShape().GetDimNum();
    if (isMxFp) {
        int64_t scaleN = x2Scale->GetViewShape().GetDim((*permX2)[NUM_TWO]);
        int64_t scaleGroupNum = x2Scale->GetViewShape().GetDim((*permX2)[1]);
        if (dimTensorScale != EXPECTED_MX_SCALE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "x2Scale", Ops::NN::FormatString("%zu", dimTensorScale).c_str(),
                Ops::NN::FormatString(
                    "In %s scene, the shape dim of %s must be %d", "MXFp8", "x2Scale",
                    static_cast<int>(EXPECTED_MX_SCALE_DIM))
                    .c_str());
            return false;
        }
        std::vector<int64_t> dims = {batch, numGroup, n, NUM_TWO};
        if (x2Scale->GetViewShape().GetDim(0) != batch || scaleN != n || scaleGroupNum != numGroup ||
            x2Scale->GetViewShape().GetDim(NUM_THREE) != NUM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "x2Scale", op::ToString(x2Scale->GetViewShape()).GetString(),
                Ops::NN::FormatString(
                    "In %s scene, the shape of %s must be %s", "MXFp8", "x2Scale",
                    Ops::NN::FormatString("[%ld, %ld, %ld, 2]", batch, dims[(*permX2)[1]], dims[(*permX2)[NUM_TWO]])
                        .c_str())
                    .c_str());
            return false;
        }
    } else {
        int64_t scaleDim0 = x2Scale->GetViewShape().GetDim(0);
        if (dimTensorScale != EXPECTED_SCALE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "x2Scale", Ops::NN::FormatString("%zu", dimTensorScale).c_str(),
                Ops::NN::FormatString("The shape dim of %s must be %zu", "x2Scale", EXPECTED_SCALE_DIM).c_str());
            return false;
        }
        if (scaleDim0 != n) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                OP_NAME, "x2Scale", Ops::NN::FormatString("%ld", scaleDim0).c_str(),
                Ops::NN::FormatString(
                    "%s of %s must be equal to %s of %s (%ld)", "Shape[0]", "x2Scale", "n-axis", "x2", n)
                    .c_str());
            return false;
        }
    }
    return true;
}

inline static bool CheckScaleValid(
    const aclTensor* x1Scale, const aclTensor* x2Scale, int64_t batch, int64_t m, int64_t n, int64_t k,
    const aclIntArray* permX2, bool isMxFp)
{
    int64_t numGroup = MathUtil::CeilDivision(MathUtil::CeilDivision(k, SUPPORTED_GROUP_SIZE), NUM_TWO);
    // 对x1Scale的维度和shape信息进行校验
    if (x1Scale != nullptr && !CheckScalex1Valid(x1Scale, batch, m, numGroup, isMxFp)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1Scale is invalid");
        return false;
    }
    // 对x2Scale的维度和shape信息进行校验
    if (x2Scale != nullptr && !CheckScalex2Valid(x2Scale, batch, n, numGroup, permX2, isMxFp)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2Scale is invalid");
        return false;
    }
    return true;
}

static bool CheckPermValid(const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY, bool isMxFp)
{
    if (permX1->Size() != EXPECTED_DIM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "permX1",
            std::to_string(permX1->Size()).c_str(), std::to_string(EXPECTED_DIM).c_str());
        return false;
    }
    if (permX2->Size() != EXPECTED_DIM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "permX2",
            std::to_string(permX2->Size()).c_str(), std::to_string(EXPECTED_DIM).c_str());
        return false;
    }
    if (permY->Size() != EXPECTED_DIM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "permY",
            std::to_string(permY->Size()).c_str(), std::to_string(EXPECTED_DIM).c_str());
        return false;
    }
    // 当前支持转置场景
    auto allowedPermX1 = ((*permX1)[0] == 1 && (*permX1)[1] == 0 && (*permX1)[2] == 2); // 1 0 2
    auto allowedPermX2 = ((*permX2)[0] == 0 && (*permX2)[1] == 1 && (*permX2)[2] == 2); // 0 1 2
    auto allowedPermY = ((*permY)[0] == 1 && (*permY)[1] == 0 && (*permY)[2] == 2);     // 1 0 2
    std::string permX2ErrorInfo = "[0, 1, 2].";
    if (isMxFp) {
        allowedPermX2 = allowedPermX2 || ((*permX2)[0] == 0 && (*permX2)[1] == NUM_TWO && (*permX2)[NUM_TWO] == 1);
        permX2ErrorInfo = "[0, 1, 2] or [0, 2, 1].";
    }

    if (!allowedPermX1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "permX1",
            Ops::NN::FormatString("[%ld, %ld, %ld]", (*permX1)[0], (*permX1)[1], (*permX1)[2]).c_str(),
            Ops::NN::FormatString("The value of %s must be %s", "permX1", "[1, 0, 2]").c_str());
        return false;
    }
    if (!allowedPermX2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "permX2",
            Ops::NN::FormatString("[%ld, %ld, %ld]", (*permX2)[0], (*permX2)[1], (*permX2)[2]).c_str(),
            Ops::NN::FormatString("The value of %s must be %s", "permX2", permX2ErrorInfo.c_str()).c_str());
        return false;
    }
    if (!allowedPermY) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "permY", Ops::NN::FormatString("[%ld, %ld, %ld]", (*permY)[0], (*permY)[1], (*permY)[2]).c_str(),
            Ops::NN::FormatString("The value of %s must be %s", "permY", "[1, 0, 2]").c_str());
        return false;
    }
    return true;
}

static bool CheckShapeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const aclIntArray* permX1, const aclIntArray* permX2, bool isMxFp)
{
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    if ((x1Shape.GetDimNum() != EXPECTED_DIM) || (x2Shape.GetDimNum() != EXPECTED_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            OP_NAME, "x1, x2",
            Ops::NN::FormatString("%zu, %zu", x1Shape.GetDimNum(), x2Shape.GetDimNum()).c_str(),
            Ops::NN::FormatString("The shape dims of %s must be %zu", "x1, x2", EXPECTED_DIM).c_str());
        return false;
    }
    int64_t x1KDim = x1->GetViewShape().GetDim((*permX1)[2]);
    int64_t x2KDim = x2->GetViewShape().GetDim((*permX2)[1]);
    int64_t batch = x1->GetViewShape().GetDim((*permX1)[0]);
    int64_t m = x1->GetViewShape().GetDim((*permX1)[1]);
    int64_t n = x2->GetViewShape().GetDim((*permX2)[2]);

    if (x1KDim != x2KDim) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "x1, x2",
            Ops::NN::FormatString("%s, %s", op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be equal", "K-axis", "x1, x2").c_str());
        return false;
    }
    if (!isMxFp) {
        // Check shape k n
        if (x1KDim != TQBMM_VALID_K || n != TQBMM_VALID_N) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "x2", op::ToString(x2Shape).GetString(),
                Ops::NN::FormatString(
                    "The k-axis and n-axis of %s must be %d and %d", "x2", TQBMM_VALID_K, TQBMM_VALID_N)
                    .c_str());
            return false;
        }
    } else {
        if (x1KDim <= 0 || static_cast<uint64_t>(x1KDim) % K_ALIGNMENT64 != 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "x1", op::ToString(x1Shape).GetString(),
                Ops::NN::FormatString("%s of %s must be exactly divisible by %lu", "K-axis", "x1", K_ALIGNMENT64)
                    .c_str());
            return false;
        }
    }

    return CheckScaleValid(x1Scale, x2Scale, batch, m, n, x1KDim, permX2, isMxFp);
}

static inline bool CheckWeightNz(const aclTensor* x2, bool isMxFp)
{
    if (!isMxFp) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            OP_NAME, "x2", "FRACTAL_NZ",
            Ops::NN::FormatString("In %s case, the format of %s cannot be %s",
                                  "non-mxfp8 mode", "x2", "FRACTAL_NZ").c_str());
        return false;
    }
    auto storageShapeDim = x2->GetStorageShape().GetDimNum();
    OP_CHECK(
        storageShapeDim == EXPECTED_NZ_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "x2", Ops::NN::FormatString("%zu", storageShapeDim).c_str(),
            Ops::NN::FormatString("The storage shape dim of %s must be %d", "x2",
                                  static_cast<int>(EXPECTED_NZ_DIM)).c_str()),
        return false);
    return true;
}

static inline bool validGroupSize(uint64_t groupSizeM, uint64_t groupSizeN, uint64_t groupSizeK)
{
    return (groupSizeM == 0 || groupSizeM == 1) && (groupSizeN == 0 || groupSizeN == 1) &&
           groupSizeK == SUPPORTED_GROUP_SIZE;
}

static inline bool InferGroupSize(int64_t& groupSize, bool isMxFp)
{
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    if (isMxFp && !validGroupSize(groupSizeM, groupSizeN, groupSizeK)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "groupSizeM, groupSizeN, groupSizeK",
            Ops::NN::FormatString("%lu, %lu, %lu", groupSizeM, groupSizeN, groupSizeK).c_str(),
            Ops::NN::FormatString(
                "The values of %s must be %s, and the value of %s must be %d", "groupSizeM and groupSizeN", "0 or 1",
                "groupSizeK", static_cast<int>(SUPPORTED_GROUP_SIZE))
                .c_str());
        return false;
    }
    OP_LOGD(
        "Inferred groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeM, groupSizeN, groupSizeK);
    groupSize = groupSizeK;
    return true;
}

inline static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, aclTensor* out,
    int32_t dtype, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batch_split_factor, int64_t groupSize)
{
    // Only support DAV_3510
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "platform",
            Ops::NN::FormatString("%d", static_cast<int>(GetCurrentPlatformInfo().GetCurNpuArch())).c_str(),
            Ops::NN::FormatString("The value of %s can only be %s", "platform", "DAV_3510").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Check null
    CHECK_RET(CheckNotNull(x1, x2, out, x1Scale, x2Scale, permX1, permX2, permY), ACLNN_ERR_PARAM_NULLPTR);
    // check dtype
    CHECK_RET(CheckDtypeValid(x1, x2, x1Scale, x2Scale, out, dtype), ACLNN_ERR_PARAM_INVALID);
    bool isMxFp = IsMicroScaling(x1Scale, x2Scale);
    // check perm
    CHECK_RET(CheckPermValid(permX1, permX2, permY, isMxFp), ACLNN_ERR_PARAM_INVALID);
    // check shape
    CHECK_RET(CheckShapeValid(x1, x2, x1Scale, x2Scale, permX1, permX2, isMxFp), ACLNN_ERR_PARAM_INVALID);
    // check nz
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ) {
        CHECK_RET(CheckWeightNz(x2, isMxFp), ACLNN_ERR_PARAM_INVALID);
    }
    // 不支持x1Format、 outFormat为NZ
    if (ge::GetPrimaryFormat(x1->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ ||
        ge::GetPrimaryFormat(out->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            OP_NAME, "x1, out",
            Ops::NN::FormatString(
                "%s, %s",
                op::ToString(x1->GetStorageFormat()).GetString(),
                op::ToString(out->GetStorageFormat()).GetString())
                .c_str(),
            Ops::NN::FormatString("The formats of %s cannot be %s", "x1, out", "FRACTAL_NZ").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // MxFp8场景groupSize判断
    CHECK_RET(InferGroupSize(groupSize, isMxFp), ACLNN_ERR_PARAM_INVALID);
    if (batch_split_factor != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "batchSplitFactor",
            Ops::NN::FormatString("%d", batch_split_factor).c_str(),
            Ops::NN::FormatString("The value of %s must be %d", "batchSplitFactor", 1).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static const aclTensor* BuildTransposeQuantBatchMatMulGraph(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, int32_t dtype,
    int64_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batchSplitFactor, aclOpExecutor* executor)
{
    // 连续性转换
    auto contiguousX1 = l0op::Contiguous(x1, executor);
    OP_CHECK(
        contiguousX1 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x1 perprocess failed, contiguous return nullptr."), return nullptr);
    auto reformX1 = l0op::ReFormat(contiguousX1, op::Format::FORMAT_ND);
    OP_CHECK(
        reformX1 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x1 perprocess failed, reformat return nullptr."), return nullptr);

    auto contiguousX2 = l0op::Contiguous(x2, executor);
    OP_CHECK(
        contiguousX2 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x2 perprocess failed, contiguous return nullptr."), return nullptr);
    auto reformX2 = contiguousX2;
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) != op::Format::FORMAT_FRACTAL_NZ) {
        reformX2 = l0op::ReFormat(contiguousX2, op::Format::FORMAT_ND);
        OP_CHECK(
            reformX2 != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x2 perprocess failed, reformat return nullptr."),
            return nullptr);
    } else {
        reformX2->SetStorageShape(x2->GetStorageShape());
    }

    auto contiguousX1Scale = x1Scale;
    if (contiguousX1Scale != nullptr) {
        contiguousX1Scale = l0op::Contiguous(x1Scale, executor);
        OP_CHECK(
            contiguousX1Scale != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x1Scale perprocess failed, contiguous return nullptr."),
            return nullptr);
    }
    auto contiguousX2Scale = x2Scale;
    if (contiguousX2Scale != nullptr) {
        contiguousX2Scale = l0op::Contiguous(x2Scale, executor);
        OP_CHECK(
            contiguousX2Scale != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x2Scale perprocess failed, contiguous return nullptr."),
            return nullptr);
    }

    // Invoke tqbmmm l0 api
    return l0op::TransposeQuantBatchMatMul(
        reformX1, reformX2, nullptr, contiguousX1Scale, contiguousX2Scale, dtype, groupSize, permX1, permX2, permY,
        batchSplitFactor, executor);
}

aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    int32_t dtype, int64_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnTransposeQuantBatchMatMul,
        DFX_IN(x1, x2, bias, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor),
        DFX_OUT(out));

    // 固定写法, 创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto ret = CheckParams(x1, x2, x1Scale, x2Scale, out, dtype, permX1, permX2, permY, batchSplitFactor, groupSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空tensor 处理
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "x1, x2",
            Ops::NN::FormatString(
                "%s, %s", op::ToString(x1->GetViewShape()).GetString(), op::ToString(x2->GetViewShape()).GetString())
                .c_str(),
            Ops::NN::FormatString("%s cannot be empty tensors", "x1, x2").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 当前暂不支持bias
    if (bias != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "bias", "", Ops::NN::FormatString("The value of %s must be %s", "bias", "null").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 构建matmul计算图
    const aclTensor* tqbmmOut = nullptr;
    tqbmmOut = BuildTransposeQuantBatchMatMulGraph(
        x1, x2, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor, unique_executor.get());
    CHECK_RET(tqbmmOut != nullptr, ACLNN_ERR_PARAM_INVALID);

    tqbmmOut = l0op::Cast(tqbmmOut, out->GetDataType(), unique_executor.get());
    CHECK_RET(tqbmmOut != nullptr, ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(tqbmmOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTransposeQuantBatchMatMul(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTransposeQuantBatchMatMul);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnTransposeQuantBatchMatMulWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    int32_t dtype, int64_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnTransposeQuantBatchMatMulWeightNz,
        DFX_IN(x1, x2, bias, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor),
        DFX_OUT(out));

    // 固定写法, 创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto ret = CheckParams(x1, x2, x1Scale, x2Scale, out, dtype, permX1, permX2, permY, batchSplitFactor, groupSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // x2 format must be NZ
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            OP_NAME, "x2",
            op::ToString(x2->GetStorageFormat()).GetString(),
            Ops::NN::FormatString("The format of %s must be %s", "x2", "FRACTAL_NZ").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 空tensor 处理
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "x1, x2",
            Ops::NN::FormatString("%s, %s", op::ToString(x1->GetViewShape()).GetString(),
                                  op::ToString(x2->GetViewShape()).GetString()).c_str(),
            Ops::NN::FormatString("The shapes of %s cannot be %s", "x1, x2", "empty").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 当前暂不支持bias
    if (bias != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "bias", "not nullptr",
            Ops::NN::FormatString("The value of %s must be %s", "bias", "null").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 构建matmul计算图
    const aclTensor* tqbmmOut = nullptr;
    tqbmmOut = BuildTransposeQuantBatchMatMulGraph(
        x1, x2, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor, unique_executor.get());
    CHECK_RET(tqbmmOut != nullptr, ACLNN_ERR_PARAM_INVALID);

    tqbmmOut = l0op::Cast(tqbmmOut, out->GetDataType(), unique_executor.get());
    CHECK_RET(tqbmmOut != nullptr, ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(tqbmmOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTransposeQuantBatchMatMulWeightNz(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTransposeQuantBatchMatMulWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}