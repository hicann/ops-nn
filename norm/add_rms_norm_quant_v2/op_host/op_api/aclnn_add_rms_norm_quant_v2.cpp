/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/format_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

#include "norm/add_rms_norm_quant/op_host/op_api/add_rms_norm_quant.h"
#include "aclnn_add_rms_norm_quant_v2.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace AddRmsNormQuantV2ACLNN {
constexpr int IDX_2 = 2;
constexpr int IDX_1 = 1;
constexpr int IDX_0 = 0;
constexpr int IDX_3 = 3;
const size_t MIN_SUPPORT_DIMS_NUMS = 1;
const uint64_t INT32_MAX_LIMIT = 2147483647;
const int64_t SIZE_MIN_LIMIT_310P = 32;
const size_t DIM_TWO = 2;

struct AddRmsNormQuantV2InputTensor {
    const aclTensor* x1;
    const aclTensor* x2;
    const aclTensor* gamma;
    const aclTensor* betaOptional;
    const aclTensor* scales1;
    const aclTensor* scales2Optional;
    const aclTensor* zeroPoints1Optional;
    const aclTensor* zeroPoints2Optional;
};

struct AddRmsNormQuantV2OutputTensor {
    aclTensor* y1Out;
    aclTensor* y2Out;
    aclTensor* resOut;
    aclTensor* xOut;
};

struct ParamStruct {
    int64_t axis;
    double epsilon;
    bool divMode;
};

static const std::initializer_list<op::DataType> EXTEND_ATB_DTYPE_SUPPORT_LIST_X = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST_ZEROPOINT = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_ZEROPOINT = {
    op::DataType::DT_INT32, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_X = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_SCALE = {
    op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckInputNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1)
{
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(scales1, return false);
    OP_CHECK_NULL(gamma, return false);
    return true;
}

static bool CheckOutputNotNull(aclTensor* y1Out, aclTensor* y2Out, const aclTensor* xOut)
{
    OP_CHECK_NULL(y1Out, return false);
    OP_CHECK_NULL(y2Out, return false);
    OP_CHECK_NULL(xOut, return false);
    return true;
}

static bool CheckInputDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gamma, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scales1, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    return true;
}

static bool CheckXDtypeValid910BC(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* betaOptional, const aclTensor* xOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gamma, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_X, return false);
    if (nullptr != betaOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(betaOptional, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_X, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(xOut, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_X, return false);
    return true;
}

static bool CheckScalesZerosDtypeValid910BC(
    const aclTensor* scales1Optional, const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional,
    const aclTensor* zeroPoints2Optional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(scales1Optional, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_SCALE, return false);
    if (nullptr != zeroPoints2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(
            zeroPoints2Optional, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }
    if (nullptr != zeroPoints1Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(
            zeroPoints1Optional, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }

    if (nullptr != scales2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scales2Optional, ASCEND910BC_AND_310P_DTYPE_SUPPORT_LIST_SCALE, return false);
    }
    return true;
}

static bool CheckOptionalParamDtypeValid(
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    const aclTensor* betaOptional)
{
    if (nullptr != zeroPoints2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(zeroPoints2Optional, ASCEND950_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }
    if (nullptr != zeroPoints1Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(zeroPoints1Optional, ASCEND950_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }

    if (nullptr != scales2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scales2Optional, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }

    if (nullptr != betaOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(betaOptional, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    return true;
}

static bool CheckOutputDtypeValid(const aclTensor* y1Out, const aclTensor* y2Out, const aclTensor* xOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(xOut, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_MATCH(y2Out, op::DataType::DT_INT8, return false); // Mandatory output
    OP_CHECK_DTYPE_NOT_MATCH(y1Out, op::DataType::DT_INT8, return false);
    return true;
}

static bool CheckOutputDtypeValid910BC(const aclTensor* y1Out, const aclTensor* y2Out)
{
    OP_CHECK_DTYPE_NOT_MATCH(y2Out, op::DataType::DT_INT8, return false); // Mandatory output
    OP_CHECK_DTYPE_NOT_MATCH(y1Out, op::DataType::DT_INT8, return false);
    return true;
}

static bool CheckInputShapeDim(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* scales1)
{
    OP_CHECK_MAX_DIM(x2, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(x1, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(scales1, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gamma, MAX_SUPPORT_DIMS_NUMS, return false);

    OP_CHECK_MIN_DIM(x2, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(x1, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(scales1, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(gamma, MIN_SUPPORT_DIMS_NUMS, return false);

    CHECK_RET(CheckDims(x2), false);
    CHECK_RET(CheckDims(x1), false);
    CHECK_RET(CheckDims(scales1), false);
    CHECK_RET(CheckDims(gamma), false);
    return true;
}

static bool CheckOptionalParamShapeDim(
    const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    const aclTensor* betaOptional)
{
    if (nullptr != zeroPoints2Optional) {
        OP_CHECK_MAX_DIM(zeroPoints2Optional, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MIN_DIM(zeroPoints2Optional, MIN_SUPPORT_DIMS_NUMS, return false);
        CHECK_RET(CheckDims(zeroPoints2Optional), false);
    }
    if (nullptr != zeroPoints1Optional) {
        OP_CHECK_MAX_DIM(zeroPoints1Optional, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MIN_DIM(zeroPoints1Optional, MIN_SUPPORT_DIMS_NUMS, return false);
        CHECK_RET(CheckDims(zeroPoints1Optional), false);
    }
    if (nullptr != scales2Optional) {
        OP_CHECK_MAX_DIM(scales2Optional, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MIN_DIM(scales2Optional, MIN_SUPPORT_DIMS_NUMS, return false);
        CHECK_RET(CheckDims(scales2Optional), false);
    }
    if (nullptr != betaOptional) {
        OP_CHECK_MAX_DIM(betaOptional, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MIN_DIM(betaOptional, MIN_SUPPORT_DIMS_NUMS, return false);
        CHECK_RET(CheckDims(betaOptional), false);
    }
    return true;
}

static bool CheckOutputShapeDim(aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut)
{
    OP_CHECK_MAX_DIM(xOut, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(y2Out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(y1Out, MAX_SUPPORT_DIMS_NUMS, return false);

    OP_CHECK_MIN_DIM(xOut, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(y2Out, MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(y1Out, MIN_SUPPORT_DIMS_NUMS, return false);

    CHECK_RET(CheckDims(xOut), false);
    CHECK_RET(CheckDims(y1Out), false);
    CHECK_RET(CheckDims(y2Out), false);
    return true;
}

static bool CheckShapeSameWithX(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* y1Out, const aclTensor* y2Out, const aclTensor* xOut)
{
    OP_CHECK_SHAPE_NOT_EQUAL(x1, x2, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(x1, y1Out, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(x1, y2Out, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(x1, xOut, return false);

    return true;
}

static bool CheckShapeSameWithGamma(
    const aclTensor* gamma, const aclTensor* scales1, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional)
{
    OP_CHECK_SHAPE_NOT_EQUAL(gamma, scales1, return false);
    if (nullptr != scales2Optional) {
        OP_CHECK_SHAPE_NOT_EQUAL(gamma, scales2Optional, return false);
    }
    if (nullptr != zeroPoints1Optional) {
        OP_CHECK_SHAPE_NOT_EQUAL(gamma, zeroPoints1Optional, return false);
    }
    if (nullptr != zeroPoints2Optional) {
        OP_CHECK_SHAPE_NOT_EQUAL(gamma, zeroPoints2Optional, return false);
    }
    return true;
}

static bool IsSocVersion310P()
{
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P) {
        return true;
    }
    return false;
}

static bool IsSocVersion910BC_310P()
{
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B ||
        op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_93 ||
        op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND310P) {
        return true;
    }
    return false;
}

static inline bool CheckShapeGammaX(const aclTensor* x1, const aclTensor* gamma) {
    const auto& xShape = x1->GetViewShape();
    const auto& gammaShape = gamma->GetViewShape();
    int64_t gammaDimNum = gammaShape.GetDimNum();
    int64_t xDimNum = xShape.GetDimNum();
    int64_t formerDimNum = xDimNum - gammaDimNum;
    // gamma的dim必须小于x的dim
    if(formerDimNum < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gamma tensor's dim %ld should not greater than x1 tensor's dim %ld.", gammaDimNum, xDimNum);
        return false;
    }
    int64_t totalGammaNum = 1;
    for(int64_t i = 0; i < gammaDimNum; i++) {
        // gamma和x的shape后N维，每一维对应的大小应该相等
        if (xShape.GetDim(formerDimNum + i) != gammaShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gamma tensor's shape dim %ld is %ld, is not equal to x1 %ld.", i, gammaShape.GetDim(i), xShape.GetDim(formerDimNum + i));
            return false;
        } else {
            totalGammaNum = totalGammaNum * gammaShape.GetDim(i);
        }
    }

    if(IsSocVersion310P()) {
        if (totalGammaNum < SIZE_MIN_LIMIT_310P) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gamma tensor's size %ld is smaller than %ld.", totalGammaNum, SIZE_MIN_LIMIT_310P);
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor)
{
    // 1. 检查必选输入/输出是否为空指针
    CHECK_RET(CheckInputNotNull(inputTensor.x1, inputTensor.x2, inputTensor.gamma, inputTensor.scales1),
    ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckOutputNotNull(outputTensor.y1Out, outputTensor.y2Out, outputTensor.xOut),
    ACLNN_ERR_PARAM_NULLPTR);

    // 3. 检查输入/输出的shape大小
    CHECK_RET(CheckInputShapeDim(inputTensor.x1, inputTensor.x2, inputTensor.gamma, inputTensor.scales1),
    ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalParamShapeDim(inputTensor.scales2Optional, inputTensor.zeroPoints1Optional, inputTensor.zeroPoints2Optional, inputTensor.betaOptional),
    ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutputShapeDim(outputTensor.y1Out, outputTensor.y2Out, outputTensor.xOut),
    ACLNN_ERR_PARAM_INVALID);

    if(IsSocVersion910BC_310P()) {
        // 3. 检查输入/输出的数据类型是否合法
        CHECK_RET(CheckScalesZerosDtypeValid910BC(inputTensor.scales1, inputTensor.scales2Optional, inputTensor.zeroPoints1Optional, inputTensor.zeroPoints2Optional), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckXDtypeValid910BC(inputTensor.x1, inputTensor.x2, inputTensor.gamma, inputTensor.betaOptional, outputTensor.xOut), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckOutputDtypeValid910BC(outputTensor.y1Out, outputTensor.y2Out), ACLNN_ERR_PARAM_INVALID);

        // 4. 检查输入输出shape的一致性
        CHECK_RET(CheckShapeSameWithX(inputTensor.x1, inputTensor.x2, outputTensor.y1Out, outputTensor.y2Out, outputTensor.xOut), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckShapeSameWithGamma(inputTensor.gamma, inputTensor.scales1, inputTensor.scales2Optional, inputTensor.zeroPoints1Optional, inputTensor.zeroPoints2Optional), ACLNN_ERR_PARAM_INVALID);
        if(nullptr != inputTensor.betaOptional) {
            OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.gamma, inputTensor.betaOptional, return ACLNN_ERR_PARAM_INVALID);
        }
        // 检查x和gamma的shape关系
        CHECK_RET(CheckShapeGammaX(inputTensor.x1, inputTensor.gamma), ACLNN_ERR_PARAM_INVALID);
    } else {
        // 2. 检查输入/输出的数据类型是否合法
        CHECK_RET(CheckInputDtypeValid(inputTensor.x1, inputTensor.x2, inputTensor.gamma, inputTensor.scales1),
        ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckOptionalParamDtypeValid(inputTensor.scales2Optional, inputTensor.zeroPoints1Optional, inputTensor.zeroPoints2Optional, inputTensor.betaOptional),
        ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckOutputDtypeValid(outputTensor.y1Out, outputTensor.y2Out, outputTensor.xOut),
        ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static bool SimpleCheckNotNull(
    AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor)
{
    OP_CHECK_NULL(inputTensor.x1, return false);
    OP_CHECK_NULL(inputTensor.x2, return false);
    OP_CHECK_NULL(inputTensor.gamma, return false);
    OP_CHECK_NULL(inputTensor.scales1, return false);
    OP_CHECK_NULL(outputTensor.y1Out, return false);
    OP_CHECK_NULL(outputTensor.y2Out, return false);
    return true;
}

static bool SimpleCheckDtypeValid(
    AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x1, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x2, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.gamma, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.scales1, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensor.x1, inputTensor.x2, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensor.x1, inputTensor.gamma, return false);
    if (nullptr != inputTensor.betaOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.betaOptional, ASCEND950_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }
    if (nullptr != inputTensor.scales2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.scales2Optional, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    if (nullptr != inputTensor.zeroPoints1Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.zeroPoints1Optional, ASCEND950_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }
    if (nullptr != inputTensor.zeroPoints2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.zeroPoints2Optional, ASCEND950_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }

    OP_CHECK_DTYPE_NOT_MATCH(outputTensor.y1Out, op::DataType::DT_INT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(outputTensor.y2Out, op::DataType::DT_INT8, return false);  // Mandatory output
    if (nullptr != outputTensor.xOut) {
        OP_CHECK_DTYPE_NOT_SUPPORT(outputTensor.xOut, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    if (nullptr != outputTensor.resOut) {
        OP_CHECK_DTYPE_NOT_SUPPORT(outputTensor.resOut, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    return true;
}

static bool SimpleCheckShapeDim(
    AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor)
{
    OP_CHECK_MAX_DIM(inputTensor.x1, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(inputTensor.x2, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(inputTensor.gamma, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(inputTensor.scales1, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, inputTensor.x2, return false);
    if (nullptr != inputTensor.betaOptional) {
        OP_CHECK_MAX_DIM(inputTensor.betaOptional, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    if (nullptr != inputTensor.scales2Optional) {
        OP_CHECK_MAX_DIM(inputTensor.scales2Optional, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    if (nullptr != inputTensor.zeroPoints1Optional) {
        OP_CHECK_MAX_DIM(inputTensor.zeroPoints1Optional, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.zeroPoints1Optional, inputTensor.scales1, return false);
    }
    if (nullptr != inputTensor.zeroPoints2Optional) {
        OP_CHECK_MAX_DIM(inputTensor.zeroPoints2Optional, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    OP_CHECK_MAX_DIM(outputTensor.y1Out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, outputTensor.y1Out, return false);
    OP_CHECK_MAX_DIM(outputTensor.y2Out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, outputTensor.y2Out, return false);
    if (nullptr != outputTensor.resOut) {
        OP_CHECK_MAX_DIM(outputTensor.resOut, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, outputTensor.resOut, return false);
    }
    if (nullptr != outputTensor.xOut) {
        OP_CHECK_MAX_DIM(outputTensor.xOut, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, outputTensor.xOut, return false);
    }
    return true;
}

static aclnnStatus SimpleCheckParams(
    AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor)
{
    // 1. 检查必选输入/输出是否为空指针
    CHECK_RET(SimpleCheckNotNull(inputTensor, outputTensor),
    ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入/输出的数据类型是否合法
    CHECK_RET(SimpleCheckDtypeValid(inputTensor, outputTensor),
    ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入/输出的shape大小
    CHECK_RET(SimpleCheckShapeDim(inputTensor, outputTensor),
    ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ComputeAddRmsNormQuantV1(AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor,
                ParamStruct& paramStruct, aclOpExecutor* executor)
{
    auto ret = CheckParams(inputTensor, outputTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    aclTensor* xKernelOut = nullptr;
    aclTensor* y2KernelOut = nullptr;
    aclTensor* y1KernelOut = nullptr;
    bool hasY2 = (nullptr != inputTensor.scales2Optional);
    auto addRmsNormQuantRes = l0op::AddRmsNormQuant(inputTensor.x1, inputTensor.x2, inputTensor.gamma, inputTensor.scales1, inputTensor.scales2Optional,
                                                    inputTensor.zeroPoints1Optional, inputTensor.zeroPoints2Optional,inputTensor.betaOptional,
                                                    paramStruct.axis, paramStruct.epsilon, paramStruct.divMode, executor);

    y2KernelOut = std::get<AddRmsNormQuantV2ACLNN::IDX_1>(addRmsNormQuantRes);
    y1KernelOut = std::get<AddRmsNormQuantV2ACLNN::IDX_0>(addRmsNormQuantRes);
    xKernelOut = std::get<AddRmsNormQuantV2ACLNN::IDX_2>(addRmsNormQuantRes);

    // 不支持空Tensor
    CHECK_RET(xKernelOut != nullptr && y2KernelOut != nullptr && y1KernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 y1ComputeOut 结果拷贝到 y1 上
    auto Y1ViewCopyResult = l0op::ViewCopy(y1KernelOut, outputTensor.y1Out, executor);
    CHECK_RET(Y1ViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 传了scales2的时候才拷贝y2
    if (hasY2) {
        // 将 y2KernelOut 结果拷贝到 y2 上
        auto viewCopyY2Result = l0op::ViewCopy(y2KernelOut, outputTensor.y2Out, executor);
        CHECK_RET(viewCopyY2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将 xKernelOut 结果拷贝到 x 上
    auto viewCopyXResult = l0op::ViewCopy(xKernelOut, outputTensor.xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static bool CheckDtypeValidV2(AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x1, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x2, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.gamma, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.scales1, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    if (nullptr != inputTensor.betaOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.betaOptional, ASCEND950_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }
    if (nullptr != inputTensor.zeroPoints1Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.zeroPoints1Optional, ASCEND950_DTYPE_SUPPORT_LIST_ZEROPOINT, return false);
    }

    OP_CHECK_DTYPE_NOT_MATCH(outputTensor.y1Out, op::DataType::DT_INT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(outputTensor.y2Out, op::DataType::DT_INT8, return false);  // Mandatory output
    if (nullptr != outputTensor.resOut) {
        OP_CHECK_DTYPE_NOT_SUPPORT(outputTensor.resOut, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    if (nullptr != outputTensor.xOut) {
        OP_CHECK_DTYPE_NOT_SUPPORT(outputTensor.xOut, ASCEND950_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    return true;
}

static bool CheckShapeDimV2(AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor)
{
    OP_CHECK_MAX_DIM(inputTensor.x1, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(inputTensor.gamma, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(inputTensor.scales1, MAX_SUPPORT_DIMS_NUMS, return false);
    if (nullptr != inputTensor.betaOptional) {
        OP_CHECK_MAX_DIM(inputTensor.betaOptional, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    if (nullptr != inputTensor.zeroPoints1Optional) {
        OP_CHECK_MAX_DIM(inputTensor.zeroPoints1Optional, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    OP_CHECK_MAX_DIM(outputTensor.y1Out, MAX_SUPPORT_DIMS_NUMS, return false);
    if (nullptr != outputTensor.resOut) {
        OP_CHECK_MAX_DIM(outputTensor.resOut, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    if (nullptr != outputTensor.xOut) {
        OP_CHECK_MAX_DIM(outputTensor.xOut, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    return true;
}

static aclnnStatus CheckParamsV2(AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor, int64_t& mode)
{
    if(inputTensor.betaOptional != nullptr && outputTensor.xOut != nullptr && outputTensor.resOut == nullptr) {
        mode = 1;
    } else if(outputTensor.xOut == nullptr && outputTensor.resOut != nullptr && inputTensor.betaOptional == nullptr) {
        mode = 0;
    } else {
        CHECK_RET(false, ACLNN_ERR_PARAM_NULLPTR);
    }

    // 1. 检查输入/输出的数据类型是否合法
    CHECK_RET(CheckDtypeValidV2(inputTensor, outputTensor),
    ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入/输出的shape大小
    CHECK_RET(CheckShapeDimV2(inputTensor, outputTensor),
    ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus SelfPreDealData(
    AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor, int64_t& mode,
    aclOpExecutor* executor)
{
    auto ret = CheckParamsV2(inputTensor, outputTensor, mode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (inputTensor.gamma->GetViewShape().GetDimNum() == DIM_TWO) {
        auto gammaReshape = l0op::Reshape(inputTensor.gamma, {inputTensor.gamma->GetViewShape()[1]}, executor);
        inputTensor.gamma = gammaReshape;
    }
    if (inputTensor.betaOptional != nullptr && inputTensor.betaOptional->GetViewShape().GetDimNum() == DIM_TWO) {
        auto biasReshape = l0op::Reshape(inputTensor.betaOptional, {inputTensor.betaOptional->GetViewShape()[1]}, executor);
        inputTensor.betaOptional = biasReshape;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ComputeAddRmsNormQuantV2(AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor,
                ParamStruct& paramStruct, aclOpExecutor* executor)
{
    int64_t mode = 0;   // 0为postRmsNormQuant，1为preRmsNormQuant
    auto ret = SelfPreDealData(inputTensor, outputTensor, mode, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    aclTensor* y1ComputeOut = nullptr;
    aclTensor* resComputeOut = nullptr;
    aclTensor* xComputeOut = nullptr;

    auto AddRmsNormQuantV2Outs = l0op::AddRmsNormQuantV2(inputTensor.x1, inputTensor.x2, inputTensor.gamma, inputTensor.betaOptional, inputTensor.scales1, inputTensor.scales2Optional,
                                                    inputTensor.zeroPoints1Optional, inputTensor.zeroPoints2Optional, outputTensor.xOut, outputTensor.resOut,
                                                    mode, paramStruct.epsilon, paramStruct.divMode, executor);
    y1ComputeOut = std::get<IDX_0>(AddRmsNormQuantV2Outs);
    xComputeOut = std::get<IDX_2>(AddRmsNormQuantV2Outs);
    resComputeOut = std::get<IDX_3>(AddRmsNormQuantV2Outs);

    // 将 y1ComputeOut 结果拷贝到 y1 上
    auto viewCopyY1Result = l0op::ViewCopy(y1ComputeOut, outputTensor.y1Out, executor);
    CHECK_RET(viewCopyY1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (outputTensor.xOut != nullptr) {
        // 将 xComputeOut 结果拷贝到 x 上
        auto viewCopyXResult = l0op::ViewCopy(xComputeOut, outputTensor.xOut, executor);
        CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (outputTensor.resOut != nullptr) {
        // 将 xComputeOut 结果拷贝到 x 上
        auto viewCopyResResult = l0op::ViewCopy(resComputeOut, outputTensor.resOut, executor);
        CHECK_RET(viewCopyResResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

const aclTensor* GetTensorContiguousV2(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

bool CheckSupportV2(AddRmsNormQuantV2InputTensor& inputTensor, AddRmsNormQuantV2OutputTensor& outputTensor)
{
    if ((GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) &&
        (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) &&
        (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P)) {
        return false;
    }

    if (inputTensor.gamma->GetViewShape().GetDimNum() == 1 && inputTensor.gamma->GetViewShape().GetDim(0) == 1 && inputTensor.betaOptional == nullptr &&
            outputTensor.resOut == nullptr) {
        return false;
    }
    if (inputTensor.zeroPoints1Optional != nullptr && inputTensor.zeroPoints1Optional->GetViewShape().GetDimNum() == 1 && inputTensor.zeroPoints1Optional->GetViewShape().GetDim(0) == 1 &&
            inputTensor.scales1->GetViewShape().GetDimNum() == 1 && inputTensor.scales1->GetViewShape().GetDim(0) == 1 && ((inputTensor.gamma->GetViewShape().GetDimNum() == DIM_TWO &&
            inputTensor.gamma->GetViewShape().GetDim(0) == 1 && inputTensor.gamma->GetViewShape().GetDim(1) == inputTensor.x1->GetViewShape().GetDim(inputTensor.x1->GetViewShape().GetDimNum() - 1)) ||
            (inputTensor.gamma->GetViewShape().GetDimNum() == 1 && inputTensor.gamma->GetViewShape().GetDim(0) == inputTensor.x1->GetViewShape().GetDim(inputTensor.x1->GetViewShape().GetDimNum() - 1)))) {
        return true;
    }
    return false;
}

void SpecialTransform(AddRmsNormQuantV2InputTensor& inputTensor, aclOpExecutor* executor) {
    if(inputTensor.scales1 != nullptr && inputTensor.scales1->GetDataType() == op::DataType::DT_FLOAT16) {
        auto scales1Cast = l0op::Cast(inputTensor.scales1, op::DataType::DT_FLOAT, executor);
        inputTensor.scales1 = scales1Cast;
    }
    if(inputTensor.zeroPoints1Optional != nullptr && inputTensor.zeroPoints1Optional->GetDataType() == op::DataType::DT_INT8) {
        auto zeroPoints1Cast = l0op::Cast(inputTensor.zeroPoints1Optional, op::DataType::DT_INT32, executor);
        inputTensor.zeroPoints1Optional = zeroPoints1Cast;
    }
}
} // namespace AddRmsNormQuantV2ACLNN

aclnnStatus aclnnAddRmsNormQuantV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma,
    const aclTensor* scales1, const aclTensor* scales2Optional, const aclTensor* zeroPoints1Optional,
    const aclTensor* zeroPoints2Optional, const aclTensor* betaOptional,
    int64_t axis, double epsilon, bool divMode,
    aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* rmsNormOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddRmsNormQuantV2GetWorkspaceSize.");
    L2_DFX_PHASE_1(aclnnAddRmsNormQuantV2,
                   DFX_IN(x1, x2, gamma, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional,
                          betaOptional, axis, epsilon, divMode),
                   DFX_OUT(y1Out, y2Out, xOut, rmsNormOut));

    // 创建OpExecutor
    auto uniqueExecutorForV2 = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutorForV2.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    AddRmsNormQuantV2ACLNN::AddRmsNormQuantV2InputTensor inputTensorOri = {x1, x2, gamma, betaOptional, scales1, scales2Optional, zeroPoints1Optional, zeroPoints2Optional};
    AddRmsNormQuantV2ACLNN::SpecialTransform(inputTensorOri, uniqueExecutorForV2.get());
    AddRmsNormQuantV2ACLNN::AddRmsNormQuantV2OutputTensor outputTensor = {y1Out, y2Out, rmsNormOut, xOut};
    auto ret = AddRmsNormQuantV2ACLNN::SimpleCheckParams(inputTensorOri, outputTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 支持空tensor
    bool anyEmptyTensor = x1->IsEmpty() || gamma->IsEmpty() || y2Out->IsEmpty();
    if (anyEmptyTensor) {
        OP_LOGW("Got empty tensor in aclnnAddRmsNormQuantV2!");
        *workspaceSize = 0;
        uniqueExecutorForV2.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto x1Cont = l0op::Contiguous(x1, uniqueExecutorForV2.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutorForV2.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutorForV2.get());

    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto biasCont = AddRmsNormQuantV2ACLNN::GetTensorContiguousV2(betaOptional, uniqueExecutorForV2.get());
    auto s1Cont = AddRmsNormQuantV2ACLNN::GetTensorContiguousV2(inputTensorOri.scales1, uniqueExecutorForV2.get());
    auto s2Cont = AddRmsNormQuantV2ACLNN::GetTensorContiguousV2(scales2Optional, uniqueExecutorForV2.get());
    auto z1Cont = AddRmsNormQuantV2ACLNN::GetTensorContiguousV2(inputTensorOri.zeroPoints1Optional, uniqueExecutorForV2.get());
    auto z2Cont = AddRmsNormQuantV2ACLNN::GetTensorContiguousV2(zeroPoints2Optional, uniqueExecutorForV2.get());

    AddRmsNormQuantV2ACLNN::AddRmsNormQuantV2InputTensor inputTensor = {x1Cont, x2Cont, gammaCont, biasCont, s1Cont, s2Cont, z1Cont, z2Cont};
    AddRmsNormQuantV2ACLNN::ParamStruct paramStruct = {axis, epsilon, divMode};

    if (AddRmsNormQuantV2ACLNN::CheckSupportV2(inputTensor, outputTensor)) {
        ret = AddRmsNormQuantV2ACLNN::ComputeAddRmsNormQuantV2(inputTensor, outputTensor, paramStruct, uniqueExecutorForV2.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else if (outputTensor.xOut != nullptr) {
        ret = AddRmsNormQuantV2ACLNN::ComputeAddRmsNormQuantV1(inputTensor, outputTensor, paramStruct, uniqueExecutorForV2.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        CHECK_RET(outputTensor.xOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutorForV2->GetWorkspaceSize();
    uniqueExecutorForV2.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddRmsNormQuantV2GetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormQuantV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddRmsNormQuantV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
