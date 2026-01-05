/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/small_vector.h"
#include "opdev/platform.h"
#include "quantize.h"
#include "aclnn_quantize.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
constexpr int64_t INDEX_FOR_X = 0;
constexpr int64_t INDEX_FOR_SCALES = 1;
constexpr int64_t INDEX_FOR_ZERO_POINTS = 2;
constexpr int64_t INDEX_FOR_OUT = 3;
constexpr int64_t DIM_MAX_NUM = 8;

const inline std::map<int64_t, std::initializer_list<op::DataType>>& GetSocSupportDtypeMap(SocVersion socVersion)
{
    static const std::map<int64_t, std::initializer_list<op::DataType>> emptyDtypesMap = {
        {INDEX_FOR_X, {}}, {INDEX_FOR_SCALES, {}}, {INDEX_FOR_ZERO_POINTS, {}}, {INDEX_FOR_OUT, {}}};

    static const std::initializer_list<op::DataType> DTYPE_SUPPORT_FLOAT_16_LIST = {
        op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
    static const std::initializer_list<op::DataType> DTYPE_SUPPORT_INT_LIST = {
        op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT32};

    static const std::initializer_list<op::DataType> DTYPE_SUPPORT_ALLF_LIST = {
        op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

    static const std::initializer_list<op::DataType> DTYPE_SUPPORT_INT_BF16_LIST = {
        op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT32, op::DataType::DT_BF16};

    static const std::initializer_list<op::DataType> DTYPE_SUPPORT_INT_FLOAT_BF16_LIST = {
        op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT32, op::DataType::DT_FLOAT,
        op::DataType::DT_BF16};

    static const std::initializer_list<op::DataType> DTYPE_SUPPORT_A5_FOR_OUT_LIST = {
        op::DataType::DT_INT8,     op::DataType::DT_UINT8,         op::DataType::DT_INT32,
        op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2};

    static const std::map<SocVersion, std::map<int64_t, std::initializer_list<op::DataType>>>
        socSupportMap = {
            {SocVersion::ASCEND310P,
             {{INDEX_FOR_X, DTYPE_SUPPORT_FLOAT_16_LIST}, {INDEX_FOR_SCALES, DTYPE_SUPPORT_FLOAT_16_LIST},
              {INDEX_FOR_ZERO_POINTS, DTYPE_SUPPORT_INT_LIST}, {INDEX_FOR_OUT, DTYPE_SUPPORT_INT_LIST}}},
            {SocVersion::ASCEND910,
             {{INDEX_FOR_X, DTYPE_SUPPORT_FLOAT_16_LIST}, {INDEX_FOR_SCALES, DTYPE_SUPPORT_FLOAT_16_LIST},
              {INDEX_FOR_ZERO_POINTS, DTYPE_SUPPORT_INT_LIST}, {INDEX_FOR_OUT, DTYPE_SUPPORT_INT_LIST}}},
            {SocVersion::ASCEND910B,
             {{INDEX_FOR_X, DTYPE_SUPPORT_ALLF_LIST}, {INDEX_FOR_SCALES, DTYPE_SUPPORT_ALLF_LIST},
              {INDEX_FOR_ZERO_POINTS, DTYPE_SUPPORT_INT_BF16_LIST}, {INDEX_FOR_OUT, DTYPE_SUPPORT_INT_LIST}}},
            {SocVersion::ASCEND910_93,
             {{INDEX_FOR_X, DTYPE_SUPPORT_ALLF_LIST}, {INDEX_FOR_SCALES, DTYPE_SUPPORT_ALLF_LIST},
              {INDEX_FOR_ZERO_POINTS, DTYPE_SUPPORT_INT_BF16_LIST}, {INDEX_FOR_OUT, DTYPE_SUPPORT_INT_LIST}}},
            {SocVersion::ASCEND910_95,
             {{INDEX_FOR_X, DTYPE_SUPPORT_ALLF_LIST}, {INDEX_FOR_SCALES, DTYPE_SUPPORT_ALLF_LIST},
              {INDEX_FOR_ZERO_POINTS, DTYPE_SUPPORT_INT_FLOAT_BF16_LIST}, 
              {INDEX_FOR_OUT, DTYPE_SUPPORT_A5_FOR_OUT_LIST}}},
        };

    auto found = socSupportMap.find(socVersion);
    if (found == socSupportMap.end()) {
        return emptyDtypesMap;
    }
    return found->second;
}

static int64_t MakeWrapDim(int64_t dim, int64_t dimPostExpr)
{
    if (dimPostExpr <= 0) {
        dimPostExpr = 1;
    }
    if (dim < 0) {
        dim += dimPostExpr;
    }
    // 异常值处理
    if (dim < -dimPostExpr || dim >= dimPostExpr) {
        dim = dimPostExpr - 1;
    }
    return dim;
}

// 空指针检测
static bool CheckNotNull(const aclTensor* x, const aclTensor* scales, aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(scales, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

// 数据类型校验
static bool CheckDtypeValid(
    const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints, aclDataType dtype, const aclTensor* out)
{
    SocVersion currentVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto socSupportMap = GetSocSupportDtypeMap(currentVersion);
    auto xSupportList = socSupportMap[INDEX_FOR_X];
    auto scalesSupportList = socSupportMap[INDEX_FOR_SCALES];
    auto zeroPointSupportList = socSupportMap[INDEX_FOR_ZERO_POINTS];
    auto outSupportList = socSupportMap[INDEX_FOR_OUT];
    // 检查输入和scales的数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(x, xSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(scales, scalesSupportList, return false);
    if (zeroPoints != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(zeroPoints, zeroPointSupportList, return false);
    }
    op::DataType dtypeOP = op::ToOpDataType(dtype);
    // 检查dtype数据类型
    if (!CheckType(dtypeOP, outSupportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Param dtype %s should be in dtype support list [%s].",
            op::ToString(dtypeOP).GetString(), op::ToString(outSupportList).GetString());
        return false;
    }
    // 检查out数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(out, outSupportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, dtypeOP, return false);
    // 当scales或zeroPoints为BF16类型时，x、scales、zeroPoints数据类型都需要是BF16
    if (zeroPoints != nullptr) {
        if (zeroPoints->GetDataType() == op::DataType::DT_BF16 || scales->GetDataType() == op::DataType::DT_BF16) {
            OP_CHECK_DTYPE_NOT_MATCH(x, op::DataType::DT_BF16, return false);
            OP_CHECK_DTYPE_NOT_MATCH(scales, op::DataType::DT_BF16, return false);
            OP_CHECK_DTYPE_NOT_MATCH(zeroPoints, op::DataType::DT_BF16, return false);
        }
    } else {
        if (scales->GetDataType() == op::DataType::DT_BF16) {
            OP_CHECK_DTYPE_NOT_MATCH(x, op::DataType::DT_BF16, return false);
        }
    }
    return true;
}

// 执行Shape相关检测
static bool CheckShapeValid(
    const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints, int32_t axis, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(out, x, return false);

    auto inputShape = x->GetViewShape();
    int64_t xDim = static_cast<int64_t>(inputShape.GetDimNum());
    auto scalesShape = scales->GetViewShape();
    if (scalesShape.GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input scales shape only support 1 dimension.");
        return false;
    }
    auto scalesSize = scalesShape.GetDim(0);
    if (zeroPoints != nullptr) {
        auto zeroPointsShape = zeroPoints->GetViewShape();
        auto zeroPointsSize = zeroPointsShape.GetDim(0);
        if (zeroPointsShape.GetDimNum() != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input zeroPoints shape only support 1 dimension.");
            return false;
        }
        if (zeroPointsSize != scalesSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Scales size should be equal to zeroPoints size.");
            return false;
        }
    }
    // Per tensor 场景
    if (scalesSize == 1) {
        return true;
    }
    // Per channel/Per head 场景， axis校验
    if (axis < -xDim || axis >= xDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Axis out of range (expected to be in range of [-%ld, %ld], but got %d)", xDim,
            xDim - 1, axis);
        return false;
    }
    int64_t positiveAxis = MakeWrapDim((int64_t)axis, xDim);
    auto xAxisSize = inputShape.GetDim(positiveAxis);
    if (scalesSize != xAxisSize) {
        // Scales' shape 不满足条件
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The size of scales must equal to 1 or the specified dimension of x, which is %ld.", xAxisSize);
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints, const aclTensor* out)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(x->GetStorageFormat()) || op::IsPrivateFormat(scales->GetStorageFormat()) ||
        op::IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of x, scales or out only support ND.");
        return false;
    }

    if ((zeroPoints != nullptr) && op::IsPrivateFormat(zeroPoints->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of zeroPoints only support ND.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints, int32_t axis, aclDataType dtype,
    aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, scales, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数shape是否合法
    CHECK_RET(CheckShapeValid(x, scales, zeroPoints, axis, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查参数数据类型是否合法
    CHECK_RET(CheckDtypeValid(x, scales, zeroPoints, dtype, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查参数数据格式是否合法
    CHECK_RET(CheckFormat(x, scales, zeroPoints, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* QuantizeReshapeTensor(
    const aclTensor* x, const aclTensor* needReshape, int32_t axis, aclOpExecutor* executor)
{
    auto xShape = x->GetViewShape();
    size_t xDimNum = xShape.GetDimNum();
    auto elewiseAxisSize = needReshape->GetViewShape().GetDim(0);
    std::vector<int64_t> reshapeValue(xDimNum);
    for (size_t i = 0; i < xDimNum; i++) {
        reshapeValue[i] = (i == static_cast<size_t>(axis)) ? elewiseAxisSize : 1; // axis is larger than 0
    }
    aclIntArray* needShape = executor->AllocIntArray(reshapeValue.data(), xDimNum);
    auto reshapeTensor = l0op::Reshape(needReshape, needShape, executor);
    return reshapeTensor;
}
}; // namespace

aclnnStatus aclnnQuantizeGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints, aclDataType dtype, int32_t axis,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantize, DFX_IN(x, scales, zeroPoints, dtype, axis), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(x, scales, zeroPoints, axis, dtype, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto positiveAxis = MakeWrapDim(axis, x->GetViewShape().GetDimNum());
    // 支持空Tensor
    if (x->IsEmpty() || scales->IsEmpty() || (zeroPoints != nullptr && zeroPoints->IsEmpty()) || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 非连续转连续
    const aclTensor* xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* scalesContiguous = l0op::Contiguous(scales, uniqueExecutor.get());
    CHECK_RET(scalesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* zeroPointsContiguous = nullptr;
    const aclTensor* quantizeOut = nullptr;
    if (zeroPoints != nullptr) {
        zeroPointsContiguous = l0op::Contiguous(zeroPoints, uniqueExecutor.get());
        CHECK_RET(zeroPointsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* scalesCast = scalesContiguous;
    // 当scales是FP16类型时，将其转换成FP32
    if (scalesContiguous->GetDataType() == op::DataType::DT_FLOAT16) {
        scalesCast = l0op::Cast(scalesContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(scalesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* scalesReshape = QuantizeReshapeTensor(xContiguous, scalesCast, positiveAxis, uniqueExecutor.get());
    CHECK_RET(scalesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* zeroPointsReshape = nullptr;
    if (zeroPointsContiguous != nullptr) {
        zeroPointsReshape =
            QuantizeReshapeTensor(xContiguous, zeroPointsContiguous, positiveAxis, uniqueExecutor.get());
        CHECK_RET(zeroPointsReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    quantizeOut = l0op::Quantize(
        xContiguous, scalesReshape, zeroPointsReshape, op::ToOpDataType(dtype), positiveAxis, uniqueExecutor.get());
    CHECK_RET(quantizeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行ViewCopy
    auto viewCopyResult = l0op::ViewCopy(quantizeOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantize(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantize);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
