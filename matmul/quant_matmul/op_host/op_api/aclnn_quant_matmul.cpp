/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quant_matmul.cpp
 * \brief
 */

#include "aclnn_quant_matmul.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/dot.h"
#include "level0/fill.h"
#include "matmul/common/op_host/op_api/matmul.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"
#include "quantBatchmatmul.h"

#include "matmul/common/op_host/op_api/matmul_util.h"

using namespace Ops::NN;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
using tupleTensor = std::tuple<const aclTensor*, const aclTensor*, const aclTensor*>;

namespace {
static constexpr int INDEX_X1 = 0;
static constexpr int INDEX_X2 = 1;
static constexpr int INDEX_BIAS = 2;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {DataType::DT_INT32};
static const std::initializer_list<op::DataType> DEQSCALE_DTYPE_SUPPORT_LIST = {DataType::DT_UINT64};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const size_t X1_X2_MIN_SUPPORT_DIMS_NUMS = 2;
static const size_t X1_X2_MAX_SUPPORT_DIMS_NUM = 3;
static const size_t BIAS_DEQSCALE_SUPPORT_DIMS_NUM = 1;
static const size_t ALIGN_NUM = 16;
static const uint64_t INPUT_DIM_MIN_VALUE = 2;
static constexpr int64_t OUTPUT_INFER_FAIL = -1;
static const int PENULTIMATE_DIM = 2;

inline static void FirstPrint(bool& isFirstToPrint, const char* interfaceName, bool isExecutorInferface)
{
    if (isFirstToPrint) {
        if (isExecutorInferface) {
            OP_LOGW("%s is deprecated, use aclnnQuantMatmulV4 instead", interfaceName);
        } else {
            OP_LOGW("%s is deprecated, use aclnnQuantMatmulV4GetWorkspaceSize instead", interfaceName);
        }
        isFirstToPrint = false;
    }
}

inline static bool IsFormatSupport(const aclTensor* input, const std::string& inputName)
{
    OP_CHECK_NULL(input, return false);
    if (input->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s's format should be ND, but actually is %s.", inputName.c_str(),
            op::ToString(input->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

inline static bool IsDimSupport(const aclTensor* input, std::vector<uint64_t>& dimRange, const std::string& inputName)
{
    OP_CHECK_NULL(input, return false);
    if (input->GetViewShape().GetDimNum() < dimRange[0] || input->GetViewShape().GetDimNum() > dimRange[1]) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s's dim-num should be in the range of [%lu, %lu], but actually is %zu.",
            inputName.c_str(), dimRange[0], dimRange[1], input->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

inline static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* deqScale, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(deqScale, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* deqScale, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_DTYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(deqScale, DEQSCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

inline static bool CheckFormatVaild(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* deqScale, const aclTensor* out)
{
    CHECK_RET(IsFormatSupport(x1, "x1"), false);
    CHECK_RET(IsFormatSupport(x2, "x2"), false);
    if (bias != nullptr) {
        CHECK_RET(IsFormatSupport(bias, "bias"), false);
    }
    CHECK_RET(IsFormatSupport(deqScale, "deqScale"), false);
    CHECK_RET(IsFormatSupport(out, "out"), false);
    return true;
}

inline static bool CheckDimVaild(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias)
{
    std::vector<uint64_t> dimRangeWithoutBatch = {X1_X2_MIN_SUPPORT_DIMS_NUMS, X1_X2_MAX_SUPPORT_DIMS_NUM};
    CHECK_RET(IsDimSupport(x1, dimRangeWithoutBatch, "x1"), false);
    CHECK_RET(IsDimSupport(x2, dimRangeWithoutBatch, "x2"), false);
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    // 检查x1与x2是否维度一致
    if (x1Shape.GetDimNum() != x2Shape.GetDimNum()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The tensor shapes of x1 and x2, %s and %s, have different dim-num",
            op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
    }
    // bias是否只支持1维
    if (bias != nullptr) {
        OP_CHECK_WRONG_DIMENSION(bias, BIAS_DEQSCALE_SUPPORT_DIMS_NUM, return false);
    }
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
    size_t vaildOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        auto shortDimValue = i < vaildOffset ? 1 : shortShapeTensor->GetViewShape().GetDim(i - vaildOffset);
        auto longDimValue = longShapeTensor->GetViewShape().GetDim(i);
        if (shortDimValue > 1 && longDimValue > 1 && shortDimValue != longDimValue) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "current short dim value %ld and long dim value %ld are not supported for boardcasting", shortDimValue,
                longDimValue);
            return OUTPUT_INFER_FAIL;
        }
        int64_t curBatchValue = static_cast<int64_t>(std::max(shortDimValue, longDimValue));
        inferedOutbatchValue = inferedOutbatchValue * curBatchValue;
        batchRecord.push_back(curBatchValue);
    }
    return inferedOutbatchValue;
}

static inline bool CheckOutShape(
    const aclTensor* out, int64_t x1MDim, int64_t x2NDim, const std::vector<int64_t>& batchRecord)
{
    auto outDimNum = out->GetViewShape().GetDimNum();
    int64_t outMDim = out->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    int64_t outNDim = out->GetViewShape().GetDim(outDimNum - 1);
    size_t inferedOutDimNum = batchRecord.size() + 2;
    if (inferedOutDimNum != outDimNum) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Infered output dim num %zu is not equal to actual out dim num %zu.",
            inferedOutDimNum, outDimNum);
        return false;
    }
    if (outMDim != x1MDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Out first dim should be equal to x1 m dim, but out first dim is %ld, x1 m is %ld.", outMDim, x1MDim);
        return false;
    }
    if (outNDim != x2NDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Out second dim should be equal to x2 n dim, but out second dim is %ld, x2 n is %ld.", outNDim, x2NDim);
        return false;
    }
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        if (out->GetViewShape().GetDim(i) != batchRecord[i]) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Output dim %ld is not equal to infered output dim num %ld at dim index %zu.",
                out->GetViewShape().GetDim(i), batchRecord[i], i);
            return false;
        }
    }
    return true;
}

static inline std::tuple<int64_t, int64_t, int64_t, int64_t> GetX1X2DimValue(
    const aclTensor* x1, const aclTensor* x2, bool adjX1, bool adjX2)
{
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim = adjX1 ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
    int64_t x1MDim = adjX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
    int64_t x2KDim = adjX2 ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = adjX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    return std::tie(x1KDim, x1MDim, x2KDim, x2NDim);
}

static bool CheckShapeValid(tupleTensor mandatoryTensors, bool adjX1, bool adjX2, const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS>(mandatoryTensors);

    int64_t x1KDim, x1MDim, x2KDim, x2NDim;
    std::tie(x1KDim, x1MDim, x2KDim, x2NDim) = GetX1X2DimValue(x1, x2, adjX1, adjX2);

    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    size_t dimTensorX = x1Shape.GetDimNum();

    // 检查x1与x2 batch是否一致
    for (size_t i = 0; i < dimTensorX - INPUT_DIM_MIN_VALUE; i++) {
        if (x1Shape[i] != x2Shape[i]) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "The tensor shapes of x1 and x2, %s and %s, have different batch-dim at shape index %zu",
                op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString(), i);
            return false;
        }
    }

    // 检查x1K与x2K是否相等
    if (x1KDim != x2KDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The tensor shapes of x1 and x2, %s and %s, have different k-dim",
            op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    }

    // 检查bias是否与n一致
    if (bias != nullptr) {
        op::Shape biasShape = bias->GetViewShape();
        int64_t biasNDim = biasShape[0]; // 0: 取bias的N
        OP_CHECK(
            biasNDim == x2NDim,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "bias 1st-dim should be equal to x2 n-dim %ld, but actually is %ld.", x2NDim,
                biasNDim),
            return false);
    }

    std::vector<int64_t> batchRecord;
    int64_t inferedOutbatchValue = InferOutputShape(x1, x2, batchRecord);
    if (inferedOutbatchValue == OUTPUT_INFER_FAIL) {
        return false;
    }

    CHECK_RET(CheckOutShape(out, x1MDim, x2NDim, batchRecord), false);
    return true;
}

static bool CheckShapeValidV2(
    tupleTensor mandatoryTensors, const aclTensor* deqScale, bool adjX1, bool adjX2, const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS>(mandatoryTensors);
    // x1、x2当前只支持2维或者3维
    OP_CHECK_MIN_DIM(x1, X1_X2_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(x2, X1_X2_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(x1, X1_X2_MAX_SUPPORT_DIMS_NUM, return false);
    OP_CHECK_MAX_DIM(x2, X1_X2_MAX_SUPPORT_DIMS_NUM, return false);

    // bias、deqScale当前只支持1维
    if (bias != nullptr) {
        OP_CHECK_WRONG_DIMENSION(bias, BIAS_DEQSCALE_SUPPORT_DIMS_NUM, return false);
    }
    OP_CHECK_WRONG_DIMENSION(deqScale, BIAS_DEQSCALE_SUPPORT_DIMS_NUM, return false);

    CHECK_RET(CheckShapeValid(mandatoryTensors, adjX1, adjX2, out), false);

    // deqScale的shape（t,)，t必须是x2的最后一维大小n按照16向上取整
    auto x2Dims = x2->GetViewShape().GetDimNum();
    auto nIndex = adjX2 ? 2 : 1; // B矩阵转置，n为倒数第2根轴
    auto n = x2->GetViewShape().GetDim(x2Dims - nIndex);
    auto t = deqScale->GetViewShape().GetDim(0);
    OP_LOGD("n is %ld, deqScale shape is %ld, transA: %d, transB: %d", n, t, adjX1, adjX2);
    if ((n + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM != t) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "deqScale shape %s is invalid.",
            op::ToString(deqScale->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckSupportSocVersion()
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        return ACLNN_SUCCESS;
    }
    OP_LOGE(
        ACLNN_ERR_RUNTIME_ERROR, "QuantBatchMatmul/V2 support for %s is not implemented",
        op::ToString(socVersion).GetString());
    return ACLNN_ERR_RUNTIME_ERROR;
}

inline static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* deqScale, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, deqScale, out), ACLNN_ERR_INNER_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x1, x2, bias, deqScale, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的数据格式是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckFormatVaild(x1, x2, bias, deqScale, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查输入的数据维度是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDimVaild(x1, x2, bias), ACLNN_ERR_PARAM_INVALID);
    // 5. 检查Shape是否支持
    CHECK_RET(CheckShapeValid(std::tie(x1, x2, bias), false, false, out), ACLNN_ERR_PARAM_INVALID);
    // 6. 检查芯片是否支持
    CHECK_RET(CheckSupportSocVersion() != ACLNN_ERR_RUNTIME_ERROR, ACLNN_ERR_RUNTIME_ERROR);
    return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckParamsV2(
    tupleTensor mandatoryTensors, const aclTensor* deqScale, bool adjX1, bool adjX2, const aclTensor* out)
{
    auto x1 = std::get<INDEX_X1>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2>(mandatoryTensors);
    auto bias = std::get<INDEX_BIAS>(mandatoryTensors);
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, deqScale, out), ACLNN_ERR_INNER_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x1, x2, bias, deqScale, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的数据格式是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckFormatVaild(x1, x2, bias, deqScale, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Shape是否支持
    CHECK_RET(CheckShapeValidV2(mandatoryTensors, deqScale, adjX1, adjX2, out), ACLNN_ERR_PARAM_INVALID);
    // 5. 检查芯片是否支持
    CHECK_RET(CheckSupportSocVersion() != ACLNN_ERR_RUNTIME_ERROR, ACLNN_ERR_RUNTIME_ERROR);
    return ACLNN_SUCCESS;
}

static const aclTensor* ProcessEmptyTensor(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* out, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    op::Shape outShape = out->GetViewShape();
    auto output = executor->AllocTensor(outShape, x1->GetDataType());
    OP_CHECK(!output->IsEmpty(), OP_LOGI("Return an empty tensor without actually doing allocation"), return output);
    if (bias != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "k(reduce) is not supported to be equal to 0 when bias exists");
        return nullptr;
    }
    FVector<int64_t> fillShape = GetShape(output);
    const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

static const aclTensor* BuildQuantMatMulGraph(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* deqScale, bool adjX1, bool adjX2,
    aclTensor* out, aclOpExecutor* executor)
{
    // 空tensor 处理
    if (x1->IsEmpty() || x2->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensor(x1, x2, bias, out, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        return emptyOut;
    }
    const aclTensor* matmulOut = l0op::QuantBatchMatmul(x1, x2, bias, deqScale, adjX1, adjX2, executor);
    CHECK_RET(matmulOut != nullptr, nullptr);
    // Reshape to out shape
    auto matReshape = l0op::Reshape(matmulOut, out->GetViewShape(), executor);
    CHECK_RET(matReshape != nullptr, nullptr);

    return matReshape;
}
} // namespace

aclnnStatus aclnnQuantMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, float deqScale, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    static bool isFirstToPrint = true;
    FirstPrint(isFirstToPrint, "aclnnQuantMatmulGetWorkspaceSize", false);
    L2_DFX_PHASE_1(aclnnQuantMatmul, DFX_IN(x1, x2, bias, deqScale), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 实现deqScale的转义
    auto fixpipeDeqScale = TransDequantScaleToM1(deqScale);
    OP_LOGD("anti_quant_scale is %f, fixpipeDeqScale is %lx", deqScale, fixpipeDeqScale);
    // 生成一个deqScaleTensor
    auto deqScaleValue = (unique_executor.get())->AllocScalar(static_cast<int64_t>(fixpipeDeqScale));
    CHECK_RET(deqScaleValue != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto deqScaleTensor = (unique_executor.get())->ConvertToTensor(deqScaleValue, DataType::DT_UINT64);
    CHECK_RET(deqScaleTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 修改deqScaleTensor的shape和format
    auto deqScaleViewShape = op::Shape{1, 1, 1, 1};
    auto deqScaleStorageShape = op::Shape{1, 1, 1, 1, 1};
    const_cast<aclTensor*>(deqScaleTensor)->SetOriginalShape(deqScaleViewShape);
    const_cast<aclTensor*>(deqScaleTensor)->SetViewShape(deqScaleViewShape);
    const_cast<aclTensor*>(deqScaleTensor)->SetStorageShape(deqScaleStorageShape);
    const_cast<aclTensor*>(deqScaleTensor)->SetOriginalFormat(op::Format::FORMAT_NHWC);
    const_cast<aclTensor*>(deqScaleTensor)->SetViewFormat(op::Format::FORMAT_NHWC);
    const_cast<aclTensor*>(deqScaleTensor)->SetStorageFormat(op::Format::FORMAT_NC1HWC0);

    // 入参检查
    auto ret = CheckParams(x1, x2, bias, deqScaleTensor, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 构建matmul计算图
    auto matmulOut = BuildQuantMatMulGraph(x1, x2, bias, deqScaleTensor, false, false, out, unique_executor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (matmulOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        unique_executor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto viewCopyResult = l0op::ViewCopy(matmulOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace
    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    static bool isFirstToPrint = true;
    FirstPrint(isFirstToPrint, "aclnnQuantMatmul", true);
    L2_DFX_PHASE_2(aclnnQuantMatmul);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnQuantMatmulV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* deqScale, bool adjX1, bool adjX2,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    static bool isFirstToPrint = true;
    FirstPrint(isFirstToPrint, "aclnnQuantMatmulV2GetWorkspaceSize", false);
    L2_DFX_PHASE_1(aclnnQuantMatmulV2, DFX_IN(x1, x2, bias, deqScale, adjX1, adjX2), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto ret = CheckParamsV2(std::tie(x1, x2, bias), deqScale, adjX1, adjX2, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 修改deqScale的shape和format
    auto t = deqScale->GetViewShape().GetDim(0);
    auto deqScaleViewShape = op::Shape{1, 1, 1, t};
    auto deqScaleStorageShape = op::Shape{1, static_cast<long>(t / ALIGN_NUM), 1, 1, ALIGN_NUM};
    const_cast<aclTensor*>(deqScale)->SetOriginalShape(deqScaleViewShape);
    const_cast<aclTensor*>(deqScale)->SetViewShape(deqScaleViewShape);
    const_cast<aclTensor*>(deqScale)->SetStorageShape(deqScaleStorageShape);
    const_cast<aclTensor*>(deqScale)->SetOriginalFormat(op::Format::FORMAT_NHWC);
    const_cast<aclTensor*>(deqScale)->SetViewFormat(op::Format::FORMAT_NHWC);
    const_cast<aclTensor*>(deqScale)->SetStorageFormat(op::Format::FORMAT_NC1HWC0);

    // 构建matmul计算图
    auto matmulOut = BuildQuantMatMulGraph(x1, x2, bias, deqScale, adjX1, adjX2, out, unique_executor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (matmulOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        unique_executor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto viewCopyResult = l0op::ViewCopy(matmulOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace
    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    static bool isFirstToPrint = true;
    FirstPrint(isFirstToPrint, "aclnnQuantMatmulV2", true);
    L2_DFX_PHASE_2(aclnnQuantMatmulV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif