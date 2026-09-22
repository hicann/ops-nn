/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include "activation/cla_gate_quant/op_api/aclnn_cla_gate_quant.h"
#include "activation/cla_gate_quant/op_api/cla_gate_quant.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "op_common/log/log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t MIN_N = 1;
constexpr int64_t MAX_N = 128;
constexpr int64_t ROW_BLOCK = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t SCALE_ALG_OCP = 0;
constexpr int64_t SCALE_ALG_CUBLAS = 1;
constexpr int64_t FP8_E4M3FN = 36;
constexpr int64_t FP8_E5M2 = 35;
constexpr int64_t FP4_E2M1 = 40;
constexpr int64_t FP4_E1M2 = 41;
constexpr const char* INPUT_ATTN_LAYOUT_TND = "TND";
constexpr const char* OP_NAME = "aclnnClaGateQuant";

static bool CheckNotNull(const aclTensor* globalAttn, const aclTensor* localAttn, const aclTensor* globalGateLogits,
                         const aclTensor* localGateLogits, const aclTensor* rowData, const aclTensor* rowScale,
                         const aclTensor* colData, const aclTensor* colScale, bool dualAxisFlag)
{
    OP_CHECK_NULL(globalAttn, return false);
    OP_CHECK_NULL(localAttn, return false);
    OP_CHECK_NULL(globalGateLogits, return false);
    OP_CHECK_NULL(localGateLogits, return false);
    OP_CHECK_NULL(rowData, return false);
    OP_CHECK_NULL(rowScale, return false);
    // Single-axis mode must pass null col outputs; dual-axis mode must provide them.
    if (dualAxisFlag) {
        OP_CHECK_NULL(colData, return false);
        OP_CHECK_NULL(colScale, return false);
    } else if (colData != nullptr || colScale != nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "colData/colScale",
                                                 "both outputs must be null when dualAxisFlag is false");
        return false;
    }
    return true;
}

static bool IsRoundModeValid(const char* mode)
{
    return mode == nullptr || std::strcmp(mode, "rint") == 0 || std::strcmp(mode, "round") == 0 ||
           std::strcmp(mode, "floor") == 0;
}

static bool IsDstTypeValid(int64_t dstType)
{
    return dstType == FP8_E5M2 || dstType == FP8_E4M3FN || dstType == FP4_E2M1 || dstType == FP4_E1M2;
}

static bool IsInputAttnLayoutValid(const char* inputAttnLayout)
{
    return inputAttnLayout == nullptr || std::strcmp(inputAttnLayout, INPUT_ATTN_LAYOUT_TND) == 0;
}

static bool CheckDtypeValid(const aclTensor* globalAttn, const aclTensor* localAttn, const aclTensor* globalGateLogits,
                            const aclTensor* localGateLogits, const aclTensor* rowData, const aclTensor* rowScale,
                            const aclTensor* colData, const aclTensor* colScale, int64_t dstType, bool dualAxisFlag)
{
    const std::initializer_list<op::DataType> inDtypeList = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
    OP_CHECK_DTYPE_NOT_SUPPORT(globalAttn, inDtypeList, return false);
    OP_CHECK_DTYPE_NOT_SAME(localAttn, globalAttn, return false);
    OP_CHECK_DTYPE_NOT_SAME(globalGateLogits, globalAttn, return false);
    OP_CHECK_DTYPE_NOT_SAME(localGateLogits, globalAttn, return false);
    if (!IsDstTypeValid(dstType)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "dstType", std::to_string(dstType).c_str(),
                                  "35(FLOAT8_E5M2), 36(FLOAT8_E4M3FN), 40(FLOAT4_E2M1) or 41(FLOAT4_E1M2)");
        return false;
    }
    const auto outputDtype = static_cast<op::DataType>(dstType);
    OP_CHECK_DTYPE_NOT_MATCH(rowData, outputDtype, return false);
    OP_CHECK_DTYPE_NOT_MATCH(rowScale, op::DataType::DT_FLOAT8_E8M0, return false);
    if (dualAxisFlag) {
        OP_CHECK_DTYPE_NOT_MATCH(colData, outputDtype, return false);
        OP_CHECK_DTYPE_NOT_MATCH(colScale, op::DataType::DT_FLOAT8_E8M0, return false);
    }
    return true;
}

static bool CheckShape(const aclTensor* globalAttn, const aclTensor* localAttn, const aclTensor* globalGateLogits,
                       const aclTensor* localGateLogits)
{
    auto globalAttnShape = globalAttn->GetViewShape();
    auto globalGateShape = globalGateLogits->GetViewShape();
    if (globalAttnShape.GetDimNum() != 3 || globalAttnShape.GetDim(0) <= 0 || globalAttnShape.GetDim(1) < MIN_N ||
        globalAttnShape.GetDim(1) > MAX_N || (globalAttnShape.GetDim(2) != 128 && globalAttnShape.GetDim(2) != 256)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "globalAttn", op::ToString(globalAttnShape).GetString(),
            "the shape must be [T,N,D], T must be positive, N must be in [1,128], and D must be 128 or 256");
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(localAttn, globalAttn, return false);
    auto gateValid = [&](const op::Shape& gate) {
        return gate.GetDimNum() == 2 && gate.GetDim(0) == globalAttnShape.GetDim(0) &&
               gate.GetDim(1) == globalAttnShape.GetDim(1);
    };
    if (!gateValid(globalGateShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "globalGateLogits", op::ToString(globalGateShape).GetString(),
                                              "the shape must be [T,N] and match globalAttn");
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(localGateLogits, globalGateLogits, return false);
    return true;
}

static bool CheckOutputShape(const aclTensor* rowData, const aclTensor* rowScale, const aclTensor* colData,
                             const aclTensor* colScale, int64_t rowCount, int64_t columnCount, bool dualAxisFlag)
{
    op::Shape expectedRowDataShape = {rowCount, columnCount};
    op::Shape expectedRowScaleShape = {rowCount, (columnCount + ROW_BLOCK - 1) / ROW_BLOCK, SCALE_LAST_DIM};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(rowData, expectedRowDataShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(rowScale, expectedRowScaleShape, return false);
    if (!dualAxisFlag) {
        return true;
    }
    op::Shape expectedColDataShape = {rowCount, columnCount};
    op::Shape expectedColScaleShape = {(rowCount + ROW_BLOCK - 1) / ROW_BLOCK, columnCount, SCALE_LAST_DIM};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(colData, expectedColDataShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(colScale, expectedColScaleShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* globalAttn, const aclTensor* localAttn,
                               const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                               const char* roundMode, int64_t scaleAlg, int64_t dstType, const char* inputAttnLayout,
                               bool dualAxisFlag, const aclTensor* rowData, const aclTensor* rowScale,
                               const aclTensor* colData, const aclTensor* colScale)
{
    CHECK_RET(CheckNotNull(globalAttn, localAttn, globalGateLogits, localGateLogits, rowData, rowScale, colData,
                           colScale, dualAxisFlag),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(globalAttn, localAttn, globalGateLogits, localGateLogits, rowData, rowScale, colData,
                              colScale, dstType, dualAxisFlag),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(globalAttn, localAttn, globalGateLogits, localGateLogits), ACLNN_ERR_PARAM_INVALID);
    if (!IsRoundModeValid(roundMode)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "roundMode", roundMode, "rint, round or floor");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsInputAttnLayoutValid(inputAttnLayout)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "inputAttnLayout", inputAttnLayout, INPUT_ATTN_LAYOUT_TND);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (scaleAlg != SCALE_ALG_OCP && scaleAlg != SCALE_ALG_CUBLAS) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "scaleAlg", std::to_string(scaleAlg).c_str(), "0 or 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((dstType == FP4_E2M1 || dstType == FP4_E1M2) && scaleAlg != SCALE_ALG_OCP) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "scaleAlg", std::to_string(scaleAlg).c_str(), "0 for FP4 output");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((dstType == FP8_E4M3FN || dstType == FP8_E5M2) && roundMode != nullptr && std::strcmp(roundMode, "rint") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "roundMode", roundMode, "rint for FP8 output");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto globalAttnShape = globalAttn->GetViewShape();
    int64_t rowCount = globalAttnShape.GetDim(0);
    int64_t columnCount = globalAttnShape.GetDim(1) * globalAttnShape.GetDim(2);
    if ((dstType == FP4_E2M1 || dstType == FP4_E1M2) && (columnCount % 4) != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "K", std::to_string(columnCount).c_str(),
                                              "K = N * D must be divisible by four for FP4 output");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckOutputShape(rowData, rowScale, colData, colScale, rowCount, columnCount, dualAxisFlag),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnClaGateQuantGetWorkspaceSize(const aclTensor* globalAttn, const aclTensor* localAttn,
                                              const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                                              const char* roundMode, int64_t scaleAlg, int64_t dstType,
                                              const char* inputAttnLayout, bool dualAxisFlag,
                                              const aclTensor* rowDataOut, const aclTensor* rowScaleOut,
                                              const aclTensor* colDataOut, const aclTensor* colScaleOut,
                                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnClaGateQuant,
                   DFX_IN(globalAttn, localAttn, globalGateLogits, localGateLogits, roundMode, scaleAlg, dstType,
                          inputAttnLayout, dualAxisFlag),
                   DFX_OUT(rowDataOut, rowScaleOut, colDataOut, colScaleOut));

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(globalAttn, localAttn, globalGateLogits, localGateLogits, roundMode, scaleAlg, dstType,
                           inputAttnLayout, dualAxisFlag, rowDataOut, rowScaleOut, colDataOut, colScaleOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto globalAttnContiguous = l0op::Contiguous(globalAttn, uniqueExecutor.get());
    CHECK_RET(globalAttnContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto localAttnContiguous = l0op::Contiguous(localAttn, uniqueExecutor.get());
    CHECK_RET(localAttnContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto globalGateContiguous = l0op::Contiguous(globalGateLogits, uniqueExecutor.get());
    CHECK_RET(globalGateContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto localGateContiguous = l0op::Contiguous(localGateLogits, uniqueExecutor.get());
    CHECK_RET(localGateContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ClaGateQuant(globalAttnContiguous, localAttnContiguous, globalGateContiguous,
                                     localGateContiguous, roundMode, scaleAlg, dstType, inputAttnLayout, dualAxisFlag,
                                     uniqueExecutor.get());
    const aclTensor* rowDataResult = std::get<0>(result);
    const aclTensor* rowScaleResult = std::get<1>(result);
    const aclTensor* colDataResult = std::get<2>(result);
    const aclTensor* colScaleResult = std::get<3>(result);
    OP_CHECK_NULL(rowDataResult, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(rowScaleResult, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(colDataResult, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(colScaleResult, return ACLNN_ERR_INNER_NULLPTR);

    auto viewRowData = l0op::ViewCopy(rowDataResult, rowDataOut, uniqueExecutor.get());
    CHECK_RET(viewRowData != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewRowScale = l0op::ViewCopy(rowScaleResult, rowScaleOut, uniqueExecutor.get());
    CHECK_RET(viewRowScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (dualAxisFlag) {
        auto viewColData = l0op::ViewCopy(colDataResult, colDataOut, uniqueExecutor.get());
        CHECK_RET(viewColData != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewColScale = l0op::ViewCopy(colScaleResult, colScaleOut, uniqueExecutor.get());
        CHECK_RET(viewColScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnClaGateQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClaGateQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
