/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <dlfcn.h>
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/quant_batch_matmul_v3/op_api/quant_matmul_checker.h"
#include "matmul/quant_batch_matmul_v4/op_host/op_api/quant_matmul_common_check.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "matmul/quant_batch_matmul_v3/op_api/quant_matmul_v3.h"
#include "fused_quant_matmul.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_fused_quant_matmul.h"
#include "aclnn_fused_quant_matmul_weight_nz.h"
#include "util/math_util.h"

using namespace op;
using Ops::NN::SwapLastTwoDimValue;
using Ops::NN::IsTransposeLastTwoDims;

namespace {
// 校验x1、x2、x1scale、x2scale、out必须存在，校验yscale、x1offset、x2offset、yoffset、x3、groupsize不支持
static inline bool CheckInputExistence(TupleInput &inputTensors, TupleQuant &quantTensors, TupleFused &fusedTensors,
                                       const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x1Offset = std::get<INDEX_X1_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto x2Offset = std::get<INDEX_X2_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto yOffset = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_QUANT_TUPLE>(quantTensors);
    auto x3 = std::get<INDEX_X3_IN_FUSED_TUPLE>(fusedTensors);
    // 必选参数
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(x1Scale, return false);
    OP_CHECK_NULL(x2Scale, return false);
    OP_CHECK_NULL(out, return false);
    // 不支持参数
    if (yScale != nullptr && !yScale->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support yScale now, yScale should be empty tensor or nullptr.");
        return false;
    }
    if (x1Offset != nullptr && !x1Offset->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support x1Offset now, x1Offset should be empty tensor or nullptr.");
        return false;
    }
    if (x2Offset != nullptr && !x2Offset->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support x2Offset now, x2Offset should be empty tensor or nullptr.");
        return false;
    }
    if (yOffset != nullptr && !yOffset->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support yOffset now, yOffset should be empty tensor or nullptr.");
        return false;
    }
    if (x3 != nullptr && !x3->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support x3 now, x3 should be empty tensor or nullptr.");
        return false;
    }
    if (groupSize != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support groupSize now, groupSize should be 0.");
        return false;
    }
    return true;
}

static inline bool CheckDimRange(TupleInput &inputTensors, TupleQuant &quantTensors, const aclTensor *out) {
    auto x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto x2StorageFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    size_t x2MaxDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MAX_DIM_NUM_NZ : MAX_DIM_NUM_ND;
    size_t x2MinDimNum = x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ ? MIN_DIM_NUM_NZ : MIN_DIM_NUM_ND;
    size_t x2DimNum = x2->GetStorageShape().GetDimNum();
    if (x2DimNum < x2MinDimNum || x2DimNum > x2MaxDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input x2's dimension should be in range[%zu, %zu], but actual is %zu.", x2MinDimNum, x2MaxDimNum, x2DimNum);
        return false;
    }
    OP_CHECK_MIN_DIM(x1, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MIN_DIM(out, MIN_DIM_NUM_ND, return false);
    OP_CHECK_MAX_DIM(x1, MAX_DIM_NUM_ND, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_NUM_ND, return false);

    OP_CHECK_WRONG_DIMENSION(x1Scale, 1, return false);
    OP_CHECK_WRONG_DIMENSION(x2Scale, 1, return false);
    OP_LOGD("QuantMatmul check dimension range success.");
    return true;
}

static aclnnStatus CheckParams(const QuantMatmulChecker& qmmV3Checker)
{
    aclnnStatus ret = qmmV3Checker.CheckParams();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    OP_LOGD("QuantMatmul check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus PreMatmulCalcProcess(TupleInput &inputTensors, TupleQuant &quantTensors,
                                        TupleFused &fusedTensors, TupleAttr &boolsTrans, const aclTensor *out,
                                        aclOpExecutor *executor) {
    auto &x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto &x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto &x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Offset = std::get<INDEX_X2_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto &yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    auto &x3 = std::get<INDEX_X3_IN_FUSED_TUPLE>(fusedTensors);
    const char* fusedOpType = std::get<INDEX_FUSEDOPTYPE_IN_FUSED_TUPLE>(fusedTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_QUANT_TUPLE>(quantTensors);
    bool transposeX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans);

    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 校验tensor是否存在或不支持。
    CHECK_RET(CheckInputExistence(inputTensors, quantTensors, fusedTensors, out), ACLNN_ERR_PARAM_NULLPTR);
    
    // 对存在的tensor进行contiguous处理
    bool tempTransposeValue = false;
    CHECK_RET(TensorContiguousProcess(x1, transposeX1, executor), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(TensorContiguousProcess(x1Scale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(TensorContiguousProcess(x2Scale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(TensorContiguousProcess(bias, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);

    auto ret = WeightNZCaseProcess(x2, transposeX2, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(CheckDimRange(inputTensors, quantTensors, out), ACLNN_ERR_PARAM_INVALID);
    // 输入int32转int4处理
    A4W4CaseProcess(x1, x2, executor);
    if (x2->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        op::Shape x2StorageShape = GetWeightNzShape(x2, transposeX2);
        x2->SetStorageShape(x2StorageShape);
    }
    return ACLNN_SUCCESS;
}

// 校验fusedOpType是否合法
bool CheckFusedOpType(const char* fusedOpType)
{
    if (std::strcmp(fusedOpType, "gelu_erf") != 0 && std::strcmp(fusedOpType, "gelu_tanh") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fusedOpType must be in the type of gelu_erf/gelu_tanh");
        return false;
    }
    return true;
}

bool CheckInputDtypeValid(const aclTensor *&x1, const aclTensor *&x2)
{   
    bool isA8W8 = x1->GetDataType() == DataType::DT_INT8 && x2->GetDataType() == DataType::DT_INT8;
    bool isA4W4 = x1->GetDataType() == DataType::DT_INT4 && x2->GetDataType() == DataType::DT_INT4;
    if (!isA8W8 && !isA4W4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnfusedQuantMatmul only support A8W8/A4W4 senario, but got x1: %s \
        x2: %s", op::ToString(x1->GetDataType()).GetString(), op::ToString(x2->GetDataType()).GetString());
        return false;
    }
    return true;
}

bool CheckQuantScaleShape(const aclTensor *&x1, const aclTensor *&x2, const aclTensor *&x1Scale, const aclTensor *&x2Scale,
                             TupleAttr &boolsTrans)
{   
    bool transposeX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans);

    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    const op::Shape x1Shape = x1->GetViewShape();
    const op::Shape x2Shape = x2->GetViewShape();
    int64_t x1MDim = transposeX1 ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
    int64_t x2NDim = transposeX2 ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];

    if (x1Scale->GetViewShape().GetDim(0) != x1MDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification of x1 is pertoken quant, \
            the shape of x1Scale should equal to the m: (%ld), but actual is (%ld).",
                x1MDim, x1Scale->GetViewShape().GetDim(0));
        return false;
    }
    if (x2Scale->GetViewShape().GetDim(0) != x2NDim && x2Scale->GetViewShape().GetDim(0) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When quantification of x2 is perchannel/pertensor quant, \
            the shape of x2Scale should equal to the n: (%ld) or (1), but actual is (%ld).",
                x2NDim, x2Scale->GetViewShape().GetDim(0));
        return false;
    }
    return true;
}

static aclnnStatus aclnnFusedQuantMatmulGetWorkspaceSizeCommonProcess(TupleInput &inputTensors, TupleQuant &quantTensors,
                                                                      TupleFused &fusedTensors, TupleAttr &boolsTrans,
                                                                      const aclTensor *out, const bool isWeightNz, aclOpExecutor *executor) {
    auto &x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors);
    auto &x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors);
    auto &x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &yScale = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors);
    auto &x1Offset = std::get<INDEX_X1_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto &x2Offset = std::get<INDEX_X2_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto &yOffset = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors);
    auto &bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_QUANT_TUPLE>(quantTensors);
    int64_t interfaceType = std::get<INDEX_INTERFACE_TYPE_IN_QUANT_TUPLE>(quantTensors);
    auto &x3 = std::get<INDEX_X3_IN_FUSED_TUPLE>(fusedTensors);
    const char* fusedOpType = std::get<INDEX_FUSEDOPTYPE_IN_FUSED_TUPLE>(fusedTensors);
    bool transposeX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans);
    bool transposeX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans);
    // 检验fusedOpType类型是否合法
    CHECK_RET(CheckFusedOpType(fusedOpType), ACLNN_ERR_PARAM_INVALID);

    auto ret = PreMatmulCalcProcess(inputTensors, quantTensors, fusedTensors, boolsTrans, out, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 校验当前只支持A8W8、A4W4
    CHECK_RET(CheckInputDtypeValid(x1, x2), ACLNN_ERR_PARAM_INVALID);

    // 校验x1scale、x2scale在pertoken、perchannel/pertensor量化下shape
    CHECK_RET(CheckQuantScaleShape(x1, x2, x1Scale, x2Scale, boolsTrans), ACLNN_ERR_PARAM_INVALID);

    auto reformatedX1 = SetTensorToNDFormat(x1);
    const aclTensor *reformatedX2 = x2;

    reformatedX2 = SetTensorToNDFormat(x2);

    const aclTensor *reformatedX1Scale = GetNDFormat(x1Scale);
    const aclTensor *reformatedX2Scale = GetNDFormat(x2Scale);
    const aclTensor *reformatedBias = GetNDFormat(bias);
    const TupleInput inputTuple = std::tie(reformatedX1, reformatedX2);
    const TupleQuant quantTuple = std::tie(reformatedX1Scale, reformatedX2Scale, yScale, x1Offset,
                                           x2Offset, yOffset, reformatedBias, groupSize, interfaceType);

    int64_t dtype = 0;
    TupleTensor inOutTuple = std::tie(reformatedX1, reformatedX2, out);
    GetDtypeAndTranspose(inOutTuple, dtype, transposeX1, transposeX2);

    QuantMatmulChecker qmmV3Checker(inputTuple, quantTuple, boolsTrans, out, isWeightNz);
    qmmV3Checker.Init();

    ret = CheckParams(qmmV3Checker);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *matmulRet = nullptr;

    matmulRet = l0op::FusedQuantMatMul(reformatedX1, reformatedX2, reformatedBias, reformatedX1Scale,
                                             reformatedX2Scale, yScale, x1Offset, x2Offset, yOffset,
                                             nullptr, x3, dtype, -1, transposeX1, transposeX2, groupSize, fusedOpType, executor);
    CHECK_RET(PostMatmulCalcProcess(matmulRet, x1, x2, out, executor) == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnFusedQuantMatmulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                                  const aclTensor *x2Scale, const aclTensor *yScaleOptional,
                                                  const aclTensor *x1OffsetOptional, const aclTensor *x2OffsetOptional,
                                                  const aclTensor *yOffsetOptional, const aclTensor *biasOptional,
                                                  const aclTensor *x3Optional, const char* fusedOpType, 
                                                  int64_t groupSizeOptional, aclTensor *out,
                                                  uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnFusedQuantMatmul,
                   DFX_IN(x1, x2, x1Scale, x2Scale, yScaleOptional, x1OffsetOptional, x2OffsetOptional, yOffsetOptional, biasOptional, x3Optional, fusedOpType,
                          groupSizeOptional),
                   DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    TupleInput inputTuple = std::tie(x1, x2);
    // 5 represents the aclnnFusedQuantMatmul interface
    const int64_t interfaceType = 5;
    TupleQuant quantTuple = std::tie(x1Scale, x2Scale, yScaleOptional, x1OffsetOptional,
                                     x2OffsetOptional, yOffsetOptional, biasOptional, groupSizeOptional, interfaceType);
    TupleFused fusedTuple = std::tie(x3Optional, fusedOpType);
    bool transposeX1 = IsTransposeLastTwoDims(x1);
 	bool transposeX2 = IsTransposeLastTwoDims(x2);
    TupleAttr attrTuple = std::tie(transposeX1, transposeX2);
    auto ret = aclnnFusedQuantMatmulGetWorkspaceSizeCommonProcess(inputTuple, quantTuple, fusedTuple, attrTuple, out, false,
                                                             uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
                                                  const aclTensor *x2Scale, const aclTensor *yScaleOptional,
                                                  const aclTensor *x1OffsetOptional, const aclTensor *x2OffsetOptional,
                                                  const aclTensor *yOffsetOptional, const aclTensor *biasOptional,
                                                  const aclTensor *x3Optional, const char* fusedOpType, 
                                                  int64_t groupSizeOptional, aclTensor *out,
                                                  uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnFusedQuantMatmulWeightNz,
                   DFX_IN(x1, x2, x1Scale, x2Scale, yScaleOptional, x1OffsetOptional, x2OffsetOptional, yOffsetOptional, biasOptional, x3Optional, fusedOpType,
                           groupSizeOptional),
                   DFX_OUT(out));
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedQuantMatmul WeightNz do not support x2 is nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t viewDimNum = x2->GetViewShape().GetDimNum();
    if(viewDimNum < MIN_DIM_NUM_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2's view dimNum should greater than 1, but is %ld.", viewDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    bool transposeX1 = IsTransposeLastTwoDims(x1);
 	bool transposeX2 = IsTransposeLastTwoDims(x2);
    if (transposeX2) {
        const_cast<aclTensor *>(x2)->SetViewShape(SwapLastTwoDimValue(x2->GetViewShape()));
    }

    op::Shape weightNzShape = GetWeightNzShape(x2, transposeX2);
    if (!CheckWeightNzStorageShape(weightNzShape, x2->GetStorageShape())) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "x2'format only support NZ, but now x2's format is not NZ(Ascend affinity format). \
aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight can be used to convert the input format from ND to Ascend \
affinity format.");
      return ACLNN_ERR_PARAM_INVALID;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    x2 = SetTensorToNZFormat(x2, weightNzShape, uniqueExecutor.get());
    TupleInput inputTuple = std::tie(x1, x2);
    // 5 represents the aclnnFusedQuantMatmul interface
    const int64_t interfaceType = 5;
    TupleQuant quantTuple = std::tie(x1Scale, x2Scale, yScaleOptional, x1OffsetOptional,
                                     x2OffsetOptional, yOffsetOptional, biasOptional, groupSizeOptional, interfaceType);
    TupleFused fusedTuple = std::tie(x3Optional, fusedOpType);
    TupleAttr attrTuple = std::tie(transposeX1, transposeX2);
    auto ret = aclnnFusedQuantMatmulGetWorkspaceSizeCommonProcess(inputTuple, quantTuple, fusedTuple, attrTuple, out, true,
                                                             uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedQuantMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFusedQuantMatmul);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFusedQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFusedQuantMatmulWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif