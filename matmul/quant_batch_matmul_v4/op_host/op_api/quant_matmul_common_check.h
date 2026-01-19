/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_MATMUL_COMMON_CHECK_H
#define OP_API_INC_QUANT_MATMUL_COMMON_CHECK_H
#include <map>
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "util/math_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/op_executor.h"

using TupleInput = std::tuple<const aclTensor *, const aclTensor *>;
using TupleQuant = std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
                              const aclTensor *, const aclTensor *, const aclTensor *, const int64_t &,
                              const int64_t &>;
using TupleFused = std::tuple<const aclTensor *, const char *>; 
using TupleAttr = std::tuple<bool, bool>;
using TupleTensor = std::tuple<const aclTensor *, const aclTensor *, const aclTensor *>;

static constexpr int INDEX_X1_IN_INPUT_TUPLE = 0;
static constexpr int INDEX_X2_IN_INPUT_TUPLE = 1;
static constexpr int INDEX_X1_SCALE_IN_QUANT_TUPLE = 0;
static constexpr int INDEX_X2_SCALE_IN_QUANT_TUPLE = 1;
static constexpr int INDEX_Y_SCALE_IN_QUANT_TUPLE = 2;
static constexpr int INDEX_X1_OFFSET_IN_QUANT_TUPLE = 3;
static constexpr int INDEX_X2_OFFSET_IN_QUANT_TUPLE = 4;
static constexpr int INDEX_Y_OFFSET_IN_QUANT_TUPLE = 5;
static constexpr int INDEX_BIAS_IN_QUANT_TUPLE = 6;
static constexpr int INDEX_GROUP_SIZE_IN_QUANT_TUPLE = 7;
static constexpr int INDEX_INTERFACE_TYPE_IN_QUANT_TUPLE = 8;
static constexpr int INDEX_X3_IN_FUSED_TUPLE = 0;
static constexpr int INDEX_FUSEDOPTYPE_IN_FUSED_TUPLE = 1;
static constexpr int INDEX_OUT_IN_TUPLE = 2;

static constexpr size_t LAST_SECOND_DIM_INDEX = 2;
static const int64_t NZ_K0_VALUE_INT8 = 16;
static const int64_t NZ_K0_VALUE_INT8_TRANS = 32;
static const int NZ_STORAGE_PENULTIMATE_DIM = 16;
static const int NZ_STORAGE_LAST_DIM = 32;

static const int64_t INT4_NUMS_IN_INT32 = 8;
static const size_t MIN_DIM_NUM_ND = 2;
static const size_t MAX_DIM_NUM_ND = 6;
static const size_t MIN_DIM_NUM_NZ = 4;
static const size_t MAX_DIM_NUM_NZ = 8;

static const size_t PENULTIMATE_DIM = 2;

bool CheckSpecialCase(const aclTensor *tensor, int64_t firstLastDim, int64_t secondLastDim);
bool GetTransposeAttrValue(const aclTensor *tensor, bool transpose);
op::Shape GetWeightNzShape(const aclTensor *input, bool transpose);
bool CheckWeightNzStorageShape(const op::Shape &nzShape, const op::Shape &storageShape);
const aclTensor *SetTensorToNZFormat(const aclTensor *input, op::Shape &shape, aclOpExecutor *executor);

inline bool TensorContiguousProcess(const aclTensor *&contiguousTensor, bool &transpose,
                                           aclOpExecutor *executor);

aclnnStatus WeightNZCaseProcess(const aclTensor *&x2, bool &transposeX2, aclOpExecutor *executor);
aclnnStatus SetSpecilNZTensorToNormalNZFormat(const aclTensor *&input, aclOpExecutor *executor);

aclTensor* ConvertTensorToInt4(const aclTensor* input, aclOpExecutor* executor);
void InputPreProcessA4W4(const aclTensor *&x1, const aclTensor *&x2, aclOpExecutor *executor);
aclnnStatus A4W4CaseProcess(const aclTensor *&x1, const aclTensor *&x2, aclOpExecutor *executor);

const aclTensor* SetTensorToNDFormat(const aclTensor *input);
const aclTensor* GetNDFormat(const aclTensor *input);

void GetDtypeAndTranspose(TupleTensor mandatoryTensors, int64_t &dtype, bool &transposeX1,
                                 bool &transposeX2);

aclnnStatus SpecialOutputProcess(const aclTensor *x1, const aclTensor *x2, const aclTensor *out,
                                        const aclTensor *&matmulRet, aclOpExecutor* executor);
aclnnStatus PostMatmulCalcProcess(const aclTensor *matmulRet, const aclTensor *x1, const aclTensor *x2,
                                         const aclTensor *out, aclOpExecutor *executor);


#endif  // OP_API_INC_QUANT_MATMUL_COMMON_CHECK_H