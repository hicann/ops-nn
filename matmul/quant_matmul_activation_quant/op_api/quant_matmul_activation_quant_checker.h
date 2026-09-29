/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_CHECKER_H
#define OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_CHECKER_H
#include <map>
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "util/math_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_executor.h"

namespace QuantMatmulActivationQuantAclnnCheck {

static constexpr int64_t NZ_K0_VALUE_BMM_BLOCK_NUM = 16;

static constexpr size_t LAST_FIRST_DIM_INDEX = 1;
static constexpr size_t LAST_SECOND_DIM_INDEX = 2;
static const int64_t NZ_K0_VALUE_INT8_TRANS = 32;
static const int64_t NZ_K0_VALUE_INT4_TRANS = 64;
static const int NZ_STORAGE_PENULTIMATE_DIM = 16;

static constexpr size_t PENULTIMATE_DIM = 2;

bool CheckSpecialCase(const aclTensor* tensor, int64_t firstLastDim, int64_t secondLastDim);
bool GetTransposeAttrValue(const aclTensor* tensor, bool transpose);
op::Shape GetWeightNzShape(const aclTensor* input, bool transpose);
bool CheckWeightNzStorageShape(const op::Shape& nzShape, const op::Shape& storageShape);
const aclTensor* SetTensorToNZFormat(const aclTensor* input, op::Shape& shape, aclOpExecutor* executor);

bool TensorContiguousProcess(const aclTensor*& contiguousTensor, bool& transpose, aclOpExecutor* executor);

aclnnStatus WeightNZCaseProcess(const aclTensor*& x2, bool& transposeX2, aclOpExecutor* executor);
aclnnStatus SetSpecilNZTensorToNormalNZFormat(const aclTensor*& input, aclOpExecutor* executor);

const aclTensor* SetTensorToNDFormat(const aclTensor* input);

} // namespace QuantMatmulActivationQuantAclnnCheck
#endif // OP_API_INC_QUANT_MATMUL_ACTIVATION_QUANT_CHECKER_H
