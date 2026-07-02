/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "aclnn_matmul_add.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace {
constexpr size_t MATRIX_DIM = 2;
constexpr size_t BIAS_DIM = 1;

const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16};

static bool CheckNotNull(
    const aclTensor* a, const aclTensor* b, const aclTensor* yOut)
{
    OP_CHECK_NULL(a, return false);
    OP_CHECK_NULL(b, return false);
    OP_CHECK_NULL(yOut, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* a, const aclTensor* b, const aclTensor* bias,
    const aclTensor* yOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(a, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(b, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(yOut, DTYPE_SUPPORT_LIST, return false);
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST, return false);
    }

    if (a->GetDataType() != b->GetDataType() ||
        a->GetDataType() != yOut->GetDataType() ||
        (bias != nullptr && a->GetDataType() != bias->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "MatmulAdd requires a, b, bias and yOut to have same dtype.");
        return false;
    }
    return true;
}

static bool CheckShapeValid(
    const aclTensor* a, const aclTensor* b, const aclTensor* bias,
    const aclTensor* yOut)
{
    auto aShape = a->GetViewShape();
    auto bShape = b->GetViewShape();
    auto yShape = yOut->GetViewShape();
    if (aShape.GetDimNum() != MATRIX_DIM || bShape.GetDimNum() != MATRIX_DIM ||
        yShape.GetDimNum() != MATRIX_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "MatmulAdd expects a, b and yOut to be 2-D.");
        return false;
    }

    int64_t m = aShape.GetDim(0);
    int64_t k = aShape.GetDim(1);
    int64_t bK = bShape.GetDim(0);
    int64_t n = bShape.GetDim(1);
    if (k != bK || yShape.GetDim(0) != m || yShape.GetDim(1) != n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "MatmulAdd shape mismatch.");
        return false;
    }

    if (bias != nullptr) {
        auto biasShape = bias->GetViewShape();
        if (biasShape.GetDimNum() != BIAS_DIM || biasShape.GetDim(0) != n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MatmulAdd bias must be 1-D and its length must equal N.");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* a, const aclTensor* b, const aclTensor* bias,
    const aclTensor* yOut)
{
    CHECK_RET(CheckNotNull(a, b, yOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(a, b, bias, yOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(a, b, bias, yOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

OP_TYPE_REGISTER(MatmulAdd);

namespace {
aclnnStatus AddMatmulAddToLauncher(
    aclOpExecutor* executor, const aclTensor* a, const aclTensor* b,
    const aclTensor* bias, aclTensor* yOut)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(
        MatmulAdd, OP_INPUT(a, b, bias), OP_OUTPUT(yOut), OP_ATTR());
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMatmulAddGetWorkspaceSize(
    const aclTensor* a, const aclTensor* b, const aclTensor* bias,
    aclTensor* yOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMatmulAdd, DFX_IN(a, b, bias), DFX_OUT(yOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(a, b, bias, yOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = INFER_SHAPE(MatmulAdd, OP_INPUT(a, b, bias), OP_OUTPUT(yOut), OP_ATTR());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = AddMatmulAddToLauncher(
        uniqueExecutor.get(), a, b, bias, yOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmulAdd(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMatmulAdd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
