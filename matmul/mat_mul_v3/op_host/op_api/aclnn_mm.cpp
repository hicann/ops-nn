/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_mm.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"

using namespace op;
using namespace std;
using namespace Ops::NN;

namespace {
static const int64_t DIMS_TWO = 2;
static const int64_t M_DIM_SELF_IDX = 0;
static const int64_t K_DIM_SELF_IDX = 1;

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(mat2, return false);
    return true;
}

static bool CheckShapeValid(const aclTensor* self, const aclTensor* mat2, bool transposeX2 = false)
{
    OP_CHECK_WRONG_DIMENSION(mat2, DIMS_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(self, DIMS_TWO, return false);
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape selfShape = self->GetViewShape();
    int64_t mat2KDim = transposeX2 ? mat2Shape.GetDim(K_DIM_SELF_IDX) : mat2Shape.GetDim(M_DIM_SELF_IDX);
    int64_t selfKDim = selfShape.GetDim(K_DIM_SELF_IDX); // self固定不转置
    if (mat2KDim != selfKDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different, self Kdim[%ld], mat2 Kdim[%ld].",
                selfKDim, mat2KDim);
        return false;
    }
    return true;
}

static bool CheckOutputShapeValid(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(out, DIMS_TWO, return false);
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    op::Shape outShape = out->GetViewShape();
    if (outShape[0] != selfShape[0] || outShape[1] != mat2Shape[1]) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "output's shape is not match input, "
                "out_m[%ld] must be same with self_m[%ld], out_n[%ld] must be same with mat2_n[%ld].",
                outShape[0], selfShape[0], outShape[1], mat2Shape[1]);
        return false;
    }
    return true;
}

inline static aclnnStatus CheckMmInputParams(const aclTensor* self, const aclTensor* mat2, const aclTensor* out,
                                             int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验。
    auto archRule = BuildRule();
    CHECK_RET(archRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(archRule->CheckInput(self, mat2, nullptr, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查Shape是否支持
    CHECK_RET(CheckShapeValid(self, mat2), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查OutputShape是否支持
    CHECK_RET(CheckOutputShapeValid(self, mat2, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* MatmulProcess(const aclTensor* mat1, const aclTensor* mat2, const aclTensor* out,
                                      int8_t cubeMathType, MmOpInfo& mmOpInfo, aclOpExecutor* executor)
{
    return MatmulCommonProcess(mat1, mat2, nullptr, out, cubeMathType, mmOpInfo, executor, false);
}

class MmMatMulGraph : public Ops::NN::MatmulGraphImpl {
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus Impl() override
    {
        // 执行 Matmul: out = matA @ matB
        const aclTensor* out = MatmulProcess(matA, matB, output, cubeMathType, opInfo, executor);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);

        convOut = out;
        return ACLNN_SUCCESS;
    };

    ~MmMatMulGraph() override = default;
};

} // namespace

aclnnStatus aclnnMmGetWorkspaceSize(const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType,
                                    size_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMm, DFX_IN(self, mat2, cubeMathType), DFX_OUT(out));

    // 入参检查
    auto ret = CheckMmInputParams(self, mat2, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空tensor处理
    if (out->IsEmpty() && (self->IsEmpty() || mat2->IsEmpty())) {
        OP_LOGI("Returning an empty tensor without actually doing calculation.");
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 构建matmul计算图
    auto matmulGraph = std::make_shared<MmMatMulGraph>(self, mat2, out, cubeMathType, uniqueExecutor.get());
    CHECK_RET(matmulGraph != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 执行计算图
    auto executeStatus = matmulGraph->Execute();
    CHECK_RET(executeStatus == ACLNN_SUCCESS, executeStatus);

    // 获取workspace
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
