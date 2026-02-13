/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_batch_matmul.h"

#include "runtime/runtime/base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "op_api/op_api_def.h"
#include "matmul/common/op_host/op_api/batch_matmul.h"
#include "matmul/common/op_host/op_api/matmul.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "matmul/common/op_host/op_api/matmul_v2tov3.h"
#include "aclnn_kernels/transdata.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/op_api/batch_matmul_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "util/math_util.h"

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;

using namespace Ops::NN;
using namespace op;

/* BatchMatMul 算子的完整计算流程如下:
                self            mat2
                 |               |
   reshape/contiguous        reshape/contiguous
                 |               |
                cast            cast
                 |               |
              transdata      transdata
                  \              /
                  batchmatmul_op_V2
                          |
                      transdata
                          |
                        cast
                          |
                       output
*/
namespace {
static const int32_t SHAPE_LIMIT = 3;
static const int32_t FIRST_DIM = 0;
static const int32_t SECOND_DIM = 1;
static const int32_t THIRD_DIM = 2;
static const int32_t PENULTIMATE_DIM = 2;
static const int32_t LAST_DIM = 1;
static const uint32_t SOC_SPEC_INFO_LEN = 32;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WEIGHTNZ = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITHOUT_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DTYPE_LIST_HALF = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(mat2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(const aclTensor* selfTensor, const aclTensor* otherTensor, const aclTensor* outTensor)
{
    // 限制DIM必须为3D否则报错
    OP_CHECK_WRONG_DIMENSION(selfTensor, SHAPE_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(otherTensor, SHAPE_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(outTensor, SHAPE_LIMIT, return false);
    auto selfDimNum = selfTensor->GetViewShape().GetDimNum();
    auto otherDimNum = otherTensor->GetViewShape().GetDimNum();
    auto outDimNum = outTensor->GetViewShape().GetDimNum();
    const op::Shape self = selfTensor->GetViewShape();
    const op::Shape other = otherTensor->GetViewShape();
    const op::Shape out = outTensor->GetViewShape();
    // selfDimNum - 1 means self's last dim, and otherDimNum - 2 means mat2's penultimate dim
    if (selfDimNum < 2 || otherDimNum < 2 || outDimNum < 2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "shapedim of self, other or out must > 2, actual selfshapeDim [%zu], otherDimNum [%zu] , outDimNum [%zu].",
            selfDimNum, otherDimNum, outDimNum);
        return false;
    }
    if (self[selfDimNum - 1] != other[otherDimNum - PENULTIMATE_DIM]) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "self's last dim and mat2's penultimate dim shoule be same, self [%ld], mat2 [%ld].",
            self[selfDimNum - LAST_DIM], other[otherDimNum - PENULTIMATE_DIM]);
        return false;
    }
    if (self[FIRST_DIM] != other[FIRST_DIM] && self[FIRST_DIM] != 1 && other[FIRST_DIM] != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "self's first dim and mat2's first dim shoule be same, or at least one of the self's first dim and "
            "mat2's first dim is 1.Now self [%ld], mat2 [%ld].",
            self[FIRST_DIM], other[FIRST_DIM]);
        return false;
    }
    auto firstDim = self[FIRST_DIM] >= other[FIRST_DIM] ? self[FIRST_DIM] : other[FIRST_DIM];
    if (out[outDimNum - PENULTIMATE_DIM] != self[selfDimNum - PENULTIMATE_DIM] ||
        out[outDimNum - LAST_DIM] != other[otherDimNum - LAST_DIM] || out[FIRST_DIM] != firstDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "output's shape is not match input, out_m[%ld] must be same with self_m[%ld], "
            "out_n[%ld] must be same with other_n[%ld], out_batch[%ld] must be same with input_batch[%ld].",
            out[outDimNum - PENULTIMATE_DIM], self[selfDimNum - PENULTIMATE_DIM], out[outDimNum - LAST_DIM],
            other[otherDimNum - LAST_DIM], out[FIRST_DIM], firstDim);
        return false;
    }
    return true;
}

static inline bool CheckBmmResIsEmpty(const aclTensor* self, const aclTensor* mat2)
{
    return self->GetViewShape().GetDim(FIRST_DIM) == 0 || self->GetViewShape().GetDim(SECOND_DIM) == 0 ||
        mat2->GetViewShape().GetDim(THIRD_DIM) == 0;
}

static aclnnStatus CheckParamsV2(const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    auto archRule = NpuArchMatMulRule::getInstance();
    CHECK_RET(archRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        archRule -> CheckInput(self, mat2, nullptr, out, cubeMathType),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和mat2的shape是否符合要求
    CHECK_RET(CheckShape(self, mat2, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static inline bool CheckStorageShape(const aclTensor* otherTensor)
{
    auto storageShape = otherTensor->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    OP_CHECK(
        storageShapeDim == 5,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only support mat2 storageShapeDim is 5, which are [%zu].", storageShapeDim),
        return false);
    return true;
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : DataType::DT_FLOAT16;
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

bool CheckDtypeValidWeightNz(const aclTensor* self, const aclTensor* mat2, const aclTensor* out)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "batchmatmulweightnz is unsupported in this npu arch");
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_WEIGHTNZ, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mat2, DTYPE_SUPPORT_LIST_WEIGHTNZ, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_WEIGHTNZ, return false);
    if (self->GetDataType() != mat2->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "self's dtype [%s] and mat2's dtype [%s] are not equal.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(mat2->GetDataType()).GetString());
            return false;
    }

    if (self->GetDataType() != out->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "self's dtype [%s] and out's dtype [%s] are not equal.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
            return false;
    }
    return true;
}

static aclnnStatus CheckParamsWeightNz(const aclTensor* self, const aclTensor* mat2, const aclTensor* out, int8_t cubeMathType)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, mat2, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValidWeightNz(self, mat2, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和mat2的shape是否符合要求
    CHECK_RET(CheckShape(self, mat2, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查mat2的storageshape是否符合要求
    CHECK_RET(CheckStorageShape(mat2), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查cubeMathType
    CHECK_RET(CheckMathType(self, mat2, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 空tensor计算图实现
class BatchmmEmptyTensorGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // 空Tensor 不做处理
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        return ACLNN_SUCCESS;
    };

    ~BatchmmEmptyTensorGraph() override = default;
};

// 正常场景计算图实现
class BatchMatmulExecBmmOpGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // 执行 out = mat1 @ mat2
        bool isBaddbmm = false;
        const aclTensor* out = ExecBmmOpV2(matA, matB, output, cubeMathType, executor, isBaddbmm);
        CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = out;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override {
        // cast
        convOut = l0op::Cast(convOut, output->GetDataType(), executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // ViewCopy
        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };

    ~BatchMatmulExecBmmOpGraph() override = default;
};

// 创建计算图
std::shared_ptr<MatmulGraphImpl> CreateBatchMatmulGraphImpl(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor)
{
    std::shared_ptr<MatmulGraphImpl> matmulGraph = nullptr;
    // 空tensor
    if (CheckBmmResIsEmpty(self, mat2)) {
        matmulGraph = std::make_shared<BatchmmEmptyTensorGraph>(self, mat2, nullptr, out, nullptr, nullptr, cubeMathType, executor);
    }
    // 正常场景
    matmulGraph = std::make_shared<BatchMatmulExecBmmOpGraph>(self, mat2, nullptr, out, nullptr, nullptr, cubeMathType, executor);
    return matmulGraph;
}
} // namespace

aclnnStatus aclnnBatchMatMulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBatchMatMul, DFX_IN(self, mat2, cubeMathType), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsV2(self, mat2, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor场景
    if (CheckBmmResIsEmpty(self, mat2)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 根据不同的输入选择不同的计算图
    std::shared_ptr<MatmulGraphImpl> matmulGraph = CreateBatchMatmulGraphImpl(self, mat2, out, cubeMathType, uniqueExecutor.get());
    CHECK_RET(matmulGraph != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行计算图
    auto executeStatus = matmulGraph -> Execute();
    CHECK_RET(executeStatus == ACLNN_SUCCESS, executeStatus);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchMatMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchMatMul);
    // 固定写法，调用框架能力，完成计算
    OP_LOGD("Entering aclnnBatchMatmul");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnBatchMatMulWeightNzGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBatchMatMulWeightNz, DFX_IN(self, mat2, cubeMathType), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsWeightNz(self, mat2, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 从最初的接口进入bmm计算
    auto bmmOut = ExecBmmOp(self, mat2, out, cubeMathType, uniqueExecutor.get());
    CHECK_RET(bmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (bmmOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto viewCopyResult = l0op::ViewCopy(bmmOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchMatMulWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchMatMulWeightNz);
    // 固定写法，调用框架能力，完成计算
    OP_LOGD("Entering aclnnBatchMatMulWeightNz");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
