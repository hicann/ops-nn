/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_addbmm.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "level0/add.h"
#include "level0/axpy.h"
#include "matmul/common/op_host/op_api/batch_matmul.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/fill.h"
#include "level0/muls.h"
#include "level0/reduce_sum_op.h"

#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/op_api/batch_matmul_util.h"

using namespace Ops::NN;
using namespace op;
namespace {
static const int32_t BMM_SHAPE_LIMIT = 3;
static const int32_t FIRST_DIM = 0;
static const int32_t SECOND_DIM = 1;
static const int32_t THIRD_DIM = 2;
static const int32_t SELF_MAX = 2;
static const int32_t SELF_MIN = 0;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> dtypeSupportList = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> dtypeSupportListWithoutBf16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static inline bool CheckInputNotNull(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(batch1, return false);
    OP_CHECK_NULL(batch2, return false);
    OP_CHECK_NULL(beta, return false);
    OP_CHECK_NULL(alpha, return false);

    return true;
}

static inline bool CheckAddbmmOutputNotNull(const aclTensor* out)
{
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckNpuArchIsSupportBf16(void)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    return (npuArch == NpuArch::DAV_2201) || (npuArch == NpuArch::DAV_3510);
}

static inline bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclTensor* out)
{
    bool bf16flag = CheckNpuArchIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto dtypeList = bf16flag ? dtypeSupportList : dtypeSupportListWithoutBf16;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(batch1, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(batch2, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    if (!bf16flag && (self->GetDataType() == op::DataType::DT_BF16 || batch1->GetDataType() == op::DataType::DT_BF16 ||
                      batch2->GetDataType() == op::DataType::DT_BF16 || out->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Bfloat16 is unsupported by the current SOC version [%s], self is %s, mat1 is %s, mat2 is %s, out is %s",
            op::ToString(socVersion).GetString(), op::ToString(self->GetDataType()).GetString(),
            op::ToString(batch1->GetDataType()).GetString(), op::ToString(batch2->GetDataType()).GetString(),
            op::ToString(out->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* selfTensor, const aclTensor* batch1Tensor, const aclTensor* batch2Tensor)
{
    // check bmm shape size is 3 or not
    OP_CHECK_WRONG_DIMENSION(batch1Tensor, BMM_SHAPE_LIMIT, return false);
    OP_CHECK_WRONG_DIMENSION(batch2Tensor, BMM_SHAPE_LIMIT, return false);

    const op::Shape batch1 = batch1Tensor->GetViewShape();
    const op::Shape batch2 = batch2Tensor->GetViewShape();
    // check batch1 last dim and batch2 penultimate dim is equal or not
    if (batch1[THIRD_DIM] != batch2[SECOND_DIM]) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "batch1's last dim and batch2's penultimate dim shoule be same, batch1 [%ld], batch2 [%ld].",
            batch1[THIRD_DIM], batch2[SECOND_DIM]);
        return false;
    }

    auto batch1DimNum = batch1Tensor->GetViewShape().GetDimNum();
    auto batch2DimNum = batch2Tensor->GetViewShape().GetDimNum();
    if (!CheckBatchDimBroadcast(batch1DimNum, batch2DimNum, batch1, batch2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self's batch dim and mat2's batch dim can not broadcast");
        return false;
    }

    // check self is empty or not
    bool selfIsEmpty = selfTensor->IsEmpty();
    // check batch1@batch2's last two dims is empty or not
    bool matIsEmpty = (batch1[SECOND_DIM] == 0 || batch2[THIRD_DIM] == 0) ? true : false;

    if ((selfIsEmpty && !matIsEmpty) || (!selfIsEmpty && matIsEmpty)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "self and batch1@batch2's last two dims should be both empty or not, self is [%d], batch1@batch2 is [%d]",
            selfIsEmpty, matIsEmpty);
        return false;
    }

    return true;
}

static bool CheckBroadCast(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclTensor* out)
{
    // self's dim should be in [1, 2]
    OP_CHECK_MAX_DIM(self, SELF_MAX, return false);
    OP_CHECK_MIN_DIM(self, SELF_MIN, return false);

    op::Shape broadcastShape;
    op::Shape bmmLastTwoShape = {(batch1->GetViewShape())[1], (batch2->GetViewShape())[2]};
    if (!BroadcastInferShape(self->GetViewShape(), bmmLastTwoShape, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of self and batch1@batch2's last two dims can't broadcast.");
        return false;
    }

    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }

    if (broadcastShape != bmmLastTwoShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(bmmLastTwoShape).GetString(), op::ToString(broadcastShape).GetString());
        return false;
    }

    return true;
}

static inline bool CheckAddbmmFormat(const aclTensor* batch1, const aclTensor* batch2)
{
    if (batch1->GetViewFormat() != batch2->GetViewFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of batch1 and batch2 should be equal, batch1 [%s], batch2 [%s].",
            op::ToString(batch1->GetViewFormat()).GetString(), op::ToString(batch2->GetViewFormat()).GetString());
        return false;
    }
    return true;
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

static aclnnStatus CheckInputParams(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha, const aclTensor* out, int8_t cubeMathType)
{
    // 1. 检查输入、输出参数是否为空指针
    CHECK_RET(CheckInputNotNull(self, batch1, batch2, beta, alpha), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckAddbmmOutputNotNull(out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    auto archRule = NpuArchMatMulRule::getInstance();
    CHECK_RET(archRule != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(archRule->CheckInput(batch1, batch2, self, out, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查batch1和batch2是否满足Shape、广播、Empty条件
    CHECK_RET(CheckShape(self, batch1, batch2), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和batch1@batch2是否能广播
    CHECK_RET(CheckBroadCast(self, batch1, batch2, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查batch1, batch2和out的format是否一致，self存在与其他输入format不一样的情况
    CHECK_RET(CheckAddbmmFormat(batch1, batch2), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 对于一个 [B, M, N] 的 batch，batch1 的 M 为 0；或者 batch2 的 N 为 0
static inline bool isAddBmmProcessEmptyTensor(const aclTensor* batch1, const aclTensor* batch2)
{
    return (batch1->GetViewShape())[SECOND_DIM] == 0 || (batch2->GetViewShape())[THIRD_DIM] == 0;
}

static const aclTensor* addBmmProcessEmptyTensor(const aclTensor* self, const aclTensor* mat2, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape selfShape = self->GetViewShape();
    op::Shape mat2Shape = mat2->GetViewShape();
    // 获取self的第2维度和mat2的最后1维度作为输出shape
    op::Shape outShape = {selfShape.GetDim(1), mat2Shape.GetDim(2)};
    auto out = executor->AllocTensor(outShape, self->GetDataType());
    OP_LOGI("Returning an empty tensor without actually doing calculation");
    return out;
}

const aclTensor* addBmmProcessBroadcastSelf(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, aclOpExecutor* executor)
{
    std::vector<int64_t> tensorShape{(batch1->GetViewShape())[SECOND_DIM], (batch2->GetViewShape())[THIRD_DIM]};
    int64_t tensorSize = tensorShape.size();
    auto outShape = executor->AllocIntArray(tensorShape.data(), tensorSize);
    auto out = l0op::BroadcastTo(self, outShape, executor);
    return out;
}

static const aclTensor* SelfMulsBetaProcess(const aclTensor* self, const aclScalar* beta, aclOpExecutor* executor){
    // self * beta
    const aclTensor* selfContiguous = l0op::Contiguous(self, executor);
    // self为bf16时cast为fp32保证精度
    if (self != nullptr && self->GetDataType() == op::DataType::DT_BF16) {
        selfContiguous = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, executor);
    }
    CHECK_RET(selfContiguous != nullptr, nullptr);
    const aclTensor* mulOut = l0op::Muls(selfContiguous, beta->ToFloat(), executor);
    CHECK_RET(mulOut != nullptr, nullptr);
    return mulOut;
}

class AddbmmAlpha0Beta0Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        FVector<int64_t> fillShape = {(matA->GetViewShape())[SECOND_DIM], (matB->GetViewShape())[THIRD_DIM]};
        const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
        aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
        const aclScalar* valueScalar = executor->AllocScalar(0);
        const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, output->GetDataType());
        const aclTensor* fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
        convOut = fillTensor;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        auto viewCopyResult = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };

    ~AddbmmAlpha0Beta0Graph() override = default;
};

class AddbmmAlpha0Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // self(bias) * beta
        const aclTensor* mulOut = SelfMulsBetaProcess(bias, beta, executor);
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        mulOut = addBmmProcessBroadcastSelf(mulOut, matA, matB, executor);
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = mulOut;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        // 固定写法，将计算结果转换成输出output的数据类型
        convOut = l0op::Cast(convOut, output->GetDataType(), executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 固定写法，将计算结果拷贝到输出output上，output可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };

    ~AddbmmAlpha0Graph() override = default;
};

class AddbmmBeta0Graph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // bmmOut = batch1(matA) @ batch2(matB)
        bool isBaddbmm = false;
        const aclTensor* bmmOut = ExecBmmOpV2(matA, matB, output, cubeMathType, executor, isBaddbmm);
        CHECK_RET(bmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const int64_t dim[] = {0};
        const aclTensor* reduceSumOut = l0op::ReduceSumOp(bmmOut, executor->AllocIntArray(dim, 1), false, executor);

        // mulOut = reduceSumOut * alpha
        const aclTensor* mulOut = l0op::Muls(reduceSumOut, alpha->ToFloat(), executor);
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = mulOut;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        // 固定写法，将计算结果转换成输出output的数据类型
        convOut = l0op::Cast(convOut, output->GetDataType(), executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 固定写法，将计算结果拷贝到输出output上，output可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };

    ~AddbmmBeta0Graph() override = default;
};

class AddbmmMmOpWithBiasGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // biasBmmOut = batch1(matA) @ batch2(matB) + self(bias)
        const aclTensor* biasBmmOut = ExecBmmOpWithBiasV2(matA, matB, bias, output, cubeMathType, executor);
        CHECK_RET(biasBmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const int64_t dim[] = {0};
        biasBmmOut = l0op::ReduceSumOp(biasBmmOut, executor->AllocIntArray(dim, 1), false, executor);
        convOut = biasBmmOut;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        // 固定写法，将计算结果转换成输出output的数据类型
        convOut = l0op::Cast(convOut, output->GetDataType(), executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 固定写法，将计算结果拷贝到输出output上，output可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };

    ~AddbmmMmOpWithBiasGraph() override = default;
};

class AddbmmMatmulGraph : public Ops::NN::MatmulGraphImpl{
public:
    using MatmulGraphImpl::MatmulGraphImpl;

    aclnnStatus PreProcess() override{
        return ACLNN_SUCCESS;
    };

    aclnnStatus Impl() override{
        // self(bias) * beta
        const aclTensor* mulOut = SelfMulsBetaProcess(bias, beta, executor);
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // bmmOut = batch1(matA) @ batch2(matB)
        bool isBaddbmm = false;
        const aclTensor* bmmOut = ExecBmmOpV2(matA, matB, output, cubeMathType, executor, isBaddbmm);
        CHECK_RET(bmmOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const int64_t dim[] = {0};
        const aclTensor* reduceSumOut = l0op::ReduceSumOp(bmmOut, executor->AllocIntArray(dim, 1), false, executor);
        CHECK_RET(reduceSumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // Add算子需要对两个输入做隐式数据类型转换，根据具体算子语义按需调用
        auto promoteTypeAdd = op::PromoteType(mulOut->GetDataType(), reduceSumOut->GetDataType());
        // 将输入的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        const aclTensor* mulOutCasted = l0op::Cast(mulOut, promoteTypeAdd, executor);
        CHECK_RET(mulOutCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将reduceSumOut的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        const aclTensor* reduceSumOutCasted = l0op::Cast(reduceSumOut, promoteTypeAdd, executor);
        CHECK_RET(reduceSumOutCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 进行Add或Axpy计算
        const aclTensor* addOut = nullptr;
        if (std::abs(alpha->ToFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) {
            // alpha == 0    addOut = mulOutCasted + bmmOutCasted
            addOut = l0op::Add(mulOutCasted, reduceSumOutCasted, executor);
        } else {
            // alpha != 0    addOut = mulOutCasted + bmmOutCasted * alpha
            addOut = l0op::Axpy(mulOutCasted, reduceSumOutCasted, alpha->ToFloat(), executor);
        }
        CHECK_RET(addOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        convOut = addOut;
        return ACLNN_SUCCESS;
    };

    aclnnStatus PostProcess() override{
        // 固定写法，将计算结果转换成输出output的数据类型
        convOut = l0op::Cast(convOut, output->GetDataType(), executor);
        CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 固定写法，将计算结果拷贝到输出output上，output可能是非连续的tensor
        auto result = l0op::ViewCopy(convOut, output, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    };

    ~AddbmmMatmulGraph() override = default;
};

// 创建计算图
std::shared_ptr<MatmulGraphImpl> CreateAddbmmGraphImpl(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta, const aclScalar* alpha,
    aclTensor* out, int8_t cubeMathType, aclOpExecutor* executor)
{
    std::shared_ptr<MatmulGraphImpl> matmulGraph = nullptr;

    // alpha == 0    batch1和batch2不参与计算
    if (std::abs(alpha->ToFloat() - 0.0f) <= std::numeric_limits<float>::epsilon()) {
        // beta == 0    self不参与计算
        if (std::abs(beta->ToFloat() - 0.0f) <= std::numeric_limits<float>::epsilon()) {
            matmulGraph = std::make_shared<AddbmmAlpha0Beta0Graph>(batch1, batch2, self, out, alpha, beta, cubeMathType, executor);
            return matmulGraph;
        }
        // beta != 0
        matmulGraph = std::make_shared<AddbmmAlpha0Graph>(batch1, batch2, self, out, alpha, beta, cubeMathType, executor);
        return matmulGraph;
    }

    // alpha != 0 && beta == 0    self不参与计算，走计算图3，即不调用add算子
    if (std::abs(beta->ToFloat() - 0.0f) <= std::numeric_limits<float>::epsilon()) {
        matmulGraph = std::make_shared<AddbmmBeta0Graph>(batch1, batch2, self, out, alpha, beta, cubeMathType, executor);
        return matmulGraph;
    }

    // 以下全部是 alpha != 0 && beta != 0
    // beta == 1 && alpha == 1 && self.shape[0] == mat2.shape[1] && 不属于切k，直接走matmul的bias模式
    if (NeedToConvertBias(self, batch1, batch2, beta, alpha)) {
        OP_LOGI("addbmm run in NeedToConvertBias branch");
        matmulGraph = std::make_shared<AddbmmMmOpWithBiasGraph>(batch1, batch2, self, out, alpha, beta, cubeMathType, executor);
        return matmulGraph;
    }
    // 多数场景  beta != 0，则self参与计算，走计算图1和2，即调用add或axpy算子
    OP_LOGI("addbmm run in bmm+add branch");
    matmulGraph = std::make_shared<AddbmmMatmulGraph>(batch1, batch2, self, out, alpha, beta, cubeMathType, executor);
    return matmulGraph;
}

} // namespace

aclnnStatus aclnnAddbmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta,
    const aclScalar* alpha, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAddbmm, DFX_IN(self, batch1, batch2, beta, alpha, cubeMathType), DFX_OUT(out));

    // 参数检查
    auto ret = CheckInputParams(self, batch1, batch2, beta, alpha, out, cubeMathType);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 输入空Tensor处理方法，这种情况下addbmm算子的最终结果就是空Tensor，因此直接返回
    // 若batch1、batch2的第一维是空Tensor，在bmm接口内部会生成一个符合shape的空Tensor，继续参与计算
    if (isAddBmmProcessEmptyTensor(batch1, batch2)) {
        auto emptyOut = addBmmProcessEmptyTensor(batch1, batch2, uniqueExecutor.get());
        CHECK_RET(emptyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(emptyOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，获取计算过程中需要使用的workspace大小
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 根据不同的输入选择不同的计算图
    std::shared_ptr<MatmulGraphImpl> matmulGraph = CreateAddbmmGraphImpl(self, batch1, batch2, beta, alpha, out, cubeMathType, uniqueExecutor.get());
    CHECK_RET(matmulGraph != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行计算图
    auto executeStatus = matmulGraph->Execute();
    CHECK_RET(executeStatus == ACLNN_SUCCESS, executeStatus);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddbmm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAddbmmGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta, const aclScalar* alpha,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = selfRef;
    return aclnnAddbmmGetWorkspaceSize(
        selfRef, batch1, batch2, beta, alpha, out, cubeMathType, workspaceSize, executor);
}

aclnnStatus aclnnInplaceAddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceAddbmm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}