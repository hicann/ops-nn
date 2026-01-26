/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_matmul.h"
#include <cstdio>
#include <cstring>

#include "fusedmatmul.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

#include "aclnn_kernels/reshape.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/op_api/cube_util.h"

using namespace op;
using namespace Ops::NN;
namespace {
const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BUILT_IN = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static constexpr size_t DIM_LEN = 2;

bool IsBuiltInScene(const char* fusedOpType)
{
    return std::strcmp(fusedOpType, "") == 0 || std::strcmp(fusedOpType, "relu") == 0 ||
           std::strcmp(fusedOpType, "add") == 0 || std::strcmp(fusedOpType, "mul") == 0;
}

// 校验fusedOpType是否合法
bool CheckFusedOpType(const char* fusedOpType)
{
    if (std::strcmp(fusedOpType, "") != 0 && std::strcmp(fusedOpType, "add") != 0 &&
        std::strcmp(fusedOpType, "mul") != 0 && std::strcmp(fusedOpType, "gelu_erf") != 0 &&
        std::strcmp(fusedOpType, "gelu_tanh") != 0 && std::strcmp(fusedOpType, "relu") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fusedOpType must be in the type of /add/mul/gelu_erf/gelu_tanh/relu");
        return false;
    }
    return true;
}
// 校验是否为空指针
bool CheckNotNull(
    const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* fusedOpType,
    const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(x2, return false);
    if (bias != nullptr && !IsBuiltInScene(fusedOpType)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "bias is not supported right now");
        return false;
    }
    if (std::strcmp(fusedOpType, "add") == 0 || std::strcmp(fusedOpType, "mul") == 0) {
        OP_CHECK_NULL(x3, return false);
    }
    OP_CHECK_NULL(y, return false);
    return true;
}

static inline bool CheckMathType(const aclTensor* self, const aclTensor* mat2, int8_t cubeMathType)
{
    bool selfFloat = self->GetDataType() == DataType::DT_FLOAT;
    bool mat2Float = mat2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = selfFloat || mat2Float ? DataType::DT_FLOAT : self->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

// 校验是否是ND格式
static bool CheckFormat(
    const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const aclTensor* y)
{
    // self格式是ND
    if (x->GetStorageFormat() != Format::FORMAT_ND || x2->GetStorageFormat() != Format::FORMAT_ND ||
        y->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND");
        return false;
    }

    if (bias != nullptr) {
        if (bias->GetStorageFormat() != Format::FORMAT_ND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND");
            return false;
        }
    }

    if (x3 != nullptr) {
        if (x3->GetStorageFormat() != Format::FORMAT_ND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND");
            return false;
        }
    }
    return true;
}
// 校验数据类型是否合法
static bool CheckDtypeValid(
    const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* fusedOpType,
    const aclTensor* y)
{
    auto dtypeSupportList = IsBuiltInScene(fusedOpType) ? DTYPE_SUPPORT_LIST_BUILT_IN : DTYPE_SUPPORT_LIST;
    // 检查x的数据类型是否在fusedmatmul算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, dtypeSupportList, return false);
    // 检查x2的数据类型是否在fusedmatmul算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, dtypeSupportList, return false);
    // 检查y的数据类型是否在fusedmatmul算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(y, dtypeSupportList, return false);

    // x和x2数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(x2, x->GetDataType(), return false);
    // x和y数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);
    if (bias != nullptr) {
        std::initializer_list<op::DataType> biasDtypeSupportList{x->GetDataType(), op::DataType::DT_FLOAT};
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, biasDtypeSupportList, return false);
    }
    if (x3 != nullptr) {
        // 检查x3的数据类型是否在fusedmatmul算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(x3, dtypeSupportList, return false);
        OP_CHECK_DTYPE_NOT_MATCH(x3, x->GetDataType(), return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensor* x, const aclTensor* x2, const aclTensor* x3, const aclTensor* y)
{
    // check x dims number is 2
    OP_CHECK_WRONG_DIMENSION(x, DIM_LEN, return false);

    // Check x2 dims number is 2
    OP_CHECK_WRONG_DIMENSION(x2, DIM_LEN, return false);

    if (x3 != nullptr) {
        // check x3 dims number is 2
        OP_CHECK_WRONG_DIMENSION(x3, DIM_LEN, return false);
        if (x3->GetViewShape() != y->GetViewShape()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Shape of x3 and y should be the same, but x3shape is %s, yshape is %s.",
                op::ToString(x3->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* fusedOpType,
    int8_t cubeMathType, const aclTensor* y)
{
    // 检验fusedOpType类型是否合法
    CHECK_RET(CheckFusedOpType(fusedOpType), ACLNN_ERR_PARAM_INVALID);
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, x2, bias, x3, fusedOpType, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查A和B是否为2维，且是否满足matmul shape MN 与传入的x3 shape Mn相同
    CHECK_RET(CheckShape(x, x2, x3, y), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在支持的数据类型之内
    CHECK_RET(CheckDtypeValid(x, x2, bias, x3, fusedOpType, y), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查Format是否支持
    CHECK_RET(CheckFormat(x, x2, bias, x3, y), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查cubeMathType
    CHECK_RET(CheckMathType(x, x2, cubeMathType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

/*
                 x               x2
                 |               |
            contiguous       contiguous
                 |               |
               cast             cast
                 |               |
                  \              /
                    fusedmatmul_op - add/mul/gelu_erf/gelu_tanh - contiguous - mat3
                          |
                        cast
                          |
                       output
*/
static const aclTensor* BuildFusedMatMulGraph(
    const aclTensor* x, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const aclTensor* y,
    const char* fusedOpType, int8_t cubeMathType, aclOpExecutor* executor)
{
    // 空tensor 处理
    if (x->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "fused matmul is not supported empty tensor handle");
        return nullptr;
    }
    // 解析当前规格matmulop支持的dtype、format能力
    MmOpInfo mmOpInfo = GetMatmulOpInfo(x, x2, cubeMathType);
    // 左输入非连续转连续
    auto selfCastOut = x;
    bool selfCastRes =
        ContiguousAndCast(x, selfCastOut, mmOpInfo.shapeInfo.transposeX1, mmOpInfo.support_info.self_dtype, executor);
    CHECK_RET(selfCastRes, nullptr);
    // 右输入非连续转连续
    auto mat2CastOut = x2;
    bool mat2CastRes =
        ContiguousAndCast(x2, mat2CastOut, mmOpInfo.shapeInfo.transposeX2, mmOpInfo.support_info.mat2_dtype, executor);
    CHECK_RET(mat2CastRes, nullptr);
    // bias非连续转连续以及转换dtype
    auto contiguousBias = bias;
    if (contiguousBias != nullptr) {
        contiguousBias = ContiguousBias(x, bias, executor);
        CHECK_RET(contiguousBias != nullptr, nullptr);
    }
    // x3非连续转连续
    auto contiguousX3 = x3;
    if (contiguousX3 != nullptr) {
        contiguousX3 = l0op::Contiguous(x3, executor);
        CHECK_RET(contiguousX3 != nullptr, nullptr);
    }
    const aclTensor* mmOut = l0op::FusedMatMulNd(
        selfCastOut, mat2CastOut, contiguousBias, contiguousX3, mmOpInfo.shapeInfo.transposeX1,
        mmOpInfo.shapeInfo.transposeX2, mmOpInfo.enableHf32, fusedOpType, executor);
    CHECK_RET(mmOut != nullptr, nullptr);
    // output cast
    auto castOut = l0op::Cast(mmOut, mmOpInfo.ori_info.output_dtype, executor);
    CHECK_RET(castOut != nullptr, nullptr);
    // Reshape to out shape
    auto matReshape = l0op::Reshape(castOut, y->GetViewShape(), executor);
    CHECK_RET(matReshape != nullptr, nullptr);
    return matReshape;
}

} // namespace

aclnnStatus aclnnFusedMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, const char* fusedOpType,
    int8_t cubeMathType, const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedMatmul, DFX_IN(x1, x2, bias, x3, fusedOpType, cubeMathType), DFX_OUT(y));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 固定写法，参数检查
    auto ret = CheckParams(x1, x2, bias, x3, fusedOpType, cubeMathType, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 构造fusedmatmul计算器
    auto matmulOut = BuildFusedMatMulGraph(x1, x2, bias, x3, y, fusedOpType, cubeMathType, uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (matmulOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto viewCopyResult = l0op::ViewCopy(matmulOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 获取workspace
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedMatmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedMatmul);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}