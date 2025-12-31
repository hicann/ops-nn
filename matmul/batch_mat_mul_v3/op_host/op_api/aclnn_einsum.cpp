/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_einsum.h"

#include <cstring>
#include <unordered_map>

#include "level0/unsqueeze.h"
#include "aclnn_kernels/cast.h"
#include "level0/mul.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace Ops::NN;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
int64_t DIM_ZERO = 0;
int64_t DIM_ONE = 1;
int64_t DIM_TWO = 2;
int64_t DIM_THREE = 3;
int64_t DIM_FOUR = 4;
int8_t g_useFP16 = 2;

static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64
};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64
};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND310P_DTYPE_SUPPORT_LIST;
    }
}

// define 回调函数类型
typedef aclnnStatus (*callback)(const aclTensorList *tensors, aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor);

typedef struct {
    std::string equation;
    callback einsumFunc;
} EinsumCallBack;

aclnnStatus CheckABCDxABCED2ABCE(const aclTensorList *tensors, const aclTensor *output) {
    auto const &tensor0Shape = (*tensors)[0]->GetViewShape();
    auto const &tensor1Shape = (*tensors)[1]->GetViewShape();
    auto const &outputShape = output->GetViewShape();
    if (tensor0Shape.GetDim(DIM_ZERO) != tensor1Shape.GetDim(DIM_ZERO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor0 shape dim0 [%ld] and tensor1 shape dim0 [%ld] should be same", tensor0Shape.GetDim(DIM_ZERO), tensor1Shape.GetDim(DIM_ZERO));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (tensor0Shape.GetDim(DIM_ONE) != tensor1Shape.GetDim(DIM_ONE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor0 shape dim1 [%ld] and tensor1 shape dim1 [%ld] should be same", tensor0Shape.GetDim(DIM_ONE), tensor1Shape.GetDim(DIM_ONE));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (tensor0Shape.GetDim(DIM_TWO) != tensor1Shape.GetDim(DIM_TWO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor0 shape dim2 [%ld] and tensor1 shape dim2 [%ld] should be same", tensor0Shape.GetDim(DIM_TWO), tensor1Shape.GetDim(DIM_TWO));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (tensor0Shape.GetDim(DIM_THREE) != tensor1Shape.GetDim(DIM_FOUR)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor0 shape dim3 [%ld] and tensor1 shape dim4 [%ld] should be same", tensor0Shape.GetDim(DIM_THREE), tensor1Shape.GetDim(DIM_FOUR));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (outputShape.GetDim(DIM_ZERO) != tensor0Shape.GetDim(DIM_ZERO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output shape dim0 [%ld] and tensor0 shape dim0 [%ld] should be same", outputShape.GetDim(DIM_ZERO), tensor0Shape.GetDim(DIM_ZERO));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (outputShape.GetDim(DIM_ONE) != tensor0Shape.GetDim(DIM_ONE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output shape dim1 [%ld] and tensor0 shape dim1 [%ld] should be same", outputShape.GetDim(DIM_ONE), tensor0Shape.GetDim(DIM_ONE));
        return ACLNN_ERR_PARAM_INVALID;
    }

    for (uint64_t i = 0; i < tensor0Shape.GetDimNum(); i++) {
        if (tensor0Shape.GetDim(i) == DIM_ZERO) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor0 shape dim%ld [0] should not be zero", i);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (tensor1Shape.GetDim(DIM_THREE) == DIM_ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tensor1 shape dim3 [0] should not be zero");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// Einsum 具体实现
aclnnStatus HandleABCDxABCED2ABCE(const aclTensorList *tensors, aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor){
    // 固定写法，创建OpExecutor
    CHECK_RET((*tensors)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET((*tensors)[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_WRONG_DIMENSION((*tensors)[0], 4, return ACLNN_ERR_PARAM_INVALID); // 校验tensorList中第1个Tensor的dimNum为4
    OP_CHECK_WRONG_DIMENSION((*tensors)[1], 5, return ACLNN_ERR_PARAM_INVALID); // 校验tensorList中第2个Tensor的dimNum为5
    OP_CHECK_WRONG_DIMENSION(output, 4, return ACLNN_ERR_PARAM_INVALID); // 校验Tensor output的dimNum为4

    auto ret = CheckABCDxABCED2ABCE(tensors, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto cubeMathType = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) ? 0 : g_useFP16;

    auto tensor0Contigous = l0op::Contiguous((*tensors)[0], uniqueExecutor.get());
    auto tensor1Contigous = l0op::Contiguous((*tensors)[1], uniqueExecutor.get());

    const aclTensor *tensor0Cast = tensor0Contigous;
    const aclTensor *tensor1Cast = tensor1Contigous;
    const aclTensor *outputCast = output;

    auto inputDtype = (*tensors)[0]->GetDataType();
    if (inputDtype != op::DataType::DT_FLOAT && inputDtype != op::DataType::DT_FLOAT16) {
      tensor0Cast = l0op::Cast(tensor0Contigous, op::DataType::DT_FLOAT16, uniqueExecutor.get());
      CHECK_RET(tensor0Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      tensor1Cast = l0op::Cast(tensor1Contigous, op::DataType::DT_FLOAT16, uniqueExecutor.get());
      CHECK_RET(tensor1Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      auto outputContigous = l0op::Contiguous(output, uniqueExecutor.get());
      outputCast = l0op::Cast(outputContigous, op::DataType::DT_FLOAT16, uniqueExecutor.get());
      CHECK_RET(outputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto expandA = l0op::UnsqueezeNd(tensor0Cast, DIM_FOUR, uniqueExecutor.get());

    auto matmulOut = ExecBmmOp(tensor1Cast, expandA, outputCast, cubeMathType, uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castResult = l0op::Cast(matmulOut, inputDtype, uniqueExecutor.get());
    const aclTensor *result = l0op::SqueezeNd(castResult, DIM_FOUR, uniqueExecutor.get());

    // 固定写法，将计算结果拷贝到输出 output output可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(result, output, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus HandleAxB2AB(const aclTensorList *tensors, aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor){
    // 固定写法，创建OpExecutor
    CHECK_RET((*tensors)[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET((*tensors)[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_WRONG_DIMENSION((*tensors)[0], 1, return ACLNN_ERR_PARAM_INVALID); // 校验tensorList中第1个Tensor的dimNum为1
    OP_CHECK_WRONG_DIMENSION((*tensors)[1], 1, return ACLNN_ERR_PARAM_INVALID); // 校验tensorList中第2个Tensor的dimNum为1
    OP_CHECK_WRONG_DIMENSION(output, 2, return ACLNN_ERR_PARAM_INVALID); // 校验Tensor output的dimNum为2

    auto tensor0Contigous = l0op::Contiguous((*tensors)[0], uniqueExecutor.get());
    CHECK_RET(tensor0Contigous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto tensor1Contigous = l0op::Contiguous((*tensors)[1], uniqueExecutor.get());
    CHECK_RET(tensor1Contigous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto leftMatrix = l0op::UnsqueezeNd(tensor0Contigous, DIM_ONE, uniqueExecutor.get());
    auto rightMatrix = l0op::UnsqueezeNd(tensor1Contigous, DIM_ZERO, uniqueExecutor.get());
    auto result = l0op::Mul(leftMatrix, rightMatrix, uniqueExecutor.get());

    // 固定写法，将计算结果拷贝到输出 output output可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(result, output, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static EinsumCallBack g_einsumFuncTable[] = {
    {"abcd,abced->abce", HandleABCDxABCED2ABCE},
    {"a,b->ab", HandleAxB2AB},
    {"", nullptr}
};

static int FindEinsumFunc(const std::string &equ) {
    for (int i = 0; g_einsumFuncTable[i].equation != ""; ++i) {
        if (equ == g_einsumFuncTable[i].equation) {
            return i;
        }
    }
    return -1;
}

// 校验空指针
static bool CheckNotNull(const aclTensorList *tensors, aclTensor *output) {
    OP_CHECK_NULL(tensors, return false);
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_NULL((*tensors)[i], return false);
    }
    OP_CHECK_NULL(output, return false);
    return true;
}

// 检查输入和输出的数据类型是否在算子的支持列表内
static inline bool CheckDtypeValid(const aclTensorList *tensors, const aclTensor* out) {
    const auto& dtypeSupportList = GetDtypeSupportList();
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_DTYPE_NOT_SUPPORT((*tensors)[i], dtypeSupportList, return false);
        if ((*tensors)[i]->GetStorageFormat() != Format::FORMAT_ND) {
	    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "format should be ND, but is [%s].",
	        op::ToString((*tensors)[i]->GetStorageFormat()).GetString());
	    return false;
	}
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);
    return true;
}

// 统一 equation 的命名
static void EquationUnification(std::string &equation) {
  OP_LOGD("original equation: [%s]", equation.c_str());
  std::unordered_map<char, char> normalize_map;
  size_t indice = 0;
  for (auto &dim_shape : equation) {
    if (isalpha(dim_shape)) {
      if (normalize_map.find(dim_shape) == normalize_map.end()) {
        normalize_map[dim_shape] = 'a' + indice;
        indice++;
      }
      dim_shape = normalize_map[dim_shape];
    }
  }
  OP_LOGD("reordered equation: [%s]", equation.c_str());
}

// CheckTensorValid
static inline bool CheckTensorValid(const aclTensorList *tensors, const aclTensor* output) {
    if (tensors->Size() != 2) { // 校验tensorList中包含2个Tensor
        OP_LOGD("invalid tensor number [%ld], should be 2", tensors->Size());
        return false;
    }

    auto input0Dtype = (*tensors)[0]->GetDataType();
    auto input1Dtype = (*tensors)[1]->GetDataType();
    auto outputDtype = output->GetDataType();
    if ((input0Dtype != input1Dtype) || (input0Dtype != outputDtype)) { // 校验三个tensor的数据类型一致
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "all inputs dtype is not equal, please check.");
        return false;
    }
    return true;
}

// 入参校验总入口
static aclnnStatus CheckParams(const aclTensorList *tensors, aclTensor *output) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(tensors, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型
    CHECK_RET(CheckDtypeValid(tensors, output), ACLNN_ERR_PARAM_INVALID);

    // 3. 校验tensor个数
    CHECK_RET(CheckTensorValid(tensors, output), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
}
// aclnnEinsum 第一段接口
aclnnStatus aclnnEinsumGetWorkspaceSize(const aclTensorList *tensors, const char *equation, aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnEinsum, DFX_IN(tensors, equation), DFX_OUT(output));

    auto ret = CheckParams(tensors, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    std::string equ = equation;
    EquationUnification(equ);
    int index = FindEinsumFunc(equ);
    if (index == -1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "failed to match equation [%s]", equ.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    aclnnStatus result = g_einsumFuncTable[index].einsumFunc(tensors, output, workspaceSize, executor);
    return result;
}

// aclnnEinsum 第二段接口
aclnnStatus aclnnEinsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnEinsum);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
