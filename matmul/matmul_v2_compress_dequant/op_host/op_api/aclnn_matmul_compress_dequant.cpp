/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul_compress_dequant.cpp
 * \brief
 */

#include "aclnn_matmul_compress_dequant.h"
#include "matmul_compress_dequant.h"
#include "level0/fill.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"

#include "aclnn/aclnn_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

#define K_DIMENSION_INDEX 2
#define DEQUANT_SCALE_ALIGN_SIZE 16

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {DataType::DT_INT32};
static const std::initializer_list<op::DataType> DEQ_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_UINT64};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};

static const std::string ALG_WEIGHT_UNZIP = "weight_unzip";
static const std::string ALG_UNKNOWN = "unknown";

static op::FVector<int64_t> GetShape(const aclTensor* tensor)
{
    op::FVector<int64_t> shape;
    if (tensor == nullptr) {
        shape.push_back(1);
        OP_LOGW("The input tensor of Func GetShape is nullptr");
        return shape;
    }
    if (tensor->GetViewShape().GetDimNum() == 0U) {
        shape.push_back(1);
    } else {
        size_t dimNum = tensor->GetViewShape().GetDimNum();
        for (size_t idx = 0U; idx < dimNum; idx++) {
            int64_t tmpVal = tensor->GetViewShape().GetDim(idx);
            shape.push_back(tmpVal);
        }
    }
    return shape;
}

enum UnzipMode { UNKNOWN = 0, WEIGHT_UNZIP = 1 };

inline static const std::string& GetAlgStr(int algMode)
{
    if (algMode == UnzipMode::WEIGHT_UNZIP) {
        return ALG_WEIGHT_UNZIP;
    }
    return ALG_UNKNOWN;
}

struct MatmulUnzipInput {
    const aclTensor* x1;
    const aclTensor* x2;
    const aclTensor* compressIndex;
    const aclTensor* bias;
    const aclTensor* deqScale;
    const aclTensor* offsetW;
};

inline static bool CheckNotNull(MatmulUnzipInput matmulUnzipInput, const aclIntArray* compressInfo,
                                const aclTensor* out)
{
    OP_CHECK_NULL(matmulUnzipInput.x1, return false);
    OP_CHECK_NULL(matmulUnzipInput.x2, return false);
    OP_CHECK_NULL(matmulUnzipInput.compressIndex, return false);
    OP_CHECK_NULL(matmulUnzipInput.bias, return false);
    OP_CHECK_NULL(matmulUnzipInput.deqScale, return false);
    OP_CHECK_NULL(compressInfo, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(MatmulUnzipInput matmulUnzipInput, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(matmulUnzipInput.x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(matmulUnzipInput.x2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(matmulUnzipInput.compressIndex, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(matmulUnzipInput.bias, BIAS_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(matmulUnzipInput.deqScale, DEQ_SCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShapeValid(const aclTensor* x1, const aclTensor* x2, const aclIntArray* compressInfo)
{
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    auto dimTensor1 = x1Shape.GetDimNum();
    auto dimTensor2 = x2Shape.GetDimNum();
    int64_t x1KDim = 0;
    int64_t x2KDim = 0;

    if (dimTensor1 != 2 || dimTensor2 != 1) { // ND format dims > 2 for x1
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MatmulUnzip not support x1 shape %s, x2 shape %s",
                op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    } else {
        x1KDim = x1Shape.GetDim(dimTensor1 - 1);
        x2KDim = (*compressInfo)[K_DIMENSION_INDEX];
        if (x1KDim != x2KDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
                    op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
            return false;
        }
    }

    return true;
}

inline static aclnnStatus CheckParam(MatmulUnzipInput matmulUnzipInput, const aclIntArray* compressInfo,
                                     const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(matmulUnzipInput, compressInfo, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(matmulUnzipInput, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查Shape是否支持
    CHECK_RET(CheckShapeValid(matmulUnzipInput.x1, matmulUnzipInput.x2, compressInfo), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* ProcessEmptyTensor(const aclTensor* x1, const aclTensor* out, aclOpExecutor* executor)
{
    // 获取shape信息
    op::Shape outShape = out->GetViewShape();
    auto output = executor->AllocTensor(outShape, x1->GetDataType());
    if (output->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        return output;
    }
    FVector<int64_t> fillShape = GetShape(output);
    const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
    return fillTensor;
}

inline static const aclTensor* TensorReformat(const aclTensor* x, const op::Format& format, aclOpExecutor* executor)
{
    auto formatTensor = executor == nullptr ? const_cast<aclTensor*>(x) :
                                              executor->CreateView(x, x->GetViewShape(), x->GetViewOffset());
    formatTensor->SetViewFormat(format);
    formatTensor->SetOriginalFormat(format);
    formatTensor->SetStorageFormat(format);
    return formatTensor;
}

static aclnnStatus InputsContiguousAndTransFormat(const aclTensor* tensor, const aclTensor*& reformatedTensor,
                                                  const std::string& tensorName, aclOpExecutor* executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    op::Format tensorFormat = tensor->GetStorageFormat();
    if (tensorFormat != Format::FORMAT_FRACTAL_NZ) {
        reformatedTensor = l0op::Contiguous(tensor, executor);
        CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "%s Contiguous failed.", tensorName.c_str());

        bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
        // only in 310P, tensor should be transdata to NZ format
        if (!is310P) {
            return ACLNN_SUCCESS;
        }

        reformatedTensor = l0op::TransData(reformatedTensor, Format::FORMAT_FRACTAL_NZ, 1, executor);
        CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "%s TransData failed.", tensorName.c_str());
        return ACLNN_SUCCESS;
    }
    reformatedTensor = tensor;
    return ACLNN_SUCCESS;
}

static const aclTensor* BuildMatMulUnzipGraph(MatmulUnzipInput matmulUnzipInput, const int offsetX,
                                              const aclIntArray* compressInfo, aclTensor* out, aclOpExecutor* executor)
{
    /*
     *       x1          x2
     *       |           |
     *  x1FractalNZ      x2ReformatFractalZ
     *       |           |
     *        \         /
     *  matmulv2_compress_unzip -- compressIndex, bias, deqScale(format?), offsetW, offsetX, compressInfo
     *             |
     *            out
     *             |
     *            outND
     */

    // 空tensor 处理
    if (matmulUnzipInput.x1->IsEmpty() || matmulUnzipInput.x2->IsEmpty()) {
        auto emptyOut = ProcessEmptyTensor(matmulUnzipInput.x1, out, executor);
        CHECK_RET(emptyOut != nullptr, nullptr);
        return emptyOut;
    }
    const aclTensor* reformatedX = nullptr;
    (void)InputsContiguousAndTransFormat(matmulUnzipInput.x1, reformatedX, "x1", executor);
    matmulUnzipInput.bias = l0op::Contiguous(matmulUnzipInput.bias, executor);
    CHECK_RET(matmulUnzipInput.bias != nullptr, nullptr);
    const aclTensor* deqScale5HD = matmulUnzipInput.deqScale;
    if (matmulUnzipInput.deqScale->Numel() % DEQUANT_SCALE_ALIGN_SIZE == 0) {
        deqScale5HD = TensorReformat(matmulUnzipInput.deqScale, op::Format::FORMAT_NC1HWC0, executor);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dequant Scale is invalid Data.");
        return ProcessEmptyTensor(matmulUnzipInput.x1, out, executor);
    }
    const aclTensor* x2ReFormatFractalZ = TensorReformat(matmulUnzipInput.x2, op::Format::FORMAT_FRACTAL_Z, executor);
    const aclTensor* matmulOut = l0op::MatMulCompressDequant(
        reformatedX, x2ReFormatFractalZ, matmulUnzipInput.compressIndex, deqScale5HD, matmulUnzipInput.bias, nullptr,
        false, false, compressInfo, offsetX, GetAlgStr(UnzipMode::WEIGHT_UNZIP), executor);
    CHECK_RET(matmulOut != nullptr, nullptr);
    // TransData out format from NZ to ND
    auto matmulOutND = l0op::TransData(matmulOut, op::Format::FORMAT_ND, 1, executor);
    CHECK_RET(matmulOutND != nullptr, nullptr);

    return matmulOutND;
}

aclnnStatus aclnnMatmulCompressDequantGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                       const aclTensor* compressIndex, const aclTensor* bias,
                                                       const aclTensor* deqScale, const aclTensor* offsetW, int offsetX,
                                                       const aclIntArray* compressInfo, aclTensor* out,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMatmulCompressDequant, DFX_IN(x1, x2, compressIndex, bias, deqScale, offsetX, compressInfo),
                   DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 入参初始化
    MatmulUnzipInput matmulUnzipInput;
    matmulUnzipInput.x1 = x1;
    matmulUnzipInput.x2 = x2;
    matmulUnzipInput.compressIndex = compressIndex;
    matmulUnzipInput.bias = bias;
    matmulUnzipInput.deqScale = deqScale;
    matmulUnzipInput.offsetW = offsetW;

    // 入参检查
    auto ret = CheckParam(matmulUnzipInput, compressInfo, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 构建matmul_unzip计算图
    auto matmulOut = BuildMatMulUnzipGraph(matmulUnzipInput, offsetX, compressInfo, out, uniqueExecutor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (matmulOut->IsEmpty()) {
        // 当输出为空tensor的场景，空tensor处理
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto viewCopyResult = l0op::ViewCopy(matmulOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmulCompressDequant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMatmulCompressDequant);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
