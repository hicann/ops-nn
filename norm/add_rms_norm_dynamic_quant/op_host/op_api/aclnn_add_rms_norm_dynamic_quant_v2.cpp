/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"

#include "add_rms_norm_dynamic_quant.h"
#include "aclnn_add_rms_norm_dynamic_quant_v2.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace AddRmsNormDynamicQuantV2ACLNN {
constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;
constexpr int IDX_4 = 4;
constexpr int OUTPUT_MASK_LEN = 2;
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;

static bool CheckFlag(const aclTensor* smoothScale1Optional, const aclTensor* smoothScale2Optional,
                      const aclBoolArray* outputMask)
{
    if (outputMask != nullptr && outputMask->Size() == OUTPUT_MASK_LEN) {
        // 只能为nullptr或者长度为2的数组
        bool outquant1 = (*outputMask)[0];
        bool outquant2 = (*outputMask)[1];
        if (smoothScale1Optional != nullptr && !outquant1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SmoothScale1Optional is not nullptr but outputMask[0] is False.");
            return false;
        }
        if (smoothScale2Optional != nullptr && !outquant2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SmoothScale2Optional is not nullptr but outputMask[1] is False.");
            return false;
        }
        // 不能两个全部为false
        if (!outquant1 && !outquant2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support both outputMask[0] and outputMask[1] are False.");
            return false;
        }
    } else if (outputMask != nullptr && outputMask->Size() != OUTPUT_MASK_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The length of outputMask must be 2, but got %zu.", outputMask->Size());
        return false;
    } else {
        // 当output_mask == nullptr时，不支持只有smooth2
        if (smoothScale1Optional == nullptr && smoothScale2Optional != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When outputMask is unavailable, it is not supported only smoothScale2Optional without "
                    "smoothScale1Optional.");
            return false;
        }
    }
    return true;
}

static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, aclTensor* y1Out,
                         aclTensor* y2Out, const aclTensor* xOut, const aclTensor* scale1Out,
                         const aclTensor* scale2Out, bool processOut1, bool processOut2)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(xOut, return false);
    if (processOut1) {
        OP_CHECK_NULL(y1Out, return false);
        OP_CHECK_NULL(scale1Out, return false);
    }
    if (processOut2) {
        OP_CHECK_NULL(y2Out, return false);
        OP_CHECK_NULL(scale2Out, return false);
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* x1, aclTensor* y1Out, aclTensor* y2Out, bool processOut1, bool processOut2)
{
    auto x1Shape = x1->GetViewShape();
    auto y1Shape = y1Out->GetViewShape();
    auto y2Shape = y2Out->GetViewShape();
    // 校验 y1 和 x1 shape 是否一致
    if (processOut1) {
        if (y1Shape != x1Shape) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y1 shape must be same as x1.");
            return false;
        }
    }

    // 校验 y2 和 x1 shape 是否一致
    if (processOut2) {
        if (y2Shape != x1Shape) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y2 shape must be same as y1.");
            return false;
        }
    }

    return true;
}

static bool CheckInt4Compatibility(const aclTensor* outTensor, int64_t xLastDim)
{
    auto outShape = outTensor->GetViewShape();
    int64_t outDimNum = static_cast<int64_t>(outShape.GetDimNum());
    int64_t outLastDim = outShape.GetDim(outDimNum - 1);
    OP_CHECK(xLastDim == outLastDim * INT4_NUMS_IN_INT32_SPACE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "For INT32 (packed int4) output, output last dim (%ld) must be 1/%ld of input x1 last dim (%ld).",
                     outLastDim, INT4_NUMS_IN_INT32_SPACE, xLastDim),
             return false);
    return true;
}
} // namespace AddRmsNormDynamicQuantV2ACLNN

static aclnnStatus AddRmsNormDynamicQuantV2Int42Int32PackedTensor(bool processOut, aclTensor*& y,
                                                                  aclOpExecutor* executor)
{
    auto viewShape = y->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    if (processOut) {
        viewShape[viewShapeDim - 1] /= AddRmsNormDynamicQuantV2ACLNN::INT4_NUMS_IN_INT32_SPACE;
    }
    auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outTemp->SetDataType(DataType::DT_INT32);
    y = outTemp;
    OP_LOGD("AddRmsNormDynamicQuantV2ACLNN output real dtype is int4, pack to int32 to out.");

    return ACLNN_SUCCESS;
}

static aclnnStatus ComputeAddRmsNormDynamicQuantV2(const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma,
                                                   const aclTensor* smoothScale1Optional,
                                                   const aclTensor* smoothScale2Optional, const aclTensor* betaOptional,
                                                   double epsilon, const aclBoolArray* outputMask, aclTensor* y1Out,
                                                   aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out,
                                                   aclTensor* scale2Out, bool processOut1, bool processOut2,
                                                   aclOpExecutor* executor)
{
    aclTensor* y1ComputeOut = nullptr;
    aclTensor* y2ComputeOut = nullptr;
    aclTensor* xComputeOut = nullptr;

    // Determine dstType from active outputs; validate dtype consistency when both outputs are active
    int32_t y1Type = processOut1 ? y1Out->GetDataType() : 0;
    int32_t y2Type = processOut2 ? y2Out->GetDataType() : 0;
    if (processOut1 && processOut2 && y1Type != y2Type) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y1 and y2 dtype must be same when both are active, got y1=%d, y2=%d.", y1Type,
                y2Type);
        return ACLNN_ERR_PARAM_INVALID;
    }
    int32_t oriType = processOut1 ? y1Type : y2Type;
    int32_t dstType = oriType;
    if (oriType == op::DataType::DT_INT32) {
        int64_t xLastDim = x1->GetViewShape().GetDim(x1->GetViewShape().GetDimNum() - 1);
        if (processOut1) {
            CHECK_RET(AddRmsNormDynamicQuantV2ACLNN::CheckInt4Compatibility(y1Out, xLastDim), ACLNN_ERR_PARAM_INVALID);
        }
        if (processOut2) {
            CHECK_RET(AddRmsNormDynamicQuantV2ACLNN::CheckInt4Compatibility(y2Out, xLastDim), ACLNN_ERR_PARAM_INVALID);
        }
        dstType = op::DataType::DT_INT4;
    } else {
        CHECK_RET(AddRmsNormDynamicQuantV2ACLNN::CheckShapeValid(x1, y1Out, y2Out, processOut1, processOut2),
                  ACLNN_ERR_INNER_TILING_ERROR);
    }
    auto addRmsNormQuantOuts = l0op::AddRmsNormDynamicQuant(x1, x2, gamma, smoothScale1Optional, smoothScale2Optional,
                                                            betaOptional, epsilon, outputMask, dstType, xOut, scale1Out,
                                                            scale2Out, executor);
    y1ComputeOut = std::get<AddRmsNormDynamicQuantV2ACLNN::IDX_0>(addRmsNormQuantOuts);
    y2ComputeOut = std::get<AddRmsNormDynamicQuantV2ACLNN::IDX_1>(addRmsNormQuantOuts);
    xComputeOut = std::get<AddRmsNormDynamicQuantV2ACLNN::IDX_2>(addRmsNormQuantOuts);
    aclTensor* scale1ComputeOut = std::get<AddRmsNormDynamicQuantV2ACLNN::IDX_3>(addRmsNormQuantOuts);
    aclTensor* scale2ComputeOut = std::get<AddRmsNormDynamicQuantV2ACLNN::IDX_4>(addRmsNormQuantOuts);

    CHECK_RET(y1ComputeOut != nullptr && y2ComputeOut != nullptr && xComputeOut != nullptr &&
                  scale1ComputeOut != nullptr && scale2ComputeOut != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    if (oriType == op::DataType::DT_INT32) {
        auto ret = AddRmsNormDynamicQuantV2Int42Int32PackedTensor(processOut1, y1ComputeOut, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = AddRmsNormDynamicQuantV2Int42Int32PackedTensor(processOut2, y2ComputeOut, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 将结果拷贝到输出tensor
    auto viewCopyY1Result = l0op::ViewCopy(y1ComputeOut, y1Out, executor);
    CHECK_RET(viewCopyY1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyScale1Result = l0op::ViewCopy(scale1ComputeOut, scale1Out, executor);
    CHECK_RET(viewCopyScale1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyY2Result = l0op::ViewCopy(y2ComputeOut, y2Out, executor);
    CHECK_RET(viewCopyY2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyScale2Result = l0op::ViewCopy(scale2ComputeOut, scale2Out, executor);
    CHECK_RET(viewCopyScale2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyXResult = l0op::ViewCopy(xComputeOut, xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static const aclTensor* ContiguousTensor(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* betaOptional, double epsilon,
    const aclBoolArray* outputMask, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out,
    aclTensor* scale2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize.");
    L2_DFX_PHASE_1(aclnnAddRmsNormDynamicQuantV2,
                   DFX_IN(x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, betaOptional, epsilon, outputMask),
                   DFX_OUT(y1Out, y2Out, xOut, scale1Out, scale2Out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // CheckFlag: validate outputMask parameter
    CHECK_RET(AddRmsNormDynamicQuantV2ACLNN::CheckFlag(smoothScale1Optional, smoothScale2Optional, outputMask),
              ACLNN_ERR_PARAM_INVALID);

    // Compute output flags
    bool processOut1 = (outputMask == nullptr) ? true : (*outputMask)[0];
    bool processOut2 = (outputMask == nullptr) ? (smoothScale1Optional != nullptr && smoothScale2Optional != nullptr) :
                                                 (*outputMask)[1];

    // CheckNotNull: validate non-null inputs/outputs
    CHECK_RET(AddRmsNormDynamicQuantV2ACLNN::CheckNotNull(x1, x2, gamma, y1Out, y2Out, xOut, scale1Out, scale2Out,
                                                          processOut1, processOut2),
              ACLNN_ERR_PARAM_NULLPTR);

    bool isRegbase = Ops::NN::AclnnUtil::IsRegbase();
    if (isRegbase && gamma->IsEmpty()) {
        // A5 norm axis empty: go through kernel empty template, do not return early
    } else {
        // A2/A3 any empty or A5 non-norm axis empty: return directly
        if (x1->IsEmpty() || x2->IsEmpty() || gamma->IsEmpty() || y2Out->IsEmpty()) {
            OP_LOGW("Got empty tensor in aclnnAddRmsNormQuantV2!");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());

    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto s1Cont = ContiguousTensor(smoothScale1Optional, uniqueExecutor.get());
    auto s2Cont = ContiguousTensor(smoothScale2Optional, uniqueExecutor.get());
    auto betaCont = ContiguousTensor(betaOptional, uniqueExecutor.get());

    auto ret = ComputeAddRmsNormDynamicQuantV2(x1Cont, x2Cont, gammaCont, s1Cont, s2Cont, betaCont, epsilon, outputMask,
                                               y1Out, y2Out, xOut, scale1Out, scale2Out, processOut1, processOut2,
                                               uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddRmsNormQuantV2GetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormDynamicQuantV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddRmsNormDynamicQuantV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
