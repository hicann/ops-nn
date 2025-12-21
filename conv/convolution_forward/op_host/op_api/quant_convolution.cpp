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
 * \file quant_convolution.cpp
 * \brief
 */

#include "quant_convolution.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ExtendConv2D);
OP_TYPE_REGISTER(QuantConv3D);

static const int64_t QUANT_CONV2D_DIM = 4;
static const int64_t QUANT_CONV3D_DIM = 5;
const int64_t DIM_2 = 2;
const int64_t DIM_3 = 3;
const int64_t QUANT_CONV2D_PAD_DIM_4 = 4;
const int64_t QUANT_CONV2D_PAD_DIM_2 = 2;
const int64_t QUANT_CONV3D_PAD_DIM_6 = 6;
const int64_t QUANT_CONV3D_PAD_DIM_3 = 6;
const int64_t PAD_4D_TOP_INDEX = 0;
const int64_t PAD_4D_BOTTOM_INDEX = 1;
const int64_t PAD_4D_LEFT_INDEX = 2;
const int64_t PAD_4D_RIGHT_INDEX = 3;
const int64_t PAD_6D_HEAD_INDEX = 0;
const int64_t PAD_6D_TAIL_INDEX = 1;
const int64_t PAD_6D_TOP_INDEX = 2;
const int64_t PAD_6D_BOTTOM_INDEX = 3;
const int64_t PAD_6D_LEFT_INDEX = 4;
const int64_t PAD_6D_RIGHT_INDEX = 5;
const int64_t PAD_2D_H_INDEX = 0;
const int64_t PAD_2D_W_INDEX = 1;
const int64_t PAD_3D_D_INDEX = 0;
const int64_t PAD_3D_H_INDEX = 1;
const int64_t PAD_3D_W_INDEX = 2;

aclIntArray* ConstructQuantConvNewPad(const aclIntArray *padding, aclOpExecutor *executor)
{
    FVector<int64_t> newPad;
    if (padding->Size() == QUANT_CONV2D_PAD_DIM_4) {
        newPad = {(*padding)[PAD_4D_TOP_INDEX], (*padding)[PAD_4D_BOTTOM_INDEX],
                  (*padding)[PAD_4D_LEFT_INDEX], (*padding)[PAD_4D_RIGHT_INDEX]};
    } else if (padding->Size() == QUANT_CONV2D_PAD_DIM_2) {
        newPad = {(*padding)[PAD_2D_H_INDEX], (*padding)[PAD_2D_H_INDEX],
                  (*padding)[PAD_2D_W_INDEX], (*padding)[PAD_2D_W_INDEX]};
    } else if (padding->Size() == QUANT_CONV3D_PAD_DIM_6) {
        newPad = {(*padding)[PAD_6D_HEAD_INDEX], (*padding)[PAD_6D_TAIL_INDEX],
                  (*padding)[PAD_6D_TOP_INDEX], (*padding)[PAD_6D_BOTTOM_INDEX],
                  (*padding)[PAD_6D_LEFT_INDEX], (*padding)[PAD_6D_RIGHT_INDEX]};
    } else if (padding->Size() == QUANT_CONV3D_PAD_DIM_3) {
        newPad = {(*padding)[PAD_3D_D_INDEX], (*padding)[PAD_3D_D_INDEX],
                  (*padding)[PAD_3D_H_INDEX], (*padding)[PAD_3D_H_INDEX],
                  (*padding)[PAD_3D_W_INDEX], (*padding)[PAD_3D_W_INDEX]};
    } else {
        newPad = {0};
    }
    return executor->AllocIntArray(newPad.data(), newPad.size());
}

static aclnnStatus ExtendConv2dL0Inner(const aclTensor *input, const aclTensor *weight, const aclTensor *scale,
                                      const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding,
                                      const aclIntArray *dilation, int groups, int32_t offsetx, const char* roundMode,
                                      aclTensor *&output, aclOpExecutor *executor)
{
    L0_DFX(ExtendConv2dL0Inner, input, weight, scale, bias, stride, padding, dilation, groups, offsetx, roundMode);
    if (stride->Size() < DIM_2) {
        OP_LOGE(ACLNN_ERR_INNER, "L0 func got stride dim < 2.");
        return ACLNN_ERR_INNER;
    }
    if (dilation->Size() < DIM_2) {
        OP_LOGE(ACLNN_ERR_INNER, "L0 func got dilation dim < 2.");
        return ACLNN_ERR_INNER;
    }
    aclIntArray *stride4;
    aclIntArray *dilation4;
    aclIntArray *pad4;
    FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1]};
    FVector<int64_t> newDilaition{1, 1, (*dilation)[0], (*dilation)[1]};

    stride4 = executor->AllocIntArray(newStrides.data(), QUANT_CONV2D_DIM);
    dilation4 = executor->AllocIntArray(newDilaition.data(), QUANT_CONV2D_DIM);
    pad4 = ConstructQuantConvNewPad(padding, executor);
    if (pad4->Size() != QUANT_CONV2D_PAD_DIM_4) {
        OP_LOGE(ACLNN_ERR_INNER, "L0 func construct quant conv2d newpad failed.");
        return ACLNN_ERR_INNER;
    }
    ge::AscendString originalFormat = op::ToString(input->GetOriginalFormat());
    const char *dataFormat = originalFormat.GetString();
    const char *tmpPadMode = "SPECIFIC";
    bool tmpEnableHf32 = false;
    bool tmpEnableRelu0 = false;
    bool tmpEnableRelu1 = false;
    bool tmpDualOutput = false;
    int tmpDtype0 = -1;
    int tmpDtype1 = -1;
    auto ret = INFER_SHAPE(ExtendConv2D,
        OP_INPUT(input, weight, bias, nullptr, scale, nullptr, nullptr, nullptr, nullptr, nullptr),
        OP_OUTPUT(output, output),
        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, offsetx, roundMode, tmpPadMode, tmpEnableHf32,
                tmpEnableRelu0, tmpEnableRelu1, tmpDualOutput, tmpDtype0, tmpDtype1));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        output = nullptr;
        return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
    }
    ADD_TO_LAUNCHER_LIST_AICORE(ExtendConv2D,
        OP_INPUT(input, weight, bias, nullptr, scale, nullptr, nullptr, nullptr, nullptr, nullptr),
        OP_OUTPUT(output, output),
        OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, offsetx, roundMode, tmpPadMode, tmpEnableHf32,
                tmpEnableRelu0, tmpEnableRelu1, tmpDualOutput, tmpDtype0, tmpDtype1));
    return ACLNN_SUCCESS;
}


static aclnnStatus QuantConv3dL0Inner(const aclTensor *input, const aclTensor *weight, const aclTensor *scale,
                                      const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding,
                                      const aclIntArray *dilation, int groups, int32_t offsetx, const char* roundMode,
                                      aclTensor *&output, aclOpExecutor *executor)
{
    L0_DFX(QuantConv3dL0Inner, input, weight, scale, bias, stride, padding, dilation, groups, offsetx, roundMode);
    if (stride->Size() < DIM_3) {
        OP_LOGE(ACLNN_ERR_INNER, "L0 func got stride dim < 3.");
        return ACLNN_ERR_INNER;
    }
    if (dilation->Size() < DIM_3) {
        OP_LOGE(ACLNN_ERR_INNER, "L0 func got dilation dim < 3.");
        return ACLNN_ERR_INNER;
    }
    aclIntArray *stride5;
    aclIntArray *dilation5;
    aclIntArray *pad6;
    FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1], (*stride)[2]};
    FVector<int64_t> newDilaition{1, 1, (*dilation)[0], (*dilation)[1], (*dilation)[2]};
    stride5 = executor->AllocIntArray(newStrides.data(), QUANT_CONV3D_DIM);
    dilation5 = executor->AllocIntArray(newDilaition.data(), QUANT_CONV3D_DIM);
    pad6 = ConstructQuantConvNewPad(padding, executor);
    if (pad6->Size() != QUANT_CONV3D_PAD_DIM_6) {
        OP_LOGE(ACLNN_ERR_INNER, "L0 func construct quant conv3d newpad failed.");
        return ACLNN_ERR_INNER;
    }
    ge::AscendString originalFormat = op::ToString(input->GetOriginalFormat());
    const char *dataFormat = originalFormat.GetString();
    const char *tmpPadMode = "SPECIFIC";
    auto ret = INFER_SHAPE(QuantConv3D, OP_INPUT(input, weight, scale, bias, nullptr), OP_OUTPUT(output),
        OP_ATTR(0, stride5, pad6, dilation5, groups, dataFormat, offsetx, roundMode, tmpPadMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        output = nullptr;
        return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
    }
    ADD_TO_LAUNCHER_LIST_AICORE(QuantConv3D, OP_INPUT(input, weight, scale, bias, nullptr), OP_OUTPUT(output),
        OP_ATTR(0, stride5, pad6, dilation5, groups, dataFormat, offsetx, roundMode, tmpPadMode));

    return ACLNN_SUCCESS;
}

const aclTensor *ExtendConv2dNCHW(const aclTensor *input, const aclTensor *weight, const aclTensor *scale,
                                 const aclTensor *bias, op::DataType outputDtype, const aclIntArray *stride,
                                 const aclIntArray *padding, const aclIntArray *dilation, int groups, int32_t offsetx,
                                 const char* roundMode, aclOpExecutor *executor)
{
    L0_DFX(ExtendConv2dNCHW, input, weight, scale, bias, outputDtype, stride, padding, dilation, groups, offsetx,
        roundMode);
    auto extendConvOut = executor->AllocTensor(outputDtype, input->GetStorageFormat(), input->GetOriginalFormat());
    ExtendConv2dL0Inner(input, weight, scale, bias, stride, padding, dilation, groups, offsetx, roundMode,
                       extendConvOut, executor);
    return extendConvOut;
}
const aclTensor *QuantConv3dNCDHW(const aclTensor *input, const aclTensor *weight, const aclTensor *scale,
                                  const aclTensor *bias, op::DataType outputDtype, const aclIntArray *stride,
                                  const aclIntArray *padding, const aclIntArray *dilation, int groups, int32_t offsetx,
                                  const char* roundMode, aclOpExecutor *executor)
{
    L0_DFX(QuantConv3dNCDHW, input, weight, scale, bias, outputDtype, stride, padding, dilation,
           groups, offsetx, roundMode);
    auto quantConvOut = executor->AllocTensor(outputDtype, input->GetStorageFormat(), input->GetOriginalFormat());
    QuantConv3dL0Inner(input, weight, scale, bias, stride, padding, dilation, groups, offsetx, roundMode,
                       quantConvOut, executor);
    return quantConvOut;
}

}  // namespace l0op