/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "avgpool_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_util.h"

using namespace op;

namespace l0op {
const uint64_t NCHW_N_IDX = 0;
const uint64_t NCHW_C_IDX = 1;
const uint64_t NCHW_H_IDX = 2;
const uint64_t NCHW_W_IDX = 3;

//AvgPoolV2的L0
constexpr int64_t DIM_N = 0;
constexpr int64_t DIM_C = 1;
constexpr int64_t DIM_H = 2;
constexpr int64_t DIM_W = 3;
constexpr int64_t TOP_IDX = 0;
constexpr int64_t BOTTOM_IDX = 1;
constexpr int64_t LEFT_IDX = 2;
constexpr int64_t RIGHT_IDX = 3;
constexpr int64_t DIGIT_TWO = 2;

inline static int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t valueNum = 0;
    if (factor == 0) {
        return valueNum;
    }
    if (value % factor == 0) {
        valueNum = value / factor;
    } else {
        valueNum = value / factor + 1;
    }
    return valueNum;
}

inline static int64_t Div(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    return value / factor;
}

OP_TYPE_REGISTER(AvgPoolV2);

// REG_OP(AvgPoolV2)
//     .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
//     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
//     .REQUIRED_ATTR(ksize, ListInt)
//     .REQUIRED_ATTR(strides, ListInt)
//     .ATTR(padding_mode, String, "CALCULATED")
//     .ATTR(pads, ListInt, {0, 0, 0, 0})
//     .ATTR(data_format, String, "NCHW")
//     .ATTR(global_pooling, Bool, false)
//     .ATTR(ceil_mode, Bool, false)
//     .ATTR(exclusive, Bool, true)
//     .ATTR(divisor_override, Int, 0)
//     .OP_END_FACTORY_REG(AvgPoolV2)
const aclTensor *AvgPoolNchw(const aclTensor *x, const aclIntArray *window, const aclIntArray *stride,
                             const std::string &paddingMode, const aclIntArray *pad, const std::string &dataFormat,
                             const bool globalPooling, const bool ceilMode, const bool exclusive,
                             const int64_t divisorOverride, aclOpExecutor *executor)
{
    L0_DFX(AvgPoolNchw, x, window, stride, paddingMode, pad, dataFormat, globalPooling, ceilMode, exclusive,
        divisorOverride);
    static internal::AicpuTaskSpace space("AvgPoolV2");

    FVector<int64_t> newWindow { (*window)[0], (*window)[1], (*window)[2], (*window)[3] };
    auto window4 = executor->AllocIntArray(newWindow.data(), 4);

    FVector<int64_t> newStride { (*stride)[0], (*stride)[1], (*stride)[2], (*stride)[3] };
    auto stride4 = executor->AllocIntArray(newStride.data(), 4);

    // modify pads if ceil_mode
    int64_t outH = 0;
    int64_t outW = 0;
    if (ceilMode) {
        // padding is symmetrical (pad[0]==pad[1], pad[2]==pad[3]), so here multiplier is 2
        // window, pad, stride are aclIntArray of length 4, so [0], [2] is index
        outH = CeilDiv(x->GetViewShape().GetDim(2) + 2 * (*pad)[0] - (*window)[2], (*stride)[2]) + 1;
        // window, pad, stride are aclIntArray of length 4, so [2], [3] is index
        outW = CeilDiv(x->GetViewShape().GetDim(3) + 2 * (*pad)[2] - (*window)[3], (*stride)[3]) + 1;
    } else {
        // window, pad, stride are aclIntArray of length 4, so [0], [2] is index
        outH = (x->GetViewShape().GetDim(2) + 2 * (*pad)[0] - (*window)[2]) / (*stride)[2] + 1;
        // window, pad, stride are aclIntArray of length 4, so [2], [3] is index
        outW = (x->GetViewShape().GetDim(3) + 2 * (*pad)[2] - (*window)[3]) / (*stride)[3] + 1;
    }

    int64_t padT = (*pad)[0]; // 0 is index of array
    int64_t padD = (*pad)[1]; // 1 is index of array
    int64_t padL = (*pad)[2]; // 2 is index of array
    int64_t padR = (*pad)[3]; // 3 is index of array

    if (ceilMode) {
        // window, pad, stride are aclIntArray of length 4, so [0], [2] is index
        if ((outH - 1) * (*stride)[2] >= x->GetViewShape().GetDim(2) + (*pad)[0]) {
            // x ViewShape is NCHW, so Dim(2) is H
            int64_t needPad = (outH - 2) * (*stride)[2] + (*window)[2] - x->GetViewShape().GetDim(2);
            padD = needPad - padT;
            outH--;
        }
        // window, pad, stride are aclIntArray of length 4, so [2], [3] is index
        if ((outW - 1) * (*stride)[3] >= x->GetViewShape().GetDim(3) + (*pad)[2]) {
            // x ViewShape is NCHW, so Dim(3) is W
            int64_t needPad = (outW - 2) * (*stride)[3] + (*window)[3] - x->GetViewShape().GetDim(3);
            padR = needPad - padL;
            outW--;
        }
    }

    FVector<int64_t> newPad { padT, padD, padL, padR };
    auto pad4 = executor->AllocIntArray(newPad.data(), 4);

    auto avgPoolingOut = executor->AllocTensor(x->GetDataType(), op::Format::FORMAT_NCHW, op::Format::FORMAT_NCHW);

    const op::Shape shape = {x->GetViewShape().GetDim(NCHW_N_IDX), x->GetViewShape().GetDim(NCHW_C_IDX), outH, outW};
    avgPoolingOut->SetStorageShape(shape);
    avgPoolingOut->SetOriginalShape(shape);
    avgPoolingOut->SetViewShape(shape);

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(AvgPoolV2, OP_ATTR_NAMES({"ksize", "strides", "padding_mode", "pads",
        "data_format", "global_pooling", "ceil_mode", "exclusive", "divisor_override"}), OP_INPUT(x),
        OP_OUTPUT(avgPoolingOut), OP_ATTR(window4, stride4, paddingMode, pad4, dataFormat, globalPooling,
        ceilMode, exclusive, divisorOverride));

    OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AvgPoolNchw ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return avgPoolingOut;
}

//AvgPoolV2的L0
static const std::initializer_list<DataType> REGBASE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                 DataType::DT_FLOAT16,
                                                                                 DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* input)
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return CheckType(input->GetDataType(), REGBASE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

// AICORE算子kernel
static inline const aclTensor* AvgPoolV2AiCore(
    const aclTensor* input, aclTensor* output, const aclIntArray *ksize, const aclIntArray *strides,
    const std::string &paddingMode, const aclIntArray *pads, const std::string &dataFormat, const bool globalPooling,
    const bool ceilMode, const bool exclusive, const int64_t divisorOverride, aclOpExecutor* executor)
{
    L0_DFX(AvgPoolV2AiCore, input, output, ksize, strides, paddingMode, pads, dataFormat,
        globalPooling, ceilMode, exclusive, divisorOverride);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AvgPoolV2, OP_INPUT(input), OP_OUTPUT(output),
    OP_ATTR(ksize, strides, paddingMode, pads, dataFormat, globalPooling, ceilMode, exclusive, divisorOverride));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
        "AvgPoolV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return output;
}

// AICPU算子kernel
static inline const aclTensor* AvgPoolV2AiCPU(
    const aclTensor* input, aclTensor* output, const aclIntArray *ksize, const aclIntArray *strides,
    const std::string &paddingMode, const aclIntArray *pads, const std::string &dataFormat, const bool globalPooling,
    const bool ceilMode, const bool exclusive, const int64_t divisorOverride, aclOpExecutor* executor)
{
    L0_DFX(AvgPoolV2AiCPU, input, output, ksize, strides, paddingMode, pads, dataFormat,
        globalPooling, ceilMode, exclusive, divisorOverride);
    static internal::AicpuTaskSpace space("AvgPoolV2");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(AvgPoolV2, OP_ATTR_NAMES({"ksize", "strides", "padding_mode", "pads",
        "data_format", "global_pooling", "ceil_mode", "exclusive", "divisor_override"}), OP_INPUT(input),
        OP_OUTPUT(output), OP_ATTR(ksize, strides, paddingMode, pads, dataFormat, globalPooling,
        ceilMode, exclusive, divisorOverride));

    OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AvgPoolV2 ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return output;
}

const aclTensor *AvgPoolV2(const aclTensor *x, const aclIntArray *ksize, const aclIntArray *strides,
                             const std::string &paddingMode, const aclIntArray *pads, const std::string &dataFormat,
                             const bool globalPooling, const bool ceilMode, const bool exclusive,
                             const int64_t divisorOverride, aclOpExecutor *executor)
{
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t topPad = (*pads)[TOP_IDX];
    int64_t bottomPad = (*pads)[BOTTOM_IDX];
    int64_t leftPad = (*pads)[LEFT_IDX];
    int64_t rightPad = (*pads)[RIGHT_IDX];
    int64_t ksizeH = (*ksize)[DIM_H];
    int64_t ksizeW = (*ksize)[DIM_W];
    int64_t strideH = (*strides)[DIM_H];
    int64_t strideW = (*strides)[DIM_W];

    if (globalPooling) {
        outH = 1;
        outW = 1;
    } else {
        if (paddingMode == "VALID") {
            outH = Div(x->GetViewShape().GetDim(DIM_H) - ksizeH + strideH, strideH);
            outW = Div(x->GetViewShape().GetDim(DIM_W) - ksizeW + strideW, strideW);
            if (ceilMode) {
                outH = Div(x->GetViewShape().GetDim(DIM_H) - ksizeH + strideH - 1, strideH) + 1;
                outW = Div(x->GetViewShape().GetDim(DIM_W) - ksizeW + strideW - 1, strideW) + 1;
            }
        } else if (paddingMode == "SAME") {
            outH = Div(x->GetViewShape().GetDim(DIM_H) + strideH - 1, strideH);
            outW = Div(x->GetViewShape().GetDim(DIM_W) + strideW - 1, strideW);
        } else {
            outH = Div(x->GetViewShape().GetDim(DIM_H) - ksizeH + topPad + bottomPad, strideH) + 1;
            outW = Div(x->GetViewShape().GetDim(DIM_W) - ksizeW + leftPad + rightPad, strideW) + 1;
            if (ceilMode) {
                outH = Div(x->GetViewShape().GetDim(DIM_H) - ksizeH + topPad + bottomPad + strideH - 1, strideH) + 1;
                outW = Div(x->GetViewShape().GetDim(DIM_W) - ksizeW + leftPad + rightPad + strideW - 1, strideW) + 1;
                if ((outH - 1) * strideH >= x->GetViewShape().GetDim(DIM_H) + topPad) {
                    outH = outH - 1;
                }
                if ((outW - 1) * strideW >= x->GetViewShape().GetDim(DIM_W) + leftPad) {
                    outW = outW - 1;
                }
            }
        }
    }

    auto outputShape = x->GetViewShape();
    outputShape.SetDim(DIM_H, outH);
    outputShape.SetDim(DIM_W, outW);
    auto output = executor->AllocTensor(outputShape, x->GetDataType(), x->GetStorageFormat());
    CHECK_RET(output != nullptr, nullptr);

    if (IsAiCoreSupport(x)) {
        return AvgPoolV2AiCore(x, output, ksize, strides, paddingMode, pads, dataFormat, globalPooling,
            ceilMode, exclusive, divisorOverride, executor);
    } else {
        return AvgPoolV2AiCPU(x, output, ksize, strides, paddingMode, pads, dataFormat, globalPooling,
            ceilMode, exclusive, divisorOverride, executor);
    }
}
} // namespace l0op
