/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pooling.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = { op::DataType::DT_FLOAT,
                                                                               op::DataType::DT_FLOAT16 };
const uint64_t NCHW_N_IDX = 0;
const uint64_t NCHW_C_IDX = 1;
const uint64_t NCHW_H_IDX = 2;
const uint64_t NCHW_W_IDX = 3;
const uint64_t C0_FP32 = 8;
const uint64_t C0_FP16 = 16;

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

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self)
{
    // 只需要判断dtype
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

OP_TYPE_REGISTER(Pooling);

// REG_OP(Pooling)
//     .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT8}))
//     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_INT32}))
//     .ATTR(mode, Int, 0)                 // 0:max pooling or 1:avg pooling
//     .ATTR(global_pooling, Bool, false)
//     .ATTR(window, ListInt, {1,1})       // kernel size
//     .ATTR(stride, ListInt, {1,1})       // stride size
//     .ATTR(pad, ListInt, {0,0,0,0})      // pad size
//     .ATTR(dilation, ListInt, {1,1,1,1})
//     .ATTR(ceil_mode, Int, 0)
//     .ATTR(data_format, String, "NCHW")
//     .OP_END_FACTORY_REG(Pooling)
const aclTensor *Pooling5Hd(const aclTensor *x, const aclTensor *weight, int64_t mode, bool globalPooling,
                            const aclIntArray *window, const aclIntArray *stride, const aclIntArray *pad,
                            const int64_t ceilMode, const char *dataFormat, aclOpExecutor *executor)
{
    L0_DFX(Pooling5Hd, x, weight, mode, globalPooling, window, stride, pad, ceilMode, dataFormat);

    if (!IsAiCoreSupport(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "data type not supports.");
        return nullptr;
    }

    FVector<int64_t> newDilation { 1, 1, 1, 1 };
    auto dilation4 = executor->AllocIntArray(newDilation.data(), 4);

    FVector<int64_t> newWindow { (*window)[0], (*window)[1], (*window)[2], (*window)[3] };
    auto window4 = executor->AllocIntArray(newWindow.data(), 4);

    FVector<int64_t> newStride { (*stride)[0], (*stride)[1], (*stride)[2], (*stride)[3] };
    auto stride4 = executor->AllocIntArray(newStride.data(), 4);

    // modify pads if ceil_mode
    int64_t outH = 0;
    int64_t outW = 0;
    if (ceilMode == 0) {
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

    if (ceilMode == 0) {
        // window, pad, stride are aclIntArray of length 4, so [0], [2] is index
        if ((outH - 1) * (*stride)[2] >= x->GetViewShape().GetDim(2) +  (*pad)[0]) {
            // x ViewShape is NCHW, so Dim(2) is H
            int64_t needPad = (outH - 2) * (*stride)[2] + (*window)[2] - x->GetViewShape().GetDim(2);
            padD = needPad - padT;
        }
        // window, pad, stride are aclIntArray of length 4, so [2], [3] is index
        if ((outW - 1) * (*stride)[3] >= x->GetViewShape().GetDim(3) +  (*pad)[2]) {
            // x ViewShape is NCHW, so Dim(3) is W
            int64_t needPad = (outW - 2) * (*stride)[3] + (*window)[3] - x->GetViewShape().GetDim(3);
            padR = needPad - padL;
        }
    }

    FVector<int64_t> newPad { padT, padD, padL, padR };
    auto pad4 = executor->AllocIntArray(newPad.data(), 4);

    auto poolingOut = executor->AllocTensor(x->GetDataType(), op::Format::FORMAT_NC1HWC0, op::Format::FORMAT_NCHW);

    const char *priAttr1 = "";
    const char *priAttr2 = "NOTSET";
    auto ret = INFER_SHAPE(Pooling, OP_INPUT(x, weight), OP_OUTPUT(poolingOut),
        OP_ATTR(mode, globalPooling, window4, stride4, pad4, dilation4, ceilMode, dataFormat, priAttr1, priAttr2));
        
    // INFER_SHAPE得到poolingOut的shape不对，因此需要手动设置poolingOut的StorageShape
    int64_t c = poolingOut->GetViewShape().GetDim(NCHW_C_IDX);
    int64_t c0 = 0;
    int64_t c1 = 0;
    if (x->GetDataType() == DataType::DT_FLOAT) {
        c0 = C0_FP32;
        c1 = CeilDiv(c, c0);
    } else {
        c0 = C0_FP16;
        c1 = CeilDiv(c, c0);
    }
    const op::Shape shape = {poolingOut->GetViewShape().GetDim(NCHW_N_IDX), c1,
                             poolingOut->GetViewShape().GetDim(NCHW_H_IDX),
                             poolingOut->GetViewShape().GetDim(NCHW_W_IDX), c0};
    poolingOut->SetStorageShape(shape);

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(Pooling, op::AI_CORE, OP_INPUT(x, weight), OP_OUTPUT(poolingOut),
        OP_ATTR(mode, globalPooling, window4, stride4, pad4, dilation4, ceilMode, dataFormat, priAttr1, priAttr2));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "Pooling ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return poolingOut;
}
} // namespace l0op
