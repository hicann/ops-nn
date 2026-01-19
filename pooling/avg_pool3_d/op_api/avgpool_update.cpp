/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "avgpool_update.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AvgPoolUpdate);

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

// REG_OP(AvgPoolUpdate)
//     .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
//     .INPUT(x2, TensorType({DA_INT4, DT_INT8, DT_FLOAT16, DT_FLOAT}))
//     .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
//     .REQUIRED_ATTR(ksize, ListInt)
//     .REQUIRED_ATTR(strides, ListInt)
//     .ATTR(padding_mode, String, "CALCULATED")
//     .ATTR(pads, ListInt, {0, 0, 0, 0})
//     .ATTR(data_format, String, "NHWC")
//     .ATTR(ceil_mode, Bool, false)
//     .ATTR(exclusive, Bool, true)
//     .OP_END_FACTORY_REG(AvgPoolUpdate)
const aclTensor *AvgPoolUpdate5Hd(const aclTensor *x1, const aclTensor *x2, const aclIntArray *ksize,
                                  const aclIntArray *strides, const char *paddingMode,
                                  const aclIntArray *pads, const char *dataFormat,
                                  const bool ceilMode, const bool exclusive, aclOpExecutor *executor)
{
    L0_DFX(AvgPoolUpdate5Hd, x1, x2, ksize, strides, paddingMode, pads, dataFormat, ceilMode, exclusive);
    
    FVector<int64_t> newKsizes{(*ksize)[0], (*ksize)[1], (*ksize)[2], (*ksize)[3]};
    auto ksize4 = executor->AllocIntArray(newKsizes.data(), 4);

    FVector<int64_t> newStride{(*strides)[0], (*strides)[1], (*strides)[2], (*strides)[3]};
    auto stride4 = executor->AllocIntArray(newStride.data(), 4);

    FVector<int64_t> newPad{(*pads)[0], (*pads)[1], (*pads)[2], (*pads)[3]};
    auto pads4 = executor->AllocIntArray(newPad.data(), 4);

    auto avgPoolUpdateOut = executor->AllocTensor(x1->GetDataType(), op::Format::FORMAT_NC1HWC0,
        op::Format::FORMAT_NCHW);

    auto ret = INFER_SHAPE(AvgPoolUpdate, OP_INPUT(x1, x2), OP_OUTPUT(avgPoolUpdateOut),
                            OP_ATTR(ksize4, stride4, paddingMode, pads4, dataFormat, ceilMode, exclusive));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }

    // INFER_SHAPE得到avgPoolUpdateOut的shape不对，因此需要手动设置avgPoolUpdateOut的StorageShape
    int64_t c = avgPoolUpdateOut->GetViewShape().GetDim(NCHW_C_IDX);
    int64_t c0 = 0;
    int64_t c1 = 0;
    if (x2->GetDataType() == DataType::DT_FLOAT) {
        c0 = C0_FP32;
        c1 = CeilDiv(c, c0);
    } else {
        c0 = C0_FP16;
        c1 = CeilDiv(c, c0);
    }
    const op::Shape shape = {avgPoolUpdateOut->GetViewShape().GetDim(NCHW_N_IDX), c1,
                             avgPoolUpdateOut->GetViewShape().GetDim(NCHW_H_IDX),
                             avgPoolUpdateOut->GetViewShape().GetDim(NCHW_W_IDX), c0};
    avgPoolUpdateOut->SetStorageShape(shape);

    ret = ADD_TO_LAUNCHER_LIST_AICORE(AvgPoolUpdate,
                                      op::AI_CORE,
                                      OP_INPUT(x1, x2),
                                      OP_OUTPUT(avgPoolUpdateOut),
                                      OP_ATTR(ksize4, stride4, paddingMode, pads4, dataFormat, ceilMode, exclusive)
                                      );
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "AvgPoolUpdate ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return avgPoolUpdateOut;
}
}  // namespace l0op
