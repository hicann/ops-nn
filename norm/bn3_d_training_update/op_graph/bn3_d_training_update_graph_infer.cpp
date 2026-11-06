/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bn3_d_training_update_graph_infer.cpp
 * \brief BN3DTrainingUpdate 图模式数据类型推导。
 */

#include "register/op_impl_registry.h" // IMPL_OP, gert::InferDataTypeContext
#include "log/log.h"                   // OP_CHECK_IF, OP_LOGE_FOR_INVALID_DTYPE, Ops::Base::ToString

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputX = 0;
constexpr size_t kInputSum = 1;
constexpr size_t kInputVariance = 6;
constexpr size_t kStatInputNum = kInputVariance - kInputSum + 1;
constexpr const char* kStatInputNames[kStatInputNum] = {"sum", "square_sum", "scale", "offset", "mean", "variance"};
constexpr size_t kOutputY = 0;
constexpr size_t kOutputStatFirst = 1;
constexpr size_t kOutputStatLast = 4;
} // namespace

static ge::graphStatus InferDataTypeForBN3DTrainingUpdate(gert::InferDataTypeContext* context)
{
    // 6 路统计量输入按 proto 契约必须是 float32。GE 的 precision_reduce 会把非法 dtype(如 bf16)
    // 自动 Cast 成 fp32,下游 tiling 看到的已是 Cast 后的 dtype,校验形同失效;此处图声明 dtype
    // 尚未被抹平,是唯一能拒绝非法输入的权威校验点。
    for (size_t i = kInputSum; i <= kInputVariance; ++i) {
        const ge::DataType dtype = context->GetInputDataType(i);
        OP_CHECK_IF(dtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), kStatInputNames[i - kInputSum],
                                              Ops::Base::ToString(dtype).c_str(), "float32"),
                    return ge::GRAPH_FAILED);
    }
    context->SetOutputDataType(kOutputY, context->GetInputDataType(kInputX));
    const ge::DataType statDtype = context->GetInputDataType(kInputSum);
    for (size_t i = kOutputStatFirst; i <= kOutputStatLast; ++i) {
        context->SetOutputDataType(i, statDtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BN3DTrainingUpdate).InferDataType(InferDataTypeForBN3DTrainingUpdate);

} // namespace ops
