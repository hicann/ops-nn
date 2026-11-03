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
 * \file sgd_graph_infer.cpp
 * \brief SGD InferDataType：唯一图输出 parameters 的 dtype 等于输入 parameters。
 *        InferDataType 仅图场景使用，故落在 op_graph；InferShape 图与单算子共用，留在 op_host。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t PARAMETERS_INDEX = 0;
} // namespace

ge::graphStatus InferDataTypeForSgd(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "InferDataTypeForSgd begin.");
    const auto dtype = context->GetInputDataType(PARAMETERS_INDEX);
    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT && dtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "parameters", Ops::Base::ToString(dtype).c_str(),
                                          "FLOAT16, FLOAT, BFLOAT16"),
                return ge::GRAPH_FAILED);
    const char* inputNames[] = {"parameters", "gradient", "learning_rate", "accum", "momentum", "stat"};
    for (size_t input = 1; input < 6; ++input) {
        const auto inputDtype = context->GetInputDataType(input);
        OP_CHECK_IF(inputDtype != dtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context->GetNodeName(), (std::string("parameters, ") + inputNames[input]).c_str(),
                        (Ops::Base::ToString(dtype) + ", " + Ops::Base::ToString(inputDtype)).c_str(),
                        "all inputs must have the same dtype as parameters"),
                    return ge::GRAPH_FAILED);
    }
    OP_LOGD(context, "InferDataTypeForSgd end.");
    return context->SetOutputDataType(PARAMETERS_INDEX, dtype);
}

IMPL_OP(SGD).InferDataType(InferDataTypeForSgd);
} // namespace ops
