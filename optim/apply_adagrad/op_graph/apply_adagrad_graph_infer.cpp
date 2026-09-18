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
 * \file apply_adagrad_graph_infer.cpp
 * \brief ApplyAdagrad graph infer resource.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "apply_adagrad_proto.h"

static constexpr size_t INPUT_VAR_INDEX = 0;
static constexpr size_t INPUT_NUM = 4;
static constexpr size_t OUTPUT_VAR_INDEX = 0;

namespace ge {
IMPLEMT_VERIFIER(ApplyAdagrad, VerifyApplyAdagrad)
{
    const DataType varDtype = op.GetInputDesc(INPUT_VAR_INDEX).GetDataType();
    for (size_t inputIdx = INPUT_VAR_INDEX + 1; inputIdx < INPUT_NUM; ++inputIdx) {
        if (op.GetInputDesc(inputIdx).GetDataType() != varDtype) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(ApplyAdagrad, VerifyApplyAdagrad);
} // namespace ge

namespace ops {
using namespace ge;

static ge::graphStatus InferDataTypeApplyAdagrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeApplyAdagrad");
    const ge::DataType varDtype = context->GetInputDataType(INPUT_VAR_INDEX);
    for (size_t inputIdx = INPUT_VAR_INDEX + 1; inputIdx < INPUT_NUM; ++inputIdx) {
        OP_CHECK_IF(context->GetInputDataType(inputIdx) != varDtype,
                    OP_LOGE(context->GetNodeName(), "All input dtypes must be the same."), return ge::GRAPH_FAILED);
    }
    context->SetOutputDataType(OUTPUT_VAR_INDEX, varDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeApplyAdagrad");
    return GRAPH_SUCCESS;
}

IMPL_OP(ApplyAdagrad).InferDataType(InferDataTypeApplyAdagrad);

} // namespace ops
