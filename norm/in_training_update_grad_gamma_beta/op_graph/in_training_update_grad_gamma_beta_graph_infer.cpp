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
 * \file in_training_update_grad_gamma_beta_graph_infer.cpp
 * \brief Graph-level data type inference for INTrainingUpdateGradGammaBeta.
 */

#include <cstddef>

#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
ge::graphStatus InferDataTypeForINTrainingUpdateGradGammaBeta(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType gammaType = context->GetInputDataType(0U);
    const ge::DataType betaType = context->GetInputDataType(1U);
    if (gammaType != ge::DT_FLOAT || betaType != ge::DT_FLOAT) {
        OP_LOGE(context, "both inputs must use float32");
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0U; index < 2U; ++index) {
        const ge::DataType declaredType = context->GetOutputDataType(index);
        if (declaredType != ge::DT_UNDEFINED && declaredType < ge::DT_MAX && declaredType != ge::DT_FLOAT) {
            OP_LOGE(context, "output[%zu] must use float32", index);
            return ge::GRAPH_FAILED;
        }
        if (context->SetOutputDataType(index, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context, "failed to set output[%zu] data type", index);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(INTrainingUpdateGradGammaBeta).InferDataType(InferDataTypeForINTrainingUpdateGradGammaBeta);
} // namespace ops
