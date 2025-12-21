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
 * \file masked_scatter_with_position_graph_infer.cpp
 * \brief masked_scatter_with_position operater graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static graphStatus MaskedScatterWithPositionInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "MaskedScatterWithPositionInferDataTypeByAttr begin.");
    std::set<ge::DataType> support_dtype = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8, ge::DT_INT8,
        ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_BF16
    };

    auto xDataType = context->GetInputDataType(0);
    OP_LOGD(context, "indices dtype:%s", Ops::Base::ToString(xDataType).c_str());
    ge::DataType yDtype = static_cast<ge::DataType>(xDataType);
    OP_CHECK_IF(support_dtype.count(yDtype) == 0,
    OP_LOGE(context, "out shape dtype only support float32, float16, double, uint8, int8, int16, int32, int64, bool, bfloat16, but got %s.",
        Ops::Base::ToString(yDtype).c_str()), return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, yDtype);
    OP_LOGD(context, "MaskedScatterWithPositionInferDataTypeByAttr end. Data type is %s.",
        Ops::Base::ToString(yDtype).c_str());

    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MaskedScatterWithPosition).InferDataType(MaskedScatterWithPositionInferDataType);
}; // namespace ops