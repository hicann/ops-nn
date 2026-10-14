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
 * \file quant_batch_matmul_v4_graph_infer.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferDataTypeQuantBatchMatmulV4(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeQuantBatchMatmulV4");
    const int64_t* dtype_num = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    ge::DataType out_dtype = static_cast<ge::DataType>(*dtype_num);
    context->SetOutputDataType(0, out_dtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeQuantBatchMatmulV4");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBatchMatmulV4).InferDataType(InferDataTypeQuantBatchMatmulV4);
} // namespace ops
