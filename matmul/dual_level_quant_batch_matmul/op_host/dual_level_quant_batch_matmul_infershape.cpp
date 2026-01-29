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
 * \file dual_level_quant_batch_matmul_infershape.cpp
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

namespace {

static ge::graphStatus InferShapeForDualQuantBatchMatmul(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(DualLevelQuantBatchMatmul).InferShape(InferShapeForDualQuantBatchMatmul);
}
