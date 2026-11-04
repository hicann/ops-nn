/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op_host/matmul_common_infershape.h"

namespace {
// BatchMatMul无bias输入,下标为占位值(公共推导对未实例化的可选输入按nullptr处理)
constexpr size_t BATCH_MATMUL_PLACEHOLDER_BIAS_IDX = 2;
static ge::graphStatus InferShapeForBatchMatMul(gert::InferShapeContext* context)
{
    return Ops::NN::InferShapeForBatchMatMul(context, 0, BATCH_MATMUL_PLACEHOLDER_BIAS_IDX);
}

static ge::graphStatus InferShapeRangeForBatchMatMul(gert::InferShapeRangeContext* context)
{
    return Ops::NN::InferShapeRangeForBatchMatMul(context, 0, BATCH_MATMUL_PLACEHOLDER_BIAS_IDX);
}
} // namespace

namespace ops {
IMPL_OP_INFERSHAPE(BatchMatMul).InferShape(InferShapeForBatchMatMul).InferShapeRange(InferShapeRangeForBatchMatMul);
}
