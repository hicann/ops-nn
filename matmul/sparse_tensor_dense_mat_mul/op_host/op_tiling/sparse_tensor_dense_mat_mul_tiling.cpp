/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_tensor_dense_mat_mul_tiling.cpp
 * \brief
 */

#include "sparse_tensor_dense_mat_mul_tiling.h"

namespace {
constexpr int64_t SIMT_MAX_THREAD_NUM = 2048; // SIMT每核能启动多少线程，暂且定2048（最大）
} // namespace

namespace Ops::NN::Optiling {
static ge::graphStatus TilingForSparseTensorDenseMatMul(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "In TilingForSparseTensorDenseMatMul(context)");

    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForSparseTensorDenseMatMul(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "In TilingParseForSparseTensorDenseMatMul(context)");

    // 由于aclnn等某些通路走不到TilingParse，暂时不使用该方式获取平台信息，compileInfo留空（不使用）
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseTensorDenseMatMul)
    .Tiling(TilingForSparseTensorDenseMatMul)
    .TilingParse<SparseTensorDenseMatMulCompileInfo>(TilingParseForSparseTensorDenseMatMul);
} // namespace Ops::NN::Optiling