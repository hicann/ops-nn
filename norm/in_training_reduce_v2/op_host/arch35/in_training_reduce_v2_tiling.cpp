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
 * \file in_training_reduce_v2_tiling.cpp
 * \brief
 */

#include "in_training_reduce_v2_tiling.h"

namespace optiling {
static ge::graphStatus Tiling4INTrainingReduceV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    OP_LOGE(context, "The operator is not supported on the current chip.");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingPrepare4INTrainingReduceV2(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepare4INTrainingReduceV2 enter.");

    auto compileInfo = context->GetCompiledInfo<INTrainingReduceV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "TilingPrepare4INTrainingReduceV2 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(INTrainingReduceV2)
    .Tiling(Tiling4INTrainingReduceV2)
    .TilingParse<INTrainingReduceV2CompileInfo>(TilingPrepare4INTrainingReduceV2);

} // namespace optiling
