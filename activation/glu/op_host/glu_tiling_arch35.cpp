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
 * \file glu_tiling_arch35.cpp
 * \brief
 */

#include "glu_tiling_arch35.h"
#include "glu_tiling_common.h"
#include "register/op_impl_registry.h"

namespace optiling {

using namespace glu_common;

static const int64_t BUFFER_SIZE_FACTOR = 6; // UB size divided by 6 for buffer allocation
constexpr int64_t RESERVED_SIZE_8K = 8 * 1024;

ge::graphStatus GluRegbaseTiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    OP_LOGD(context, "RunFusionKernelTiling enter.");

    auto compileInfo = context->GetCompileInfo<GluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    int64_t ubSizePlatForm = compileInfo->ubSizePlatForm - RESERVED_SIZE_8K;
    OP_CHECK_IF(
        (ubSizePlatForm <= 0), OP_LOGE(context, "RunFusionKernelTiling fail to get ub size."), return ge::GRAPH_FAILED);

    int64_t commonBufferSize = ubSizePlatForm / (BUFFER_SIZE_FACTOR * sizeof(float));
    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    if (dtype == ge::DT_BF16) {
        commonBufferSize = commonBufferSize * 2;
    }

    OP_LOGD(context, "RunFusionKernelTiling ubSize: %ld, commonBufferSize: %ld", ubSizePlatForm, commonBufferSize);
    return RunCommonTiling(context, commonBufferSize, commonBufferSize, 0);
}

} // namespace optiling