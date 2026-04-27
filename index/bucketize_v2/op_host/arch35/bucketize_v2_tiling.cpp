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
 * \file bucketize_v2_tiling.cpp
 * \brief
 */

#include "op_host/tiling_templates_registry.h"
#include "bucketize_v2_tiling.h"

namespace optiling {

ge::graphStatus Tiling4BucketizeV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling for bucketize_v2 is running.");
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4BucketizeV2(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<BucketizeV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BucketizeV2).Tiling(Tiling4BucketizeV2).TilingParse<BucketizeV2CompileInfo>(TilingPrepare4BucketizeV2);

} // namespace optiling