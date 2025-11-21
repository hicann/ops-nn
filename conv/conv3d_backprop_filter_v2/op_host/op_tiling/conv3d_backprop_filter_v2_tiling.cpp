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
 * \file conv3d_backprop_filter_v2_tiling.cc
 * \brief
 */
#include "arch32/conv3d_backprop_filter_v2_base_tiling.h"

#include <map>
#include <numeric>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "common/op_host/op_tiling/platform_util.h"
#include "error_util.h"

namespace Ops {
namespace NN {
namespace Conv {

static ge::graphStatus Conv3DBackpropFilterV2TilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForConv3DBackpropFilterV2(gert::TilingParseContext *context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null ");

    auto compileInfoPtr = context->GetCompiledInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");

    // 此函数会将platformInfoPtr中的soc_version传递给compileInfoPtr，在tiling中尽量使用compileInfoPtr->soc_version来获取平台信息
    PlatformUtil::ParseRuntimePlatformInfo(*compileInfoPtr, context->GetNodeName(), *platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->shortSocVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv3DBackpropFilterV2)
    .Tiling(Conv3DBackpropFilterV2TilingFunc)
    .TilingParse<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>(TilingParseForConv3DBackpropFilterV2);

}  // namespace Conv
}  // namespace NN
}  // namespace Ops
