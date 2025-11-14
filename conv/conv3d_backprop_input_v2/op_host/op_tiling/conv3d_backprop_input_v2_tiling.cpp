/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_input_v2_tiling.cpp
 * \brief
 */

#include "arch32/conv3d_backprop_input_v2_base_tiling.h"

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv/common/op_host/op_tiling/math_util.h"
#include "error_util.h"

namespace Ops {
namespace NN {
namespace Conv {

static ge::graphStatus Conv3DBackpropInputV2TilingFunc(gert::TilingContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "Conv3DBackpropInputV2", "context is null");
    auto compileInfoPtr = context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null ");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForConv3DBackpropInputV2(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");

    PlatformUtil::ParseRuntimePlatformInfo(*compileInfoPtr, context->GetNodeName(), *platformInfoPtr);

    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->shortSocVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv3DBackpropInputV2)
    .Tiling(Conv3DBackpropInputV2TilingFunc)
    .TilingParse<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>(TilingParseForConv3DBackpropInputV2);

} // namespace Conv
} // namespace NN
} // namespace Ops
