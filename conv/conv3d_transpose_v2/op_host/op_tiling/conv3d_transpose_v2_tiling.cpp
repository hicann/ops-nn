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
 * \file conv3d_transpose_v2_tiling.cc
 * \brief
 */

#include <register/op_impl_registry.h>
#include <log/log.h>
#include "tiling_base/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "error_util.h"
#include "../../op_kernel/conv3d_transpose_v2_tiling_key.h"
#include "conv/conv3d_backprop_input_v2/op_kernel/arch32/conv3d_backprop_input_v2_tiling_data.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch32/conv3d_backprop_input_v2_base_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_base_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

static ge::graphStatus Conv3DTransposeV2TilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForConv3DTransposeV2(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    PlatformUtil::ParseRuntimePlatformInfo(*compileInfoPtr, context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->shortSocVersion = ascendcPlatform.GetSocVersion();
	compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    OP_LOGD(context->GetNodeName(), "compileInfoPtr npuarch: %d", compileInfoPtr->npuArch);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv3DTransposeV2)
    .Tiling(Conv3DTransposeV2TilingFunc)
    .TilingParse<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>(TilingParseForConv3DTransposeV2);

}  // namespace Conv
}  // namespace NN
}  // namespace Ops
