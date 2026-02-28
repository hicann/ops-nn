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
 * \file conv2d_transpose_v2_tiling.cc
 * \brief
 */
#include "extend_conv_transpose.h"

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "op_host/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv/common/op_host/op_tiling/math_util.h"
#include "error_util.h"

namespace {
using ExtendConvTransposeCompileInfo = Ops::NN::Conv::Conv3DBackpropV2CompileInfo;
}

namespace Ops {
namespace NN {
namespace Conv {
REGISTER_TILING_TEMPLATE("ExtendConvTranspose", ExtendConvTransposeKernelSplitFullLoadTiling, 97);
REGISTER_TILING_TEMPLATE("ExtendConvTranspose", ExtendConvTransposeKernelSplitTiling, 98);
REGISTER_TILING_TEMPLATE("ExtendConvTranspose", ExtendConvTransposeSmallShapeTiling, 99);
REGISTER_TILING_TEMPLATE("ExtendConvTranspose", ExtendConvTransposeFullLoadTiling, 100);
REGISTER_TILING_TEMPLATE("ExtendConvTranspose", ExtendConvTransposeInnerProductTiling, 101);
REGISTER_TILING_TEMPLATE("ExtendConvTranspose", ExtendConvTransposeTiling, 102);

static ge::graphStatus ExtendConvTransposeTilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForExtendConvTranspose(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<Conv3DBackpropV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    PlatformUtil::ParseRuntimePlatformInfo(*compileInfoPtr, context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->shortSocVersion = ascendcPlatform.GetSocVersion();
	compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    OP_LOGD(context->GetNodeName(), "compileInfoPtr npuarch: %d", compileInfoPtr->npuArch);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ExtendConvTranspose)
    .Tiling(ExtendConvTransposeTilingFunc)
    .TilingParse<ExtendConvTransposeCompileInfo>(TilingParseForExtendConvTranspose);
} // namespace Conv
} // namespace NN
} // namespace Ops