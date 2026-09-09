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
 * \file conv2d_transpose_v2_tiling.cc
 * \brief
 */
#include "extend_conv_transpose_v2.h"

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "op_host/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"
#include "conv/common/op_host/op_tiling/conv_math_util.h"
#include "error_util.h"

namespace {
using ExtendConvTransposeV2CompileInfo = Ops::NN::Conv::Conv3DBackpropV2CompileInfo;
}

namespace Ops {
namespace NN {
namespace Conv {
REGISTER_TILING_TEMPLATE("ExtendConvTransposeV2", ExtendConvTransposeV2SmallKernelTiling, 96);
REGISTER_TILING_TEMPLATE("ExtendConvTransposeV2", ExtendConvTransposeV2KernelSplitFullLoadTiling, 97);
REGISTER_TILING_TEMPLATE("ExtendConvTransposeV2", ExtendConvTransposeV2KernelSplitTiling, 98);
REGISTER_TILING_TEMPLATE("ExtendConvTransposeV2", ExtendConvTransposeV2SmallShapeTiling, 99);
REGISTER_TILING_TEMPLATE("ExtendConvTransposeV2", ExtendConvTransposeV2FullLoadTiling, 100);
REGISTER_TILING_TEMPLATE("ExtendConvTransposeV2", ExtendConvTransposeV2InnerProductTiling, 101);
REGISTER_TILING_TEMPLATE("ExtendConvTransposeV2", ExtendConvTransposeV2Tiling, 102);

static ge::graphStatus ExtendConvTransposeV2TilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForExtendConvTransposeV2(gert::TilingParseContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null."),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<Conv3DBackpropV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfo is null."),
                return ge::GRAPH_FAILED);
    PlatformUtil::ParseRuntimePlatformInfo(*compileInfoPtr, context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->shortSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    OP_LOGD(context->GetNodeName(), "compileInfoPtr npuarch: %d", compileInfoPtr->npuArch);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ExtendConvTransposeV2)
    .Tiling(ExtendConvTransposeV2TilingFunc)
    .TilingParse<ExtendConvTransposeV2CompileInfo>(TilingParseForExtendConvTransposeV2);
} // namespace Conv
} // namespace NN
} // namespace Ops
