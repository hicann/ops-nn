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
 * \file ascend_quant_tiling.cpp
 * \brief
 */
#include "ascend_quant_tiling.h"
#include "ascend_quant_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "tiling_base/tiling_base.h"

using namespace Ops::NN::Optiling;
using namespace Ops::Base;
using namespace gert;

namespace optiling {

static ge::graphStatus Tiling4AscendQuant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "AscendQuantTiling running.");
    auto compile_info = reinterpret_cast<const AscendQuantCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_IF(
        compile_info == nullptr, OP_LOGE(context->GetNodeName(), "compile_info is null."), return ge::GRAPH_FAILED);

    ascendquantregbase::AscendQuantRegbase tiling(context);
    ge::graphStatus status = tiling.DoAscendQuantTiling();
    return status;
}

static ge::graphStatus TilingPrepare4AscendQuant(gert::TilingParseContext* context)
{
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AscendQuant).Tiling(Tiling4AscendQuant).TilingParse<AscendQuantCompileInfo>(TilingPrepare4AscendQuant);
} // namespace optiling
