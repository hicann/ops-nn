/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mse_loss_tiling.cpp
 * \brief
 */
#include <cstdint>
#include <vector>
#include <string>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "mse_loss_tiling_arch35.h"
#include "mse_loss_tiling.h"

namespace optiling {
ge::graphStatus TilingForMseLoss(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "start tiling");
    auto compileInfo = reinterpret_cast<const MseLossCompileInfo*>(context->GetCompileInfo());

    auto predictInputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictInputShape);
    auto& oriShape = Ops::Base::EnsureNotScalar(predictInputShape->GetOriginShape());

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    std::vector<gert::Shape> dimShapes = {oriShape};

    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 走新的模板tiling

    OP_LOGD("MseLossTiling", "Enter new MseLossTiling");
    MseLossTiling tiling(context);
    return tiling.RunTiling(compileInfo);
}

ge::graphStatus TilingPrePareForMseLoss(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the MseLoss op.
IMPL_OP_OPTILING(MseLoss).Tiling(TilingForMseLoss).TilingParse<MseLossCompileInfo>(TilingPrePareForMseLoss);
} // namespace optiling
