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
 * \file conv3d_v2_base_tiling_tilingkey.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"
#include "conv/common/op_kernel/arch35/conv_tilingkey.h"
#include "conv/conv3d_v2/op_kernel/arch35/conv3d_v2_tilingkey.h"
#include "conv3d_v2_base_tiling_template_tilingkey.h"

namespace optiling {
namespace conv_ops_tiling {
using namespace Conv3DV2Key;

ge::graphStatus Conv3dBaseTilingV2::SetTilingKey()
{
    Conv3dV2BaseTilingKey conv3dV2BaseTilingKey(tilingData_, flagInfo_, 
    descInfo_, shapeInfo_, numBlocksRes, convOpsConstParams_);
    conv3dV2BaseTilingKey.GetTemplateTilingKey(tilingKeyPara_);
    
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyPara_.fmpTiling,
                                    tilingKeyPara_.weightTiling,
                                    tilingKeyPara_.l1PingPong,
                                    tilingKeyPara_.l0PingPong,
                                    tilingKeyPara_.outputOrder,
                                    tilingKeyPara_.iterOrder,
                                    tilingKeyPara_.groupType);

    OP_LOGD(context_->GetNodeName(), "%s AscendC: tiling key: %lu.", paramInfo_.nodeType.c_str(), tilingKey_);
    return ge::GRAPH_SUCCESS;
}
}
}